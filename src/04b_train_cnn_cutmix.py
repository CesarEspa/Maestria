"""
04b - CNN con CutMix (aumento de datos avanzado, "intercambio de píxeles")

El resumen de la propuesta 1 del TFM menciona "técnicas avanzadas de
aumento de datos, como el intercambio de píxeles" — el proyecto solo
tenía aumento geométrico estándar (flip/rotación/zoom/contraste, en
04_train_cnn_aug.py). CutMix es exactamente eso: recorta una región
rectangular de una imagen y la pega sobre otra, mezclando las etiquetas
en proporción al área intercambiada (Yun et al., 2019).

Misma arquitectura, misma normalización, mismo optimizador/loss/paciencia
que 04_train_cnn_aug.py — el ÚNICO cambio es qué aumento de datos se
aplica, para que la comparación entre ambos sea limpia.

Hipótesis a contrastar (dada explícitamente por el usuario): el modelo
con aumento estándar colapsa (exactitud 50.91%, predice "Maligno" para
todo, ver README Hallazgo #3). ¿CutMix, al mezclar etiquetas de forma
continua en vez de transformar geométricamente, evita ese colapso?

Nota técnica sobre class_weight: con CutMix las etiquetas dejan de ser
one-hot puro (pasan a ser una combinación como [0.3, 0.7, 0.0]), así que
`class_weight` de Keras — pensado para una clase por muestra— ya no es
aplicable directamente. Para mantener el mismo espíritu de balanceo de
clases que 04_train_cnn_aug.py, cada muestra mezclada recibe un
`sample_weight` igual a la combinación (en la misma proporción lambda
que las etiquetas) de los pesos de clase de las dos imágenes que la
formaron.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    SPLITS_DIR, MODELS_DIR, FIGURES_DIR,
    IMG_SIZE, NUM_CLASSES, BATCH_SIZE,
    EPOCHS_BASE, LEARNING_RATE, PATIENCE, SEED
)
from progress_tracker import WebProgressCallback, mark_status

MODEL_KEY = "cnn_cutmix"
DISPLAY_NAME = "CNN + CutMix"
CUTMIX_ALPHA = 1.0  # Beta(1,1) = uniforme en [0,1], igual que el paper original

EPOCHS_BASE = int(os.environ.get("TFM_EPOCHS_OVERRIDE", EPOCHS_BASE))

tf.random.set_seed(SEED)
np.random.seed(SEED)


def sample_beta(alpha):
    """Beta(alpha, alpha) vía dos Gamma(alpha,1) — sin depender de
    tensorflow_probability, que no es una dependencia del proyecto."""
    gamma1 = tf.random.gamma([], alpha, 1.0)
    gamma2 = tf.random.gamma([], alpha, 1.0)
    return gamma1 / (gamma1 + gamma2)


def cutmix_batch(images, labels, class_weights_tensor, alpha=CUTMIX_ALPHA):
    """
    Aplica CutMix a un batch completo: cada imagen se mezcla con otra del
    mismo batch (batch barajado), con UN solo recuadro aleatorio para todo
    el batch (misma técnica que la implementación de referencia del paper
    y el tutorial oficial de Keras) — vectorizado, sin bucles Python.
    """
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.float32)

    batch_size = tf.shape(images)[0]
    h, w = tf.shape(images)[1], tf.shape(images)[2]

    shuffled_idx = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, shuffled_idx)
    shuffled_labels = tf.gather(labels, shuffled_idx)

    lam = sample_beta(alpha)
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(cut_ratio * tf.cast(h, tf.float32), tf.int32)
    cut_w = tf.cast(cut_ratio * tf.cast(w, tf.float32), tf.int32)

    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)

    # Máscara (H,W,1): 0 dentro del recuadro (se usa la imagen barajada),
    # 1 fuera (se conserva la imagen original).
    row_idx = tf.range(h)[:, None]
    col_idx = tf.range(w)[None, :]
    inside_box = (row_idx >= y1) & (row_idx < y2) & (col_idx >= x1) & (col_idx < x2)
    mask = tf.where(inside_box, 0.0, 1.0)
    mask = mask[None, :, :, None]  # (1,H,W,1), broadcast sobre el batch y los 3 canales

    mixed_images = images * mask + shuffled_images * (1.0 - mask)

    # Lambda REAL según el área del recuadro ya recortada a los bordes
    # (puede diferir un poco del lambda muestreado si el recuadro se sale
    # de la imagen).
    box_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
    img_area = tf.cast(h * w, tf.float32)
    lam_adj = 1.0 - (box_area / img_area)

    mixed_labels = lam_adj * labels + (1.0 - lam_adj) * shuffled_labels

    # sample_weight: misma proporción lambda, aplicada a los pesos de
    # clase de las dos imágenes que forman la mezcla (ver nota del
    # docstring del módulo sobre por qué no se usa class_weight aquí).
    w1 = tf.gather(class_weights_tensor, tf.argmax(labels, axis=1))
    w2 = tf.gather(class_weights_tensor, tf.argmax(shuffled_labels, axis=1))
    sample_weight = lam_adj * w1 + (1.0 - lam_adj) * w2

    return mixed_images, mixed_labels, sample_weight


def build_cnn_cutmix():
    """Misma arquitectura EXACTA que build_cnn_augmented() en
    04_train_cnn_aug.py, pero sin la capa de augmentation geométrica
    integrada — el aumento (CutMix) se aplica fuera del modelo, sobre los
    batches del dataset de entrenamiento."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    return model


def plot_cutmix_samples(X_train, y_train_cat, class_weights_tensor, filename="muestras_cutmix.png"):
    """Visualiza el efecto de CutMix sobre un batch de muestra."""
    batch_images = tf.constant(X_train[:8])
    batch_labels = tf.constant(y_train_cat[:8])
    mixed_images, mixed_labels, _ = cutmix_batch(batch_images, batch_labels, class_weights_tensor)
    mixed_images = mixed_images.numpy()
    mixed_labels = mixed_labels.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i in range(8):
        row, col = divmod(i, 4)
        axes[row, col].imshow(np.clip(mixed_images[i], 0, 1))
        label_str = "/".join(f"{v:.2f}" for v in mixed_labels[i])
        axes[row, col].set_title(f"Etiqueta mezclada: [{label_str}]", fontsize=9)
        axes[row, col].axis("off")
    fig.suptitle("Efecto de CutMix sobre un batch de muestra (etiquetas [Benigno, Maligno, Normal])",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def plot_training_history(history, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["loss"], label="Entrenamiento", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Validación", linewidth=2)
    ax1.set_title("Pérdida (Loss)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"], label="Entrenamiento", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Validación", linewidth=2)
    ax2.set_title("Exactitud (Accuracy)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("CNN + CutMix — Curvas de Entrenamiento", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO — CNN con CutMix")
    print("=" * 60)

    mark_status(MODEL_KEY, DISPLAY_NAME, "running", current_epoch=0,
                total_epochs=EPOCHS_BASE, phase="", history=[])

    try:
        print("\n[1/5] Cargando datos preprocesados...")
        data = np.load(SPLITS_DIR / "dataset_splits.npz")
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

        y_train_cat = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

        print("\n[2/5] Visualizando efecto de CutMix...")
        plot_cutmix_samples(X_train, y_train_cat, class_weights_tensor)

        print("\n[3/5] Construyendo pipeline de datos con CutMix...")
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
            .shuffle(len(X_train), seed=SEED)
            .batch(BATCH_SIZE)
            .map(lambda imgs, lbls: cutmix_batch(imgs, lbls, class_weights_tensor),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        print("\n[4/5] Construyendo modelo...")
        model = build_cnn_cutmix()
        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
            WebProgressCallback(MODEL_KEY, DISPLAY_NAME, EPOCHS_BASE),
        ]

        print("\n  Entrenando...")
        history = model.fit(
            train_ds,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS_BASE,
            callbacks=callbacks,
            verbose=1
        )

        print("\n[5/5] Guardando modelo y curvas...")
        model.save(MODELS_DIR / "cnn_cutmix.keras")
        print(f"  → Modelo: {MODELS_DIR / 'cnn_cutmix.keras'}")

        plot_training_history(history, "curvas_cnn_cutmix.png")

        val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"\n  Resultado en validación → Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print("\n✓ Entrenamiento CNN + CutMix completado.")

        mark_status(MODEL_KEY, DISPLAY_NAME, "completed",
                    final_val_loss=float(val_loss), final_val_accuracy=float(val_acc))
    except Exception as e:
        mark_status(MODEL_KEY, DISPLAY_NAME, "failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
