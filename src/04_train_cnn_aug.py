"""
04 - CNN con Aumento de Datos (Data Augmentation)
Misma arquitectura que la CNN base, pero con augmentation en tiempo real.
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

MODEL_KEY = "cnn_augmented"
DISPLAY_NAME = "CNN + Augmentation"

# Permite que la app web sobreescriba el número de épocas para una
# ejecución concreta sin tocar config.py (ver training_control.launch_training).
EPOCHS_BASE = int(os.environ.get("TFM_EPOCHS_OVERRIDE", EPOCHS_BASE))

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_augmentation_layer():
    """
    Capa de aumento de datos integrada en el modelo.

    Reducido respecto a la versión original: se quitó RandomFlip("vertical")
    (una TC de tórax no se ve al revés — un volteo vertical no es una
    variación anatómicamente plausible) y RandomTranslation (podía sacar la
    lesión del encuadre). RandomRotation y RandomZoom se redujeron en
    magnitud para que la augmentation sea más suave sobre imágenes médicas.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def build_cnn_augmented():
    """CNN de 3 bloques con augmentation integrada."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Augmentation — SIN fijar training=True explícito: así hereda el modo
    # (train/inference) de la llamada al modelo completo. Si se fija
    # training=True aquí, Keras lo deja "clavado" para este nodo del grafo y
    # la augmentation se sigue aplicando incluso en evaluate()/predict(),
    # contaminando las métricas de validación/test con aleatoriedad.
    x = build_augmentation_layer()(inputs)

    # Bloque 1 (LayerNormalization en vez de BatchNormalization: con pocos
    # batches por época, las estadísticas de BatchNorm no se estabilizan
    # bien — ver misma nota en 03_train_cnn_base.py)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Bloque 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Bloque 3
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Clasificador
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        # CategoricalCrossentropy + label_smoothing=0.1 (SparseCategorical
        # no soporta label_smoothing) — ver main() para el one-hot de y.
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    return model


def plot_augmented_samples(X_train):
    """Visualiza el efecto del augmentation sobre una imagen."""
    aug_layer = build_augmentation_layer()
    sample = X_train[0:1]  # Una imagen

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes[0, 0].imshow(sample[0])
    axes[0, 0].set_title("Original", fontweight="bold")
    axes[0, 0].axis("off")

    for i in range(1, 10):
        row, col = divmod(i, 5)
        augmented = aug_layer(sample, training=True)
        axes[row, col].imshow(augmented[0].numpy())
        axes[row, col].set_title(f"Augmented {i}")
        axes[row, col].axis("off")

    fig.suptitle("Efecto del Aumento de Datos sobre una imagen de muestra", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "muestras_augmentation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / 'muestras_augmentation.png'}")


def plot_training_history(history, filename):
    """Curvas de pérdida y exactitud."""
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

    fig.suptitle("CNN + Aumento de Datos — Curvas de Entrenamiento", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO — CNN con Aumento de Datos")
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
        class_weight_dict = dict(enumerate(class_weights))

        # CategoricalCrossentropy (para label_smoothing) necesita one-hot
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

        print("\n[2/5] Visualizando efecto del augmentation...")
        plot_augmented_samples(X_train)

        print("\n[3/5] Construyendo modelo...")
        model = build_cnn_augmented()
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

        print("\n[4/5] Entrenando...")
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS_BASE,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        print("\n[5/5] Guardando modelo y curvas...")
        model.save(MODELS_DIR / "cnn_augmented.keras")
        print(f"  → Modelo: {MODELS_DIR / 'cnn_augmented.keras'}")

        plot_training_history(history, "curvas_cnn_augmented.png")

        val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"\n  Resultado en validación → Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print("\n✓ Entrenamiento CNN + Augmentation completado.")

        mark_status(MODEL_KEY, DISPLAY_NAME, "completed",
                    final_val_loss=float(val_loss), final_val_accuracy=float(val_acc))
    except Exception as e:
        mark_status(MODEL_KEY, DISPLAY_NAME, "failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
