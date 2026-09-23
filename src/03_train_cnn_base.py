"""
03 - CNN Base (sin aumento de datos)
Arquitectura sencilla de 3 bloques conv + dense.
Sirve como línea base para comparar con augmentation y transfer learning.
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

MODEL_KEY = "cnn_base"
DISPLAY_NAME = "CNN Base"

# Permite que la app web sobreescriba el número de épocas para una
# ejecución concreta sin tocar config.py (ver training_control.launch_training).
EPOCHS_BASE = int(os.environ.get("TFM_EPOCHS_OVERRIDE", EPOCHS_BASE))

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_cnn_base():
    """
    CNN de 3 bloques convolucionales.

    Cambios respecto a la versión original (ver README.md / Hallazgos para
    el diagnóstico completo del colapso de entrenamiento):
      - LayerNormalization en vez de BatchNormalization: con solo 24 batches
        por época, las estadísticas de BatchNorm (media/varianza por batch)
        no llegaban a estabilizarse. LayerNormalization normaliza por
        muestra (no depende de estadísticas acumuladas del batch), lo que
        la hace más robusta con datasets pequeños y pocos pasos por época.
      - learning_rate más bajo (ver config.LEARNING_RATE): el valor anterior
        (1e-3) era demasiado alto para este dataset.
      - Loss con label_smoothing=0.1: suaviza las etiquetas objetivo
        (en vez de exigir 100% de confianza en la clase correcta), lo que
        penaliza menos el error y reduce el sobreajuste temprano. Requiere
        CategoricalCrossentropy (SparseCategoricalCrossentropy no soporta
        label_smoothing en esta versión de Keras), por eso main() convierte
        las etiquetas a one-hot antes de entrenar.
    """
    model = keras.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.LayerNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.LayerNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.LayerNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Clasificador
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    return model


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

    fig.suptitle("CNN Base — Curvas de Entrenamiento", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO — CNN Base (sin aumento de datos)")
    print("=" * 60)

    mark_status(MODEL_KEY, DISPLAY_NAME, "running", current_epoch=0,
                total_epochs=EPOCHS_BASE, phase="", history=[])

    try:
        # Cargar datos
        print("\n[1/4] Cargando datos preprocesados...")
        data = np.load(SPLITS_DIR / "dataset_splits.npz")
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        print(f"  → Train: {X_train.shape}, Val: {X_val.shape}")

        # Pesos de clase para manejar desbalance (a partir de las etiquetas
        # sparse originales; el diccionario resultante funciona igual con
        # fit() aunque y sea one-hot, ver más abajo)
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        print(f"  → Pesos de clase: {class_weight_dict}")

        # CategoricalCrossentropy (para poder usar label_smoothing) necesita
        # etiquetas one-hot en vez de índices de clase.
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

        # Construir modelo
        print("\n[2/4] Construyendo modelo...")
        model = build_cnn_base()
        model.summary()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
            WebProgressCallback(MODEL_KEY, DISPLAY_NAME, EPOCHS_BASE),
        ]

        # Entrenar
        print("\n[3/4] Entrenando...")
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS_BASE,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        # Guardar
        print("\n[4/4] Guardando modelo y curvas...")
        model.save(MODELS_DIR / "cnn_base.keras")
        print(f"  → Modelo: {MODELS_DIR / 'cnn_base.keras'}")

        plot_training_history(history, "curvas_cnn_base.png")

        # Evaluación rápida en validación
        val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"\n  Resultado en validación → Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print("\n✓ Entrenamiento CNN Base completado.")

        mark_status(MODEL_KEY, DISPLAY_NAME, "completed",
                    final_val_loss=float(val_loss), final_val_accuracy=float(val_acc))
    except Exception as e:
        mark_status(MODEL_KEY, DISPLAY_NAME, "failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
