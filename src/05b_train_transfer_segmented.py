"""
05b - Transfer Learning (EfficientNetB0) sobre datos SEGMENTADOS

Mismo modelo, mismos hiperparámetros y mismo procedimiento que
05_train_transfer.py, pero entrenado sobre
outputs/splits/dataset_splits_segmented.npz (generado por
02b_segmentation.py) en vez del dataset original — para poder comparar
de forma limpia el efecto de la segmentación, sin que ningún otro
cambio contamine la comparación.

Por instrucción explícita: SOLO se reentrena este modelo con datos
segmentados (no CNN Base ni CNN + Augmentation). Se guarda como un
modelo NUEVO (transfer_efficientnet_segmented.keras), sin sobreescribir
transfer_efficientnet.keras.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from config import (
    SPLITS_DIR, MODELS_DIR, FIGURES_DIR,
    IMG_SIZE, NUM_CLASSES, BATCH_SIZE,
    EPOCHS_TRANSFER, LEARNING_RATE_TRANSFER, PATIENCE, SEED
)
from progress_tracker import WebProgressCallback, mark_status

MODEL_KEY = "transfer_efficientnet_segmented"
DISPLAY_NAME = "Transfer Learning + Segmentación"
PHASE1_EPOCHS = 25
UNFROZEN_LAYERS = 50  # igual que 05_train_transfer.py, para que la comparación sea limpia

EPOCHS_TRANSFER = int(os.environ.get("TFM_EPOCHS_OVERRIDE", EPOCHS_TRANSFER))

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_augmentation_layer():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def build_transfer_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = build_augmentation_layer()(inputs)
    x = keras.applications.efficientnet.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def plot_training_history(history, history_ft, filename):
    loss = history.history["loss"] + history_ft.history["loss"]
    val_loss = history.history["val_loss"] + history_ft.history["val_loss"]
    acc = history.history["accuracy"] + history_ft.history["accuracy"]
    val_acc = history.history["val_accuracy"] + history_ft.history["val_accuracy"]
    ft_start = len(history.history["loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(loss, label="Entrenamiento", linewidth=2)
    ax1.plot(val_loss, label="Validación", linewidth=2)
    ax1.axvline(ft_start, color="gray", linestyle="--", alpha=0.7, label="Inicio fine-tuning")
    ax1.set_title("Pérdida (Loss)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(acc, label="Entrenamiento", linewidth=2)
    ax2.plot(val_acc, label="Validación", linewidth=2)
    ax2.axvline(ft_start, color="gray", linestyle="--", alpha=0.7, label="Inicio fine-tuning")
    ax2.set_title("Exactitud (Accuracy)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Transfer Learning + Segmentación — Curvas de Entrenamiento", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO — Transfer Learning + Segmentación")
    print("=" * 60)

    total_epochs_budget = PHASE1_EPOCHS + EPOCHS_TRANSFER
    mark_status(MODEL_KEY, DISPLAY_NAME, "running", current_epoch=0,
                total_epochs=total_epochs_budget, phase="Fase 1 - Cabeza congelada",
                history=[])

    try:
        print("\n[1/5] Cargando datos SEGMENTADOS (02b_segmentation.py)...")
        seg_path = SPLITS_DIR / "dataset_splits_segmented.npz"
        if not seg_path.exists():
            raise FileNotFoundError(
                f"No se encontró {seg_path}. Ejecuta primero 02b_segmentation.py."
            )
        data = np.load(seg_path)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        print(f"  → Train: {X_train.shape}, Val: {X_val.shape}")

        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        print("\n[2/5] Construyendo modelo (base congelada)...")
        model, base_model = build_transfer_model()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_TRANSFER),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()

        callbacks_phase1 = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            WebProgressCallback(MODEL_KEY, DISPLAY_NAME, PHASE1_EPOCHS,
                                 phase="Fase 1 - Cabeza congelada",
                                 epoch_offset=0,
                                 total_epochs_all_phases=total_epochs_budget),
        ]

        print("\n[3/5] Fase 1 — Entrenando cabeza clasificadora...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=PHASE1_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks_phase1,
            verbose=1
        )

        print(f"\n[4/5] Fase 2 — Fine-tuning (últimas {UNFROZEN_LAYERS} capas)...")
        mark_status(MODEL_KEY, DISPLAY_NAME, "running",
                    phase="Fase 2 - Fine-tuning")
        base_model.trainable = True
        for layer in base_model.layers[:-UNFROZEN_LAYERS]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_TRANSFER / 10),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        phase1_completed = len(history.history.get("loss", []))

        callbacks_phase2 = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
            ),
            WebProgressCallback(MODEL_KEY, DISPLAY_NAME, EPOCHS_TRANSFER,
                                 phase="Fase 2 - Fine-tuning",
                                 epoch_offset=phase1_completed,
                                 total_epochs_all_phases=total_epochs_budget),
        ]

        history_ft = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS_TRANSFER,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks_phase2,
            verbose=1
        )

        print("\n[5/5] Guardando modelo y curvas...")
        model.save(MODELS_DIR / "transfer_efficientnet_segmented.keras")
        print(f"  → Modelo: {MODELS_DIR / 'transfer_efficientnet_segmented.keras'}")

        plot_training_history(history, history_ft, "curvas_transfer_learning_segmentado.png")

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\n  Resultado en validación → Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print("\n✓ Entrenamiento Transfer Learning + Segmentación completado.")

        mark_status(MODEL_KEY, DISPLAY_NAME, "completed",
                    final_val_loss=float(val_loss), final_val_accuracy=float(val_acc))
    except Exception as e:
        mark_status(MODEL_KEY, DISPLAY_NAME, "failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
