"""
07 - Grad-CAM: Mapas de activación de clases
Genera visualizaciones de las regiones que el mejor modelo usa para clasificar.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from config import (
    SPLITS_DIR, MODELS_DIR, FIGURES_DIR, GRADCAM_DIR,
    CLASS_LABELS, IMG_SIZE, SEED
)
from gradcam_utils import (
    find_last_conv_layer_name as find_last_conv_layer,
    make_gradcam_heatmap as _make_gradcam_heatmap,
    overlay_heatmap,
)

np.random.seed(SEED)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Genera el mapa de calor Grad-CAM (wrapper que descarta el índice de clase)."""
    heatmap, _ = _make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    return heatmap


def generate_gradcam_grid(model, X_test, y_test, last_conv_name, n_per_class=3):
    """Genera grid de Grad-CAM para muestras de cada clase."""
    fig, axes = plt.subplots(3, n_per_class * 2, figsize=(4 * n_per_class * 2, 12))

    for cls_idx in range(3):
        cls_mask = y_test == cls_idx
        cls_indices = np.where(cls_mask)[0]
        selected = np.random.choice(cls_indices, size=min(n_per_class, len(cls_indices)), replace=False)

        for j, idx in enumerate(selected):
            img = X_test[idx]
            img_array = np.expand_dims(img, axis=0)

            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred[0])
            confidence = pred[0][pred_class]

            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_name)
                overlay = overlay_heatmap(img, heatmap)
            except Exception as e:
                print(f"  ⚠ Grad-CAM falló para imagen {idx}: {e}")
                overlay = img

            # Imagen original
            col_orig = j * 2
            axes[cls_idx, col_orig].imshow(img)
            axes[cls_idx, col_orig].axis("off")
            if j == 0:
                axes[cls_idx, col_orig].set_ylabel(CLASS_LABELS[cls_idx],
                                                    fontsize=14, fontweight="bold")

            # Grad-CAM
            col_cam = j * 2 + 1
            axes[cls_idx, col_cam].imshow(overlay)
            axes[cls_idx, col_cam].axis("off")

            status = "✓" if pred_class == cls_idx else "✗"
            axes[cls_idx, col_cam].set_title(
                f"{status} Pred: {CLASS_LABELS[pred_class]} ({confidence:.1%})",
                fontsize=10, color="green" if pred_class == cls_idx else "red"
            )

    fig.suptitle("Mapas de Activación Grad-CAM — Mejor Modelo", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(GRADCAM_DIR / "gradcam_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {GRADCAM_DIR / 'gradcam_grid.png'}")


def main():
    print("=" * 60)
    print("  GRAD-CAM — Explicabilidad del Modelo")
    print("=" * 60)

    print("\n[1/3] Cargando datos y modelo...")
    data = np.load(SPLITS_DIR / "dataset_splits.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    # Intentar cargar el mejor modelo (transfer learning primero)
    model_path = MODELS_DIR / "transfer_efficientnet.keras"
    model_name = "Transfer Learning"
    if not model_path.exists():
        model_path = MODELS_DIR / "cnn_augmented.keras"
        model_name = "CNN + Augmentation"
    if not model_path.exists():
        model_path = MODELS_DIR / "cnn_base.keras"
        model_name = "CNN Base"

    print(f"  → Usando modelo: {model_name}")
    model = keras.models.load_model(model_path)

    # Encontrar última capa convolucional
    print("\n[2/3] Identificando capa convolucional...")
    last_conv_name = find_last_conv_layer(model)

    if last_conv_name is None:
        print("  ⚠ No se encontró capa convolucional. Usando la CNN augmented...")
        model = keras.models.load_model(MODELS_DIR / "cnn_augmented.keras")
        last_conv_name = find_last_conv_layer(model)

    print(f"  → Capa: {last_conv_name}")

    print("\n[3/3] Generando mapas Grad-CAM...")
    generate_gradcam_grid(model, X_test, y_test, last_conv_name)

    print("\n✓ Grad-CAM completado.")


if __name__ == "__main__":
    main()
