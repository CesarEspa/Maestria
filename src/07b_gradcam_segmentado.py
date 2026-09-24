"""
07b - Grad-CAM sobre el modelo Transfer Learning + Segmentación

Misma lógica que 07_gradcam.py, pero apuntando al modelo entrenado sobre
datos segmentados (05b_train_transfer_segmented.py) y a su conjunto de
test correspondiente (dataset_splits_segmented.npz) — para responder la
pregunta concreta de la Fase 10 / Tarea 2: ¿desaparece el artefacto de la
esquina saturada (Hallazgo 5.3 / README Hallazgo #6) al eliminar el fondo
y la mesa del escáner de la imagen de entrada?

Guarda en un archivo aparte (gradcam_grid_segmentado.png) para no
sobreescribir el grid del modelo original y poder comparar ambos.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from config import SPLITS_DIR, MODELS_DIR, GRADCAM_DIR, CLASS_LABELS, SEED
from gradcam_utils import (
    find_last_conv_layer_name as find_last_conv_layer,
    make_gradcam_heatmap as _make_gradcam_heatmap,
    overlay_heatmap,
)

np.random.seed(SEED)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    heatmap, _ = _make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    return heatmap


def generate_gradcam_grid(model, X_test, y_test, last_conv_name, out_path, n_per_class=3):
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

            col_orig = j * 2
            axes[cls_idx, col_orig].imshow(img)
            axes[cls_idx, col_orig].axis("off")
            if j == 0:
                axes[cls_idx, col_orig].set_ylabel(CLASS_LABELS[cls_idx],
                                                    fontsize=14, fontweight="bold")

            col_cam = j * 2 + 1
            axes[cls_idx, col_cam].imshow(overlay)
            axes[cls_idx, col_cam].axis("off")

            status = "✓" if pred_class == cls_idx else "✗"
            axes[cls_idx, col_cam].set_title(
                f"{status} Pred: {CLASS_LABELS[pred_class]} ({confidence:.1%})",
                fontsize=10, color="green" if pred_class == cls_idx else "red"
            )

    fig.suptitle("Mapas de Activación Grad-CAM — Transfer Learning + Segmentación",
                 fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {out_path}")


def main():
    print("=" * 60)
    print("  GRAD-CAM — Transfer Learning + Segmentación")
    print("=" * 60)

    print("\n[1/3] Cargando datos segmentados y modelo...")
    data = np.load(SPLITS_DIR / "dataset_splits_segmented.npz")
    X_test, y_test = data["X_test"], data["y_test"]
    model = keras.models.load_model(MODELS_DIR / "transfer_efficientnet_segmented.keras")

    print("\n[2/3] Identificando capa convolucional...")
    last_conv_name = find_last_conv_layer(model)
    print(f"  → Capa: {last_conv_name}")

    print("\n[3/3] Generando mapas Grad-CAM...")
    out_path = GRADCAM_DIR / "gradcam_grid_segmentado.png"
    generate_gradcam_grid(model, X_test, y_test, last_conv_name, out_path)

    print("\n✓ Grad-CAM (segmentado) completado.")


if __name__ == "__main__":
    main()
