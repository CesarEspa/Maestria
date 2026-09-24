"""
02b - Segmentación pulmonar clásica (sin deep learning)

Objetivo específico 1 de la propuesta 1 del TFM: "preprocesar... implementando
técnicas de segmentación...". Aísla los campos pulmonares en cada imagen ya
preprocesada por 02_preprocessing.py, poniendo a cero todo lo que no sea
pulmón (mesa del escáner, contorno corporal, huesos, mediastino).

Pipeline (validado visualmente sobre 24+150 imágenes de muestra antes de
aplicarlo a todo el dataset, ver PROYECTO_CLAUDE_CONTEXTO.txt Fase 10):
  1. Escala de grises + umbral de Otsu (no se asumen unidades Hounsfield:
     estas imágenes son JPG ya exportados, no DICOM crudo).
  2. Silueta corporal = el componente CLARO más grande (la mesa del escáner
     también es clara, pero mucho más pequeña que el cuerpo, así que nunca
     gana esta selección) + cierre morfológico + relleno de huecos, para
     tener una región corporal sólida.
  3. Candidatos a pulmón = píxeles OSCUROS que caen DENTRO de esa silueta
     corporal. Restringir la búsqueda al interior del cuerpo es lo que
     evita que el hueco bajo la mesa del escáner se cuele como si fuera un
     pulmón (el fallo real que se encontró y corrigió en el primer intento).
  4. Apertura (quita ruido puntual) + cierre (rellena huecos pequeños) +
     descartar componentes diminutos, y quedarse con los 2 componentes
     conexos más grandes = los dos pulmones.
  5. Cierre final más agresivo + relleno de huecos, para no amputar masas
     brillantes que tocan el borde del pulmón (crean una "mordida" en vez
     de un agujero cerrado si no se hace este último paso).

Guarda outputs/splits/dataset_splits_segmented.npz con LA MISMA partición
train/val/test y las mismas etiquetas que dataset_splits.npz (se deriva
directamente de ese archivo, no se vuelve a hacer el split — así las
particiones coinciden exactamente, sin depender de que la semilla
reproduzca el mismo orden dos veces).
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing, disk, remove_small_objects
from scipy.ndimage import binary_fill_holes

from config import SPLITS_DIR, FIGURES_DIR, CLASS_LABELS

# Las llamadas siguen la API vigente en scikit-image==0.26.0 (versión fija
# del proyecto); algunos nombres de parámetro están deprecados a favor de
# otros que llegarán en 2.0, pero eso no afecta el resultado aquí.
warnings.filterwarnings("ignore", category=FutureWarning)

MIN_COMPONENT_PX = 150   # descarta motas de ruido tras la apertura
BODY_CLOSE_RADIUS = 6    # cierre de la silueta corporal
CANDIDATE_OPEN_RADIUS = 2
CANDIDATE_CLOSE_RADIUS = 4
FINAL_CLOSE_RADIUS = 9   # cierre final, agresivo a propósito (ver punto 5)


def _largest_component(mask):
    labeled = label(mask)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    regions = regionprops(labeled)
    biggest = max(regions, key=lambda r: r.area)
    return labeled == biggest.label


def segment_lungs(img01):
    """
    img01: array (H, W, 3) float32 en [0, 1] (una imagen ya preprocesada por
    02_preprocessing.py). Devuelve (mask bool HxW, imagen segmentada H,W,3).
    """
    gray = np.mean(img01, axis=2)
    gray_u8 = (gray * 255).astype(np.uint8)
    thresh = threshold_otsu(gray_u8)

    # Silueta corporal sólida (ver punto 2 del docstring del módulo)
    bright_mask = gray_u8 >= thresh
    body = _largest_component(bright_mask)
    body = closing(body, disk(BODY_CLOSE_RADIUS))
    body_filled = binary_fill_holes(body)

    # Candidatos a pulmón: oscuros y dentro del cuerpo (punto 3)
    dark_mask = gray_u8 < thresh
    candidates = dark_mask & body_filled
    candidates = clear_border(candidates)
    candidates = opening(candidates, disk(CANDIDATE_OPEN_RADIUS))
    candidates = closing(candidates, disk(CANDIDATE_CLOSE_RADIUS))
    candidates = remove_small_objects(candidates, min_size=MIN_COMPONENT_PX)

    # Los 2 componentes más grandes = los dos pulmones (punto 4)
    labeled = label(candidates)
    regions = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
    mask = np.zeros_like(candidates, dtype=bool)
    for r in regions[:2]:
        mask |= (labeled == r.label)

    # Cierre final + relleno (punto 5)
    mask = closing(mask, disk(FINAL_CLOSE_RADIUS))
    mask = binary_fill_holes(mask)
    mask = mask & body_filled

    segmented = img01 * mask[:, :, None]
    return mask, segmented.astype(np.float32)


def segment_batch(X, label_prefix=""):
    """Segmenta un array (N,H,W,3) completo, con progreso cada 100 imágenes."""
    n = X.shape[0]
    out = np.empty_like(X)
    for i in range(n):
        _, seg = segment_lungs(X[i])
        out[i] = seg
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  {label_prefix}: {i + 1}/{n}")
    return out


def plot_examples(X, y, filename="ejemplos_segmentacion.png"):
    """Rejilla: original / máscara / segmentada, una fila por clase."""
    n_classes = len(CLASS_LABELS)
    fig, axes = plt.subplots(n_classes, 3, figsize=(9, 3 * n_classes))
    for row, cls in enumerate(range(n_classes)):
        idx = int(np.where(y == cls)[0][0])
        img = X[idx]
        mask, seg = segment_lungs(img)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original — {CLASS_LABELS[cls]}")
        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("Máscara binaria")
        axes[row, 2].imshow(seg)
        axes[row, 2].set_title("Segmentada")
        for c in range(3):
            axes[row, c].axis("off")
    fig.suptitle("Segmentación pulmonar clásica (Otsu + morfología) — ejemplos por clase",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def main():
    print("=" * 60)
    print("  02b - SEGMENTACIÓN PULMONAR CLÁSICA")
    print("=" * 60)

    print("\n[1/3] Cargando dataset_splits.npz...")
    data = np.load(SPLITS_DIR / "dataset_splits.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"  → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n[2/3] Generando ejemplos_segmentacion.png (uno por clase)...")
    plot_examples(X_train, y_train)

    print("\n[3/3] Segmentando los 3 splits (esto tarda unos minutos)...")
    X_train_seg = segment_batch(X_train, "train")
    X_val_seg = segment_batch(X_val, "val")
    X_test_seg = segment_batch(X_test, "test")

    out_path = SPLITS_DIR / "dataset_splits_segmented.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train_seg, y_train=y_train,
        X_val=X_val_seg, y_val=y_val,
        X_test=X_test_seg, y_test=y_test,
    )
    print(f"\n  → Guardado: {out_path}")
    print("\n✓ Segmentación completada. Mismas particiones y etiquetas que "
          "dataset_splits.npz (derivadas directamente de él, no se rehizo el split).")


if __name__ == "__main__":
    main()
