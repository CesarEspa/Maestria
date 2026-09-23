"""
01 - Análisis Exploratorio del Dataset IQ-OTH/NCCD
Genera figuras para el TFM: distribución de clases, muestras, distribución de tamaños.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter

from config import (
    DATA_DIR, FIGURES_DIR, CLASS_NAMES, CLASS_LABELS, SEED
)

np.random.seed(SEED)
sns.set_theme(style="whitegrid", font_scale=1.2)


def load_image_paths():
    """Carga todas las rutas de imágenes organizadas por clase."""
    data = {}
    for cls in CLASS_NAMES:
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"ERROR: No se encontró la carpeta '{cls_dir}'")
            print(f"Asegúrate de descargar el dataset en: {DATA_DIR}")
            sys.exit(1)
        # Filtrar solo archivos de imagen
        imgs = [p for p in sorted(cls_dir.iterdir())
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        data[cls] = imgs
    return data


def plot_class_distribution(data):
    """Gráfico de barras con la distribución de clases."""
    counts = [len(data[c]) for c in CLASS_NAMES]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_LABELS, counts, color=colors, edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.set_xlabel("Clase")
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Distribución de imágenes por clase — Dataset IQ-OTH/NCCD")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribucion_clases.png", dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / 'distribucion_clases.png'}")
    return counts


def plot_sample_images(data, n=3):
    """Muestra n imágenes de cada clase."""
    fig, axes = plt.subplots(NUM_CLASSES := 3, n, figsize=(4 * n, 4 * NUM_CLASSES))
    for i, cls in enumerate(CLASS_NAMES):
        samples = np.random.choice(len(data[cls]), size=n, replace=False)
        for j, idx in enumerate(samples):
            img = Image.open(data[cls][idx]).convert("RGB")
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(CLASS_LABELS[i], fontsize=14, fontweight="bold")

    fig.suptitle("Muestras representativas por clase", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "muestras_por_clase.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / 'muestras_por_clase.png'}")


def plot_image_sizes(data):
    """Distribución de tamaños de imagen (ancho x alto)."""
    widths, heights = [], []
    for cls in CLASS_NAMES:
        for p in data[cls]:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths, bins=30, color="#3498db", edgecolor="black")
    axes[0].set_title("Distribución de anchos")
    axes[0].set_xlabel("Píxeles")
    axes[0].set_ylabel("Frecuencia")

    axes[1].hist(heights, bins=30, color="#e74c3c", edgecolor="black")
    axes[1].set_title("Distribución de altos")
    axes[1].set_xlabel("Píxeles")

    fig.suptitle("Tamaños originales de las imágenes", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribucion_tamanos.png", dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / 'distribucion_tamanos.png'}")

    print(f"\n  Ancho  — min: {min(widths)}, max: {max(widths)}, media: {np.mean(widths):.0f}")
    print(f"  Alto   — min: {min(heights)}, max: {max(heights)}, media: {np.mean(heights):.0f}")


def plot_pixel_intensity(data, n_per_class=50):
    """Distribución de intensidad de píxeles por clase."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, cls in enumerate(CLASS_NAMES):
        idxs = np.random.choice(len(data[cls]), size=min(n_per_class, len(data[cls])), replace=False)
        all_pixels = []
        for idx in idxs:
            img = np.array(Image.open(data[cls][idx]).convert("L"))  # Escala de grises
            all_pixels.extend(img.flatten().tolist())

        ax.hist(all_pixels, bins=50, alpha=0.5, color=colors[i],
                label=CLASS_LABELS[i], density=True)

    ax.set_xlabel("Intensidad de píxel (0-255)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de intensidad de píxeles por clase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribucion_intensidad.png", dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / 'distribucion_intensidad.png'}")


def main():
    print("=" * 60)
    print("  ANÁLISIS EXPLORATORIO — IQ-OTH/NCCD Lung Cancer Dataset")
    print("=" * 60)

    data = load_image_paths()

    total = sum(len(v) for v in data.values())
    print(f"\n  Total de imágenes: {total}")
    for cls, label in zip(CLASS_NAMES, CLASS_LABELS):
        n = len(data[cls])
        print(f"    {label:12s}: {n:4d} ({100 * n / total:.1f}%)")

    print("\n[1/4] Distribución de clases...")
    plot_class_distribution(data)

    print("[2/4] Muestras representativas...")
    plot_sample_images(data)

    print("[3/4] Distribución de tamaños...")
    plot_image_sizes(data)

    print("[4/4] Distribución de intensidad de píxeles...")
    plot_pixel_intensity(data)

    print("\n✓ EDA completado. Figuras guardadas en:", FIGURES_DIR)


if __name__ == "__main__":
    main()
