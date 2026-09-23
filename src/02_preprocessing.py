"""
02 - Preprocesamiento y división del dataset
Carga imágenes, redimensiona a 224x224, normaliza [0,1] y divide en train/val/test estratificado.
Guarda los arrays en .npz para que los scripts de entrenamiento carguen rápido.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, SPLITS_DIR, CLASS_NAMES,
    IMG_SIZE, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


def load_and_resize_images():
    """Carga todas las imágenes, redimensiona y asigna etiquetas."""
    images, labels = [], []
    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = DATA_DIR / cls
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                images.append(np.array(img))
                labels.append(label_idx)
            except Exception as e:
                print(f"  ⚠ Error cargando {img_path.name}: {e}")

    X = np.array(images, dtype=np.float32) / 255.0  # Normalizar a [0, 1]
    y = np.array(labels, dtype=np.int32)
    return X, y


def stratified_split(X, y):
    """Split estratificado en train (70%), val (15%), test (15%)."""
    # Primero separar test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=y
    )

    # Luego separar val del resto
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        random_state=SEED,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    print("=" * 60)
    print("  PREPROCESAMIENTO — Carga, redimensionado y split")
    print("=" * 60)

    print(f"\n[1/3] Cargando imágenes y redimensionando a {IMG_SIZE}x{IMG_SIZE}...")
    X, y = load_and_resize_images()
    print(f"  → Total: {len(X)} imágenes, shape: {X.shape}")

    print("[2/3] División estratificada (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    print(f"  → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    for i, name in enumerate(CLASS_NAMES):
        tr = (y_train == i).sum()
        va = (y_val == i).sum()
        te = (y_test == i).sum()
        print(f"    {name:20s}: train={tr}, val={va}, test={te}")

    print("[3/3] Guardando splits en disco...")
    np.savez_compressed(
        SPLITS_DIR / "dataset_splits.npz",
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test
    )
    print(f"  → Guardado: {SPLITS_DIR / 'dataset_splits.npz'}")
    print(f"  → Tamaño: {(SPLITS_DIR / 'dataset_splits.npz').stat().st_size / 1e6:.1f} MB")

    print("\n✓ Preprocesamiento completado.")


if __name__ == "__main__":
    main()
