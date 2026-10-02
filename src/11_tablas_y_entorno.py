"""
11 - Tablas descriptivas, figura de preprocesamiento y entorno (Tarea 7,
correcciones del TFM)

1. outputs/experimentos/imagenes_por_subconjunto.csv: imágenes y grupos por
   clase en train/val/test, para cada semilla y tipo de partición (grupos e
   imagen). Fila "TOTAL" por semilla/tipo con, para la semilla 0 y partición
   por grupos, el número de muestras vistas durante el entrenamiento
   (imágenes de train x épocas efectivas) en M1, M2 y M3 — el aumento es en
   línea: no aumenta el número de imágenes distintas, pero cada época ve
   versiones transformadas.
2. outputs/figures/pipeline_preprocesamiento.png: un ejemplo por clase, con
   las columnas original, redimensionada y normalizada, máscara, segmentada,
   aumento geométrico y CutMix.
3. outputs/experimentos/entorno.json: SO, CPU, RAM, GPU (y si TensorFlow la
   usó), y versiones de Python y las librerías clave del pipeline.
"""
import json
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import protocolo as pr
from protocolo import EXPERIMENTOS_DIR, RUNS_DIR
from config import FIGURES_DIR, CLASS_LABELS

SEMILLAS = [0, 1, 2]
TIPOS_PARTICION = ["grupos", "imagen"]
MODELOS_MUESTRAS_VISTAS = ["M1", "M2", "M3"]


# ───────────────── 1. Imágenes y grupos por subconjunto ────────────────────

def _grupos_df():
    return pd.read_csv(pr.SPLITS_DIR / "grupos_paciente.csv")


def _conteo_filas(tipo, seed, df_grupos, y):
    tr, va, te = pr.particion(seed, tipo=tipo)
    filas = []
    for clase_idx, clase_lbl in enumerate(CLASS_LABELS):
        fila = {"tipo_particion": tipo, "semilla": seed, "clase": clase_lbl}
        for nombre_subset, idx_subset in (("train", tr), ("val", va), ("test", te)):
            mask_clase = y[idx_subset] == clase_idx
            fila[nombre_subset] = int(mask_clase.sum())
            if tipo == "grupos":
                grupos_subset = df_grupos.iloc[idx_subset[mask_clase]]["grupo"].unique()
                fila[f"grupos_{nombre_subset}"] = int(len(grupos_subset))
            else:
                fila[f"grupos_{nombre_subset}"] = None
        filas.append(fila)

    fila_total = {"tipo_particion": tipo, "semilla": seed, "clase": "TOTAL"}
    for nombre_subset, idx_subset in (("train", tr), ("val", va), ("test", te)):
        fila_total[nombre_subset] = int(len(idx_subset))
        if tipo == "grupos":
            fila_total[f"grupos_{nombre_subset}"] = int(df_grupos.iloc[idx_subset]["grupo"].nunique())
        else:
            fila_total[f"grupos_{nombre_subset}"] = None
    filas.append(fila_total)
    return filas, tr


def _muestras_vistas(modelo, hp, seed=0):
    """n_train x épocas_efectivas (sumadas si hay varias fases), leído del
    hp.json ya guardado por entrenar() (Tareas 3/4)."""
    run_dir = RUNS_DIR / pr.nombre_run(modelo, hp, seed)
    hp_path = run_dir / "hp.json"
    if not hp_path.exists():
        return None
    with open(hp_path, encoding="utf-8") as f:
        info = json.load(f)
    epocas_total = sum(info["epocas_efectivas"])
    tr, _, _ = pr.particion(seed, tipo="grupos")
    return int(len(tr) * epocas_total)


def generar_imagenes_por_subconjunto(config_final):
    print("--- imagenes_por_subconjunto.csv ---")
    df_grupos = _grupos_df()
    _, y = pr.cargar_X_y(segmentado=False)

    todas_filas = []
    for tipo in TIPOS_PARTICION:
        for seed in SEMILLAS:
            filas, _ = _conteo_filas(tipo, seed, df_grupos, y)
            todas_filas.extend(filas)

    df = pd.DataFrame(todas_filas)

    for modelo in MODELOS_MUESTRAS_VISTAS:
        col = f"muestras_vistas_{modelo}"
        df[col] = None
        hp = config_final.get(modelo)
        if hp is None:
            continue
        valor = _muestras_vistas(modelo, hp, seed=0)
        mask = (df["tipo_particion"] == "grupos") & (df["semilla"] == 0) & (df["clase"] == "TOTAL")
        df.loc[mask, col] = valor

    df.to_csv(EXPERIMENTOS_DIR / "imagenes_por_subconjunto.csv", index=False)
    print(f"  → {EXPERIMENTOS_DIR / 'imagenes_por_subconjunto.csv'}")
    return df


# ──────────────────── 2. Figura de preprocesamiento ─────────────────────────

def generar_pipeline_preprocesamiento(filename="pipeline_preprocesamiento.png"):
    print("--- pipeline_preprocesamiento.png ---")
    import importlib
    seg_mod = importlib.import_module("02b_segmentation")
    cutmix_mod = importlib.import_module("04b_train_cnn_cutmix")

    df_grupos = _grupos_df()
    X, y = pr.cargar_X_y(segmentado=False)
    X_seg, masks = pr.cargar_segmentadas()
    aug_layer = pr.build_augmentation_geometrico()

    columnas = ["Original", "Redim. + normalizada", "Máscara", "Segmentada",
                "Aumento geométrico", "CutMix"]
    fig, axes = plt.subplots(3, len(columnas), figsize=(3 * len(columnas), 9))

    rng = np.random.default_rng(0)
    for fila, clase_lbl in enumerate(CLASS_LABELS):
        idx = int(np.where(y == fila)[0][0])
        img_original_raw = _cargar_imagen_cruda(df_grupos.iloc[idx])
        img_norm = X[idx]
        mask = masks[idx]
        img_seg = X_seg[idx]

        img_batch = tf.constant(img_norm[np.newaxis, ...], dtype=tf.float32)
        img_aug = aug_layer(img_batch, training=True)[0].numpy()

        otro_idx = int(rng.choice(np.where(y != fila)[0]))
        batch2 = tf.constant(np.stack([img_norm, X[otro_idx]]), dtype=tf.float32)
        labels2 = tf.one_hot([fila, int(y[otro_idx])], depth=3)
        cw = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
        mezclada, _, _ = cutmix_mod.cutmix_batch(batch2, labels2, cw)
        img_cutmix = mezclada[0].numpy()

        imagenes = [img_original_raw, img_norm, mask, img_seg, img_aug, img_cutmix]
        cmaps = [None, None, "gray", None, None, None]
        for col, (img, cmap) in enumerate(zip(imagenes, cmaps)):
            ax = axes[fila, col]
            ax.imshow(np.clip(img, 0, 1) if cmap is None else img, cmap=cmap)
            ax.axis("off")
            if fila == 0:
                ax.set_title(columnas[col], fontsize=11)
            if col == 0:
                ax.text(-0.08, 0.5, clase_lbl, transform=ax.transAxes, fontsize=13,
                        fontweight="bold", rotation=90, va="center", ha="center")

    fig.suptitle("Pipeline de preprocesamiento — un ejemplo por clase", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIGURES_DIR / filename}")


def _cargar_imagen_cruda(fila_grupo):
    from PIL import Image
    p = pr.DATA_DIR / pr.LABEL_TO_FOLDER[fila_grupo["clase"]] / fila_grupo["archivo"]
    img = Image.open(p).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


# ─────────────────────────────── 3. Entorno ──────────────────────────────────

def generar_entorno_json():
    print("--- entorno.json ---")
    import sklearn
    import skimage
    import cv2
    import statsmodels
    import psutil

    gpus = tf.config.list_physical_devices("GPU")
    info = {
        "sistema_operativo": f"{platform.system()} {platform.release()} ({platform.version()})",
        "arquitectura": platform.machine(),
        "cpu": platform.processor() or platform.uname().processor,
        "cpu_nucleos_fisicos": psutil.cpu_count(logical=False),
        "cpu_nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "gpu_detectada_por_tensorflow": [str(g) for g in gpus],
        "tensorflow_uso_gpu": len(gpus) > 0,
        "nota_gpu": ("Sin GPU utilizable por TensorFlow en este equipo pese a tener una "
                     "GPU NVIDIA física: desde TF 2.11, pip install tensorflow no trae "
                     "soporte nativo de CUDA en Windows (ver README, Limitaciones)."
                     if len(gpus) == 0 else ""),
        "versiones": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scikit-image": skimage.__version__,
            "opencv-python": cv2.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    with open(EXPERIMENTOS_DIR / "entorno.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  → {EXPERIMENTOS_DIR / 'entorno.json'}")
    return info


# ───────────────────────────────── main ──────────────────────────────────────

def main():
    print("=" * 70)
    print("TAREA 7 — Tablas descriptivas, figura de preprocesamiento y entorno")
    print("=" * 70)

    with open(EXPERIMENTOS_DIR / "hiperparametros_finales.json", encoding="utf-8") as f:
        config_final = json.load(f)

    generar_imagenes_por_subconjunto(config_final)
    generar_pipeline_preprocesamiento()
    generar_entorno_json()

    print("\n✓ Tarea 7 completada.")


if __name__ == "__main__":
    main()
