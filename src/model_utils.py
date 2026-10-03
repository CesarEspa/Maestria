"""
Utilidades de inferencia compartidas por la app web.

Sirve los modelos del PROTOCOLO FINAL del TFM — M1 (CNN desde cero), M4 y M5
(EfficientNetB0, con y sin segmentación) y la línea base HOG+SVM, todos de la
semilla 0, partición por grupos de paciente (ver README.md, "Protocolo
corregido") — NO los 4 modelos preliminares de partición por imagen (esos se
siguen entrenando/sirviendo desde la pestaña "Entrenamiento", ver
training_control.py, que apunta a outputs/legacy_particion_imagen/).

Todo el preprocesamiento (resize, segmentación, HOG) reutiliza exactamente
las mismas funciones que usaron el entrenamiento y 09_evaluacion_final.py
(protocolo.py, 02b_segmentation.py, 08_baseline_ml.py) — no se reimplementa
nada aquí, para garantizar que la app clasifica igual que el protocolo
evaluado en el TFM. Ver src/verificar_app_vs_protocolo.py para la
verificación numérica de esta equivalencia.
"""
import importlib
from pathlib import Path

import numpy as np
from PIL import Image

from config import PROJECT_ROOT, IMG_SIZE

RUNS_DIR = PROJECT_ROOT / "outputs" / "experimentos" / "runs"

import protocolo as pr  # noqa: E402  (reutiliza cargar_X_y / particion / etc.)
seg_mod = importlib.import_module("02b_segmentation")  # noqa: E402
baseline_mod = importlib.import_module("08_baseline_ml")  # noqa: E402

# Los 4 modelos del protocolo final, semilla 0. M5 es el modelo recomendado
# (índice de Youden — ver outputs/experimentos/seleccion_modelo_recomendado.json).
MODEL_SPECS = [
    {"key": "m5", "name": "M5 · EfficientNetB0 + segmentación (recomendado)",
     "run_dir": "M5_aumentogeometrico_capas_descongeladas20_lr_ajuste0.0001_seed0",
     "file": "modelo.keras", "type": "keras", "segmentado": True},
    {"key": "m4", "name": "M4 · EfficientNetB0",
     "run_dir": "M4_aumentogeometrico_capas_descongeladas20_lr_ajuste0.0001_seed0",
     "file": "modelo.keras", "type": "keras", "segmentado": False},
    {"key": "m1", "name": "M1 · CNN desde cero",
     "run_dir": "M1_dropout_bloques0.25_lr0.0001_seed0",
     "file": "modelo.keras", "type": "keras", "segmentado": False},
    {"key": "svm", "name": "Línea base HOG + SVM",
     "run_dir": "SVM_grupos_seed0",
     "file": "modelo.joblib", "type": "sklearn", "segmentado": False},
]


def preprocess_pil_image(pil_img):
    """
    Idéntico al preprocesamiento de protocolo.cargar_imagenes(): RGB, resize
    a IMG_SIZE x IMG_SIZE con LANCZOS, normalizado a [0, 1].
    Devuelve un array float32 (IMG_SIZE, IMG_SIZE, 3).
    """
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def available_models():
    """Los 4 modelos del protocolo final, con su ruta real en
    outputs/experimentos/runs/ y si el archivo existe en disco."""
    models = []
    for spec in MODEL_SPECS:
        path = RUNS_DIR / spec["run_dir"] / spec["file"]
        models.append({**spec, "path": path, "exists": path.exists()})
    return models


def model_input_image(spec, img_array01):
    """
    Imagen que REALMENTE entra al modelo. Para M5 (segmentado=True), aplica
    02b_segmentation.segment_lungs() — la misma función que usa
    protocolo.cargar_segmentadas() para construir X_seg — seguida de la
    misma cuantización a uint8/255 que esa función aplica antes de cachear
    (necesaria para reproducir bit a bit las predicciones de pred_test.npz,
    que se calcularon sobre la versión cuantizada). Para el resto, la imagen
    sin cambios.
    """
    if not spec["segmentado"]:
        return img_array01
    _, seg = seg_mod.segment_lungs(img_array01)
    seg_u8 = (np.clip(seg, 0, 1) * 255).astype(np.uint8)
    return seg_u8.astype(np.float32) / 255.0


def predict_keras(model, img_array01, spec):
    """Predice probabilidades por clase con un modelo Keras del protocolo
    final. Devuelve (proba, imagen_que_entro_al_modelo)."""
    x = model_input_image(spec, img_array01)
    proba = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    return proba, x


def predict_svm(bundle, img_array01):
    """
    Predice probabilidades por clase con la línea base SVM+HOG (SVM_grupos).
    bundle: dict cargado con joblib, con claves 'svm', 'scaler', 'hog_params'.
    Reutiliza 08_baseline_ml.extract_hog_features (mismos parámetros HOG,
    misma conversión a escala de grises) en vez de reimplementar la
    extracción de características.
    """
    feat = baseline_mod.extract_hog_features(img_array01[np.newaxis, ...])
    feat_scaled = bundle["scaler"].transform(feat)
    return bundle["svm"].predict_proba(feat_scaled)[0]


# ──────────── Conjunto de prueba real (partición por grupos, semilla 0) ────
# El mismo conjunto (156 imágenes) que evaluó 09_evaluacion_final.py para
# M1-M5 y SVM_grupos — ninguna de ellas participó en el entrenamiento de
# ningún modelo de esa semilla. Ver incidencia de fuga de datos en README.md:
# a diferencia del holdout de partición por imagen (91-95% de gemelos casi
# idénticos en entrenamiento), esta partición es por grupo de paciente.

SEED_DEMO = 0
_test_pool_cache = None


def cargar_test_grupos_seed0():
    """
    (X_test, y_test, indices_globales) de la partición por grupos de
    paciente, semilla 0 — cargados con las mismas funciones de protocolo.py
    que usa el resto del protocolo (cargar_X_y + particion), no una
    reimplementación ni una fuente de datos distinta.
    """
    global _test_pool_cache
    if _test_pool_cache is not None:
        return _test_pool_cache
    X, y = pr.cargar_X_y(segmentado=False)
    _, _, te_idx = pr.particion(SEED_DEMO, tipo="grupos")
    _test_pool_cache = (X[te_idx], y[te_idx], te_idx)
    return _test_pool_cache


# ──────────── Imágenes de ejemplo ilustrativas (pestaña "usar ejemplo") ────
# Sin relación con el conjunto de prueba: son solo una forma rápida de
# probar la clasificación sin subir un archivo propio, tomadas del dataset
# original. Pueden coincidir con imágenes vistas en entrenamiento por
# cualquiera de los modelos — no se presentan como datos no vistos.

from config import DATA_DIR, CLASS_NAMES, CLASS_LABELS  # noqa: E402


def sample_images_by_class(n_per_class=4):
    """
    Devuelve rutas de imágenes de ejemplo del dataset original, agrupadas por
    clase, para que el usuario pueda probar la clasificación sin subir su
    propio archivo.
    """
    samples = {}
    for cls, label in zip(CLASS_NAMES, CLASS_LABELS):
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            samples[label] = []
            continue
        imgs = sorted([
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ])
        # Tomar muestras espaciadas en vez de las primeras N (más variedad)
        if len(imgs) > n_per_class:
            idxs = np.linspace(0, len(imgs) - 1, n_per_class, dtype=int)
            imgs = [imgs[i] for i in idxs]
        samples[label] = imgs
    return samples
