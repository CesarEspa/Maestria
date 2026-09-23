"""
Utilidades de inferencia compartidas por la app web: preprocesamiento de una
imagen subida por el usuario (idéntico al usado en 02_preprocessing.py) y
carga/predicción con cualquiera de los 4 modelos entrenados (3 redes Keras +
la línea base SVM).
"""
import numpy as np
from pathlib import Path
from PIL import Image

from config import DATA_DIR, MODELS_DIR, IMG_SIZE, CLASS_NAMES, CLASS_LABELS

# Modelos Keras disponibles, en el mismo orden que se presentan en el TFM.
KERAS_MODEL_SPECS = [
    ("cnn_base", "CNN Base", "cnn_base.keras"),
    ("cnn_augmented", "CNN + Augmentation", "cnn_augmented.keras"),
    ("transfer_efficientnet", "Transfer Learning (EfficientNetB0)", "transfer_efficientnet.keras"),
]
SVM_MODEL_SPEC = ("svm_baseline", "SVM (HOG) — Línea Base", "svm_baseline.joblib")


def preprocess_pil_image(pil_img):
    """
    Replica exacta del preprocesamiento de 02_preprocessing.py:
    RGB, resize a IMG_SIZE x IMG_SIZE con LANCZOS, normalizado a [0, 1].
    Devuelve un array float32 (IMG_SIZE, IMG_SIZE, 3).
    """
    img = pil_img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def available_models():
    """
    Devuelve la lista de modelos entrenados disponibles en disco, con su tipo
    ('keras' o 'sklearn') para que la app sepa cómo cargarlos y predecir.
    """
    models = []
    for key, name, filename in KERAS_MODEL_SPECS:
        path = MODELS_DIR / filename
        models.append({"key": key, "name": name, "path": path,
                        "type": "keras", "exists": path.exists()})

    key, name, filename = SVM_MODEL_SPEC
    path = MODELS_DIR / filename
    models.append({"key": key, "name": name, "path": path,
                    "type": "sklearn", "exists": path.exists()})
    return models


def predict_keras(model, img_array01):
    """Predice probabilidades por clase con un modelo Keras. img_array01: (H,W,3) en [0,1]."""
    batch = np.expand_dims(img_array01, axis=0)
    proba = model.predict(batch, verbose=0)[0]
    return proba


def predict_svm(bundle, img_array01):
    """
    Predice probabilidades por clase con la línea base SVM+HOG.
    bundle: dict cargado con joblib, con claves 'svm', 'scaler', 'hog_params'.
    """
    from skimage.feature import hog

    gray = np.mean(img_array01, axis=2)
    feat = hog(gray, **bundle["hog_params"]).reshape(1, -1)
    feat_scaled = bundle["scaler"].transform(feat)
    proba = bundle["svm"].predict_proba(feat_scaled)[0]
    return proba


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
