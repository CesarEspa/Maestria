"""
Backend FastAPI del TFM — Detección y Clasificación de Cáncer de Pulmón.

Expone una API REST sobre la lógica de negocio del proyecto
(training_control.py, model_utils.py, gradcam_utils.py, results_utils.py) y
sirve el frontend estático (HTML/CSS/JS) en frontend/, todo en un único
proceso y un único puerto (evita problemas de CORS).

Ejecutar con:
    cd src
    uvicorn api:app --reload --port 8000

Luego abrir http://localhost:8000 en el navegador.

Este backend es deliberadamente ligero (sin base de datos, sin autenticación,
un solo usuario) porque su propósito es servir de interfaz visual sobre el
pipeline de scripts ya existente. Estructurarlo como una API REST separada
del frontend permite hacerlo crecer más adelante: añadir autenticación, una
base de datos para historial de clasificaciones, desplegarlo en un servidor
real, servir el frontend por separado (p. ej. como una SPA en React), etc.,
sin tener que reescribir la lógica de negocio.
"""
import base64
import io
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLASS_LABELS, DATA_DIR, SPLITS_DIR  # noqa: E402
import model_utils  # noqa: E402
import gradcam_utils  # noqa: E402
import results_utils  # noqa: E402
from training_control import (  # noqa: E402
    TRAINABLE, TRAINABLE_BY_KEY,
    read_progress, effective_status, launch_training, kill_process, read_log_tail,
)
from progress_tracker import mark_status  # noqa: E402

FRONTEND_DIR = SRC_DIR / "frontend"

app = FastAPI(title="TFM Lung Cancer API", version="1.0")

# ───────────────────────── Caché de modelos cargados ──────────────────────
# Evita recargar un modelo de varios MB en cada petición. Se invalida si el
# archivo cambia (mtime).
_model_cache = {}


def _load_model_cached(spec):
    path = spec["path"]
    mtime = path.stat().st_mtime
    cache_key = str(path)
    cached = _model_cache.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    if spec["type"] == "keras":
        from tensorflow import keras
        obj = keras.models.load_model(path)
    else:
        obj = joblib.load(path)

    _model_cache[cache_key] = (mtime, obj)
    return obj


# ───────── Caché del "holdout" (30% no usado en entrenamiento) ────────────
# Val (15%) + Test (15%) combinados = el 30% de dataset_splits.npz que
# ningún modelo vio durante su entrenamiento. Se usa para el botón "Elegir
# imagen aleatoria" en la pestaña Clasificar, para poder probar de forma
# honesta "qué tal funciona" sobre datos realmente no vistos.
_holdout_cache = None


def _load_holdout_pool():
    global _holdout_cache
    if _holdout_cache is not None:
        return _holdout_cache

    npz_path = SPLITS_DIR / "dataset_splits.npz"
    if not npz_path.exists():
        raise HTTPException(
            404,
            "No se encontró outputs/splits/dataset_splits.npz. "
            "Ejecuta primero 02_preprocessing.py."
        )
    data = np.load(npz_path)
    X_holdout = np.concatenate([data["X_val"], data["X_test"]], axis=0)
    y_holdout = np.concatenate([data["y_val"], data["y_test"]], axis=0)
    _holdout_cache = (X_holdout, y_holdout)
    return _holdout_cache


# ────────────────────────────────── Salud ──────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ────────────────────────────── Entrenamiento ──────────────────────────────

@app.get("/api/models")
def list_models():
    """Estado de los 4 modelos entrenables, para pintar las tarjetas del frontend."""
    out = []
    for entry in TRAINABLE:
        state = read_progress(entry["key"])
        status = effective_status(entry, state)
        out.append({
            "key": entry["key"],
            "name": entry["name"],
            "note": entry["note"],
            "status": status,
            "progress": state,
            "default_epochs": entry.get("default_epochs"),
            "epochs_label": entry.get("epochs_label"),
        })
    return out


@app.post("/api/models/{key}/train")
def train_model(key: str, epochs: int | None = None):
    entry = TRAINABLE_BY_KEY.get(key)
    if entry is None:
        raise HTTPException(404, f"Modelo desconocido: {key}")

    state = read_progress(key)
    status = effective_status(entry, state)
    if status == "running":
        raise HTTPException(409, "Este modelo ya se está entrenando.")

    max_epochs = entry.get("default_epochs")
    if epochs is not None:
        if max_epochs is None:
            raise HTTPException(400, f"El modelo '{key}' no admite un número de épocas configurable.")
        if not (1 <= epochs <= max_epochs):
            raise HTTPException(400, f"El número de épocas debe estar entre 1 y {max_epochs} para este modelo.")

    launch_training(entry, epochs=epochs)
    return {"launched": True, "key": key, "epochs": epochs}


@app.post("/api/models/{key}/stop")
def stop_model(key: str):
    entry = TRAINABLE_BY_KEY.get(key)
    if entry is None:
        raise HTTPException(404, f"Modelo desconocido: {key}")

    state = read_progress(key)
    if not state or not state.get("pid"):
        raise HTTPException(409, "No hay un proceso en curso para detener.")

    ok = kill_process(state["pid"])
    if not ok:
        raise HTTPException(
            500,
            "El proceso no respondió a la señal de detener y puede seguir "
            "en ejecución en segundo plano. Intenta de nuevo en unos segundos."
        )
    # Importante: sin esto, el JSON de progreso se queda con status="running"
    # (y el frontend sigue mostrando "Entrenando…" con la barra congelada)
    # hasta que pasan los 5 minutos del umbral de "stale".
    mark_status(key, entry["name"], "stopped")
    return {"stopped": True, "key": key}


@app.get("/api/models/{key}/log")
def model_log(key: str):
    if key not in TRAINABLE_BY_KEY:
        raise HTTPException(404, f"Modelo desconocido: {key}")
    log = read_log_tail(key)
    return {"log": log or ""}


# ─────────────────────────────── Clasificación ─────────────────────────────

@app.get("/api/samples")
def list_samples():
    """Imágenes de ejemplo del propio dataset, agrupadas por clase, para
    probar la clasificación sin necesidad de subir un archivo."""
    samples = model_utils.sample_images_by_class(n_per_class=6)
    return {
        label: [{"index": i, "name": p.name} for i, p in enumerate(paths)]
        for label, paths in samples.items()
    }


@app.get("/api/samples/{label}/{index}")
def get_sample_image(label: str, index: int):
    samples = model_utils.sample_images_by_class(n_per_class=6)
    paths = samples.get(label)
    if not paths or index < 0 or index >= len(paths):
        raise HTTPException(404, "Imagen de ejemplo no encontrada.")
    return FileResponse(paths[index])


def _predict_and_explain(pil_img, model_key):
    """Clasifica una imagen PIL: la preprocesa y delega en la versión que
    trabaja directamente sobre el array ya normalizado."""
    img_array01 = model_utils.preprocess_pil_image(pil_img)
    return _predict_and_explain_array(img_array01, model_key)


def _predict_and_explain_array(img_array01, model_key):
    """
    Núcleo de la clasificación: recibe una imagen YA preprocesada (float32,
    224x224x3, en [0,1]) y devuelve predicción + Grad-CAM. Se usa tanto para
    imágenes subidas/de ejemplo (previamente preprocesadas desde un archivo)
    como para imágenes del conjunto de test, que ya vienen preprocesadas tal
    cual se guardaron en dataset_splits.npz — así se clasifican exactamente
    los mismos píxeles que se usaron para calcular las métricas de 06_evaluate.py.
    """
    specs = {m["key"]: m for m in model_utils.available_models()}
    spec = specs.get(model_key)
    if spec is None:
        raise HTTPException(404, f"Modelo desconocido: {model_key}")
    if not spec["exists"]:
        raise HTTPException(409, f"El modelo '{model_key}' todavía no está entrenado.")

    obj = _load_model_cached(spec)

    if spec["type"] == "keras":
        proba = model_utils.predict_keras(obj, img_array01)
    else:
        proba = model_utils.predict_svm(obj, img_array01)

    pred_idx = int(np.argmax(proba))
    result = {
        "predicted_class": CLASS_LABELS[pred_idx],
        "confidence": float(proba[pred_idx]),
        "probabilities": {CLASS_LABELS[i]: float(p) for i, p in enumerate(proba)},
        "gradcam_base64": None,
    }

    if spec["type"] == "keras":
        try:
            overlay, _, _ = gradcam_utils.gradcam_overlay_for_image(obj, img_array01, pred_index=pred_idx)
            if overlay is not None:
                buf = io.BytesIO()
                Image.fromarray((overlay * 255).astype("uint8")).save(buf, format="PNG")
                result["gradcam_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass  # Grad-CAM es un extra; un fallo ahí no debe tumbar la predicción

    return result


@app.post("/api/classify")
async def classify(model_key: str, file: UploadFile = File(...)):
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.load()
    except Exception:
        raise HTTPException(400, "No se pudo leer el archivo como imagen.")

    return _predict_and_explain(pil_img, model_key)


@app.post("/api/classify_sample")
def classify_sample(model_key: str, label: str, index: int):
    samples = model_utils.sample_images_by_class(n_per_class=6)
    paths = samples.get(label)
    if not paths or index < 0 or index >= len(paths):
        raise HTTPException(404, "Imagen de ejemplo no encontrada.")
    pil_img = Image.open(paths[index])
    result = _predict_and_explain(pil_img, model_key)
    result["true_label"] = label
    return result


# ───────────── Imagen aleatoria del holdout (30%: validación + test) ──────

@app.get("/api/holdout_random")
def holdout_random():
    """
    Elige un índice al azar dentro del 30% de datos no usado en
    entrenamiento (validación + test combinados). Es lo que consume el
    botón "Elegir imagen aleatoria" de la pestaña Clasificar, para probar
    de forma honesta cómo generaliza el modelo.
    """
    X_holdout, y_holdout = _load_holdout_pool()
    index = random.randrange(len(X_holdout))
    return {"index": index, "true_label": CLASS_LABELS[int(y_holdout[index])]}


@app.get("/api/holdout_sample/{index}")
def get_holdout_sample_image(index: int):
    X_holdout, _ = _load_holdout_pool()
    if index < 0 or index >= len(X_holdout):
        raise HTTPException(404, "Índice fuera de rango del holdout.")
    img01 = X_holdout[index]
    buf = io.BytesIO()
    Image.fromarray((np.clip(img01, 0, 1) * 255).astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/classify_holdout_sample")
def classify_holdout_sample(model_key: str, index: int):
    X_holdout, y_holdout = _load_holdout_pool()
    if index < 0 or index >= len(X_holdout):
        raise HTTPException(404, "Índice fuera de rango del holdout.")
    result = _predict_and_explain_array(X_holdout[index], model_key)
    result["true_label"] = CLASS_LABELS[int(y_holdout[index])]
    return result


# ────────────────────────────── Resultados ─────────────────────────────────

@app.get("/api/results")
def results():
    metrics = results_utils.load_all_metrics()
    return {
        "metrics": metrics,
        "leakage_warning": results_utils.has_leakage_signal(metrics),
        "leakage_warning_text": results_utils.LEAKAGE_WARNING,
        "best_model": results_utils.best_model_name(metrics),
    }


@app.get("/api/figures")
def figures():
    return results_utils.available_figures()


@app.get("/api/figures/{filename}")
def get_figure(filename: str):
    from config import FIGURES_DIR
    path = FIGURES_DIR / filename
    if ".." in filename or not path.exists():
        raise HTTPException(404, "Figura no encontrada.")
    return FileResponse(path)


@app.get("/api/gradcam_grid")
def gradcam_grid():
    path = results_utils.gradcam_grid_path()
    if path is None:
        raise HTTPException(404, "Aún no se generó el grid de Grad-CAM.")
    return FileResponse(path)


@app.get("/api/dataset_summary")
def dataset_summary():
    from config import CLASS_NAMES
    counts = {}
    total = 0
    for cls, label in zip(CLASS_NAMES, CLASS_LABELS):
        cls_dir = DATA_DIR / cls
        n = 0
        if cls_dir.exists():
            n = sum(
                1 for p in cls_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            )
        counts[label] = n
        total += n
    return {"total": total, "per_class": counts}


# ─────────────────────── Frontend estático (al final: catch-all) ──────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
