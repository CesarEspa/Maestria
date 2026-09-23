"""
Control de entrenamiento compartido por la app Streamlit (app.py) y el
backend FastAPI (backend/main.py): qué modelos existen, cómo lanzarlos como
subprocesos independientes, cómo detenerlos y cómo determinar su estado
actual combinando el JSON de progreso en vivo con la presencia del archivo
del modelo en disco.

Centralizar esto en un solo módulo evita que las dos interfaces (Streamlit
y la futura web con backend) diverjan en su lógica de estado.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from config import MODELS_DIR
from progress_tracker import PROGRESS_DIR, progress_path

SRC_DIR = Path(__file__).resolve().parent
STALE_SECONDS = 300  # si "running" lleva más de esto sin actualizar, se asume caído

TRAINABLE = [
    {"key": "cnn_base", "name": "CNN Base", "script": "03_train_cnn_base.py",
     "model_file": "cnn_base.keras",
     "note": "LayerNorm + LR 1e-4, hasta 80 épocas. En CPU: ~40-60 min."},
    {"key": "cnn_augmented", "name": "CNN + Augmentation", "script": "04_train_cnn_aug.py",
     "model_file": "cnn_augmented.keras",
     "note": "Misma arquitectura con aumento de datos reducido. En CPU: ~50-80 min."},
    {"key": "transfer_efficientnet", "name": "Transfer Learning (EfficientNetB0)",
     "script": "05_train_transfer.py", "model_file": "transfer_efficientnet.keras",
     "note": "Dos fases (cabeza + fine-tuning de 50 capas). En CPU: ~20-35 min."},
    {"key": "svm_baseline", "name": "SVM (HOG) — Línea Base", "script": "08_baseline_ml.py",
     "model_file": "svm_baseline.joblib",
     "note": "Extracción HOG + SVM. Rápido: ~1-2 min."},
]

TRAINABLE_BY_KEY = {t["key"]: t for t in TRAINABLE}

STATUS_BADGES = {
    "idle": ("⚪", "No entrenado"),
    "running": ("🟡", "Entrenando…"),
    "completed": ("🟢", "Completado"),
    "failed": ("🔴", "Falló"),
    "stale": ("🟠", "Sin actividad reciente (¿se detuvo?)"),
}


def read_progress(key):
    """Lee el JSON de progreso de un modelo, o None si no existe/es inválido."""
    import json
    path = progress_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def effective_status(entry, state):
    """
    Determina el estado a mostrar ('idle' | 'running' | 'completed' | 'failed'
    | 'stale'), combinando el JSON de progreso con la presencia del archivo
    del modelo en disco (por si el modelo se entrenó antes de que existiera
    el sistema de progreso, o el progreso quedó desactualizado).
    """
    model_path = MODELS_DIR / entry["model_file"]
    if state is None:
        return "completed" if model_path.exists() else "idle"

    status = state.get("status", "idle")
    if status == "running":
        stale = (time.time() - state.get("updated_at", 0)) > STALE_SECONDS
        return "stale" if stale else "running"
    if status == "phase_completed":
        return "running"  # transición entre fase 1 y fase 2 (transfer learning)
    if status in ("completed", "failed"):
        return status
    return "completed" if model_path.exists() else "idle"


def model_status(key):
    """Atajo: estado efectivo de un modelo por su clave, más su progreso crudo."""
    entry = TRAINABLE_BY_KEY[key]
    state = read_progress(key)
    return effective_status(entry, state), state


def launch_training(entry):
    """
    Lanza el script de entrenamiento correspondiente como un subproceso
    independiente (no bloqueante). El subproceso sigue corriendo aunque el
    proceso que lo lanzó (Streamlit o el backend) se reinicie o se cierre la
    pestaña del navegador; el progreso se sigue leyendo desde su archivo JSON.
    """
    script_path = SRC_DIR / entry["script"]
    log_path = PROGRESS_DIR / f"{entry['key']}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    log_file = open(log_path, "w", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(SRC_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def kill_process(pid):
    """Detiene un proceso de entrenamiento por PID (Windows y POSIX)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True)
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


def read_log_tail(key, max_chars=4000):
    """Devuelve las últimas `max_chars` del log de un modelo, o None si no existe."""
    log_path = PROGRESS_DIR / f"{key}.log"
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]
    except Exception:
        return None
