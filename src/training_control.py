"""
Control de entrenamiento usado por el backend FastAPI (api.py): qué modelos
existen, cómo lanzarlos como subprocesos independientes, cómo detenerlos y
cómo determinar su estado actual combinando el JSON de progreso en vivo con
la presencia del archivo del modelo en disco.
"""
import os
import re
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
     "model_file": "cnn_base.keras", "default_epochs": 120, "epochs_label": "Épocas",
     "note": "Red neuronal entrenada desde cero, sin variar las imágenes. Tarda entre 35 y 50 minutos."},
    {"key": "cnn_augmented", "name": "CNN + Augmentation", "script": "04_train_cnn_aug.py",
     "model_file": "cnn_augmented.keras", "default_epochs": 120, "epochs_label": "Épocas",
     "note": "Misma red, pero variando cada imagen al entrenar (rotación, zoom, etc). Suele terminar rápido: 15 a 25 minutos."},
    {"key": "transfer_efficientnet", "name": "Transfer Learning (EfficientNetB0)",
     "script": "05_train_transfer.py", "model_file": "transfer_efficientnet.keras",
     "default_epochs": 100, "epochs_label": "Épocas (ajuste fino)",
     "note": "Parte de una red ya entrenada en millones de imágenes y la adapta a este problema. Tarda entre 20 y 30 minutos."},
    {"key": "svm_baseline", "name": "SVM (HOG) — Línea Base", "script": "08_baseline_ml.py",
     "model_file": "svm_baseline.joblib", "default_epochs": None, "epochs_label": None,
     "note": "Método clásico de aprendizaje automático, sin redes neuronales. El más rápido: 1 a 2 minutos."},
]

TRAINABLE_BY_KEY = {t["key"]: t for t in TRAINABLE}

STATUS_BADGES = {
    "idle": ("⚪", "No entrenado"),
    "running": ("🟡", "Entrenando…"),
    "completed": ("🟢", "Completado"),
    "failed": ("🔴", "Falló"),
    "stopped": ("⚫", "Detenido por el usuario"),
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
    | 'stopped' | 'stale'), combinando el JSON de progreso con la presencia
    del archivo del modelo en disco (por si el modelo se entrenó antes de que
    existiera el sistema de progreso, o el progreso quedó desactualizado).
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
    if status in ("completed", "failed", "stopped"):
        return status
    return "completed" if model_path.exists() else "idle"


def model_status(key):
    """Atajo: estado efectivo de un modelo por su clave, más su progreso crudo."""
    entry = TRAINABLE_BY_KEY[key]
    state = read_progress(key)
    return effective_status(entry, state), state


def launch_training(entry, epochs=None):
    """
    Lanza el script de entrenamiento correspondiente como un subproceso
    independiente (no bloqueante). El subproceso sigue corriendo aunque el
    backend se reinicie o se cierre la pestaña del navegador; el progreso
    se sigue leyendo desde su archivo JSON.

    `epochs`, si se indica, sobreescribe el número de épocas configurado por
    defecto en config.py para ESTA ejecución concreta (vía variable de
    entorno, sin tocar el archivo de configuración). Los scripts 03/04/05 lo
    leen como TFM_EPOCHS_OVERRIDE; en transfer learning solo afecta a la
    fase 2 (fine-tuning), la fase 1 se mantiene fija.
    """
    script_path = SRC_DIR / entry["script"]
    log_path = PROGRESS_DIR / f"{entry['key']}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if epochs and entry.get("default_epochs") is not None:
        env["TFM_EPOCHS_OVERRIDE"] = str(int(epochs))
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(SRC_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    finally:
        # Popen duplica el handle para el subproceso, así que cerrarlo aquí
        # no afecta su salida; sin este cierre, el proceso servidor (de larga
        # duración) acumula un descriptor de archivo abierto por cada
        # entrenamiento lanzado durante toda su vida.
        log_file.close()


def _pid_alive(pid):
    """Comprueba si un PID sigue vivo (Windows y POSIX)."""
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def kill_process(pid, wait_seconds=8):
    """
    Detiene un proceso de entrenamiento por PID (Windows y POSIX) y espera
    hasta `wait_seconds` a que realmente termine antes de reportar éxito.

    Un `taskkill`/`kill` que "no da error" no garantiza que el proceso ya
    haya muerto (el árbol de procesos de TensorFlow puede tardar un poco en
    cerrarse); si se reportara éxito sin comprobarlo, el JSON de progreso se
    marcaría como detenido mientras el proceso seguía vivo de fondo.
    """
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True, timeout=15)
        else:
            os.kill(pid, 15)
    except Exception:
        pass

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.3)
    return not _pid_alive(pid)


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def read_log_tail(key, max_chars=4000):
    """Devuelve las últimas `max_chars` del log de un modelo, o None si no existe.

    La barra de progreso de Keras escribe códigos ANSI de color/cursor aunque
    la salida esté redirigida a un archivo; se limpian aquí para que el log
    sea legible en el panel del frontend (texto plano, sin colores de terminal).
    """
    log_path = PROGRESS_DIR / f"{key}.log"
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        text = _ANSI_ESCAPE_RE.sub("", text)
        return text[-max_chars:]
    except Exception:
        return None
