"""
Utilidades para registrar el progreso del entrenamiento en un archivo JSON.
La app web lee estos archivos para mostrar el estado en vivo (época actual,
curvas de loss/accuracy, fase de entrenamiento) sin acoplarse directamente
al proceso de entrenamiento, que corre en un subproceso aparte.
"""
import json
import os
import time
from pathlib import Path

from config import OUTPUT_DIR

PROGRESS_DIR = OUTPUT_DIR / "progress"
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def progress_path(model_key):
    """Ruta del JSON de progreso para un modelo dado (p.ej. 'cnn_base')."""
    return PROGRESS_DIR / f"{model_key}.json"


def _atomic_write(path, state):
    """
    Escribe el JSON de forma atómica (escribe a .tmp y renombra).
    El progreso es solo para la UI: cualquier fallo al escribirlo (carpeta
    borrada/movida, problema de disco, etc.) se ignora en silencio para que
    NUNCA interrumpa el entrenamiento real del modelo.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except OSError:
        pass


def mark_status(model_key, display_name, status, **extra):
    """Actualiza solo el estado global (idle/running/completed/failed) de un modelo."""
    path = progress_path(model_key)
    state = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    state.update({
        "model_key": model_key,
        "display_name": display_name,
        "status": status,
        "updated_at": time.time(),
        "pid": os.getpid(),
    })
    state.update(extra)
    _atomic_write(path, state)
    return state


try:
    from tensorflow import keras

    class WebProgressCallback(keras.callbacks.Callback):
        """
        Callback de Keras que escribe el progreso del entrenamiento a un JSON
        después de cada batch y cada época, para que la app web lo muestre en vivo.

        phase: etiqueta libre para distinguir fases dentro de un mismo script
               (p.ej. "Fase 1 - Cabeza congelada" / "Fase 2 - Fine-tuning" en
               el transfer learning).
        epoch_offset: número de épocas ya completadas en fases anteriores,
                      para que el contador de época sea acumulado y no se
                      reinicie visualmente en la fase 2.
        """

        def __init__(self, model_key, display_name, total_epochs, phase="",
                     epoch_offset=0, total_epochs_all_phases=None):
            super().__init__()
            self.model_key = model_key
            self.display_name = display_name
            self.total_epochs = total_epochs
            self.phase = phase
            self.epoch_offset = epoch_offset
            self.total_epochs_all_phases = total_epochs_all_phases or total_epochs
            self.path = progress_path(model_key)
            self.history = []
            # Si ya existe historial de una fase anterior, lo conservamos
            if self.path.exists():
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        prev = json.load(f)
                    if prev.get("model_key") == model_key:
                        self.history = prev.get("history", [])
                except Exception:
                    self.history = []

        def _write(self, **extra):
            state = {
                "model_key": self.model_key,
                "display_name": self.display_name,
                "status": "running",
                "phase": self.phase,
                "total_epochs": self.total_epochs_all_phases,
                "updated_at": time.time(),
                "pid": os.getpid(),
                "history": self.history[-500:],
            }
            state.update(extra)
            _atomic_write(self.path, state)

        def on_train_begin(self, logs=None):
            self._current_epoch = self.epoch_offset
            self._write(current_epoch=self._current_epoch, status="running")

        def on_epoch_begin(self, epoch, logs=None):
            self._current_epoch = self.epoch_offset + epoch + 1
            self._write(current_epoch=self._current_epoch)

        def on_batch_end(self, batch, logs=None):
            if batch % 3 != 0:
                return
            logs = logs or {}
            safe_logs = {k: float(v) for k, v in logs.items() if _is_number(v)}
            self._write(
                current_epoch=self._current_epoch,
                batch=int(batch),
                batch_logs=safe_logs,
            )

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            entry = {
                "epoch": self.epoch_offset + epoch + 1,
                "phase": self.phase,
                **{k: float(v) for k, v in logs.items() if _is_number(v)},
            }
            self.history.append(entry)
            self._current_epoch = self.epoch_offset + epoch + 1
            self._write(current_epoch=self._current_epoch)

        def on_train_end(self, logs=None):
            self._write(current_epoch=self._current_epoch, status="phase_completed")

except ImportError:
    # TensorFlow puede no estar disponible en un proceso que solo lee
    # progreso sin necesitar entrenar nada. No es necesario aquí.
    WebProgressCallback = None


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
