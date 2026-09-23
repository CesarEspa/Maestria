"""
Configuración central del proyecto TFM.
Todos los scripts importan de aquí para mantener consistencia.
"""
import os
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Datos" / "The IQ-OTHNCCD lung cancer dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
SPLITS_DIR = OUTPUT_DIR / "splits"

# Crear directorios si no existen
for d in [FIGURES_DIR, MODELS_DIR, GRADCAM_DIR, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Clases ─────────────────────────────────────────────────────────
CLASS_NAMES = ["Bengin cases", "Malignant cases", "Normal cases"]
CLASS_LABELS = ["Benigno", "Maligno", "Normal"]  # Para gráficos en español
NUM_CLASSES = 3

# ── Hiperparámetros ───────────────────────────────────────────────
IMG_SIZE = 224          # Resolución de entrada (224x224 para compatibilidad con transfer learning)
BATCH_SIZE = 32
EPOCHS_BASE = 50        # CNN base y CNN+aug
EPOCHS_TRANSFER = 30    # Transfer learning (converge más rápido)
LEARNING_RATE = 1e-3
LEARNING_RATE_TRANSFER = 1e-4
PATIENCE = 7            # Early stopping

# ── Semilla para reproducibilidad ─────────────────────────────────
SEED = 42

# ── Split ratios ──────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
