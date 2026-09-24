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
EPOCHS_BASE = 120       # CNN base y CNN+aug (subido de 80 en la 3ra ronda: ambas ya paraban
                        # muy por debajo de 80 con EarlyStopping, pero se sube el techo para
                        # confirmar que ninguna quedaba limitada por el presupuesto)
EPOCHS_TRANSFER = 100   # Transfer learning, fase de fine-tuning (subido de 50: en la ronda
                        # anterior agotó el presupuesto completo sin activar EarlyStopping,
                        # señal de que seguía mejorando y necesitaba más margen)
LEARNING_RATE = 1e-4    # Bajado de 1e-3: el valor anterior era parte de la causa de la
                        # inestabilidad de la CNN base/aug (ver Hallazgos en README.md)
LEARNING_RATE_TRANSFER = 1e-4
PATIENCE = 12           # Early stopping (subido de 7, para dar más margen con el LR más bajo)

# ── Semilla para reproducibilidad ─────────────────────────────────
SEED = 42

# ── Split ratios ──────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
