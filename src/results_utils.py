"""
Carga de métricas y figuras de evaluación, compartida por la app Streamlit
y el backend FastAPI, para que ambas muestren exactamente los mismos números
y no diverjan.
"""
import json

from config import FIGURES_DIR, GRADCAM_DIR

# (nombre de archivo, título a mostrar)
FIGURE_SPECS = [
    ("tabla_comparativa.png", "Tabla comparativa"),
    ("confusion_matrix_cnn_base.png", "Matriz de confusión — CNN Base"),
    ("confusion_matrix_cnn_augmentation.png", "Matriz de confusión — CNN + Augmentation"),
    ("confusion_matrix_transfer_learning.png", "Matriz de confusión — Transfer Learning"),
    ("confusion_matrix_svm_baseline.png", "Matriz de confusión — SVM baseline"),
    ("curvas_cnn_base.png", "Curvas de entrenamiento — CNN Base"),
    ("curvas_cnn_augmented.png", "Curvas de entrenamiento — CNN + Augmentation"),
    ("curvas_transfer_learning.png", "Curvas de entrenamiento — Transfer Learning"),
]

LEAKAGE_WARNING = (
    "Uno o más modelos alcanzan ~100% de exactitud en el conjunto de prueba. "
    "Para este dataset (sin identificador de paciente por imagen y con "
    "múltiples cortes de TC por paciente), esto es indicativo de fuga de "
    "datos (data leakage) entre train/val/test a nivel de paciente, y no "
    "debe interpretarse como una capacidad de generalización real."
)


def load_all_metrics():
    """
    Devuelve la lista combinada de métricas de los 4 modelos (3 redes Keras
    + SVM baseline), leyendo los JSON que generan 06_evaluate.py y
    08_baseline_ml.py. Modelos aún no evaluados simplemente no aparecen.
    """
    metrics = []

    comp_path = FIGURES_DIR / "metricas_comparativas.json"
    if comp_path.exists():
        with open(comp_path, "r", encoding="utf-8") as f:
            metrics.extend(json.load(f))

    svm_path = FIGURES_DIR / "metricas_svm_baseline.json"
    if svm_path.exists():
        with open(svm_path, "r", encoding="utf-8") as f:
            metrics.append(json.load(f))

    return metrics


def has_leakage_signal(metrics, threshold=0.999):
    """True si algún modelo alcanza una exactitud sospechosamente perfecta."""
    return any((m.get("accuracy") or 0) >= threshold for m in metrics)


def best_model_name(metrics, exclude_suspicious=True, threshold=0.999):
    """
    Nombre del modelo con mayor exactitud, excluyendo por defecto los que
    superan `threshold` (sospechosos de fuga de datos) para no señalar la
    SVM como "mejor modelo" solo por su resultado inflado.
    """
    candidates = metrics
    if exclude_suspicious:
        non_suspicious = [m for m in metrics if (m.get("accuracy") or 0) < threshold]
        if non_suspicious:
            candidates = non_suspicious
    if not candidates:
        return None
    best = max(candidates, key=lambda m: m.get("accuracy") or 0)
    return best.get("name")


def available_figures():
    """Lista (filename, caption, exists) de las figuras de evaluación conocidas."""
    result = []
    for filename, caption in FIGURE_SPECS:
        path = FIGURES_DIR / filename
        result.append({"filename": filename, "caption": caption, "exists": path.exists()})
    return result


def gradcam_grid_path():
    path = GRADCAM_DIR / "gradcam_grid.png"
    return path if path.exists() else None
