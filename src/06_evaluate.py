"""
06 - Evaluación Comparativa de los 4 modelos en el conjunto de TEST
Genera matrices de confusión, métricas por clase, tabla comparativa,
curvas ROC superpuestas y reportes de clasificación como imagen.

La SVM baseline se carga aquí en modo SOLO LECTURA (nunca se reentrena,
ver 08_baseline_ml.py) únicamente para incluirla en la comparación visual
(curvas ROC, classification report). Sus métricas "oficiales" siguen
siendo las que genera 08_baseline_ml.py en metricas_svm_baseline.json;
este script no las sobreescribe.
"""
import importlib
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc as sk_auc, accuracy_score
)
from sklearn.preprocessing import label_binarize

from config import (
    SPLITS_DIR, MODELS_DIR, FIGURES_DIR,
    CLASS_LABELS, NUM_CLASSES
)

sns.set_theme(style="whitegrid", font_scale=1.1)

MODEL_SPECS = [
    ("CNN Base", "cnn_base.keras"),
    ("CNN + Augmentation", "cnn_augmented.keras"),
    ("Transfer Learning", "transfer_efficientnet.keras"),
]
SVM_MODEL_NAME = "SVM (HOG) — Línea Base"
ROC_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]


def evaluate_from_proba(y_test, y_proba, model_name):
    """Calcula todas las métricas a partir de las probabilidades ya predichas
    (independiente de si vienen de un modelo Keras o de la SVM)."""
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)

    try:
        y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
        auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    report = classification_report(
        y_test, y_pred, target_names=CLASS_LABELS, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "name": model_name,
        "accuracy": acc,
        "auc_roc": auc,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def evaluate_model(model, X_test, y_test, model_name):
    """Evalúa un modelo Keras (predict directo sobre los píxeles)."""
    y_proba = model.predict(X_test, verbose=0)
    return evaluate_from_proba(y_test, y_proba, model_name)


def load_svm_probabilities(X_test):
    """
    Carga la SVM baseline ya entrenada (outputs/models/svm_baseline.joblib)
    y calcula sus probabilidades sobre el test set, reutilizando la misma
    extracción HOG que 08_baseline_ml.py (importando su función para no
    duplicar la lógica). No reentrena ni modifica el modelo.
    """
    svm_path = MODELS_DIR / "svm_baseline.joblib"
    if not svm_path.exists():
        return None

    bundle = joblib.load(svm_path)
    baseline_mod = importlib.import_module("08_baseline_ml")
    X_test_hog = baseline_mod.extract_hog_features(X_test)
    X_test_hog = bundle["scaler"].transform(X_test_hog)
    return bundle["svm"].predict_proba(X_test_hog)


def macro_roc_curve(y_test, y_proba):
    """ROC macro-average (one-vs-rest) — misma técnica que scikit-learn
    recomienda para promediar curvas ROC multiclase."""
    y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    fpr, tpr = {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES

    return all_fpr, mean_tpr, sk_auc(all_fpr, mean_tpr)


def plot_confusion_matrix(cm, model_name, filename):
    """Matriz de confusión con heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                ax=ax, linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor Real")
    ax.set_title(f"Matriz de Confusión — {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def plot_comparison_table(results):
    """Tabla comparativa de los 3 modelos."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    headers = ["Modelo", "Exactitud", "AUC-ROC",
               "Sensib.\nBenigno", "Sensib.\nMaligno", "Sensib.\nNormal",
               "F1\nMacro"]

    rows = []
    for r in results:
        rep = r["report"]
        rows.append([
            r["name"],
            f"{r['accuracy']:.4f}",
            f"{r['auc_roc']:.4f}" if r["auc_roc"] else "N/A",
            f"{rep['Benigno']['recall']:.4f}",
            f"{rep['Maligno']['recall']:.4f}",
            f"{rep['Normal']['recall']:.4f}",
            f"{rep['macro avg']['f1-score']:.4f}",
        ])

    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colColours=["#3498db"] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Colorear headers
    for j in range(len(headers)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Resaltar mejor accuracy
    best_idx = np.argmax([r["accuracy"] for r in results])
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor("#d4efdf")

    fig.suptitle("Comparativa de Rendimiento — Conjunto de Prueba", fontsize=14, y=0.95)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tabla_comparativa.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / 'tabla_comparativa.png'}")


def plot_roc_comparison(roc_entries, filename="curvas_roc_comparativas.png"):
    """
    Curvas ROC macro-average (one-vs-rest) de todos los modelos disponibles,
    superpuestas en una sola figura. roc_entries: lista de
    (nombre, fpr, tpr, auc).
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for (name, fpr, tpr, auc_val), color in zip(roc_entries, ROC_COLORS):
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.3,
            label="Clasificador aleatorio (AUC = 0.500)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curvas ROC comparativas — macro-average (one-vs-rest)\nConjunto de Prueba")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def plot_classification_report_image(report, model_name, filename):
    """Guarda el classification_report (precision/recall/f1/support por
    clase, más macro y weighted avg) como una tabla-imagen."""
    row_labels = list(CLASS_LABELS) + ["macro avg", "weighted avg"]
    rows = []
    for key in row_labels:
        r = report[key]
        rows.append([
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1-score']:.3f}",
            f"{int(r['support'])}",
        ])

    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(rows) + 1.8))
    ax.axis("off")
    headers = ["Precision", "Recall", "F1-score", "Support"]

    table = ax.table(
        cellText=rows, rowLabels=row_labels, colLabels=headers,
        cellLoc="center", rowLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.9)
    for j in range(len(headers)):
        table[0, j].set_text_props(color="white", fontweight="bold")
        table[0, j].set_facecolor("#3498db")

    fig.suptitle(f"Classification Report — {model_name}\n"
                 f"Exactitud global: {report['accuracy']:.4f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def main():
    print("=" * 60)
    print("  EVALUACIÓN COMPARATIVA — Conjunto de Prueba")
    print("=" * 60)

    print("\n[1/3] Cargando datos de prueba...")
    data = np.load(SPLITS_DIR / "dataset_splits.npz")
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"  → Test: {X_test.shape}")

    print("\n[2/4] Evaluando modelos Keras...")
    results = []
    for model_name, model_file in MODEL_SPECS:
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            print(f"  ⚠ Modelo no encontrado: {model_path}, saltando...")
            continue

        print(f"\n  ── {model_name} ──")
        model = keras.models.load_model(model_path)
        result = evaluate_model(model, X_test, y_test, model_name)
        results.append(result)

        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC-ROC:  {result['auc_roc']:.4f}" if result["auc_roc"] else "  AUC-ROC: N/A")

        # Classification report
        print(classification_report(y_test, result["y_pred"], target_names=CLASS_LABELS))

        # Confusion matrix
        safe_name = model_name.lower().replace("+", "").replace(" ", "_")
        while "__" in safe_name:
            safe_name = safe_name.replace("__", "_")
        plot_confusion_matrix(result["confusion_matrix"], model_name,
                              f"confusion_matrix_{safe_name}.png")
        plot_classification_report_image(
            result["report"], model_name, f"classification_report_{safe_name}.png"
        )

    if len(results) >= 2:
        print("\n[3/4] Generando tabla comparativa...")
        plot_comparison_table(results)

    # ── SVM baseline: SOLO LECTURA, no se reentrena ni se modifica ──────
    # Se incluye aquí únicamente para las curvas ROC comparativas y su
    # classification report como imagen. Sus métricas "oficiales" siguen
    # siendo las de metricas_svm_baseline.json (08_baseline_ml.py).
    print("\n[4/4] Cargando SVM baseline (solo lectura, sin reentrenar)...")
    svm_proba = load_svm_probabilities(X_test)
    svm_result = None
    if svm_proba is not None:
        svm_result = evaluate_from_proba(y_test, svm_proba, SVM_MODEL_NAME)
        print(f"  Accuracy: {svm_result['accuracy']:.4f} (ya reportado en "
              f"metricas_svm_baseline.json; no se sobreescribe)")
        plot_classification_report_image(
            svm_result["report"], SVM_MODEL_NAME, "classification_report_svm_baseline.png"
        )
    else:
        print("  ⚠ No se encontró outputs/models/svm_baseline.joblib, se omite del ROC comparativo.")

    # ── Curvas ROC comparativas (los modelos Keras + SVM si está disponible) ──
    print("\n  Generando curvas ROC comparativas...")
    roc_entries = []
    for r in results:
        fpr, tpr, auc_val = macro_roc_curve(y_test, r["y_proba"])
        roc_entries.append((r["name"], fpr, tpr, auc_val))
    if svm_result is not None:
        fpr, tpr, auc_val = macro_roc_curve(y_test, svm_result["y_proba"])
        roc_entries.append((svm_result["name"], fpr, tpr, auc_val))
    if roc_entries:
        plot_roc_comparison(roc_entries)

    # Guardar métricas como JSON (solo los 3 modelos Keras — el JSON de la
    # SVM lo genera y posee 08_baseline_ml.py, no se toca desde aquí)
    metrics_json = []
    for r in results:
        metrics_json.append({
            "name": r["name"],
            "accuracy": r["accuracy"],
            "auc_roc": r["auc_roc"],
            "f1_macro": r["report"]["macro avg"]["f1-score"],
            "sensitivity_per_class": {
                cls: r["report"][cls]["recall"] for cls in CLASS_LABELS
            }
        })
    with open(FIGURES_DIR / "metricas_comparativas.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  → Métricas JSON: {FIGURES_DIR / 'metricas_comparativas.json'}")

    print("\n✓ Evaluación comparativa completada.")


if __name__ == "__main__":
    main()
