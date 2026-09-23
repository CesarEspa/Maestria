"""
06 - Evaluación Comparativa de los 3 modelos en el conjunto de TEST
Genera matrices de confusión, métricas por clase y tabla comparativa.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

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


def evaluate_model(model, X_test, y_test, model_name):
    """Evalúa un modelo y retorna métricas."""
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)

    # AUC-ROC (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
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


def main():
    print("=" * 60)
    print("  EVALUACIÓN COMPARATIVA — Conjunto de Prueba")
    print("=" * 60)

    print("\n[1/3] Cargando datos de prueba...")
    data = np.load(SPLITS_DIR / "dataset_splits.npz")
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"  → Test: {X_test.shape}")

    print("\n[2/3] Evaluando modelos...")
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

    if len(results) >= 2:
        print("\n[3/3] Generando tabla comparativa...")
        plot_comparison_table(results)

    # Guardar métricas como JSON
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
