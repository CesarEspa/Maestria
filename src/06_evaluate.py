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


def specificity_per_class_from_cm(cm):
    """
    Especificidad por clase (uno-contra-el-resto) a partir de una matriz de
    confusión NxN, según la propuesta 1 del TFM (objetivo específico 3):
      FP_i = suma de la columna i, excluyendo la diagonal
      VN_i = celdas que no están ni en la fila i ni en la columna i
           = total - fila_i - columna_i + cm[i,i]
      Especificidad_i = VN_i / (VN_i + FP_i)
    Devuelve un dict {CLASS_LABELS[i]: especificidad}.
    """
    n = cm.shape[0]
    total = cm.sum()
    out = {}
    for i in range(n):
        col_i = cm[:, i].sum()
        row_i = cm[i, :].sum()
        fp = col_i - cm[i, i]
        vn = total - row_i - col_i + cm[i, i]
        out[CLASS_LABELS[i]] = float(vn / (vn + fp)) if (vn + fp) > 0 else 0.0
    return out


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
    specificity = specificity_per_class_from_cm(cm)

    return {
        "name": model_name,
        "accuracy": acc,
        "auc_roc": auc,
        "report": report,
        "confusion_matrix": cm,
        "specificity_per_class": specificity,
        "specificity_macro": float(np.mean(list(specificity.values()))),
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


def plot_specificity_table(results, filename="tabla_comparativa_especificidad.png"):
    """Tabla comparativa de especificidad por clase (contraparte de
    plot_comparison_table, separada para no saturar una sola tabla con
    10 columnas — ver objetivo específico 3 de la propuesta 1 del TFM)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    headers = ["Modelo", "Exactitud",
               "Especif.\nBenigno", "Especif.\nMaligno", "Especif.\nNormal",
               "Especif.\nMacro"]

    rows = []
    for r in results:
        spec = r["specificity_per_class"]
        rows.append([
            r["name"],
            f"{r['accuracy']:.4f}",
            f"{spec['Benigno']:.4f}",
            f"{spec['Maligno']:.4f}",
            f"{spec['Normal']:.4f}",
            f"{r['specificity_macro']:.4f}",
        ])

    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colColours=["#e67e22"] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for j in range(len(headers)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    best_idx = np.argmax([r["specificity_macro"] for r in results])
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor("#fae5d3")

    fig.suptitle("Comparativa de Especificidad por Clase — Conjunto de Prueba",
                 fontsize=14, y=0.95)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def plot_sensitivity_specificity_bars(all_results, filename="sensibilidad_especificidad.png"):
    """
    Gráfico de barras agrupadas: sensibilidad vs especificidad por clase,
    para los 4 modelos, en 3 subgráficos (uno por clase) — pedido
    explícitamente en la propuesta 1 del TFM (objetivo específico 3).
    """
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(16, 5.5), sharey=True)
    model_names = [r["name"] for r in all_results]
    x = np.arange(len(model_names))
    width = 0.35
    bar_colors = {"Sensibilidad": "#2e86ab", "Especificidad": "#e67e22"}

    for i, cls in enumerate(CLASS_LABELS):
        ax = axes[i]
        sens = [r["report"][cls]["recall"] for r in all_results]
        spec = [r["specificity_per_class"][cls] for r in all_results]
        ax.bar(x - width / 2, sens, width, label="Sensibilidad", color=bar_colors["Sensibilidad"])
        ax.bar(x + width / 2, spec, width, label="Especificidad", color=bar_colors["Especificidad"])
        ax.set_title(cls, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel("Valor")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=11)
    fig.suptitle("Sensibilidad vs. Especificidad por clase — 4 modelos", fontsize=15, y=1.1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


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
        plot_specificity_table(results)

    # ── SVM baseline: SOLO LECTURA, no se reentrena ni se modifica ──────
    # Se incluye aquí únicamente para las curvas ROC comparativas y su
    # classification report como imagen. Sus métricas "oficiales" siguen
    # siendo las de metricas_svm_baseline.json (08_baseline_ml.py); esta
    # evaluación de solo lectura SÍ se usa para AÑADIRLE la especificidad
    # a ese JSON ya existente, sin volver a entrenar el modelo (objetivo
    # específico 3 de la propuesta 1 del TFM).
    print("\n[4/4] Cargando SVM baseline (solo lectura, sin reentrenar)...")
    svm_proba = load_svm_probabilities(X_test)
    svm_result = None
    if svm_proba is not None:
        svm_result = evaluate_from_proba(y_test, svm_proba, SVM_MODEL_NAME)
        print(f"  Accuracy: {svm_result['accuracy']:.4f} (ya reportado en "
              f"metricas_svm_baseline.json; no se sobreescribe salvo la especificidad)")
        plot_classification_report_image(
            svm_result["report"], SVM_MODEL_NAME, "classification_report_svm_baseline.png"
        )

        # Parchear metricas_svm_baseline.json con la especificidad, SIN
        # reentrenar el SVM ni tocar ninguno de sus campos existentes.
        svm_json_path = FIGURES_DIR / "metricas_svm_baseline.json"
        if svm_json_path.exists():
            with open(svm_json_path, "r", encoding="utf-8") as f:
                svm_metrics = json.load(f)
            svm_metrics["specificity_per_class"] = svm_result["specificity_per_class"]
            svm_metrics["specificity_macro"] = svm_result["specificity_macro"]
            with open(svm_json_path, "w", encoding="utf-8") as f:
                json.dump(svm_metrics, f, indent=2)
            print(f"  → Especificidad añadida a: {svm_json_path} (sin reentrenar)")
    else:
        print("  ⚠ No se encontró outputs/models/svm_baseline.joblib, se omite del ROC comparativo.")

    # ── Sensibilidad vs. especificidad, los 4 modelos (bar chart agrupado) ──
    all_four = list(results)
    if svm_result is not None:
        all_four.append(svm_result)
    if len(all_four) >= 2:
        print("\n  Generando gráfico de sensibilidad vs. especificidad (4 modelos)...")
        plot_sensitivity_specificity_bars(all_four)

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
            },
            "specificity_per_class": r["specificity_per_class"],
            "specificity_macro": r["specificity_macro"],
        })
    with open(FIGURES_DIR / "metricas_comparativas.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  → Métricas JSON: {FIGURES_DIR / 'metricas_comparativas.json'}")

    print("\n✓ Evaluación comparativa completada.")


if __name__ == "__main__":
    main()
