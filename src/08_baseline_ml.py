"""
08 - Línea Base de Machine Learning Clásico (SVM)
Extrae características HOG y entrena un SVM como punto de comparación.
Esto demuestra la ventaja de deep learning sobre métodos tradicionales.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

from config import (
    SPLITS_DIR, MODELS_DIR, FIGURES_DIR, CLASS_LABELS, NUM_CLASSES, SEED
)
from progress_tracker import mark_status

MODEL_KEY = "svm_baseline"
DISPLAY_NAME = "SVM (HOG) — Línea Base"

# Parámetros de HOG centralizados aquí para que la app web use exactamente
# la misma extracción de características al clasificar una imagen nueva.
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), feature_vector=True)

np.random.seed(SEED)
sns.set_theme(style="whitegrid", font_scale=1.1)


def extract_hog_features(images):
    """Extrae características HOG de cada imagen."""
    features = []
    for img in images:
        # Convertir a escala de grises (promedio de canales)
        gray = np.mean(img, axis=2)
        feat = hog(gray, **HOG_PARAMS)
        features.append(feat)
    return np.array(features)


def plot_confusion_matrix(cm, filename):
    """Matriz de confusión del SVM."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                ax=ax, linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor Real")
    ax.set_title("Matriz de Confusión — Línea Base SVM (HOG)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → Guardado: {FIGURES_DIR / filename}")


def main():
    print("=" * 60)
    print("  LÍNEA BASE — SVM con características HOG")
    print("=" * 60)

    mark_status(MODEL_KEY, DISPLAY_NAME, "running", current_epoch=0, total_epochs=1)

    try:
        print("\n[1/4] Cargando datos preprocesados...")
        data = np.load(SPLITS_DIR / "dataset_splits.npz")
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        print(f"\n[2/4] Extrayendo características HOG...")
        print(f"  → Train: {X_train.shape[0]} imágenes...")
        X_train_hog = extract_hog_features(X_train)
        print(f"  → Test: {X_test.shape[0]} imágenes...")
        X_test_hog = extract_hog_features(X_test)
        print(f"  → Shape de features: {X_train_hog.shape[1]} dimensiones")

        # Normalizar
        scaler = StandardScaler()
        X_train_hog = scaler.fit_transform(X_train_hog)
        X_test_hog = scaler.transform(X_test_hog)

        print(f"\n[3/4] Entrenando SVM (kernel RBF)...")
        svm = SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=SEED,
        )
        svm.fit(X_train_hog, y_train)

        print(f"\n[4/4] Evaluando en test...")
        y_pred = svm.predict(X_test_hog)
        y_proba = svm.predict_proba(X_test_hog)

        acc = accuracy_score(y_test, y_pred)
        try:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
            auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = None

        print(f"\n  Accuracy: {acc:.4f}")
        print(f"  AUC-ROC:  {auc:.4f}" if auc else "  AUC-ROC: N/A")
        print()
        print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, "confusion_matrix_svm_baseline.png")

        # Guardar métricas
        report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, output_dict=True)
        metrics = {
            "name": "SVM (HOG) — Línea Base",
            "accuracy": acc,
            "auc_roc": auc,
            "f1_macro": report["macro avg"]["f1-score"],
            "sensitivity_per_class": {
                cls: report[cls]["recall"] for cls in CLASS_LABELS
            }
        }
        with open(FIGURES_DIR / "metricas_svm_baseline.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  → Métricas JSON: {FIGURES_DIR / 'metricas_svm_baseline.json'}")

        # Guardar modelo + scaler + parámetros HOG para que la app web
        # pueda clasificar imágenes nuevas sin reentrenar.
        joblib.dump(
            {"svm": svm, "scaler": scaler, "hog_params": HOG_PARAMS},
            MODELS_DIR / "svm_baseline.joblib",
        )
        print(f"  → Modelo: {MODELS_DIR / 'svm_baseline.joblib'}")

        print("\n✓ Línea base SVM completada.")

        mark_status(MODEL_KEY, DISPLAY_NAME, "completed", current_epoch=1,
                    final_val_accuracy=float(acc))
    except Exception as e:
        mark_status(MODEL_KEY, DISPLAY_NAME, "failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
