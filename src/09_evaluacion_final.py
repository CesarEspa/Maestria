"""
09 - Evaluación en prueba (una sola vez) y estadística (Tarea 5, correcciones
del TFM)

Predice sobre el conjunto de prueba UNA ÚNICA VEZ por modelo y semilla (nunca
antes de este script — ver Tarea 4, seleccion_modelo_recomendado.json se
escribe solo con cifras de validación). Modelos evaluados:
  M1, M2, M3, M4, M5          — partición por grupos de paciente
  SVM_grupos                  — SVM+HOG, partición por grupos
  SVM_imagen                  — SVM+HOG, partición por imagen (referencia,
                                 para cuantificar la inflación por fuga)

Nota sobre McNemar y SVM_imagen: McNemar exige predicciones PAREADAS sobre
los MISMOS casos de prueba. SVM_imagen usa una partición distinta (distintos
índices de test) a la de M1-M5/SVM_grupos, así que no es pareable contra el
modelo recomendado — se excluye de mcnemar.csv, pero se evalúa igual en
metricas_por_semilla.csv, resumen_ic95.csv y su propia figura comparativa
(svm_imagen_vs_grupos.png), que es su propósito real (cuantificar la
inflación, no competir en igualdad de condiciones).
"""
import importlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve, auc as sk_auc,
)
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
from statsmodels.stats.multitest import multipletests

import protocolo as pr
from protocolo import EXPERIMENTOS_DIR, RUNS_DIR
from config import FIGURES_DIR, CLASS_LABELS

baseline_mod = importlib.import_module("08_baseline_ml")

SEMILLAS = [0, 1, 2]
MODELOS = ["M1", "M2", "M3", "M4", "M5", "SVM_grupos", "SVM_imagen"]
METRICAS_BOOTSTRAP = ["exactitud", "f1_macro", "auc_macro", "sens_maligno", "sens_benigno"]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
sns.set_theme(style="whitegrid", font_scale=1.0)


def _cargar_config_final():
    with open(EXPERIMENTOS_DIR / "hiperparametros_finales.json", encoding="utf-8") as f:
        return json.load(f)


def specificity_per_class_from_cm(cm):
    """Misma fórmula que 06_evaluate.py / 08_baseline_ml.py (uno-contra-el-resto)."""
    n = cm.shape[0]
    total = cm.sum()
    out = []
    for i in range(n):
        col_i = cm[:, i].sum()
        row_i = cm[i, :].sum()
        fp = col_i - cm[i, i]
        vn = total - row_i - col_i + cm[i, i]
        out.append(float(vn / (vn + fp)) if (vn + fp) > 0 else 0.0)
    return out  # [Benigno, Maligno, Normal]


# ─────────────────────── Predicción de test (una vez) ──────────────────────

def _run_dir_modelo(modelo, seed, config_final):
    if modelo.startswith("SVM"):
        tipo = modelo.split("_", 1)[1]
        return RUNS_DIR / f"SVM_{tipo}_seed{seed}"
    hp = config_final[modelo]
    return RUNS_DIR / pr.nombre_run(modelo, hp, seed)


def _predecir(modelo, seed, config_final):
    run_dir = _run_dir_modelo(modelo, seed, config_final)
    pred_path = run_dir / "pred_test.npz"
    if pred_path.exists():
        data = np.load(pred_path)
        return data["y"], data["proba"]

    if modelo.startswith("SVM"):
        tipo = modelo.split("_", 1)[1]
        bundle = joblib.load(run_dir / "modelo.joblib")
        X, y = pr.cargar_X_y(segmentado=False)
        _, _, te = pr.particion(seed, tipo=tipo)
        X_test, y_test = X[te], y[te]
        X_test_hog = baseline_mod.extract_hog_features(X_test)
        X_test_hog = bundle["scaler"].transform(X_test_hog)
        proba = bundle["svm"].predict_proba(X_test_hog)
    else:
        segmentado = (modelo == "M5")
        model = keras.models.load_model(run_dir / "modelo.keras")
        X, y = pr.cargar_X_y(segmentado=segmentado)
        _, _, te = pr.particion(seed, tipo="grupos")
        X_test, y_test = X[te], y[te]
        proba = model.predict(X_test, batch_size=32, verbose=0)

    np.savez(pred_path, proba=proba, y=y_test)
    return y_test, proba


# ─────────────────────── Métricas por modelo/semilla ────────────────────────

def fila_metricas(modelo, seed, y_test, proba):
    pred = proba.argmax(1)
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    sens = recall_score(y_test, pred, labels=[0, 1, 2], average=None, zero_division=0)
    esp = specificity_per_class_from_cm(cm)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    try:
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {
        "modelo": modelo, "semilla": seed, "n_test": int(len(y_test)),
        "exactitud": float(acc), "f1_macro": float(f1), "auc_macro": float(auc),
        "sens_benigno": float(sens[0]), "sens_maligno": float(sens[1]), "sens_normal": float(sens[2]),
        "esp_benigno": esp[0], "esp_maligno": esp[1], "esp_normal": esp[2],
        "mal_a_normal": int(cm[1, 2]), "mal_a_benigno": int(cm[1, 0]),
    }


# ───────────────────── Resumen con IC 95% por bootstrap ────────────────────

def _metric_value(y_true, proba, metrica):
    pred = proba.argmax(1)
    if metrica == "exactitud":
        return accuracy_score(y_true, pred)
    if metrica == "f1_macro":
        return f1_score(y_true, pred, average="macro")
    if metrica == "auc_macro":
        try:
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            return roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        except ValueError:
            return float("nan")
    if metrica == "sens_maligno":
        return float(recall_score(y_true, pred, labels=[1], average=None, zero_division=0)[0])
    if metrica == "sens_benigno":
        return float(recall_score(y_true, pred, labels=[0], average=None, zero_division=0)[0])
    raise ValueError(metrica)


def bootstrap_ci(y_true, proba, metrica, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """IC 95% por bootstrap (remuestreo de imágenes con reemplazo), semilla fija."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    valores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        v = _metric_value(y_true[idx], proba[idx], metrica)
        if not (isinstance(v, float) and np.isnan(v)):
            valores.append(v)
    valores = np.asarray(valores)
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def filas_resumen_modelo(modelo, filas_semilla_modelo, y_agregado, proba_agregado):
    filas = []
    for metrica in METRICAS_BOOTSTRAP:
        valores = [f[metrica] for f in filas_semilla_modelo
                   if not (isinstance(f[metrica], float) and np.isnan(f[metrica]))]
        media = float(np.mean(valores))
        de = float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0
        ic_inf, ic_sup = bootstrap_ci(y_agregado, proba_agregado, metrica)
        filas.append({"modelo": modelo, "metrica": metrica, "media": media, "de": de,
                       "ic95_inf": ic_inf, "ic95_sup": ic_sup})
    return filas


# ──────────────────────────────── McNemar ───────────────────────────────────

def calcular_mcnemar(predicciones, modelo_recomendado, modelos_comparables):
    filas = []
    for seed_key in SEMILLAS + ["agregado"]:
        y_a, proba_a = predicciones[modelo_recomendado][seed_key]
        pred_a_ok = proba_a.argmax(1) == y_a

        pvals, comparaciones = [], []
        for modelo_b in modelos_comparables:
            if modelo_b == modelo_recomendado:
                continue
            y_b, proba_b = predicciones[modelo_b][seed_key]
            if not np.array_equal(y_a, y_b):
                raise ValueError(
                    f"y distinto entre {modelo_recomendado} y {modelo_b} en {seed_key} "
                    "(no son parejables para McNemar)."
                )
            pred_b_ok = proba_b.argmax(1) == y_b
            a = int(np.sum(pred_a_ok & pred_b_ok))
            b = int(np.sum(pred_a_ok & ~pred_b_ok))
            c = int(np.sum(~pred_a_ok & pred_b_ok))
            d = int(np.sum(~pred_a_ok & ~pred_b_ok))
            resultado = mcnemar_test([[a, b], [c, d]], exact=True)
            pvals.append(resultado.pvalue)
            comparaciones.append((modelo_b, b, c, resultado.pvalue))

        if pvals:
            _, p_holm, _, _ = multipletests(pvals, method="holm")
            for (modelo_b, b, c, p), ph in zip(comparaciones, p_holm):
                filas.append({
                    "modelo_a": modelo_recomendado, "modelo_b": modelo_b,
                    "semilla": seed_key, "b": b, "c": c,
                    "p_valor": float(p), "p_holm": float(ph),
                })
    return filas


# ───────────────────── Errores clínicos + matrices agregadas ───────────────

def calcular_errores_y_matrices(predicciones):
    filas, matrices = [], {}
    for modelo in MODELOS:
        y_agg, proba_agg = predicciones[modelo]["agregado"]
        pred_agg = proba_agg.argmax(1)
        cm = confusion_matrix(y_agg, pred_agg, labels=[0, 1, 2])
        matrices[modelo] = cm.tolist()
        filas.append({
            "modelo": modelo,
            "malignos_total": int(cm[1, :].sum()),
            "mal_a_normal": int(cm[1, 2]),
            "mal_a_benigno": int(cm[1, 0]),
        })
    return filas, matrices


# ──────────────────────────────── Figuras ───────────────────────────────────

def plot_resumen_ic(df_resumen, filename="resumen_modelos_ic.png"):
    modelos_fig = ["M1", "M2", "M3", "M4", "M5", "SVM_grupos"]
    metricas_fig = [("f1_macro", "F1 Macro"), ("sens_maligno", "Sensibilidad Maligno"),
                     ("sens_benigno", "Sensibilidad Benigno")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (metrica, titulo) in zip(axes, metricas_fig):
        medias, errs_lo, errs_hi = [], [], []
        for m in modelos_fig:
            row = df_resumen[(df_resumen.modelo == m) & (df_resumen.metrica == metrica)].iloc[0]
            medias.append(row["media"])
            errs_lo.append(max(0.0, row["media"] - row["ic95_inf"]))
            errs_hi.append(max(0.0, row["ic95_sup"] - row["media"]))
        x = np.arange(len(modelos_fig))
        ax.bar(x, medias, yerr=[errs_lo, errs_hi], capsize=4, color="#3498db")
        ax.set_xticks(x); ax.set_xticklabels(modelos_fig, rotation=20)
        ax.set_title(titulo); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Resumen de modelos — media e IC 95% bootstrap (test agregado, 3 semillas)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIGURES_DIR / filename}")


def plot_matrices_confusion(matrices, filename="matrices_confusion_agregadas.png"):
    cols = 4
    rows = int(np.ceil(len(MODELOS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i, modelo in enumerate(MODELOS):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        cm = np.array(matrices[modelo])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{cm[a, b]}\n({cm_pct[a, b]:.0f}%)" for b in range(3)] for a in range(3)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=CLASS_LABELS,
                    yticklabels=CLASS_LABELS, ax=ax, cbar=False)
        ax.set_title(modelo); ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    for j in range(len(MODELOS), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.suptitle("Matrices de confusión agregadas (suma de 3 semillas) — recuento y % por fila",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIGURES_DIR / filename}")


def _macro_roc(y_true, proba):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr = {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    return all_fpr, mean_tpr, sk_auc(all_fpr, mean_tpr)


def plot_roc_final(predicciones, filename="curvas_roc_final.png"):
    colores = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#1abc9c", "#c0392b", "#7f8c8d"]
    fig, ax = plt.subplots(figsize=(8, 7))
    for modelo, color in zip(MODELOS, colores):
        y_agg, proba_agg = predicciones[modelo]["agregado"]
        fpr, tpr, auc_val = _macro_roc(y_agg, proba_agg)
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{modelo} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curvas ROC macro (one-vs-rest) — predicciones de test agregadas (3 semillas)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → {FIGURES_DIR / filename}")


def plot_svm_comparacion(df_resumen, filename="svm_imagen_vs_grupos.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, metrica, titulo in zip(axes, ["exactitud", "f1_macro"], ["Exactitud", "F1 Macro"]):
        modelos_fig = ["SVM_grupos", "SVM_imagen"]
        medias, errs_lo, errs_hi = [], [], []
        for m in modelos_fig:
            row = df_resumen[(df_resumen.modelo == m) & (df_resumen.metrica == metrica)].iloc[0]
            medias.append(row["media"])
            errs_lo.append(max(0.0, row["media"] - row["ic95_inf"]))
            errs_hi.append(max(0.0, row["ic95_sup"] - row["media"]))
        x = np.arange(2)
        ax.bar(x, medias, yerr=[errs_lo, errs_hi], capsize=5, color=["#2ecc71", "#e74c3c"])
        ax.set_xticks(x); ax.set_xticklabels(["Por grupos\n(sin fuga)", "Por imagen\n(con fuga)"])
        ax.set_title(titulo); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("SVM+HOG: partición por grupos vs. por imagen — inflación por fuga de datos",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIGURES_DIR / filename}")


# ───────────────────────────────── main ──────────────────────────────────────

def main():
    config_final = _cargar_config_final()
    sel_path = EXPERIMENTOS_DIR / "seleccion_modelo_recomendado.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"No existe {sel_path}. Ejecuta primero 08c_entrenamiento_final.py (Tarea 4).")
    with open(sel_path, encoding="utf-8") as f:
        seleccion = json.load(f)
    modelo_recomendado = seleccion["modelo_recomendado"]

    print("=" * 70)
    print(f"TAREA 5 — Evaluación en prueba (una sola vez). Modelo recomendado: {modelo_recomendado}")
    print("=" * 70)

    predicciones = {}
    filas_semilla = []
    for modelo in MODELOS:
        predicciones[modelo] = {}
        ys, probas = [], []
        for seed in SEMILLAS:
            y_test, proba = _predecir(modelo, seed, config_final)
            predicciones[modelo][seed] = (y_test, proba)
            ys.append(y_test); probas.append(proba)
            fila = fila_metricas(modelo, seed, y_test, proba)
            filas_semilla.append(fila)
            print(f"  {modelo} semilla {seed}: n_test={fila['n_test']} "
                  f"exactitud={fila['exactitud']:.4f} f1_macro={fila['f1_macro']:.4f} "
                  f"auc_macro={fila['auc_macro']:.4f}")
        y_agg, proba_agg = np.concatenate(ys), np.concatenate(probas)
        predicciones[modelo]["agregado"] = (y_agg, proba_agg)

    df_semilla = pd.DataFrame(filas_semilla)
    df_semilla.to_csv(EXPERIMENTOS_DIR / "metricas_por_semilla.csv", index=False)
    print(f"\n→ {EXPERIMENTOS_DIR / 'metricas_por_semilla.csv'}")

    print("\n--- Resumen con IC 95% (bootstrap, 2000 remuestreos) ---")
    filas_resumen = []
    for modelo in MODELOS:
        filas_modelo = [f for f in filas_semilla if f["modelo"] == modelo]
        y_agg, proba_agg = predicciones[modelo]["agregado"]
        filas_resumen += filas_resumen_modelo(modelo, filas_modelo, y_agg, proba_agg)
    df_resumen = pd.DataFrame(filas_resumen)
    df_resumen.to_csv(EXPERIMENTOS_DIR / "resumen_ic95.csv", index=False)
    print(f"→ {EXPERIMENTOS_DIR / 'resumen_ic95.csv'}")

    print("\n--- McNemar exacto (modelo recomendado vs. cada uno de los demás) ---")
    modelos_comparables = [m for m in MODELOS if m != "SVM_imagen"]
    filas_mcnemar = calcular_mcnemar(predicciones, modelo_recomendado, modelos_comparables)
    pd.DataFrame(filas_mcnemar).to_csv(EXPERIMENTOS_DIR / "mcnemar.csv", index=False)
    print(f"→ {EXPERIMENTOS_DIR / 'mcnemar.csv'} (SVM_imagen excluido: partición distinta, no pareable)")

    print("\n--- Errores clínicos (maligno mal clasificado) y matrices agregadas ---")
    filas_errores, matrices = calcular_errores_y_matrices(predicciones)
    pd.DataFrame(filas_errores).to_csv(EXPERIMENTOS_DIR / "errores_maligno.csv", index=False)
    with open(EXPERIMENTOS_DIR / "matrices_confusion_agregadas.json", "w", encoding="utf-8") as f:
        json.dump(matrices, f, indent=2)
    print(f"→ {EXPERIMENTOS_DIR / 'errores_maligno.csv'}")
    print(f"→ {EXPERIMENTOS_DIR / 'matrices_confusion_agregadas.json'}")

    print("\n--- Figuras ---")
    plot_resumen_ic(df_resumen)
    plot_matrices_confusion(matrices)
    plot_roc_final(predicciones)
    plot_svm_comparacion(df_resumen)

    print("\n✓ Tarea 5 completada.")


if __name__ == "__main__":
    main()
