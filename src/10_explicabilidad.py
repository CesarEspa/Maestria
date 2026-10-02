"""
10 - Explicabilidad (Tarea 6, correcciones del TFM)

Usa los modelos de semilla 0 y su conjunto de prueba (partición por grupos):
  - Grad-CAM sobre M1 y sobre el modelo recomendado (si es distinto de M1).
  - Grad-CAM++ y Score-CAM sobre M4 (capa top_conv de EfficientNetB0),
    además del Grad-CAM que ya existía para ese modelo.

Casos: por clase, hasta 2 aciertos y 2 errores (o los que existan). Para
Maligno se prioriza incluir un error maligno->normal si lo hay (el más
relevante clínicamente).

Métricas objetivas de informatividad (explicabilidad.csv):
  - fraccion_torax: fracción de la energía del mapa dentro de la envolvente
    convexa por pulmón (rellenada, dilatada 5 px) — se usa la envolvente, no
    la máscara de Otsu cruda, porque esta última excluye las masas densas y
    penalizaría injustamente a un mapa que sí señala el tumor.
  - correlacion_entre_clases: correlación de Pearson media entre TODOS los
    pares de mapas de clases distintas, dentro de cada (modelo, método) — en
    la Fase 10 anterior salió 0.999 (mapa prácticamente constante).
  - fraccion_esquina: fracción de la energía del mapa en la celda 1/7 x 1/7
    superior izquierda (en la resolución NATIVA del mapa de cada método, sin
    redimensionar), para comprobar si persiste el artefacto de la esquina.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from skimage.morphology import convex_hull_image, binary_dilation, disk
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import binary_fill_holes

import gradcam_utils as gc
import protocolo as pr
from protocolo import EXPERIMENTOS_DIR
from config import GRADCAM_DIR, CLASS_LABELS

SEED = 0
N_ACIERTOS = 2
N_ERRORES = 2
SCORECAM_TOPK = 256
NOMBRES_METODO = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "scorecam": "Score-CAM"}


def _cargar_config_final():
    with open(EXPERIMENTOS_DIR / "hiperparametros_finales.json", encoding="utf-8") as f:
        return json.load(f)


def _run_dir(modelo, seed, config_final):
    hp = config_final[modelo]
    return pr.RUNS_DIR / pr.nombre_run(modelo, hp, seed)


def _cargar_modelo_y_test(modelo, seed, config_final):
    segmentado = (modelo == "M5")
    run_dir = _run_dir(modelo, seed, config_final)
    model = keras.models.load_model(run_dir / "modelo.keras")
    X, y = pr.cargar_X_y(segmentado=segmentado)
    _, _, te = pr.particion(seed, tipo="grupos")
    return model, X[te], y[te], te


# ───────────────────────── Selección de casos ──────────────────────────────

def seleccionar_casos_clase(y_true, pred, clase_idx, n_aciertos=N_ACIERTOS,
                             n_errores=N_ERRORES, priorizar_error_a=None, seed=SEED):
    rng = np.random.default_rng(seed)
    idx_clase = np.where(y_true == clase_idx)[0]
    aciertos = idx_clase[pred[idx_clase] == clase_idx]
    errores = idx_clase[pred[idx_clase] != clase_idx]

    elegidos_aciertos = list(rng.choice(aciertos, size=min(n_aciertos, len(aciertos)), replace=False)) \
        if len(aciertos) else []

    elegidos_errores = []
    restantes = errores
    if priorizar_error_a is not None and len(errores):
        prioritarios = errores[pred[errores] == priorizar_error_a]
        if len(prioritarios):
            elegido = rng.choice(prioritarios, size=1)
            elegidos_errores.append(int(elegido[0]))
            restantes = np.array([i for i in errores if i != elegido[0]])
    faltan = n_errores - len(elegidos_errores)
    if faltan > 0 and len(restantes):
        extra = rng.choice(restantes, size=min(faltan, len(restantes)), replace=False)
        elegidos_errores += [int(i) for i in extra]

    return [int(i) for i in elegidos_aciertos], elegidos_errores


def generar_casos(y_test, pred, seed=SEED):
    casos = []
    for clase_idx in range(3):
        priorizar = 2 if clase_idx == 1 else None  # Maligno(1) -> Normal(2), el error más grave
        aciertos, errores = seleccionar_casos_clase(y_test, pred, clase_idx,
                                                      priorizar_error_a=priorizar, seed=seed)
        for idx in aciertos:
            casos.append({"clase_idx": clase_idx, "acierto": True, "idx_local": idx})
        for idx in errores:
            casos.append({"clase_idx": clase_idx, "acierto": False, "idx_local": idx})
    return casos


# ────────────────────────── Métricas objetivas ─────────────────────────────

def mascara_envolvente(mask, dilate_px=5):
    labeled = sk_label(mask)
    regions = sorted(regionprops(labeled), key=lambda r: -r.area)[:2]
    envolvente = np.zeros_like(mask, dtype=bool)
    for r in regions:
        envolvente |= convex_hull_image(labeled == r.label)
    envolvente = binary_fill_holes(envolvente)
    envolvente = binary_dilation(envolvente, disk(dilate_px))
    return envolvente


def fraccion_energia_en_mascara(heatmap, mascara_hw):
    if mascara_hw.shape != heatmap.shape:
        m = Image.fromarray((mascara_hw * 255).astype(np.uint8)).resize(
            (heatmap.shape[1], heatmap.shape[0]), Image.NEAREST
        )
        mascara_r = np.array(m) > 127
    else:
        mascara_r = mascara_hw
    total = heatmap.sum()
    return float(heatmap[mascara_r].sum() / total) if total > 0 else 0.0


def fraccion_esquina(heatmap):
    h, w = heatmap.shape
    ch, cw = max(1, h // 7), max(1, w // 7)
    total = heatmap.sum()
    return float(heatmap[:ch, :cw].sum() / total) if total > 0 else 0.0


def correlacion_entre_clases(heatmaps_por_clase):
    items = [(cls, hm.flatten()) for cls, hms in heatmaps_por_clase.items() for hm in hms]
    corrs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][0] == items[j][0]:
                continue
            a, b = items[i][1], items[j][1]
            if a.std() == 0 or b.std() == 0:
                continue
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else float("nan")


def calcular_correlacion(lista_resultados):
    """Agrupa una lista de resultados (como las que produce procesar_modelo)
    por clase_idx y calcula correlacion_entre_clases sobre los heatmaps."""
    heatmaps_por_clase = {}
    for r in lista_resultados:
        heatmaps_por_clase.setdefault(r["clase_idx"], []).append(r["heatmap"])
    return correlacion_entre_clases(heatmaps_por_clase)


# ─────────────────────────── Cálculo de mapas ───────────────────────────────

def calcular_heatmap(metodo, model, last_conv, img_array):
    if metodo == "gradcam":
        return gc.make_gradcam_heatmap(img_array, model, last_conv)
    if metodo == "gradcampp":
        return gc.make_gradcampp_heatmap(img_array, model, last_conv)
    if metodo == "scorecam":
        return gc.make_scorecam_heatmap(img_array, model, last_conv, top_k=SCORECAM_TOPK)
    raise ValueError(metodo)


def procesar_modelo(modelo_nombre, metodos, config_final, masks_global):
    model, X_test, y_test, te_idx = _cargar_modelo_y_test(modelo_nombre, SEED, config_final)
    proba = model.predict(X_test, batch_size=32, verbose=0)
    pred = proba.argmax(1)
    last_conv = gc.find_last_conv_layer_name(model)
    if last_conv is None:
        raise ValueError(f"No se encontró capa convolucional en {modelo_nombre}.")

    casos = generar_casos(y_test, pred, seed=SEED)
    print(f"  {modelo_nombre}: {len(casos)} casos seleccionados "
          f"({sum(c['acierto'] for c in casos)} aciertos, {sum(not c['acierto'] for c in casos)} errores)")

    resultados_por_metodo = {}
    for metodo in metodos:
        lista = []
        for caso in casos:
            img_array = X_test[caso["idx_local"]][np.newaxis, ...]
            heatmap, pred_idx = calcular_heatmap(metodo, model, last_conv, img_array)
            idx_global = int(te_idx[caso["idx_local"]])
            envolvente = mascara_envolvente(masks_global[idx_global], dilate_px=5)
            lista.append({
                **caso,
                "heatmap": heatmap, "pred_idx": pred_idx,
                "fraccion_torax": fraccion_energia_en_mascara(heatmap, envolvente),
                "fraccion_esquina": fraccion_esquina(heatmap),
                "imagen": X_test[caso["idx_local"]],
            })
        resultados_por_metodo[metodo] = lista
    return resultados_por_metodo


# ──────────────────────────────── Figuras ───────────────────────────────────

def plot_grid_un_metodo(lista_resultados, titulo, out_path):
    n = len(lista_resultados)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, r in enumerate(lista_resultados):
        overlay = gc.overlay_heatmap(r["imagen"], r["heatmap"])
        axes[i, 0].imshow(np.clip(r["imagen"], 0, 1)); axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay); axes[i, 1].axis("off")
        estado = "Acierto" if r["acierto"] else "Error"
        axes[i, 1].set_title(
            f"{CLASS_LABELS[r['clase_idx']]} ({estado}) → pred. {CLASS_LABELS[r['pred_idx']]}",
            fontsize=9, color="green" if r["acierto"] else "red")
    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_comparacion_metodos(resultados_por_metodo, metodos, out_path):
    casos_ref = resultados_por_metodo[metodos[0]]
    n = len(casos_ref)
    cols = 1 + len(metodos)
    fig, axes = plt.subplots(n, cols, figsize=(3.3 * cols, 3.2 * n))
    for i in range(n):
        r0 = casos_ref[i]
        axes[i, 0].imshow(np.clip(r0["imagen"], 0, 1)); axes[i, 0].axis("off")
        estado = "Acierto" if r0["acierto"] else "Error"
        axes[i, 0].set_title(f"{CLASS_LABELS[r0['clase_idx']]} ({estado})", fontsize=9)
        for j, metodo in enumerate(metodos):
            r = resultados_por_metodo[metodo][i]
            overlay = gc.overlay_heatmap(r["imagen"], r["heatmap"])
            axes[i, j + 1].imshow(overlay); axes[i, j + 1].axis("off")
            if i == 0:
                axes[i, j + 1].set_title(NOMBRES_METODO[metodo], fontsize=11)
    fig.suptitle("Comparación de métodos de explicabilidad — M4 (EfficientNetB0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ───────────────────────────────── main ──────────────────────────────────────

def _filas_desde_resultados(modelo, metodo, lista, corr):
    return [{
        "modelo": modelo, "metodo": metodo, "clase": CLASS_LABELS[r["clase_idx"]],
        "acierto": bool(r["acierto"]), "fraccion_torax": r["fraccion_torax"],
        "correlacion_entre_clases": corr, "fraccion_esquina": r["fraccion_esquina"],
    } for r in lista]


def main():
    config_final = _cargar_config_final()
    with open(EXPERIMENTOS_DIR / "seleccion_modelo_recomendado.json", encoding="utf-8") as f:
        seleccion = json.load(f)
    modelo_recomendado = seleccion["modelo_recomendado"]

    print("=" * 70)
    print(f"TAREA 6 — Explicabilidad. Modelo recomendado: {modelo_recomendado}")
    print("=" * 70)

    _, masks_global = pr.cargar_segmentadas()
    todas_filas = []

    print("\n--- M1: Grad-CAM ---")
    res_m1 = procesar_modelo("M1", ["gradcam"], config_final, masks_global)
    plot_grid_un_metodo(res_m1["gradcam"], "Grad-CAM — M1 (CNN propia, semilla 0)",
                         GRADCAM_DIR / "gradcam_M1.png")
    corr_m1 = calcular_correlacion(res_m1["gradcam"])
    todas_filas += _filas_desde_resultados("M1", "gradcam", res_m1["gradcam"], corr_m1)
    print(f"  correlación entre clases (M1, Grad-CAM): {corr_m1:.3f}")

    print("\n--- M4: Grad-CAM, Grad-CAM++, Score-CAM ---")
    res_m4 = procesar_modelo("M4", ["gradcam", "gradcampp", "scorecam"], config_final, masks_global)
    plot_comparacion_metodos(res_m4, ["gradcam", "gradcampp", "scorecam"],
                              GRADCAM_DIR / "comparacion_metodos_M4.png")
    for metodo in ("gradcam", "gradcampp", "scorecam"):
        corr = calcular_correlacion(res_m4[metodo])
        todas_filas += _filas_desde_resultados("M4", metodo, res_m4[metodo], corr)
        print(f"  correlación entre clases (M4, {metodo}): {corr:.3f}")

    if modelo_recomendado == "M1":
        print("\n  (modelo recomendado = M1, Grad-CAM ya generado — no se duplica)")
    else:
        print(f"\n--- {modelo_recomendado} (recomendado): Grad-CAM ---")
        if modelo_recomendado == "M4":
            res_rec_gradcam = res_m4["gradcam"]
            duplica_en_csv = False
        else:
            res_rec = procesar_modelo(modelo_recomendado, ["gradcam"], config_final, masks_global)
            res_rec_gradcam = res_rec["gradcam"]
            duplica_en_csv = True
        plot_grid_un_metodo(res_rec_gradcam,
                             f"Grad-CAM — {modelo_recomendado} (modelo recomendado, semilla 0)",
                             GRADCAM_DIR / "gradcam_recomendado.png")
        corr_rec = calcular_correlacion(res_rec_gradcam)
        if duplica_en_csv:
            todas_filas += _filas_desde_resultados(modelo_recomendado, "gradcam", res_rec_gradcam, corr_rec)
        print(f"  correlación entre clases ({modelo_recomendado}, gradcam): {corr_rec:.3f}")

    df = pd.DataFrame(todas_filas)
    df.to_csv(EXPERIMENTOS_DIR / "explicabilidad.csv", index=False)
    print(f"\n→ {EXPERIMENTOS_DIR / 'explicabilidad.csv'}")

    pct_pulmonares = (df["fraccion_torax"] >= 0.5).mean() * 100
    print(f"\n  Mapas \"pulmonares\" (fraccion_torax >= 0.5): {pct_pulmonares:.1f}% de {len(df)}")

    print("\n✓ Tarea 6 completada.")


if __name__ == "__main__":
    main()
