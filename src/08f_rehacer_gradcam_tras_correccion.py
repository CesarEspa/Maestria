"""
08f - Rehace SOLO el Grad-CAM de M1 (nuevo, tras la corrección de schedule)
y del modelo recomendado (nuevo criterio de Youden) si es distinto de M1 y
M4. Las filas de M4 (Grad-CAM/Grad-CAM++/Score-CAM) y sus figuras NO se
tocan — M4 no cambió (salvo que 08d haya recalculado el aumento ganador, en
cuyo caso M4 ya son modelos nuevos con los mismos archivos de salida, y este
script simplemente regenera gradcam_recomendado.png apuntando a él sin
recalcular Grad-CAM++/Score-CAM, que son costosos).

Ver outputs/experimentos/registro.md, incidencia "schedule de la tasa de
aprendizaje y criterio de selección", punto 3.
"""
import importlib
import json

import pandas as pd

import protocolo as pr
from protocolo import EXPERIMENTOS_DIR
from config import GRADCAM_DIR

mod = importlib.import_module("10_explicabilidad")


def main():
    with open(EXPERIMENTOS_DIR / "hiperparametros_finales.json", encoding="utf-8") as f:
        config_final = json.load(f)
    with open(EXPERIMENTOS_DIR / "seleccion_modelo_recomendado.json", encoding="utf-8") as f:
        seleccion = json.load(f)
    modelo_recomendado = seleccion["modelo_recomendado"]

    print("=" * 70)
    print(f"Rehaciendo Grad-CAM: M1 (nuevo) + modelo recomendado ({modelo_recomendado})")
    print("=" * 70)

    _, masks_global = pr.cargar_segmentadas()

    df_existente = pd.read_csv(EXPERIMENTOS_DIR / "explicabilidad.csv")
    modelos_fijos = {"M1", "M4"}
    modelo_recomendado_anterior = next(
        (m for m in df_existente["modelo"].unique() if m not in modelos_fijos), None
    )
    print(f"  Modelo recomendado anterior (se descartan sus filas de explicabilidad.csv): "
          f"{modelo_recomendado_anterior}")

    modelos_a_quitar = {"M1"}
    if modelo_recomendado_anterior:
        modelos_a_quitar.add(modelo_recomendado_anterior)
    conservadas = df_existente[~df_existente["modelo"].isin(modelos_a_quitar)]

    nuevas_filas = []

    print("\n--- M1 (nuevo): Grad-CAM ---")
    res_m1 = mod.procesar_modelo("M1", ["gradcam"], config_final, masks_global)
    mod.plot_grid_un_metodo(res_m1["gradcam"], "Grad-CAM — M1 (CNN propia, semilla 0, corregido)",
                             GRADCAM_DIR / "gradcam_M1.png")
    corr_m1 = mod.calcular_correlacion(res_m1["gradcam"])
    nuevas_filas += mod._filas_desde_resultados("M1", "gradcam", res_m1["gradcam"], corr_m1)
    print(f"  correlación entre clases (M1, Grad-CAM): {corr_m1:.3f}")

    if modelo_recomendado == "M1":
        print("\n  (modelo recomendado = M1, ya generado arriba — no se duplica)")
    elif modelo_recomendado == "M4":
        print(f"\n--- {modelo_recomendado} (recomendado = M4): solo Grad-CAM para la figura ---")
        print("  (Grad-CAM++/Score-CAM de M4 NO se recalculan — no cambiaron)")
        res_rec = mod.procesar_modelo("M4", ["gradcam"], config_final, masks_global)
        mod.plot_grid_un_metodo(res_rec["gradcam"], "Grad-CAM — M4 (modelo recomendado, semilla 0)",
                                 GRADCAM_DIR / "gradcam_recomendado.png")
        # No se añaden filas nuevas: las de M4 (incl. gradcam) ya están conservadas.
    else:
        print(f"\n--- {modelo_recomendado} (recomendado): Grad-CAM ---")
        res_rec = mod.procesar_modelo(modelo_recomendado, ["gradcam"], config_final, masks_global)
        mod.plot_grid_un_metodo(
            res_rec["gradcam"], f"Grad-CAM — {modelo_recomendado} (modelo recomendado, semilla 0)",
            GRADCAM_DIR / "gradcam_recomendado.png")
        corr_rec = mod.calcular_correlacion(res_rec["gradcam"])
        nuevas_filas += mod._filas_desde_resultados(modelo_recomendado, "gradcam", res_rec["gradcam"], corr_rec)
        print(f"  correlación entre clases ({modelo_recomendado}, gradcam): {corr_rec:.3f}")

    df_final = pd.concat([conservadas, pd.DataFrame(nuevas_filas)], ignore_index=True)
    df_final.to_csv(EXPERIMENTOS_DIR / "explicabilidad.csv", index=False)
    print(f"\n→ {EXPERIMENTOS_DIR / 'explicabilidad.csv'} (actualizado, filas de M4 conservadas)")

    pct_pulmonares = (df_final["fraccion_torax"] >= 0.5).mean() * 100
    print(f"  Mapas \"pulmonares\" (fraccion_torax >= 0.5): {pct_pulmonares:.1f}% de {len(df_final)}")

    print("\n✓ Grad-CAM parcial rehecho.")


if __name__ == "__main__":
    main()
