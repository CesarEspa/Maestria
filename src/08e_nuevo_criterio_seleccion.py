"""
08e - Nuevo criterio de selección del modelo recomendado (incidencia
"schedule de la tasa de aprendizaje y criterio de selección", ver
outputs/experimentos/registro.md)

El criterio anterior (mayor sensibilidad media en Maligno, validación) lo
maximiza trivialmente un modelo que predice "Maligno" para todo — fue
exactamente lo que pasó con M3 bajo el criterio v1. Nuevo criterio: mayor
**índice de Youden de la clase Maligno** en VALIDACIÓN (sensibilidad Maligno
+ especificidad Maligno − 1), media de las 3 semillas, entre M1-M5.
Desempate si la diferencia es menor de 0.02: mayor F1 macro medio de
validación. Un clasificador constante tiene Youden = 0 exactamente.

Requiere que M1-M5 ya estén entrenados en las 3 semillas con la
configuración FINAL (hiperparametros_finales.json, ya corregido por
08d_correccion_schedule_cnn.py). Se ejecuta ANTES de recalcular ninguna
métrica de prueba — solo usa pred_val.npz.

seleccion_modelo_recomendado.json (el del criterio v1) se renombra a
seleccion_modelo_recomendado_v1_descartada.json; este script escribe el
nuevo con el criterio de Youden.
"""
import json
from datetime import datetime

import numpy as np

import protocolo as pr
from protocolo import EXPERIMENTOS_DIR, RUNS_DIR

SEMILLAS = [0, 1, 2]
ESCENARIOS = ["M1", "M2", "M3", "M4", "M5"]


def _especificidad_maligno(y_true, pred):
    """Especificidad de la clase Maligno (índice 1), uno-contra-el-resto."""
    es_maligno_real = (y_true == 1)
    es_maligno_pred = (pred == 1)
    vn = int(np.sum(~es_maligno_real & ~es_maligno_pred))
    fp = int(np.sum(~es_maligno_real & es_maligno_pred))
    return vn / (vn + fp) if (vn + fp) > 0 else 0.0


def _sensibilidad_maligno(y_true, pred):
    es_maligno_real = (y_true == 1)
    vp = int(np.sum(es_maligno_real & (pred == 1)))
    total = int(np.sum(es_maligno_real))
    return vp / total if total > 0 else 0.0


def _f1_macro(y_true, pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, pred, average="macro"))


def main():
    with open(EXPERIMENTOS_DIR / "hiperparametros_finales.json", encoding="utf-8") as f:
        config_final = json.load(f)

    print("=" * 70)
    print("Nuevo criterio de selección: índice de Youden de Maligno (validación)")
    print("=" * 70)

    cifras = {}
    for escenario in ESCENARIOS:
        hp = config_final[escenario]
        youden_por_semilla, f1_por_semilla = [], []
        sens_por_semilla, esp_por_semilla = [], []
        for seed in SEMILLAS:
            nombre = pr.nombre_run(escenario, hp, seed)
            run_dir = RUNS_DIR / nombre
            data = np.load(run_dir / "pred_val.npz")
            proba, y = data["proba"], data["y"]
            pred = proba.argmax(1)
            sens = _sensibilidad_maligno(y, pred)
            esp = _especificidad_maligno(y, pred)
            youden = sens + esp - 1
            sens_por_semilla.append(sens)
            esp_por_semilla.append(esp)
            youden_por_semilla.append(youden)
            f1_por_semilla.append(_f1_macro(y, pred))

        cifras[escenario] = {
            "sens_maligno_por_semilla": sens_por_semilla,
            "esp_maligno_por_semilla": esp_por_semilla,
            "youden_maligno_por_semilla": youden_por_semilla,
            "youden_maligno_medio": float(np.mean(youden_por_semilla)),
            "f1_macro_por_semilla": f1_por_semilla,
            "f1_macro_medio": float(np.mean(f1_por_semilla)),
            "segmentado": (escenario == "M5"),
        }
        print(f"  {escenario}: youden_maligno_medio={cifras[escenario]['youden_maligno_medio']:.4f} "
              f"(sens={np.mean(sens_por_semilla):.3f}, esp={np.mean(esp_por_semilla):.3f}) "
              f"f1_macro_medio={cifras[escenario]['f1_macro_medio']:.4f}")

    ordenados = sorted(ESCENARIOS, key=lambda e: -cifras[e]["youden_maligno_medio"])
    mejor = ordenados[0]
    empatados = [e for e in ordenados
                 if cifras[mejor]["youden_maligno_medio"] - cifras[e]["youden_maligno_medio"] < 0.02]
    if len(empatados) > 1:
        empatados.sort(key=lambda e: -cifras[e]["f1_macro_medio"])
        mejor = empatados[0]

    # ── Renombrar el JSON v1 (criterio defectuoso) y escribir el nuevo ──
    v1_path = EXPERIMENTOS_DIR / "seleccion_modelo_recomendado.json"
    v1_descartada_path = EXPERIMENTOS_DIR / "seleccion_modelo_recomendado_v1_descartada.json"
    if v1_path.exists() and not v1_descartada_path.exists():
        v1_path.rename(v1_descartada_path)
        print(f"\n  → {v1_path.name} renombrado a {v1_descartada_path.name}")

    resultado = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "criterio": ("Mayor ÍNDICE DE YOUDEN de la clase Maligno en VALIDACIÓN "
                     "(sensibilidad Maligno + especificidad Maligno - 1), media de 3 "
                     "semillas, entre M1-M5. Desempate (<0.02 de diferencia): mayor F1 "
                     "macro medio de validación. Un clasificador constante tiene Youden=0. "
                     "Sustituye al criterio v1 (solo sensibilidad Maligno), que un modelo "
                     "colapsado a predecir siempre Maligno maximizaba trivialmente — ver "
                     "incidencia 'schedule de la tasa de aprendizaje y criterio de "
                     "selección' en registro.md."),
        "modelo_recomendado": mejor,
        "cifras_validacion": cifras,
        "semillas": SEMILLAS,
        "criterio_anterior_descartado": "seleccion_modelo_recomendado_v1_descartada.json",
    }
    with open(v1_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\n  → Modelo recomendado (nuevo criterio): {mejor}")
    print(f"  → {v1_path}")
    return resultado


if __name__ == "__main__":
    main()
