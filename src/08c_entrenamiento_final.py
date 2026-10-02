"""
08c - Entrenamiento final multisemilla (Tarea 4, correcciones del TFM)

Para cada semilla en SEMILLAS, entrena M1-M5 con la configuración elegida en
la Tarea 3 (outputs/experimentos/hiperparametros_finales.json) sobre la
partición por grupos de esa semilla. El entrenamiento de semilla 0 con esa
configuración YA EXISTE (es el mismo run de la búsqueda, Tarea 3) — entrenar()
lo detecta y lo salta, no se repite.

También entrena la línea base SVM+HOG (mismos parámetros HOG que
08_baseline_ml.py), eligiendo C por F1 macro de validación en cada semilla,
sobre la partición por grupos Y por imagen (para cuantificar la inflación
por fuga de datos).

No predice sobre el conjunto de prueba en ningún momento (eso es la Tarea 5).
Antes de terminar, escribe seleccion_modelo_recomendado.json con el criterio
fijado de antemano (sensibilidad media en Maligno, validación, media de
semillas) — ANTES de que exista ninguna métrica de test.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score

import protocolo as pr
from protocolo import EXPERIMENTOS_DIR, RUNS_DIR
from config import CLASS_LABELS

import importlib
baseline_mod = importlib.import_module("08_baseline_ml")

SEMILLAS = [0, 1, 2]
ESCENARIOS = ["M1", "M2", "M3", "M4", "M5"]


def _cargar_config_final():
    path = EXPERIMENTOS_DIR / "hiperparametros_finales.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero 08b_busqueda_hiperparametros.py (Tarea 3).")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _val_f1_sens(run_dir):
    data = np.load(run_dir / "pred_val.npz")
    proba, y = data["proba"], data["y"]
    pred = proba.argmax(1)
    f1 = f1_score(y, pred, average="macro")
    sens = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return float(f1), [float(s) for s in sens]


# ─────────────────────── M1-M5: multisemilla ───────────────────────────────

def entrenar_m1_m5(config_final):
    print("=" * 70)
    print("TAREA 4 — Entrenamiento final multisemilla: M1-M5")
    print("=" * 70)
    for seed in SEMILLAS:
        for escenario in ESCENARIOS:
            hp = config_final[escenario]
            segmentado = (escenario == "M5")
            print(f"\n--- {escenario} semilla {seed} (segmentado={segmentado}) ---")
            pr.entrenar(escenario, hp, seed, tipo_particion="grupos", segmentado=segmentado)


# ───────────────────────── SVM+HOG multisemilla ─────────────────────────────

def _svm_run_dir(tipo_particion, seed):
    d = RUNS_DIR / f"SVM_{tipo_particion}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def entrenar_svm(tipos_particion=("grupos", "imagen")):
    print("\n" + "=" * 70)
    print("TAREA 4 — Línea base SVM+HOG multisemilla (grupos e imagen)")
    print("=" * 70)

    X, y = pr.cargar_X_y(segmentado=False)

    for seed in SEMILLAS:
        for tipo in tipos_particion:
            run_dir = _svm_run_dir(tipo, seed)
            modelo_path = run_dir / "modelo.joblib"
            if modelo_path.exists():
                print(f"  [saltado, ya existe] SVM_{tipo}_seed{seed}")
                continue

            print(f"\n--- SVM {tipo} semilla {seed} ---")
            tr, va, te = pr.particion(seed, tipo=tipo)
            X_train, y_train = X[tr], y[tr]
            X_val, y_val = X[va], y[va]

            t0 = time.time()
            print(f"  Extrayendo HOG de {len(X_train)} (train) + {len(X_val)} (val)...")
            X_train_hog = baseline_mod.extract_hog_features(X_train)
            X_val_hog = baseline_mod.extract_hog_features(X_val)

            scaler = StandardScaler()
            X_train_hog = scaler.fit_transform(X_train_hog)
            X_val_hog = scaler.transform(X_val_hog)

            mejor = None
            for C in (1, 10, 100):
                svm = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced",
                          probability=True, random_state=seed)
                svm.fit(X_train_hog, y_train)
                proba_val = svm.predict_proba(X_val_hog)
                f1 = f1_score(y_val, proba_val.argmax(1), average="macro")
                print(f"    C={C} -> f1_macro_val={f1:.4f}")
                if mejor is None or f1 > mejor["f1"]:
                    mejor = {"C": C, "f1": f1, "svm": svm, "proba_val": proba_val}

            tiempo_s = time.time() - t0
            print(f"  → C elegido: {mejor['C']} (f1_macro_val={mejor['f1']:.4f}), {tiempo_s/60:.1f} min")

            joblib.dump(
                {"svm": mejor["svm"], "scaler": scaler, "hog_params": baseline_mod.HOG_PARAMS},
                modelo_path,
            )
            np.savez(run_dir / "pred_val.npz", proba=mejor["proba_val"], y=y_val)
            with open(run_dir / "tiempo.json", "w", encoding="utf-8") as f:
                json.dump({"segundos": tiempo_s, "minutos": tiempo_s / 60}, f, indent=2)
            with open(run_dir / "hp.json", "w", encoding="utf-8") as f:
                json.dump({"escenario": f"SVM_{tipo}", "C": mejor["C"], "seed": seed,
                           "tipo_particion": tipo}, f, indent=2, ensure_ascii=False)


# ───────────────── Selección del modelo recomendado (pre-test) ─────────────

def seleccionar_modelo_recomendado(config_final):
    """
    Criterio fijado de antemano (corrección #8 del protocolo corregido): mayor
    sensibilidad media en Maligno sobre VALIDACIÓN (media de las 3
    semillas), entre M1-M5 (la SVM es línea base de comparación, no
    candidata a "modelo recomendado"). Desempate <0.02: F1 macro medio de
    validación. Se escribe ANTES de calcular ninguna métrica de test.
    """
    print("\n" + "=" * 70)
    print("Selección del modelo recomendado (criterio fijado de antemano, solo validación)")
    print("=" * 70)

    cifras = {}
    for escenario in ESCENARIOS:
        hp = config_final[escenario]
        segmentado = (escenario == "M5")
        sens_maligno_por_semilla, f1_por_semilla = [], []
        for seed in SEMILLAS:
            nombre = pr.nombre_run(escenario, hp, seed)
            run_dir = RUNS_DIR / nombre
            f1, sens = _val_f1_sens(run_dir)
            sens_maligno_por_semilla.append(sens[1])  # índice 1 = Maligno
            f1_por_semilla.append(f1)
        cifras[escenario] = {
            "sens_maligno_por_semilla": sens_maligno_por_semilla,
            "sens_maligno_medio": float(np.mean(sens_maligno_por_semilla)),
            "f1_macro_por_semilla": f1_por_semilla,
            "f1_macro_medio": float(np.mean(f1_por_semilla)),
            "segmentado": segmentado,
        }
        print(f"  {escenario}: sens_maligno_medio={cifras[escenario]['sens_maligno_medio']:.4f} "
              f"f1_macro_medio={cifras[escenario]['f1_macro_medio']:.4f}")

    ordenados = sorted(ESCENARIOS, key=lambda e: -cifras[e]["sens_maligno_medio"])
    mejor = ordenados[0]
    empatados = [e for e in ordenados
                 if cifras[mejor]["sens_maligno_medio"] - cifras[e]["sens_maligno_medio"] < 0.02]
    if len(empatados) > 1:
        empatados.sort(key=lambda e: -cifras[e]["f1_macro_medio"])
        mejor = empatados[0]

    resultado = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "criterio": ("Mayor sensibilidad media sobre la clase Maligno en VALIDACIÓN "
                     "(media de 3 semillas), entre M1-M5. Desempate (<0.02 de diferencia): "
                     "mayor F1 macro medio de validación. Fijado antes de calcular cualquier "
                     "métrica sobre el conjunto de prueba."),
        "modelo_recomendado": mejor,
        "cifras_validacion": cifras,
        "semillas": SEMILLAS,
    }
    out_path = EXPERIMENTOS_DIR / "seleccion_modelo_recomendado.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\n  → Modelo recomendado: {mejor}")
    print(f"  → {out_path}")
    return resultado


def main():
    config_final = _cargar_config_final()
    entrenar_m1_m5(config_final)
    entrenar_svm()
    seleccionar_modelo_recomendado(config_final)
    print("\n✓ Tarea 4 completada.")


if __name__ == "__main__":
    main()
