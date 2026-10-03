"""
Verificación de fidelidad: ¿clasifica la app web EXACTAMENTE igual que el
protocolo final evaluado en el TFM (09_evaluacion_final.py)?

No es parte del pipeline numerado (01-11) del TFM — es una herramienta de
control de calidad para la app (src/api.py, src/model_utils.py), que debe
reejecutarse cada vez que cambie el preprocesamiento de model_utils.py.

Para M1, M4, M5 y SVM_grupos (semilla 0, partición por grupos de paciente):
  1. Toma una muestra de N_MUESTRA imágenes del conjunto de prueba (156) y
     compara las probabilidades que produce la ruta de inferencia de la app
     (model_utils.predict_keras / predict_svm) contra las guardadas en
     pred_test.npz de cada run — deben coincidir con tolerancia TOLERANCIA.
  2. Calcula la exactitud de la app sobre las 156 imágenes completas y la
     compara con la de metricas_por_semilla.csv para la semilla 0.

Si algo no coincide, el preprocesamiento de la app NO es idéntico al del
protocolo y sus predicciones no deben presentarse como equivalentes a las
cifras del TFM.
"""
import sys
from pathlib import Path

import numpy as np
import joblib

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model_utils  # noqa: E402
from protocolo import RUNS_DIR  # noqa: E402

N_MUESTRA = 20
TOLERANCIA = 1e-4

# outputs/experimentos/metricas_por_semilla.csv, semilla 0 (ver registro.md).
EXACTITUD_ESPERADA = {"m1": 0.801, "m4": 0.654, "m5": 0.724, "svm": 0.859}
TOLERANCIA_EXACTITUD = 1e-3


def _cargar_pred_test(run_dir_name):
    data = np.load(RUNS_DIR / run_dir_name / "pred_test.npz")
    return data["y"], data["proba"]


def _predecir_una(spec, modelo_cargado, img01):
    if spec["type"] == "keras":
        proba, _ = model_utils.predict_keras(modelo_cargado, img01, spec)
    else:
        proba = model_utils.predict_svm(modelo_cargado, img01)
    return proba


def main():
    X_test, y_test, _ = model_utils.cargar_test_grupos_seed0()
    specs = {m["key"]: m for m in model_utils.available_models()}

    print("=" * 72)
    print(f"Verificación app vs. protocolo final — {len(X_test)} imágenes de "
          f"prueba (semilla 0, partición por grupos)")
    print("=" * 72)

    modelos_cargados = {}
    for key, spec in specs.items():
        if not spec["exists"]:
            print(f"\n⚠ {key}: no se encontró {spec['path']} — se omite.")
            continue
        if spec["type"] == "keras":
            from tensorflow import keras
            modelos_cargados[key] = keras.models.load_model(spec["path"])
        else:
            modelos_cargados[key] = joblib.load(spec["path"])

    resumen = {}
    rng = np.random.default_rng(0)
    todo_ok = True

    for key, spec in specs.items():
        if key not in modelos_cargados:
            continue

        y_pred_test, proba_guardada = _cargar_pred_test(spec["run_dir"])
        if not np.array_equal(y_pred_test, y_test):
            print(f"\n{key}: ✗ las etiquetas de pred_test.npz no coinciden con "
                  f"cargar_test_grupos_seed0() — partición distinta, abortando.")
            todo_ok = False
            continue

        # 1) Diferencia máxima de probabilidad en una muestra ──────────────
        muestra = rng.choice(len(X_test), size=min(N_MUESTRA, len(X_test)), replace=False)
        max_diff = 0.0
        for i in muestra:
            proba_app = _predecir_una(spec, modelos_cargados[key], X_test[i])
            diff = float(np.max(np.abs(proba_app - proba_guardada[i])))
            max_diff = max(max_diff, diff)

        # 2) Exactitud sobre las 156 imágenes completas, vía la ruta de la app ─
        preds_app = np.array([
            int(np.argmax(_predecir_una(spec, modelos_cargados[key], img)))
            for img in X_test
        ])
        exactitud_app = float(np.mean(preds_app == y_test))

        esperado = EXACTITUD_ESPERADA.get(key)
        ok_diff = max_diff <= TOLERANCIA
        ok_acc = esperado is None or abs(exactitud_app - esperado) < TOLERANCIA_EXACTITUD
        estado_ok = ok_diff and ok_acc
        todo_ok = todo_ok and estado_ok

        resumen[key] = {
            "max_diff_proba": max_diff, "exactitud_app": exactitud_app,
            "exactitud_esperada": esperado, "ok": estado_ok,
        }

        print(f"\n{key} ({spec['name']}): {'OK' if estado_ok else 'FALLA'}")
        print(f"  diferencia máxima de probabilidad ({len(muestra)} imágenes): "
              f"{max_diff:.6f} ({'<=' if ok_diff else '>'} {TOLERANCIA})")
        if esperado is not None:
            print(f"  exactitud vía la app (156 imágenes): {exactitud_app:.4f} "
                  f"(esperado semilla 0: {esperado:.3f}) {'✓' if ok_acc else '✗'}")
        else:
            print(f"  exactitud vía la app (156 imágenes): {exactitud_app:.4f} "
                  f"(sin valor esperado configurado)")

    print("\n" + "=" * 72)
    print("✓ TODO CORRECTO — la app reproduce exactamente las cifras del protocolo"
          if todo_ok else
          "⚠ HAY DISCREPANCIAS — no confiar en la app hasta corregir model_utils.py")
    print("=" * 72)
    return resumen, todo_ok


if __name__ == "__main__":
    _, ok = main()
    sys.exit(0 if ok else 1)
