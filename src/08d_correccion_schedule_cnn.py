"""
08d - Corrección de la incidencia "schedule de la tasa de aprendizaje" (ver
outputs/experimentos/registro.md)

Diagnóstico (señalado en las correcciones del TFM, no detectado por este pipeline): en los 9
runs de M1/M2/M3 de la Tarea 3+4, la pérdida de entrenamiento nunca bajó de
ln(3)≈1.0986 y la exactitud de entrenamiento no pasó de 0.53 — la CNN nunca
llegó a entrenar de verdad. Causa: `ReduceLROnPlateau(monitor="val_f1_macro",
patience=5)` bajaba el LR hasta ~1e-6 antes de que la CNN saliera de la
meseta inicial (en el protocolo preliminar, a LR constante 1e-4, la CNN
salía de la meseta hacia la época 11 — ver curvas_cnn_base.png).

Corrección v2 (solo familia CNN — M4, M5 y SVM NO se tocan, no tienen este
problema): el primer intento (EarlyStopping estándar, start_from_epoch=15,
sin ReduceLROnPlateau) mejoró mucho el entrenamiento, pero en la rejilla de
semilla 0 la CNN ganadora no salió de su meseta inicial hasta la ÉPOCA 27 —
a solo 3 épocas de que patience=15 (límite en época 30) la hubiera cortado
sin aprender nada. Corregido con `protocolo.EarlyStoppingTrasMeseta`: solo
empieza a contar "mejor época" y paciencia cuando se cumplen dos
condiciones — época >= 15 Y la red ya salió de la meseta (loss de
entrenamiento < 1.0 en alguna época). Antes de eso no cuenta nada. Ver
protocolo.py para el detalle completo y la motivación.

Este script:
  1. Repite la rejilla de M1 (lr x dropout, semilla 0) con el callback
     corregido. Si NINGUNA de las 4 configuraciones sale de la meseta en
     60 épocas, se detiene (sys.exit) — en ese caso sí sería una señal real
     de que el modelo no puede aprender con ninguno de los hiperparámetros
     probados. En caso contrario, continúa aunque la configuración ganadora
     haya tardado en salir de la meseta (ya no hay límite de época fijo:
     la propia lógica del callback asegura margen suficiente tras salir).
  2. M2 y M3 con la configuración ganadora de M1.
  3. Recalcula qué aumento gana entre los NUEVOS M2/M3. Si cambia respecto
     al anterior (registrado en hiperparametros_finales.json), reentrena la
     rejilla de M4 y M5 con el nuevo aumento; si no cambia, no los toca.
  4. M1, M2, M3 en semillas 1 y 2 (con la config ganadora, reanudable).
  5. Actualiza hiperparametros_finales.json y busqueda_hiperparametros.csv,
     regenera curvas_M1/M2/M3.png (y curvas_M4/M5.png si cambiaron).

Los runs anteriores de M1/M2/M3 ya se movieron (fuera de este script, ver
registro.md) a outputs/experimentos/runs_descartados_schedule/ — no se
borraron, solo se sacaron de runs/ para que entrenar() no los reutilice.
"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

import protocolo as pr
from protocolo import EXPERIMENTOS_DIR, RUNS_DIR

busqueda_mod = importlib.import_module("08b_busqueda_hiperparametros")

SEED_BUSQUEDA = 0
SEMILLAS_FINALES = [1, 2]


def _cargar_config_anterior():
    with open(EXPERIMENTOS_DIR / "hiperparametros_finales.json", encoding="utf-8") as f:
        return json.load(f)


def _info_meseta(run_dir):
    """Lee el campo 'meseta' que entrenar() ya guarda en hp.json."""
    with open(run_dir / "hp.json", encoding="utf-8") as f:
        info = json.load(f)
    return info.get("meseta") or {"salio_meseta": False, "epoca_salida_meseta": None}


def main():
    print("=" * 70)
    print("TAREA 08d — Corrección del schedule de LR para la familia CNN (M1-M3)")
    print("=" * 70)

    config_anterior = _cargar_config_anterior()
    aumento_m4_anterior = config_anterior["aumento_m4_elegido"]

    # ── 1. Rejilla M1 (semilla 0, callbacks corregidos) ─────────────────
    print("\n--- M1: rejilla lr x dropout_bloques (callbacks corregidos) ---")
    filas = []
    m1_filas = []
    for lr in (3e-4, 1e-4):
        for dropout in (0.25, 0.40):
            hp = {"arquitectura": "cnn", "aumento": None, "lr": lr, "dropout_bloques": dropout}
            run_dir = pr.entrenar("M1", hp, SEED_BUSQUEDA, tipo_particion="grupos")
            fila = busqueda_mod._registrar(filas, "M1", hp, run_dir)
            m1_filas.append(fila)
            print(f"  lr={lr} dropout_bloques={dropout} -> f1_macro_val={fila['f1_macro_val']:.4f} "
                  f"({fila['tiempo_min']:.1f} min)")

    m1_mejor = busqueda_mod._elegir_mejor(m1_filas, criterio_desempate_key="lr")
    m1_mejor_run_dir = Path(m1_mejor["run_dir"])
    print(f"  → M1 elegido: lr={m1_mejor['lr']} dropout_bloques={m1_mejor['dropout_bloques']} "
          f"(f1_macro_val={m1_mejor['f1_macro_val']:.4f})")

    # ── Criterio de aceptación v2: ¿salió de la meseta ALGUNA configuración? ──
    alguna_salio = False
    for fila in m1_filas:
        info = _info_meseta(Path(fila["run_dir"]))
        estado = (f"salió en época {info['epoca_salida_meseta']}" if info["salio_meseta"]
                  else "NO SALIÓ DE LA MESETA")
        print(f"    lr={fila['lr']} dropout={fila['dropout_bloques']}: {estado}")
        alguna_salio = alguna_salio or info["salio_meseta"]

    info_ganadora = _info_meseta(m1_mejor_run_dir)
    print(f"\n  Configuración ganadora: {'salió de la meseta en época ' + str(info_ganadora['epoca_salida_meseta']) if info_ganadora['salio_meseta'] else 'NO salió de la meseta'}")

    if not alguna_salio:
        print("\n  ⚠ NINGUNA de las 4 configuraciones salió de la meseta en 60 épocas.")
        print("  DETENIÉNDOSE — revisar manualmente antes de continuar (ver registro.md).")
        sys.exit(1)
    if not info_ganadora["salio_meseta"]:
        print("\n  ⚠ La configuración GANADORA (mayor F1 macro) no salió de la meseta, "
              "pero otra configuración del grid sí lo logró. Continuando con la ganadora "
              "de todas formas (gana por F1 macro de validación, el criterio establecido) "
              "— queda documentado en registro.md.")

    # ── 2. M2 y M3 con la configuración de M1 ───────────────────────────
    print("\n--- M2: CNN + aumento geométrico (hp de M1, callbacks corregidos) ---")
    hp_m2 = {"arquitectura": "cnn", "aumento": "geometrico",
              "lr": m1_mejor["lr"], "dropout_bloques": m1_mejor["dropout_bloques"]}
    run_m2 = pr.entrenar("M2", hp_m2, SEED_BUSQUEDA, tipo_particion="grupos")
    fila_m2 = busqueda_mod._registrar(filas, "M2", hp_m2, run_m2)
    print(f"  M2 f1_macro_val={fila_m2['f1_macro_val']:.4f} ({fila_m2['tiempo_min']:.1f} min)")

    print("\n--- M3: CNN + CutMix (hp de M1, callbacks corregidos) ---")
    hp_m3 = {"arquitectura": "cnn", "aumento": "cutmix",
              "lr": m1_mejor["lr"], "dropout_bloques": m1_mejor["dropout_bloques"]}
    run_m3 = pr.entrenar("M3", hp_m3, SEED_BUSQUEDA, tipo_particion="grupos")
    fila_m3 = busqueda_mod._registrar(filas, "M3", hp_m3, run_m3)
    print(f"  M3 f1_macro_val={fila_m3['f1_macro_val']:.4f} ({fila_m3['tiempo_min']:.1f} min)")

    aumento_m4_nuevo = "geometrico" if fila_m2["f1_macro_val"] >= fila_m3["f1_macro_val"] else "cutmix"
    razon_m4 = (f"M2 (geometrico) f1_macro_val={fila_m2['f1_macro_val']:.4f} vs. "
                f"M3 (cutmix) f1_macro_val={fila_m3['f1_macro_val']:.4f} [recalculado tras "
                f"corrección de schedule]")
    print(f"\n  → Aumento recalculado para M4: {aumento_m4_nuevo} ({razon_m4})")
    cambia_aumento = (aumento_m4_nuevo != aumento_m4_anterior)
    print(f"  → ¿Cambia respecto al anterior ({aumento_m4_anterior})? "
          f"{'SÍ — se reentrena M4/M5' if cambia_aumento else 'No — se conservan M4/M5 actuales'}")

    # ── 3. M4/M5: reentrenar solo si cambió el aumento ──────────────────
    m4_mejor_hp = config_anterior["M4"]
    m5_hp = config_anterior["M5"]
    if cambia_aumento:
        print("\n--- M4: rejilla lr_ajuste x capas_descongeladas (nuevo aumento) ---")
        m4_filas = []
        for lr_ajuste in (1e-4, 1e-5):
            for capas in (20, 50):
                hp = {"arquitectura": "efficientnet", "aumento": aumento_m4_nuevo,
                      "lr_ajuste": lr_ajuste, "capas_descongeladas": capas}
                run_dir = pr.entrenar("M4", hp, SEED_BUSQUEDA, tipo_particion="grupos")
                fila = busqueda_mod._registrar(filas, "M4", hp, run_dir)
                m4_filas.append(fila)
                print(f"  lr_ajuste={lr_ajuste} capas_descongeladas={capas} -> "
                      f"f1_macro_val={fila['f1_macro_val']:.4f} ({fila['tiempo_min']:.1f} min)")
        m4_mejor = busqueda_mod._elegir_mejor(m4_filas, criterio_desempate_key="lr_ajuste")
        m4_mejor_hp = {"arquitectura": "efficientnet", "aumento": aumento_m4_nuevo,
                        "lr_ajuste": m4_mejor["lr_ajuste"], "capas_descongeladas": m4_mejor["capas_descongeladas"]}
        print(f"  → M4 elegido: lr_ajuste={m4_mejor['lr_ajuste']} "
              f"capas_descongeladas={m4_mejor['capas_descongeladas']}")

        print("\n--- M5: EfficientNet + segmentación (hp de M4, nuevo aumento) ---")
        hp_m5 = {**m4_mejor_hp}
        run_m5 = pr.entrenar("M5", hp_m5, SEED_BUSQUEDA, tipo_particion="grupos", segmentado=True)
        busqueda_mod._registrar(filas, "M5", hp_m5, run_m5)
        m5_hp = hp_m5
    else:
        print("\n  (M4/M5 no se reentrenan — conservan la configuración geométrica anterior)")

    # ── 4. M1, M2, M3 en semillas 1 y 2 ─────────────────────────────────
    print("\n--- M1/M2/M3 en semillas 1 y 2 (config corregida) ---")
    hp_m1_final = {"arquitectura": "cnn", "aumento": None,
                    "lr": m1_mejor["lr"], "dropout_bloques": m1_mejor["dropout_bloques"]}
    for seed in SEMILLAS_FINALES:
        for escenario, hp in (("M1", hp_m1_final), ("M2", hp_m2), ("M3", hp_m3)):
            print(f"  {escenario} semilla {seed}...")
            pr.entrenar(escenario, hp, seed, tipo_particion="grupos")

    # ── 5. Actualizar hiperparametros_finales.json ──────────────────────
    config_final = {
        "M1": hp_m1_final, "M2": hp_m2, "M3": hp_m3,
        "aumento_m4_elegido": aumento_m4_nuevo, "aumento_m4_razon": razon_m4,
        "M4": m4_mejor_hp, "M5": m5_hp,
        "semilla_busqueda": SEED_BUSQUEDA, "tipo_particion": "grupos",
        "correccion_schedule_cnn": True,
    }
    with open(EXPERIMENTOS_DIR / "hiperparametros_finales.json", "w", encoding="utf-8") as f:
        json.dump(config_final, f, indent=2, ensure_ascii=False)
    print(f"\n→ {EXPERIMENTOS_DIR / 'hiperparametros_finales.json'} (actualizado)")

    df_nuevo = pd.DataFrame(filas)
    busqueda_path = EXPERIMENTOS_DIR / "busqueda_hiperparametros_correccion_schedule.csv"
    df_nuevo.to_csv(busqueda_path, index=False)
    print(f"→ {busqueda_path}")

    # ── 6. Curvas de aprendizaje actualizadas ───────────────────────────
    print("\n--- Curvas de aprendizaje actualizadas ---")
    run_dirs = {"M1": str(m1_mejor_run_dir), "M2": str(run_m2), "M3": str(run_m3)}
    if cambia_aumento:
        run_dirs["M4"] = str(RUNS_DIR / pr.nombre_run("M4", m4_mejor_hp, SEED_BUSQUEDA))
        run_dirs["M5"] = str(RUNS_DIR / pr.nombre_run("M5", m5_hp, SEED_BUSQUEDA))
    for escenario, run_dir in run_dirs.items():
        busqueda_mod.plot_curvas(Path(run_dir), escenario, f"curvas_{escenario}.png")

    print("\n✓ Corrección de schedule CNN completada.")
    print(f"  Aumento M4/M5: {'CAMBIÓ a ' + aumento_m4_nuevo if cambia_aumento else 'sin cambios (' + aumento_m4_anterior + ')'}")


if __name__ == "__main__":
    main()
