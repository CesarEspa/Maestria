"""
08b - Búsqueda de hiperparámetros (Tarea 3, correcciones del TFM)

Solo validación, semilla 0, partición por grupos de paciente. Cada
escenario (M1-M5) cambia un solo factor respecto al anterior:

  M1: CNN propia, sin aumento            — referencia
  M2: CNN propia, aumento geométrico     — aísla el aumento geométrico
  M3: CNN propia, CutMix                 — aísla el aumento avanzado
  M4: EfficientNetB0, el mejor de M2/M3  — aísla la transferencia
  M5: EfficientNetB0, igual que M4, pero sobre imagen segmentada — aísla
      la segmentación

No se predice sobre el conjunto de prueba en ningún momento de este script
(ver PROMPT_CORRECCIONES_TUTORA.md, regla general y Tarea 3).
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score

import protocolo as pr
from config import FIGURES_DIR
from protocolo import EXPERIMENTOS_DIR

SEED = 0


def _val_f1_sens(run_dir):
    """F1 macro y sensibilidad por clase ([Benigno, Maligno, Normal]) sobre
    las predicciones de validación ya guardadas por entrenar()."""
    data = np.load(run_dir / "pred_val.npz")
    proba, y = data["proba"], data["y"]
    pred = proba.argmax(1)
    f1 = f1_score(y, pred, average="macro")
    sens = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return float(f1), [float(s) for s in sens]


def _tiempo_min(run_dir):
    with open(run_dir / "tiempo.json", encoding="utf-8") as f:
        return json.load(f)["minutos"]


def _registrar(filas, escenario, hp, run_dir):
    f1, sens = _val_f1_sens(run_dir)
    fila = {
        "escenario": escenario, **hp,
        "f1_macro_val": f1,
        "sens_benigno_val": sens[0], "sens_maligno_val": sens[1], "sens_normal_val": sens[2],
        "tiempo_min": _tiempo_min(run_dir), "run_dir": str(run_dir),
    }
    filas.append(fila)
    return fila


def _elegir_mejor(filas, criterio_desempate_key=None):
    """Mayor f1_macro_val; si dos quedan a <0.01, desempata por mayor
    sensibilidad maligna; si persiste, por el valor más bajo de
    criterio_desempate_key (p.ej. 'lr' o 'lr_ajuste')."""
    ordenadas = sorted(filas, key=lambda r: -r["f1_macro_val"])
    mejor = ordenadas[0]
    empatados_f1 = [r for r in ordenadas if mejor["f1_macro_val"] - r["f1_macro_val"] < 0.01]
    if len(empatados_f1) > 1:
        empatados_f1.sort(key=lambda r: -r["sens_maligno_val"])
        mejor = empatados_f1[0]
        empatados_sens = [r for r in empatados_f1
                           if mejor["sens_maligno_val"] - r["sens_maligno_val"] < 1e-9]
        if len(empatados_sens) > 1 and criterio_desempate_key:
            empatados_sens.sort(key=lambda r: r[criterio_desempate_key])
            mejor = empatados_sens[0]
    return mejor


def plot_curvas(run_dir, escenario, filename):
    """3 paneles: pérdida, exactitud y F1 macro de validación, con la época
    restaurada (máximo val_f1_macro) marcada; si hay 2 fases (EfficientNet),
    marca también el inicio del fine-tuning."""
    hist = pd.read_csv(run_dir / "historial.csv")
    x = np.arange(len(hist))
    best_i = int(hist["val_f1_macro"].idxmax())
    fases = list(dict.fromkeys(hist["fase"]))  # orden de aparición, sin duplicar
    fase2_inicio = int((hist["fase"] == fases[0]).sum()) if len(fases) > 1 else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(x, hist["loss"], label="Entrenamiento", linewidth=2)
    axes[0].plot(x, hist["val_loss"], label="Validación", linewidth=2)
    axes[0].set_title("Pérdida"); axes[0].set_xlabel("Época"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(x, hist["accuracy"], label="Entrenamiento", linewidth=2)
    axes[1].plot(x, hist["val_accuracy"], label="Validación", linewidth=2)
    axes[1].set_title("Exactitud"); axes[1].set_xlabel("Época"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(x, hist["val_f1_macro"], label="F1 macro (validación)", color="green", linewidth=2)
    axes[2].set_title("F1 macro de validación"); axes[2].set_xlabel("Época"); axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.axvline(best_i, color="gray", linestyle="--", alpha=0.8)
        if fase2_inicio is not None:
            ax.axvline(fase2_inicio, color="black", linestyle=":", alpha=0.5)
    axes[2].plot([], [], color="gray", linestyle="--", label="Época restaurada")
    if fase2_inicio is not None:
        axes[2].plot([], [], color="black", linestyle=":", label="Inicio fine-tuning")
    axes[2].legend()

    fig.suptitle(f"{escenario} — Curvas de aprendizaje (configuración elegida, semilla {SEED})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"  → {FIGURES_DIR / filename}")


def main():
    filas = []
    print("=" * 70)
    print("TAREA 3 — Búsqueda de hiperparámetros (semilla 0, partición por grupos)")
    print("=" * 70)

    # ── M1: rejilla CNN propia ──────────────────────────────────────────
    print("\n--- M1: rejilla lr x dropout_bloques (4 entrenamientos) ---")
    m1_filas = []
    for lr in (3e-4, 1e-4):
        for dropout in (0.25, 0.40):
            hp = {"arquitectura": "cnn", "aumento": None, "lr": lr, "dropout_bloques": dropout}
            run_dir = pr.entrenar("M1", hp, SEED, tipo_particion="grupos")
            fila = _registrar(filas, "M1", hp, run_dir)
            m1_filas.append(fila)
            print(f"  lr={lr} dropout_bloques={dropout} -> f1_macro_val={fila['f1_macro_val']:.4f} "
                  f"({fila['tiempo_min']:.1f} min)")

    m1_mejor = _elegir_mejor(m1_filas, criterio_desempate_key="lr")
    print(f"  → M1 elegido: lr={m1_mejor['lr']} dropout_bloques={m1_mejor['dropout_bloques']} "
          f"(f1_macro_val={m1_mejor['f1_macro_val']:.4f})")

    # ── M2: CNN + geométrico, hp de M1 ──────────────────────────────────
    print("\n--- M2: CNN + aumento geométrico (hp de M1) ---")
    hp_m2 = {"arquitectura": "cnn", "aumento": "geometrico",
              "lr": m1_mejor["lr"], "dropout_bloques": m1_mejor["dropout_bloques"]}
    run_m2 = pr.entrenar("M2", hp_m2, SEED, tipo_particion="grupos")
    fila_m2 = _registrar(filas, "M2", hp_m2, run_m2)
    print(f"  M2 f1_macro_val={fila_m2['f1_macro_val']:.4f} ({fila_m2['tiempo_min']:.1f} min)")

    # ── M3: CNN + CutMix, hp de M1 ──────────────────────────────────────
    print("\n--- M3: CNN + CutMix (hp de M1) ---")
    hp_m3 = {"arquitectura": "cnn", "aumento": "cutmix",
              "lr": m1_mejor["lr"], "dropout_bloques": m1_mejor["dropout_bloques"]}
    run_m3 = pr.entrenar("M3", hp_m3, SEED, tipo_particion="grupos")
    fila_m3 = _registrar(filas, "M3", hp_m3, run_m3)
    print(f"  M3 f1_macro_val={fila_m3['f1_macro_val']:.4f} ({fila_m3['tiempo_min']:.1f} min)")

    aumento_m4 = "geometrico" if fila_m2["f1_macro_val"] >= fila_m3["f1_macro_val"] else "cutmix"
    razon_m4 = (f"M2 (geometrico) f1_macro_val={fila_m2['f1_macro_val']:.4f} vs. "
                f"M3 (cutmix) f1_macro_val={fila_m3['f1_macro_val']:.4f}")
    print(f"\n  → Aumento elegido para M4: {aumento_m4} ({razon_m4})")

    # ── M4: rejilla EfficientNet ─────────────────────────────────────────
    print("\n--- M4: rejilla lr_ajuste x capas_descongeladas (4 entrenamientos) ---")
    m4_filas = []
    for lr_ajuste in (1e-4, 1e-5):
        for capas in (20, 50):
            hp = {"arquitectura": "efficientnet", "aumento": aumento_m4,
                  "lr_ajuste": lr_ajuste, "capas_descongeladas": capas}
            run_dir = pr.entrenar("M4", hp, SEED, tipo_particion="grupos")
            fila = _registrar(filas, "M4", hp, run_dir)
            m4_filas.append(fila)
            print(f"  lr_ajuste={lr_ajuste} capas_descongeladas={capas} -> "
                  f"f1_macro_val={fila['f1_macro_val']:.4f} ({fila['tiempo_min']:.1f} min)")

    m4_mejor = _elegir_mejor(m4_filas, criterio_desempate_key="lr_ajuste")
    print(f"  → M4 elegido: lr_ajuste={m4_mejor['lr_ajuste']} "
          f"capas_descongeladas={m4_mejor['capas_descongeladas']} "
          f"(f1_macro_val={m4_mejor['f1_macro_val']:.4f})")

    # ── M5: EfficientNet segmentado, hp de M4 ───────────────────────────
    print("\n--- M5: EfficientNet + segmentación (hp de M4) ---")
    hp_m5 = {"arquitectura": "efficientnet", "aumento": aumento_m4,
              "lr_ajuste": m4_mejor["lr_ajuste"], "capas_descongeladas": m4_mejor["capas_descongeladas"]}
    run_m5 = pr.entrenar("M5", hp_m5, SEED, tipo_particion="grupos", segmentado=True)
    fila_m5 = _registrar(filas, "M5", hp_m5, run_m5)
    print(f"  M5 f1_macro_val={fila_m5['f1_macro_val']:.4f} ({fila_m5['tiempo_min']:.1f} min)")

    # ── Guardar CSV + config final ──────────────────────────────────────
    df = pd.DataFrame(filas)
    df.to_csv(EXPERIMENTOS_DIR / "busqueda_hiperparametros.csv", index=False)
    print(f"\n→ {EXPERIMENTOS_DIR / 'busqueda_hiperparametros.csv'}")

    config_final = {
        "M1": {"arquitectura": "cnn", "aumento": None,
               "lr": m1_mejor["lr"], "dropout_bloques": m1_mejor["dropout_bloques"]},
        "M2": hp_m2,
        "M3": hp_m3,
        "aumento_m4_elegido": aumento_m4,
        "aumento_m4_razon": razon_m4,
        "M4": {"arquitectura": "efficientnet", "aumento": aumento_m4,
               "lr_ajuste": m4_mejor["lr_ajuste"], "capas_descongeladas": m4_mejor["capas_descongeladas"]},
        "M5": hp_m5,
        "semilla_busqueda": SEED,
        "tipo_particion": "grupos",
    }
    with open(EXPERIMENTOS_DIR / "hiperparametros_finales.json", "w", encoding="utf-8") as f:
        json.dump(config_final, f, indent=2, ensure_ascii=False)
    print(f"→ {EXPERIMENTOS_DIR / 'hiperparametros_finales.json'}")

    # ── Curvas de aprendizaje, configuración elegida, semilla 0 ─────────
    print("\n--- Curvas de aprendizaje (curvas_M1..M5.png) ---")
    run_dirs = {"M1": m1_mejor["run_dir"], "M2": str(run_m2), "M3": str(run_m3),
                "M4": m4_mejor["run_dir"], "M5": str(run_m5)}
    from pathlib import Path
    for escenario, run_dir in run_dirs.items():
        plot_curvas(Path(run_dir), escenario, f"curvas_{escenario}.png")

    print("\n✓ Tarea 3 completada.")


if __name__ == "__main__":
    main()
