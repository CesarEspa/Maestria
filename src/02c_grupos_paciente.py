"""
02c - Reconstrucción aproximada de grupos por paciente y partición por grupos

El conjunto IQ-OTH/NCCD no trae un identificador de paciente por imagen, pero
los archivos están numerados en orden ("Malignant case (1).jpg", "(2)", ...) y
los cortes consecutivos de un mismo paciente son casi idénticos. Este script:

  1. Ordena cada clase por el número del archivo (orden natural, no alfabético).
  2. Calcula la correlación de Pearson entre cada imagen y la siguiente
     (escala de grises, 64x64, tras el mismo redimensionado a 224 del
     preprocesamiento).
  3. Corta la secuencia de cada clase en los K-1 enlaces más débiles, donde K es
     el número de casos documentado por los autores del conjunto
     (15 benignos, 40 malignos, 55 normales).
  4. Fusiona dentro de cada clase los grupos que tengan algún par de imágenes con
     correlación > UMBRAL_FUSION (cortes casi duplicados que el paso 3 hubiera
     separado). Fusionar de más es seguro frente a la fuga; separar de más no.
  5. Mide la fuga residual: porcentaje de imágenes de prueba que tienen un
     "gemelo" en entrenamiento (correlación > 0,95 y > 0,98), comparando la
     partición por imagen con la partición por grupos.

Salidas:
  outputs/splits/grupos_paciente.csv           archivo, clase, numero, grupo
  outputs/splits/particion_grupos_seed{S}.npz  índices train/val/test por semilla
  outputs/splits/particion_imagen_seed{S}.npz  índices de la partición por imagen (referencia)
  outputs/figures/fuga_gemelos_particion.png   histograma de similitud prueba→entrenamiento
  outputs/figures/montaje_grupos_paciente.png  muestra visual de grupos (verificación)
  outputs/figures/grupos_paciente_resumen.json  resumen numérico
"""
import json
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from config import DATA_DIR, SPLITS_DIR, FIGURES_DIR, CLASS_NAMES, CLASS_LABELS, IMG_SIZE

CASOS_DOCUMENTADOS = {"Bengin cases": 15, "Malignant cases": 40, "Normal cases": 55}
UMBRAL_FUSION = 0.97
SEMILLAS = [0, 1, 2, 3, 4]
PATRON = re.compile(r"\((\d+)\)")


def numero_archivo(path):
    m = PATRON.search(path.name)
    if m is None:
        raise ValueError(f"Nombre sin número entre paréntesis: {path.name}")
    return int(m.group(1))


def cargar_ordenado():
    """Devuelve rutas, clases, números y vectores de rasgo (64x64 normalizados)."""
    rutas, clases, numeros, rasgos = [], [], [], []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for c, nombre in enumerate(CLASS_NAMES):
        archivos = [p for p in (DATA_DIR / nombre).iterdir() if p.suffix.lower() in exts]
        archivos.sort(key=numero_archivo)
        for p in archivos:
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            g = np.asarray(img, dtype=np.float32).mean(-1) / 255.0
            g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).ravel()
            g = g - g.mean()
            g = g / (np.linalg.norm(g) + 1e-8)
            rutas.append(p.name); clases.append(c); numeros.append(numero_archivo(p)); rasgos.append(g)
    return rutas, np.array(clases), np.array(numeros), np.stack(rasgos)


def agrupar(clases, F):
    N = len(clases)
    C = F @ F.T
    grupo = np.zeros(N, dtype=int)
    g = 0
    consecutivas = {}
    for c, nombre in enumerate(CLASS_NAMES):
        idx = np.where(clases == c)[0]          # ya está en orden natural
        cs = np.array([C[idx[k], idx[k + 1]] for k in range(len(idx) - 1)])
        consecutivas[CLASS_LABELS[c]] = cs
        K = CASOS_DOCUMENTADOS[nombre]
        cortes = set(np.argsort(cs)[:K - 1])
        for k, i in enumerate(idx):
            grupo[i] = g
            if k in cortes:
                g += 1
        g += 1

    # Fusión de grupos con cortes casi duplicados (union-find)
    G = grupo.max() + 1
    padre = list(range(G))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    miembros = [np.where(grupo == a)[0] for a in range(G)]
    for a in range(G):
        for b in range(a + 1, G):
            if clases[miembros[a][0]] != clases[miembros[b][0]]:
                continue
            if C[np.ix_(miembros[a], miembros[b])].max() > UMBRAL_FUSION:
                padre[raiz(a)] = raiz(b)
    final = np.array([raiz(x) for x in grupo])
    _, final = np.unique(final, return_inverse=True)
    return final, C, consecutivas


def particion_grupos(clases, grupos, seed):
    """~70/15/15 por grupos y estratificada: 1/7 para prueba, 1/6 del resto para validación."""
    N = len(clases)
    sg = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    resto, test = next(sg.split(np.zeros(N), clases, grupos))
    sg2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(sg2.split(np.zeros(len(resto)), clases[resto], grupos[resto]))
    return resto[tr_rel], resto[va_rel], test


def particion_imagen(clases, seed):
    idx = np.arange(len(clases))
    resto, test = train_test_split(idx, test_size=0.15, random_state=seed, stratify=clases)
    tr, va = train_test_split(resto, test_size=0.15 / 0.85, random_state=seed, stratify=clases[resto])
    return tr, va, test


def gemelos(C, tr, te):
    nn = C[np.ix_(te, tr)].max(1)
    return nn, float((nn > 0.95).mean()), float((nn > 0.98).mean()), float(np.median(nn))


def main():
    print("Cargando y ordenando imágenes por número de archivo...")
    rutas, clases, numeros, F = cargar_ordenado()
    print(f"  {len(rutas)} imágenes: {np.bincount(clases).tolist()}")

    grupos, C, consecutivas = agrupar(clases, F)
    por_clase = {CLASS_LABELS[c]: int(len(np.unique(grupos[clases == c]))) for c in range(3)}
    tam = np.bincount(grupos)
    print(f"  Grupos: {grupos.max() + 1} → {por_clase}; tamaño mín/mediana/máx = "
          f"{tam.min()}/{int(np.median(tam))}/{tam.max()}")

    pd.DataFrame({"archivo": rutas, "clase": [CLASS_LABELS[c] for c in clases],
                  "numero": numeros, "grupo": grupos}).to_csv(SPLITS_DIR / "grupos_paciente.csv", index=False)

    resumen = {"n_imagenes": len(rutas), "grupos_total": int(grupos.max() + 1),
               "grupos_por_clase": por_clase, "umbral_fusion": UMBRAL_FUSION,
               "casos_documentados": CASOS_DOCUMENTADOS,
               "correlacion_consecutiva_mediana": {k: float(np.median(v)) for k, v in consecutivas.items()},
               "por_semilla": {}}

    nn_img_all, nn_grp_all = [], []
    for s in SEMILLAS:
        tr, va, te = particion_grupos(clases, grupos, s)
        assert not (set(grupos[tr]) & set(grupos[te])) and not (set(grupos[va]) & set(grupos[te]))
        np.savez(SPLITS_DIR / f"particion_grupos_seed{s}.npz", train=tr, val=va, test=te,
                 archivos=np.array(rutas), clases=clases)
        tri, vai, tei = particion_imagen(clases, s)
        np.savez(SPLITS_DIR / f"particion_imagen_seed{s}.npz", train=tri, val=vai, test=tei,
                 archivos=np.array(rutas), clases=clases)
        nn_g, g95, g98, gmed = gemelos(C, tr, te)
        nn_i, i95, i98, imed = gemelos(C, tri, tei)
        nn_img_all.append(nn_i); nn_grp_all.append(nn_g)
        resumen["por_semilla"][s] = {
            "n_train_val_test": [len(tr), len(va), len(te)],
            "test_por_clase": np.bincount(clases[te], minlength=3).tolist(),
            "imagen": {"gemelo_095": i95, "gemelo_098": i98, "mediana": imed},
            "grupos": {"gemelo_095": g95, "gemelo_098": g98, "mediana": gmed},
        }
        print(f"  Semilla {s}: {len(tr)}/{len(va)}/{len(te)} | prueba por clase "
              f"{np.bincount(clases[te], minlength=3).tolist()} | gemelos>0,95 por imagen "
              f"{i95:.1%} vs por grupos {g95:.1%}")

    with open(FIGURES_DIR / "grupos_paciente_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # Figura: similitud de cada imagen de prueba con su vecina más parecida de entrenamiento
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0.4, 1.0, 61)
    ax.hist(np.concatenate(nn_img_all), bins=bins, alpha=0.65, label="Partición por imagen")
    ax.hist(np.concatenate(nn_grp_all), bins=bins, alpha=0.65, label="Partición por grupos de paciente")
    ax.axvline(0.95, color="k", ls="--", lw=1)
    ax.set_xlabel("Correlación con la imagen de entrenamiento más parecida")
    ax.set_ylabel("Imágenes de prueba")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fuga_gemelos_particion.png", dpi=300)
    plt.close(fig)

    # Montaje de verificación visual: 2 grupos por clase, hasta 6 cortes cada uno
    rng = np.random.default_rng(0)
    filas = []
    for c in range(3):
        gs = np.unique(grupos[clases == c])
        filas += [(c, g) for g in rng.choice(gs, size=min(2, len(gs)), replace=False)]
    fig, axes = plt.subplots(len(filas), 6, figsize=(12, 2.1 * len(filas)))
    for r, (c, g) in enumerate(filas):
        idx = np.where(grupos == g)[0][:6]
        for k in range(6):
            ax = axes[r, k]; ax.axis("off")
            if k < len(idx):
                ax.imshow(F[idx[k]].reshape(64, 64), cmap="gray")
                if k == 0:
                    ax.set_title(f"{CLASS_LABELS[c]} · grupo {g}", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "montaje_grupos_paciente.png", dpi=200)
    plt.close(fig)
    print("✓ Grupos y particiones guardados.")


if __name__ == "__main__":
    main()
