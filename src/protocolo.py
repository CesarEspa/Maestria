"""
protocolo.py — Infraestructura común del protocolo corregido (correcciones
de la tutora, ver PROMPT_CORRECCIONES_TUTORA.md): partición por grupos de
paciente, parada temprana por F1 macro de validación, aumento de datos
unificado entre arquitecturas, y entrenamiento reanudable con registro en
disco. Los scripts de la Tarea 3 en adelante importan de aquí.

Numeración de secciones = numeración del prompt de la tutora (Tarea 2).
"""
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from skimage.measure import label as sk_label
from skimage.segmentation import clear_border

from config import (
    PROJECT_ROOT, DATA_DIR, SPLITS_DIR, FIGURES_DIR,
    CLASS_NAMES, CLASS_LABELS, NUM_CLASSES, IMG_SIZE, BATCH_SIZE,
)

EXPERIMENTOS_DIR = PROJECT_ROOT / "outputs" / "experimentos"
RUNS_DIR = EXPERIMENTOS_DIR / "runs"
for _d in (EXPERIMENTOS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABEL_TO_FOLDER = dict(zip(CLASS_LABELS, CLASS_NAMES))
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}

# Techos de épocas (2.5). Sobreescribibles por variable de entorno para
# pruebas rápidas de la infraestructura, igual que TFM_EPOCHS_OVERRIDE en
# 03/04/04b/05_train_*.py — no se usa en las ejecuciones reales del protocolo.
# MAX_EPOCHS_CNN bajado de 80 a 60 (incidencia "schedule de la tasa de
# aprendizaje", ver registro.md): con el nuevo EarlyStopping
# start_from_epoch=15 y sin ReduceLROnPlateau, 60 es el techo pedido.
MAX_EPOCHS_CNN = int(os.environ.get("TFM_MAX_EPOCHS_CNN", 60))
MAX_EPOCHS_FASE1 = int(os.environ.get("TFM_MAX_EPOCHS_FASE1", 20))
MAX_EPOCHS_FASE2 = int(os.environ.get("TFM_MAX_EPOCHS_FASE2", 60))


# ════════════════════════════════ 2.1 Datos ════════════════════════════════

def _grupos_df():
    path = SPLITS_DIR / "grupos_paciente.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero src/02c_grupos_paciente.py (Tarea 1)."
        )
    return pd.read_csv(path)


_imagenes_cache = None


def cargar_imagenes():
    """
    Todas las imágenes, en el MISMO ORDEN que grupos_paciente.csv, 224x224
    (LANCZOS), devueltas como float32 en [0,1]. La caché en disco se guarda
    en uint8 (outputs/splits/imagenes_224.npz) para pesar ~4x menos; la
    conversión a float ocurre solo en memoria, tras cargar la caché.
    """
    global _imagenes_cache
    if _imagenes_cache is not None:
        return _imagenes_cache

    cache_path = SPLITS_DIR / "imagenes_224.npz"
    df = _grupos_df()

    if cache_path.exists():
        X_u8 = np.load(cache_path)["X"]
        if len(X_u8) != len(df):
            raise ValueError(
                f"{cache_path} tiene {len(X_u8)} imágenes pero grupos_paciente.csv "
                f"tiene {len(df)}. Borra la caché y vuelve a generarla."
            )
    else:
        from PIL import Image
        print(f"Cargando y redimensionando {len(df)} imágenes (224x224, LANCZOS)...")
        X_u8 = np.empty((len(df), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for i, row in enumerate(df.itertuples()):
            p = DATA_DIR / LABEL_TO_FOLDER[row.clase] / row.archivo
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            X_u8[i] = np.asarray(img, dtype=np.uint8)
            if (i + 1) % 200 == 0 or (i + 1) == len(df):
                print(f"  {i + 1}/{len(df)}")
        np.savez_compressed(cache_path, X=X_u8)
        print(f"  → Caché guardada: {cache_path}")

    _imagenes_cache = X_u8.astype(np.float32) / 255.0
    return _imagenes_cache


_segmentadas_cache = None


def cargar_segmentadas():
    """
    Aplica segment_lungs() de 02b_segmentation.py a TODAS las imágenes, una
    sola vez. Devuelve (X_seg float32 [0,1], masks bool). Caché en disco en
    outputs/splits/imagenes_224_segmentadas.npz (imagen uint8 + máscara).
    """
    global _segmentadas_cache
    if _segmentadas_cache is not None:
        return _segmentadas_cache

    cache_path = SPLITS_DIR / "imagenes_224_segmentadas.npz"
    X = cargar_imagenes()

    if cache_path.exists():
        data = np.load(cache_path)
        X_seg_u8, masks = data["X_seg"], data["masks"]
        if len(X_seg_u8) != len(X):
            raise ValueError(f"{cache_path} desincronizada con imagenes_224.npz.")
    else:
        seg_mod = importlib.import_module("02b_segmentation")
        n = len(X)
        print(f"Segmentando {n} imágenes (02b_segmentation.segment_lungs)...")
        X_seg_u8 = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        masks = np.empty((n, IMG_SIZE, IMG_SIZE), dtype=bool)
        for i in range(n):
            mask, seg = seg_mod.segment_lungs(X[i])
            X_seg_u8[i] = (np.clip(seg, 0, 1) * 255).astype(np.uint8)
            masks[i] = mask
            if (i + 1) % 100 == 0 or (i + 1) == n:
                print(f"  {i + 1}/{n}")
        np.savez_compressed(cache_path, X_seg=X_seg_u8, masks=masks)
        print(f"  → Caché guardada: {cache_path}")

    _segmentadas_cache = (X_seg_u8.astype(np.float32) / 255.0, masks)
    return _segmentadas_cache


def cargar_X_y(segmentado=False):
    """Atajo: (X, y) listos para indexar con una partición. y son índices
    0/1/2 en el orden de CLASS_LABELS, igual que en el resto del proyecto."""
    X = cargar_segmentadas()[0] if segmentado else cargar_imagenes()
    df = _grupos_df()
    y = df["clase"].map(LABEL_TO_IDX).to_numpy()
    return X, y


def particion(seed, tipo="grupos"):
    """
    Lee outputs/splits/particion_{tipo}_seed{S}.npz (generado por
    02c_grupos_paciente.py). tipo: "grupos" (protocolo nuevo) o "imagen"
    (referencia, para cuantificar la inflación por fuga). Devuelve
    (train_idx, val_idx, test_idx).
    """
    path = SPLITS_DIR / f"particion_{tipo}_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta 02c_grupos_paciente.py primero.")
    data = np.load(path)
    return data["train"], data["val"], data["test"]


# ═══════════════════ 2.2 Control de calidad de las máscaras ════════════════

def control_calidad_mascaras():
    """
    Para cada imagen: % de área de la máscara, nº de componentes conexos, y
    si toca el borde. Marca sospechosas (área<5% o >60%, o >2 componentes).
    Guarda cifras en outputs/experimentos/calidad_mascaras.json y una figura
    de 30 máscaras al azar (10 por clase) superpuestas en rojo semi-
    transparente sobre la imagen ORIGINAL (para poder juzgar visualmente si
    la máscara respeta la anatomía real).
    """
    X_orig = cargar_imagenes()
    _, masks = cargar_segmentadas()
    df = _grupos_df()
    n = len(masks)

    areas = masks.reshape(n, -1).mean(axis=1)
    n_componentes = np.zeros(n, dtype=int)
    toca_borde = np.zeros(n, dtype=bool)
    for i in range(n):
        n_componentes[i] = int(sk_label(masks[i]).max())
        cleared = clear_border(masks[i])
        toca_borde[i] = not np.array_equal(cleared, masks[i])

    sospechosa = (areas < 0.05) | (areas > 0.60) | (n_componentes > 2)

    resumen = {
        "total": int(n),
        "sospechosas_total": int(sospechosa.sum()),
        "toca_borde_total": int(toca_borde.sum()),
        "umbral_area_min": 0.05,
        "umbral_area_max": 0.60,
        "umbral_componentes": 2,
        "por_clase": {},
    }
    for lbl in CLASS_LABELS:
        m = (df["clase"] == lbl).to_numpy()
        resumen["por_clase"][lbl] = {
            "n": int(m.sum()),
            "sospechosas": int((sospechosa & m).sum()),
            "toca_borde": int((toca_borde & m).sum()),
            "area_media": float(areas[m].mean()),
            "area_mediana": float(np.median(areas[m])),
            "componentes_media": float(n_componentes[m].mean()),
        }

    with open(EXPERIMENTOS_DIR / "calidad_mascaras.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"  → {EXPERIMENTOS_DIR / 'calidad_mascaras.json'}: "
          f"{sospechosa.sum()}/{n} sospechosas, {toca_borde.sum()}/{n} tocan el borde")

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(10, 3, figsize=(9, 30))
    for col, lbl in enumerate(CLASS_LABELS):
        idx_clase = np.where((df["clase"] == lbl).to_numpy())[0]
        elegidos = rng.choice(idx_clase, size=min(10, len(idx_clase)), replace=False)
        for row in range(10):
            ax = axes[row, col]
            ax.axis("off")
            if row >= len(elegidos):
                continue
            idx = elegidos[row]
            ax.imshow(X_orig[idx])
            overlay = np.zeros((*masks[idx].shape, 4))
            overlay[masks[idx]] = [1, 0, 0, 0.35]
            ax.imshow(overlay)
            if row == 0:
                ax.set_title(lbl, fontsize=11)
    fig.suptitle("Control de calidad de las máscaras de segmentación (10 por clase, al azar)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "control_calidad_mascaras.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIGURES_DIR / 'control_calidad_mascaras.png'}")
    return resumen


# ═══════════════ 2.3 Aumento de datos (unificado para ambas familias) ══════

def build_augmentation_geometrico():
    """
    Aumento geométrico UNIFICADO: antes la transferencia usaba
    RandomRotation(0.1)/RandomZoom(0.1) mientras la CNN usaba 0.08/0.05 — un
    segundo factor que cambiaba a la vez entre escenarios (corrección #5 de
    la tutora). Ahora ambas familias usan exactamente esta misma capa.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.1),
    ], name="augmentation_geometrico")


def _dataset_entrenamiento(X_train, y_train_cat, aumento, class_weights_tensor, seed):
    """
    tf.data.Dataset de entrenamiento, con el aumento elegido aplicado SOLO
    al conjunto de entrenamiento. aumento: None | "geometrico" | "cutmix".
    Para "cutmix" el dataset rinde (x, y_mezclada, sample_weight) — ver nota
    de class_weight en entrenar(). Para None/"geometrico" rinde (x, y).
    """
    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    ds = ds.shuffle(len(X_train), seed=seed)
    ds = ds.batch(BATCH_SIZE)

    if aumento == "geometrico":
        aug_layer = build_augmentation_geometrico()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    elif aumento == "cutmix":
        cutmix_mod = importlib.import_module("04b_train_cnn_cutmix")
        ds = ds.map(lambda x, y: cutmix_mod.cutmix_batch(x, y, class_weights_tensor),
                    num_parallel_calls=tf.data.AUTOTUNE)
    elif aumento is not None:
        raise ValueError(f"aumento desconocido: {aumento!r}")

    return ds.prefetch(tf.data.AUTOTUNE)


# ══════════════ 2.4 Parada temprana por F1 macro de validación ═════════════
# Código dado literalmente por la tutora, sin modificar.

class F1MacroValidacion(keras.callbacks.Callback):
    """Debe ir PRIMERO en la lista de callbacks: añade val_f1_macro a logs
    para que EarlyStopping, ReduceLROnPlateau y ModelCheckpoint lo vean."""
    def __init__(self, X_val, y_val, batch_size=32):
        super().__init__()
        self.X_val, self.y_val, self.bs = X_val, y_val, batch_size

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.X_val, batch_size=self.bs, verbose=0).argmax(1)
        sens = recall_score(self.y_val, p, labels=[0, 1, 2], average=None, zero_division=0)
        if logs is not None:
            logs["val_f1_macro"] = f1_score(self.y_val, p, average="macro")
            logs["val_sens_benigno"], logs["val_sens_maligno"], logs["val_sens_normal"] = map(float, sens)


def _callbacks_f1_efficientnet(X_val, y_val, patience=15):
    """EfficientNet (M4/M5) — SIN CAMBIOS (no afectada por la incidencia de
    schedule de LR, ver registro.md 'Incidencia: schedule de la tasa de
    aprendizaje y criterio de selección')."""
    return [
        F1MacroValidacion(X_val, y_val, batch_size=BATCH_SIZE),
        keras.callbacks.EarlyStopping(monitor="val_f1_macro", mode="max", patience=patience,
                                       start_from_epoch=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_f1_macro", mode="max",
                                           factor=0.5, patience=5, min_lr=1e-6),
    ]


def _callbacks_f1_cnn(X_val, y_val):
    """CNN propia (M1/M2/M3) — CORREGIDO (incidencia 'schedule de la tasa de
    aprendizaje'): SIN ReduceLROnPlateau. Con val_f1_macro + patience=5, el
    LR caía a ~1e-6 antes de que la CNN saliera de la meseta inicial (loss
    de entrenamiento nunca bajaba de ln(3)≈1.0986) — confirmado comparando
    con curvas_cnn_base.png del protocolo preliminar, donde a LR constante
    1e-4 la CNN sale de la meseta hacia la época 11. EarlyStopping con
    start_from_epoch=15 (antes 5) le da más margen a esa meseta inicial."""
    return [
        F1MacroValidacion(X_val, y_val, batch_size=BATCH_SIZE),
        keras.callbacks.EarlyStopping(monitor="val_f1_macro", mode="max", patience=15,
                                       start_from_epoch=15, restore_best_weights=True),
    ]


# ════════════════════════════ 2.5 Constructores ═════════════════════════════

def cnn_propia(lr, dropout_bloques):
    """Arquitectura actual de 03_train_cnn_base.py (3 bloques, LayerNorm,
    label smoothing 0.1, Adam). dropout_bloques es ahora un parámetro; el
    dropout de la cabeza se mantiene fijo en 0.5."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for filtros in (32, 64, 128):
        x = layers.Conv2D(filtros, (3, 3), activation="relu", padding="same")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Conv2D(filtros, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_bloques)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def efficientnet(lr_ajuste, capas_descongeladas):
    """
    Arquitectura actual de 05_train_transfer.py. El aumento ya NO se
    construye dentro del grafo (antes sí) — ahora se aplica fuera, en el
    pipeline de datos, igual que para cnn_propia (ver 2.3, unificación).
    Devuelve (model, base_model) ya compilados para la FASE 1 (base
    congelada, lr fijo 1e-3). Llama a activar_fine_tuning() para la fase 2.
    """
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model, base_model


def activar_fine_tuning(model, base_model, lr_ajuste, capas_descongeladas):
    """Descongela las últimas `capas_descongeladas` capas de la base y
    recompila con lr_ajuste, para la fase 2."""
    base_model.trainable = True
    for layer in base_model.layers[:-capas_descongeladas]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_ajuste),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )


# ══════════════════════ 2.6 entrenar() — orquestador ═══════════════════════

def nombre_run(escenario, hp, seed):
    """Nombre único y determinista {escenario}_{config}_seed{S}."""
    partes = [escenario]
    for k in sorted(hp):
        if k == "arquitectura" or hp[k] is None:
            continue
        partes.append(f"{k}{hp[k]}")
    partes.append(f"seed{seed}")
    return "_".join(str(p) for p in partes)


def entrenar(escenario, hp, seed, tipo_particion="grupos", segmentado=False):
    """
    Entrena un escenario (M1-M5) con una configuración de hiperparámetros
    sobre una semilla y partición. NO predice sobre el conjunto de prueba
    (eso es exclusivo de la Tarea 5). Reanudable: si modelo.keras ya existe
    para este nombre, se salta.

    hp admite: arquitectura ("cnn"|"efficientnet"), aumento
    (None|"geometrico"|"cutmix"), y según arquitectura: lr/dropout_bloques
    (cnn) o lr_ajuste/capas_descongeladas (efficientnet).

    Guarda en outputs/experimentos/runs/{nombre}/: modelo.keras,
    historial.csv, pred_val.npz, tiempo.json, hp.json.
    """
    nombre = nombre_run(escenario, hp, seed)
    run_dir = RUNS_DIR / nombre
    run_dir.mkdir(parents=True, exist_ok=True)
    modelo_path = run_dir / "modelo.keras"

    if modelo_path.exists():
        print(f"  [saltado, ya existe] {nombre}")
        return run_dir

    keras.utils.set_random_seed(seed)

    X, y = cargar_X_y(segmentado=segmentado)
    tr_idx, va_idx, te_idx = particion(seed, tipo=tipo_particion)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    y_train_cat = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

    arquitectura = hp["arquitectura"]
    aumento = hp.get("aumento")
    # class_weight de Keras no admite etiquetas mezcladas (CutMix): en ese
    # caso el peso de clase ya va dentro del sample_weight del dataset (ver
    # 04b_train_cnn_cutmix.py).
    class_weight_arg = None if aumento == "cutmix" else class_weight_dict

    t0 = time.time()
    historiales = []  # lista de (etiqueta_fase, history)

    train_ds = _dataset_entrenamiento(X_train, y_train_cat, aumento, class_weights_tensor, seed)

    if arquitectura == "cnn":
        model = cnn_propia(hp["lr"], hp["dropout_bloques"])
        history = model.fit(
            train_ds, validation_data=(X_val, y_val_cat), epochs=MAX_EPOCHS_CNN,
            class_weight=class_weight_arg, callbacks=_callbacks_f1_cnn(X_val, y_val),
            verbose=1,
        )
        historiales.append(("unica", history))

    elif arquitectura == "efficientnet":
        model, base_model = efficientnet(hp["lr_ajuste"], hp["capas_descongeladas"])
        history1 = model.fit(
            train_ds, validation_data=(X_val, y_val_cat), epochs=MAX_EPOCHS_FASE1,
            class_weight=class_weight_arg, callbacks=_callbacks_f1_efficientnet(X_val, y_val, patience=15),
            verbose=1,
        )
        historiales.append(("fase1", history1))

        activar_fine_tuning(model, base_model, hp["lr_ajuste"], hp["capas_descongeladas"])
        history2 = model.fit(
            train_ds, validation_data=(X_val, y_val_cat), epochs=MAX_EPOCHS_FASE2,
            class_weight=class_weight_arg, callbacks=_callbacks_f1_efficientnet(X_val, y_val, patience=15),
            verbose=1,
        )
        historiales.append(("fase2", history2))

    else:
        raise ValueError(f"arquitectura desconocida: {arquitectura!r}")

    tiempo_s = time.time() - t0

    model.save(modelo_path)

    filas = []
    for fase, h in historiales:
        n_ep = len(h.history["loss"])
        for e in range(n_ep):
            fila = {"fase": fase, "epoca": e + 1}
            for k, v in h.history.items():
                fila[k] = v[e]
            filas.append(fila)
    pd.DataFrame(filas).to_csv(run_dir / "historial.csv", index=False)

    proba_val = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    np.savez(run_dir / "pred_val.npz", proba=proba_val, y=y_val)

    with open(run_dir / "tiempo.json", "w", encoding="utf-8") as f:
        json.dump({"segundos": tiempo_s, "minutos": tiempo_s / 60}, f, indent=2)
    with open(run_dir / "hp.json", "w", encoding="utf-8") as f:
        json.dump({
            "escenario": escenario, "hp": hp, "seed": seed,
            "tipo_particion": tipo_particion, "segmentado": segmentado,
            "epocas_efectivas": [len(h.history["loss"]) for _, h in historiales],
        }, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {nombre} — {tiempo_s / 60:.1f} min "
          f"({'+'.join(str(len(h.history['loss'])) for _, h in historiales)} épocas)")
    return run_dir


# ═══════════════════════════ 2.7 Presupuesto de tiempo ══════════════════════

def benchmark_tiempo(n_epocas=2, seed=0):
    """
    Mide el tiempo de n_epocas épocas de la CNN propia y de EfficientNet
    (fase 1) en este equipo, sin aumento (cota inferior razonable), para
    estimar el presupuesto total de las Tareas 3 y 4 antes de lanzarlas.
    Guarda outputs/experimentos/benchmark_tiempo.json.
    """
    X, y = cargar_X_y(segmentado=False)
    tr, va, _ = particion(seed, tipo="grupos")
    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    resultados = {"n_epocas_medidas": n_epocas, "n_train": int(len(tr)), "n_val": int(len(va))}

    print("Benchmark: CNN propia...")
    keras.utils.set_random_seed(seed)
    m = cnn_propia(1e-4, 0.25)
    t0 = time.time()
    m.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
          epochs=n_epocas, batch_size=BATCH_SIZE, class_weight=class_weight_dict, verbose=1)
    resultados["cnn_seg_por_epoca"] = (time.time() - t0) / n_epocas

    print("Benchmark: EfficientNet (fase 1, base congelada)...")
    keras.utils.set_random_seed(seed)
    m2, _ = efficientnet(1e-4, 50)
    t0 = time.time()
    m2.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
           epochs=n_epocas, batch_size=BATCH_SIZE, class_weight=class_weight_dict, verbose=1)
    resultados["efficientnet_seg_por_epoca"] = (time.time() - t0) / n_epocas

    print(f"\nResultados del benchmark: {resultados}")
    with open(EXPERIMENTOS_DIR / "benchmark_tiempo.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
    return resultados


if __name__ == "__main__":
    benchmark_tiempo()
