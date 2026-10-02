"""
Funciones compartidas de Grad-CAM, usadas tanto por 07_gradcam.py (modo script,
genera la figura comparativa para el TFM) como por la app web (inferencia
interactiva sobre una imagen subida por el usuario).
"""
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from PIL import Image


def find_last_conv_layer_name(model):
    """
    Busca el nombre de la última capa convolucional del modelo (o del último
    submodelo que contenga una, p.ej. la base de EfficientNet en transfer
    learning). Devuelve None si no encuentra ninguna.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, keras.layers.Conv2D):
                    return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Genera el mapa de calor Grad-CAM (valores en [0, 1]) para una imagen.

    Reproduce el forward pass manualmente, capa a capa, en vez de construir
    un keras.Model recortado vía `layer.output` (el patrón "clásico" de
    Grad-CAM). En Keras 3, reconstruir un modelo desde `model.inputs` hasta
    la salida de un submodelo anidado (p.ej. la base de EfficientNet dentro
    del modelo de transfer learning) falla con
    "Output with path `0` is not connected to `inputs`" porque el trazado
    de grafo no sigue correctamente las conexiones internas del submodelo
    anidado. Reejecutar cada capa de nivel superior directamente (en modo
    eager, dentro de un GradientTape) evita ese problema por completo y
    funciona igual para arquitecturas simples (CNN base/aug) y anidadas
    (transfer learning), siempre que el modelo sea una cadena lineal de
    capas de nivel superior — que es el caso de las 3 arquitecturas de
    este proyecto.
    """
    x = tf.convert_to_tensor(img_array)
    conv_output = None

    with tf.GradientTape() as tape:
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)
        predictions = x
        if conv_output is None:
            raise ValueError(
                f"No se pudo localizar la capa '{last_conv_layer_name}' "
                "durante el forward pass manual de Grad-CAM."
            )
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def _forward_hasta_capa(img_array, model, last_conv_layer_name):
    """Reproduce el forward pass manual (ver docstring de make_gradcam_heatmap,
    necesario por los submodelos anidados de EfficientNet en Keras 3) hasta
    la capa objetivo, sin gradientes — usado por Score-CAM. Devuelve
    (conv_output, predictions)."""
    x = tf.convert_to_tensor(img_array)
    conv_output = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        x = layer(x, training=False)
        if layer.name == last_conv_layer_name:
            conv_output = x
    if conv_output is None:
        raise ValueError(f"No se pudo localizar la capa '{last_conv_layer_name}'.")
    return conv_output, x


def make_gradcampp_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM++ (Chattopadhyay et al., 2018). Fórmula (Tarea 6, correcciones de
    la tutora): alpha = g^2 / (2*g^2 + SUM_espacial(A)*g^3) por canal, con el
    denominador protegido contra cero; peso de canal = SUM(alpha * ReLU(g));
    mapa = ReLU(SUM(peso * A)). Reutiliza el mismo forward pass manual que
    make_gradcam_heatmap (necesario para los submodelos anidados de
    EfficientNet en Keras 3).
    """
    x = tf.convert_to_tensor(img_array)
    conv_output = None
    with tf.GradientTape() as tape:
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if layer.name == last_conv_layer_name:
                conv_output = x
                tape.watch(conv_output)
        predictions = x
        if conv_output is None:
            raise ValueError(f"No se pudo localizar la capa '{last_conv_layer_name}'.")
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    A = conv_output[0]          # (H,W,C)
    g = grads[0]                # (H,W,C)
    g2 = tf.square(g)
    g3 = g2 * g
    sum_A = tf.reduce_sum(A, axis=(0, 1))  # (C,) — suma espacial por canal
    denom = 2.0 * g2 + sum_A[tf.newaxis, tf.newaxis, :] * g3
    denom = tf.where(tf.abs(denom) < 1e-8, tf.ones_like(denom) * 1e-8, denom)
    alpha = g2 / denom

    pesos_canal = tf.reduce_sum(alpha * tf.nn.relu(g), axis=(0, 1))  # (C,)
    heatmap = tf.nn.relu(tf.reduce_sum(pesos_canal[tf.newaxis, tf.newaxis, :] * A, axis=-1))
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def _normalizar01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def make_scorecam_heatmap(img_array, model, last_conv_layer_name, pred_index=None,
                           top_k=256, batch_size=32):
    """
    Score-CAM (Wang et al., 2020). Para cada uno de los `top_k` canales con
    mayor activación media (recorte explícito para acotar el coste en CPU,
    Tarea 6 de las correcciones de la tutora — evaluar los ~1280 canales de
    EfficientNetB0 sería demasiado lento sin GPU): se sobremuestrea el canal
    a la resolución de entrada, se normaliza a [0,1], se multiplica por la
    imagen original, y se mide la probabilidad de la clase objetivo en esa
    imagen enmascarada. Los pesos finales son el softmax de esas
    probabilidades.
    """
    conv_output, predictions = _forward_hasta_capa(img_array, model, last_conv_layer_name)
    if pred_index is None:
        pred_index = int(tf.argmax(predictions[0]))

    A = conv_output[0].numpy()  # (h, w, C)
    H, W = img_array.shape[1], img_array.shape[2]
    img0 = img_array[0]  # (H, W, 3)

    mean_activation = A.mean(axis=(0, 1))
    C = A.shape[-1]
    k = min(top_k, C)
    top_idx = np.argsort(mean_activation)[::-1][:k]

    mapas_resized = []
    for idx in top_idx:
        canal = _normalizar01(A[:, :, idx])
        resized = np.array(
            Image.fromarray(np.uint8(255 * canal)).resize((W, H), Image.LANCZOS)
        ) / 255.0
        mapas_resized.append(resized)
    mapas_resized = np.stack(mapas_resized, axis=0)  # (k, H, W)

    enmascaradas = img0[np.newaxis, ...] * mapas_resized[..., np.newaxis]  # (k,H,W,3)
    probas = model.predict(enmascaradas, batch_size=batch_size, verbose=0)  # (k,3)
    scores = probas[:, pred_index]
    pesos = tf.nn.softmax(scores).numpy()  # (k,)

    heatmap = np.tensordot(pesos, mapas_resized, axes=(0, 0))  # (H,W)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.astype(np.float32), pred_index


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Superpone el heatmap (0-1, HxW) sobre la imagen original (0-1, HxWx3)."""
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(
            (img.shape[1], img.shape[0]), Image.LANCZOS
        )
    ) / 255.0

    colormap = cm.jet(heatmap_resized)[:, :, :3]
    superimposed = img * (1 - alpha) + colormap * alpha
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed


def gradcam_overlay_for_image(model, img_float01, pred_index=None):
    """
    Atajo de alto nivel: dada una imagen (H, W, 3) en [0,1] y un modelo Keras,
    calcula la predicción, el heatmap Grad-CAM y la superposición.
    Devuelve (overlay_rgb01, pred_index, heatmap) o (None, pred_index, None)
    si el modelo no tiene una capa convolucional identificable.
    """
    last_conv_name = find_last_conv_layer_name(model)
    img_array = np.expand_dims(img_float01, axis=0)

    if last_conv_name is None:
        preds = model.predict(img_array, verbose=0)
        return None, int(np.argmax(preds[0])), None

    heatmap, used_pred_index = make_gradcam_heatmap(
        img_array, model, last_conv_name, pred_index=pred_index
    )
    overlay = overlay_heatmap(img_float01, heatmap)
    return overlay, used_pred_index, heatmap
