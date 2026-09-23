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
