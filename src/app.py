"""
App web (Streamlit) del TFM — Detección y Clasificación de Cáncer de Pulmón.

Permite:
  1) Lanzar el entrenamiento de cada modelo y observar el progreso en vivo
     (época actual, curvas de loss/accuracy, fase de entrenamiento).
  2) Subir una imagen de TC (o elegir una de ejemplo del dataset) y ver la
     clasificación de cada modelo entrenado, con mapa Grad-CAM cuando aplica.
  3) Consultar la tabla comparativa de métricas de los 4 modelos.

Ejecutar con:
    streamlit run src/app.py

Los scripts de entrenamiento (03/04/05/08) se lanzan como subprocesos
independientes; escriben su progreso en outputs/progress/<modelo>.json,
que esta app sondea periódicamente. Así el entrenamiento sigue corriendo
aunque se cierre o recargue la pestaña del navegador.
"""
import io
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    DATA_DIR, MODELS_DIR, FIGURES_DIR, GRADCAM_DIR,
    CLASS_LABELS, CLASS_NAMES, IMG_SIZE,
)
from progress_tracker import PROGRESS_DIR
import model_utils
import results_utils
import gradcam_utils
from training_control import (
    TRAINABLE, STATUS_BADGES, STALE_SECONDS,
    read_progress, effective_status, launch_training, kill_process,
)

SRC_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="TFM — Cáncer de Pulmón (TC)",
    page_icon="🫁",
    layout="wide",
)


# ───────────────────────────── Página: Inicio ─────────────────────────────

def page_home():
    st.title("🫁 Detección y Clasificación de Cáncer de Pulmón en TC")
    st.caption(
        "TFM — Máster en Inteligencia Artificial. CNNs, aumento de datos y "
        "transfer learning sobre el dataset IQ-OTH/NCCD."
    )

    st.warning(
        "**Herramienta de apoyo al diagnóstico (CADe/CADx), no un sustituto "
        "del criterio médico.** Los resultados de este sistema son "
        "orientativos y deben ser siempre confirmados por un radiólogo o "
        "especialista."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("¿Qué hace esta app?")
        st.markdown(
            "- **Entrenamiento**: lanza y monitorea en vivo el entrenamiento "
            "de los 4 modelos (época actual, curvas de pérdida/exactitud).\n"
            "- **Clasificar imagen**: sube una TC de tórax (o usa una de "
            "ejemplo) y observa la predicción de cada modelo entrenado, "
            "con mapa de activación Grad-CAM.\n"
            "- **Resultados**: tabla comparativa de métricas sobre el "
            "conjunto de prueba (exactitud, AUC-ROC, sensibilidad por clase)."
        )
    with col2:
        st.subheader("Dataset")
        st.markdown(
            f"**IQ-OTH/NCCD** — {sum(1 for _ in DATA_DIR.rglob('*.jpg'))} "
            "imágenes de TC de tórax en 3 clases: Benigno, Maligno, Normal."
        )
        st.markdown(
            "⚠️ **Limitación importante**: las imágenes exportadas no "
            "incluyen un identificador de paciente, por lo que la partición "
            "train/val/test se hizo a nivel de *imagen* y no de *paciente*. "
            "Como este dataset contiene múltiples cortes de TC por paciente, "
            "es probable que existan cortes muy similares entre conjuntos "
            "(*data leakage*). La línea base SVM alcanzó 100% de exactitud "
            "en test, un resultado atípicamente alto para un método clásico "
            "que es consistente con esta fuga de datos. Ver la pestaña "
            "**Resultados** para más detalle."
        )

    st.divider()
    st.subheader("Estado actual de los modelos")
    cols = st.columns(4)
    for col, entry in zip(cols, TRAINABLE):
        state = read_progress(entry["key"])
        status = effective_status(entry, state)
        icon, label = STATUS_BADGES[status]
        with col:
            st.metric(entry["name"], f"{icon} {label}")


# ─────────────────────────── Página: Entrenamiento ─────────────────────────

def page_training():
    st.title("Entrenamiento de modelos")
    st.caption(
        "No hay GPU disponible en este equipo: el entrenamiento corre sobre "
        "CPU y puede tardar. Los tiempos son estimaciones orientativas."
    )

    auto_refresh = st.checkbox("🔄 Auto-actualizar cada 3 s", value=True)

    any_running = False

    for entry in TRAINABLE:
        state = read_progress(entry["key"])
        status = effective_status(entry, state)
        icon, label = STATUS_BADGES[status]

        with st.container(border=True):
            top = st.columns([3, 1, 1])
            top[0].markdown(f"### {icon} {entry['name']}")
            top[0].caption(entry["note"])

            disabled = status == "running"
            btn_label = "Reentrenar" if status in ("completed", "failed", "stale") else "Entrenar"
            if top[1].button(btn_label, key=f"btn_{entry['key']}", disabled=disabled,
                              width='stretch'):
                launch_training(entry)
                st.success(f"Lanzado el entrenamiento de {entry['name']}.")
                time.sleep(1)
                st.rerun()

            if status == "running" and state and state.get("pid"):
                if top[2].button("Detener", key=f"stop_{entry['key']}",
                                  width='stretch'):
                    if kill_process(state["pid"]):
                        st.warning("Proceso detenido.")
                        time.sleep(1)
                        st.rerun()

            if status in ("running",) and state:
                any_running = True
                total = state.get("total_epochs") or 1
                current = state.get("current_epoch", 0)
                frac = min(max(current / total, 0.0), 1.0)
                phase = state.get("phase", "")
                st.progress(frac, text=f"Época {current}/{total}" + (f" — {phase}" if phase else ""))

                batch_logs = state.get("batch_logs")
                if batch_logs:
                    metric_cols = st.columns(len(batch_logs))
                    for c, (k, v) in zip(metric_cols, batch_logs.items()):
                        c.metric(k, f"{v:.4f}")

                history = state.get("history", [])
                if history:
                    df = pd.DataFrame(history).set_index("epoch")
                    chart_cols = [c for c in ["loss", "val_loss"] if c in df.columns]
                    if chart_cols:
                        st.line_chart(df[chart_cols])
                    chart_cols_acc = [c for c in ["accuracy", "val_accuracy"] if c in df.columns]
                    if chart_cols_acc:
                        st.line_chart(df[chart_cols_acc])

            elif status == "completed" and state:
                val_acc = state.get("final_val_accuracy")
                val_loss = state.get("final_val_loss")
                msg = "Entrenamiento completado."
                if val_acc is not None:
                    msg += f" Accuracy (val): **{val_acc:.4f}**"
                if val_loss is not None:
                    msg += f" · Loss (val): **{val_loss:.4f}**"
                st.info(msg)

            elif status == "failed" and state:
                st.error(f"El entrenamiento falló: {state.get('error', 'error desconocido')}")

            elif status == "stale":
                st.warning(
                    "El proceso dejó de reportar progreso hace más de "
                    f"{STALE_SECONDS // 60} min. Puede haberse detenido o "
                    "fallado sin capturar el error. Revisa el log o vuelve a lanzarlo."
                )

            log_path = PROGRESS_DIR / f"{entry['key']}.log"
            if log_path.exists():
                with st.expander("Ver log del proceso"):
                    try:
                        text = log_path.read_text(encoding="utf-8", errors="replace")
                        st.code(text[-4000:], language=None)
                    except Exception as e:
                        st.caption(f"No se pudo leer el log: {e}")

    if auto_refresh and any_running:
        time.sleep(3)
        st.rerun()


# ─────────────────────────── Página: Clasificar imagen ─────────────────────

@st.cache_resource(show_spinner="Cargando modelo…")
def load_keras_model(path_str, mtime):
    from tensorflow import keras
    return keras.models.load_model(path_str)


@st.cache_resource(show_spinner="Cargando modelo SVM…")
def load_svm_bundle(path_str, mtime):
    return joblib.load(path_str)


def page_classify():
    st.title("Clasificar una imagen de TC")

    models = model_utils.available_models()
    available = [m for m in models if m["exists"]]
    if not available:
        st.info("Todavía no hay ningún modelo entrenado. Ve a la pestaña "
                 "**Entrenamiento** para entrenar al menos uno.")
        return

    source = st.radio("Origen de la imagen", ["Imagen de ejemplo del dataset", "Subir mi propia imagen"],
                       horizontal=True)

    img = None
    true_label = None

    if source == "Subir mi propia imagen":
        uploaded = st.file_uploader("Imagen de TC (jpg/png)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        if uploaded is not None:
            img = Image.open(io.BytesIO(uploaded.read()))
    else:
        samples = model_utils.sample_images_by_class(n_per_class=4)
        cols = st.columns(3)
        chosen_path = None
        for col, label in zip(cols, CLASS_LABELS):
            with col:
                st.markdown(f"**{label}**")
                for p in samples.get(label, []):
                    if st.button(p.name, key=f"sample_{p.name}", width='stretch'):
                        st.session_state["chosen_sample"] = str(p)
                        st.session_state["chosen_label"] = label
        if "chosen_sample" in st.session_state:
            chosen_path = Path(st.session_state["chosen_sample"])
            true_label = st.session_state.get("chosen_label")
            img = Image.open(chosen_path)

    if img is None:
        st.caption("Elige una imagen de ejemplo o sube tu propia TC para continuar.")
        return

    model_names = [m["name"] for m in available]
    chosen_name = st.selectbox("Modelo a usar", model_names)
    chosen = next(m for m in available if m["name"] == chosen_name)

    col_img, col_result = st.columns([1, 1])
    with col_img:
        st.image(img, caption="Imagen original" + (f" — clase real: {true_label}" if true_label else ""),
                  width='stretch')

    img_array01 = model_utils.preprocess_pil_image(img)

    with col_result:
        mtime = chosen["path"].stat().st_mtime
        if chosen["type"] == "keras":
            model = load_keras_model(str(chosen["path"]), mtime)
            proba = model_utils.predict_keras(model, img_array01)
        else:
            bundle = load_svm_bundle(str(chosen["path"]), mtime)
            proba = model_utils.predict_svm(bundle, img_array01)

        pred_idx = int(np.argmax(proba))
        pred_label = CLASS_LABELS[pred_idx]
        confidence = float(proba[pred_idx])

        st.markdown(f"### Predicción: **{pred_label}**")
        st.caption(f"Confianza: {confidence:.1%}")

        df_proba = pd.DataFrame({"Clase": CLASS_LABELS, "Probabilidad": proba})
        st.bar_chart(df_proba.set_index("Clase"))

        if true_label is not None:
            if pred_label == true_label:
                st.success("Coincide con la clase real de la imagen de ejemplo.")
            else:
                st.error(f"No coincide con la clase real ({true_label}).")

    if chosen["type"] == "keras":
        st.divider()
        st.subheader("Explicabilidad — Grad-CAM")
        st.caption(
            "Regiones de la imagen que más influyeron en la predicción del modelo. "
            "Colores cálidos (rojo/amarillo) indican mayor relevancia."
        )
        try:
            overlay, _, _ = gradcam_utils.gradcam_overlay_for_image(model, img_array01, pred_index=pred_idx)
            if overlay is not None:
                gc_cols = st.columns(2)
                gc_cols[0].image(img_array01, caption="Original", width='stretch')
                gc_cols[1].image(overlay, caption="Grad-CAM", width='stretch')
            else:
                st.caption("No se encontró una capa convolucional para generar Grad-CAM en este modelo.")
        except Exception as e:
            st.caption(f"No se pudo generar Grad-CAM: {e}")

    st.divider()
    st.caption(
        "Recordatorio: esta predicción es generada por un modelo de apoyo (CADx) "
        "entrenado sobre un dataset académico limitado, y no constituye un diagnóstico médico."
    )


# ─────────────────────────── Página: Resultados ─────────────────────────

def page_results():
    st.title("Resultados comparativos")

    metrics = results_utils.load_all_metrics()

    if not metrics:
        st.info("Todavía no hay métricas de evaluación. Ejecuta el "
                 "entrenamiento y luego `06_evaluate.py` y `08_baseline_ml.py`.")
        return

    rows = []
    for m in metrics:
        row = {
            "Modelo": m["name"],
            "Exactitud": m.get("accuracy"),
            "AUC-ROC": m.get("auc_roc"),
            "F1 Macro": m.get("f1_macro"),
        }
        for cls, val in m.get("sensitivity_per_class", {}).items():
            row[f"Sensib. {cls}"] = val
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Modelo")
    st.dataframe(df.style.format("{:.4f}"), width='stretch')

    if results_utils.has_leakage_signal(metrics):
        st.error("⚠️ " + results_utils.LEAKAGE_WARNING + " Ver la nota en la pestaña Inicio.")

    st.divider()
    st.subheader("Figuras")

    shown = 0
    cols = st.columns(2)
    for fig in results_utils.available_figures():
        if fig["exists"]:
            path = FIGURES_DIR / fig["filename"]
            cols[shown % 2].image(str(path), caption=fig["caption"], width='stretch')
            shown += 1

    gradcam_path = results_utils.gradcam_grid_path()
    if gradcam_path:
        st.divider()
        st.subheader("Grad-CAM — mejor modelo")
        st.image(str(gradcam_path), width='stretch')

    if shown == 0:
        st.caption("Aún no se generaron figuras de evaluación.")


# ───────────────────────────────── Main ────────────────────────────────────

PAGES = {
    "Inicio": page_home,
    "Entrenamiento": page_training,
    "Clasificar imagen": page_classify,
    "Resultados comparativos": page_results,
}

st.sidebar.title("🫁 TFM — Cáncer de Pulmón")
page = st.sidebar.radio("Navegación", list(PAGES.keys()))
PAGES[page]()
