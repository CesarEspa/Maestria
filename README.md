# TFM — Detección y Clasificación Automática de Cáncer de Pulmón en Imágenes de TC mediante CNN y Aumento de Datos

**Máster en Inteligencia Artificial — Università G. Marconi / AICAD Business School**
Autor: César Libardo España Salguero

> ⚠️ **Este sistema es una herramienta de apoyo al diagnóstico (CADe/CADx) con
> fines académicos.** No sustituye el criterio de un radiólogo ni de ningún
> profesional médico, y no debe usarse para tomar decisiones clínicas reales.
> Ver la sección [Limitaciones](#limitaciones-y-consideraciones-éticas) antes
> de interpretar cualquier resultado.

---

## Índice

1. [Objetivo del proyecto](#objetivo-del-proyecto)
2. [Dataset](#dataset)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Instalación](#instalación)
5. [Cómo ejecutar el pipeline](#cómo-ejecutar-el-pipeline)
6. [Apps web interactivas](#apps-web-interactivas)
7. [Metodología](#metodología)
8. [Resultados](#resultados)
9. [Hallazgos y análisis](#hallazgos-y-análisis)
10. [Limitaciones y consideraciones éticas](#limitaciones-y-consideraciones-éticas)
11. [Trabajo futuro](#trabajo-futuro)

---

## Objetivo del proyecto

Desarrollar y evaluar un sistema de clasificación automática de imágenes de
TC de tórax en tres categorías (**Normal**, **Benigno**, **Maligno**),
basado en redes neuronales convolucionales (CNN) y técnicas de aumento de
datos, orientado al apoyo del proceso de cribado en servicios de radiología.

**Objetivos específicos:**

- Preprocesar y analizar el dataset IQ-OTH/NCCD, aplicando normalización,
  partición estratificada (70/15/15) y *data augmentation* para mitigar el
  desbalance entre clases.
- Diseñar, entrenar y comparar tres arquitecturas: CNN base, CNN con
  augmentation, y transfer learning (EfficientNetB0).
- Evaluar con sensibilidad, especificidad, exactitud, AUC-ROC y matriz de
  confusión; contrastar con una línea base de ML clásico (SVM + HOG) y con
  la literatura.
- Generar mapas Grad-CAM sobre el mejor modelo y analizar su aplicabilidad
  en un flujo de trabajo de radiología.

## Dataset

**IQ-OTH/NCCD Lung Cancer Dataset** (Kaggle) — TC de tórax de pacientes
iraquíes, recopiladas por el Hospital Especializado de Oncología de Irak
(IOSH) y el Centro Nacional de Cáncer y Enfermedades del Tórax (NCCD).

| Clase | Nº imágenes | % |
|---|---:|---:|
| Benigno (*Bengin cases*) | 120 | 10.9% |
| Maligno (*Malignant cases*) | 561 | 51.1% |
| Normal (*Normal cases*) | 416 | 37.9% |
| **Total** | **1097** | 100% |

El dataset está claramente **desbalanceado** (la clase Benigno tiene menos
de la cuarta parte de imágenes que Maligno), lo cual se compensa
parcialmente con `class_weight="balanced"` durante el entrenamiento y se
documenta como factor relevante en el análisis de resultados.

Las imágenes tienen tamaños originales heterogéneos (ancho: 506–801 px, alto:
331–512 px) y se redimensionan a 224×224 px para todos los modelos.

> ⚠️ **Nota crítica sobre el dataset**: las imágenes exportadas en Kaggle
> **no incluyen un identificador de paciente por imagen**. El nombre de
> archivo (p. ej. `Malignant case (37).jpg`) es solo un índice secuencial.
> Como es habitual en datasets de TC, cada paciente aporta *varios cortes
> axiales* al conjunto, que suelen ser visualmente muy similares entre sí.
> Al no poder agrupar por paciente, la partición train/val/test de este
> proyecto se hizo **a nivel de imagen**, no a nivel de paciente — ver la
> discusión completa en [Limitaciones](#limitaciones-y-consideraciones-éticas).

## Estructura del proyecto

```
tfm-lung-cancer/
├── Datos/The IQ-OTHNCCD lung cancer dataset/
│   ├── Bengin cases/           # 120 imágenes (nombre de carpeta tal cual viene en el dataset)
│   ├── Malignant cases/        # 561 imágenes
│   └── Normal cases/           # 416 imágenes
├── src/
│   ├── config.py                # Configuración central (rutas, hiperparámetros, clases)
│   ├── 01_eda.py                 # Análisis exploratorio de datos
│   ├── 02_preprocessing.py       # Carga, resize 224×224, normalización y split 70/15/15
│   ├── 03_train_cnn_base.py      # CNN sencilla sin aumento de datos
│   ├── 04_train_cnn_aug.py       # Misma CNN con aumento de datos en tiempo real
│   ├── 05_train_transfer.py      # Transfer learning EfficientNetB0 (2 fases)
│   ├── 06_evaluate.py            # Evaluación comparativa de los 3 modelos Keras en test
│   ├── 07_gradcam.py             # Mapas Grad-CAM sobre el mejor modelo
│   ├── 08_baseline_ml.py         # Línea base clásica: HOG + SVM
│   ├── progress_tracker.py       # Callback de Keras que registra progreso para las apps web
│   ├── gradcam_utils.py          # Funciones Grad-CAM compartidas
│   ├── model_utils.py            # Preprocesamiento e inferencia compartidos
│   ├── results_utils.py          # Carga de métricas/figuras compartida
│   ├── training_control.py       # Lanzar/detener entrenamientos, estado — compartido
│   ├── app.py                    # App web #1 (Streamlit): entrenar, monitorear, clasificar
│   ├── api.py                    # App web #2: backend FastAPI (API REST)
│   └── frontend/                 # App web #2: frontend HTML/CSS/JS servido por api.py
│       ├── index.html
│       ├── style.css
│       └── app.js
├── outputs/
│   ├── figures/                  # Gráficos y JSON de métricas para el TFM
│   ├── models/                   # Modelos entrenados (.keras y .joblib)
│   ├── gradcam/                  # Imágenes Grad-CAM
│   ├── splits/                   # dataset_splits.npz (arrays preprocesados)
│   └── progress/                 # Estado de entrenamiento en vivo (JSON) + logs
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

**Notas para Windows:**

- Los scripts imprimen caracteres especiales (`→`, `✓`, `⚠`) que pueden
  fallar con `UnicodeEncodeError` en la consola por defecto (cp1252). Antes
  de ejecutar cualquier script, fija la codificación:
  ```powershell
  $env:PYTHONIOENCODING = "utf-8"
  ```
  (en bash/Git Bash: `export PYTHONIOENCODING=utf-8`)
- Instalar `streamlit` puede actualizar `protobuf` a una versión
  incompatible con TensorFlow (`protobuf>=5`, pero TensorFlow 2.17
  requiere `<5`). Si `import tensorflow` falla tras instalar Streamlit,
  ejecuta:
  ```bash
  pip install "protobuf>=3.20.3,<5.0.0dev"
  ```
  `requirements.txt` ya fija este rango para evitar el problema en una
  instalación limpia.
- Este proyecto se desarrolló y probó **sin GPU** (solo CPU), pese a que el
  equipo de desarrollo sí tiene una GPU NVIDIA física (RTX 3050). El motivo:
  desde TensorFlow 2.11, el paquete `pip install tensorflow` estándar **ya
  no incluye soporte nativo de GPU/CUDA en Windows** — `tf.config.list_physical_devices('GPU')`
  devuelve una lista vacía aunque `nvidia-smi` detecte la tarjeta
  correctamente. Para usar la GPU en Windows hoy hacen falta pasos
  adicionales: WSL2 con una instalación Linux de TensorFlow, o el plugin
  `tensorflow-directml-plugin` de Microsoft (typically requiere fijar una
  versión de TensorFlow más antigua, con el riesgo de romper código escrito
  para Keras 3). No se intentó ninguna de las dos por el riesgo de dejar el
  entorno en un estado peor sin garantía de que funcionara; el
  entrenamiento es funcional pero lento en CPU. Ver tiempos orientativos en
  la sección [Metodología](#metodología), y la nota correspondiente en
  [Trabajo futuro](#trabajo-futuro).

## Cómo ejecutar el pipeline

Los scripts se ejecutan en orden desde `src/`, y cada uno depende de los
artefactos generados por el anterior (splits en `outputs/splits/`, modelos en
`outputs/models/`):

```bash
cd src
export PYTHONIOENCODING=utf-8   # o $env:PYTHONIOENCODING="utf-8" en PowerShell

python 01_eda.py                # Análisis exploratorio → outputs/figures/
python 02_preprocessing.py      # Split 70/15/15 → outputs/splits/dataset_splits.npz
python 03_train_cnn_base.py     # ~40-60 min en CPU (hasta 80 épocas, paciencia 12)
python 04_train_cnn_aug.py      # ~50-80 min en CPU (hasta 80 épocas, paciencia 12)
python 05_train_transfer.py     # ~20-35 min en CPU (2 fases, hasta 15+50 épocas)
python 06_evaluate.py           # Evaluación comparativa en test
python 07_gradcam.py            # Mapas de activación del mejor modelo
python 08_baseline_ml.py        # Línea base SVM + HOG (~1-2 min)
```

Cada script guarda sus resultados (figuras, modelos, métricas) en `outputs/`.
Los scripts 03, 04, 05 y 08 también actualizan `outputs/progress/<modelo>.json`
en vivo, que consume la app web descrita a continuación.

## Apps web interactivas

El proyecto incluye **dos** interfaces web equivalentes en funcionalidad,
para no depender únicamente de la terminal. Ambas comparten la misma lógica
de negocio (`training_control.py`, `model_utils.py`, `gradcam_utils.py`,
`results_utils.py`, `progress_tracker.py`), así que se comportan igual y
muestran siempre los mismos números — solo cambia la interfaz.

### Opción 1 — Streamlit (`src/app.py`)

La más rápida de lanzar, pensada como herramienta interna:

```bash
cd src
streamlit run app.py
```

Se abre en `http://localhost:8501` con 4 secciones (Inicio, Entrenamiento,
Clasificar imagen, Resultados comparativos) — ver el detalle de cada una
más abajo, es el mismo para las dos apps.

### Opción 2 — Frontend propio + backend FastAPI (`src/api.py` + `src/frontend/`)

Una interfaz visual más cuidada (HTML/CSS/JS propio, sin dependencias de
build) sobre una **API REST** separada del frontend:

```bash
cd src
uvicorn api:app --reload --port 8000
```

Abre `http://localhost:8000` en el navegador (el backend sirve el frontend
directamente, todo en un único puerto — sin problemas de CORS). La API
queda disponible bajo `http://localhost:8000/api/...` (ver `/docs` para la
documentación interactiva autogenerada por FastAPI) y puede consumirse desde
cualquier otro cliente, no solo desde este frontend.

Esta opción se separó deliberadamente en frontend + backend (en vez de
todo-en-uno como hace Streamlit) para poder crecer más adelante sin
reescribir nada: añadir autenticación, guardar un historial de
clasificaciones en una base de datos, desplegar el backend en un servidor
real y servir el frontend aparte (p. ej. como SPA), etc. El backend actual
es deliberadamente ligero (sin base de datos, sin autenticación, pensado
para un único usuario en local) — es el punto de partida sobre el que
construir esas extensiones, no una versión final.

**Endpoints principales de la API:**

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/api/models` | Estado de los 4 modelos (idle/running/completed/failed) + progreso en vivo |
| POST | `/api/models/{key}/train` | Lanza el entrenamiento de un modelo |
| POST | `/api/models/{key}/stop` | Detiene un entrenamiento en curso |
| GET | `/api/models/{key}/log` | Log del proceso de entrenamiento |
| GET | `/api/samples` | Imágenes de ejemplo del dataset, por clase |
| POST | `/api/classify` | Clasifica una imagen subida (`multipart/form-data`) |
| POST | `/api/classify_sample` | Clasifica una imagen de ejemplo del dataset |
| GET | `/api/results` | Métricas comparativas de los 4 modelos + alerta de fuga de datos |
| GET | `/api/figures`, `/api/gradcam_grid` | Figuras de evaluación y grid Grad-CAM |

### Qué ofrecen ambas interfaces

1. **Inicio** — resumen del proyecto, estado de los 4 modelos y el aviso
   sobre la limitación de fuga de datos.
2. **Entrenamiento** — permite **lanzar el entrenamiento de cualquiera de
   los 4 modelos con un clic**, y observar en vivo:
   - la época actual y el total (barra de progreso),
   - la fase (para transfer learning: cabeza congelada / fine-tuning),
   - curvas de *loss*/*accuracy* de entrenamiento y validación, actualizadas
     cada pocos segundos,
   - el log completo del proceso, y un botón para detenerlo si es necesario.
   Cada entrenamiento se lanza como un **subproceso independiente**: sigue
   corriendo aunque cierres el navegador, y la app simplemente vuelve a leer
   su archivo de progreso al reabrirla.
3. **Clasificar imagen** — sube tu propia imagen de TC (o elige una de
   ejemplo del propio dataset) y dile a la app con qué modelo entrenado
   quieres clasificarla. Muestra la clase predicha, la confianza, la
   distribución de probabilidad por clase y, para los 3 modelos basados en
   CNN, el **mapa Grad-CAM** superpuesto sobre la imagen original.
4. **Resultados comparativos** — tabla con las métricas de los 4 modelos
   sobre el conjunto de prueba (exactitud, AUC-ROC, F1 macro, sensibilidad
   por clase), las figuras generadas por `06_evaluate.py`/`08_baseline_ml.py`,
   destacando el mejor modelo *fiable* y marcando con ⚠️ el que sea
   sospechoso de fuga de datos, además de una alerta automática si algún
   modelo alcanza ~100% de exactitud.

## Metodología

### Preprocesamiento (`02_preprocessing.py`)

- Carga de las 1097 imágenes, conversión a RGB, redimensionado a 224×224 px
  (interpolación LANCZOS) y normalización a `[0, 1]`.
- Partición estratificada 70% train (767) / 15% val (165) / 15% test (165),
  manteniendo la proporción de clases en cada subconjunto, con semilla fija
  (`SEED=42`) para reproducibilidad.
- Los arrays resultantes se guardan comprimidos en
  `outputs/splits/dataset_splits.npz` (~63 MB) para que los scripts de
  entrenamiento no tengan que releer y reprocesar las imágenes cada vez.

### Modelos

*(Arquitecturas tal como quedaron tras la segunda ronda de ajustes — ver
[Hallazgos](#hallazgos-y-análisis) para el porqué de cada cambio.)*

| Modelo | Arquitectura | Particularidad |
|---|---|---|
| **CNN Base** | 3 bloques Conv2D(32→64→128) + **LayerNorm** + MaxPooling + GAP + Dense(256) | Sin aumento de datos. `CategoricalCrossentropy(label_smoothing=0.1)` |
| **CNN + Augmentation** | Misma arquitectura | Augmentation reducido: flip horizontal, rotación ±0.08, zoom ±0.05, contraste (sin flip vertical ni traslación) |
| **Transfer Learning** | EfficientNetB0 preentrenada en ImageNet + cabeza personalizada | Fase 1: base congelada (hasta 15 épocas). Fase 2: fine-tuning de las **últimas 50 capas** (hasta 50 épocas) |
| **SVM (línea base)** | HOG (9 orientaciones, celdas 16×16) + SVM kernel RBF | Método clásico de ML, sin deep learning — sin cambios entre rondas |

Todos los modelos de Keras usan `class_weight="balanced"` para compensar el
desbalance de clases, `EarlyStopping` (paciencia 12) sobre `val_loss` y
`ReduceLROnPlateau`. Learning rate 1×10⁻⁴ (CNN base/aug) y 1×10⁻⁴ /
1×10⁻⁵ (transfer learning, fase 1/fase 2). El entrenamiento se ejecutó
**enteramente en CPU** (ver nota sobre GPU en
[Instalación](#instalación)).

## Resultados

*(Tabla generada a partir de `outputs/figures/metricas_comparativas.json` y
`outputs/figures/metricas_svm_baseline.json` tras ejecutar `06_evaluate.py`
y `08_baseline_ml.py`. Valores medidos sobre el conjunto de **test**, 165
imágenes.)*

**Segunda ronda de entrenamiento** (LayerNormalization en vez de
BatchNormalization, learning rate 1e-4, label smoothing 0.1, patience 12,
hasta 80/50 épocas, augmentation reducido en la CNN+Aug, y fine-tuning de
las últimas 50 capas en vez de 30 en transfer learning — ver
[Hallazgos](#hallazgos-y-análisis) para el detalle de cada cambio y su
efecto). Se entrenó íntegramente en CPU: no había soporte de GPU disponible
en la instalación de TensorFlow del equipo (ver nota en
[Instalación](#instalación)).

| Modelo | Exactitud | AUC-ROC | F1 Macro | Sensib. Benigno | Sensib. Maligno | Sensib. Normal |
|---|---:|---:|---:|---:|---:|---:|
| **CNN Base** | **0.8485** | 0.9386 | 0.7305 | 0.44 | 0.99 | 0.78 |
| CNN + Augmentation | 0.5091 ⚠️ | 0.7280 | 0.2249 | 0.00 | 1.00 | 0.00 |
| **Transfer Learning (EfficientNetB0)** | 0.8303 | **0.9761** | **0.7764** | **0.89** | 0.87 | 0.76 |
| SVM (HOG) — Línea base | 1.0000 ⚠️ | 1.0000 ⚠️ | 1.0000 ⚠️ | 1.00 | 1.00 | 1.00 |

⚠️ Los resultados perfectos de la SVM siguen sin ser creíbles (fuga de datos,
sin cambios — no se retocó ese modelo). La CNN + Augmentation mejoró su
AUC-ROC pero sigue colapsando a nivel de predicción final (ver Hallazgos).

**¿Cuál es "el mejor" ahora?** Depende del criterio: la **CNN Base** tiene
la exactitud global más alta (84.85%), pero la **Transfer Learning** tiene
mejor AUC-ROC (0.976), mejor F1 macro, y sobre todo una sensibilidad mucho
más equilibrada entre clases — incluida la clase minoritaria Benigno
(0.89, frente a 0.44 de la CNN Base). En un problema médico, esa
sensibilidad equilibrada por clase pesa más que un punto extra de
exactitud global, así que **Transfer Learning sigue siendo el modelo
recomendado**, aunque ahora con una CNN Base mucho más competitiva de lo
que era en la primera ronda.

Figuras nuevas de esta ronda: `curvas_roc_comparativas.png` (las 4 curvas
ROC macro-average superpuestas) y `classification_report_<modelo>.png`
(precision/recall/f1/support por clase, como imagen, para cada uno de los
4 modelos) en `outputs/figures/`.

## Hallazgos y análisis

### 1. Primera ronda: las CNN entrenadas desde cero colapsan a predecir una única clase

En la primera ronda de entrenamiento, tanto la **CNN Base** como la
**CNN + Augmentation** mostraban un patrón idéntico e inestable: la
exactitud en entrenamiento subía con normalidad (hasta 78-84%), pero la
pérdida de validación **crecía de forma monótona desde la primera época**
(p. ej., CNN Base: de 1.23 en época 1 a 10.64 en época 8) mientras la
exactitud de validación quedaba **congelada** en un único valor. Ambos
modelos colapsaban a predecir **una sola clase para todo el conjunto de
prueba**: CNN Base siempre "Maligno" (exactitud 50.9%, la proporción de
Maligno en el conjunto), CNN + Augmentation siempre "Normal" (exactitud
38.2%). El `EarlyStopping` (paciencia 7) terminaba restaurando los pesos
de la época 1, la única con pérdida de validación razonable.

**Hipótesis de causa raíz** planteada en esa ronda: la combinación de
`BatchNormalization` en cada bloque + `class_weight="balanced"` agresivo
(peso ×3.04 para Benigno) + *learning rate* alto (1×10⁻³) + muy pocos
pasos por época (24) producía una optimización inestable: las
estadísticas de `BatchNormalization` no llegaban a estabilizarse con tan
pocos batches por época.

Durante esa ronda también se detectó y corrigió un **bug real**: la capa
de augmentation se conectaba con `training=True` fijado explícitamente,
lo que dejaba la augmentation activa incluso en `evaluate()`/`predict()`.
Se corrigió, y **se confirmó que ese bug no era la causa de la
inestabilidad** (el colapso se reprodujo igual tras corregirlo), aunque sí
mejoró la evaluación del modelo de transfer learning de esa ronda.

### 2. Segunda ronda: el diagnóstico se confirmó — la CNN Base se arregló por completo

Con GPU solicitada pero no disponible de forma nativa en Windows para esta
instalación de TensorFlow (ver [Instalación](#instalación)), se reentrenó
en CPU aplicando directamente la hipótesis de causa raíz de la ronda
anterior:

- `LayerNormalization` en vez de `BatchNormalization` en los 3 bloques
  convolucionales (normaliza por muestra, no depende de estadísticas de
  batch — inmune al problema de pocos batches/época).
- Learning rate bajado de 1×10⁻³ a **1×10⁻⁴**.
- `label_smoothing=0.1` en la loss (requirió cambiar a
  `CategoricalCrossentropy` + etiquetas one-hot, ver `03_train_cnn_base.py`).
- `EarlyStopping` con paciencia subida de 7 a **12**, y presupuesto de
  épocas subido de 50 a **80**.

**Resultado: el colapso de la CNN Base desapareció por completo.**
Exactitud en test: de 50.9% a **84.85%**; AUC-ROC: de 0.536 a **0.939**;
sensibilidad en Maligno 0.99, en Normal 0.78, y en Benigno 0.44 (la más
baja, pero ya no es cero). Las curvas de entrenamiento muestran una
convergencia normal, sin la explosión de pérdida de validación de la
primera ronda. **Esto confirma la hipótesis de causa raíz**: el problema
no era la arquitectura en sí ni el dataset en sí, sino específicamente la
combinación de `BatchNormalization` con pocos batches por época y un
learning rate demasiado alto para ese régimen.

### 3. La CNN + Augmentation solo mejoró parcialmente: el colapso persiste a nivel de decisión final

Se aplicaron exactamente los mismos cambios a la CNN + Augmentation, más
una reducción del augmentation en sí (se quitó `RandomFlip("vertical")` y
`RandomTranslation`, y se redujo la magnitud de `RandomRotation` y
`RandomZoom` — ver `04_train_cnn_aug.py`). El resultado es mixto:

- El **AUC-ROC mejoró sustancialmente**, de 0.503 a **0.728** — la red sí
  está aprendiendo a ordenar mejor sus probabilidades entre clases.
- Pero la **exactitud final (50.9%) sigue siendo la de un colapso**: la
  matriz de confusión confirma que el modelo predice "Maligno" para
  prácticamente todo el conjunto de test (sensibilidad 1.00 en Maligno,
  0.00 en Benigno y Normal) — el mismo patrón de la primera ronda, solo
  que ahora colapsando hacia la clase mayoritaria en vez de hacia Normal.

**Lectura de este resultado:** dado que la CNN Base (arquitectura
idéntica, mismos hiperparámetros, sin augmentation) sí se arregló por
completo con los mismos cambios, la diferencia entre ambos resultados solo
puede deberse al augmentation en tiempo real. La hipótesis más plausible
es que, incluso reducido, el augmentation introduce suficiente variabilidad
adicional en cada batch como para que, combinado con un dataset ya pequeño
(24 batches/época) y el desbalance de clases, el optimizador siga
encontrando un mínimo donde predecir la clase mayoritaria es "más fácil"
que aprender la variación real entre clases. Que el AUC mejore sin que la
exactitud lo haga sugiere que el modelo aprende información útil (mejor
*ranking* de probabilidades) pero el umbral de decisión final (argmax) no
llega a cruzarse hacia las otras clases — sería candidato a ajustar
umbrales de decisión por clase en vez de usar argmax puro, como trabajo
futuro. Siguiendo el criterio del propio usuario del proyecto, este
resultado se documenta como válido en vez de perseguir más iteraciones:
confirma que **el augmentation, no solo la normalización o el learning
rate, es un factor causal de la inestabilidad en este dataset pequeño**.

### 4. Transfer learning mejoró más al descongelar más capas y entrenar más épocas

Con GPU no disponible, se reentrenó igualmente en CPU, descongelando las
**últimas 50 capas** de EfficientNetB0 en la Fase 2 (antes 30) y subiendo
el presupuesto de la Fase 2 de 30 a **50 épocas**. Resultado: exactitud de
70.9% → **83.0%**, AUC-ROC de 0.913 → **0.976**, y sensibilidad en Benigno
de 0.50 → **0.89** (la mejora más notable, en la clase con menos datos).
Maligno bajó ligeramente de recall (0.87 vs 0.75 antes, en realidad
*sube*) y Normal se mantiene similar (0.76). La hipótesis: con más
capas descongeladas y más épocas de fine-tuning, el modelo tiene más
capacidad de adaptar las representaciones preentrenadas de ImageNet al
dominio específico de TC de tórax, sin perder la estabilidad que le da
partir de pesos preentrenados (a diferencia de las CNN entrenadas desde
cero).

Con estos resultados, la comparación entre CNN Base (84.85% exactitud) y
Transfer Learning (83.0% exactitud, pero mejor AUC-ROC y mejor
sensibilidad equilibrada) ya no es tan unilateral como en la primera
ronda — ver la discusión en la sección [Resultados](#resultados) sobre
cuál conviene reportar como "el mejor modelo".

### 5. La línea base SVM+HOG con 100% de exactitud es una señal de alarma, no un éxito

Como se anticipó en las [Limitaciones](#limitaciones-y-consideraciones-éticas),
la SVM con características HOG alcanza exactitud, AUC-ROC y F1 **perfectos
(1.00)** en el conjunto de prueba — superando incluso al modelo de deep
learning más sofisticado. Para un clasificador clásico sin aprendizaje de
representaciones, sobre un problema médico de 3 clases, esto **no es
creíble como capacidad de generalización real**. Es la evidencia más clara
del problema de fuga de datos descrito en el dataset: al no existir
identificador de paciente, es muy probable que cortes de TC casi idénticos
del mismo paciente aparezcan tanto en entrenamiento como en prueba,
permitiendo que un modelo memorice la apariencia de un paciente en lugar
de aprender patrones patológicos generalizables. **Este resultado se
reporta íntegramente por transparencia metodológica, pero no debe leerse
como que "SVM + HOG supera a las CNN"** — la comparación correcta,
imposible de hacer con los datos disponibles, requeriría un split a nivel
de paciente.

### 6. Grad-CAM revela un problema de interpretabilidad en el modelo de transfer learning

Al generar los mapas Grad-CAM sobre el modelo de Transfer Learning (el
mejor de los 3 modelos Keras) se encontró un problema técnico adicional,
más allá de los ya mencionados: el mapa de calor crudo es **exactamente
cero en las 48 de las 49 celdas de la rejilla 7×7 de activaciones**, con
un único píxel en la esquina inferior derecha saturado a 1.0 — y este
patrón se repite **idéntico** para imágenes de entrada completamente
distintas y con distinta clase predicha. Es decir, el mapa de calor que
produce Grad-CAM en este modelo **no es informativo**: no está resaltando
regiones anatómicas relevantes para la predicción.

Se investigó si esto era un error de implementación (de hecho, se
encontró y corrigió un bug real y distinto: en Keras 3, reconstruir un
`keras.Model` recortado desde `model.inputs` hasta la salida de un
submodelo anidado —la base de EfficientNet dentro del modelo de transfer
learning— falla con `Output with path 0 is not connected to inputs`; se
solucionó reproduciendo el *forward pass* manualmente capa por capa dentro
de un `GradientTape`, ver `gradcam_utils.py`). Sin embargo, **tras corregir
ese bug, el patrón de saturación en una esquina persiste**, y los valores
crudos confirman que no es un artefacto de normalización: la activación en
esa celda es órdenes de magnitud mayor que en el resto de la rejilla,
para cualquier imagen de entrada.

La explicación más plausible es un fenómeno conocido en redes
convolucionales/transformers preentrenadas: la aparición de canales con
activaciones de magnitud anormalmente alta y aproximadamente constantes en
posiciones fijas (a menudo cerca de los bordes, por efectos de padding en
convoluciones con stride), que dominan el promedio ponderado por gradiente
de Grad-CAM y eclipsan las contribuciones —más pequeñas pero
semánticamente relevantes— del resto del mapa. Este es un **límite conocido
de Grad-CAM "clásico"** (no de este modelo en particular) frente a ciertas
arquitecturas preentrenadas, y una razón adicional para no confiar
ciegamente en la interpretabilidad de este sistema sin validación adicional
(p. ej. Grad-CAM++, o excluir/normalizar canales atípicos antes de
generar el mapa).

**Implicación práctica:** las imágenes Grad-CAM generadas por
`07_gradcam.py` y mostradas en la app web para el modelo de transfer
learning deben interpretarse con cautela — actualmente no ofrecen una
explicación fiable de en qué se basa el modelo para clasificar cada
imagen, lo cual es en sí mismo un hallazgo relevante sobre las
limitaciones de explicabilidad del sistema.

**Verificación tras la segunda ronda de entrenamiento:** se regeneró el
grid de Grad-CAM con el modelo de transfer learning reentrenado (50 capas
descongeladas, 83.0% de exactitud) y **el mismo patrón de saturación en
una esquina persiste, sin cambios**. Esto refuerza que no depende del
punto de entrenamiento concreto del modelo, sino de una propiedad más
estructural de cómo EfficientNetB0 representa la información en su última
capa convolucional.

## Limitaciones y consideraciones éticas

1. **Posible fuga de datos (*data leakage*) por ausencia de identificador de
   paciente.** El dataset exportado desde Kaggle no incluye metadatos DICOM
   ni ningún campo que identifique a qué paciente pertenece cada corte de
   TC. Es una práctica bien documentada en datasets públicos de TC que un
   mismo paciente aporte docenas de cortes axiales consecutivos, muy
   similares entre sí. Al hacer la partición train/val/test **a nivel de
   imagen** (única opción posible con los datos disponibles), es probable
   que cortes casi idénticos del mismo paciente terminen repartidos entre
   entrenamiento y prueba. Esto **infla artificialmente las métricas de
   generalización**: un modelo puede "memorizar" la apariencia de un
   paciente visto en entrenamiento en lugar de aprender patrones
   patológicos generalizables. La línea base SVM+HOG alcanzando 100% de
   exactitud y 100% de AUC-ROC en test —un resultado atípico para un
   método clásico sin aprendizaje profundo— es la evidencia más clara de
   este problema, y probablemente también infla en algún grado las métricas
   de los modelos CNN. **Este es el límite metodológico más importante del
   proyecto** y debe mencionarse explícitamente en cualquier presentación
   de estos resultados.
2. **Tamaño de dataset reducido para deep learning** (1097 imágenes, 767
   para entrenamiento). Esto favorece el overfitting y limita la capacidad
   de las arquitecturas entrenadas desde cero.
3. **Desbalance de clases** (Benigno: 10.9% del total). Se mitigó con
   `class_weight`, pero no se aplicó remuestreo ni generación sintética
   adicional (p. ej. SMOTE a nivel de características, u oversampling de
   imágenes).
4. **Un único origen de datos** (hospitales de Irak, dataset de 2019). No
   se evaluó generalización a equipos de TC, protocolos de adquisición ni
   poblaciones de otros centros — un paso obligatorio antes de cualquier
   uso clínico real.
5. **Entrenamiento sin GPU y con presupuesto de tiempo limitado**: los
   hiperparámetros (arquitectura, learning rate, nº de épocas) no se
   optimizaron exhaustivamente; ver el análisis de inestabilidad de las
   CNN entrenadas desde cero en la sección de Hallazgos.
6. **No es un dispositivo médico.** Este sistema es un prototipo académico.
   No cuenta con validación clínica, no está certificado como *software as
   a medical device*, y **no debe usarse para apoyar decisiones de
   diagnóstico o tratamiento reales**. Su único propósito es demostrar,
   con fines educativos, la aplicación de CNNs y transfer learning a
   imágenes médicas.

## Trabajo futuro

- Reconstruir un split a **nivel de paciente** (requiere acceso a los
  metadatos DICOM originales del NCCD/IOSH, no solo a las imágenes
  exportadas), y volver a medir todas las métricas — se espera una caída
  sustancial de exactitud, especialmente en la línea base SVM.
- Investigar específicamente **por qué el augmentation en tiempo real
  sigue causando colapso en la CNN + Augmentation** aunque la misma
  arquitectura sin augmentation (CNN Base) ya no colapsa con los mismos
  hiperparámetros (ver Hallazgos #3) — por ejemplo, probar con
  augmentation aún más suave, o aplicado solo a partir de cierta época
  (*curriculum*), en vez de desde el principio.
- Probar **umbrales de decisión por clase** en vez de `argmax` puro para
  la CNN + Augmentation: su AUC-ROC (0.728) sugiere que el modelo sí
  aprende información útil aunque la decisión final colapse, lo que podría
  indicar que el problema está más en el umbral que en el modelo en sí.
- Configurar soporte de GPU real para este proyecto (WSL2, o el plugin
  DirectML de Microsoft para Windows) — permitiría iterar mucho más rápido
  sobre estas hipótesis; en esta ronda se intentó pero la instalación de
  TensorFlow del equipo no tenía soporte nativo de GPU en Windows (ver
  [Instalación](#instalación)), así que todo el reentrenamiento se hizo en
  CPU.
- Validación cruzada (k-fold) en lugar de un único split, dado el tamaño
  reducido del dataset.
- Evaluar en un conjunto externo (otro hospital/dataset público de TC de
  tórax) para medir generalización real.
- Explorar arquitecturas 3D que aprovechen la información volumétrica de
  cortes consecutivos, en lugar de clasificar cortes 2D de forma
  independiente.
