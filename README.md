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
- Este proyecto se desarrolló y probó **sin GPU** (solo CPU). El
  entrenamiento es funcional pero lento; ver tiempos orientativos en la
  sección [Metodología](#metodología).

## Cómo ejecutar el pipeline

Los scripts se ejecutan en orden desde `src/`, y cada uno depende de los
artefactos generados por el anterior (splits en `outputs/splits/`, modelos en
`outputs/models/`):

```bash
cd src
export PYTHONIOENCODING=utf-8   # o $env:PYTHONIOENCODING="utf-8" en PowerShell

python 01_eda.py                # Análisis exploratorio → outputs/figures/
python 02_preprocessing.py      # Split 70/15/15 → outputs/splits/dataset_splits.npz
python 03_train_cnn_base.py     # ~10-20 min en CPU
python 04_train_cnn_aug.py      # ~15-30 min en CPU
python 05_train_transfer.py     # ~20-45 min en CPU (2 fases)
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

| Modelo | Arquitectura | Particularidad |
|---|---|---|
| **CNN Base** | 3 bloques Conv2D(32→64→128) + BatchNorm + MaxPooling + GAP + Dense(256) | Sin aumento de datos |
| **CNN + Augmentation** | Misma arquitectura | Capa de augmentation integrada (flip, rotación, zoom, contraste, traslación) |
| **Transfer Learning** | EfficientNetB0 preentrenada en ImageNet + cabeza personalizada | Fase 1: base congelada (15 épocas). Fase 2: fine-tuning de las últimas 30 capas (hasta 30 épocas) |
| **SVM (línea base)** | HOG (9 orientaciones, celdas 16×16) + SVM kernel RBF | Método clásico de ML, sin deep learning |

Todos los modelos de Keras usan `class_weight="balanced"` para compensar el
desbalance de clases, `EarlyStopping` sobre `val_loss` y
`ReduceLROnPlateau`. El entrenamiento se ejecutó **enteramente en CPU** (sin
GPU disponible en el equipo de desarrollo).

## Resultados

*(Tabla generada a partir de `outputs/figures/metricas_comparativas.json` y
`outputs/figures/metricas_svm_baseline.json` tras ejecutar `06_evaluate.py`
y `08_baseline_ml.py`. Valores medidos sobre el conjunto de **test**, 165
imágenes.)*

| Modelo | Exactitud | AUC-ROC | F1 Macro | Sensib. Benigno | Sensib. Maligno | Sensib. Normal |
|---|---:|---:|---:|---:|---:|---:|
| CNN Base | 0.5091 | 0.5364 | 0.2249 | 0.00 | 1.00 | 0.00 |
| CNN + Augmentation | 0.3818 | 0.5027 | 0.1842 | 0.00 | 0.00 | 1.00 |
| **Transfer Learning (EfficientNetB0)** | **0.7091** | **0.9134** | **0.6292** | 0.50 | 0.75 | 0.71 |
| SVM (HOG) — Línea base | 1.0000 ⚠️ | 1.0000 ⚠️ | 1.0000 ⚠️ | 1.00 | 1.00 | 1.00 |

⚠️ Los resultados perfectos de la línea base SVM no reflejan una capacidad de
generalización real — ver el análisis de fuga de datos más abajo. **El mejor
modelo *evaluable de forma fiable* es la Transfer Learning con EfficientNetB0.**

## Hallazgos y análisis

### 1. Las CNN entrenadas desde cero colapsan a predecir una única clase

Tanto la **CNN Base** como la **CNN + Augmentation** muestran un patrón de
entrenamiento idéntico e inestable: la exactitud en entrenamiento sube con
normalidad (hasta 78-84%), pero la pérdida de validación **crece de forma
monótona desde la primera época** (p. ej., CNN Base: de 1.23 en época 1 a
10.64 en época 8) mientras la exactitud de validación queda **congelada** en
un único valor durante todo el entrenamiento. Al inspeccionar las
predicciones se confirma que ambos modelos colapsan a predecir **una sola
clase para todo el conjunto de prueba**:

- CNN Base → predice siempre "Maligno" (sensibilidad 1.00 en Maligno, 0.00 en
  las otras dos clases; exactitud 50.9%, que coincide con la proporción de
  Maligno en el conjunto).
- CNN + Augmentation → predice siempre "Normal" (sensibilidad 1.00 en Normal,
  0.00 en las otras dos; exactitud 38.2%).

El `EarlyStopping` (paciencia 7, `restore_best_weights=True`) termina
restaurando los pesos de la **época 1**, la única con una pérdida de
validación razonable, porque todas las épocas posteriores empeoran.

**Hipótesis de causa raíz** (no se descarta con los recursos de este
proyecto, ver Trabajo futuro): la combinación de `BatchNormalization` en
cada bloque convolucional + `class_weight="balanced"` agresivo (peso ×3.04
para Benigno) + *learning rate* inicial relativamente alto (1×10⁻³) + muy
pocos pasos por época (24, dado el tamaño del dataset) parece producir una
optimización inestable desde el arranque: las estadísticas de
`BatchNormalization` no llegan a estabilizarse con tan pocos batches por
época, y el modelo cae rápidamente en un mínimo local donde predecir la
clase mayoritaria (o la más "fácil" de sobreajustar con el augmentation
activo) minimiza la pérdida de entrenamiento a costa de la generalización.

Durante el desarrollo se detectó y corrigió un **bug real** en ambos
scripts (`04_train_cnn_aug.py`, `05_train_transfer.py`): la capa de
aumento de datos se conectaba con `training=True` fijado explícitamente al
construir el grafo del modelo, lo cual —a diferencia de lo que cabría
esperar— **deja la augmentation activa permanentemente**, incluso dentro de
`model.evaluate()` o `model.predict()`, porque en Keras un valor de
`training` fijado explícitamente en la llamada a una capa queda "clavado"
para ese nodo del grafo y ya no seguirá el modo (entrenamiento/inferencia)
de la llamada al modelo completo. Se corrigió eliminando ese argumento
para que el modo se propague correctamente. **Se confirmó que este bug no
era la causa de la inestabilidad**: tras corregirlo y reentrenar la CNN +
Augmentation desde cero, el colapso a una única clase se reprodujo de forma
idéntica. Sin embargo, la corrección sí era necesaria por corrección
metodológica (evaluación determinista) y **mejoró de forma medible** los
resultados del modelo de Transfer Learning (ver punto 2).

### 2. Transfer learning es sustancialmente más estable y preciso

El modelo de **Transfer Learning (EfficientNetB0)** no muestra el colapso
de los modelos anteriores: alcanza 70.9% de exactitud y **0.913 de AUC-ROC**
en test, con sensibilidad razonable en las 3 clases (Benigno 0.50, Maligno
0.75, Normal 0.71). La hipótesis más plausible es que, al partir de una
base preentrenada en ImageNet con sus estadísticas de
`BatchNormalization` ya calibradas (y congeladas en la Fase 1), el modelo
evita la inestabilidad de optimización que afecta a las CNN entrenadas
desde cero con este dataset pequeño y desbalanceado.

Después de corregir el bug de `training=True` descrito arriba, el
AUC-ROC del modelo de transfer learning mejoró de 0.907 a **0.913** y, más
notablemente, la sensibilidad en la clase minoritaria (Benigno) pasó de
**0.11 a 0.50** — evidencia de que evaluar con augmentation activo
introducía ruido que perjudicaba especialmente a la clase con menos
ejemplos (120 imágenes en total, 18 en test).

Aun así, la sensibilidad en Benigno (0.50) sigue siendo la más baja de las
tres clases, coherente con que es la clase minoritaria del dataset (10.9%
del total). Esto es clínicamente relevante: un falso negativo en un caso
maligno o benigno tiene consecuencias muy distintas a un falso negativo en
"normal", por lo que la sensibilidad por clase —no solo la exactitud
global— debe ser el criterio principal de evaluación en este dominio.

### 3. La línea base SVM+HOG con 100% de exactitud es una señal de alarma, no un éxito

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

### 4. Grad-CAM revela un problema de interpretabilidad en el modelo de transfer learning

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
- Ajustar hiperparámetros (learning rate, regularización, arquitectura) de
  la CNN base y la CNN con augmentation para resolver la inestabilidad de
  entrenamiento observada.
- Validación cruzada (k-fold) en lugar de un único split, dado el tamaño
  reducido del dataset.
- Evaluar en un conjunto externo (otro hospital/dataset público de TC de
  tórax) para medir generalización real.
- Explorar arquitecturas 3D que aprovechen la información volumétrica de
  cortes consecutivos, en lugar de clasificar cortes 2D de forma
  independiente.
