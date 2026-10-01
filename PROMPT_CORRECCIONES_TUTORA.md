# CORRECCIONES DE LA TUTORA — Nuevo protocolo experimental del TFM

## Contexto

Ya conoces este repositorio (lee de nuevo README.md y PROYECTO_CLAUDE_CONTEXTO.txt si hace falta). La tutora revisó el TFM y pidió cambios que afectan a los experimentos. Entrega: **miércoles 7 de octubre**. Lo que hay que resolver:

1. **Fuga de datos.** Todo se partió por imagen. Ya se comprobó que los archivos vienen numerados en orden y que los cortes consecutivos de un mismo paciente son casi idénticos, así que **se pueden reconstruir grupos por paciente**. Con la partición por imagen, el 89–95 % de las imágenes de prueba tiene un "gemelo" casi idéntico en entrenamiento. Con la partición por grupos, ese porcentaje baja a 0–2,5 %. El SVM+HOG pasa de 99–100 % a 81–87 % de exactitud. **Todos los modelos se rehacen con partición por grupos.**
2. **El colapso de los modelos con aumento.** La hipótesis es que `EarlyStopping(monitor="val_loss")` restauraba los pesos de la época 1. La parada temprana debe vigilar el **F1 macro de validación**.
3. **El conjunto de prueba se usó para decidir.** En las rondas anteriores se miró el conjunto de prueba antes de cambiar la configuración. A partir de ahora: **búsqueda de hiperparámetros solo con validación, y el conjunto de prueba se evalúa una única vez al final**.
4. **Faltaba un ajuste sistemático de hiperparámetros.** Hace falta una búsqueda documentada, con sus rangos.
5. **La comparación no estaba controlada.** Cada escenario debe cambiar **un solo factor** respecto al anterior. Además, el aumento de la transferencia (rotación 0,1, zoom 0,1) no era igual al de la CNN (0,08 y 0,05).
6. **Faltaba estadística.** Hay que entrenar con varias semillas e informar intervalos de confianza por bootstrap y pruebas de McNemar.
7. **La explicabilidad no era informativa.** Hay que aplicar Grad-CAM a la red desde cero, y Grad-CAM++ y Score-CAM a la transferencia.
8. **El modelo recomendado se elegía por la clase benigna.** El criterio debe centrarse en la sensibilidad sobre la clase **maligna**, y debe fijarse **antes** de mirar el conjunto de prueba.

## Reglas generales (aplican a todo)

- **NO modifiques ningún .docx.** El documento lo corrijo aparte con las salidas que generes.
- **NO borres nada antiguo.** Mueve los modelos, figuras y JSON actuales a `outputs/legacy_particion_imagen/` (con la misma estructura de subcarpetas), porque el documento los cita como "ensayos preliminares". Después de moverlos, `outputs/figures/`, `outputs/models/` y `outputs/gradcam/` quedan para el protocolo nuevo.
- **El conjunto de prueba no se usa para ninguna decisión.** En la fase de búsqueda ni siquiera se predice sobre él. Solo se predice en la Tarea 5, una vez por modelo y semilla, cuando todas las decisiones ya están escritas en disco.
- **Comparaciones pareadas:** en cada semilla, todos los modelos usan exactamente la misma partición train/val/test.
- **Ejecuciones reanudables:** cada entrenamiento guarda modelo, historial y predicciones con un nombre único (`{escenario}_{config}_seed{S}`). Si el archivo ya existe, se salta. Así, si el equipo se apaga a mitad de la noche, basta con relanzar.
- **Registro:** anota cada decisión, tiempo de ejecución e incidencia en `outputs/experimentos/registro.md`.
- **Commit y push al terminar cada tarea**, no todo al final. Sin los archivos de modelo pesados si superan el límite de GitHub (en ese caso, sube solo predicciones, historiales, CSV, JSON y figuras).
- Windows: `$env:PYTHONIOENCODING="utf-8"` antes de ejecutar.
- Si algo falla o un resultado sale raro, **documéntalo y repórtalo tal cual**. Los resultados negativos también son resultados.

---

## TAREA 1 — Grupos por paciente y particiones (rápida, hacer primero)

Copia el script adjunto `02c_grupos_paciente.py` en `src/` y ejecútalo. Ya está probado con las imágenes del NPZ del repositorio. Lo que hace:

1. Ordena los archivos de cada clase por el número entre paréntesis (orden natural: "(2)" va antes que "(10)").
2. Calcula la correlación entre cada corte y el siguiente (escala de grises, 64×64).
3. Corta la secuencia en los K−1 enlaces más débiles, con K = 15 benignos, 40 malignos y 55 normales (número de casos documentado por los autores del conjunto).
4. Fusiona los grupos que tengan algún par de cortes con correlación > 0,97.
5. Genera las particiones por grupos (~70/15/15, estratificadas) para las semillas 0 a 4, y también las particiones por imagen equivalentes como referencia.

**Verificación esperada** (si no sale algo muy parecido, detente y avísame):
- 1097 imágenes, **83 grupos** (Benigno 15, Maligno 30, Normal 38).
- Correlación consecutiva mediana: 0,97 / 0,99 / 0,99.
- Gemelos > 0,95 en prueba: **89–95 %** por imagen frente a **0–2,5 %** por grupos.
- Revisa `outputs/figures/montaje_grupos_paciente.png`. Cada fila debe mostrar cortes del mismo estudio. Si ves filas que mezclan claramente pacientes distintos (en la misma clase), anótalo. Fusionar de más no causa fuga, separar de más sí.

Si algún archivo no sigue el patrón `nombre (N).ext`, el script se detiene. Dime qué nombres aparecen.

---

## TAREA 2 — Infraestructura común del protocolo

Crea `src/protocolo.py`, un módulo con todo lo compartido. Los scripts de la Tarea 3 en adelante lo importan.

### 2.1 Datos
- `cargar_imagenes()`: todas las imágenes en el **mismo orden que `grupos_paciente.csv`**, redimensionadas a 224×224 con LANCZOS y escaladas a [0,1] (dividiendo entre 255). Guárdalas en caché en `outputs/splits/imagenes_224.npz` (uint8, para que pese poco).
- `cargar_segmentadas()`: aplica la función de segmentación de `02b_segmentation.py` a **todas** las imágenes una sola vez. Caché en `outputs/splits/imagenes_224_segmentadas.npz`, guardando también las máscaras.
- `particion(seed, tipo="grupos")`: lee `particion_grupos_seed{S}.npz` o `particion_imagen_seed{S}.npz`.

### 2.2 Control de calidad de las máscaras (la tutora lo pide en Métodos, no en Conclusiones)
- Para cada imagen: porcentaje del área ocupado por la máscara, número de componentes y si toca el borde.
- Marca como sospechosas las máscaras con área < 5 % o > 60 %, o con más de 2 componentes. Informa cuántas hay por clase.
- `outputs/figures/control_calidad_mascaras.png`: montaje de 30 máscaras al azar (10 por clase) superpuestas en rojo semitransparente sobre la imagen.
- Guarda las cifras en `outputs/experimentos/calidad_mascaras.json`.

### 2.3 Aumento de datos (UNIFICADO para ambas familias)
- `geometrico`: `RandomFlip("horizontal")`, `RandomRotation(0.08)`, `RandomZoom(0.05)`, `RandomContrast(0.1)`. Es el mismo para la CNN y para EfficientNet. Antes la transferencia usaba 0,1/0,1, y eso era un segundo factor que cambiaba a la vez.
- `cutmix`: la implementación actual de `04b_train_cnn_cutmix.py` (alpha = 1,0), sin cambios.
- Solo se aplica al conjunto de entrenamiento.

### 2.4 Parada temprana por F1 macro de validación

```python
from sklearn.metrics import f1_score, recall_score

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

callbacks = [
    F1MacroValidacion(X_val, y_val),
    keras.callbacks.EarlyStopping(monitor="val_f1_macro", mode="max", patience=15,
                                  start_from_epoch=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_f1_macro", mode="max",
                                      factor=0.5, patience=5, min_lr=1e-6),
]
```
Comprueba en el primer entrenamiento que `history.history` contiene `val_f1_macro`. Si no aparece, el orden de los callbacks está mal.

### 2.5 Constructores
- `cnn_propia(lr, dropout_bloques)`: la arquitectura actual de `03_train_cnn_base.py` (3 bloques, LayerNormalization, label smoothing 0,1, Adam). El dropout de los bloques queda como parámetro y el de la cabeza se mantiene en 0,5.
- `efficientnet(lr_ajuste, capas_descongeladas)`: la arquitectura actual de `05_train_transfer.py`. Fase 1 con la base congelada (lr 1e-3, máx. 20 épocas) y fase 2 de ajuste fino (máx. 60 épocas). Ambas fases usan la parada temprana por F1.
- Comunes: `class_weight` balanceado, lote de 32, máx. 80 épocas para la CNN. Semilla: `keras.utils.set_random_seed(seed)` al inicio de cada entrenamiento.

### 2.6 `entrenar(escenario, hp, seed, tipo_particion="grupos")`
Guarda en `outputs/experimentos/runs/{nombre}/`: `modelo.keras`, `historial.csv` (con val_f1_macro y las sensibilidades de validación), `pred_val.npz` (probabilidades + etiquetas), `tiempo.json` y `hp.json`. **No predice sobre la prueba.**

### 2.7 Presupuesto de tiempo
Antes de lanzar nada largo, mide el tiempo de 2 épocas de la CNN y de 2 de EfficientNet en este equipo, y estima el total de las Tareas 3 y 4. Anota la estimación en `registro.md`. Si el total en CPU supera ~14 h, usa 3 semillas en vez de 5 (esto ya está previsto abajo) y avísame de la estimación.

---

## TAREA 3 — Búsqueda de hiperparámetros (solo validación, semilla 0, partición por grupos)

El diseño sigue la tabla de la guía de la tutora. Cada escenario cambia un solo factor:

| Escenario | Arquitectura | Aumento | Entrada | Qué aísla |
|---|---|---|---|---|
| M1 | CNN propia | Ninguno | Imagen completa | Referencia |
| M2 | CNN propia | Geométrico | Imagen completa | Aumento geométrico |
| M3 | CNN propia | CutMix | Imagen completa | Aumento avanzado |
| M4 | EfficientNetB0 | El mejor de M2/M3 | Imagen completa | Transferencia |
| M5 | EfficientNetB0 | El de M4 | Imagen segmentada | Segmentación |

Secuencia:
1. **Rejilla de M1:** `lr ∈ {3e-4, 1e-4}` × `dropout_bloques ∈ {0.25, 0.40}`, es decir, 4 entrenamientos. Elige la configuración con mayor **F1 macro de validación**. Si dos configuraciones quedan a menos de 0,01, desempata la mayor sensibilidad maligna de validación, y si persiste el empate, la lr más baja.
2. **M2 y M3** se entrenan con los hiperparámetros elegidos para M1 (semilla 0). Solo cambia el aumento.
3. **Aumento para M4** = el de M2 o M3 con mayor F1 macro de validación (aunque los dos queden por debajo de M1: así lo define la guía, y si pasa se comenta en el documento).
4. **Rejilla de M4:** `lr_ajuste ∈ {1e-4, 1e-5}` × `capas_descongeladas ∈ {20, 50}`, es decir, 4 entrenamientos. Mismo criterio de selección.
5. **M5** usa los hiperparámetros elegidos para M4, sobre imágenes segmentadas.

Salidas:
- `outputs/experimentos/busqueda_hiperparametros.csv`: una fila por entrenamiento, con escenario, todos los hiperparámetros, épocas efectivas, época del mejor valor, F1 macro de validación, sensibilidades de validación por clase y tiempo.
- `outputs/experimentos/hiperparametros_finales.json`: la configuración completa de los cinco escenarios (todas las variables, incluidas las fijas), más el aumento elegido para M4 y por qué.
- Curvas de aprendizaje de la configuración elegida de cada escenario, semilla 0: `outputs/figures/curvas_M{1..5}.png`, con 3 paneles (pérdida, exactitud y F1 macro de validación) y la época restaurada marcada.

**Reutilización:** el entrenamiento de semilla 0 con la configuración elegida de cada escenario ES el entrenamiento de semilla 0 de la Tarea 4. No lo repitas.

---

## TAREA 4 — Entrenamiento final multisemilla

- Semillas: **0, 1, 2** como mínimo; **0 a 4** si el tiempo lo permite.
- Para cada semilla, entrena M1–M5 con su configuración final sobre la partición por grupos de esa semilla.
- **Línea base SVM+HOG** (mismos parámetros HOG que `08_baseline_ml.py`): elige `C ∈ {1, 10, 100}` por F1 macro de validación en cada semilla. Se entrena en las mismas particiones por grupos, y **también** en las particiones por imagen de las mismas semillas, para cuantificar la inflación por fuga.
- **Opcional, solo si sobra tiempo:** M1 con partición por imagen en las mismas semillas, para mostrar la inflación también en una CNN.

**Antes de pasar a la Tarea 5**, escribe `outputs/experimentos/seleccion_modelo_recomendado.json` con fecha y hora:
- Criterio fijado de antemano: **mayor sensibilidad media sobre la clase maligna en VALIDACIÓN** (media de semillas). Si dos escenarios quedan a menos de 0,02, desempata el F1 macro medio de validación.
- Las cifras de validación que justifican la elección.

Este archivo se escribe y se sube a git **antes** de calcular cualquier métrica de prueba.

---

## TAREA 5 — Evaluación en prueba (una sola vez) y estadística

Crea `src/09_evaluacion_final.py`:

1. Para cada modelo y semilla, predice sobre su conjunto de prueba **una vez**. Guarda `pred_test.npz`.
2. Métricas por modelo y semilla: exactitud, F1 macro, AUC-ROC macro uno-contra-el-resto, sensibilidad y especificidad por clase, matriz de confusión, y recuentos **maligno→normal** y **maligno→benigno**. Salida: `outputs/experimentos/metricas_por_semilla.csv`, que va al Anexo C.
3. Resumen por modelo, en `outputs/experimentos/resumen_ic95.csv`:
   - media ± desviación estándar entre semillas;
   - **IC 95 % por bootstrap** sobre las predicciones de prueba agregadas de todas las semillas (2000 remuestreos de imágenes, semilla fija), para exactitud, F1 macro, AUC, sensibilidad maligna y sensibilidad benigna.
4. **McNemar exacto** (`statsmodels.stats.contingency_tables.mcnemar(exact=True)`) entre el modelo recomendado y cada uno de los demás, sobre las predicciones agregadas y también por semilla, con corrección de Holm. Salida: `outputs/experimentos/mcnemar.csv`.
5. Tabla de errores clínicos: por modelo, número total de malignos de prueba (sumando semillas), cuántos acabaron como normal y cuántos como benigno. Salida: `outputs/experimentos/errores_maligno.csv`.
6. Figuras:
   - `outputs/figures/resumen_modelos_ic.png`: barras con IC 95 % de F1 macro, sensibilidad maligna y sensibilidad benigna para M1–M5 y SVM.
   - `outputs/figures/matrices_confusion_agregadas.png`: rejilla con la matriz sumada sobre semillas de cada modelo, con recuentos y porcentaje por fila.
   - `outputs/figures/curvas_roc_final.png`: curvas ROC macro de los 6 modelos con las predicciones agregadas.
   - `outputs/figures/svm_imagen_vs_grupos.png`: la misma línea base partida por imagen y por grupos (exactitud y F1 macro con IC).

---

## TAREA 6 — Explicabilidad

Usa los modelos de **semilla 0** y su conjunto de prueba.

1. **Grad-CAM** sobre M1 (última capa convolucional) y sobre el modelo recomendado, si es distinto.
2. **Grad-CAM++ y Score-CAM** sobre M4 (capa `top_conv` de EfficientNetB0), además del Grad-CAM que ya existía.
   - Grad-CAM++: pesos α = g² / (2g² + ΣA·g³), con el denominador protegido contra cero. Peso del canal = Σ α·ReLU(g). Mapa = ReLU(Σ w·A).
   - Score-CAM: para cada canal, se sobremuestrea a 224×224, se normaliza a [0,1], se multiplica por la imagen de entrada y se toma la probabilidad de la clase objetivo. Los pesos son el softmax de esas probabilidades. Para acotar el coste en CPU, usa solo los **256 canales con mayor activación media** y documenta esta decisión.
3. Casos: por clase, **2 aciertos y 2 errores** (o los que existan). Incluye los errores maligno→normal si los hay. Figuras:
   - `outputs/gradcam/gradcam_M1.png`
   - `outputs/gradcam/comparacion_metodos_M4.png`: filas = casos, columnas = imagen, Grad-CAM, Grad-CAM++ y Score-CAM.
   - `outputs/gradcam/gradcam_recomendado.png`
4. **Métricas objetivas de informatividad** (`outputs/experimentos/explicabilidad.csv`), por modelo y método:
   - **Energía dentro del tórax pulmonar:** fracción de la suma del mapa que cae dentro de la máscara pulmonar **rellenada y con envolvente convexa por pulmón, dilatada 5 px**. Hay que usar la envolvente porque la máscara de Otsu excluye las masas densas, y sin ella un mapa que apunte al tumor contaría como "fuera". Criterio: un mapa se considera "pulmonar" si esa fracción es ≥ 0,5. Informa el porcentaje de mapas pulmonares.
   - **Correlación media entre mapas de imágenes de clases distintas.** Antes salía 0,999: un mapa constante. Calcúlala sobre todos los pares entre clases del conjunto de casos.
   - **Fracción de energía en la esquina** (celda 1/7 × 1/7 superior izquierda), para comprobar si persiste el artefacto anterior.

---

## TAREA 7 — Tablas descriptivas, figura de preprocesamiento y entorno

1. `outputs/experimentos/imagenes_por_subconjunto.csv`: imágenes y grupos por clase en train/val/test para cada semilla. Añade, para la semilla 0, el número de **muestras vistas durante el entrenamiento** (imágenes × épocas efectivas) en M1, M2 y M3. El aumento es en línea: no aumenta el número de imágenes distintas, pero cada época ve versiones transformadas.
2. `outputs/figures/pipeline_preprocesamiento.png`: para un ejemplo de cada clase, las columnas original, redimensionada y normalizada, máscara, segmentada, aumento geométrico y CutMix.
3. `outputs/experimentos/entorno.json`: sistema operativo, CPU, RAM, GPU (y si TensorFlow la usó o no), y versiones de Python, TensorFlow, Keras, scikit-learn, scikit-image, OpenCV, NumPy y statsmodels. Actualiza `requirements.txt` con versiones fijas.

---

## Al terminar

1. Actualiza README.md y PROYECTO_CLAUDE_CONTEXTO.txt con el protocolo nuevo, separando claramente "ensayos preliminares (partición por imagen)" y "protocolo final (partición por grupos)".
2. Escribe `outputs/experimentos/RESUMEN_PARA_DOCUMENTO.md` con:
   - la tabla final de M1–M5 + SVM (media ± DE e IC 95 %) y las sensibilidades/especificidades por clase;
   - el modelo recomendado y su justificación con cifras de validación;
   - los errores maligno→normal / maligno→benigno por modelo;
   - McNemar;
   - si el aumento sigue colapsando o no con la parada por F1 (la pregunta clave);
   - qué muestran Grad-CAM++ y Score-CAM frente a Grad-CAM (con las tres métricas);
   - la lista de todas las figuras y CSV nuevos, con su ruta.
3. Commit y push.

## Prioridad si el tiempo no alcanza
Tareas 1 → 2 → 3 → 4 (con 3 semillas) → 5 → 6 → 7. Lo que no llegue a hacerse se documenta como pendiente. Lo mínimo imprescindible para el documento es: Tarea 1, la parada por F1, M1–M5 + SVM en partición por grupos con al menos 3 semillas, la Tarea 5 y el Grad-CAM de M1 con al menos una variante (Grad-CAM++) sobre M4.
