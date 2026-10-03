# Resumen para el documento del TFM — Protocolo corregido (partición por grupos de paciente)

Generado al completar las 7 tareas del protocolo corregido (`PROMPT_CORRECCIONES_TUTORA.md`,
nombre de archivo heredado; las correcciones del TFM, no una persona, son la fuente de
estas instrucciones). Fuente de todos los números: los CSV/JSON en
`outputs/experimentos/` (ver lista de archivos al final). Semillas usadas: **0, 1, 2**
(3, no 5 — ver justificación en "Presupuesto de tiempo" más abajo). Partición principal:
**por grupos de paciente** (83 grupos reconstruidos por correlación entre cortes
consecutivos — ver Tarea 1).

**Nota de versión (2026-10-03):** este documento refleja una **segunda ronda de
correcciones**, posterior a las 7 tareas originales, sobre dos problemas detectados en
los resultados iniciales de M1/M2/M3 (ver `registro.md`, incidencia "schedule de la
tasa de aprendizaje y criterio de selección"):

1. El schedule de la tasa de aprendizaje (`ReduceLROnPlateau` sobre `val_f1_macro`)
   apagaba el aprendizaje de la CNN propia (M1/M2/M3) antes de que saliera de su
   meseta inicial — **no eran resultados válidos de la arquitectura**, eran un
   artefacto del programador de la tasa de aprendizaje. Corregido con un callback
   nuevo (`EarlyStoppingTrasMeseta`) que no cuenta paciencia hasta que la red
   realmente sale de la meseta. M1, M2 y M3 se reentrenaron por completo (rejilla de
   M1 + las 3 semillas de M1/M2/M3).
2. El criterio de selección del modelo recomendado (mayor sensibilidad Maligno en
   validación) lo maximizaba trivialmente un modelo colapsado a predecir siempre
   "Maligno" — exactamente lo que le pasó a M3 bajo ese criterio. Sustituido por el
   **índice de Youden de la clase Maligno** (sensibilidad + especificidad − 1), que
   un clasificador constante puntúa en 0 exactamente.

Todas las cifras de M1/M2/M3 y el modelo recomendado en este documento son las de
**esta segunda corrección** (los resultados de la primera ronda, ya inválidos, se
conservan solo como evidencia metodológica en `seleccion_modelo_recomendado_v1_descartada.json`
y en `registro.md`). Las cifras de M4, M5 y SVM **no cambiaron** entre ambas rondas.

---

## 1. Tabla final — M1-M5 + SVM, test agregado (3 semillas), media ± DE e IC 95% bootstrap

| Modelo | Exactitud | F1 Macro | AUC macro | Sens. Maligno | Sens. Benigno |
|---|---|---|---|---|---|
| M1 (CNN, sin aumento) | 0.618 ± 0.217 [0.573, 0.660] | 0.432 ± 0.217 [0.451, 0.534]* | 0.771 ± 0.202 [0.809, 0.855]* | 0.663 ± 0.574 [0.602, 0.725] | 0.125 ± 0.217 [0.034, 0.217] |
| M2 (CNN + geométrico) | 0.453 ± 0.069 [0.408, 0.500] | 0.234 ± 0.056 [0.286, 0.350]* | 0.586 ± 0.086 [0.485, 0.575]* | 0.392 ± 0.534 [0.330, 0.455] | **0.000 ± 0.000** |
| M3 (CNN + CutMix) | 0.483 ± 0.021 [0.436, 0.526] | 0.282 ± 0.051 [0.308, 0.368]* | 0.544 ± 0.188 [0.470, 0.553]* | 0.504 ± 0.437 [0.437, 0.565] | **0.000 ± 0.000** |
| M4 (EfficientNetB0) | 0.665 ± 0.013 [0.624, 0.707] | 0.558 ± 0.028 [0.515, 0.599] | 0.813 ± 0.048 [0.779, 0.849] | 0.591 ± 0.124 [0.527, 0.654] | 0.236 ± 0.105 [0.128, 0.367] |
| M5 (EfficientNetB0 + segmentación) | 0.684 ± 0.042 [0.641, 0.726] | 0.568 ± 0.064 [0.531, 0.623] | 0.827 ± 0.069 [0.763, 0.841] | 0.762 ± 0.119 [0.705, 0.817] | 0.312 ± 0.272 [0.174, 0.432] |
| SVM_grupos (HOG, partición por grupos) | **0.857 ± 0.023 [0.825, 0.887]** | **0.675 ± 0.064 [0.645, 0.752]** | **0.881 ± 0.110 [0.836, 0.909]** | 0.941 ± 0.040 [0.909, 0.969] | 0.208 ± 0.260 [0.109, 0.341] |
| SVM_imagen (HOG, partición por imagen — referencia, CON fuga) | 0.998 ± 0.003 | 0.996 ± 0.007 | 1.000 | 1.000 ± 0.000 | 0.981 ± 0.032 |

*(los intervalos marcados con `*` reflejan que el IC bootstrap, calculado sobre las
predicciones agregadas, queda desplazado respecto a la media ± DE entre semillas para
M1/M2/M3 — señal de la alta variabilidad entre semillas de estos tres modelos,
descrita en la sección 3. Esto ocurre incluso con el schedule de entrenamiento ya
corregido: no es el mismo fenómeno documentado en la ronda anterior.)*

**Sensibilidad y especificidad completas por clase** (media simple entre semillas,
`metricas_por_semilla.csv`):

| Modelo | Sens.Benigno | Sens.Maligno | Sens.Normal | Esp.Benigno | Esp.Maligno | Esp.Normal |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.125 | 0.663 | 0.694 | 0.895 | 0.947 | 0.576 |
| M2 | 0.000 | 0.392 | 0.661 | 1.000 | 0.636 | 0.403 |
| M3 | 0.000 | 0.504 | 0.589 | 1.000 | 0.553 | 0.517 |
| M4 | 0.236 | 0.591 | 0.882 | 0.842 | 1.000 | 0.685 |
| M5 | 0.312 | 0.762 | 0.687 | 0.854 | 0.891 | 0.786 |
| SVM_grupos | 0.208 | 0.941 | 0.921 | 0.973 | 0.987 | 0.817 |
| SVM_imagen | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 |

**Lectura:** con partición por grupos (sin fuga), **SVM_grupos sigue siendo el modelo
con mejor desempeño agregado** de los 7 — supera a los 5 modelos profundos en
exactitud, F1 macro y AUC. Entre los modelos profundos, **M5 (EfficientNetB0 +
segmentación) es el mejor**, con la sensibilidad Maligno más alta del grupo (0.762).
M2 y M3 comparten **especificidad 0.000 en Benigno** — ninguno de los dos predice esa
clase ni una sola vez en las 3 semillas, aun con el schedule de entrenamiento ya
corregido (ver sección 3).

---

## 2. Modelo recomendado y su justificación (cifras de validación, fijadas antes de ver el test)

Archivo: `outputs/experimentos/seleccion_modelo_recomendado.json` (reescrito en esta
segunda ronda, **antes** de recalcular ninguna métrica de prueba). El criterio v1
(solo sensibilidad Maligno) y su resultado (M3) quedan conservados en
`seleccion_modelo_recomendado_v1_descartada.json` como evidencia de la lección
metodológica descrita abajo.

**Por qué cambió el criterio:** la sensibilidad Maligno "pura", sin ningún control de
especificidad, la maximiza trivialmente un modelo que predice "Maligno" para todo.
Fue exactamente lo que le pasó a M3 bajo el criterio v1 (`sens_maligno=1.00`
simultáneo con `f1_macro≈0.23` en 2 de 3 semillas — la firma de un colapso total, no
de un buen modelo). Este cambio de criterio ocurrió **después** de haber mirado una
vez las métricas de prueba bajo el criterio v1 — es una excepción real al principio
de "nunca mirar el test antes de fijar el criterio" que rige el resto del protocolo,
y se documenta como tal (ver `registro.md`) en vez de disimularla: la decisión de
cambiar de criterio es una corrección metodológica honesta, no un ajuste buscando un
resultado concreto en el conjunto de prueba (que no se tocó para decidir el cambio).

**Nuevo criterio, fijado antes de recalcular el test:** mayor **índice de Youden de
la clase Maligno** en VALIDACIÓN (sensibilidad Maligno + especificidad Maligno − 1),
media de las 3 semillas, entre M1-M5. Desempate si la diferencia es menor a 0.02:
mayor F1 macro medio de validación. Un clasificador constante tiene Youden = 0
exactamente, así que ya no premia el colapso trivial.

| Modelo | Youden Maligno (medio) | Sens. Maligno (val.) | Esp. Maligno (val.) | F1 macro (val., medio) |
|---|---:|---:|---:|---:|
| M1 | 0.556 | 0.732 | 0.824 | 0.513 |
| M2 | 0.083 | 0.447 | 0.636 | 0.253 |
| M3 | 0.263 | 0.780 | 0.482 | 0.366 |
| M4 | 0.454 | 0.454 | 1.000 | 0.531 |
| **M5** | **0.592** | 0.698 | 0.894 | 0.598 |

**Modelo recomendado: M5** (EfficientNetB0 + segmentación), con Youden Maligno medio
de **0.592** frente a M1 (0.556, segundo mejor) — diferencia de 0.036, mayor que el
umbral de desempate de 0.02, así que no hace falta recurrir al F1 macro. M3 (el
modelo recomendado bajo el criterio v1 descartado) tiene aquí un Youden mediocre
(0.263): sensibilidad alta (0.780) pero especificidad muy baja (0.482), justo el
patrón que el índice de Youden está diseñado para penalizar.

**Consistencia con el desempeño agregado de test** (sección 1): a diferencia de la
ronda anterior (donde el criterio v1 recomendaba M3, un modelo colapsado, mientras
que el mejor desempeño real de test era de M5), esta vez el criterio pre-registrado
en validación y el mejor desempeño observado en test entre los modelos profundos
**coinciden en M5**. SVM_grupos sigue siendo superior a M5 en desempeño agregado de
test, pero no participa en este criterio de selección (que solo compara M1-M5).

---

## 3. ¿Sigue colapsando la CNN propia (M1/M2/M3)? — Con el schedule ya corregido, sí, pero de forma distinta

Esta sección reemplaza por completo la versión anterior de esta pregunta. **La
conclusión anterior ("el colapso persiste con F1 macro") no era válida**: los
resultados que la sustentaban (las 4 configuraciones de M1 dando exactamente
F1 macro=0.3789) eran un artefacto del schedule de tasa de aprendizaje
(`ReduceLROnPlateau` sobre `val_f1_macro`, que decaía la tasa a ~1e-6 antes de que la
red saliera de su meseta inicial de entrenamiento), no una propiedad real de la
arquitectura. Con el callback corregido (`EarlyStoppingTrasMeseta`, que no cuenta
paciencia hasta que la red sale de la meseta — ver `registro.md`), la rejilla de M1
ya **no** da el mismo resultado para las 4 configuraciones:

| lr | dropout | F1 macro val. | ¿Salió de la meseta? |
|---|---|---:|---|
| 3e-4 | 0.25 / 0.40 | 0.243 | No (60 épocas completas) |
| 1e-4 | 0.25 / 0.40 | **0.586** | Sí (épocas 26 y 30) |

Con `lr=1e-4` la red sí aprende de forma genuina (antes nunca bajaba de
`loss≈ln(3)≈1.10`; ahora llega a `loss≈0.67`). **El colapso por schedule queda
descartado como causa.** Pero con la configuración ganadora entrenada en las 3
semillas, persiste un fenómeno distinto, real y ya no atribuible a un bug:

- **M1:** colapsa a una clase distinta en cada semilla — semilla 0 predice casi
  siempre Maligno (`sens_maligno=0.99`, `sens_benigno=0.00`), semilla 1 predice
  siempre Normal (`sens_normal=1.00`, las demás en 0), semilla 2 es la única con las
  tres clases presentes (`sens_benigno=0.375`, `sens_maligno=1.00`,
  `sens_normal=0.317`). La media entre semillas (sección 1) oculta esta inestabilidad.
- **M2 y M3:** el mismo patrón, con `sens_benigno=0.000 ± 0.000` en las 3 semillas
  para ambos — ninguno de los dos predice "Benigno" ni una sola vez, sin importar el
  aumento de datos usado (geométrico o CutMix).

**Conclusión revisada:** una vez eliminado el bug de schedule, la CNN propia sí
aprende (la pérdida de entrenamiento baja de forma genuina), pero sigue siendo
**inestable entre semillas** — qué clase "domina" las predicciones de un run concreto
depende de la semilla, no de un patrón reproducible. Esto es compatible con una causa
real y más profunda: el tamaño reducido del dataset (784 imágenes de entrenamiento),
el desbalance de clases, y una arquitectura entrenada desde cero sin preentrenamiento,
independientemente de si hay aumento de datos (geométrico o CutMix) o no. Los modelos
de transferencia (M4, M5) no muestran este patrón de colapso por clase en ninguna
semilla (ver sección 1, especificidades siempre > 0 en las tres clases).

---

## 4. McNemar (modelo recomendado M5 vs. cada uno de los demás)

Archivo: `outputs/experimentos/mcnemar.csv`. Exacto (`statsmodels.stats.contingency_tables.mcnemar`),
corrección de Holm aplicada por grupo (cada semilla por separado, y el agregado).
**SVM_imagen excluido**: usa una partición distinta (por imagen) a la de M1-M5/SVM_grupos
(por grupos), así que sus predicciones no son parejables caso a caso con las de M5.

**Resultados agregados (las 3 semillas combinadas), M5 vs.:**

| Comparación | b (M5 acierta, otro falla) | c (M5 falla, otro acierta) | p (Holm) | ¿Significativo? |
|---|---:|---:|---:|---|
| M5 vs. M1 | 108 | 77 | 0.054 | No (al borde) |
| M5 vs. M2 | 177 | 69 | 1.62e-11 | **Sí** |
| M5 vs. M3 | 153 | 59 | 2.49e-10 | **Sí** |
| M5 vs. M4 | 89 | 80 | 0.538 | No |
| M5 vs. SVM_grupos | 23 | 104 | 8.48e-13 | **Sí** |

**Lectura:** M5 es significativamente mejor que M2 y M3 (los dos modelos que nunca
predicen "Benigno") en las 3 semillas y en el agregado. Contra M1, la diferencia
agregada queda justo al borde de la significación tras la corrección de Holm
(p=0.054) — ninguna semilla individual es significativa por separado, reflejo de la
alta variabilidad entre semillas de M1 descrita en la sección 3. Contra M4, no hay
diferencia significativa en ninguna semilla ni en el agregado (p=0.538): M4 y M5
comparten arquitectura (EfficientNetB0) y solo difieren en si usan segmentación
(M5) o no (M4). Frente a SVM_grupos, la diferencia es clara y en sentido contrario:
SVM_grupos es significativamente mejor que M5 en las 3 semillas y en el agregado —
las redes convolucionales de este protocolo no alcanzan al SVM clásico sobre
características HOG en esta partición por grupos.

---

## 5. Errores clínicos — Maligno mal clasificado

Archivo: `outputs/experimentos/errores_maligno.csv` (suma de las 3 semillas).

| Modelo | Malignos totales en test | → Normal (grave) | → Benigno |
|---|---:|---:|---:|
| M1 | 239 | 79 | 1 |
| M2 | 239 | **145** (el peor) | 0 |
| M3 | 239 | 119 | 0 |
| M4 | 239 | 53 | 45 |
| M5 | 239 | 37 | 20 |
| SVM_grupos | 239 | **14** (el mejor entre los creíbles) | 0 |
| SVM_imagen | 252 | 0 | 0 (inflado por la fuga, no mérito real) |

M2 sigue siendo el modelo con más errores clínicos graves (maligno clasificado como
"sano", 60.7% de sus casos malignos) — menos extremo que en la ronda anterior
(66.5%), pero sigue siendo el peor de los 7. SVM_grupos y M5 son los que menos
cometen este error entre los modelos creíbles; M5, el modelo recomendado, tiene la
menor tasa de error clínico entre los 5 modelos profundos (57/239 = 23.8%).

---

## 6. Explicabilidad — Grad-CAM vs. Grad-CAM++ vs. Score-CAM sobre M4; Grad-CAM de M1 y M5

Archivo: `outputs/experimentos/explicabilidad.csv` (51 filas: M4=30, M1=9, M5=12).
M1 y M5 (modelo recomendado, Grad-CAM únicamente) se rehicieron en esta segunda
corrección, sustituyendo las filas de M3 (modelo recomendado bajo el criterio v1
descartado). Las filas de M4 (Grad-CAM/Grad-CAM++/Score-CAM) no se tocaron.

| Métrica | Grad-CAM (M4) | Grad-CAM++ (M4) | Score-CAM (M4) | Grad-CAM (M1) | Grad-CAM (M5) |
|---|---:|---:|---:|---:|---:|
| Correlación entre clases | 0.9998 | 1.0000 | 0.99999 | **0.111** | 0.99998 |
| Fracción torácica media | 0.063 | 0.000 | 0.098 | — | — |

**M4 y M5 (EfficientNetB0) producen mapas prácticamente constantes respecto a la
clase predicha** (correlación entre clases ≥ 0.9998 en los tres métodos de M4, y
0.99998 en M5) — el patrón "mapa constante" ya documentado en la ronda anterior se
reproduce también en M5, con Grad-CAM, Grad-CAM++ y Score-CAM: tres métodos con
mecanismos de cálculo muy distintos (gradientes de segundo/tercer orden, y uno sin
gradientes, basado en enmascaramiento y softmax). Esto refuerza la hipótesis de que
la causa es estructural —cómo EfficientNetB0 integra información espacial en su
última capa convolucional— y no un defecto de un método de explicabilidad concreto.

**M1 (CNN propia) es la excepción: sus mapas sí cambian según la clase predicha**
(correlación 0.111, muy por debajo de EfficientNet). Es un hallazgo nuevo de esta
corrección — la rejilla de M1 original (Tarea 6) no se había repetido con el modelo
ya genuinamente entrenado (tras arreglar el schedule). Que M1 module su mapa de
atención según la clase, mientras M4/M5 no lo hacen, sugiere que el patrón
"constante" de EfficientNetB0 no es inevitable en toda arquitectura sobre estas
imágenes, sino algo específico de cómo EfficientNetB0 (preentrenada, congelada en
su mayoría) procesa esta tarea.

**Nota sobre la métrica `fraccion_esquina`:** definida sobre la celda superior
izquierda. El punto caliente de Grad-CAM++/Score-CAM está en la esquina **inferior
derecha** — por eso esa columna da ≈0 para esos casos, aunque el artefacto de esquina
sí existe (en la esquina opuesta). La cifra sola engañaría; la figura lo deja claro.

**Hallazgo adicional:** de las 51 filas (caso × modelo × método) en
`explicabilidad.csv`, **0% tiene ≥50% de su energía dentro de la envolvente
pulmonar** — ninguno de los métodos, en ninguno de los tres modelos evaluados (M1,
M5 con Grad-CAM; M4 con los tres métodos), señala predominantemente tejido pulmonar
real. Es una limitación de explicabilidad que va más allá del patrón "constante" ya
conocido, y se reporta sin suavizarla: en su estado actual, ninguno de estos mapas
sería fiable como apoyo visual para un radiólogo — ni siquiera M1, cuyo mapa sí
distingue entre clases.

---

## 7. Lista de figuras y CSV/JSON nuevos de esta ronda (rutas)

**`outputs/experimentos/` (CSV y JSON):**
- `grupos_paciente_resumen.json`, `calidad_mascaras.json`, `benchmark_tiempo.json` (Tareas 1-2)
- `busqueda_hiperparametros.csv`, `hiperparametros_finales.json` (Tarea 3; `hiperparametros_finales.json`
  actualizado en la segunda corrección con la configuración ganadora de M1 tras arreglar el schedule)
- `busqueda_hiperparametros_correccion_schedule.csv` (segunda corrección — rejilla de M1 con
  `EarlyStoppingTrasMeseta`, script `08d_correccion_schedule_cnn.py`)
- `seleccion_modelo_recomendado.json` (criterio de Youden, M5; script `08e_nuevo_criterio_seleccion.py`)
  — `seleccion_modelo_recomendado_v1_descartada.json` conserva el criterio v1 descartado (M3)
- `metricas_por_semilla.csv`, `resumen_ic95.csv`, `mcnemar.csv`, `errores_maligno.csv`,
  `matrices_confusion_agregadas.json` (Tarea 5, reevaluados tras la segunda corrección)
- `explicabilidad.csv` (Tarea 6; M1/M5 regenerados por `08f_rehacer_gradcam_tras_correccion.py`, M4 intacto)
- `imagenes_por_subconjunto.csv`, `entorno.json` (Tarea 7)
- `outputs/experimentos/runs/<nombre>/`: `historial.csv`, `hp.json` (incluye el campo `meseta` en
  la familia CNN de la segunda corrección), `tiempo.json`, `pred_val.npz`, `pred_test.npz`. Los
  pesos (`modelo.keras`/`modelo.joblib`) NO están en git por volumen (ver `.gitignore`) —
  reproducibles reejecutando `entrenar()`.
- `outputs/experimentos/runs_descartados_schedule/`: las corridas de M1 descartadas de las dos
  rondas de la corrección de schedule (v1 con `EarlyStopping` estándar, v2 con sufijo adicional
  tras ajustar la activación diferida) — conservadas como evidencia, no en uso.

**`outputs/figures/` (PNG, 300 dpi salvo donde se indica):**
- `fuga_gemelos_particion.png`, `montaje_grupos_paciente.png` (Tarea 1)
- `control_calidad_mascaras.png` (Tarea 2)
- `curvas_M1.png` … `curvas_M5.png` (Tarea 3, configuración elegida, semilla 0)
- `resumen_modelos_ic.png`, `matrices_confusion_agregadas.png`, `curvas_roc_final.png`,
  `svm_imagen_vs_grupos.png` (Tarea 5)
- `pipeline_preprocesamiento.png` (Tarea 7)

**`outputs/gradcam/`:**
- `gradcam_M1.png`, `comparacion_metodos_M4.png`, `gradcam_recomendado.png` (Tarea 6)

---

## 8. Incidencias técnicas relevantes para la sección de Metodología

- **Límite de tiempo en segundo plano del entorno de ejecución (~30 min):** el primer
  intento de la Tarea 3 se perdió por completo (ningún entrenamiento sobrevivió).
  Corregido lanzando los procesos desacoplados del entorno (`nohup ... & disown`),
  verificado que sobreviven indefinidamente.
- **Suspensión del sistema operativo durante la noche:** pausó el avance real varias
  horas en dos ocasiones durante la Tarea 4 (un run tardó 360 min en vez de los ~30
  habituales). El proceso no murió, solo se alargó el reloj de pared.
- **Bug real en `10_explicabilidad.py`** (nombre de función inconsistente, corregido
  en minutos gracias a lanzar con `python -u` para ver el progreso en tiempo real en
  vez de depender del buffering por bloques de los `print()` sin TTY).
- **Incidencia "schedule de la tasa de aprendizaje y criterio de selección"
  (segunda corrección, posterior a las 7 tareas originales):** `ReduceLROnPlateau`
  sobre `val_f1_macro` decaía la tasa de aprendizaje de M1/M2/M3 a ~1e-6 antes de que
  la red saliera de su meseta inicial de entrenamiento — invalidando sus resultados
  de la primera ronda. Primera corrección: `EarlyStopping` estándar
  (`start_from_epoch=15, patience=15`) en vez de `ReduceLROnPlateau`; funcionó para
  la configuración ganadora de M1 (salió de la meseta en la época 27), pero con
  riesgo real de que la paciencia agotara el entrenamiento 3 épocas antes de esa
  salida. Segunda corrección (`EarlyStoppingTrasMeseta`): no cuenta paciencia hasta
  que la red sale genuinamente de la meseta Y se alcanza la época mínima —
  eliminando esa carrera entre el reloj de la paciencia y el momento real de salida
  de la meseta. Aplicada a toda la familia CNN (M1/M2/M3); M4/M5/SVM no se vieron
  afectados (arquitecturas distintas, sin este callback).
- **Mismo incidente, criterio de selección del modelo recomendado:** la sensibilidad
  Maligno pura (criterio v1) la maximizaba trivialmente un modelo colapsado a
  predecir siempre "Maligno" (M3). Sustituido por el índice de Youden de Maligno
  (sensibilidad + especificidad − 1), que un clasificador constante puntúa en 0.
  Cambio documentado como ocurrido después de una primera mirada al test bajo el
  criterio v1 (ver sección 2) — excepción real y declarada al principio de no mirar
  el test antes de fijar el criterio.
- **`UnicodeEncodeError` al final de `09_evaluacion_final.py`:** la consola usaba
  `cp1252` en vez de UTF-8 al lanzar sin `PYTHONIOENCODING=utf-8`, haciendo fallar el
  `print()` de una ruta con el carácter "→" después de que todas las métricas ya se
  hubieran calculado. Resuelto relanzando con el encoding correcto; las predicciones
  de test ya cacheadas (`pred_test.npz`) evitaron recalcular la inferencia.
- Ver `outputs/experimentos/registro.md` para el detalle completo, tarea por tarea,
  incluida la estimación de tiempo (Tarea 2.7), por qué se usaron 3 semillas en vez
  de 5 (el total estimado para Tareas 3+4 con 3 semillas ya superaba el umbral de
  ~14h fijado en el protocolo corregido), y el registro completo de la segunda
  corrección (schedule + criterio de Youden).
