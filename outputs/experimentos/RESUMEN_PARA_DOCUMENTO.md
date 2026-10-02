# Resumen para el documento del TFM — Protocolo corregido (partición por grupos de paciente)

Generado al completar las 7 tareas de `PROMPT_CORRECCIONES_TUTORA.md`. Fuente de todos
los números: los CSV/JSON en `outputs/experimentos/` (ver lista de archivos al final).
Semillas usadas: **0, 1, 2** (3, no 5 — ver justificación en "Presupuesto de tiempo"
más abajo). Partición principal: **por grupos de paciente** (83 grupos reconstruidos
por correlación entre cortes consecutivos — ver Tarea 1).

---

## 1. Tabla final — M1-M5 + SVM, test agregado (3 semillas), media ± DE e IC 95% bootstrap

| Modelo | Exactitud | F1 Macro | AUC macro | Sens. Maligno | Sens. Benigno |
|---|---|---|---|---|---|
| M1 (CNN, sin aumento) | 0.502 ± 0.327 [0.455, 0.547] | 0.346 ± 0.229 [0.398, 0.479]* | 0.779 ± 0.066 | 0.426 ± 0.516 [0.362, 0.490] | 0.241 ± 0.417 [0.143, 0.382] |
| M2 (CNN + geométrico) | 0.447 ± 0.113 [0.400, 0.496] | 0.258 ± 0.128 [0.338, 0.449]* | 0.586 ± 0.353 | 0.333 ± 0.577 [0.274, 0.394] | 0.208 ± 0.361 [0.096, 0.321] |
| M3 (CNN + CutMix) | 0.468 ± 0.072 [0.423, 0.515] | 0.212 ± 0.023 [0.281, 0.343]* | 0.614 ± 0.070 | 0.667 ± 0.577 [0.602, 0.725] | **0.000 ± 0.000 [0.000, 0.000]** |
| M4 (EfficientNetB0) | 0.665 ± 0.013 [0.624, 0.707] | 0.558 ± 0.028 [0.515, 0.599] | 0.813 ± 0.048 | 0.591 ± 0.124 [0.527, 0.654] | 0.236 ± 0.105 [0.128, 0.367] |
| M5 (EfficientNetB0 + segmentación) | 0.684 ± 0.042 [0.641, 0.726] | 0.568 ± 0.064 [0.531, 0.623] | 0.827 ± 0.069 | 0.762 ± 0.119 [0.705, 0.817] | 0.312 ± 0.272 [0.174, 0.432] |
| SVM_grupos (HOG, partición por grupos) | **0.857 ± 0.023 [0.825, 0.887]** | **0.675 ± 0.064 [0.645, 0.752]** | **0.881 ± 0.110** | 0.941 ± 0.040 [0.909, 0.969] | 0.208 ± 0.260 [0.109, 0.341] |
| SVM_imagen (HOG, partición por imagen — referencia, CON fuga) | 0.998 ± 0.003 | 0.996 ± 0.007 | 1.000 | 1.000 ± 0.000 | 0.981 ± 0.032 |

*(los intervalos marcados con `*` en F1 Macro reflejan que el IC bootstrap, calculado
sobre las predicciones agregadas, queda por encima de la media ± DE entre semillas
para M1/M2/M3 — señal adicional de la alta variabilidad entre semillas de estos tres
modelos, coherente con el colapso descrito en la sección 3.)*

**Sensibilidad y especificidad completas por clase** (media simple entre semillas,
`metricas_por_semilla.csv`):

| Modelo | Sens.Benigno | Sens.Maligno | Sens.Normal | Esp.Benigno | Esp.Maligno | Esp.Normal |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.241 | 0.426 | 0.667 | 0.729 | 0.982 | 0.594 |
| M2 | 0.208 | 0.333 | 0.667 | 0.955 | 0.794 | 0.333 |
| M3 | 0.000 | 0.667 | 0.333 | 1.000 | 0.333 | 0.667 |
| M4 | 0.236 | 0.591 | 0.882 | 0.842 | 1.000 | 0.685 |
| M5 | 0.312 | 0.762 | 0.687 | 0.854 | 0.891 | 0.786 |
| SVM_grupos | 0.208 | 0.941 | 0.921 | 0.973 | 0.987 | 0.817 |
| SVM_imagen | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 |

**Lectura:** con partición por grupos (sin fuga), **SVM_grupos es el modelo con mejor
desempeño agregado** de los 7 — supera a los 5 modelos profundos en exactitud, F1
macro y AUC. Entre los modelos profundos, **M5 (EfficientNetB0 + segmentación) es el
mejor**, con la sensibilidad Maligno más alta del grupo (0.762) y sin señales de
colapso. M3 destaca por tener **especificidad 0.000 en Benigno con desviación 0.000**
— nunca, en ninguna semilla, predice esa clase (ver sección 3).

---

## 2. Modelo recomendado y su justificación (cifras de validación, fijadas antes de ver el test)

Archivo: `outputs/experimentos/seleccion_modelo_recomendado.json` (escrito el
2026-10-02T10:16:28, **antes** de calcular ninguna métrica de prueba).

**Criterio fijado de antemano:** mayor sensibilidad media sobre la clase Maligno en
VALIDACIÓN (media de las 3 semillas), entre M1-M5. Desempate si la diferencia es
menor a 0.02: mayor F1 macro medio de validación.

**Modelo recomendado: M3** (CNN + CutMix), con sensibilidad Maligno media en
validación de **0.780** (por semilla: 0.341, 1.000, 1.000) frente a M5 (0.698),
SVM no participa en este criterio (solo M1-M5).

**Esto es una paradoja metodológica real, no un error, y se reporta como tal:** al
examinar el detalle por semilla, `sens_maligno=1.00` simultáneo con `f1_macro≈0.23`
en 2 de 3 semillas es la firma de un modelo que **colapsó a predecir "Maligno" para
todo** (sensibilidad perfecta trivial). El criterio de sensibilidad "pura" —sin
ningún control de especificidad o balance entre clases— es, en este caso, vulnerable
al colapso: lo que maximiza la métrica no es el mejor modelo, es el modelo más
degenerado. **No se alteró el criterio ni el resultado tras detectar esto** —
hacerlo violaría el principio central de este protocolo (fijar criterios antes de
ver resultados). Las comparaciones de la sección 4 (McNemar) usan M3 como modelo
recomendado tal como exige el protocolo, y confirman estadísticamente que M3 se
comporta distinto (peor en términos prácticos) que M4, M5 y SVM_grupos.

**Si el criterio se hubiera basado en desempeño agregado de test** (sección 1), el
modelo con mejor comportamiento real habría sido **SVM_grupos**, y entre los modelos
profundos, **M5**. Esta discrepancia entre "lo que el criterio pre-registrado elige"
y "lo que el test agregado muestra como mejor" es en sí misma uno de los hallazgos
metodológicos más valiosos de esta ronda de correcciones.

---

## 3. ¿Sigue colapsando el aumento de datos con la parada por F1 macro? — Sí, y de forma más clara que antes

**Pregunta clave de la corrección #2 del protocolo corregido:** la hipótesis era que
`EarlyStopping(monitor="val_loss")` restauraba los pesos de la época 1, y que vigilar
F1 macro en su lugar evitaría el colapso. **No lo evita.**

- **M1 (CNN, sin aumento):** las 4 configuraciones de la rejilla de hiperparámetros
  (Tarea 3) dieron EXACTAMENTE la misma F1 macro de validación (0.3789) y el mismo
  patrón (`sens_benigno=0.00`), sin importar `lr` ni `dropout_bloques`. La curva de
  aprendizaje (`curvas_M1.png`) muestra F1 macro de validación oscilando entre 0.06 y
  0.38 sin converger, con la época restaurada en la época 1 (la más temprana posible
  tras `start_from_epoch=5`... en este caso incluso antes, en la época 1 literal).
- **M3 (CNN + CutMix):** reproduce esos mismos números de forma IDÉNTICA a M1 en
  validación (Tarea 3), y en TEST (sección 1) tiene `sens_benigno = 0.000 ± 0.000`
  — nunca predijo "Benigno" ni una sola vez en ninguna de las 3 semillas.
- **M2 (CNN + geométrico):** mejora relativamente sobre M1/M3, pero también colapsa
  a predecir "Normal" en 2 de 3 semillas en test (`sens_benigno=0.00`,
  `mal_a_normal=159/239`, el peor registro de errores clínicos de los 7 modelos).

**Conclusión:** el colapso NO dependía únicamente del criterio `monitor="val_loss"`
ni de la fuga de datos por partición incorrecta de rondas anteriores. Persiste con
partición por grupos de paciente y parada por F1 macro. Esto apunta a una causa más
profunda: la interacción entre el tamaño reducido del dataset (784 imágenes de
entrenamiento), el desbalance de clases, y la arquitectura CNN propia entrenada desde
cero — con o sin aumento de datos, geométrico o CutMix. Los modelos de transferencia
(M4, M5) no muestran este patrón de colapso en ninguna semilla.

---

## 4. McNemar (modelo recomendado M3 vs. cada uno de los demás)

Archivo: `outputs/experimentos/mcnemar.csv`. Exacto (`statsmodels.stats.contingency_tables.mcnemar`),
corrección de Holm aplicada por grupo (cada semilla por separado, y el agregado).
**SVM_imagen excluido**: usa una partición distinta (por imagen) a la de M1-M5/SVM_grupos
(por grupos), así que sus predicciones no son parejables caso a caso con las de M3.

**Resultados agregados (las 3 semillas combinadas), M3 vs.:**

| Comparación | b (M3 acierta, otro falla) | c (M3 falla, otro acierta) | p (Holm) | ¿Significativo? |
|---|---:|---:|---:|---|
| M3 vs. M1 | 57 | 73 | 0.376 | No |
| M3 vs. M2 | 79 | 69 | 0.460 | No |
| M3 vs. M4 | 56 | 148 | 2.71e-10 | **Sí** |
| M3 vs. M5 | 66 | 167 | 1.12e-10 | **Sí** |
| M3 vs. SVM_grupos | 8 | 190 | 1.32e-45 | **Sí** |

**Lectura:** M3 es estadísticamente indistinguible de M1 y M2 (los tres colapsan, de
formas distintas según la semilla) pero significativamente peor que M4, M5 y
SVM_grupos — con la diferencia más extrema frente a SVM_grupos.

---

## 5. Errores clínicos — Maligno mal clasificado

Archivo: `outputs/experimentos/errores_maligno.csv` (suma de las 3 semillas).

| Modelo | Malignos totales en test | → Normal (grave) | → Benigno |
|---|---:|---:|---:|
| M1 | 239 | 80 | 57 |
| M2 | 239 | **159** (el peor) | 0 |
| M3 | 239 | 80 | 0 |
| M4 | 239 | 53 | 45 |
| M5 | 239 | 37 | 20 |
| SVM_grupos | 239 | **14** (el mejor entre los creíbles) | 0 |
| SVM_imagen | 252 | 0 | 0 (inflado por la fuga, no mérito real) |

M2 es el modelo con más errores clínicos graves (maligno clasificado como "sano",
66.5% de sus casos malignos). SVM_grupos y M5 son los que menos cometen este error
entre los modelos creíbles.

---

## 6. Explicabilidad — Grad-CAM vs. Grad-CAM++ vs. Score-CAM sobre M4

Archivo: `outputs/experimentos/explicabilidad.csv`. Casos: 2 aciertos + 2 errores por
clase (o los disponibles) sobre M1, M3 (recomendado) y M4, semilla 0.

| Métrica | Grad-CAM (M4) | Grad-CAM++ (M4) | Score-CAM (M4) |
|---|---:|---:|---:|
| Correlación entre clases | 0.9998 | 1.0000 | 0.99999 |
| Fracción torácica media | 0.063 | 0.000 | 0.098 |
| Patrón visual (`comparacion_metodos_M4.png`) | Degradado izquierda-derecha, constante | Punto caliente fijo, esquina inferior derecha | Punto caliente fijo, esquina inferior derecha (igual que Grad-CAM++) |

**Los tres métodos son, en la práctica, constantes respecto al contenido de la
imagen** (correlación entre clases distintas ≥ 0.9998 en los tres) — el patrón "mapa
constante" ya documentado en la ronda anterior del proyecto NO es un artefacto de
Grad-CAM específicamente: se reproduce con Grad-CAM++ y Score-CAM, dos métodos con
mecanismos de cálculo muy distintos (uno basado en gradientes de segundo/tercer
orden, el otro sin gradientes en absoluto, basado en enmascaramiento y softmax de
probabilidades). Esto refuerza con fuerza la hipótesis de que la causa es estructural
—cómo EfficientNetB0 integra información espacial en su última capa convolucional—
y no un defecto de implementación de un método de explicabilidad en particular.

**Nota sobre la métrica `fraccion_esquina`:** definida sobre la celda superior
izquierda. El punto caliente de Grad-CAM++/Score-CAM está en la esquina **inferior
derecha** — por eso esa columna da ≈0 para esos casos, aunque el artefacto de esquina
sí existe (en la esquina opuesta). La cifra sola engañaría; la figura lo deja claro.

**Hallazgo adicional:** en los 42 (caso × modelo × método) evaluados, **0% tiene
≥50% de su energía dentro de la envolvente pulmonar** — ninguno de los métodos, en
ninguno de los dos modelos evaluados (M1/M3 con Grad-CAM, M4 con los tres), señala
predominantemente tejido pulmonar real. Es una limitación de explicabilidad que va
más allá del patrón "constante" ya conocido, y se reporta sin suavizarla: en su
estado actual, ninguno de estos mapas sería fiable como apoyo visual para un
radiólogo.

---

## 7. Lista de figuras y CSV/JSON nuevos de esta ronda (rutas)

**`outputs/experimentos/` (CSV y JSON):**
- `grupos_paciente_resumen.json`, `calidad_mascaras.json`, `benchmark_tiempo.json` (Tareas 1-2)
- `busqueda_hiperparametros.csv`, `hiperparametros_finales.json` (Tarea 3)
- `seleccion_modelo_recomendado.json` (Tarea 4, escrito antes de ver test)
- `metricas_por_semilla.csv`, `resumen_ic95.csv`, `mcnemar.csv`, `errores_maligno.csv`,
  `matrices_confusion_agregadas.json` (Tarea 5)
- `explicabilidad.csv` (Tarea 6)
- `imagenes_por_subconjunto.csv`, `entorno.json` (Tarea 7)
- `outputs/experimentos/runs/<nombre>/`: `historial.csv`, `hp.json`, `tiempo.json`,
  `pred_val.npz`, `pred_test.npz` por cada uno de los 21 (M1-M5 × 3 semillas) + 6
  (SVM × 2 particiones × 3 semillas) entrenamientos. Los pesos (`modelo.keras`/
  `modelo.joblib`) NO están en git por volumen (ver `.gitignore`) — reproducibles
  reejecutando `entrenar()`.

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
- Ver `outputs/experimentos/registro.md` para el detalle completo, tarea por tarea,
  incluida la estimación de tiempo (Tarea 2.7) y por qué se usaron 3 semillas en vez
  de 5 (el total estimado para Tareas 3+4 con 3 semillas ya superaba el umbral de
  ~14h fijado en el prompt).
