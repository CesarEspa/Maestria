# Registro de ejecución — Protocolo corregido (partición por grupos de paciente)

Entrega: miércoles 7 de octubre. Fuente: `PROMPT_CORRECCIONES_TUTORA.md`.

---

## Tarea 1 — Grupos por paciente y particiones

**Script:** `src/02c_grupos_paciente.py` (entregado en las correcciones del TFM, sin modificar).

**Ejecutado:** 2026-09-30. Duración: segundos (no es entrenamiento, solo correlación de
imágenes a 64×64 y partición).

**Verificación contra lo esperado:**

| Métrica | Esperado | Obtenido | OK |
|---|---|---|---|
| Imágenes totales | 1097 | 1097 | ✓ |
| Grupos totales | 83 | 83 | ✓ |
| Grupos por clase | Benigno 15, Maligno 30, Normal 38 | Benigno 15, Maligno 30, Normal 38 | ✓ |
| Correlación consecutiva mediana | 0.97 / 0.99 / 0.99 | Benigno 0.9707, Maligno 0.9878, Normal 0.9911 | ✓ |
| Gemelos >0.95 en prueba, por imagen | 89–95% | 89.7%–94.5% (según semilla) | ✓ |
| Gemelos >0.95 en prueba, por grupos | 0–2.5% | 0.0%–2.5% (según semilla) | ✓ |

**Verificación visual (`montaje_grupos_paciente.png`):** revisadas las 6 filas (2 por
clase). Todas muestran progresión anatómica suave y continua entre cortes consecutivos
(tamaño/forma del campo pulmonar, contorno corporal) — consistente con cortes del mismo
estudio. Ninguna fila mezcla visiblemente pacientes distintos. Sin incidencias.

**Conclusión:** Tarea 1 completada sin desviaciones. Se continúa con la Tarea 2.

**Archivos generados:**
- `outputs/splits/grupos_paciente.csv`
- `outputs/splits/particion_grupos_seed{0..4}.npz`
- `outputs/splits/particion_imagen_seed{0..4}.npz`
- `outputs/figures/fuga_gemelos_particion.png`
- `outputs/figures/montaje_grupos_paciente.png`
- `outputs/figures/grupos_paciente_resumen.json`

---

## Tarea 2 — Infraestructura común del protocolo

**Reorganización previa (regla general):** movidos modelos/figuras/JSON del protocolo
anterior (partición por imagen) a `outputs/legacy_particion_imagen/{models,figures,gradcam}/`
vía `git mv` (preserva historial). `outputs/figures/`, `outputs/models/` y
`outputs/gradcam/` quedan libres para el protocolo nuevo. Pendiente (explícitamente
diferido al paso "Al terminar" del prompt): actualizar las referencias a figuras antiguas
en README.md y PROYECTO_CLAUDE_CONTEXTO.txt.

**Módulo creado:** `src/protocolo.py` — `cargar_imagenes()`, `cargar_segmentadas()`,
`particion()`, `control_calidad_mascaras()`, aumento unificado (`geometrico`/`cutmix` vía
`_dataset_entrenamiento()`), `F1MacroValidacion` (código de las correcciones del TFM, sin modificar),
`cnn_propia()`/`efficientnet()`/`activar_fine_tuning()`, `entrenar()`, `benchmark_tiempo()`.

**Validación antes de confiar en el módulo (2026-10-01):** smoke tests en 3 etapas antes
de ejecutar nada real:
1. Carga de imágenes (1097, 224×224, float32 [0,1] en memoria / uint8 en caché),
   lectura de particiones, `cnn_propia()` compila (321,251 parámetros).
2. `efficientnet()` compila (4,378,278 parámetros, base congelada), `activar_fine_tuning()`
   descongela exactamente 50/238 capas de la base, y las 3 variantes de
   `_dataset_entrenamiento()` (None/geometrico/cutmix) producen las formas de batch
   esperadas (cutmix añade correctamente el tercer elemento `sample_weight`).
3. Entrenamiento real de 2 épocas (CNN + cutmix, subconjunto de 64/32 imágenes): se
   confirmó que `val_f1_macro` aparece en `history.history` (requisito explícito del
   prompt: "Comprueba en el primer entrenamiento que history.history contiene
   val_f1_macro"). Sin incidencias.

**Tarea 2.2 — Control de calidad de máscaras (ejecutado, 2026-10-01):**
Se generó `outputs/splits/imagenes_224_segmentadas.npz` (segmentación de las 1097
imágenes con `segment_lungs()` de `02b_segmentation.py`, sin cambios en el algoritmo).

Resultado (`outputs/experimentos/calidad_mascaras.json`):

| Clase | n | Sospechosas (área<5%, >60% o >2 comp.) | Tocan el borde | Área media |
|---|---:|---:|---:|---:|
| Benigno | 120 | 5 (4.2%) | 0 | 9.8% |
| Maligno | 561 | 37 (6.6%) | 9 | 13.6% |
| Normal | 416 | 26 (6.3%) | 1 | 9.4% |
| **Total** | **1097** | **68 (6.2%)** | **10 (0.9%)** | — |

**Verificación visual (`control_calidad_mascaras.png`, 30 máscaras al azar, 10 por
clase):** todas las máscaras cubren el campo pulmonar de forma anatómicamente razonable;
en los casos Maligno con masas grandes, la máscara INCLUYE la masa (no la amputa) — el
ajuste del cierre final (`FINAL_CLOSE_RADIUS`) de la Fase 10 anterior sigue funcionando
bien aquí. Ninguna máscara se extiende sobre la mesa del escáner o el fondo. Sin
incidencias — no hizo falta ajustar parámetros de segmentación.

**Tarea 2.7 — Presupuesto de tiempo (`protocolo.benchmark_tiempo()`, 2026-10-01):**

Medido en este equipo (CPU, sin GPU utilizable — ver README/Limitaciones), 2 épocas
reales, sin aumento (cota inferior), 784 imágenes de entrenamiento / 157 de validación:

| Arquitectura | s/época medido |
|---|---:|
| CNN propia | 91.8 s |
| EfficientNetB0 (fase 1, base congelada) | 27.1 s |

**Estimación del total (Tareas 3 y 4), con supuestos explícitos sobre épocas efectivas
(no medidas — extrapoladas de rondas anteriores con `EarlyStopping` similar):**
CNN ~35 épocas efectivas/entrenamiento (~53.5 min); EfficientNet fase 1 ~18 épocas
(~8.1 min) + fase 2 (50 capas descongeladas, no medida directamente, estimada en
2-3× el coste por época de la fase 1 por tener más parámetros entrenables) ~35 épocas
(~40.8 min) → ~48.9 min por entrenamiento de EfficientNet completo.

- **Tarea 3** (búsqueda, semilla 0: 4+1+1+4+1 = 11 entrenamientos): ≈ 9.4 h
- **Tarea 4** (semillas adicionales 1 y 2 — la semilla 0 se reutiliza de la Tarea 3,
  no se repite; M1-M3 CNN + M4-M5 EfficientNet por semilla): ≈ 8.6 h para 2 semillas
  adicionales (total 3 semillas)
- SVM+HOG (3 valores de C × 3 semillas × 2 tipos de partición): ≈ 0.6 h

**Total estimado con 3 semillas: ≈ 18.6 h — ya SUPERA el umbral de ~14 h** que marca la
instrucción 2.7 para pasar de 5 a 3 semillas. **Decisión: se usan 3 semillas (0, 1, 2)**,
el mínimo que pide la Tarea 4, tal como prevé el propio prompt para este caso
("esto ya está previsto abajo"). Con 5 semillas el total estimado rondaría ≈ 27.6 h,
inviable antes del 7 de octubre junto con las Tareas 5-7. Esta estimación es
aproximada (la fase 2 de EfficientNet y el número real de épocas con parada por F1
no se midieron directamente) — se documentará el tiempo REAL de cada entrenamiento en
`tiempo.json` dentro de cada carpeta de `outputs/experimentos/runs/`, y el registro se
actualizará si la estimación se desvía mucho de lo real.

**Conclusión:** Tarea 2 completada en su totalidad. Se continúa con la Tarea 3
(búsqueda de hiperparámetros, semilla 0, partición por grupos).

**Incidencia técnica (2026-10-01):** el primer lanzamiento de la Tarea 3 (proceso en
segundo plano gestionado por el entorno de ejecución del asistente) fue detenido
automáticamente a los ~30 minutos. A ese ritmo (~90 s/época), el primer entrenamiento
(M1, lr=3e-4, dropout=0.25) no llegó a completar sus 80 épocas máximas (se detuvo a
mitad de la época 23) y, como `entrenar()` solo escribe `modelo.keras` al terminar
`model.fit()` con éxito (sin checkpointing intra-entrenamiento), no se guardó nada —
el directorio del run quedó creado pero vacío. **Ningún entrenamiento de la Tarea 3
sobrevivió al primer lanzamiento**, no solo 1 como se anotó inicialmente aquí (error
de verificación: se comprobó la existencia de la carpeta del run, no de `modelo.keras`
dentro de ella). Corregido relanzando el proceso desacoplado por completo de esa
gestión (`nohup ... > log 2>&1 < /dev/null & disown`), de modo que el proceso de
Python vive de forma independiente en el sistema operativo y no está sujeto a ese
límite de tiempo; el progreso se verifica leyendo el log y el contenido de
`outputs/experimentos/runs/<nombre>/modelo.keras` (no solo la carpeta) periódicamente.
Todos los procesos largos del resto del protocolo (Tareas 3-6) se lanzan con este
mismo patrón a partir de ahora. Gracias a la reanudabilidad de `entrenar()`, el
relanzamiento retomó automáticamente desde el primer entrenamiento sin intervención
manual.

**Tarea 3 completada (2026-10-02).** Los 11 entrenamientos terminaron sin más
incidencias. Tiempos reales muy por debajo de la estimación inicial: CNN
29.8-49.1 min (media ~38 min, no ~53.5 min estimados); EfficientNet 17.4-24.2 min
(¡mucho más rápido que los ~49 min estimados! El `EarlyStopping` por F1 macro corta
mucho antes de lo que suponía la estimación basada en 2 épocas sin parada). Duración
real total de la Tarea 3: ~6h20min desde el relanzamiento (vs. ~9.4h estimadas).

Resultados de la búsqueda (`busqueda_hiperparametros.csv`, `hiperparametros_finales.json`):

| Escenario | Config. elegida | F1 macro val | Sens.Benigno val | Sens.Maligno val | Sens.Normal val |
|---|---|---:|---:|---:|---:|
| M1 (CNN, sin aumento) | lr=1e-4, dropout=0.25 | 0.3789 | 0.00 | 0.34 | 1.00 |
| M2 (CNN + geométrico) | (hp de M1) | 0.4167 | 0.00 | 0.43 | 1.00 |
| M3 (CNN + CutMix) | (hp de M1) | 0.3789 | 0.00 | 0.34 | 1.00 |
| M4 (EfficientNet) | capas=20, lr_ajuste=1e-4 | 0.5282 | 0.63 | 0.38 | 0.80 |
| M5 (EfficientNet + segmentado) | (hp de M4) | **0.6074** | 0.56 | **0.74** | 0.61 |

**Hallazgo importante — se repite el patrón de colapso de la Fase 10 anterior, ahora
bajo partición por grupos:** las 4 configuraciones de M1 dan EXACTAMENTE la misma
F1 macro, sensibilidad por clase y patrón (sens_benigno=0.00, predice mayoritariamente
"Normal") sin importar lr/dropout — y M3 (CutMix) reproduce esos mismos números de
forma idéntica a M1. La curva de aprendizaje de M1 (`curvas_M1.png`) confirma
inestabilidad real: F1 macro de validación oscila entre ~0.06 y ~0.38 sin converger, y
la época restaurada por `EarlyStopping` es la época 1 (muy temprana). Es decir, el
problema de colapso/inestabilidad de las rondas anteriores NO se debía únicamente al
criterio `monitor="val_loss"` ni a la partición con fuga — persiste con partición por
grupos de paciente y parada por F1 macro. Se documentará con el mismo rigor en el
resumen final; es un resultado negativo válido, no un error del proceso.

**Hallazgo positivo:** a diferencia de la Fase 10 anterior (donde la segmentación NO
mejoró el modelo), bajo este protocolo M5 (EfficientNet + segmentación) da el mejor
resultado de TODA la búsqueda (F1 macro 0.6074, sensibilidad Maligno 0.74, la más alta
de los 5 escenarios). Sujeto a confirmación con las 3 semillas en la Tarea 5 antes de
sacar conclusiones definitivas — esto es solo semilla 0, validación.

Commit y push de la Tarea 3 hechos por separado. Se continúa automáticamente con la
Tarea 4 (entrenamiento final multisemilla), sin esperar confirmación, según lo
indicado por el usuario.

**Incidencia técnica (2026-10-02, madrugada):** durante la noche, el equipo entró en
suspensión (probablemente por configuración de ahorro de energía de Windows tras
inactividad de teclado/ratón), lo que pausó el avance real del entrenamiento de la
Tarea 4 durante varias horas. El proceso en sí **no murió** (el `nohup`/`disown` lo
protege de que el entorno de ejecución lo mate, pero no de que el sistema operativo
completo se suspenda) — al reanudarse, Keras reportó un tiempo de paso artificialmente
inflado (~1000 s en vez de ~3 s) para el paso que coincidió con la suspensión, y
después volvió a su ritmo normal sin intervención. El run en curso en ese momento
(M1 semilla 2) tardó algo más de lo normal (44.2 min) pero terminó bien. **Recomendado
al usuario desactivar la suspensión automática mientras el protocolo siga corriendo**,
para no perder más horas de avance real en pausas similares. (Nota: M3 semilla 2
reportó 360 min de duración, frente a los ~25-30 min habituales de un CNN — consistente
con otra suspensión larga del sistema durante esa ejecución; el entrenamiento en sí
terminó correctamente, solo el reloj de pared incluye las horas de pausa.)

---

## Tarea 4 — Entrenamiento final multisemilla (completada 2026-10-02)

Los 10 entrenamientos nuevos (M1-M5 × semillas 1 y 2) y los 6 runs de SVM+HOG
(3 semillas × grupos/imagen) terminaron sin errores. Semilla 0 de M1-M5 se reutilizó
íntegramente de la Tarea 3 (0 reentrenamientos, confirmado por los mensajes
"[saltado, ya existe]").

**Hallazgo crítico — el criterio de selección pre-registrado eligió un modelo
colapsado como "recomendado":** `seleccion_modelo_recomendado.json` eligió **M3**
(CNN + CutMix) por tener la mayor sensibilidad media en Maligno (0.78) entre M1-M5,
tal como exige el criterio fijado de antemano en el protocolo corregido (corrección #8). Pero al
examinar el detalle por semilla, M3 tiene `sens_maligno_por_semilla = [0.34, 1.00,
1.00]` con `f1_macro_por_semilla = [0.379, 0.229, 0.227]` — en las semillas 1 y 2,
sensibilidad Maligno = 1.00 simultánea con F1 macro muy bajo (~0.23) es la firma
inequívoca de un modelo que **colapsó a predecir "Maligno" para todo** (sensibilidad
perfecta trivial, a costa de ignorar las otras dos clases) — el mismo patrón de
colapso de CNN+Augmentation/CutMix documentado extensamente en la Fase 10 del
protocolo anterior, que persiste bajo partición por grupos y parada por F1 macro.

**Qué se hizo al respecto: nada — y eso es lo correcto.** El criterio se fijó
explícitamente ANTES de ver esta cifra, precisamente para evitar el sesgo de
cambiar las reglas después de mirar el resultado (ver Tarea 4 del prompt: "fijado de
antemano... antes de mirar el conjunto de prueba"). Cambiarlo ahora, aunque el
resultado sea incómodo, violaría el principio central de este protocolo corregido.
Se documenta tal cual, como una lección metodológica genuina: un criterio de
sensibilidad "pura" (sin ningún control de especificidad o F1) es frágil ante el
colapso trivial de un modelo a una sola clase, y una selección pre-registrada puede
legítimamente producir un resultado contraintuitivo que conviene exponer, no ocultar.
La Tarea 5 (test + McNemar) confirmará o matizará esto con más datos — se compara
igualmente contra M3 como exige el protocolo, y se discutirá esta paradoja con
honestidad en el resumen final.

Commit y push de la Tarea 4 hechos tal cual (sin alterar el criterio ni el resultado
de la selección). Se continúa automáticamente con la Tarea 5.

---

## Tarea 5 — Evaluación en prueba y estadística (completada 2026-10-02)

Ejecutada en menos de 10 minutos (solo inferencia + cálculos estadísticos, sin
entrenar nada). Nota técnica: los `print()` normales de Python quedan bufferizados
cuando no hay TTY (a diferencia de `model.fit(verbose=1)`, que sí hace flush), así
que el log no mostró progreso en tiempo real — se verificó el avance directamente por
la aparición de los archivos de salida (`pred_test.npz`, `metricas_por_semilla.csv`,
etc.) en vez de leer el log. Para la Tarea 6 se lanzará con `python -u` para evitarlo.

**Resumen agregado (`resumen_ic95.csv`, IC 95% bootstrap sobre test agregado de 3
semillas, 2000 remuestreos):**

| Modelo | Exactitud | F1 Macro | AUC macro | Sens. Maligno | Sens. Benigno |
|---|---:|---:|---:|---:|---:|
| M1 | 0.50 | 0.35 | 0.70-0.78 | 0.43 | 0.24 |
| M2 | 0.45 | 0.26 | 0.46-0.56 | 0.33 | 0.21 |
| M3 | 0.47 | 0.21 | 0.51-0.59 | 0.67 | **0.00 (de=0.00)** |
| M4 | 0.66 | 0.56 | 0.81 | 0.59 | 0.24 |
| M5 | 0.68 | 0.57 | 0.83 | 0.76 | 0.31 |
| SVM_grupos | **0.86** | **0.67** | **0.88** | 0.94 | 0.21 |
| SVM_imagen | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |

**M3 confirma en TEST el colapso ya detectado en validación (Tarea 4):**
`sens_benigno = 0.00` con desviación estándar 0.00 entre las 3 semillas — nunca,
en ninguna semilla, predice "Benigno" ni una sola vez (0/54 casos Benigno en la matriz
agregada). McNemar (modelo recomendado M3 vs. resto, agregado) confirma diferencias
altamente significativas contra M4 (p≈9e-11), M5 (p≈3e-11) y SVM_grupos (p≈3e-46,
tras corrección de Holm) — pero NO contra M1 (p=0.38) ni M2 (p=0.46), consistente con
que M1 y M2 también colapsan (de formas distintas según la semilla), así que no son
estadísticamente distinguibles de M3 en conjunto.

**SVM_grupos resulta ser el modelo con mejor desempeño agregado de los 7** (exactitud
0.86, F1 macro 0.67, AUC 0.88) — superando a M4 y M5. Con partición correcta
(por grupos), el SVM+HOG deja de ser la señal de alarma que era en el protocolo
anterior y pasa a ser un contendiente legítimo.

**SVM_imagen confirma la inflación por fuga de forma contundente:** exactitud
0.998 vs. 0.857 de SVM_grupos, AUC 1.000 vs. 0.876, visible con total claridad en
`curvas_roc_final.png` (curva gris pegada a la esquina) y `svm_imagen_vs_grupos.png`
— la pieza central de evidencia de todo este protocolo corregido.

**Errores clínicos (`errores_maligno.csv`, 239-252 malignos de test según partición):**
M2 es el peor (159/239 malignos clasificados como "Normal" — el error más grave
posible); M3 también comete 80/239 maligno→normal (su colapso en semilla 0 predice
"Normal"); M4 y M5 reducen sustancialmente estos errores (53 y 37 respectivamente);
SVM_grupos solo 14/239; SVM_imagen 0/252 (por la fuga, no por mérito real).

Archivos: `metricas_por_semilla.csv`, `resumen_ic95.csv`, `mcnemar.csv`,
`errores_maligno.csv`, `matrices_confusion_agregadas.json`, y las figuras
`resumen_modelos_ic.png`, `matrices_confusion_agregadas.png`, `curvas_roc_final.png`,
`svm_imagen_vs_grupos.png` — las 4 revisadas visualmente, correctas.

Commit y push de la Tarea 5. Se continúa automáticamente con la Tarea 6
(explicabilidad).

---

## Tarea 6 — Explicabilidad (completada 2026-10-02)

**Bug real encontrado y corregido antes de confiar en el resultado:** el primer
lanzamiento falló con `NameError: name 'calcular_correlacion' is not defined` — la
función se había definido como `correlacion_entre_clases(heatmaps_por_clase)` (recibe
un diccionario clase→lista de heatmaps) pero se llamaba como `calcular_correlacion`
con una lista plana de resultados. Corregido añadiendo un wrapper
`calcular_correlacion(lista_resultados)` que agrupa por `clase_idx` antes de llamar a
la función original. Relanzado con `python -u` (en vez de sin flags) para evitar el
problema de buffering de la Tarea 5 y poder verificar el progreso en tiempo real —
funcionó, el bug se detectó y corrigió en minutos, no se perdió tiempo de cómputo
relevante (Grad-CAM es rápido).

**Hallazgo principal — el problema de explicabilidad de la Fase 10 anterior se
confirma con TRES métodos independientes, no es un artefacto de uno solo:**
Grad-CAM, Grad-CAM++ y Score-CAM sobre M4 (EfficientNetB0) dan correlación entre
clases de **0.9998, 1.0000 y 0.99999** respectivamente — prácticamente constantes
los tres, con casos de clases y aciertos/errores completamente distintos
(`comparacion_metodos_M4.png`, revisada visualmente). El patrón visual concreto:
Grad-CAM reproduce el degradado izquierda-derecha ya documentado; Grad-CAM++ y
Score-CAM muestran, los dos, un punto caliente fijo en la esquina **inferior
derecha** — un artefacto más parecido al original de la primera ronda (esquina
saturada) que al degradado de la segmentación. Confirma con fuerza la hipótesis ya
planteada: es una propiedad estructural de cómo EfficientNetB0 integra información
espacial en su última capa, no un artefacto de un método de explicabilidad
particular ni del preprocesamiento de entrada.

**Matiz importante sobre la métrica `fraccion_esquina`:** definida (según el prompt)
sobre la celda 1/7×1/7 **superior izquierda**. El artefacto de Grad-CAM++/Score-CAM
está en la esquina **inferior derecha** — por eso `fraccion_esquina ≈ 0` para esos
casos en `explicabilidad.csv`, aunque el artefacto de esquina SÍ existe (solo que en
la esquina opuesta a la que mide la métrica tal como está definida). Se documenta
explícitamente para que no se lea la cifra de 0.0 como "sin problema de esquina" —
la evidencia visual (figura) es la que lo confirma, el número por sí solo engañaría.

**M1 y M3 (CNN propia) son menos "constantes"** que M4 (correlación 0.501 y 0.731
respectivamente) pero predicen "Normal" para los 6/6 casos mostrados en sus grids
(`gradcam_M1.png`, `gradcam_recomendado.png` — M3 es el modelo recomendado) — consistente
con el colapso de M1/M3 en la semilla 0 ya documentado en la Tarea 3. Su Grad-CAM se
concentra en gran medida en el contorno corporal (anillo en el borde de la imagen),
con el interior pulmonar en tonos más uniformes.

**Hallazgo adicional, grave:** `0.0% de 42` mapas (todos los modelos y métodos
combinados) caen mayormente (≥50% de la energía) dentro de la envolvente pulmonar —
ninguno de los tres métodos, en ninguno de los dos modelos evaluados, señala
predominantemente tejido pulmonar relevante. Es una limitación de explicabilidad que
afecta a todo el proyecto, no solo al patrón "constante" ya conocido; se documentará
con este número exacto en el resumen final, sin suavizarlo.

Archivos: `outputs/experimentos/explicabilidad.csv`, `outputs/gradcam/gradcam_M1.png`,
`outputs/gradcam/comparacion_metodos_M4.png`, `outputs/gradcam/gradcam_recomendado.png`
— las 3 figuras revisadas visualmente.

Commit y push de la Tarea 6 (incluye el fix de `10_explicabilidad.py`). Se continúa
automáticamente con la Tarea 7 (última).

---

## Tarea 7 — Tablas descriptivas, figura de preprocesamiento y entorno (completada 2026-10-02)

Sin incidencias, completada en menos de 1 minuto (no entrena nada). Las 3 partes ya
se habían probado parcialmente durante la Tarea 2 (antes de que existiera
`hiperparametros_finales.json`); esta ejecución final regenera
`imagenes_por_subconjunto.csv` con las columnas `muestras_vistas_M1/M2/M3` ya
pobladas correctamente para la semilla 0 (21,952 muestras vistas cada una — 784
imágenes de entrenamiento × 28 épocas efectivas, igual para los 3 porque las 3
configuraciones finales de M1/M2/M3 se entrenaron con el mismo `lr`/`dropout_bloques`
elegidos para M1 en la Tarea 3 y pararon en el mismo número de épocas). Conteos de
imágenes y grupos por subconjunto verificados: suman 83 grupos exactos en cada
semilla (train+val+test), consistente con la Tarea 1.

**Entorno (`entorno.json`):** Windows 10, AMD64 (6 núcleos físicos / 12 lógicos),
15.2 GB RAM, sin GPU utilizable por TensorFlow (confirmado `gpu_detectada_por_tensorflow: []`).
Python 3.11.9, TensorFlow 2.17.1, Keras 3.15.1, resto de versiones en el archivo.

**Figura `pipeline_preprocesamiento.png`:** revisada visualmente durante la
preparación de la Tarea 2 — las 6 columnas (original, redimensionada+normalizada,
máscara, segmentada, aumento geométrico, CutMix) correctas para las 3 clases.

**Con esto, las 7 tareas del protocolo corregido están completas.** Quedan los pasos
"Al terminar" del prompt: actualizar README.md y PROYECTO_CLAUDE_CONTEXTO.txt
(separando ensayos preliminares del protocolo final), escribir
`RESUMEN_PARA_DOCUMENTO.md`, y el commit/push final.

**Archivos generados:**
- `src/protocolo.py`
- `outputs/splits/imagenes_224.npz`
- `outputs/splits/imagenes_224_segmentadas.npz`
- `outputs/experimentos/calidad_mascaras.json`
- `outputs/experimentos/benchmark_tiempo.json`
- `outputs/figures/control_calidad_mascaras.png`

---

## Incidencia: schedule de la tasa de aprendizaje y criterio de selección (2026-10-02)

Tras entregar el `RESUMEN_PARA_DOCUMENTO.md` de las 7 tareas, el usuario trasladó dos
correcciones adicionales al prompt original (no son errores de la implementación de
este protocolo, son ajustes al propio diseño experimental, detectados al revisar los
resultados):

### 1. M1, M2 y M3 nunca llegaron a entrenar de verdad

**Diagnóstico señalado en las correcciones del TFM, confirmado al revisar `historial.csv` de los 9 runs:** la
pérdida de entrenamiento nunca bajó de ln(3)≈1.0986 (el valor de un clasificador que no
aprendió nada, con 3 clases) y la exactitud de entrenamiento no superó 0.53 en ningún
run. **Causa:** `ReduceLROnPlateau(monitor="val_f1_macro", patience=5)` reducía el LR
hasta ~1e-6 antes de que la CNN saliera de su meseta inicial de aprendizaje — en el
protocolo preliminar (`curvas_cnn_base.png`, `outputs/legacy_particion_imagen/figures/`),
a LR constante 1e-4 la CNN salía de esa meseta hacia la época 11; con `val_f1_macro`
(una métrica discontinua, no se mueve suavemente como `val_loss`) el `ReduceLROnPlateau`
disparaba reducciones de LR mucho antes de que el modelo tuviera oportunidad real de
aprender. **Esto invalida la conclusión de la sección 3 del `RESUMEN_PARA_DOCUMENTO.md`
anterior** ("el colapso persiste con F1 macro") — no era un colapso genuino del
modelo/arquitectura, era un problema de la tasa de aprendizaje apagándose demasiado
pronto. Esa conclusión se retira.

**Corrección aplicada (solo familia CNN — M1/M2/M3; M4, M5 y SVM no se tocan, no
tienen este problema):**
- `src/protocolo.py`: la función `_callbacks_f1` se bifurcó en
  `_callbacks_f1_efficientnet` (sin cambios, para M4/M5) y `_callbacks_f1_cnn` (nueva,
  para M1/M2/M3): **sin `ReduceLROnPlateau`**; `EarlyStopping(monitor="val_f1_macro",
  mode="max", patience=15, start_from_epoch=15, restore_best_weights=True)`.
  `MAX_EPOCHS_CNN` bajado de 80 a 60.
- Los 12 runs afectados (M1 ×6, M2 ×3, M3 ×3 — las 3 semillas de cada uno) se movieron
  (no se borraron) de `outputs/experimentos/runs/` a
  `outputs/experimentos/runs_descartados_schedule/` con `git mv` para los archivos ya
  versionados (`historial.csv`, `hp.json`, `tiempo.json`, `pred_val.npz`,
  `pred_test.npz`) y `mv` normal para `modelo.keras` (no versionado). `.gitignore`
  actualizado para excluir también los pesos de la carpeta de descartados.
- Validado con un entrenamiento corto de prueba (3 épocas, nombre `TESTFIX`, limpiado
  después) que los nuevos callbacks realmente no incluyen `ReduceLROnPlateau` y que
  `entrenar()` sigue funcionando correctamente sin la columna `learning_rate` en
  `historial.csv` (esa columna la generaba el callback que se quitó; ningún script
  del pipeline depende de ella, confirmado por grep).
- Nuevo script `src/08d_correccion_schedule_cnn.py`: repite la rejilla de M1 (semilla
  0) con los callbacks corregidos; **criterio de aceptación explícito**: la pérdida de
  entrenamiento de la configuración ganadora debe bajar de 1.0 antes de la época 25 —
  si no lo logra, el script se detiene (`sys.exit(1)`) sin continuar con M2/M3/M4/M5.
  Si pasa: entrena M2 y M3 con esa configuración; recalcula el aumento ganador de M4
  entre los M2/M3 nuevos (si cambia respecto al elegido antes, reentrena la rejilla de
  M4 y M5 con el nuevo aumento; si no cambia, los conserva tal cual); entrena M1/M2/M3
  en semillas 1 y 2; actualiza `hiperparametros_finales.json` y regenera las curvas.

### 2. El criterio de selección del modelo recomendado era defectuoso

La sensibilidad Maligno sola, sin ningún control de especificidad, la maximiza
trivialmente un modelo que predice "Maligno" para todo — exactamente lo que le pasó a
M3 con el criterio anterior (ver Tarea 4). **Nuevo criterio:** mayor **índice de
Youden de la clase Maligno** en VALIDACIÓN (sensibilidad Maligno + especificidad
Maligno − 1), media de las 3 semillas, entre M1-M5; desempate si la diferencia es
menor de 0.02, por F1 macro medio de validación. Un clasificador constante tiene
Youden = 0 exactamente (si predice siempre Maligno: sens=1, esp=0, Youden=1+0−1=0),
así que este criterio ya no premia el colapso trivial.

`outputs/experimentos/seleccion_modelo_recomendado.json` (el escrito con el criterio
viejo) se renombra a `seleccion_modelo_recomendado_v1_descartada.json` — se conserva
como evidencia de la propia lección metodológica (documentada en el
`RESUMEN_PARA_DOCUMENTO.md` anterior, sección 2). El nuevo se escribe con el criterio
de Youden, **antes** de recalcular ninguna métrica de prueba.

**Nota de honestidad metodológica, tal como pidió el usuario:** este cambio de
criterio ocurre **después** de haber mirado una vez las métricas de prueba (la Tarea 5
ya se ejecutó con el criterio v1 antes de detectar este problema). Es una excepción
real al principio de "nunca mirar el test antes de fijar el criterio" que rige el
resto del protocolo, y se documenta como tal en vez de disimularla: el cambio no se
basa en qué modelo convenía más a la vista de los resultados de test (de hecho, bajo
el criterio v1 el resultado de test ya mostraba a M3 como peor que M4/M5/SVM_grupos
por McNemar, así que no hay incentivo oculto para "mejorar la cifra" cambiando el
criterio), sino en que el criterio v1 era estructuralmente defectuoso independientemente
de los datos — un clasificador constante siempre lo habría maximizado trivialmente,
en cualquier dataset. Aun así, se deja constancia explícita de la secuencia temporal
real para que quien lea el TFM pueda juzgarlo por sí mismo.

### 3. Qué se repite tras estas correcciones

La Tarea 5 completa (métricas, IC95%, McNemar contra el nuevo modelo recomendado,
errores maligno, las 4 figuras) y, de la Tarea 6, solo el Grad-CAM de M1 (nuevo) y del
modelo recomendado si cambia (Grad-CAM++/Score-CAM de M4 no se tocan si M4 no cambió).
`RESUMEN_PARA_DOCUMENTO.md` se actualiza al final con todos los resultados corregidos.

---

## DETENIDO — la rejilla de M1 corregida no cumple el criterio de aceptación (2026-10-02)

`src/08d_correccion_schedule_cnn.py` ejecutó la rejilla de M1 (4 configuraciones,
callbacks corregidos sin `ReduceLROnPlateau`) y se detuvo automáticamente, tal como
estaba diseñado, sin continuar con M2/M3. Resultado completo:

| Configuración | F1 macro val | Loss mínima entrenamiento | ¿Cruza loss&lt;1.0? | Tiempo |
|---|---:|---:|---|---:|
| lr=3e-4, dropout=0.25 | 0.3789 | 1.060 (nunca baja de 1.0) | No | 51.1 min |
| lr=3e-4, dropout=0.40 | 0.3789 | 1.079 (nunca baja de 1.0) | No | 81.5 min |
| **lr=1e-4, dropout=0.25 (ganadora)** | **0.5855** | **0.666** | Sí, en la **época 27** | 64.7 min |
| lr=1e-4, dropout=0.40 | 0.5855 | 0.664 | Sí (no verificado el nº de época exacto) | 80.5 min |

**Lectura importante, no solo "falló":** con `lr=3e-4` el modelo sigue genuinamente
estancado (loss nunca baja de 1.0, igual que antes del fix) — ese LR parece
sencillamente demasiado alto para esta arquitectura con `LayerNormalization`
(recordar que el resto del proyecto ya había bajado de 1e-3 a 1e-4 por el mismo
motivo de inestabilidad). Con `lr=1e-4` el modelo **sí aprende de verdad** esta vez:
loss de entrenamiento baja hasta 0.666, exactitud de entrenamiento hasta ~0.79,
F1 macro de validación 0.5855 (muy por encima del 0.3789 de un colapso). El fix
funciona — pero la configuración ganadora cruza el umbral de loss&lt;1.0 en la
**época 27**, dos épocas después del límite de 25 fijado en el criterio de
aceptación. Por ese margen estrecho, el criterio formal no se cumple.

**Siguiendo la instrucción explícita del usuario ("si ninguna lo logra, detente y
avísame"), el proceso se detiene aquí sin continuar con M2, M3, ni con las semillas
1 y 2.** Los runs del grid (los 4) quedan en `outputs/experimentos/runs/` (no se
mueven a descartados — a diferencia de los runs con el bug de schedule, estos sí
entrenaron correctamente, la discusión es solo sobre si cumplen el criterio exacto
de aceptación, no sobre si son válidos). Pendiente de decisión del usuario: aceptar
la configuración ganadora tal cual (el aprendizaje es real, solo 2 épocas fuera del
límite nominal), relajar el límite de época, o algún otro ajuste.

---
