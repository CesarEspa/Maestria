# Registro de ejecución — Protocolo corregido (partición por grupos de paciente)

Entrega: miércoles 7 de octubre. Fuente: `PROMPT_CORRECCIONES_TUTORA.md`.

---

## Tarea 1 — Grupos por paciente y particiones

**Script:** `src/02c_grupos_paciente.py` (entregado por la tutora, sin modificar).

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
`_dataset_entrenamiento()`), `F1MacroValidacion` (código de la tutora, sin modificar),
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

**Archivos generados:**
- `src/protocolo.py`
- `outputs/splits/imagenes_224.npz`
- `outputs/splits/imagenes_224_segmentadas.npz`
- `outputs/experimentos/calidad_mascaras.json`
- `outputs/experimentos/benchmark_tiempo.json`
- `outputs/figures/control_calidad_mascaras.png`

---
