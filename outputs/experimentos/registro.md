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
