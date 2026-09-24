// App del TFM — Cáncer de Pulmón. Vanilla JS, sin build step.
// Habla con el backend FastAPI (api.py) servido en el mismo origen.

const API = ""; // mismo origen: rutas relativas "/api/..."

const STATUS_LABELS = {
  idle: "No entrenado",
  running: "Entrenando…",
  completed: "Completado",
  failed: "Falló",
  stopped: "Detenido",
  stale: "Sin actividad reciente",
};

// Iconos SVG en línea reutilizables (evitan depender de emoji, más
// consistentes visualmente entre sistemas operativos y navegadores).
const ICONS = {
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 5-5"/></svg>`,
  cross: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>`,
  warning: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="1" fill="currentColor" stroke="none"/></svg>`,
  star: `<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><path d="M12 2.5l2.9 6.6 7.1.6-5.4 4.7 1.6 7-6.2-3.8L5.8 21.4l1.6-7L2 9.7l7.1-.6L12 2.5z"/></svg>`,
};
function icon(name, cls = "") {
  return `<span class="inline-icon ${cls}">${ICONS[name]}</span>`;
}

const state = {
  models: [],
  charts: {}, // key -> {loss: Chart, acc: Chart}
  selectedImage: null, // { kind: 'upload'|'sample', file?, label?, index?, previewUrl }
  pollTimer: null,
  openLogs: new Set(), // claves de modelo cuyo log está desplegado (sobrevive al re-render)
  epochOverrides: {}, // key -> épocas elegidas por el usuario (string del <input>)
  trainAll: { active: false, queue: [], current: null, total: 0, done: 0, launchedAt: 0 },
};

// ───────────────────────────── Navegación ─────────────────────────────

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => showPage(btn.dataset.page));
});

function showPage(page) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${page}`));
  if (page === "resultados") loadResults();
  if (page === "clasificar") loadSamplesIfNeeded();
}

// ───────────────────────────── Utilidades ─────────────────────────────

function badgeHtml(status) {
  const label = STATUS_LABELS[status] || status;
  return `<span class="badge badge-${status}"><span class="badge-dot"></span>${label}</span>`;
}

async function apiGet(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function apiPost(path, opts = {}) {
  const res = await fetch(API + path, { method: "POST", ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function fmtPct(x) {
  return x == null ? "—" : (x * 100).toFixed(1) + "%";
}
function fmt4(x) {
  return x == null ? "—" : Number(x).toFixed(4);
}

// ───────────────────────────── Estado de la API ─────────────────────────────

async function checkApiHealth() {
  const el = document.getElementById("api-status");
  try {
    await apiGet("/api/health");
    el.textContent = "API conectada";
    el.className = "api-status ok";
  } catch (e) {
    el.textContent = "API no disponible";
    el.className = "api-status err";
  }
}

// ───────────────────────────── Resumen del dataset (Inicio) ─────────────────────────────

async function loadDatasetSummary() {
  try {
    const data = await apiGet("/api/dataset_summary");
    document.getElementById("dataset-summary-text").innerHTML =
      `<strong>IQ-OTH/NCCD</strong> — ${data.total} imágenes de TC de tórax en 3 clases.`;

    const colors = { Benigno: "#2563a8", Maligno: "#c0392b", Normal: "#1f9d55" };
    const bars = Object.entries(data.per_class).map(([label, n]) => {
      const pct = data.total ? (100 * n / data.total).toFixed(1) : 0;
      return `<div class="dataset-bar-row">
        <span>${label}</span>
        <div class="dataset-bar-track"><div class="dataset-bar-fill" style="width:${pct}%;background:${colors[label] || "var(--primary)"}"></div></div>
        <span>${n} (${pct}%)</span>
      </div>`;
    }).join("");
    document.getElementById("dataset-bars").innerHTML = bars;
  } catch (e) {
    document.getElementById("dataset-summary-text").textContent = "No se pudo cargar el resumen del dataset.";
  }
}

// ───────────────────────────── Modelos: polling ─────────────────────────────

async function pollModels() {
  try {
    state.models = await apiGet("/api/models");
    renderHomeCards();
    renderTrainingCards();
    renderMiniStatus();
    populateModelSelect();
    checkTrainAllProgress();
  } catch (e) {
    // silencioso: se reintenta en el siguiente ciclo
  }
}

function renderMiniStatus() {
  const el = document.getElementById("mini-status");
  el.innerHTML = state.models.map((m) => {
    const dotColor = {
      idle: "var(--idle)", running: "var(--warn)", completed: "var(--ok)",
      failed: "var(--danger)", stopped: "var(--idle)", stale: "var(--warn)",
    }[m.status] || "var(--idle)";
    return `<div class="mini-row"><span>${m.name}</span><span style="color:${dotColor}">●</span></div>`;
  }).join("");
}

function renderHomeCards() {
  const el = document.getElementById("home-model-cards");
  el.innerHTML = state.models.map((m) => `
    <div class="status-mini-card">
      <div class="name">${m.name}</div>
      ${badgeHtml(m.status)}
    </div>
  `).join("");
}

// ───────────────────────────── Página Entrenamiento ─────────────────────────────

function renderTrainingCards() {
  const el = document.getElementById("training-cards");
  el.innerHTML = state.models.map((m) => trainingCardHtml(m)).join("");

  state.models.forEach((m) => {
    const trainBtn = document.getElementById(`train-btn-${m.key}`);
    if (trainBtn) trainBtn.addEventListener("click", () => handleTrain(m.key));
    const stopBtn = document.getElementById(`stop-btn-${m.key}`);
    if (stopBtn) stopBtn.addEventListener("click", () => handleStop(m.key));
    const logToggle = document.getElementById(`log-toggle-${m.key}`);
    if (logToggle) logToggle.addEventListener("click", () => toggleLog(m.key));
    const epochInput = document.getElementById(`epoch-input-${m.key}`);
    if (epochInput) {
      epochInput.addEventListener("input", () => {
        const max = m.default_epochs;
        let v = parseInt(epochInput.value, 10);
        if (!Number.isNaN(v)) {
          if (max != null && v > max) v = max;
          if (v < 1) v = 1;
          if (String(v) !== epochInput.value) epochInput.value = v;
        }
        state.epochOverrides[m.key] = epochInput.value;
      });
    }

    if (m.status === "running" && m.progress) {
      renderCharts(m.key, m.progress.history || []);
    }
  });

  // El log-box se recrea desde cero en cada re-render (cada 3s mientras hay
  // entrenamientos activos); volvemos a aplicarle la clase "open" y a
  // refrescar su contenido para los logs que el usuario dejó desplegados,
  // en vez de que el toggle se pierda en cada ciclo de polling.
  state.openLogs.forEach((key) => {
    const box = document.getElementById(`log-box-${key}`);
    if (!box) { state.openLogs.delete(key); return; }
    box.classList.add("open");
    loadLogContent(key);
  });
}

function trainingCardHtml(m) {
  const p = m.progress || {};
  const running = m.status === "running";
  const total = p.total_epochs || 1;
  const current = p.current_epoch || 0;
  const frac = Math.min(Math.max(current / total, 0), 1) * 100;

  let progressBlock = "";
  if (running) {
    const phaseTxt = p.phase ? ` — ${p.phase}` : "";
    progressBlock = `
      <div class="progress-track">
        <div class="progress-fill" style="width:${frac}%"></div>
        <div class="progress-label">Época ${current}/${total}${phaseTxt}</div>
      </div>`;

    if (p.batch_logs) {
      progressBlock += `<div class="metric-chips">` + Object.entries(p.batch_logs).map(([k, v]) =>
        `<div class="metric-chip">${k}<span class="val">${Number(v).toFixed(4)}</span></div>`
      ).join("") + `</div>`;
    }

    progressBlock += `
      <div class="charts-row">
        <div class="chart-box"><canvas id="chart-loss-${m.key}"></canvas></div>
        <div class="chart-box"><canvas id="chart-acc-${m.key}"></canvas></div>
      </div>`;
  } else if (m.status === "completed" && p.final_val_accuracy != null) {
    const epochsChip = (m.default_epochs != null && p.current_epoch != null)
      ? `<div class="metric-chip">Épocas entrenadas<span class="val">${p.current_epoch}</span></div>`
      : "";
    progressBlock = `<div class="metric-chips">
      ${epochsChip}
      <div class="metric-chip">Accuracy val<span class="val">${fmt4(p.final_val_accuracy)}</span></div>
      ${p.final_val_loss != null ? `<div class="metric-chip">Loss val<span class="val">${fmt4(p.final_val_loss)}</span></div>` : ""}
    </div>`;
  } else if (m.status === "failed") {
    progressBlock = `<div class="banner banner-danger small">Error: ${p.error || "desconocido"}</div>`;
  } else if (m.status === "stopped") {
    progressBlock = `<div class="banner small" style="background:var(--idle-soft);color:var(--text);">Entrenamiento detenido manualmente en la época ${p.current_epoch || "?"}. Puedes reentrenar cuando quieras.</div>`;
  } else if (m.status === "stale") {
    progressBlock = `<div class="banner banner-warning small">El proceso dejó de reportar progreso. Puede haberse detenido o fallado.</div>`;
  }

  const trainLabel = ["completed", "failed", "stopped", "stale"].includes(m.status) ? "Reentrenar" : "Entrenar";
  const trainDisabled = (running || state.trainAll.active) ? "disabled" : "";

  // Campo de épocas compacto (una sola línea, con el detalle en un
  // tooltip) para que quepa junto al botón sin cambiar la altura de la
  // fila — así los 4 botones siguen alineados aunque el SVM no tenga
  // campo de épocas.
  let epochField = "";
  if (!running && m.default_epochs != null) {
    const val = state.epochOverrides[m.key] ?? m.default_epochs;
    const currentModelText = (m.status === "completed" && p.current_epoch != null)
      ? `Modelo actual entrenado con ${p.current_epoch} épocas. `
      : "";
    const tooltip = `${currentModelText}Máximo permitido: ${m.default_epochs} épocas.`;
    epochField = `
      <label class="epoch-field" title="${tooltip}">
        <span class="epoch-field-label">${m.epochs_label || "Épocas"}</span>
        <input type="number" min="1" max="${m.default_epochs}" step="1" value="${val}" id="epoch-input-${m.key}" ${trainDisabled} />
        <span class="epoch-max">/ ${m.default_epochs}</span>
      </label>`;
  }

  const logIsOpen = state.openLogs.has(m.key);

  return `
    <div class="training-card">
      <div class="training-card-top">
        <div>
          <div class="training-card-title">
            <h3>${m.name}</h3>
            ${badgeHtml(m.status)}
          </div>
          <div class="training-card-note">${m.note}</div>
        </div>
        <div class="training-actions">
          ${epochField}
          <button class="btn btn-primary" id="train-btn-${m.key}" ${trainDisabled}>${trainLabel}</button>
          ${running ? `<button class="btn btn-danger" id="stop-btn-${m.key}">Detener</button>` : ""}
        </div>
      </div>
      ${progressBlock}
      <button class="log-toggle" id="log-toggle-${m.key}">Ver log del proceso ${logIsOpen ? "▴" : "▾"}</button>
      <div class="log-box${logIsOpen ? " open" : ""}" id="log-box-${m.key}"></div>
    </div>
  `;
}

function renderCharts(key, history) {
  const lossCanvas = document.getElementById(`chart-loss-${key}`);
  const accCanvas = document.getElementById(`chart-acc-${key}`);
  if (!lossCanvas || !accCanvas || typeof Chart === "undefined") return;

  const epochs = history.map((h) => h.epoch);
  const loss = history.map((h) => h.loss);
  const valLoss = history.map((h) => h.val_loss);
  const acc = history.map((h) => h.accuracy);
  const valAcc = history.map((h) => h.val_accuracy);

  if (state.charts[key]) {
    state.charts[key].loss.destroy();
    state.charts[key].acc.destroy();
  }

  const baseOpts = {
    responsive: true, maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } },
    scales: { x: { ticks: { font: { size: 9 } } }, y: { ticks: { font: { size: 9 } } } },
  };

  state.charts[key] = {
    loss: new Chart(lossCanvas, {
      type: "line",
      data: { labels: epochs, datasets: [
        { label: "loss", data: loss, borderColor: "#2563a8", borderWidth: 2, pointRadius: 0 },
        { label: "val_loss", data: valLoss, borderColor: "#c0392b", borderWidth: 2, pointRadius: 0 },
      ]},
      options: baseOpts,
    }),
    acc: new Chart(accCanvas, {
      type: "line",
      data: { labels: epochs, datasets: [
        { label: "accuracy", data: acc, borderColor: "#0f9b8e", borderWidth: 2, pointRadius: 0 },
        { label: "val_accuracy", data: valAcc, borderColor: "#b8860b", borderWidth: 2, pointRadius: 0 },
      ]},
      options: baseOpts,
    }),
  };
}

function epochsFor(key) {
  const raw = state.epochOverrides[key];
  let n = raw != null ? Number(raw) : null;
  if (!n || n <= 0) return null;
  const m = state.models.find((x) => x.key === key);
  if (m && m.default_epochs != null && n > m.default_epochs) n = m.default_epochs;
  return n;
}

async function handleTrain(key) {
  const btn = document.getElementById(`train-btn-${key}`);
  if (btn) { btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Lanzando…`; }
  try {
    const epochs = epochsFor(key);
    const qs = epochs ? `?epochs=${epochs}` : "";
    await apiPost(`/api/models/${key}/train${qs}`, {});
    await pollModels();
  } catch (e) {
    alert("No se pudo lanzar el entrenamiento: " + e.message);
    await pollModels();
  }
}

async function handleStop(key) {
  const btn = document.getElementById(`stop-btn-${key}`);
  if (btn) { btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Deteniendo…`; }
  try {
    await apiPost(`/api/models/${key}/stop`, {});
    await pollModels();
  } catch (e) {
    alert("No se pudo detener el proceso: " + e.message);
    await pollModels();
  }
}

function toggleLog(key) {
  const box = document.getElementById(`log-box-${key}`);
  const toggleBtn = document.getElementById(`log-toggle-${key}`);
  const isOpen = state.openLogs.has(key);
  if (isOpen) {
    state.openLogs.delete(key);
    box.classList.remove("open");
    if (toggleBtn) toggleBtn.textContent = "Ver log del proceso ▾";
    return;
  }
  state.openLogs.add(key);
  box.classList.add("open");
  if (toggleBtn) toggleBtn.textContent = "Ver log del proceso ▴";
  box.textContent = "Cargando log…";
  loadLogContent(key);
}

// Refresca el contenido de un log ya desplegado sin tocar su estado
// abierto/cerrado. Si el usuario está viendo el final del log, lo sigue
// desplazando hacia abajo (comportamiento tipo "tail -f"); si se desplazó
// hacia arriba para leer algo anterior, respeta su posición de scroll.
async function loadLogContent(key) {
  const box = document.getElementById(`log-box-${key}`);
  if (!box) return;
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  try {
    const data = await apiGet(`/api/models/${key}/log`);
    box.textContent = data.log || "(log vacío)";
  } catch (e) {
    box.textContent = "No se pudo cargar el log.";
  }
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

// ───────────────────────── Entrenar todos (cola secuencial) ─────────────────────────
// En esta máquina (solo CPU) entrenar varios modelos a la vez los haría
// competir por el mismo procesador y tardarían todos más, así que la cola
// lanza un modelo, espera a que termine (o se estanque) y recién entonces
// lanza el siguiente.

document.getElementById("train-all-btn").addEventListener("click", handleTrainAll);
document.getElementById("train-all-cancel-btn").addEventListener("click", cancelTrainAllQueue);

function handleTrainAll() {
  if (state.trainAll.active || !state.models.length) return;
  const keys = state.models.map((m) => m.key);
  state.trainAll = {
    active: true, queue: keys.slice(1), current: keys[0],
    total: keys.length, done: 0, launchedAt: 0,
  };
  renderTrainAllStatus();
  renderTrainingCards();
  launchTrainAllCurrent();
}

function cancelTrainAllQueue() {
  state.trainAll.queue = [];
  state.trainAll.active = false;
  state.trainAll.current = null;
  renderTrainAllStatus();
  renderTrainingCards();
}

async function launchTrainAllCurrent() {
  const key = state.trainAll.current;
  try {
    const epochs = epochsFor(key);
    const qs = epochs ? `?epochs=${epochs}` : "";
    await apiPost(`/api/models/${key}/train${qs}`, {});
  } catch (e) {
    // Puede que ya estuviera entrenándose (409) u otro error puntual: de
    // todos modos lo vigilamos hasta que termine antes de seguir con el próximo.
  }
  state.trainAll.launchedAt = Date.now();
  await pollModels();
}

function checkTrainAllProgress() {
  if (!state.trainAll.active || !state.trainAll.current) return;
  // Margen de gracia tras el lanzamiento para que el subproceso alcance a
  // reportar "running" antes de evaluar si ya terminó (evita avanzar la
  // cola de forma prematura leyendo el estado "idle" previo al arranque).
  if (Date.now() - state.trainAll.launchedAt < 4000) return;
  const m = state.models.find((x) => x.key === state.trainAll.current);
  if (!m) return;
  if (["completed", "failed", "stopped", "stale"].includes(m.status)) {
    advanceTrainAllQueue();
  }
}

function advanceTrainAllQueue() {
  state.trainAll.done += 1;
  if (!state.trainAll.queue.length) {
    state.trainAll.active = false;
    state.trainAll.current = null;
    renderTrainAllStatus();
    renderTrainingCards();
    return;
  }
  state.trainAll.current = state.trainAll.queue.shift();
  renderTrainAllStatus();
  renderTrainingCards();
  launchTrainAllCurrent();
}

function renderTrainAllStatus() {
  const el = document.getElementById("train-all-status");
  const btn = document.getElementById("train-all-btn");
  const cancelBtn = document.getElementById("train-all-cancel-btn");
  if (!state.trainAll.active) {
    el.textContent = "";
    btn.disabled = false;
    btn.textContent = "Entrenar todos los modelos";
    cancelBtn.style.display = "none";
    return;
  }
  btn.disabled = true;
  btn.textContent = "Entrenando todos…";
  cancelBtn.style.display = "";
  const m = state.models.find((x) => x.key === state.trainAll.current);
  const name = m ? m.name : state.trainAll.current;
  el.textContent = `(${state.trainAll.done + 1}/${state.trainAll.total}) ${name}`;
}

// ───────────────────────────── Página Clasificar ─────────────────────────────

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${btn.dataset.tab}`));
  });
});

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) selectUploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) selectUploadFile(fileInput.files[0]);
});

function selectUploadFile(file) {
  state.selectedImage = { kind: "upload", file, previewUrl: URL.createObjectURL(file) };
  showClassifyControls();
}

// Botón "Elegir imagen aleatoria": pide al backend un índice al azar del
// 30% de datos no usado en entrenamiento (validación + test combinados) y
// lo deja listo para clasificar, igual que subir un archivo o elegir un
// ejemplo del dataset.
document.getElementById("random-holdout-btn").addEventListener("click", handleRandomHoldout);

async function handleRandomHoldout() {
  const btn = document.getElementById("random-holdout-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Eligiendo…`;
  try {
    const pick = await apiGet("/api/holdout_random");
    state.selectedImage = {
      kind: "holdout",
      index: pick.index,
      label: pick.true_label,
      previewUrl: `/api/holdout_sample/${pick.index}`,
    };
    showClassifyControls();
  } catch (e) {
    alert("No se pudo elegir una imagen aleatoria: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Elegir imagen aleatoria";
  }
}

let samplesLoaded = false;
async function loadSamplesIfNeeded() {
  if (samplesLoaded) return;
  try {
    const samples = await apiGet("/api/samples");
    const grid = document.getElementById("samples-grid");
    let html = "";
    for (const [label, items] of Object.entries(samples)) {
      html += `<div class="samples-class-title">${label}</div><div class="samples-grid">`;
      html += items.map((it) => `
        <div class="sample-thumb" data-label="${label}" data-index="${it.index}">
          <img src="/api/samples/${encodeURIComponent(label)}/${it.index}" loading="lazy" alt="${label}" />
          <div class="label">${label}</div>
        </div>`).join("");
      html += `</div>`;
    }
    document.getElementById("samples-grid").outerHTML = `<div id="samples-grid">${html}</div>`;
    document.querySelectorAll("#samples-grid .sample-thumb").forEach((el) => {
      el.addEventListener("click", () => {
        const label = el.dataset.label;
        const index = Number(el.dataset.index);
        state.selectedImage = {
          kind: "sample", label, index,
          previewUrl: `/api/samples/${encodeURIComponent(label)}/${index}`,
        };
        showClassifyControls();
      });
    });
    samplesLoaded = true;
  } catch (e) {
    document.getElementById("samples-grid").textContent = "No se pudieron cargar las imágenes de ejemplo.";
  }
}

const SELECTED_IMAGE_LABELS = {
  upload: () => "Imagen subida",
  sample: (img) => `Ejemplo del dataset — clase real: ${img.label}`,
  holdout: (img) => `Aleatoria (30% no usado en entrenamiento) — clase real: ${img.label}`,
};

function showClassifyControls() {
  document.getElementById("classify-controls").style.display = "";
  document.getElementById("selected-image-preview").src = state.selectedImage.previewUrl;
  document.getElementById("selected-image-label").textContent =
    SELECTED_IMAGE_LABELS[state.selectedImage.kind](state.selectedImage);
  document.getElementById("classify-result").style.display = "none";
}

function populateModelSelect() {
  const select = document.getElementById("model-select");
  const trained = state.models.filter((m) => m.status === "completed");
  if (!trained.length) {
    select.innerHTML = `<option value="">Ningún modelo entrenado todavía</option>`;
    return;
  }
  const prevValue = select.value;
  select.innerHTML = trained.map((m) => `<option value="${m.key}">${m.name}</option>`).join("");
  if (trained.some((m) => m.key === prevValue)) select.value = prevValue;
}

document.getElementById("classify-btn").addEventListener("click", handleClassify);

async function handleClassify() {
  const modelKey = document.getElementById("model-select").value;
  if (!modelKey) { alert("No hay ningún modelo entrenado disponible todavía."); return; }
  if (!state.selectedImage) return;

  const btn = document.getElementById("classify-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Clasificando…`;

  try {
    let result;
    if (state.selectedImage.kind === "upload") {
      const form = new FormData();
      form.append("file", state.selectedImage.file);
      result = await apiPost(`/api/classify?model_key=${encodeURIComponent(modelKey)}`, { body: form });
    } else if (state.selectedImage.kind === "holdout") {
      const { index } = state.selectedImage;
      result = await apiPost(
        `/api/classify_holdout_sample?model_key=${encodeURIComponent(modelKey)}&index=${index}`,
        {}
      );
    } else {
      const { label, index } = state.selectedImage;
      result = await apiPost(
        `/api/classify_sample?model_key=${encodeURIComponent(modelKey)}&label=${encodeURIComponent(label)}&index=${index}`,
        {}
      );
    }
    renderClassifyResult(result);
  } catch (e) {
    alert("No se pudo clasificar la imagen: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Clasificar";
  }
}

function renderClassifyResult(result) {
  const el = document.getElementById("classify-result");
  el.style.display = "";

  const matchNote = result.true_label
    ? (result.true_label === result.predicted_class
        ? `<div class="banner" style="background:var(--ok-soft);color:var(--ok);">${icon("check", "banner-icon")}<span>Coincide con la clase real de la imagen de ejemplo.</span></div>`
        : `<div class="banner banner-danger">${icon("cross", "banner-icon")}<span>No coincide con la clase real (${result.true_label}).</span></div>`)
    : "";

  const probBars = Object.entries(result.probabilities).map(([label, p]) => `
    <div class="prob-bar-row">
      <span>${label}</span>
      <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
      <span>${fmtPct(p)}</span>
    </div>
  `).join("");

  let gradcamHtml = "";
  if (result.gradcam_base64) {
    gradcamHtml = `
      <div class="gradcam-row">
        <div><img src="${state.selectedImage.previewUrl}" /><div class="gradcam-caption">Original</div></div>
        <div><img src="data:image/png;base64,${result.gradcam_base64}" /><div class="gradcam-caption">Grad-CAM</div></div>
      </div>`;
  }

  el.innerHTML = `
    <div class="result-header">
      <div class="result-pred">Predicción: ${result.predicted_class}</div>
      <div class="result-confidence">Confianza: ${fmtPct(result.confidence)}</div>
    </div>
    ${matchNote}
    <div class="prob-bars">${probBars}</div>
    ${gradcamHtml}
    <div class="banner banner-warning small" style="margin-top:16px;">
      ${icon("warning", "banner-icon")}
      <span>Recordatorio: esta predicción es generada por un modelo de apoyo (CADx) entrenado sobre un dataset
      académico limitado, y no constituye un diagnóstico médico.</span>
    </div>
  `;
}

// ───────────────────────────── Página Resultados ─────────────────────────────

let resultsLoaded = false;
async function loadResults() {
  if (resultsLoaded) return;
  resultsLoaded = true;
  try {
    const data = await apiGet("/api/results");
    renderResultsTable(data);
  } catch (e) {
    document.getElementById("results-tbody").innerHTML = `<tr><td colspan="7">No se pudieron cargar las métricas.</td></tr>`;
  }
  try {
    const figs = await apiGet("/api/figures");
    renderFigures(figs);
  } catch (e) { /* silencioso */ }
}

function renderResultsTable(data) {
  const { metrics, leakage_warning, leakage_warning_text, best_model } = data;

  document.getElementById("results-warning").innerHTML = leakage_warning
    ? `<div class="banner banner-danger">${icon("warning", "banner-icon")}<span>${leakage_warning_text}</span></div>`
    : "";

  if (!metrics.length) {
    document.getElementById("results-tbody").innerHTML =
      `<tr><td colspan="7" class="muted">Todavía no hay métricas de evaluación.</td></tr>`;
    return;
  }

  document.getElementById("results-tbody").innerHTML = metrics.map((m) => {
    const suspicious = (m.accuracy || 0) >= 0.999;
    const isBest = m.name === best_model;
    const sens = m.sensitivity_per_class || {};
    return `<tr class="${isBest ? "best-row" : ""} ${suspicious ? "suspicious-row" : ""}">
      <td>${m.name}${suspicious ? " " + icon("warning", "inline-icon warn") : ""}${isBest ? " " + icon("star", "inline-icon star") : ""}</td>
      <td>${fmt4(m.accuracy)}</td>
      <td>${fmt4(m.auc_roc)}</td>
      <td>${fmt4(m.f1_macro)}</td>
      <td>${fmt4(sens.Benigno)}</td>
      <td>${fmt4(sens.Maligno)}</td>
      <td>${fmt4(sens.Normal)}</td>
    </tr>`;
  }).join("");
}

function renderFigures(figs) {
  const existing = figs.filter((f) => f.exists);
  document.getElementById("figures-grid").innerHTML = existing.map((f) => `
    <div class="figure-card">
      <img src="/api/figures/${encodeURIComponent(f.filename)}" loading="lazy" alt="${f.caption}" />
      <div class="caption">${f.caption}</div>
    </div>
  `).join("") || `<p class="muted">Aún no se generaron figuras de evaluación.</p>`;

  const gradcamSection = document.getElementById("gradcam-section");
  fetch("/api/gradcam_grid").then((res) => {
    if (res.ok) {
      gradcamSection.innerHTML = `
        <h3 class="section-title">Grad-CAM — mejor modelo</h3>
        <div class="card"><img src="/api/gradcam_grid" style="width:100%;border-radius:8px;" /></div>`;
    }
  }).catch(() => {});
}

// ───────────────────────────── Arranque ─────────────────────────────

checkApiHealth();
loadDatasetSummary();
pollModels();
state.pollTimer = setInterval(pollModels, 3000);
setInterval(checkApiHealth, 15000);
