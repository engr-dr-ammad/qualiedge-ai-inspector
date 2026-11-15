/* global tf */
// QualiEdge AI Inspector — developed by Ammad Aslam. © 2026 Ammad Aslam. All rights reserved.
import * as db from './db.js';
import { Camera, grabCells, drawOverlay } from './camera.js';
import { loadBackbone, activeBackbone, embed, fileToImage, canvasToDataUrl, BACKBONES } from './features.js';
import { RecipeModel, ensureFeatures, evaluate, featKey, labelOf } from './engine.js';

export const APP_VERSION = '1.0.0';

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? '' : v);
  }
  kids.flat().forEach((c) => c != null && n.append(c));
  return n;
};

const DEFAULT_SETTINGS = {
  rows: 1, cols: 1, roiScale: 0.7, backbone: 'balanced', aug: 4, mirror: false, jig: false,
  passZ: 3, rejectZ: 6, pPass: 0.25, pReject: 0.75, fps: 3, burst: 8, operator: '',
  defectTypes: ['defect'], epochs: 40,
};

const state = {
  recipes: [], recipe: null, samples: [], model: new RecipeModel(),
  batch: null, live: false, busy: false, filter: 'all', lastResults: null, overlayTimer: null,
  tab: 'inspect', addToTraining: true,
};

const camera = new Camera($('video'));

// ---------------- utilities ----------------
let toastTimer;
function toast(msg, ms = 2600) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

function download(name, content, type) {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const a = el('a', { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
}

function flash() {
  const f = $('flash');
  f.classList.add('on');
  requestAnimationFrame(() => requestAnimationFrame(() => f.classList.remove('on')));
}

const fmt = (n, d = 1) => (n === null || n === undefined || Number.isNaN(n) ? '–' : Number(n).toFixed(d));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const safeName = (s) => (s || 'qualiedge').replace(/[^\w.-]+/g, '_').slice(0, 60);

function setProgress(frac) {
  const p = $('trainProgress');
  if (frac === null) { p.hidden = true; return; }
  p.hidden = false;
  p.firstElementChild.style.width = `${Math.round(frac * 100)}%`;
}

// ---------------- recipes ----------------
function newRecipe(name) {
  return { id: db.uid('r_'), name, createdAt: Date.now(), settings: { ...DEFAULT_SETTINGS }, model: null };
}

async function saveRecipe() { await db.put('recipes', state.recipe); }

function renderRecipePicker() {
  const sel = $('recipeSelect');
  sel.replaceChildren(...state.recipes.map((r) => el('option', { value: r.id, text: r.name })));
  sel.value = state.recipe?.id || '';
}

async function loadRecipe(id) {
  stopLive();
  state.recipe = state.recipes.find((r) => r.id === id) || state.recipes[0];
  state.recipe.settings = { ...DEFAULT_SETTINGS, ...state.recipe.settings };
  localStorage.setItem('qe.recipe', state.recipe.id);
  state.samples = (await db.byRecipe('samples', state.recipe.id)).sort((a, b) => a.ts - b.ts);
  state.batch = null;
  const open = (await db.byRecipe('batches', state.recipe.id)).find((b) => !b.closedAt);
  if (open) state.batch = open;
  state.model = new RecipeModel();
  renderRecipePicker();
  renderSettings();
  renderGallery();
  renderBatchBar();
  renderBatches();
  $('resultCard').hidden = true;
  $('evalResult').hidden = true;
  await prepareModel();
}

async function prepareModel() {
  const r = state.recipe;
  setPill('Loading AI…', 'muted');
  const want = r.settings.backbone;
  const got = await loadBackbone(want);
  if (got !== want) toast('Feature network could not be downloaded (offline?). Using classic features for now.', 4500);
  $('engineInfo').textContent = `Engine: TensorFlow.js ${tf.version.tfjs} on ${tf.getBackend()} · features: ${BACKBONES[activeBackbone()].label}`;
  if (r.model && r.model.backbone === activeBackbone()) {
    const ok = await state.model.restore(r, state.samples);
    if (!ok) state.model = new RecipeModel();
  }
  updatePill();
}

function setPill(text, kind, title = text) {
  const p = $('modelPill');
  p.textContent = text;
  p.title = title;
  p.className = `pill pill-${kind}`;
}

function modelOutdated() {
  const info = state.recipe.model;
  if (!info) return false;
  const ids = state.samples.map((s) => s.id).sort().join(',');
  return ids !== info.sampleIds || info.featKey !== featKey(state.recipe);
}

function updatePill() {
  const nGood = state.samples.filter((s) => s.label === 'good').length;
  if (!state.model.ready) {
    if (state.recipe.model) setPill('Retrain', 'warn', 'Model settings or feature network changed: retrain the model');
    else setPill(nGood ? 'Untrained' : 'No model', 'muted', `${state.samples.length} samples, model not trained yet`);
  } else if (modelOutdated()) setPill('Ready · update', 'warn', 'Model works; new or changed samples: retrain to include them');
  else setPill('Ready', 'ok', state.model.head ? 'Model ready: anomaly detector + defect classifier' : 'Model ready: anomaly detector (good parts)');
  $('setupHint').hidden = state.model.ready;
}

// ---------------- camera ----------------
async function startCamera(deviceId) {
  if (!Camera.supported()) {
    toast('This browser cannot access cameras. Use HTTPS and a modern browser, or inspect photo files.', 5000);
    return;
  }
  try {
    const info = await camera.start(deviceId || localStorage.getItem('qe.cam') || undefined);
    localStorage.setItem('qe.camOn', '1');
    if (info.deviceId) localStorage.setItem('qe.cam', info.deviceId);
    $('camPlaceholder').hidden = true;
    const W = camera.width, H = camera.height;
    const wrap = $('videoWrap');
    wrap.style.aspectRatio = `${W} / ${H}`;
    wrap.style.maxWidth = `calc(58vh * ${W / H})`;
    const cams = await camera.list();
    const sel = $('camSelect');
    sel.replaceChildren(...cams.map((c, i) => el('option', { value: c.deviceId, text: c.label || `Camera ${i + 1}` })));
    if (info.deviceId) sel.value = info.deviceId;
    $('torchBtn').hidden = !camera.hasTorch();
    redrawOverlay();
  } catch (err) {
    if (deviceId || localStorage.getItem('qe.cam')) {
      localStorage.removeItem('qe.cam');
      if (deviceId === undefined) return startCamera(null);
    }
    const msg = err.name === 'NotAllowedError' ? 'Camera permission was denied. Allow it in the browser settings.'
      : err.name === 'NotFoundError' ? 'No camera found. You can still inspect photo files.'
        : `Camera error: ${err.message}`;
    toast(msg, 5000);
    $('camPlaceholder').hidden = false;
  }
}

function redrawOverlay() {
  drawOverlay($('overlay'), state.recipe.settings, camera.width, camera.height, state.lastResults);
}

function showResultsOnOverlay(results) {
  state.lastResults = results;
  redrawOverlay();
  clearTimeout(state.overlayTimer);
  if (!state.live) state.overlayTimer = setTimeout(() => { state.lastResults = null; redrawOverlay(); }, 3500);
}

function currentCells() {
  if (!camera.active || !camera.width) {
    toast('Start the camera first, or use a photo file.');
    return null;
  }
  return grabCells(camera.video, camera.width, camera.height, state.recipe.settings);
}

// ---------------- training samples ----------------
async function addSamples(label, canvases, type) {
  const recs = canvases.map((c) => ({
    id: db.uid('s_'), recipeId: state.recipe.id, label,
    type: label === 'good' ? null : (type || 'defect'),
    image: canvasToDataUrl(c, 224, 0.88), ts: Date.now(),
  }));
  await ensureFeatures(state.recipe, recs);
  await db.putMany('samples', recs);
  state.samples.push(...recs);
  renderGallery();
  updatePill();
  return recs.length;
}

async function captureSamples(label) {
  if (state.busy) return;
  const type = label === 'good' ? null : $('defectType').value;
  const shots = $('burstToggle').checked ? state.recipe.settings.burst : 1;
  state.busy = true;
  try {
    let n = 0;
    for (let i = 0; i < shots; i++) {
      const cells = currentCells();
      if (!cells) return;
      flash();
      n += await addSamples(label, cells.map((c) => c.canvas), type);
      if (shots > 1) await sleep(300);
    }
    toast(`Added ${n} ${label === 'good' ? 'good' : type} sample${n === 1 ? '' : 's'}`);
  } catch (err) {
    toast(err.message, 4000);
  } finally { state.busy = false; }
}

async function uploadSamples(files, label) {
  const type = label === 'good' ? null : $('defectType').value;
  let n = 0;
  setProgress(0);
  try {
    for (let i = 0; i < files.length; i++) {
      const img = await fileToImage(files[i]);
      const cells = grabCells(img, img.naturalWidth, img.naturalHeight, { ...state.recipe.settings, roiScale: 1 });
      n += await addSamples(label, cells.map((c) => c.canvas), type);
      setProgress((i + 1) / files.length);
    }
    toast(`Imported ${n} sample${n === 1 ? '' : 's'}`);
  } catch (err) { toast(err.message, 4000); }
  setProgress(null);
}

function renderGallery() {
  const g = $('gallery');
  const list = state.samples.filter((s) => state.filter === 'all' || (state.filter === 'good' ? s.label === 'good' : s.label !== 'good'));
  g.replaceChildren(...list.slice().reverse().map((s) => el('button', {
    class: `thumb ${s.label === 'good' ? 'good' : 'defect'}`, title: s.label === 'good' ? 'Good' : `Defect: ${s.type}`,
    onclick: () => openSample(s),
  }, el('img', { src: s.image, alt: '', loading: 'lazy' }), el('span', { class: 'tag', text: s.label === 'good' ? 'good' : s.type }))));
  $('galleryEmpty').hidden = list.length > 0;
  const nG = state.samples.filter((s) => s.label === 'good').length;
  $('sampleCounts').textContent = `${nG} good · ${state.samples.length - nG} defect`;
}

let dialogSample = null;
function openSample(s) {
  dialogSample = s;
  $('sampleDialogImg').src = s.image;
  $('sampleDialogInfo').textContent = `${s.label === 'good' ? 'Good' : 'Defect: ' + s.type} · added ${new Date(s.ts).toLocaleString()}`;
  $('sampleDialog').showModal();
}

$('sampleDialog').addEventListener('click', async (e) => {
  const act = e.target.dataset?.act;
  if (!act || !dialogSample) return;
  const s = dialogSample;
  if (act === 'delete') {
    await db.del('samples', s.id);
    state.samples = state.samples.filter((x) => x.id !== s.id);
  } else if (act === 'good' || act === 'defect') {
    s.label = act;
    s.type = act === 'good' ? null : $('defectType').value;
    await db.put('samples', s);
  }
  $('sampleDialog').close();
  dialogSample = null;
  renderGallery();
  updatePill();
});

function renderDefectTypes() {
  const sel = $('defectType');
  const cur = sel.value;
  sel.replaceChildren(...state.recipe.settings.defectTypes.map((t) => el('option', { value: t, text: t })));
  if (state.recipe.settings.defectTypes.includes(cur)) sel.value = cur;
}

// ---------------- training & evaluation ----------------
function drawLoss(history) {
  const c = $('lossChart');
  c.hidden = false;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height, pad = 24;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#8b9bab';
  ctx.font = '12px system-ui';
  ctx.fillText('training loss', pad, 16);
  if (history.length < 2) return;
  const max = Math.max(...history), min = Math.min(...history, 0);
  ctx.strokeStyle = '#f5a524';
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((v, i) => {
    const x = pad + (i / (history.length - 1)) * (W - 2 * pad);
    const y = H - pad - ((v - min) / (max - min || 1)) * (H - 2 * pad - 10);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillText(history[history.length - 1].toFixed(3), W - pad - 30, 16);
}

async function featuresForAll(label) {
  $('trainStatus').textContent = `${label}: computing image features…`;
  const changed = await ensureFeatures(state.recipe, state.samples, (f) => setProgress(f));
  if (changed.length) await db.putMany('samples', changed);
}

async function train() {
  if (state.busy) return;
  state.busy = true;
  $('trainBtn').disabled = true;
  $('evalBtn').disabled = true;
  try {
    await loadBackbone(state.recipe.settings.backbone);
    await featuresForAll('Training');
    const losses = [];
    $('lossChart').hidden = true;
    $('trainStatus').textContent = 'Training…';
    const info = await state.model.train(state.recipe, state.samples, {
      onEpoch: (e, logs) => {
        losses.push(logs.loss);
        setProgress((e + 1) / (state.recipe.settings.epochs || 40));
        drawLoss(losses);
        $('trainStatus').textContent = `Training classifier · epoch ${e + 1} · accuracy ${fmt(logs.acc * 100, 0)}%`;
      },
    });
    await state.model.save(state.recipe.id);
    state.recipe.model = info;
    await saveRecipe();
    $('trainStatus').textContent = info.hasHead
      ? `Trained on ${info.nGood} good + ${info.nDefect} defect photos (${info.classes.length - 1} defect type${info.classes.length > 2 ? 's' : ''}). Anomaly detector + classifier active.`
      : `Trained on ${info.nGood} good photos. Anomaly detector active. Add 3+ defect photos to also train the classifier.`;
    toast('Model trained');
  } catch (err) {
    console.error(err);
    $('trainStatus').textContent = err.message;
    toast(err.message, 4500);
  } finally {
    setProgress(null);
    state.busy = false;
    $('trainBtn').disabled = false;
    $('evalBtn').disabled = false;
    updatePill();
  }
}

async function runEvaluation() {
  if (state.busy) return;
  state.busy = true;
  $('evalBtn').disabled = true;
  $('trainBtn').disabled = true;
  try {
    await loadBackbone(state.recipe.settings.backbone);
    await featuresForAll('Evaluating');
    $('trainStatus').textContent = 'Cross-validating (each photo is judged by a model that never saw it)…';
    const r = await evaluate(state.recipe, state.samples, (f) => setProgress(f));
    renderEval(r);
    $('trainStatus').textContent = '';
  } catch (err) {
    $('trainStatus').textContent = err.message;
    toast(err.message, 4000);
  } finally {
    setProgress(null);
    state.busy = false;
    $('evalBtn').disabled = false;
    $('trainBtn').disabled = false;
  }
}

function renderEval(r) {
  const box = $('evalResult');
  box.hidden = false;
  const kpi = (val, label, cls = '') => el('div', { class: `kpi ${cls}` }, el('b', { text: val }), el('span', { text: label }));
  const kids = [el('h2', { text: 'Evaluation (honest hold-out)' })];
  const k = [
    kpi(`${r.good.pass}/${r.good.n}`, 'good parts passed', 'good'),
    kpi(String(r.good.reject), 'good parts wrongly rejected', r.good.reject ? 'bad' : ''),
    kpi(String(r.good.review + r.defect.review), 'sent to human review'),
  ];
  if (r.defect.n) {
    k.push(kpi(`${r.defect.reject}/${r.defect.n}`, 'defects rejected', 'good'));
    k.push(kpi(String(r.defect.pass), 'defects MISSED (passed)', r.defect.pass ? 'bad' : 'good'));
  }
  kids.push(el('div', { class: 'kpis' }, k));
  const table = el('table', {},
    el('tr', {}, el('th', { text: 'Truth \\ decision' }), el('th', { text: 'Pass' }), el('th', { text: 'Review' }), el('th', { text: 'Reject' })),
    el('tr', {}, el('td', { text: `Good (${r.good.n})` }), el('td', { class: 'num', text: r.good.pass }), el('td', { class: 'num', text: r.good.review }), el('td', { class: 'num', text: r.good.reject })),
    r.defect.n ? el('tr', {}, el('td', { text: `Defect (${r.defect.n})` }), el('td', { class: 'num', text: r.defect.pass }), el('td', { class: 'num', text: r.defect.review }), el('td', { class: 'num', text: r.defect.reject })) : null,
  );
  kids.push(table);
  kids.push(el('p', {
    class: 'hint',
    text: r.withHead ? 'Good parts: leave-one-out anomaly score. Classifier: grouped cross-validation.'
      : 'Only good parts available: shows how often a normal part would be flagged. Add defect photos to measure detection.',
  }));
  if (r.suggestedPassZ !== null) {
    const sug = Math.max(1, Math.ceil(r.suggestedPassZ * 2) / 2);
    kids.push(el('div', { class: 'row wrap' },
      el('span', { class: 'muted', text: `Suggested pass limit from your data: ${sug}` }),
      el('button', {
        class: 'btn small', text: 'Apply',
        onclick: async () => {
          state.recipe.settings.passZ = Math.min(8, sug);
          state.recipe.settings.rejectZ = Math.max(state.recipe.settings.rejectZ, state.recipe.settings.passZ + 1);
          await saveRecipe(); renderSettings(); toast('Thresholds updated');
        },
      })));
  }
  box.replaceChildren(...kids);
}

// ---------------- inspection ----------------
async function inspectCells(cells, record) {
  if (!state.model.ready) {
    toast(state.recipe.model ? 'Retrain the model (Train tab) before inspecting.' : 'Train the model first (Train tab).', 3500);
    return null;
  }
  const feats = await embed(cells.map((c) => c.canvas), { wantClassic: !!state.model.classic });
  const preds = state.model.predict(feats, state.recipe.settings);
  const results = preds.map((p, i) => ({ ...p, row: cells[i].row, col: cells[i].col, canvas: cells[i].canvas }));
  renderResult(results);
  if (record) await recordResults(results);
  return results;
}

async function inspectNow() {
  if (state.busy) return;
  const cells = currentCells();
  if (!cells) return;
  state.busy = true;
  try {
    flash();
    const res = await inspectCells(cells, true);
    if (res) showResultsOnOverlay(res);
  } catch (err) { console.error(err); toast(err.message, 4000); } finally { state.busy = false; }
}

async function inspectFile(file) {
  try {
    const img = await fileToImage(file);
    const cells = grabCells(img, img.naturalWidth, img.naturalHeight, { ...state.recipe.settings, roiScale: 1 });
    await inspectCells(cells, true);
  } catch (err) { toast(err.message, 4000); }
}

function verdictOfSet(verdicts) {
  if (!verdicts.length) return null;
  if (verdicts.includes('reject')) return 'reject';
  if (verdicts.includes('review')) return 'review';
  return 'pass';
}

function renderResult(results) {
  $('resultCard').hidden = false;
  const v = verdictOfSet(results.map((r) => r.verdict));
  const vEl = $('resultVerdict');
  vEl.textContent = { pass: 'PASS', review: 'REVIEW', reject: 'REJECT' }[v];
  vEl.className = `verdict ${v}`;
  const nRej = results.filter((r) => r.verdict === 'reject').length;
  $('resultMeta').textContent = results.length > 1
    ? `${results.length} parts · ${nRej} rejected · ${results.filter((r) => r.verdict === 'review').length} for review`
    : `deviation ${fmt(results[0].z)}σ${results[0].pDef !== null ? ` · defect probability ${fmt(results[0].pDef * 100, 0)}%` : ''}`;
  $('resultCells').replaceChildren(...results.map((r) => el('div', { class: `cell ${r.verdict}` },
    el('img', { src: r.canvas.toDataURL('image/jpeg', 0.7), alt: '' }),
    el('div', { class: 'lbl' }, el('b', { text: r.verdict.toUpperCase() }), el('span', { class: 'muted', text: results.length > 1 ? `R${r.row + 1}C${r.col + 1}` : '' })),
    el('div', { class: 'why', text: `dev ${fmt(r.z)}σ${r.pDef !== null ? ` · p ${fmt(r.pDef * 100, 0)}%` : ''}${r.verdict !== 'pass' && r.type ? ` · ${r.type}` : ''}` }))));
}

// ---------------- batches ----------------
function batchStatus(b) {
  const v = verdictOfSet(b.items.map((i) => i.final || i.verdict));
  return v === 'reject' ? 'defective' : v === 'review' ? 'review' : v === 'pass' ? 'ok' : 'empty';
}
const STATUS_TEXT = { defective: 'BATCH HAS DEFECTS', review: 'NEEDS REVIEW', ok: 'BATCH OK', empty: 'Batch open · no parts yet' };

function countsOf(b) {
  const c = { pass: 0, review: 0, reject: 0 };
  b.items.forEach((i) => { c[i.final || i.verdict]++; });
  return c;
}

async function startBatch() {
  const lot = $('lotInput').value.trim() || `Batch ${new Date().toLocaleString()}`;
  state.batch = {
    id: db.uid('b_'), recipeId: state.recipe.id, recipeName: state.recipe.name, lot,
    operator: state.recipe.settings.operator, startedAt: Date.now(), closedAt: null, items: [],
  };
  await db.put('batches', state.batch);
  $('lotInput').value = lot;
  renderBatchBar();
  renderBatches();
}

async function closeBatch() {
  if (!state.batch) return;
  state.batch.closedAt = Date.now();
  await db.put('batches', state.batch);
  const s = batchStatus(state.batch);
  toast(`Batch closed: ${STATUS_TEXT[s]}`);
  state.batch = null;
  $('lotInput').value = '';
  renderBatchBar();
  renderBatches();
}

async function recordResults(results) {
  if (!state.batch) await startBatch();
  results.forEach((r) => state.batch.items.push({
    id: db.uid('i_'), ts: Date.now(), row: r.row, col: r.col, verdict: r.verdict, z: r.z, pDef: r.pDef,
    type: r.verdict === 'pass' ? null : r.type, image: canvasToDataUrl(r.canvas, 224, 0.75), final: null,
  }));
  await db.put('batches', state.batch);
  renderBatchBar();
}

function renderBatchBar() {
  const b = state.batch;
  $('batchToggleBtn').textContent = b ? 'Close batch' : 'Start batch';
  $('batchBar').hidden = !b;
  $('lotInput').disabled = !!b;
  if (!b) return;
  $('lotInput').value = b.lot;
  const c = countsOf(b);
  $('cPass').textContent = c.pass; $('cReview').textContent = c.review; $('cReject').textContent = c.reject;
  const s = batchStatus(b);
  const v = $('batchVerdict');
  v.textContent = STATUS_TEXT[s];
  v.className = `batch-verdict ${s === 'empty' ? '' : s}`;
}

async function renderBatches() {
  const list = (await db.byRecipe('batches', state.recipe.id)).sort((a, b) => b.startedAt - a.startedAt);
  const box = $('batchList');
  if (!list.length) {
    box.replaceChildren(el('div', { class: 'card' }, el('p', { class: 'hint center', text: 'No batches yet. Inspected parts are recorded into batches.' })));
    return;
  }
  box.replaceChildren(...list.map((b) => {
    const s = batchStatus(b);
    const c = countsOf(b);
    return el('div', { class: 'card batch-item', onclick: () => openBatch(b.id) },
      el('div', {}, el('strong', { text: b.lot }),
        el('div', { class: 'meta', text: `${new Date(b.startedAt).toLocaleString()} · ${b.items.length} parts · ${c.pass} pass / ${c.review} review / ${c.reject} reject${b.closedAt ? '' : ' · OPEN'}` })),
      el('span', { class: `batch-verdict ${s === 'empty' ? '' : s}`, text: s === 'empty' ? 'EMPTY' : STATUS_TEXT[s].replace('BATCH ', '') }));
  }));
}

async function openBatch(id) {
  const b = state.batch?.id === id ? state.batch : await db.get('batches', id);
  if (!b) return;
  $('batchList').hidden = true;
  const box = $('batchDetail');
  box.hidden = false;
  const render = () => {
    const s = batchStatus(b);
    const c = countsOf(b);
    box.replaceChildren(
      el('div', { class: 'row wrap no-print' },
        el('button', { class: 'btn small', text: '← Back', onclick: () => { box.hidden = true; $('batchList').hidden = false; renderBatches(); } }),
        el('button', { class: 'btn small', text: 'CSV', onclick: () => exportBatchCsv(b) }),
        el('button', { class: 'btn small', text: 'JSON', onclick: () => download(`${safeName(b.lot)}.json`, JSON.stringify(b, null, 1), 'application/json') }),
        el('button', { class: 'btn small', text: 'Print / PDF', onclick: () => window.print() }),
        !b.closedAt && state.batch?.id === b.id ? el('button', { class: 'btn small', text: 'Close batch', onclick: async () => { await closeBatch(); b.closedAt = Date.now(); render(); } }) : null,
        el('button', {
          class: 'btn small danger', text: 'Delete',
          onclick: async () => {
            if (!confirm(`Delete batch "${b.lot}"?`)) return;
            await db.del('batches', b.id);
            if (state.batch?.id === b.id) { state.batch = null; renderBatchBar(); }
            box.hidden = true; $('batchList').hidden = false; renderBatches();
          },
        })),
      el('div', { class: 'card' },
        el('h2', { text: `Inspection report · ${b.lot}` }),
        el('p', { class: 'muted', text: `Recipe: ${b.recipeName} · Operator: ${b.operator || '–'} · Started ${new Date(b.startedAt).toLocaleString()}${b.closedAt ? ` · Closed ${new Date(b.closedAt).toLocaleString()}` : ' · open'}` }),
        el('div', { class: 'batch-bar' },
          el('div', { class: 'counts' },
            el('span', { class: 'chip pass' }, 'Pass ', el('b', { text: c.pass })),
            el('span', { class: 'chip review' }, 'Review ', el('b', { text: c.review })),
            el('span', { class: 'chip reject' }, 'Reject ', el('b', { text: c.reject }))),
          el('div', { class: `batch-verdict ${s === 'empty' ? '' : s}`, text: STATUS_TEXT[s] })),
        el('label', { class: 'toggle no-print' }, el('input', {
          type: 'checkbox', checked: state.addToTraining, onchange: (e) => { state.addToTraining = e.target.checked; },
        }), 'Add my decisions to the training set'),
        el('p', { class: 'hint', text: 'Tap Good or Defect on any part to confirm or overrule the AI. Uncertain (REVIEW) parts need a decision before the batch can be released.' }),
        el('div', { class: 'cells' }, b.items.map((it, idx) => {
          const v = it.final || it.verdict;
          return el('div', { class: `cell ${v}` },
            el('img', { src: it.image, alt: '' }),
            el('div', { class: 'lbl' }, el('b', { text: `#${idx + 1} ${v.toUpperCase()}` }), el('span', { class: 'muted', text: it.final ? 'operator' : 'AI' })),
            el('div', { class: 'why', text: `dev ${fmt(it.z)}σ${it.pDef !== null && it.pDef !== undefined ? ` · p ${fmt(it.pDef * 100, 0)}%` : ''}` }),
            el('div', { class: 'review-actions no-print' },
              el('button', { class: 'btn pass-btn', text: 'Good', onclick: () => decideItem(b, it, 'pass', render) }),
              el('button', { class: 'btn reject-btn', text: 'Defect', onclick: () => decideItem(b, it, 'reject', render) })));
        }))),
    );
  };
  render();
}

async function decideItem(b, it, final, render) {
  it.final = final;
  await db.put('batches', b);
  if (state.batch?.id === b.id) { state.batch = b; renderBatchBar(); }
  if (state.addToTraining && !it.trained) {
    const img = new Image();
    img.src = it.image;
    await img.decode();
    const c = document.createElement('canvas');
    c.width = 224; c.height = 224;
    c.getContext('2d').drawImage(img, 0, 0, 224, 224);
    await addSamples(final === 'pass' ? 'good' : 'defect', [c], it.type || $('defectType').value || 'defect');
    it.trained = true;
    await db.put('batches', b);
    toast('Saved and added to training set. Retrain to learn from it.');
  }
  render();
}

function exportBatchCsv(b) {
  const head = ['lot', 'recipe', 'operator', 'part', 'time', 'tray_row', 'tray_col', 'ai_verdict', 'final_verdict', 'decided_by', 'deviation_sigma', 'p_defect', 'defect_type'];
  const q = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [head.join(',')];
  b.items.forEach((it, i) => lines.push([
    b.lot, b.recipeName, b.operator, i + 1, new Date(it.ts).toISOString(), it.row + 1, it.col + 1, it.verdict,
    it.final || it.verdict, it.final ? 'operator' : 'ai', fmt(it.z, 2), it.pDef === null || it.pDef === undefined ? '' : fmt(it.pDef, 3), it.type || '',
  ].map(q).join(',')));
  lines.push('');
  lines.push(`${q('batch_status')},${q(STATUS_TEXT[batchStatus(b)])}`);
  download(`${safeName(b.lot)}.csv`, lines.join('\n'), 'text/csv');
}

// ---------------- live mode ----------------
async function liveLoop() {
  while (state.live) {
    const t0 = performance.now();
    if (!state.busy && camera.active && state.model.ready) {
      state.busy = true;
      try {
        const cells = grabCells(camera.video, camera.width, camera.height, state.recipe.settings);
        const res = await inspectCells(cells, false);
        if (res) showResultsOnOverlay(res);
      } catch (err) { console.error(err); } finally { state.busy = false; }
    }
    const wait = 1000 / Math.max(1, state.recipe.settings.fps) - (performance.now() - t0);
    await sleep(Math.max(30, wait));
  }
}

function startLive() {
  if (!state.model.ready) { $('liveToggle').checked = false; toast('Train the model first.'); return; }
  if (!camera.active) { $('liveToggle').checked = false; toast('Start the camera first.'); return; }
  state.live = true;
  liveLoop();
}

function stopLive() {
  state.live = false;
  $('liveToggle').checked = false;
  state.lastResults = null;
  if (state.recipe) redrawOverlay();
}

// ---------------- settings ----------------
const SETTINGS_MAP = [
  ['setRows', 'rows', 'int'], ['setCols', 'cols', 'int'], ['setRoi', 'roiScale', 'float'],
  ['setBackbone', 'backbone', 'str'], ['setAug', 'aug', 'int'], ['setMirror', 'mirror', 'bool'], ['setJig', 'jig', 'bool'],
  ['setPassZ', 'passZ', 'float'], ['setRejectZ', 'rejectZ', 'float'], ['setPPass', 'pPass', 'float'], ['setPReject', 'pReject', 'float'],
  ['setOperator', 'operator', 'str'], ['setFps', 'fps', 'int'], ['setBurst', 'burst', 'int'],
];

function renderSettings() {
  const s = state.recipe.settings;
  $('recipeName').value = state.recipe.name;
  for (const [id, key, kind] of SETTINGS_MAP) {
    if (document.activeElement === $(id)) continue;
    if (kind === 'bool') $(id).checked = !!s[key];
    else $(id).value = s[key];
  }
  $('roiVal').textContent = `${Math.round(s.roiScale * 100)}%`;
  $('augVal').textContent = s.aug;
  $('passZVal').textContent = `${s.passZ}σ`;
  $('rejectZVal').textContent = `${s.rejectZ}σ`;
  $('pPassVal').textContent = `${Math.round(s.pPass * 100)}%`;
  $('pRejectVal').textContent = `${Math.round(s.pReject * 100)}%`;
  $('gridHint').textContent = s.rows * s.cols > 1 ? `Tray mode: ${s.rows} × ${s.cols} = ${s.rows * s.cols} parts per photo` : '';
  renderDefectTypes();
  if (state.recipe) redrawOverlay();
}

for (const [id, key, kind] of SETTINGS_MAP) {
  $(id).addEventListener('change', async (e) => {
    const s = state.recipe.settings;
    let v = kind === 'bool' ? e.target.checked : e.target.value;
    if (kind === 'int') v = Math.max(+e.target.min || 0, Math.min(+e.target.max || 99, parseInt(v, 10) || 0));
    if (kind === 'float') v = parseFloat(v);
    s[key] = v;
    if (key === 'passZ' && s.rejectZ <= v) s.rejectZ = v + 0.5;
    if (key === 'rejectZ' && s.passZ >= v) s.passZ = v - 0.5;
    if (key === 'pPass' && s.pReject <= v) s.pReject = Math.min(0.95, v + 0.25);
    await saveRecipe();
    renderSettings();
    if (['backbone', 'aug', 'mirror', 'jig'].includes(key)) {
      state.model = new RecipeModel();
      await prepareModel();
      if (state.recipe.model) toast('Model settings changed. Retrain the model to apply them.', 3500);
    }
  });
  if ($(id).type === 'range') $(id).addEventListener('input', (e) => {
    const s = { ...state.recipe.settings, [key]: parseFloat(e.target.value) };
    $('roiVal').textContent = `${Math.round(s.roiScale * 100)}%`;
    $('augVal').textContent = s.aug;
    $('passZVal').textContent = `${s.passZ}σ`;
    $('rejectZVal').textContent = `${s.rejectZ}σ`;
    $('pPassVal').textContent = `${Math.round(s.pPass * 100)}%`;
    $('pRejectVal').textContent = `${Math.round(s.pReject * 100)}%`;
    if (key === 'roiScale') drawOverlay($('overlay'), s, camera.width, camera.height, null);
  });
}

$('recipeName').addEventListener('change', async (e) => {
  state.recipe.name = e.target.value.trim() || state.recipe.name;
  await saveRecipe();
  renderRecipePicker();
});

$('deleteRecipeBtn').addEventListener('click', async () => {
  if (!confirm(`Delete recipe "${state.recipe.name}" with all its photos, model and batch reports?`)) return;
  await db.deleteRecipeData(state.recipe.id);
  await RecipeModel.removeSaved(state.recipe.id);
  state.recipes = state.recipes.filter((r) => r.id !== state.recipe.id);
  if (!state.recipes.length) { const r = newRecipe('My part'); await db.put('recipes', r); state.recipes.push(r); }
  await loadRecipe(state.recipes[0].id);
  toast('Recipe deleted');
});

$('exportRecipeBtn').addEventListener('click', () => {
  const data = {
    format: 'qualiedge-recipe', version: 1, app: `QualiEdge AI Inspector ${APP_VERSION}`, exportedAt: new Date().toISOString(),
    recipe: { name: state.recipe.name, settings: state.recipe.settings },
    samples: state.samples.map((s) => ({ label: s.label, type: s.type, image: s.image, ts: s.ts })),
  };
  download(`${safeName(state.recipe.name)}.qerecipe.json`, JSON.stringify(data), 'application/json');
});

$('importRecipe').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (data.format !== 'qualiedge-recipe' || !Array.isArray(data.samples)) throw new Error('Not a QualiEdge recipe file');
    const r = newRecipe(`${data.recipe?.name || 'Imported'}${state.recipes.some((x) => x.name === data.recipe?.name) ? ' (imported)' : ''}`);
    r.settings = { ...DEFAULT_SETTINGS, ...(data.recipe?.settings || {}) };
    await db.put('recipes', r);
    const recs = data.samples.filter((s) => typeof s.image === 'string' && s.image.startsWith('data:image/')).map((s) => ({
      id: db.uid('s_'), recipeId: r.id, label: s.label === 'good' ? 'good' : 'defect',
      type: s.label === 'good' ? null : String(s.type || 'defect').slice(0, 40), image: s.image, ts: s.ts || Date.now(),
    }));
    await db.putMany('samples', recs);
    state.recipes.push(r);
    await loadRecipe(r.id);
    toast(`Imported "${r.name}" with ${recs.length} photos. Train the model to use it.`, 4000);
  } catch (err) { toast(`Import failed: ${err.message}`, 4500); }
});

// ---------------- wiring ----------------
function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.tabbar button').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${tab}`));
  $('stage').hidden = !(tab === 'inspect' || tab === 'train');
  if (tab !== 'inspect') stopLive();
  if (tab === 'batches') { $('batchDetail').hidden = true; $('batchList').hidden = false; renderBatches(); }
  if (!$('stage').hidden) requestAnimationFrame(redrawOverlay);
}

document.querySelectorAll('.tabbar button').forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
document.querySelectorAll('.seg').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.seg').forEach((x) => x.classList.toggle('active', x === b));
  state.filter = b.dataset.filter;
  renderGallery();
}));

$('camStartBtn').addEventListener('click', () => startCamera());
$('camStopBtn').addEventListener('click', () => { stopLive(); camera.stop(); localStorage.removeItem('qe.camOn'); $('camPlaceholder').hidden = false; redrawOverlay(); });
$('camSelect').addEventListener('change', (e) => startCamera(e.target.value));
$('camFlipBtn').addEventListener('click', () => { camera.facing = camera.facing === 'environment' ? 'user' : 'environment'; localStorage.removeItem('qe.cam'); startCamera(null); });
$('torchBtn').addEventListener('click', async () => { try { const on = await camera.toggleTorch(); $('torchBtn').classList.toggle('primary', on); } catch { toast('Light not supported on this camera'); } });

$('addGoodBtn').addEventListener('click', () => captureSamples('good'));
$('addDefectBtn').addEventListener('click', () => captureSamples('defect'));
$('uploadGood').addEventListener('change', (e) => { uploadSamples([...e.target.files], 'good'); e.target.value = ''; });
$('uploadDefect').addEventListener('change', (e) => { uploadSamples([...e.target.files], 'defect'); e.target.value = ''; });
$('newTypeBtn').addEventListener('click', async () => {
  const t = (prompt('Name of the defect type (e.g. scratch, missing clip, crack):') || '').trim().slice(0, 40);
  if (!t) return;
  if (!state.recipe.settings.defectTypes.includes(t)) state.recipe.settings.defectTypes.push(t);
  await saveRecipe();
  renderDefectTypes();
  $('defectType').value = t;
});
$('trainBtn').addEventListener('click', train);
$('evalBtn').addEventListener('click', runEvaluation);

$('inspectBtn').addEventListener('click', inspectNow);
$('inspectFile').addEventListener('change', (e) => { if (e.target.files[0]) inspectFile(e.target.files[0]); e.target.value = ''; });
$('liveToggle').addEventListener('change', (e) => (e.target.checked ? startLive() : stopLive()));
$('batchToggleBtn').addEventListener('click', () => (state.batch ? closeBatch() : startBatch()));

$('recipeSelect').addEventListener('change', (e) => loadRecipe(e.target.value));
$('newRecipeBtn').addEventListener('click', async () => {
  const name = (prompt('Name of the part / product for the new recipe:') || '').trim().slice(0, 60);
  if (!name) return;
  const r = newRecipe(name);
  await db.put('recipes', r);
  state.recipes.push(r);
  await loadRecipe(r.id);
  switchTab('train');
  toast(`Recipe "${name}" created. Add good-part photos.`);
});

window.addEventListener('resize', () => redrawOverlay());
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.code === 'Space' && state.tab === 'inspect') { e.preventDefault(); inspectNow(); }
  if (e.key === 'g' && state.tab === 'train') captureSamples('good');
  if (e.key === 'd' && state.tab === 'train') captureSamples('defect');
});

// ---------------- boot ----------------
async function boot() {
  $('appVersion').textContent = `v${APP_VERSION}`;
  try { await tf.ready(); } catch { await tf.setBackend('cpu'); }
  state.recipes = await db.all('recipes');
  if (!state.recipes.length) {
    const r = newRecipe('My part');
    await db.put('recipes', r);
    state.recipes.push(r);
  }
  await loadRecipe(localStorage.getItem('qe.recipe'));
  if (localStorage.getItem('qe.camOn') && Camera.supported()) startCamera();
  if ('serviceWorker' in navigator && location.protocol === 'https:') {
    navigator.serviceWorker.register('sw.js').catch((e) => console.warn('SW', e));
  }
  window.qualiedge = { state, camera, APP_VERSION };
}

boot().catch((err) => { console.error(err); toast(`Start-up error: ${err.message}`, 8000); });
