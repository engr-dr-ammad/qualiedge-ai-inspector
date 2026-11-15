/* global tf */
// QualiEdge decision engine.
// 1) Anomaly channel: k-nearest-neighbour distance to GOOD parts in feature space, normalised by
//    how far good parts are from each other (leave-one-out, robust median/MAD). Needs only good parts.
// 2) Supervised channel (when defect examples exist): a small neural head trained on the
//    embeddings, with one class per defect type.
// 3) Three-way decision: PASS / REVIEW / REJECT. Uncertain parts go to a human instead of a guess.
import { embed, augment, loadImage, hashString, activeBackbone } from './features.js';

const K = 3;

function cosDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return 1 - s;
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export class KnnDetector {
  // refs: [{ v: Float32Array, g: groupId }]; LOO normalisation excludes the whole group
  // (a photo and its augmented copies), so the spread reflects real part-to-part variation.
  fit(refs, origIdx) {
    this.refs = refs;
    this.k = Math.max(1, Math.min(K, new Set(refs.map((r) => r.g)).size - 1));
    const loo = origIdx.map((i) => this._dist(refs[i].v, refs[i].g));
    this.mu = median(loo);
    const mad = median(loo.map((d) => Math.abs(d - this.mu))) * 1.4826;
    this.sigma = Math.max(mad, 0.15 * this.mu, 1e-4);
    this.loo = loo;
    return this;
  }

  _dist(v, excludeGroup) {
    const ds = [];
    for (const r of this.refs) if (r.g !== excludeGroup) ds.push(cosDist(v, r.v));
    ds.sort((a, b) => a - b);
    const k = Math.min(this.k, ds.length);
    let s = 0;
    for (let i = 0; i < k; i++) s += ds[i];
    return s / Math.max(1, k);
  }

  score(v, excludeGroup = null) {
    const d = this._dist(v, excludeGroup);
    return (d - this.mu) / this.sigma;
  }
}

function buildHead(dim, nClasses) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [dim], units: 64, activation: 'relu', kernelRegularizer: tf.regularizers.l2({ l2: 1e-3 }) }));
  m.add(tf.layers.dropout({ rate: 0.3 }));
  m.add(tf.layers.dense({ units: nClasses, activation: 'softmax' }));
  m.compile({ optimizer: tf.train.adam(1e-3), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

// Embeddings are unit-length; rescale so each entry has roughly unit variance for the dense head.
function headInput(vectors) {
  const dim = vectors[0].length;
  return tf.tensor2d(vectors.map((v) => Array.from(v))).mul(Math.sqrt(dim));
}

async function fitHead(X, y, classes, { epochs = 40, onEpoch } = {}) {
  const dim = X[0].length;
  const head = buildHead(dim, classes.length);
  const counts = classes.map((_, c) => y.filter((v) => v === c).length);
  const classWeight = {};
  counts.forEach((n, c) => { classWeight[c] = n ? X.length / (classes.length * n) : 1; });
  const xs = headInput(X);
  const ys = tf.oneHot(tf.tensor1d(y, 'int32'), classes.length).toFloat();
  await head.fit(xs, ys, {
    epochs, batchSize: 16, shuffle: true, classWeight, verbose: 0,
    callbacks: { onEpochEnd: async (e, logs) => { onEpoch?.(e, logs); await tf.nextFrame(); } },
  });
  xs.dispose(); ys.dispose();
  return head;
}

export function labelOf(sample) {
  return sample.label === 'good' ? 'good' : 'defect:' + (sample.type || 'defect');
}

// ---------- feature cache (stored on the sample record) ----------
export function featKey(recipe) {
  return `${activeBackbone()}|a${recipe.settings.aug}|m${recipe.settings.mirror ? 1 : 0}`;
}

// Ensures every sample has embeddings for the current backbone + augmentation settings.
// Returns the samples whose cache changed (to be saved by the caller).
export async function ensureFeatures(recipe, samples, onProgress) {
  const key = featKey(recipe);
  const todo = samples.filter((s) => !s.feat || s.feat.key !== key);
  for (let i = 0; i < todo.length; i++) {
    const s = todo[i];
    const img = await loadImage(s.image);
    const base = document.createElement('canvas');
    base.width = 224; base.height = 224;
    base.getContext('2d').drawImage(img, 0, 0, 224, 224);
    const views = [base, ...augment(img, recipe.settings.aug, recipe.settings.mirror, hashString(s.id))];
    const f = await embed(views, { wantClassic: true });
    s.feat = { key, deep: f.deep, classic: f.classic };
    onProgress?.((i + 1) / todo.length);
  }
  return todo;
}

// ---------- the trained recipe model ----------
export class RecipeModel {
  constructor() { this.deep = null; this.classic = null; this.head = null; this.classes = ['good']; this.info = null; }

  get ready() { return !!this.deep; }

  static buildDetectors(samples, useJig) {
    const goods = samples.filter((s) => s.label === 'good');
    const refsD = [], refsC = [], origD = [], origC = [];
    goods.forEach((s) => {
      s.feat.deep.forEach((v, j) => { if (j === 0) origD.push(refsD.length); refsD.push({ v, g: s.id }); });
      if (useJig) s.feat.classic.forEach((v, j) => { if (j === 0) origC.push(refsC.length); refsC.push({ v, g: s.id }); });
    });
    const deep = new KnnDetector().fit(refsD, origD);
    const classic = useJig ? new KnnDetector().fit(refsC, origC) : null;
    return { deep, classic };
  }

  static classesFor(samples) {
    const types = [...new Set(samples.filter((s) => s.label !== 'good').map(labelOf))].sort();
    return ['good', ...types];
  }

  static canTrainHead(samples) {
    const nGood = samples.filter((s) => s.label === 'good').length;
    const nDef = samples.filter((s) => s.label !== 'good').length;
    return nGood >= 5 && nDef >= 3;
  }

  static async trainHeadOn(samples, classes, opts) {
    const X = [], y = [];
    samples.forEach((s) => {
      const c = classes.indexOf(labelOf(s));
      if (c < 0) return;
      s.feat.deep.forEach((v) => { X.push(v); y.push(c); });
    });
    return fitHead(X, y, classes, opts);
  }

  async train(recipe, samples, { onEpoch } = {}) {
    const goods = samples.filter((s) => s.label === 'good');
    if (goods.length < 5) throw new Error(`Need at least 5 good samples (have ${goods.length}). 10 or more is better.`);
    const { deep, classic } = RecipeModel.buildDetectors(samples, recipe.settings.jig);
    this.deep = deep; this.classic = classic;
    this.head?.dispose();
    this.head = null;
    this.classes = RecipeModel.classesFor(samples);
    if (RecipeModel.canTrainHead(samples)) {
      this.head = await RecipeModel.trainHeadOn(samples, this.classes, { epochs: recipe.settings.epochs || 40, onEpoch });
    }
    this.info = {
      trainedAt: Date.now(), backbone: activeBackbone(), nGood: goods.length,
      nDefect: samples.length - goods.length, classes: this.classes, hasHead: !!this.head,
      featKey: featKey(recipe), sampleIds: samples.map((s) => s.id).sort().join(','),
    };
    return this.info;
  }

  async save(recipeId) {
    if (this.head) await this.head.save(`indexeddb://qualiedge-head-${recipeId}`);
    else await RecipeModel.removeSaved(recipeId);
  }

  static async removeSaved(recipeId) {
    try { await tf.io.removeModel(`indexeddb://qualiedge-head-${recipeId}`); } catch { /* none saved */ }
  }

  // Restore from stored info + cached sample features (no retraining needed).
  async restore(recipe, samples) {
    const info = recipe.model;
    if (!info) return false;
    const used = samples.filter((s) => info.sampleIds.split(',').includes(s.id) && s.feat?.key === info.featKey);
    if (used.filter((s) => s.label === 'good').length < 5) return false;
    const { deep, classic } = RecipeModel.buildDetectors(used, recipe.settings.jig);
    this.deep = deep; this.classic = classic;
    this.classes = info.classes;
    this.head = null;
    if (info.hasHead) {
      try { this.head = await tf.loadLayersModel(`indexeddb://qualiedge-head-${recipe.id}`); } catch { this.head = null; }
    }
    this.info = info;
    return true;
  }

  // feats: { deep: Float32Array[], classic: Float32Array[] } for N items
  predict(feats, settings, exclude = null) {
    const n = feats.deep.length;
    let probs = null;
    if (this.head) {
      probs = tf.tidy(() => this.head.predict(headInput(feats.deep)).arraySync());
    }
    const results = [];
    for (let i = 0; i < n; i++) {
      const zDeep = this.deep.score(feats.deep[i], exclude);
      const zPix = this.classic && feats.classic ? this.classic.score(feats.classic[i], exclude) : null;
      const z = zPix === null ? zDeep : Math.max(zDeep, zPix);
      let pDef = null, type = null;
      if (probs) {
        pDef = 1 - probs[i][0];
        let best = 1;
        for (let c = 2; c < probs[i].length; c++) if (probs[i][c] > probs[i][best]) best = c;
        type = this.classes[best]?.replace(/^defect:/, '') || null;
      }
      results.push({ z, zDeep, zPix, pDef, type, verdict: decide(z, pDef, settings) });
    }
    return results;
  }
}

export function decide(z, pDef, s) {
  if (pDef === null || pDef === undefined) {
    if (z <= s.passZ) return 'pass';
    if (z >= s.rejectZ) return 'reject';
    return 'review';
  }
  if (pDef >= s.pReject || z >= s.rejectZ) return 'reject';
  if (pDef <= s.pPass && z <= s.passZ) return 'pass';
  return 'review';
}

// ---------- evaluation: leave-one-out (anomaly) + grouped k-fold (classifier) ----------
export async function evaluate(recipe, samples, onProgress) {
  const goods = samples.filter((s) => s.label === 'good');
  if (goods.length < 6) throw new Error('Need at least 6 good samples to evaluate.');
  const settings = recipe.settings;
  const withHead = RecipeModel.canTrainHead(samples);
  const classes = RecipeModel.classesFor(samples);
  const pDefById = {};

  if (withHead) {
    const folds = Math.min(5, samples.filter((s) => s.label !== 'good').length);
    const order = [...samples].sort((a, b) => hashString(a.id) - hashString(b.id));
    const goodsO = order.filter((s) => s.label === 'good'), defO = order.filter((s) => s.label !== 'good');
    const foldOf = {};
    goodsO.forEach((s, i) => { foldOf[s.id] = i % folds; });
    defO.forEach((s, i) => { foldOf[s.id] = i % folds; });
    for (let f = 0; f < folds; f++) {
      const train = samples.filter((s) => foldOf[s.id] !== f);
      const test = samples.filter((s) => foldOf[s.id] === f);
      const head = await RecipeModel.trainHeadOn(train, classes, { epochs: settings.epochs || 40 });
      const probs = tf.tidy(() => head.predict(headInput(test.map((s) => s.feat.deep[0]))).arraySync());
      test.forEach((s, i) => { pDefById[s.id] = 1 - probs[i][0]; });
      head.dispose();
      onProgress?.((f + 1) / (folds + 1));
    }
  }

  const { deep, classic } = RecipeModel.buildDetectors(samples, settings.jig);
  const rows = samples.map((s) => {
    const zDeep = deep.score(s.feat.deep[0], s.label === 'good' ? s.id : null);
    const zPix = classic ? classic.score(s.feat.classic[0], s.label === 'good' ? s.id : null) : null;
    const z = zPix === null ? zDeep : Math.max(zDeep, zPix);
    const pDef = withHead ? pDefById[s.id] : null;
    return { id: s.id, truth: s.label === 'good' ? 'good' : 'defect', z, pDef, verdict: decide(z, pDef, settings) };
  });
  onProgress?.(1);

  const g = rows.filter((r) => r.truth === 'good'), d = rows.filter((r) => r.truth === 'defect');
  const cnt = (arr, v) => arr.filter((r) => r.verdict === v).length;
  return {
    withHead,
    good: { n: g.length, pass: cnt(g, 'pass'), review: cnt(g, 'review'), reject: cnt(g, 'reject') },
    defect: { n: d.length, pass: cnt(d, 'pass'), review: cnt(d, 'review'), reject: cnt(d, 'reject') },
    rows,
    // suggested pass limit: 97.5th percentile of honest good-part scores
    suggestedPassZ: g.length ? [...g.map((r) => r.z)].sort((a, b) => a - b)[Math.min(g.length - 1, Math.floor(g.length * 0.975))] : null,
  };
}
