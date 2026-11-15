/* global tf */
// Feature extraction: a pretrained MobileNet (transfer learning) turns each 224x224 image into an
// embedding vector. A "classic" pixel descriptor (lighting-normalised) is always available as an
// offline fallback and as the pixel-reference channel for fixed-position (jig) inspection.

const BASE = 'https://storage.googleapis.com/tfjs-models/tfjs/';
export const BACKBONES = {
  fast: { url: BASE + 'mobilenet_v1_0.25_224/model.json', label: 'MobileNet v1 0.25' },
  balanced: { url: BASE + 'mobilenet_v1_0.50_224/model.json', label: 'MobileNet v1 0.50' },
  accurate: { url: BASE + 'mobilenet_v1_1.0_224/model.json', label: 'MobileNet v1 1.00' },
  classic: { url: null, label: 'Classic pixel features' },
};

let net = null;
let netKey = null;
let loading = null;

export function activeBackbone() { return net ? netKey : 'classic'; }

export async function loadBackbone(key) {
  if (key === 'classic') { netKey = 'classic'; net = null; return 'classic'; }
  if (net && netKey === key) return key;
  if (loading?.key === key) return loading.promise;
  const override = new URLSearchParams(location.search).get('backboneUrl');
  const url = override || BACKBONES[key].url;
  const promise = (async () => {
    try {
      const m = await tf.loadLayersModel(url);
      const layer = m.getLayer('conv_pw_13_relu');
      const trunk = tf.model({ inputs: m.inputs, outputs: layer.output });
      tf.tidy(() => trunk.predict(tf.zeros([1, 224, 224, 3])));
      net = trunk; netKey = key;
      return key;
    } catch (err) {
      console.warn('Feature network failed to load, using classic features', err);
      net = null; netKey = 'classic';
      return 'classic';
    } finally {
      loading = null;
    }
  })();
  loading = { key, promise };
  return promise;
}

function l2(t) {
  return t.div(t.norm('euclidean', 1, true).add(1e-8));
}

// Lighting-normalised small image: each channel standardised per image (removes global
// brightness/contrast change), so the reference comparison is about shape and presence.
function classicTensor(x) {
  // Area-average 224 -> 16x16 (each value covers a 14x14 block), so small shifts change it smoothly.
  const small = tf.avgPool(x, 14, 14, 'valid');
  const mean = small.mean([1, 2], true);
  const std = small.sub(mean).square().mean([1, 2], true).sqrt().add(1e-3);
  const norm = small.sub(mean).div(std);
  return l2(norm.reshape([x.shape[0], -1]));
}

// sources: array of 224x224 canvases. Returns { deep: Float32Array[] | null, classic: Float32Array[] }
export async function embed(sources, { wantClassic = true } = {}) {
  const out = { deep: null, classic: null, backbone: activeBackbone() };
  const chunk = 16;
  const deep = [];
  const classic = [];
  for (let i = 0; i < sources.length; i += chunk) {
    const part = sources.slice(i, i + chunk);
    const [d, c] = tf.tidy(() => {
      const x = tf.stack(part.map((s) => tf.browser.fromPixels(s))).toFloat();
      const ct = wantClassic || !net ? classicTensor(x) : null;
      const dt = net ? l2(net.predict(x.div(127.5).sub(1)).mean([1, 2])) : null;
      return [dt, ct];
    });
    if (d) { const arr = await d.data(); const dim = d.shape[1]; d.dispose(); for (let k = 0; k < part.length; k++) deep.push(arr.slice(k * dim, (k + 1) * dim)); }
    if (c) { const arr = await c.data(); const dim = c.shape[1]; c.dispose(); for (let k = 0; k < part.length; k++) classic.push(arr.slice(k * dim, (k + 1) * dim)); }
    await tf.nextFrame();
  }
  out.deep = net ? deep : classic;
  out.classic = classic.length ? classic : null;
  return out;
}

// ---- image helpers ----
export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not read image'));
    img.src = src;
  });
}

export function fileToImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { resolve(img); setTimeout(() => URL.revokeObjectURL(url), 1000); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Not an image: ' + file.name)); };
    img.src = url;
  });
}

export function toCanvas224(source, sw, sh) {
  const c = document.createElement('canvas');
  c.width = 224; c.height = 224;
  c.getContext('2d').drawImage(source, 0, 0, sw ?? source.width, sh ?? source.height, 0, 0, 224, 224);
  return c;
}

export function canvasToDataUrl(canvas, size = 224, quality = 0.85) {
  if (canvas.width === size && canvas.height === size) return canvas.toDataURL('image/jpeg', quality);
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  c.getContext('2d').drawImage(canvas, 0, 0, size, size);
  return c.toDataURL('image/jpeg', quality);
}

// Deterministic pseudo-random so augmentations are reproducible per sample.
function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) % 100000) / 100000; };
}

// Small, realistic variations: shift, slight rotation/zoom, brightness/contrast; flips and
// 90° rotations only when the recipe allows them (orientation can itself be a defect).
export function augment(img, count, allowMirror, seed) {
  const out = [];
  const r = rng(seed);
  for (let i = 0; i < count; i++) {
    const c = document.createElement('canvas');
    c.width = 224; c.height = 224;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, 224, 224);
    ctx.translate(112 + (r() - 0.5) * 18, 112 + (r() - 0.5) * 18);
    ctx.rotate(((r() - 0.5) * 10 * Math.PI) / 180);
    const zoom = 1 + (r() - 0.3) * 0.12;
    ctx.scale(zoom, zoom);
    if (allowMirror) {
      if (r() < 0.5) ctx.scale(-1, 1);
      ctx.rotate(Math.floor(r() * 4) * Math.PI / 2);
    }
    ctx.drawImage(img, -112, -112, 224, 224);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const gain = 0.75 + r() * 0.5;
    const bias = (r() - 0.5) * 40;
    const id = ctx.getImageData(0, 0, 224, 224);
    const d = id.data;
    for (let p = 0; p < d.length; p += 4) {
      d[p] = d[p] * gain + bias; d[p + 1] = d[p + 1] * gain + bias; d[p + 2] = d[p + 2] * gain + bias;
    }
    ctx.putImageData(id, 0, 0);
    out.push(c);
  }
  return out;
}

export function hashString(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
