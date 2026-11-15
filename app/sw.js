// QualiEdge AI Inspector service worker: offline app shell + cached feature-network weights.
const VERSION = 'qe-v1.0.0';
const SHELL = [
  './', 'index.html', 'css/app.css', 'js/main.js', 'js/db.js', 'js/camera.js', 'js/features.js', 'js/engine.js',
  'vendor/tf.min.js', 'manifest.webmanifest', 'icons/icon.svg', 'icons/icon-192.png', 'icons/icon-512.png',
];
const MODEL_CACHE = 'qe-models-v1';

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(
    keys.filter((k) => k !== VERSION && k !== MODEL_CACHE).map((k) => caches.delete(k)),
  )).then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;
  // Pretrained weights never change at a given URL: cache-first.
  if (url.hostname === 'storage.googleapis.com' && url.pathname.startsWith('/tfjs-models/')) {
    e.respondWith(caches.open(MODEL_CACHE).then(async (c) => {
      const hit = await c.match(e.request);
      if (hit) return hit;
      const res = await fetch(e.request);
      if (res.ok) c.put(e.request, res.clone());
      return res;
    }));
    return;
  }
  if (url.origin !== location.origin) return;
  // App files: network first so updates arrive, cache when offline.
  e.respondWith(fetch(e.request).then((res) => {
    if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(e.request, copy)); }
    return res;
  }).catch(() => caches.match(e.request, { ignoreSearch: true }).then((r) => r || caches.match('index.html'))));
});
