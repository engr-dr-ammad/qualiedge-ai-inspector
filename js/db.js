// Tiny IndexedDB wrapper. All data stays on the device.
const DB_NAME = 'qualiedge';
const DB_VERSION = 1;
let dbPromise = null;

function open() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('recipes')) db.createObjectStore('recipes', { keyPath: 'id' });
        if (!db.objectStoreNames.contains('samples')) {
          db.createObjectStore('samples', { keyPath: 'id' }).createIndex('recipeId', 'recipeId');
        }
        if (!db.objectStoreNames.contains('batches')) {
          db.createObjectStore('batches', { keyPath: 'id' }).createIndex('recipeId', 'recipeId');
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

async function run(store, mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, mode);
    const req = fn(tx.objectStore(store));
    let out;
    if (req) req.onsuccess = () => { out = req.result; };
    tx.oncomplete = () => resolve(out);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error('Storage transaction aborted (device storage full?)'));
  });
}

export const put = (store, value) => run(store, 'readwrite', (s) => s.put(value));
export const get = (store, key) => run(store, 'readonly', (s) => s.get(key));
export const del = (store, key) => run(store, 'readwrite', (s) => s.delete(key));
export const all = (store) => run(store, 'readonly', (s) => s.getAll());
export const byRecipe = (store, recipeId) => run(store, 'readonly', (s) => s.index('recipeId').getAll(recipeId));

export async function putMany(store, values) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    const s = tx.objectStore(store);
    values.forEach((v) => s.put(v));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error('Storage transaction aborted'));
  });
}

export async function deleteRecipeData(recipeId) {
  for (const store of ['samples', 'batches']) {
    const items = await byRecipe(store, recipeId);
    for (const it of items) await del(store, it.id);
  }
  await del('recipes', recipeId);
}

export function uid(prefix = '') {
  const rnd = crypto.getRandomValues(new Uint32Array(2));
  return prefix + Date.now().toString(36) + rnd[0].toString(36) + rnd[1].toString(36);
}
