/**
 * Browser Noise Free Mic — Shared IndexedDB & Preferences Store (appStore.js)
 *
 * Bridges state seamlessly across input.html -> processing.html -> output.html:
 * - IndexedDB `BNFM_DB`:
 *   1. `activeJob`     — Active 48 kHz job (rawPCM, enhancedPCM, masteredPCM, telemetry, metadata)
 *   2. `takesHistory`  — Local Last-5 Takes History (auto-pruning oldest)
 *   3. `calibration`   — Loop A 48 kHz 2-second Room Snapshot noise profile
 * - localStorage `BNFM_PREFS`:
 *   theme ('light'|'dark'), nerdsMode (false), selectedEngine ('df3'),
 *   selectedPreset ('podcast'), comfortNoise (true), edgeTrim (true)
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.BNFMStore = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const DB_NAME = 'BNFM_DB';
  const DB_VERSION = 1;
  const STORE_ACTIVE = 'activeJob';
  const STORE_HISTORY = 'takesHistory';
  const STORE_CALIBRATION = 'calibration';
  const ACTIVE_JOB_KEY = 'current';
  const ROOM_PROFILE_KEY = 'roomProfile';
  const MAX_HISTORY_TAKES = 5;
  const PREFS_STORAGE_KEY = 'BNFM_PREFS';

  const DEFAULT_PREFS = Object.freeze({
    theme: 'light',
    nerdsMode: false,
    selectedEngine: 'df3',
    selectedPreset: 'podcast',
    comfortNoise: true,
    edgeTrim: true,
    dryWetPct: 100
  });

  let dbPromise = null;

  function openDB() {
    if (dbPromise) return dbPromise;
    if (typeof indexedDB === 'undefined') {
      return Promise.reject(new Error('IndexedDB is not available in this environment.'));
    }

    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains(STORE_ACTIVE)) {
          db.createObjectStore(STORE_ACTIVE, { keyPath: 'id' });
        }

        if (!db.objectStoreNames.contains(STORE_HISTORY)) {
          const histStore = db.createObjectStore(STORE_HISTORY, { keyPath: 'id' });
          histStore.createIndex('createdAt', 'createdAt', { unique: false });
        }

        if (!db.objectStoreNames.contains(STORE_CALIBRATION)) {
          db.createObjectStore(STORE_CALIBRATION, { keyPath: 'id' });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => {
        dbPromise = null;
        reject(request.error || new Error('Failed to open BNFM_DB.'));
      };
    });

    return dbPromise;
  }

  function toStorablePCM(pcm) {
    if (!pcm) return null;
    if (pcm instanceof Float32Array) {
      return pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength);
    }
    if (pcm instanceof ArrayBuffer) {
      return pcm.slice(0);
    }
    if (ArrayBuffer.isView(pcm)) {
      const f32 = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i];
      return f32.buffer;
    }
    return null;
  }

  function fromStorablePCM(buf) {
    if (!buf) return null;
    if (buf instanceof Float32Array) return buf;
    if (buf instanceof ArrayBuffer) return new Float32Array(buf);
    if (ArrayBuffer.isView(buf)) {
      return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
    }
    return null;
  }

  function hydrateJob(record) {
    if (!record) return null;
    return {
      ...record,
      rawPCM: fromStorablePCM(record.rawPCM),
      enhancedPCM: fromStorablePCM(record.enhancedPCM),
      masteredPCM: fromStorablePCM(record.masteredPCM),
      sampleRate: record.sampleRate || 48000
    };
  }

  function serializeJob(job, id) {
    const rawBuf = toStorablePCM(job.rawPCM);
    const enhBuf = toStorablePCM(job.enhancedPCM);
    const mastBuf = toStorablePCM(job.masteredPCM);
    const sampleRate = job.sampleRate || 48000;
    const rawLen = rawBuf ? rawBuf.byteLength / 4 : 0;

    return {
      id: id || job.id || ACTIVE_JOB_KEY,
      fileName: job.fileName || 'untitled_48k.wav',
      mode: job.mode || 'upload',
      engine: job.engine || getPrefs().selectedEngine,
      preset: job.preset || getPrefs().selectedPreset,
      sampleRate,
      durationSec: job.durationSec || (rawLen > 0 ? rawLen / sampleRate : 0),
      rawPCM: rawBuf,
      enhancedPCM: enhBuf,
      masteredPCM: mastBuf,
      telemetry: job.telemetry ? { ...job.telemetry } : null,
      createdAt: job.createdAt || new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
  }

  function runTx(storeName, mode, executor) {
    return openDB().then((db) => new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, mode);
      const store = tx.objectStore(storeName);
      let result;

      try {
        result = executor(store, tx);
      } catch (err) {
        tx.abort();
        reject(err);
        return;
      }

      if (result && typeof result.then === 'function') {
        result.then(
          (val) => {
            tx.oncomplete = () => resolve(val);
            tx.onerror = () => reject(tx.error);
          },
          (err) => {
            try { tx.abort(); } catch (_) {}
            reject(err);
          }
        );
      } else {
        tx.oncomplete = () => resolve(result);
        tx.onerror = () => reject(tx.error);
      }
    }));
  }

  function reqToPromise(req) {
    return new Promise((resolve, reject) => {
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  // ---------------------------------------------------------------------------
  // Active Job API (shared across input.html -> processing.html -> output.html)
  // ---------------------------------------------------------------------------

  async function saveActiveJob(job) {
    const serialized = serializeJob(job, ACTIVE_JOB_KEY);
    await runTx(STORE_ACTIVE, 'readwrite', (store) => reqToPromise(store.put(serialized)));
    return hydrateJob(serialized);
  }

  async function getActiveJob() {
    const record = await runTx(STORE_ACTIVE, 'readonly', (store) => reqToPromise(store.get(ACTIVE_JOB_KEY)));
    return hydrateJob(record);
  }

  async function patchActiveJob(patch) {
    const existing = (await getActiveJob()) || {};
    const merged = {
      ...existing,
      ...patch,
      telemetry: patch.telemetry
        ? { ...(existing.telemetry || {}), ...patch.telemetry }
        : existing.telemetry
    };
    return saveActiveJob(merged);
  }

  async function clearActiveJob() {
    await runTx(STORE_ACTIVE, 'readwrite', (store) => reqToPromise(store.delete(ACTIVE_JOB_KEY)));
  }

  // ---------------------------------------------------------------------------
  // Last-5 Takes History API (auto-prunes oldest beyond MAX_HISTORY_TAKES)
  // ---------------------------------------------------------------------------

  function generateTakeId() {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return 'take_' + crypto.randomUUID();
    }
    return 'take_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
  }

  async function saveTake(take) {
    const id = take.id || generateTakeId();
    const record = serializeJob({ ...take, createdAt: take.createdAt || new Date().toISOString() }, id);

    await runTx(STORE_HISTORY, 'readwrite', async (store) => {
      await reqToPromise(store.put(record));
      const all = await reqToPromise(store.getAll());
      if (all.length > MAX_HISTORY_TAKES) {
        all.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
        const toDelete = all.slice(MAX_HISTORY_TAKES);
        for (const old of toDelete) {
          await reqToPromise(store.delete(old.id));
        }
      }
    });

    return hydrateJob(record);
  }

  async function getTakesHistory(includePCM = false) {
    const all = await runTx(STORE_HISTORY, 'readonly', (store) => reqToPromise(store.getAll()));
    all.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

    if (includePCM) {
      return all.map(hydrateJob);
    }

    return all.map((r) => ({
      id: r.id,
      fileName: r.fileName,
      mode: r.mode,
      engine: r.engine,
      preset: r.preset,
      sampleRate: r.sampleRate || 48000,
      durationSec: r.durationSec || 0,
      telemetry: r.telemetry || null,
      createdAt: r.createdAt,
      updatedAt: r.updatedAt,
      hasRawPCM: !!r.rawPCM,
      hasEnhancedPCM: !!r.enhancedPCM
    }));
  }

  async function getTakeById(id) {
    const record = await runTx(STORE_HISTORY, 'readonly', (store) => reqToPromise(store.get(id)));
    return hydrateJob(record);
  }

  async function restoreTakeToActive(id) {
    const take = await getTakeById(id);
    if (!take) return null;
    return saveActiveJob(take);
  }

  async function deleteTake(id) {
    await runTx(STORE_HISTORY, 'readwrite', (store) => reqToPromise(store.delete(id)));
  }

  async function clearTakesHistory() {
    await runTx(STORE_HISTORY, 'readwrite', (store) => reqToPromise(store.clear()));
  }

  // ---------------------------------------------------------------------------
  // Loop A Room Calibration Profile API (2-second room snapshot @ 48 kHz)
  // ---------------------------------------------------------------------------

  async function saveRoomProfile(profile) {
    const record = {
      id: ROOM_PROFILE_KEY,
      sampleRate: profile.sampleRate || 48000,
      fftSize: profile.fftSize || 2048,
      noiseFloorDb: typeof profile.noiseFloorDb === 'number' ? profile.noiseFloorDb : -60,
      durationSec: profile.durationSec || 2.0,
      spectrum: toStorablePCM(profile.spectrum),
      capturedAt: profile.capturedAt || new Date().toISOString()
    };
    await runTx(STORE_CALIBRATION, 'readwrite', (store) => reqToPromise(store.put(record)));
    return {
      ...record,
      spectrum: fromStorablePCM(record.spectrum)
    };
  }

  async function getRoomProfile() {
    const record = await runTx(STORE_CALIBRATION, 'readonly', (store) => reqToPromise(store.get(ROOM_PROFILE_KEY)));
    if (!record) return null;
    return {
      ...record,
      spectrum: fromStorablePCM(record.spectrum)
    };
  }

  async function clearRoomProfile() {
    await runTx(STORE_CALIBRATION, 'readwrite', (store) => reqToPromise(store.delete(ROOM_PROFILE_KEY)));
  }

  // ---------------------------------------------------------------------------
  // localStorage Preferences API
  // ---------------------------------------------------------------------------

  function getPrefs() {
    if (typeof localStorage === 'undefined') {
      return { ...DEFAULT_PREFS };
    }
    try {
      const raw = localStorage.getItem(PREFS_STORAGE_KEY);
      if (!raw) return { ...DEFAULT_PREFS };
      const parsed = JSON.parse(raw);
      return { ...DEFAULT_PREFS, ...parsed };
    } catch (_) {
      return { ...DEFAULT_PREFS };
    }
  }

  function setPrefs(partial) {
    const next = { ...getPrefs(), ...partial };
    if (typeof localStorage !== 'undefined') {
      try {
        localStorage.setItem(PREFS_STORAGE_KEY, JSON.stringify(next));
      } catch (_) {}
    }
    if (partial && typeof partial.theme === 'string') {
      applyTheme(next.theme);
    }
    return next;
  }

  function setPref(key, value) {
    return setPrefs({ [key]: value });
  }

  function resetPrefs() {
    if (typeof localStorage !== 'undefined') {
      try {
        localStorage.removeItem(PREFS_STORAGE_KEY);
      } catch (_) {}
    }
    applyTheme(DEFAULT_PREFS.theme);
    return { ...DEFAULT_PREFS };
  }

  function applyTheme(theme) {
    if (typeof document === 'undefined' || !document.documentElement) return;
    const resolved = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', resolved);
  }

  function onPrefsChange(callback) {
    if (typeof window === 'undefined' || typeof callback !== 'function') {
      return () => {};
    }
    const handler = (event) => {
      if (event.key === PREFS_STORAGE_KEY) {
        callback(getPrefs());
      }
    };
    window.addEventListener('storage', handler);
    return () => window.removeEventListener('storage', handler);
  }

  return {
    DB_NAME,
    DB_VERSION,
    MAX_HISTORY_TAKES,
    DEFAULT_PREFS,
    openDB,
    saveActiveJob,
    getActiveJob,
    patchActiveJob,
    clearActiveJob,
    saveTake,
    getTakesHistory,
    getTakeById,
    restoreTakeToActive,
    deleteTake,
    clearTakesHistory,
    saveRoomProfile,
    getRoomProfile,
    clearRoomProfile,
    getPrefs,
    setPref,
    setPrefs,
    resetPrefs,
    applyTheme,
    onPrefsChange
  };
});
