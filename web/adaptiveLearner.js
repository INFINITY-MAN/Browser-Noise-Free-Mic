/**
 * NEON Browser Noise-Free Mic — Adaptive Learning & Telemetry Engine
 *
 * Loop A (On-Device Local Calibration):
 *   - Energy + Spectral-Flatness VAD (speech absent >= 150 ms -> EMA 0.92 noise profile across 257 bins)
 *   - Gentle residual floor damper
 *   - IndexedDB ("NoiseFreeMicDB") persistence for room noise profile, preferred Dry/Wet mix,
 *     rolling RTF/SNR metrics, and session count.
 *
 * Loop B (Federated / Aggregated Cloud Telemetry & Opt-In Feedback):
 *   - POST /api/telemetry
 *   - POST /api/feedback
 *   - GET  /api/metrics-dashboard
 */

(function (global) {
  "use strict";

  const DB_NAME = "NoiseFreeMicDB";
  const DB_VERSION = 1;
  const STORE_NAME = "adaptive_state";
  const STATE_KEY = "neon_user_profile_v1";
  const NUM_BINS = 257;
  const EMA_DECAY = 0.92;
  const SILENCE_THRESHOLD_MS = 150;

  class AdaptiveLearner {
    constructor() {
      this.db = null;
      this.noiseProfile = new Float32Array(NUM_BINS);
      this.hasProfile = false;
      this.calibrationFrames = 0;
      this.silenceAccumMs = 0;
      this.preferredWetMix = 1.0;
      this.sessionCount = 0;
      this.rollingRtf = [];
      this.rollingSnr = [];
      this.lastUpdated = null;
      this.lastVadState = "idle";
    }

    async init() {
      try {
        this.db = await this._openDB();
        await this.loadState();
        this.sessionCount += 1;
        await this.saveState();
      } catch (err) {
        console.warn("[AdaptiveLearner] IndexedDB unavailable, running in memory:", err);
      }
      return this.getSummary();
    }

    _openDB() {
      return new Promise((resolve, reject) => {
        if (!global.indexedDB) {
          reject(new Error("IndexedDB not supported"));
          return;
        }
        const req = global.indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME, { keyPath: "id" });
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }

    async loadState() {
      if (!this.db) return null;
      return new Promise((resolve, reject) => {
        const tx = this.db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(STATE_KEY);
        req.onsuccess = () => {
          const rec = req.result;
          if (rec) {
            if (rec.noiseProfile && rec.noiseProfile.length === NUM_BINS) {
              this.noiseProfile = new Float32Array(rec.noiseProfile);
              this.hasProfile = Boolean(rec.hasProfile);
            }
            this.calibrationFrames = rec.calibrationFrames || 0;
            this.preferredWetMix = typeof rec.preferredWetMix === "number" ? rec.preferredWetMix : 1.0;
            this.sessionCount = rec.sessionCount || 0;
            this.rollingRtf = Array.isArray(rec.rollingRtf) ? rec.rollingRtf.slice(-50) : [];
            this.rollingSnr = Array.isArray(rec.rollingSnr) ? rec.rollingSnr.slice(-50) : [];
            this.lastUpdated = rec.lastUpdated || null;
          }
          resolve(rec || null);
        };
        req.onerror = () => reject(req.error);
      });
    }

    async saveState() {
      if (!this.db) return;
      const payload = {
        id: STATE_KEY,
        noiseProfile: Array.from(this.noiseProfile),
        hasProfile: this.hasProfile,
        calibrationFrames: this.calibrationFrames,
        preferredWetMix: this.preferredWetMix,
        sessionCount: this.sessionCount,
        rollingRtf: this.rollingRtf.slice(-50),
        rollingSnr: this.rollingSnr.slice(-50),
        lastUpdated: new Date().toISOString()
      };
      this.lastUpdated = payload.lastUpdated;

      return new Promise((resolve, reject) => {
        const tx = this.db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        const req = store.put(payload);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    }

    async resetRoomProfile() {
      this.noiseProfile = new Float32Array(NUM_BINS);
      this.hasProfile = false;
      this.calibrationFrames = 0;
      this.silenceAccumMs = 0;
      this.lastVadState = "reset";
      await this.saveState();
      return this.getSummary();
    }

    /**
     * Loop A: Energy + Spectral-Flatness VAD.
     * When speech is absent for >= 150 ms, updates 257-bin room noise profile with EMA decay 0.92.
     */
    updateVadAndProfile(magBins, frameMs = 16) {
      if (!magBins || magBins.length < NUM_BINS) return { isSpeech: true, updated: false };

      let powerSum = 0;
      let logPowerSum = 0;
      let speechBandEnergy = 0;
      const eps = 1e-10;

      for (let k = 0; k < NUM_BINS; k++) {
        const p = magBins[k] * magBins[k] + eps;
        powerSum += p;
        logPowerSum += Math.log(p);
        // 300 Hz - 3400 Hz corresponds to bins ~10..109 at 16 kHz / 512 FFT
        if (k >= 10 && k <= 109) speechBandEnergy += p;
      }

      const arithMean = powerSum / NUM_BINS;
      const geoMean = Math.exp(logPowerSum / NUM_BINS);
      const spectralFlatness = geoMean / (arithMean + eps);
      const rmsEnergy = Math.sqrt(arithMean);
      const speechRatio = speechBandEnergy / (powerSum + eps);

      // Speech absent if low energy OR noise-like high flatness without concentrated speech formants
      const isSpeech = rmsEnergy > 0.035 && (spectralFlatness < 0.35 || speechRatio > 0.65);
      let updated = false;

      if (!isSpeech) {
        this.silenceAccumMs += frameMs;
        this.lastVadState = "noise_only";
        if (this.silenceAccumMs >= SILENCE_THRESHOLD_MS) {
          if (!this.hasProfile) {
            for (let k = 0; k < NUM_BINS; k++) {
              this.noiseProfile[k] = magBins[k];
            }
            this.hasProfile = true;
          } else {
            for (let k = 0; k < NUM_BINS; k++) {
              this.noiseProfile[k] = EMA_DECAY * this.noiseProfile[k] + (1 - EMA_DECAY) * magBins[k];
            }
          }
          this.calibrationFrames += 1;
          updated = true;
        }
      } else {
        this.silenceAccumMs = 0;
        this.lastVadState = "speech";
      }

      return { isSpeech, updated, spectralFlatness, rmsEnergy };
    }

    /**
     * Gentle residual floor damper applied to post-GTCRN spectrum.
     */
    applyResidualDamper(realBins, imagBins, strength = 0.35) {
      if (!this.hasProfile || this.calibrationFrames < 5) return;
      const floorScale = 0.25;
      for (let k = 0; k < NUM_BINS; k++) {
        const r = realBins[k];
        const im = imagBins[k];
        const mag = Math.hypot(r, im);
        const refFloor = this.noiseProfile[k] * floorScale;
        if (mag < refFloor && refFloor > 1e-8) {
          const ratio = mag / refFloor;
          const gain = Math.max(0.15, 1.0 - strength * (1.0 - ratio));
          realBins[k] = r * gain;
          imagBins[k] = im * gain;
        }
      }
    }

    syncProfileFromWorker(profileArray, calibrationFrames) {
      if (profileArray && profileArray.length === NUM_BINS) {
        this.noiseProfile = new Float32Array(profileArray);
        this.calibrationFrames = calibrationFrames || this.calibrationFrames;
        this.hasProfile = this.calibrationFrames > 0;
      }
    }

    async setPreferredWetMix(wetMix) {
      this.preferredWetMix = Math.max(0, Math.min(1, wetMix));
      await this.saveState();
    }

    async recordSessionMetrics(rtf, snrDb) {
      if (Number.isFinite(rtf)) {
        this.rollingRtf.push(Number(rtf.toFixed(4)));
        if (this.rollingRtf.length > 50) this.rollingRtf.shift();
      }
      if (Number.isFinite(snrDb)) {
        this.rollingSnr.push(Number(snrDb.toFixed(2)));
        if (this.rollingSnr.length > 50) this.rollingSnr.shift();
      }
      await this.saveState();
    }

    getSummary() {
      const avgRtf = this.rollingRtf.length
        ? this.rollingRtf.reduce((a, b) => a + b, 0) / this.rollingRtf.length
        : null;
      const avgSnr = this.rollingSnr.length
        ? this.rollingSnr.reduce((a, b) => a + b, 0) / this.rollingSnr.length
        : null;
      return {
        hasProfile: this.hasProfile,
        calibrationFrames: this.calibrationFrames,
        preferredWetMix: this.preferredWetMix,
        sessionCount: this.sessionCount,
        avgRtf,
        avgSnr,
        lastUpdated: this.lastUpdated,
        lastVadState: this.lastVadState
      };
    }

    // ==================== LOOP B: CLOUD TELEMETRY & FEEDBACK ====================

    async sendTelemetry(payload) {
      try {
        const res = await fetch("/api/telemetry", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            timestamp: new Date().toISOString(),
            sessionCount: this.sessionCount,
            ...payload
          })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (err) {
        return { ok: false, offline: true, error: err.message };
      }
    }

    async sendFeedback(payload) {
      try {
        const res = await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            timestamp: new Date().toISOString(),
            ...payload
          })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (err) {
        return { ok: false, offline: true, error: err.message };
      }
    }

    async fetchDashboard() {
      try {
        const res = await fetch("/api/metrics-dashboard", {
          method: "GET",
          headers: { Accept: "application/json" }
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        return { ok: true, data };
      } catch (err) {
        return { ok: false, offline: true, error: err.message };
      }
    }
  }

  global.AdaptiveLearner = AdaptiveLearner;
})(typeof self !== "undefined" ? self : window);
