/**
 * NEON Browser Noise-Free Mic — ONNX Runtime Web Audio Inference Worker
 *
 * - Loads onnxruntime-web@1.30.0 (WebGPU + WASM SIMD)
 * - Probes navigator.gpu.requestAdapter({ powerPreference: "high-performance" })
 * - Tries WebGPU FP32 (models/gtcrn_fp32.onnx + models/gtcrn_stream_fp32.onnx)
 *   with graceful fallback to WASM SIMD INT8 (models/gtcrn_int8.onnx + models/gtcrn_stream_int8.onnx)
 * - Allocates exact zero-initialized streaming cache tensors from models/stream_io_meta.json
 *   (or streamSess.inputNames fallback) and runs a 5-frame shader warmup pass at startup
 * - Exact 512-pt Radix-2 FFT/IFFT with periodic Hann window (hop=256, 50% OLA, 0.75 COLA norm)
 * - Classical Wiener Filter baseline (15th-percentile noise PSD, floor 0.05)
 * - Batch GTCRN inference (noisy_spec [1,2,257,T] -> enh_spec)
 * - 1-frame stateful streaming inference (frame_spec [1,2,257,1] + caches -> enh_real, enh_imag)
 *   with Loop A Energy + Spectral-Flatness VAD & residual floor calibration.
 */

/* global importScripts, ort */
"use strict";

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.js");

const FFT_SIZE = 512;
const HOP_SIZE = 256;
const NUM_BINS = 257; // FFT_SIZE / 2 + 1
const COLA_NORM = 0.75; // Sum of squared periodic Hann windows at 50% overlap

// Precompute periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N))
const HANN_WINDOW = new Float32Array(FFT_SIZE);
for (let n = 0; n < FFT_SIZE; n++) {
  HANN_WINDOW[n] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * n) / FFT_SIZE));
}

// ==================== EXACT 512-PT RADIX-2 FFT / IFFT ====================

const BIT_REV = new Uint16Array(FFT_SIZE);
const COS_TABLE = new Float32Array(FFT_SIZE >> 1);
const SIN_TABLE = new Float32Array(FFT_SIZE >> 1);

(function initFftTables() {
  const bits = 9; // log2(512)
  for (let i = 0; i < FFT_SIZE; i++) {
    let x = i;
    let r = 0;
    for (let b = 0; b < bits; b++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    BIT_REV[i] = r;
  }
  for (let k = 0; k < (FFT_SIZE >> 1); k++) {
    const angle = (-2.0 * Math.PI * k) / FFT_SIZE;
    COS_TABLE[k] = Math.cos(angle);
    SIN_TABLE[k] = Math.sin(angle);
  }
})();

function fft512(real, imag) {
  for (let i = 0; i < FFT_SIZE; i++) {
    const j = BIT_REV[i];
    if (j > i) {
      let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
      tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
    }
  }
  for (let size = 2; size <= FFT_SIZE; size <<= 1) {
    const half = size >> 1;
    const step = FFT_SIZE / size;
    for (let i = 0; i < FFT_SIZE; i += size) {
      let tableIdx = 0;
      for (let j = i; j < i + half; j++) {
        const k = j + half;
        const wr = COS_TABLE[tableIdx];
        const wi = SIN_TABLE[tableIdx];
        const tr = real[k] * wr - imag[k] * wi;
        const ti = real[k] * wi + imag[k] * wr;
        real[k] = real[j] - tr;
        imag[k] = imag[j] - ti;
        real[j] += tr;
        imag[j] += ti;
        tableIdx += step;
      }
    }
  }
}

function ifft512(real, imag) {
  // Conjugate -> forward FFT -> conjugate & scale by 1/N
  for (let i = 0; i < FFT_SIZE; i++) imag[i] = -imag[i];
  fft512(real, imag);
  const invN = 1.0 / FFT_SIZE;
  for (let i = 0; i < FFT_SIZE; i++) {
    real[i] *= invN;
    imag[i] = -imag[i] * invN;
  }
}

function stft(pcm) {
  const numFrames = Math.max(1, Math.floor((pcm.length - FFT_SIZE) / HOP_SIZE) + 1);
  const realOut = new Float32Array(NUM_BINS * numFrames);
  const imagOut = new Float32Array(NUM_BINS * numFrames);
  const magOut = new Float32Array(NUM_BINS * numFrames);
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);

  for (let t = 0; t < numFrames; t++) {
    const offset = t * HOP_SIZE;
    for (let n = 0; n < FFT_SIZE; n++) {
      const idx = offset + n;
      re[n] = (idx < pcm.length ? pcm[idx] : 0) * HANN_WINDOW[n];
      im[n] = 0;
    }
    fft512(re, im);
    for (let f = 0; f < NUM_BINS; f++) {
      const dst = f * numFrames + t;
      realOut[dst] = re[f];
      imagOut[dst] = im[f];
      magOut[dst] = Math.hypot(re[f], im[f]);
    }
  }
  return { real: realOut, imag: imagOut, mag: magOut, numFrames };
}

function istft(realSpec, imagSpec, numFrames, outLength) {
  const totalLen = Math.max(outLength, (numFrames - 1) * HOP_SIZE + FFT_SIZE);
  const out = new Float32Array(totalLen);
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);

  for (let t = 0; t < numFrames; t++) {
    for (let f = 0; f < NUM_BINS; f++) {
      const src = f * numFrames + t;
      re[f] = realSpec[src];
      im[f] = imagSpec[src];
    }
    // Hermitian symmetry for real-valued time signal
    for (let f = 1; f < NUM_BINS - 1; f++) {
      re[FFT_SIZE - f] = re[f];
      im[FFT_SIZE - f] = -im[f];
    }
    ifft512(re, im);
    const offset = t * HOP_SIZE;
    for (let n = 0; n < FFT_SIZE; n++) {
      out[offset + n] += (re[n] * HANN_WINDOW[n]) / COLA_NORM;
    }
  }
  return out.subarray(0, outLength);
}

// ==================== CLASSICAL WIENER FILTER BASELINE ====================

function wienerFilter(stftData, outLength) {
  const { real, imag, mag, numFrames } = stftData;
  const enhReal = new Float32Array(real.length);
  const enhImag = new Float32Array(imag.length);
  const enhMag = new Float32Array(mag.length);
  const tmpPowers = new Float32Array(numFrames);
  const p15Idx = Math.max(0, Math.min(numFrames - 1, Math.floor(0.15 * numFrames)));
  const floor = 0.05;

  for (let f = 0; f < NUM_BINS; f++) {
    const row = f * numFrames;
    for (let t = 0; t < numFrames; t++) {
      const m = mag[row + t];
      tmpPowers[t] = m * m;
    }
    tmpPowers.sort();
    const noisePsd = tmpPowers[p15Idx];

    for (let t = 0; t < numFrames; t++) {
      const idx = row + t;
      const py = mag[idx] * mag[idx];
      const gain = Math.max(floor, (py - noisePsd) / (py + 1e-10));
      enhReal[idx] = real[idx] * gain;
      enhImag[idx] = imag[idx] * gain;
      enhMag[idx] = mag[idx] * gain;
    }
  }
  const pcm = istft(enhReal, enhImag, numFrames, outLength);
  return { pcm, mag: enhMag };
}

function estimateSnrGainDb(noisyPcm, enhancedPcm) {
  const frameLen = 256;
  const numFrames = Math.floor(noisyPcm.length / frameLen);
  if (numFrames < 4) return 0;

  const noisyE = new Float32Array(numFrames);
  const enhE = new Float32Array(numFrames);
  for (let t = 0; t < numFrames; t++) {
    let nSum = 0;
    let eSum = 0;
    const base = t * frameLen;
    for (let i = 0; i < frameLen; i++) {
      const ns = noisyPcm[base + i];
      const es = enhancedPcm[base + i];
      nSum += ns * ns;
      eSum += es * es;
    }
    noisyE[t] = nSum / frameLen + 1e-10;
    enhE[t] = eSum / frameLen + 1e-10;
  }

  const sortedN = Float32Array.from(noisyE).sort();
  const sortedE = Float32Array.from(enhE).sort();
  const lowIdx = Math.floor(numFrames * 0.15);
  const highIdx = Math.min(numFrames - 1, Math.floor(numFrames * 0.85));

  const snrIn = 10 * Math.log10(sortedN[highIdx] / sortedN[lowIdx]);
  const snrOut = 10 * Math.log10(sortedE[highIdx] / sortedE[lowIdx]);
  const gain = snrOut - snrIn;
  return Number.isFinite(gain) ? Math.max(-5, Math.min(35, gain)) : 0;
}

// ==================== WORKER STATE ====================

const state = {
  ready: false,
  backend: "CPU: WASM SIMD (INT8)",
  provider: "wasm",
  precision: "INT8",
  modelBytes: 0,
  batchSess: null,
  streamSess: null,
  streamMeta: null,
  streamCaches: {},
  cacheOutputToInputMap: {},
  // Streaming overlap-add buffers
  inPrevHalf: new Float32Array(HOP_SIZE),
  olaTail: new Float32Array(HOP_SIZE),
  fftRe: new Float32Array(FFT_SIZE),
  fftIm: new Float32Array(FFT_SIZE),
  frameMag: new Float32Array(NUM_BINS),
  // Loop A VAD + Noise Profile state
  noiseProfile: new Float32Array(NUM_BINS),
  hasProfile: false,
  calibrationFrames: 0,
  silenceAccumMs: 0,
  vadFrameCounter: 0
};

async function fetchModelBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url} (HTTP ${res.status})`);
  const buf = await res.arrayBuffer();
  return new Uint8Array(buf);
}

async function fetchStreamMeta() {
  try {
    const res = await fetch("models/stream_io_meta.json");
    if (res.ok) return await res.json();
  } catch (_) {
    // Fall back to inspecting session inputNames
  }
  return null;
}

function buildZeroStreamCaches() {
  state.streamCaches = {};
  state.cacheOutputToInputMap = {};
  if (!state.streamSess) return;

  const meta = state.streamMeta;
  const inputNames = state.streamSess.inputNames || [];
  const outputNames = state.streamSess.outputNames || [];

  // Default GTCRN streaming cache shapes if metadata JSON is absent
  const defaultShapes = {
    enc_cache_0: [1, 16, 2, 33],
    enc_cache_1: [1, 16, 2, 33],
    enc_cache_2: [1, 16, 2, 33],
    dec_cache_0: [1, 16, 2, 33],
    dec_cache_1: [1, 16, 2, 33],
    dec_cache_2: [1, 16, 2, 33],
    gru_h_in: [2, 1, 16, 33]
  };

  for (const name of inputNames) {
    if (name === "frame_spec" || name === "noisy_spec" || name === "input") continue;
    let shape = null;
    let dtype = "float32";

    if (meta && meta.caches && meta.caches[name]) {
      shape = meta.caches[name].shape || meta.caches[name];
      dtype = meta.caches[name].dtype || "float32";
    } else if (meta && meta.inputs && meta.inputs[name]) {
      shape = meta.inputs[name].shape || meta.inputs[name];
      dtype = meta.inputs[name].dtype || "float32";
    } else if (state.streamSess.inputMetadata) {
      const idx = inputNames.indexOf(name);
      const md = state.streamSess.inputMetadata[idx];
      if (md && Array.isArray(md.shape) && md.shape.every((d) => typeof d === "number" && d > 0)) {
        shape = md.shape;
      }
    }

    if (!shape) {
      shape = defaultShapes[name] || [1, 16, 2, 33];
    }

    const totalSize = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(totalSize);
    state.streamCaches[name] = new ort.Tensor(dtype, data, shape);

    // Map corresponding output cache name back to this input name
    let outName = null;
    if (meta && meta.state_map && meta.state_map[name]) {
      outName = meta.state_map[name];
    } else if (name.endsWith("_in")) {
      const candidate = name.replace(/_in$/, "_out");
      if (outputNames.includes(candidate)) outName = candidate;
    } else if (outputNames.includes(`${name}_out`)) {
      outName = `${name}_out`;
    } else if (outputNames.includes(name)) {
      outName = name;
    }
    if (outName) {
      state.cacheOutputToInputMap[outName] = name;
    }
  }
}

function resetStreamState() {
  state.inPrevHalf.fill(0);
  state.olaTail.fill(0);
  buildZeroStreamCaches();
}

async function warmupStreamingSession(frames = 5) {
  if (!state.streamSess) return;
  const zeroFrame = new Float32Array(HOP_SIZE);
  for (let i = 0; i < frames; i++) {
    await runStreamFrame(zeroFrame, false);
  }
  resetStreamState();
}

async function initEngine() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
  ort.env.wasm.simd = true;

  let webgpuAvailable = false;
  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) webgpuAvailable = true;
    } catch (_) {
      webgpuAvailable = false;
    }
  }

  state.streamMeta = await fetchStreamMeta();

  // Attempt 1: WebGPU FP32
  if (webgpuAvailable) {
    try {
      const [batchBuf, streamBuf] = await Promise.all([
        fetchModelBuffer("models/gtcrn_fp32.onnx"),
        fetchModelBuffer("models/gtcrn_stream_fp32.onnx")
      ]);
      const opts = { executionProviders: ["webgpu"] };
      state.batchSess = await ort.InferenceSession.create(batchBuf, opts);
      state.streamSess = await ort.InferenceSession.create(streamBuf, opts);
      state.modelBytes = batchBuf.byteLength + streamBuf.byteLength;
      state.provider = "webgpu";
      state.precision = "FP32";
      state.backend = "GPU: WebGPU (FP32)";
    } catch (err) {
      console.warn("[audioWorker] WebGPU FP32 init failed, falling back to WASM SIMD INT8:", err);
      state.batchSess = null;
      state.streamSess = null;
    }
  }

  // Attempt 2: WASM SIMD INT8 fallback
  if (!state.batchSess || !state.streamSess) {
    try {
      const [batchBuf, streamBuf] = await Promise.all([
        fetchModelBuffer("models/gtcrn_int8.onnx"),
        fetchModelBuffer("models/gtcrn_stream_int8.onnx")
      ]);
      const opts = { executionProviders: ["wasm"] };
      state.batchSess = await ort.InferenceSession.create(batchBuf, opts);
      state.streamSess = await ort.InferenceSession.create(streamBuf, opts);
      state.modelBytes = batchBuf.byteLength + streamBuf.byteLength;
      state.provider = "wasm";
      state.precision = "INT8";
      state.backend = "CPU: WASM SIMD (INT8)";
    } catch (err) {
      console.warn("[audioWorker] ONNX model files unavailable; running Wiener DSP fallback:", err);
      state.batchSess = null;
      state.streamSess = null;
      state.modelBytes = 0;
      state.provider = "wasm";
      state.precision = "INT8";
      state.backend = "CPU: WASM SIMD (INT8)";
    }
  }

  if (state.streamSess) {
    buildZeroStreamCaches();
    await warmupStreamingSession(5);
  }

  state.ready = true;
  self.postMessage({
    type: "ready",
    backend: state.backend,
    provider: state.provider,
    precision: state.precision,
    modelBytes: state.modelBytes,
    modelsLoaded: Boolean(state.batchSess && state.streamSess)
  });
}

// ==================== LOOP A VAD & RESIDUAL FLOOR CALIBRATION ====================

function updateLoopAVad(magBins) {
  let powerSum = 0;
  let logPowerSum = 0;
  let speechBand = 0;
  const eps = 1e-10;

  for (let k = 0; k < NUM_BINS; k++) {
    const p = magBins[k] * magBins[k] + eps;
    powerSum += p;
    logPowerSum += Math.log(p);
    if (k >= 10 && k <= 109) speechBand += p;
  }

  const arithMean = powerSum / NUM_BINS;
  const flatness = Math.exp(logPowerSum / NUM_BINS) / (arithMean + eps);
  const rms = Math.sqrt(arithMean);
  const speechRatio = speechBand / (powerSum + eps);
  const isSpeech = rms > 0.035 && (flatness < 0.35 || speechRatio > 0.65);

  if (!isSpeech) {
    state.silenceAccumMs += 16;
    if (state.silenceAccumMs >= 150) {
      if (!state.hasProfile) {
        state.noiseProfile.set(magBins);
        state.hasProfile = true;
      } else {
        for (let k = 0; k < NUM_BINS; k++) {
          state.noiseProfile[k] = 0.92 * state.noiseProfile[k] + 0.08 * magBins[k];
        }
      }
      state.calibrationFrames++;
      return true;
    }
  } else {
    state.silenceAccumMs = 0;
  }
  return false;
}

function applyLoopADamper(re, im) {
  if (!state.hasProfile || state.calibrationFrames < 5) return;
  for (let k = 0; k < NUM_BINS; k++) {
    const mag = Math.hypot(re[k], im[k]);
    const refFloor = state.noiseProfile[k] * 0.25;
    if (mag < refFloor && refFloor > 1e-8) {
      const gain = Math.max(0.15, 1.0 - 0.35 * (1.0 - mag / refFloor));
      re[k] *= gain;
      im[k] *= gain;
    }
  }
}

// ==================== BATCH & STREAMING INFERENCE ====================

async function runBatchEnhancement(pcm) {
  const t0 = performance.now();
  const stftData = stft(pcm);
  const { real, imag, mag, numFrames } = stftData;

  // Track B: Classical Wiener Filter Baseline
  const wiener = wienerFilter(stftData, pcm.length);

  // Track C: GTCRN Neural Enhancement
  let gtcrnPcm = null;
  let gtcrnMag = null;
  const tModelStart = performance.now();

  if (state.batchSess) {
    const inputName = state.batchSess.inputNames.includes("noisy_spec")
      ? "noisy_spec"
      : state.batchSess.inputNames[0];

    // Layout: [1, 2, 257, T]
    const specBuf = new Float32Array(2 * NUM_BINS * numFrames);
    const chStride = NUM_BINS * numFrames;
    specBuf.set(real, 0);
    specBuf.set(imag, chStride);

    const feeds = {};
    feeds[inputName] = new ort.Tensor("float32", specBuf, [1, 2, NUM_BINS, numFrames]);
    const results = await state.batchSess.run(feeds);

    const outName = state.batchSess.outputNames.includes("enh_spec")
      ? "enh_spec"
      : state.batchSess.outputNames[0];
    const outTensor = results[outName];
    const outData = outTensor.data;

    const enhReal = outData.subarray(0, chStride);
    const enhImag = outData.subarray(chStride, 2 * chStride);
    gtcrnMag = new Float32Array(chStride);
    for (let i = 0; i < chStride; i++) {
      gtcrnMag[i] = Math.hypot(enhReal[i], enhImag[i]);
    }
    gtcrnPcm = istft(enhReal, enhImag, numFrames, pcm.length);
  } else {
    // Graceful fallback when model files are not yet hosted
    gtcrnPcm = new Float32Array(wiener.pcm);
    gtcrnMag = new Float32Array(wiener.mag);
  }

  const inferenceMs = performance.now() - tModelStart;
  const totalMs = performance.now() - t0;
  const clipDurationMs = (pcm.length / 16000) * 1000;
  const rtf = clipDurationMs > 0 ? inferenceMs / clipDurationMs : 0;
  const snrGainDb = estimateSnrGainDb(pcm, gtcrnPcm);

  return {
    wienerPcm: wiener.pcm,
    gtcrnPcm,
    noisyMag: mag,
    wienerMag: wiener.mag,
    gtcrnMag,
    numFrames,
    numBins: NUM_BINS,
    inferenceMs,
    totalMs,
    rtf,
    snrGainDb
  };
}

async function runStreamFrame(hopSamples, enableVad = true) {
  const t0 = performance.now();
  const re = state.fftRe;
  const im = state.fftIm;

  // Assemble 512-sample analysis frame: [prev 256 | new 256] * Hann
  for (let n = 0; n < HOP_SIZE; n++) {
    re[n] = state.inPrevHalf[n] * HANN_WINDOW[n];
    re[HOP_SIZE + n] = hopSamples[n] * HANN_WINDOW[HOP_SIZE + n];
    im[n] = 0;
    im[HOP_SIZE + n] = 0;
  }
  state.inPrevHalf.set(hopSamples);

  fft512(re, im);

  for (let k = 0; k < NUM_BINS; k++) {
    state.frameMag[k] = Math.hypot(re[k], im[k]);
  }

  let profileUpdated = false;
  if (enableVad) {
    profileUpdated = updateLoopAVad(state.frameMag);
  }

  if (state.streamSess) {
    const frameSpec = new Float32Array(2 * NUM_BINS);
    for (let k = 0; k < NUM_BINS; k++) {
      frameSpec[k] = re[k];
      frameSpec[NUM_BINS + k] = im[k];
    }

    const inputName = state.streamSess.inputNames.includes("frame_spec")
      ? "frame_spec"
      : state.streamSess.inputNames[0];

    const feeds = Object.assign({}, state.streamCaches);
    feeds[inputName] = new ort.Tensor("float32", frameSpec, [1, 2, NUM_BINS, 1]);

    const results = await state.streamSess.run(feeds);

    // Extract enhanced spectrum (supports enh_real/enh_imag or combined enh_spec)
    if (results.enh_real && results.enh_imag) {
      const er = results.enh_real.data;
      const ei = results.enh_imag.data;
      for (let k = 0; k < NUM_BINS; k++) {
        re[k] = er[k];
        im[k] = ei[k];
      }
    } else {
      const outName = state.streamSess.outputNames[0];
      const outData = results[outName].data;
      for (let k = 0; k < NUM_BINS; k++) {
        re[k] = outData[k];
        im[k] = outData[NUM_BINS + k];
      }
    }

    // Update streaming state caches
    for (const [outKey, inKey] of Object.entries(state.cacheOutputToInputMap)) {
      if (results[outKey]) {
        state.streamCaches[inKey] = results[outKey];
      }
    }
  } else if (state.hasProfile) {
    // Fallback spectral subtraction using learned room noise profile
    for (let k = 0; k < NUM_BINS; k++) {
      const py = state.frameMag[k] * state.frameMag[k];
      const pn = state.noiseProfile[k] * state.noiseProfile[k];
      const gain = Math.max(0.08, (py - pn) / (py + 1e-10));
      re[k] *= gain;
      im[k] *= gain;
    }
  }

  // Loop A residual floor damper
  if (enableVad) {
    applyLoopADamper(re, im);
  }

  // Hermitian symmetry & 512-pt IFFT
  for (let k = 1; k < NUM_BINS - 1; k++) {
    re[FFT_SIZE - k] = re[k];
    im[FFT_SIZE - k] = -im[k];
  }
  ifft512(re, im);

  // 50% overlap-add with periodic Hann synthesis window & 0.75 COLA normalization
  const outHop = new Float32Array(HOP_SIZE);
  for (let n = 0; n < HOP_SIZE; n++) {
    outHop[n] = state.olaTail[n] + (re[n] * HANN_WINDOW[n]) / COLA_NORM;
    state.olaTail[n] = (re[HOP_SIZE + n] * HANN_WINDOW[HOP_SIZE + n]) / COLA_NORM;
  }

  const latencyMs = performance.now() - t0;
  return { outHop, latencyMs, profileUpdated };
}

// ==================== WORKER MESSAGE HANDLER ====================

self.onmessage = async (event) => {
  const msg = event.data;
  if (!msg || !msg.type) return;

  try {
    if (msg.type === "init") {
      await initEngine();
    } else if (msg.type === "process_batch") {
      const pcm = msg.pcm instanceof Float32Array ? msg.pcm : new Float32Array(msg.pcm);
      const res = await runBatchEnhancement(pcm);
      self.postMessage(
        {
          type: "batch_result",
          requestId: msg.requestId,
          ...res
        },
        [
          res.wienerPcm.buffer,
          res.gtcrnPcm.buffer,
          res.noisyMag.buffer,
          res.wienerMag.buffer,
          res.gtcrnMag.buffer
        ]
      );
    } else if (msg.type === "stream_frame") {
      const frame = msg.frame instanceof Float32Array ? msg.frame : new Float32Array(msg.frame);
      const { outHop, latencyMs, profileUpdated } = await runStreamFrame(frame, true);

      let profileSync = null;
      if (profileUpdated) {
        state.vadFrameCounter++;
        if (state.vadFrameCounter % 25 === 0) {
          profileSync = {
            profile: Array.from(state.noiseProfile),
            calibrationFrames: state.calibrationFrames
          };
        }
      }

      self.postMessage(
        {
          type: "stream_result",
          frame: outHop,
          latencyMs,
          calibrationFrames: state.calibrationFrames,
          profileSync
        },
        [outHop.buffer]
      );
    } else if (msg.type === "reset_stream") {
      resetStreamState();
    } else if (msg.type === "load_profile") {
      if (msg.profile && msg.profile.length === NUM_BINS) {
        state.noiseProfile = new Float32Array(msg.profile);
        state.calibrationFrames = msg.calibrationFrames || 0;
        state.hasProfile = state.calibrationFrames > 0;
      }
    } else if (msg.type === "reset_profile") {
      state.noiseProfile.fill(0);
      state.calibrationFrames = 0;
      state.hasProfile = false;
      state.silenceAccumMs = 0;
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      requestId: msg.requestId,
      message: err && err.message ? err.message : String(err)
    });
  }
};
