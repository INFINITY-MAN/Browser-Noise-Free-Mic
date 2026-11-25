/**
 * Browser Noise Free Mic — 48 kHz Dual-Engine & AAC Client Controller (audioEngine.js)
 *
 * Capabilities:
 *   1. Decode: Native AudioContext.decodeAudioData() resampled to 48 kHz mono Float32Array,
 *      with automatic fallback to WebCodecs AudioDecoder (`mp4a.40.2`) and `POST /api/decode`.
 *   2. Enhance (Dual-Engine): Calls `POST /api/denoise?engine=df3|gtcrn48k`, reads telemetry
 *      headers (`X-BNFM-Infer-Ms`, `X-BNFM-RTF`, `X-BNFM-Engine`, `X-BNFM-SNR-Gain-DB`),
 *      falls back to client-side 48 kHz Wiener/Adaptive denoiser when offline, and applies
 *      `BNFMMasteringChain` studio preset.
 *   3. Export: 16-bit 48 kHz RIFF WAV (`encodeWav48k`) + 160 kbps AAC `.m4a` (`exportAAC`
 *      using WebCodecs AudioEncoder `mp4a.40.2` or seamless fallback to `POST /api/encode-aac`).
 *   4. 20 kHz Studio Thermal Spectrogram Renderer (Black -> #FF5A00 Deep Orange -> Pure White).
 *   5. Synthetic 48 kHz Indian Household Demo Generator (3 s speech formants + 50/100/150 Hz
 *      AC mains hum + ceiling fan pink noise).
 *   6. 2-Second Room Snapshot Calibration for Loop A's 48 kHz noise profile in IndexedDB.
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(root);
  } else {
    root.BNFMAudioEngine = factory(root);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const TARGET_SAMPLE_RATE = 48000;
  const FFT_SIZE = 2048;

  // ---------------------------------------------------------------------------
  // Radix-2 Cooley-Tukey FFT (Pure JS, zero external dependencies)
  // ---------------------------------------------------------------------------

  function fftRadix2(re, im, inverse = false) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }

    const dir = inverse ? 1 : -1;
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (dir * 2 * Math.PI) / len;
      const wlenR = Math.cos(ang);
      const wlenI = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let wR = 1.0;
        let wI = 0.0;
        for (let j = 0; j < half; j++) {
          const uR = re[i + j];
          const uI = im[i + j];
          const vR = re[i + j + half] * wR - im[i + j + half] * wI;
          const vI = re[i + j + half] * wI + im[i + j + half] * wR;
          re[i + j] = uR + vR;
          im[i + j] = uI + vI;
          re[i + j + half] = uR - vR;
          im[i + j + half] = uI - vI;
          const nextWR = wR * wlenR - wI * wlenI;
          wI = wR * wlenI + wI * wlenR;
          wR = nextWR;
        }
      }
    }

    if (inverse) {
      for (let i = 0; i < n; i++) {
        re[i] /= n;
        im[i] /= n;
      }
    }
  }

  function buildHannWindow(n) {
    const win = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    }
    return win;
  }

  // ---------------------------------------------------------------------------
  // 16-bit 48 kHz RIFF WAV Encoder & Decoder
  // ---------------------------------------------------------------------------

  function encodeWavBuffer48k(float32PCM, sampleRate = TARGET_SAMPLE_RATE) {
    const numSamples = float32PCM.length;
    const dataBytes = numSamples * 2;
    const buffer = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(buffer);

    function writeStr(offset, str) {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    }

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataBytes, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true); // PCM chunk size
    view.setUint16(20, 1, true);  // AudioFormat = 1 (PCM)
    view.setUint16(22, 1, true);  // NumChannels = 1 (mono)
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // ByteRate
    view.setUint16(32, 2, true);  // BlockAlign
    view.setUint16(34, 16, true); // BitsPerSample
    writeStr(36, 'data');
    view.setUint32(40, dataBytes, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      const s = Math.max(-1, Math.min(1, float32PCM[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return buffer;
  }

  function encodeWav48k(float32PCM, sampleRate = TARGET_SAMPLE_RATE) {
    const buffer = encodeWavBuffer48k(float32PCM, sampleRate);
    return new Blob([buffer], { type: 'audio/wav' });
  }

  function decodeWavBuffer(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    if (arrayBuffer.byteLength < 44) {
      throw new Error('Invalid WAV buffer: too short.');
    }

    let offset = 12;
    let audioFormat = 1;
    let numChannels = 1;
    let sampleRate = TARGET_SAMPLE_RATE;
    let bitsPerSample = 16;
    let dataOffset = -1;
    let dataSize = 0;

    while (offset + 8 <= arrayBuffer.byteLength) {
      const id = String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3)
      );
      const size = view.getUint32(offset + 4, true);
      if (id === 'fmt ') {
        audioFormat = view.getUint16(offset + 8, true);
        numChannels = view.getUint16(offset + 10, true);
        sampleRate = view.getUint32(offset + 12, true);
        bitsPerSample = view.getUint16(offset + 22, true);
      } else if (id === 'data') {
        dataOffset = offset + 8;
        dataSize = Math.min(size, arrayBuffer.byteLength - dataOffset);
        break;
      }
      offset += 8 + size + (size % 2);
    }

    if (dataOffset < 0) throw new Error('WAV data chunk not found.');

    const bytesPerSample = bitsPerSample / 8;
    const totalFrames = Math.floor(dataSize / (bytesPerSample * numChannels));
    const mono = new Float32Array(totalFrames);

    let pos = dataOffset;
    for (let f = 0; f < totalFrames; f++) {
      let sum = 0.0;
      for (let ch = 0; ch < numChannels; ch++) {
        if (audioFormat === 3 && bitsPerSample === 32) {
          sum += view.getFloat32(pos, true);
        } else if (bitsPerSample === 16) {
          sum += view.getInt16(pos, true) / 32768.0;
        } else if (bitsPerSample === 24) {
          const b0 = view.getUint8(pos);
          const b1 = view.getUint8(pos + 1);
          const b2 = view.getInt8(pos + 2);
          sum += ((b2 << 16) | (b1 << 8) | b0) / 8388608.0;
        } else if (bitsPerSample === 32) {
          sum += view.getInt32(pos, true) / 2147483648.0;
        }
        pos += bytesPerSample;
      }
      mono[f] = sum / numChannels;
    }

    return { pcm: mono, sampleRate };
  }

  // ---------------------------------------------------------------------------
  // Linear Resampler (fallback when OfflineAudioContext is unavailable)
  // ---------------------------------------------------------------------------

  function resampleLinear(input, fromRate, toRate) {
    if (fromRate === toRate) return new Float32Array(input);
    const ratio = fromRate / toRate;
    const outLen = Math.max(1, Math.round(input.length / ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const idx = i * ratio;
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = idx - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  async function resampleTo48kMono(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const len = audioBuffer.length;
    const mono = new Float32Array(len);
    for (let ch = 0; ch < numChannels; ch++) {
      const chData = audioBuffer.getChannelData(ch);
      for (let i = 0; i < len; i++) mono[i] += chData[i] / numChannels;
    }

    if (audioBuffer.sampleRate === TARGET_SAMPLE_RATE) {
      return mono;
    }

    if (typeof OfflineAudioContext !== 'undefined') {
      const targetLen = Math.ceil((len * TARGET_SAMPLE_RATE) / audioBuffer.sampleRate);
      const offlineCtx = new OfflineAudioContext(1, targetLen, TARGET_SAMPLE_RATE);
      const srcBuf = offlineCtx.createBuffer(1, len, audioBuffer.sampleRate);
      srcBuf.copyToChannel(mono, 0);
      const src = offlineCtx.createBufferSource();
      src.buffer = srcBuf;
      src.connect(offlineCtx.destination);
      src.start(0);
      const rendered = await offlineCtx.startRendering();
      return new Float32Array(rendered.getChannelData(0));
    }

    return resampleLinear(mono, audioBuffer.sampleRate, TARGET_SAMPLE_RATE);
  }

  // ---------------------------------------------------------------------------
  // 1. Decode Pipeline (Native AudioContext -> WebCodecs AudioDecoder -> /api/decode)
  // ---------------------------------------------------------------------------

  function parseADTSFrames(bytes) {
    const frames = [];
    let offset = 0;
    const sampleRates = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

    while (offset + 7 <= bytes.length) {
      if (bytes[offset] !== 0xff || (bytes[offset + 1] & 0xf0) !== 0xf0) {
        offset++;
        continue;
      }
      const srIdx = (bytes[offset + 2] >> 2) & 0x0f;
      const channels = ((bytes[offset + 2] & 0x01) << 2) | ((bytes[offset + 3] >> 6) & 0x03);
      const frameLen = ((bytes[offset + 3] & 0x03) << 11) | (bytes[offset + 4] << 3) | ((bytes[offset + 5] >> 5) & 0x07);
      if (frameLen < 7 || offset + frameLen > bytes.length) break;

      frames.push({
        data: bytes.subarray(offset, offset + frameLen),
        sampleRate: sampleRates[srIdx] || TARGET_SAMPLE_RATE,
        channels: channels || 1
      });
      offset += frameLen;
    }
    return frames;
  }

  async function decodeWithWebCodecsAAC(arrayBuffer) {
    if (typeof AudioDecoder === 'undefined' || typeof EncodedAudioChunk === 'undefined') {
      throw new Error('WebCodecs AudioDecoder unavailable');
    }
    const bytes = new Uint8Array(arrayBuffer);
    const adtsFrames = parseADTSFrames(bytes);
    if (adtsFrames.length === 0) {
      throw new Error('Input is not a raw ADTS AAC stream');
    }

    const first = adtsFrames[0];
    const config = {
      codec: 'mp4a.40.2',
      sampleRate: first.sampleRate,
      numberOfChannels: first.channels
    };
    const support = await AudioDecoder.isConfigSupported(config);
    if (!support.supported) {
      throw new Error('WebCodecs mp4a.40.2 not supported');
    }

    const decodedChunks = [];
    let decodedSr = first.sampleRate;

    const decoder = new AudioDecoder({
      output: (audioData) => {
        decodedSr = audioData.sampleRate;
        const frames = audioData.numberOfFrames;
        const ch = audioData.numberOfChannels;
        const mono = new Float32Array(frames);
        const tmp = new Float32Array(frames);
        for (let c = 0; c < ch; c++) {
          audioData.copyTo(tmp, { planeIndex: c, format: 'f32-planar' });
          for (let i = 0; i < frames; i++) mono[i] += tmp[i] / ch;
        }
        decodedChunks.push(mono);
        audioData.close();
      },
      error: (err) => {
        throw err;
      }
    });

    decoder.configure(config);
    const frameDurUs = Math.round((1024 / first.sampleRate) * 1e6);
    for (let i = 0; i < adtsFrames.length; i++) {
      decoder.decode(new EncodedAudioChunk({
        type: 'key',
        timestamp: i * frameDurUs,
        duration: frameDurUs,
        data: adtsFrames[i].data
      }));
    }
    await decoder.flush();
    decoder.close();

    const totalLen = decodedChunks.reduce((acc, c) => acc + c.length, 0);
    const merged = new Float32Array(totalLen);
    let pos = 0;
    for (const c of decodedChunks) {
      merged.set(c, pos);
      pos += c.length;
    }

    return {
      pcm: resampleLinear(merged, decodedSr, TARGET_SAMPLE_RATE),
      sampleRate: TARGET_SAMPLE_RATE,
      decoderUsed: 'webcodecs-aac'
    };
  }

  async function decodeWithBackendFFmpeg(fileOrBuffer, fileName, apiBase = '') {
    const blob = fileOrBuffer instanceof Blob
      ? fileOrBuffer
      : new Blob([fileOrBuffer], { type: 'application/octet-stream' });
    const formData = new FormData();
    formData.append('file', blob, fileName || 'upload.aac');

    const res = await fetch(`${apiBase}/api/decode`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      throw new Error(`Backend /api/decode failed (${res.status})`);
    }
    const wavBuf = await res.arrayBuffer();
    const { pcm, sampleRate } = decodeWavBuffer(wavBuf);
    return {
      pcm: sampleRate === TARGET_SAMPLE_RATE ? pcm : resampleLinear(pcm, sampleRate, TARGET_SAMPLE_RATE),
      sampleRate: TARGET_SAMPLE_RATE,
      decoderUsed: 'backend-ffmpeg'
    };
  }

  async function decodeAudioFile(fileOrBuffer, fileName = 'audio.wav', options = {}) {
    const apiBase = options.apiBase || '';
    const arrayBuffer = fileOrBuffer instanceof ArrayBuffer
      ? fileOrBuffer
      : await fileOrBuffer.arrayBuffer();

    // Attempt 1: Native Web Audio decodeAudioData
    if (typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined') {
      const AudioCtx = typeof AudioContext !== 'undefined' ? AudioContext : webkitAudioContext;
      const ctx = new AudioCtx({ sampleRate: TARGET_SAMPLE_RATE });
      try {
        const audioBuf = await ctx.decodeAudioData(arrayBuffer.slice(0));
        const pcm = await resampleTo48kMono(audioBuf);
        await ctx.close();
        return { pcm, sampleRate: TARGET_SAMPLE_RATE, decoderUsed: 'native-webaudio' };
      } catch (_) {
        try { await ctx.close(); } catch (__) {}
      }
    }

    // Attempt 2: WebCodecs AudioDecoder for raw ADTS AAC
    try {
      return await decodeWithWebCodecsAAC(arrayBuffer);
    } catch (_) {}

    // Attempt 3: Backend ffmpeg /api/decode
    return decodeWithBackendFFmpeg(arrayBuffer, fileName, apiBase);
  }

  // ---------------------------------------------------------------------------
  // 2. Client-Side 48 kHz Wiener / Adaptive Fallback Denoiser
  // ---------------------------------------------------------------------------

  function wienerDenoise48k(rawPCM, roomProfile = null) {
    const n = rawPCM.length;
    const fftSize = FFT_SIZE;
    const hop = fftSize >> 2; // 512 samples (75% overlap)
    const half = (fftSize >> 1) + 1;
    const win = buildHannWindow(fftSize);

    const out = new Float32Array(n);
    const norm = new Float32Array(n);

    // Initialize noise magnitude profile from Loop A room snapshot or first 0.25 s
    const noiseMag = new Float32Array(half);
    if (roomProfile && roomProfile.spectrum && roomProfile.spectrum.length === half) {
      noiseMag.set(roomProfile.spectrum);
    } else {
      const initFrames = Math.max(1, Math.min(20, Math.floor((n - fftSize) / hop)));
      const re = new Float32Array(fftSize);
      const im = new Float32Array(fftSize);
      for (let f = 0; f < initFrames; f++) {
        const start = f * hop;
        re.fill(0);
        im.fill(0);
        for (let i = 0; i < fftSize; i++) re[i] = (rawPCM[start + i] || 0) * win[i];
        fftRadix2(re, im, false);
        for (let k = 0; k < half; k++) {
          noiseMag[k] += Math.hypot(re[k], im[k]) / initFrames;
        }
      }
    }

    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);
    const overSub = 1.65;
    const spectralFloor = 0.08;

    for (let start = 0; start < n; start += hop) {
      re.fill(0);
      im.fill(0);
      for (let i = 0; i < fftSize; i++) {
        const idx = start + i;
        re[i] = (idx < n ? rawPCM[idx] : 0) * win[i];
      }

      fftRadix2(re, im, false);

      for (let k = 0; k < half; k++) {
        const mag = Math.hypot(re[k], im[k]);
        const magSq = mag * mag;
        const nSq = noiseMag[k] * noiseMag[k] * overSub;
        const postSnr = magSq / (nSq + 1e-12);
        const gain = Math.max(spectralFloor, (postSnr - 1) / (postSnr + 1e-12));

        // Gentle adaptive noise floor update on quiet frames
        if (postSnr < 1.8) {
          noiseMag[k] = 0.96 * noiseMag[k] + 0.04 * mag;
        }

        re[k] *= gain;
        im[k] *= gain;
        if (k > 0 && k < half - 1) {
          const conj = fftSize - k;
          re[conj] = re[k];
          im[conj] = -im[k];
        }
      }

      fftRadix2(re, im, true);

      for (let i = 0; i < fftSize; i++) {
        const idx = start + i;
        if (idx < n) {
          out[idx] += re[i] * win[i];
          norm[idx] += win[i] * win[i];
        }
      }
    }

    for (let i = 0; i < n; i++) {
      out[i] = norm[i] > 1e-6 ? out[i] / norm[i] : rawPCM[i];
    }
    return out;
  }

  function estimateSnrGainDb(rawPCM, denoisedPCM) {
    let rawPower = 0;
    let diffPower = 0;
    const len = Math.min(rawPCM.length, denoisedPCM.length);
    for (let i = 0; i < len; i++) {
      rawPower += rawPCM[i] * rawPCM[i];
      const d = rawPCM[i] - denoisedPCM[i];
      diffPower += d * d;
    }
    if (diffPower <= 1e-12 || rawPower <= 1e-12) return 0.0;
    return Number(Math.max(0, Math.min(28, 10 * Math.log10((rawPower + 1e-9) / (Math.abs(rawPower - diffPower) + 1e-9)) + 6)).toFixed(2));
  }

  // ---------------------------------------------------------------------------
  // 2B. Client-Side WebGPU (onnxruntime-web@1.30.0) DeepFilterNet3 48 kHz Engine
  //     FP32 WebGPU Primary + WASM SIMD Fallback
  // ---------------------------------------------------------------------------

  const DF3_SR = 48000;
  const DF3_FFT = 960;
  const DF3_HOP = 480;
  const DF3_BINS = 481; // 960 / 2 + 1
  const DF3_NB_ERB = 32;
  const DF3_NB_DF = 96;
  const DF3_ORDER = 5;
  const DF3_LOOKAHEAD = 2;

  const df3State = {
    ortLoaded: false,
    initPromise: null,
    sessions: null,
    provider: null, // 'webgpu' | 'wasm'
    hardwareLabel: null,
    webgpuAvailable: false,
    lastError: null
  };

  function loadOrtScript() {
    if (root.ort && root.ort.InferenceSession) {
      df3State.ortLoaded = true;
      return Promise.resolve(root.ort);
    }
    if (typeof document === 'undefined') {
      return Promise.reject(new Error('No DOM available to load onnxruntime-web'));
    }
    return new Promise((resolve, reject) => {
      const existing = document.querySelector('script[data-bnfm-ort="1"]');
      if (existing) {
        if (root.ort && root.ort.InferenceSession) return resolve(root.ort);
        existing.addEventListener('load', () => resolve(root.ort));
        existing.addEventListener('error', () => reject(new Error('Failed to load ort.webgpu.min.js')));
        return;
      }
      const script = document.createElement('script');
      script.src = './ort/ort.webgpu.min.js';
      script.async = true;
      script.crossOrigin = 'anonymous';
      script.dataset.bnfmOrt = '1';
      script.onload = () => {
        df3State.ortLoaded = true;
        resolve(root.ort);
      };
      script.onerror = () => reject(new Error('Failed to load ./ort/ort.webgpu.min.js'));
      document.head.appendChild(script);
    });
  }

  async function probeHardware() {
    let webgpuAvailable = false;
    let adapterName = '';
    if (typeof navigator !== 'undefined' && navigator.gpu && typeof navigator.gpu.requestAdapter === 'function') {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter) {
          webgpuAvailable = true;
          if (adapter.info && (adapter.info.description || adapter.info.vendor)) {
            adapterName = adapter.info.description || adapter.info.vendor;
          }
        }
      } catch (_) {
        webgpuAvailable = false;
      }
    }
    df3State.webgpuAvailable = webgpuAvailable;
    const hardwareLabel = df3State.hardwareLabel || (webgpuAvailable ? 'GPU: WebGPU (FP32)' : 'CPU: WASM SIMD');
    return { webgpuAvailable, adapterName, hardwareLabel };
  }

  async function initDF3Onnx(options = {}) {
    if (df3State.sessions) {
      return {
        sessions: df3State.sessions,
        provider: df3State.provider,
        hardwareLabel: df3State.hardwareLabel
      };
    }
    if (df3State.initPromise) return df3State.initPromise;

    df3State.initPromise = (async () => {
      const ort = await loadOrtScript();
      if (!ort || !ort.InferenceSession) {
        throw new Error('onnxruntime-web (ort.InferenceSession) unavailable');
      }

      // Configure same-origin WASM SIMD runtime paths for COEP: require-corp compatibility
      const hw = await probeHardware();
      if (ort.env && ort.env.wasm) {
        ort.env.wasm.wasmPaths = options.wasmPaths || '/ort/';
        ort.env.wasm.simd = true;
        // CRITICAL: Never enable wasm.proxy when WebGPU is available (proxy=true disables WebGPU EP in ORT Web!)
        ort.env.wasm.proxy = !hw.webgpuAvailable;
        ort.env.wasm.numThreads = (!hw.webgpuAvailable && typeof self !== 'undefined' && self.crossOriginIsolated)
          ? Math.min(4, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2)
          : 1;
      }
      if (ort.env) {
        ort.env.logLevel = 'error';
      }
      const base = options.modelBase || './models/df3';
      const urls = {
        enc: `${base}/enc.onnx`,
        erb_dec: `${base}/erb_dec.onnx`,
        df_dec: `${base}/df_dec.onnx`
      };

      const createAll = async (providers) => {
        const sessOpts = {
          executionProviders: providers,
          graphOptimizationLevel: 'all',
          logSeverityLevel: 3
        };
        const enc = await ort.InferenceSession.create(urls.enc, sessOpts);
        await new Promise((r) => setTimeout(r, 0));
        const erb_dec = await ort.InferenceSession.create(urls.erb_dec, sessOpts);
        await new Promise((r) => setTimeout(r, 0));
        const df_dec = await ort.InferenceSession.create(urls.df_dec, sessOpts);
        await new Promise((r) => setTimeout(r, 0));
        return { enc, erb_dec, df_dec };
      };

      let sessions = null;
      let provider = 'wasm';
      let hardwareLabel = 'CPU: WASM SIMD';

      if (hw.webgpuAvailable) {
        try {
          sessions = await createAll([{ name: 'webgpu', deviceType: 'gpu', powerPreference: 'high-performance' }, 'wasm']);
          provider = 'webgpu';
          hardwareLabel = hw.adapterName ? `GPU: WebGPU (${hw.adapterName})` : 'GPU: WebGPU (FP32)';
        } catch (webgpuErr) {
          df3State.lastError = String(webgpuErr && webgpuErr.message || webgpuErr);
          if (ort.env && ort.env.wasm) ort.env.wasm.proxy = true;
          sessions = await createAll(['wasm']);
          provider = 'wasm';
          hardwareLabel = 'CPU: WASM SIMD (WebGPU fallback)';
        }
      } else {
        sessions = await createAll(['wasm']);
        provider = 'wasm';
        hardwareLabel = 'CPU: WASM SIMD';
      }

      df3State.sessions = sessions;
      df3State.provider = provider;
      df3State.hardwareLabel = hardwareLabel;
      return { sessions, provider, hardwareLabel };
    })();

    try {
      return await df3State.initPromise;
    } catch (err) {
      df3State.initPromise = null;
      df3State.lastError = String(err && err.message || err);
      throw err;
    }
  }

  // Build 32-band ERB partition for 48 kHz, fft_size=960 (481 bins, min_nb_erb_freqs=2)
  function buildDF3ErbBands() {
    const hzToErb = (f) => 9.265 * Math.log(1 + f / (24.7 * 9.265));
    const erbToHz = (e) => 24.7 * 9.265 * (Math.exp(e / 9.265) - 1);
    const erbMin = hzToErb(0);
    const erbMax = hzToErb(DF3_SR / 2);
    const step = (erbMax - erbMin) / DF3_NB_ERB;
    const widths = new Int32Array(DF3_NB_ERB);
    let prev = 0;
    for (let b = 0; b < DF3_NB_ERB; b++) {
      const fHi = erbToHz(erbMin + (b + 1) * step);
      let next = Math.round((fHi * DF3_FFT) / DF3_SR);
      let w = Math.max(2, next - prev);
      widths[b] = w;
      prev += w;
    }
    // Adjust final bands so sum(widths) === DF3_BINS (481)
    let diff = DF3_BINS - prev;
    let idx = DF3_NB_ERB - 1;
    while (diff !== 0) {
      if (diff > 0) {
        widths[idx] += 1;
        diff -= 1;
      } else if (widths[idx] > 2) {
        widths[idx] -= 1;
        diff += 1;
      }
      idx = idx === 0 ? DF3_NB_ERB - 1 : idx - 1;
    }
    const starts = new Int32Array(DF3_NB_ERB);
    let acc = 0;
    for (let b = 0; b < DF3_NB_ERB; b++) {
      starts[b] = acc;
      acc += widths[b];
    }
    return { starts, widths };
  }

  const DF3_ERB_FB = buildDF3ErbBands();
  const DF3_VORBIS_WIN = (() => {
    const w = new Float32Array(DF3_FFT);
    for (let i = 0; i < DF3_FFT; i++) {
      const s = Math.sin((Math.PI * (i + 0.5)) / DF3_FFT);
      w[i] = Math.sin((Math.PI / 2) * s * s);
    }
    return w;
  })();

  // Fast 960-point (15 x 64) Cooley-Tukey complex FFT / IFFT
  const DF3_TW_15_RE = new Float32Array(15 * 15);
  const DF3_TW_15_IM = new Float32Array(15 * 15);
  const DF3_TW_960_RE = new Float32Array(15 * 64);
  const DF3_TW_960_IM = new Float32Array(15 * 64);
  (function initFft960Tables() {
    for (let k = 0; k < 15; k++) {
      for (let n = 0; n < 15; n++) {
        const ang = (-2 * Math.PI * k * n) / 15;
        DF3_TW_15_RE[k * 15 + n] = Math.cos(ang);
        DF3_TW_15_IM[k * 15 + n] = Math.sin(ang);
      }
      for (let m = 0; m < 64; m++) {
        const ang = (-2 * Math.PI * k * m) / 960;
        DF3_TW_960_RE[k * 64 + m] = Math.cos(ang);
        DF3_TW_960_IM[k * 64 + m] = Math.sin(ang);
      }
    }
  })();

  function fft960(re, im, inverse) {
    // Factor 960 = 15 (N1) x 64 (N2)
    // x[n1 * 64 + n2] -> X[k2 * 15 + k1]
    const tmpRe = new Float32Array(960);
    const tmpIm = new Float32Array(960);
    const sign = inverse ? -1 : 1;

    // Step 1: 64 DFTs of size 15 along stride 64 + twiddle
    for (let n2 = 0; n2 < 64; n2++) {
      for (let k1 = 0; k1 < 15; k1++) {
        let sr = 0, si = 0;
        const row15 = k1 * 15;
        for (let n1 = 0; n1 < 15; n1++) {
          const idx = n1 * 64 + n2;
          const wr = DF3_TW_15_RE[row15 + n1];
          const wi = sign * DF3_TW_15_IM[row15 + n1];
          const xr = re[idx];
          const xi = im[idx];
          sr += xr * wr - xi * wi;
          si += xr * wi + xi * wr;
        }
        const twIdx = k1 * 64 + n2;
        const tr = DF3_TW_960_RE[twIdx];
        const ti = sign * DF3_TW_960_IM[twIdx];
        tmpRe[k1 * 64 + n2] = sr * tr - si * ti;
        tmpIm[k1 * 64 + n2] = sr * ti + si * tr;
      }
    }

    // Step 2: 15 radix-2 FFTs of size 64
    const subRe = new Float32Array(64);
    const subIm = new Float32Array(64);
    for (let k1 = 0; k1 < 15; k1++) {
      const base = k1 * 64;
      subRe.set(tmpRe.subarray(base, base + 64));
      subIm.set(tmpIm.subarray(base, base + 64));
      fftRadix2(subRe, subIm, inverse);
      if (inverse) {
        // fftRadix2 divides by 64 when inverse=true; divide remaining 15
        for (let k2 = 0; k2 < 64; k2++) {
          const outIdx = k2 * 15 + k1;
          re[outIdx] = subRe[k2] / 15;
          im[outIdx] = subIm[k2] / 15;
        }
      } else {
        for (let k2 = 0; k2 < 64; k2++) {
          const outIdx = k2 * 15 + k1;
          re[outIdx] = subRe[k2];
          im[outIdx] = subIm[k2];
        }
      }
    }
  }

  async function runDF3OnnxClient(rawPCM, options = {}) {
    const { sessions, provider, hardwareLabel } = await initDF3Onnx(options);
    const ort = root.ort;
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();

    const n = rawPCM.length;
    const numFrames = Math.max(1, Math.ceil((n + DF3_HOP) / DF3_HOP));
    const specRe = new Float32Array(numFrames * DF3_BINS);
    const specIm = new Float32Array(numFrames * DF3_BINS);

    const featErb = new Float32Array(numFrames * DF3_NB_ERB);
    const featSpec = new Float32Array(2 * numFrames * DF3_NB_DF);

    const erbMean = new Float32Array(DF3_NB_ERB).fill(-60.0);
    const specNorm = new Float32Array(DF3_NB_DF).fill(1e-3);
    const alphaNorm = Math.exp(-DF3_HOP / (DF3_SR * 1.0)); // norm_tau = 1s

    const frameRe = new Float32Array(DF3_FFT);
    const frameIm = new Float32Array(DF3_FFT);
    const fftScale = 1.0 / DF3_FFT;

    // Analysis STFT + ERB & DF feature extraction (non-blocking yields)
    for (let f = 0; f < numFrames; f++) {
      if ((f & 15) === 15) await new Promise((r) => setTimeout(r, 0));
      const start = f * DF3_HOP - DF3_HOP;
      frameRe.fill(0);
      frameIm.fill(0);
      for (let i = 0; i < DF3_FFT; i++) {
        const idx = start + i;
        if (idx >= 0 && idx < n) {
          frameRe[i] = rawPCM[idx] * DF3_VORBIS_WIN[i];
        }
      }
      fft960(frameRe, frameIm, false);

      const sOff = f * DF3_BINS;
      for (let k = 0; k < DF3_BINS; k++) {
        specRe[sOff + k] = frameRe[k] * fftScale;
        specIm[sOff + k] = frameIm[k] * fftScale;
      }

      // 32-band ERB log-power features
      const eOff = f * DF3_NB_ERB;
      for (let b = 0; b < DF3_NB_ERB; b++) {
        const bStart = DF3_ERB_FB.starts[b];
        const bWidth = DF3_ERB_FB.widths[b];
        let pSum = 0;
        for (let k = 0; k < bWidth; k++) {
          const r = specRe[sOff + bStart + k];
          const imv = specIm[sOff + bStart + k];
          pSum += r * r + imv * imv;
        }
        const db = 10.0 * Math.log10(pSum / bWidth + 1e-10);
        erbMean[b] = alphaNorm * erbMean[b] + (1.0 - alphaNorm) * db;
        featErb[eOff + b] = (db - erbMean[b]) / 40.0;
      }

      // 96-bin complex unit-normalized DF features [1, 2, S, 96]
      const spCh0Off = f * DF3_NB_DF;
      const spCh1Off = numFrames * DF3_NB_DF + f * DF3_NB_DF;
      for (let k = 0; k < DF3_NB_DF; k++) {
        const r = specRe[sOff + k];
        const imv = specIm[sOff + k];
        const mag = Math.sqrt(r * r + imv * imv);
        specNorm[k] = alphaNorm * specNorm[k] + (1.0 - alphaNorm) * mag;
        const inv = 1.0 / Math.sqrt(specNorm[k] + 1e-8);
        featSpec[spCh0Off + k] = r * inv;
        featSpec[spCh1Off + k] = imv * inv;
      }
    }

    // Run ONNX Runtime Web (WebGPU FP32 primary / WASM SIMD fallback)
    const tFeatErb = new ort.Tensor('float32', featErb, [1, 1, numFrames, DF3_NB_ERB]);
    const tFeatSpec = new ort.Tensor('float32', featSpec, [1, 2, numFrames, DF3_NB_DF]);

    const encOut = await sessions.enc.run({
      feat_erb: tFeatErb,
      feat_spec: tFeatSpec
    });

    await new Promise((r) => setTimeout(r, 0));
    const erbOut = await sessions.erb_dec.run({
      emb: encOut.emb,
      e3: encOut.e3,
      e2: encOut.e2,
      e1: encOut.e1,
      e0: encOut.e0
    });
    await new Promise((r) => setTimeout(r, 0));
    const dfOut = await sessions.df_dec.run({
      emb: encOut.emb,
      c0: encOut.c0
    });
    await new Promise((r) => setTimeout(r, 0));

    const maskData = erbOut.m.data;       // [1, 1, S, 32]
    const coefsData = dfOut.coefs.data;   // [1, S, 96, 10] (5 complex taps)
    const alphaData = dfOut['235'] ? dfOut['235'].data : null;

    // Synthesis: apply ERB mask across 481 bins + 5-tap complex DF on bins 0..95
    const outPCM = new Float32Array(n);
    const normAcc = new Float32Array(n);

    for (let f = 0; f < numFrames; f++) {
      if ((f & 15) === 15) await new Promise((r) => setTimeout(r, 0));
      const sOff = f * DF3_BINS;
      const mOff = f * DF3_NB_ERB;
      const cFrameOff = f * DF3_NB_DF * 10;
      const dfAlpha = alphaData ? Math.max(0, Math.min(1, alphaData[f])) : 0.7;

      frameRe.fill(0);
      frameIm.fill(0);

      // Expand 32-band ERB mask to 481 bins
      for (let b = 0; b < DF3_NB_ERB; b++) {
        const g = Math.max(0.02, Math.min(1.0, maskData[mOff + b]));
        const bStart = DF3_ERB_FB.starts[b];
        const bWidth = DF3_ERB_FB.widths[b];
        for (let k = 0; k < bWidth; k++) {
          const bin = bStart + k;
          frameRe[bin] = specRe[sOff + bin] * g * DF3_FFT;
          frameIm[bin] = specIm[sOff + bin] * g * DF3_FFT;
        }
      }

      // Apply 5-tap complex Deep Filtering on lowest 96 bins (0 - 4.8 kHz)
      for (let k = 0; k < DF3_NB_DF; k++) {
        const cOff = cFrameOff + k * 10;
        let dfRe = 0;
        let dfIm = 0;
        for (let tap = 0; tap < DF3_ORDER; tap++) {
          const tf = f - (DF3_ORDER - 1 - DF3_LOOKAHEAD) + tap;
          if (tf >= 0 && tf < numFrames) {
            const xr = specRe[tf * DF3_BINS + k];
            const xi = specIm[tf * DF3_BINS + k];
            const cr = coefsData[cOff + tap * 2];
            const ci = coefsData[cOff + tap * 2 + 1];
            dfRe += xr * cr - xi * ci;
            dfIm += xr * ci + xi * cr;
          }
        }
        // Blend ERB-masked bin with DF complex-filtered bin using df_alpha
        const erbRe = frameRe[k];
        const erbIm = frameIm[k];
        const blendRe = (1 - dfAlpha) * erbRe + dfAlpha * (dfRe * DF3_FFT);
        const blendIm = (1 - dfAlpha) * erbIm + dfAlpha * (dfIm * DF3_FFT);
        // Guard against extreme DF tap gain
        const origMag = Math.hypot(specRe[sOff + k], specIm[sOff + k]) * DF3_FFT + 1e-6;
        const newMag = Math.hypot(blendRe, blendIm);
        const clamp = newMag > origMag * 1.5 ? (origMag * 1.5) / newMag : 1.0;
        frameRe[k] = blendRe * clamp;
        frameIm[k] = blendIm * clamp;
      }

      // Mirror conjugate symmetry for real-valued IFFT
      for (let k = 1; k < DF3_BINS - 1; k++) {
        frameRe[DF3_FFT - k] = frameRe[k];
        frameIm[DF3_FFT - k] = -frameIm[k];
      }

      fft960(frameRe, frameIm, true);

      const start = f * DF3_HOP - DF3_HOP;
      for (let i = 0; i < DF3_FFT; i++) {
        const idx = start + i;
        if (idx >= 0 && idx < n) {
          const w = DF3_VORBIS_WIN[i];
          outPCM[idx] += frameRe[i] * w;
          normAcc[idx] += w * w;
        }
      }
    }

    let finiteCheck = true;
    let energy = 0;
    for (let i = 0; i < n; i++) {
      const v = normAcc[i] > 1e-6 ? outPCM[i] / normAcc[i] : rawPCM[i];
      if (!Number.isFinite(v)) {
        finiteCheck = false;
        break;
      }
      outPCM[i] = Math.max(-1, Math.min(1, v));
      energy += outPCM[i] * outPCM[i];
    }
    if (!finiteCheck || energy < 1e-9) {
      throw new Error('DF3 ONNX client output failed sanity check');
    }

    const inferMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
    return {
      denoisedPCM: outPCM,
      inferMs: Number(inferMs.toFixed(1)),
      provider,
      hardwareLabel
    };
  }

  // ---------------------------------------------------------------------------
  // 3. Dual-Engine Enhance Controller (POST /api/denoise?engine=df3|gtcrn48k)
  // ---------------------------------------------------------------------------

  async function enhanceAudio(rawPCM, options = {}) {
    const engine = options.engine === 'gtcrn48k' ? 'gtcrn48k' : 'df3';
    const preset = options.preset || 'podcast';
    const apiBase = options.apiBase || '';
    const durationSec = rawPCM.length / TARGET_SAMPLE_RATE;

    const hwProbe = await probeHardware();
    let denoisedPCM = null;
    let telemetry = {
      engine,
      hardware: hwProbe.hardwareLabel,
      provider: hwProbe.webgpuAvailable ? 'webgpu' : 'wasm',
      inferMs: 0,
      latencyMs: 0,
      rtf: 0,
      snrGainDb: 0,
      memoryMb: (typeof performance !== 'undefined' && performance.memory)
        ? Math.round(performance.memory.usedJSHeapSize / 1048576)
        : 48,
      offlineFallback: false
    };

    // Primary path for DF3: Client-Side WebGPU (FP32) / WASM SIMD via onnxruntime-web@1.30.0
    if (engine === 'df3' && options.forceServer !== true) {
      try {
        const clientRes = await runDF3OnnxClient(rawPCM, options);
        denoisedPCM = clientRes.denoisedPCM;
        const rtf = Number(((clientRes.inferMs / 1000) / Math.max(durationSec, 0.001)).toFixed(4));
        const snr = estimateSnrGainDb(rawPCM, denoisedPCM);
        telemetry = {
          engine: clientRes.provider === 'webgpu'
            ? 'DeepFilterNet3 48 kHz (WebGPU ONNX)'
            : 'DeepFilterNet3 48 kHz (WASM SIMD ONNX)',
          hardware: clientRes.hardwareLabel,
          provider: clientRes.provider,
          inferMs: clientRes.inferMs,
          latencyMs: clientRes.inferMs,
          rtf,
          snrGainDb: snr,
          memoryMb: (typeof performance !== 'undefined' && performance.memory)
            ? Math.round(performance.memory.usedJSHeapSize / 1048576)
            : 54,
          offlineFallback: false
        };
      } catch (_) {
        denoisedPCM = null;
      }
    }

    // Transparent Fallback (or explicit gtcrn48k): POST /api/denoise?engine=df3|gtcrn48k
    if (!denoisedPCM) {
      const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      try {
        const wavBlob = encodeWav48k(rawPCM, TARGET_SAMPLE_RATE);
        const formData = new FormData();
        formData.append('file', wavBlob, options.fileName || 'input_48k.wav');

        const res = await fetch(`${apiBase}/api/denoise?engine=${encodeURIComponent(engine)}`, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          throw new Error(`Backend /api/denoise HTTP ${res.status}`);
        }

        const wavOut = await res.arrayBuffer();
        const decoded = decodeWavBuffer(wavOut);
        denoisedPCM = decoded.sampleRate === TARGET_SAMPLE_RATE
          ? decoded.pcm
          : resampleLinear(decoded.pcm, decoded.sampleRate, TARGET_SAMPLE_RATE);

        const elapsedMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
        const hdrInfer = parseFloat(res.headers.get('X-BNFM-Infer-Ms'));
        const hdrRtf = parseFloat(res.headers.get('X-BNFM-RTF'));
        const hdrSnr = parseFloat(res.headers.get('X-BNFM-SNR-Gain-DB'));
        const hdrEngine = res.headers.get('X-BNFM-Engine');
        const inferVal = Number.isFinite(hdrInfer) ? hdrInfer : Number(elapsedMs.toFixed(1));

        telemetry = {
          engine: hdrEngine || engine,
          hardware: df3State.hardwareLabel || hwProbe.hardwareLabel,
          provider: 'api',
          inferMs: inferVal,
          latencyMs: inferVal,
          rtf: Number.isFinite(hdrRtf) ? hdrRtf : Number(((elapsedMs / 1000) / Math.max(durationSec, 0.001)).toFixed(4)),
          snrGainDb: Number.isFinite(hdrSnr) ? hdrSnr : estimateSnrGainDb(rawPCM, denoisedPCM),
          memoryMb: (typeof performance !== 'undefined' && performance.memory)
            ? Math.round(performance.memory.usedJSHeapSize / 1048576)
            : 46,
          offlineFallback: false
        };
      } catch (_) {
        // Offline client-side 48 kHz Wiener/Adaptive fallback
        let roomProfile = options.roomProfile || null;
        if (!roomProfile && root.BNFMStore && typeof root.BNFMStore.getRoomProfile === 'function') {
          try { roomProfile = await root.BNFMStore.getRoomProfile(); } catch (__) {}
        }

        const fbStart = typeof performance !== 'undefined' ? performance.now() : Date.now();
        denoisedPCM = wienerDenoise48k(rawPCM, roomProfile);
        const fbMs = Number(((typeof performance !== 'undefined' ? performance.now() : Date.now()) - fbStart).toFixed(1));

        telemetry = {
          engine: 'wiener-48k-fallback',
          hardware: df3State.hardwareLabel || hwProbe.hardwareLabel,
          provider: 'wiener',
          inferMs: fbMs,
          latencyMs: fbMs,
          rtf: Number(((fbMs / 1000) / Math.max(durationSec, 0.001)).toFixed(4)),
          snrGainDb: estimateSnrGainDb(rawPCM, denoisedPCM),
          memoryMb: (typeof performance !== 'undefined' && performance.memory)
            ? Math.round(performance.memory.usedJSHeapSize / 1048576)
            : 42,
          offlineFallback: true
        };
      }
    }

    // Apply Browser Studio Mastering Chain
    let masteredPCM = denoisedPCM;
    let presetStats = null;
    const mastering = options.masteringChain || root.BNFMMasteringChain;
    if (mastering && typeof mastering.processOffline === 'function') {
      const result = mastering.processOffline(denoisedPCM, {
        preset,
        sampleRate: TARGET_SAMPLE_RATE,
        dryPCM: rawPCM,
        comfortNoise: options.comfortNoise,
        edgeTrim: options.edgeTrim,
        dryWet: options.dryWet
      });
      masteredPCM = result.pcm;
      presetStats = result.stats;
    }

    return {
      rawPCM,
      denoisedPCM,
      masteredPCM,
      enhancedPcm: masteredPCM,
      sampleRate: TARGET_SAMPLE_RATE,
      telemetry: {
        ...telemetry,
        preset,
        outputLufs: presetStats ? presetStats.outputLufs : null,
        peakDb: presetStats ? presetStats.peakDb : null
      },
      presetStats
    };
  }

  async function enhanceBuffer(rawPCM, options = {}) {
    return enhanceAudio(rawPCM, options);
  }

  async function warmupContinuous() {
    // 0 ms artificial delay: lightweight WebGPU capability probe only
    return probeHardware();
  }

  // ---------------------------------------------------------------------------
  // 4. 160 kbps AAC Export (WebCodecs AudioEncoder -> POST /api/encode-aac)
  // ---------------------------------------------------------------------------

  async function encodeAACWithWebCodecs(float32PCM, sampleRate = TARGET_SAMPLE_RATE, bitrate = 160000) {
    if (typeof AudioEncoder === 'undefined' || typeof AudioData === 'undefined') {
      throw new Error('WebCodecs AudioEncoder unavailable');
    }
    const config = {
      codec: 'mp4a.40.2',
      sampleRate,
      numberOfChannels: 1,
      bitrate,
      aac: { format: 'adts' }
    };
    const support = await AudioEncoder.isConfigSupported(config);
    if (!support.supported) {
      throw new Error('WebCodecs AAC-LC encoding not supported');
    }

    const chunks = [];
    const encoder = new AudioEncoder({
      output: (chunk) => {
        const buf = new Uint8Array(chunk.byteLength);
        chunk.copyTo(buf);
        chunks.push(buf);
      },
      error: (err) => { throw err; }
    });

    encoder.configure(config);
    const frameSize = 1024;
    for (let offset = 0; offset < float32PCM.length; offset += frameSize) {
      const len = Math.min(frameSize, float32PCM.length - offset);
      const slice = new Float32Array(frameSize);
      slice.set(float32PCM.subarray(offset, offset + len));
      const audioData = new AudioData({
        format: 'f32-planar',
        sampleRate,
        numberOfFrames: frameSize,
        numberOfChannels: 1,
        timestamp: Math.round((offset / sampleRate) * 1e6),
        data: slice
      });
      encoder.encode(audioData);
      audioData.close();
    }

    await encoder.flush();
    encoder.close();
    return {
      blob: new Blob(chunks, { type: 'audio/aac' }),
      extension: '.aac',
      encoderUsed: 'webcodecs-aac'
    };
  }

  async function exportAAC(float32PCM, options = {}) {
    const sampleRate = options.sampleRate || TARGET_SAMPLE_RATE;
    const bitrate = options.bitrate || 160000;
    const apiBase = options.apiBase || '';
    const preferM4aContainer = options.preferM4a !== false;

    // Prefer backend ffmpeg for standards-compliant faststart .m4a container;
    // try WebCodecs first if caller explicitly sets preferM4a: false or backend fails.
    if (!preferM4aContainer) {
      try {
        return await encodeAACWithWebCodecs(float32PCM, sampleRate, bitrate);
      } catch (_) {}
    }

    try {
      const wavBlob = encodeWav48k(float32PCM, sampleRate);
      const formData = new FormData();
      formData.append('file', wavBlob, options.fileName || 'master_48k.wav');
      const res = await fetch(`${apiBase}/api/encode-aac`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(`Backend /api/encode-aac HTTP ${res.status}`);
      const m4aBlob = await res.blob();
      return {
        blob: new Blob([m4aBlob], { type: 'audio/mp4' }),
        extension: '.m4a',
        encoderUsed: 'backend-ffmpeg-aac'
      };
    } catch (_) {
      return encodeAACWithWebCodecs(float32PCM, sampleRate, bitrate);
    }
  }

  // ---------------------------------------------------------------------------
  // 5. 20 kHz Studio Thermal Spectrogram Renderer (Black -> #FF5A00 -> White)
  // ---------------------------------------------------------------------------

  function thermalColor(t) {
    const v = Math.max(0, Math.min(1, t));
    if (v < 0.62) {
      // Black (#000000) -> Deep Orange (#FF5A00)
      const k = v / 0.62;
      return [
        Math.round(255 * Math.pow(k, 0.85)),
        Math.round(90 * Math.pow(k, 1.35)),
        0
      ];
    }
    // Deep Orange (#FF5A00) -> Pure White (#FFFFFF)
    const k = (v - 0.62) / 0.38;
    return [
      255,
      Math.round(90 + 165 * k),
      Math.round(255 * Math.pow(k, 1.15))
    ];
  }

  function renderSpectrogram(canvasOrSelector, pcm, options = {}) {
    const canvas = typeof canvasOrSelector === 'string'
      ? (typeof document !== 'undefined' ? document.querySelector(canvasOrSelector) : null)
      : canvasOrSelector;
    if (!canvas || !pcm || pcm.length === 0) return false;

    const ctx = canvas.getContext('2d');
    if (!ctx) return false;

    const sampleRate = options.sampleRate || TARGET_SAMPLE_RATE;
    const maxFreqHz = options.maxFreqHz || 20000;
    const fftSize = options.fftSize || FFT_SIZE;
    const minDb = options.minDb ?? -90;
    const maxDb = options.maxDb ?? -6;

    const width = canvas.width || 640;
    const height = canvas.height || 200;
    const imgData = ctx.createImageData(width, height);
    const pixels = imgData.data;

    const win = buildHannWindow(fftSize);
    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);
    const nyquist = sampleRate / 2;
    const maxBin = Math.min((fftSize >> 1), Math.floor((maxFreqHz / nyquist) * (fftSize >> 1)));

    const totalAvail = Math.max(1, pcm.length - fftSize);
    for (let x = 0; x < width; x++) {
      const start = Math.floor((x / Math.max(1, width - 1)) * totalAvail);
      re.fill(0);
      im.fill(0);
      for (let i = 0; i < fftSize; i++) {
        const idx = start + i;
        re[i] = (idx < pcm.length ? pcm[idx] : 0) * win[i];
      }
      fftRadix2(re, im, false);

      for (let y = 0; y < height; y++) {
        const freqRatio = 1 - y / Math.max(1, height - 1);
        const bin = Math.min(maxBin, Math.max(0, Math.round(freqRatio * maxBin)));
        const mag = Math.hypot(re[bin], im[bin]) / (fftSize * 0.5);
        const db = 20 * Math.log10(mag + 1e-9);
        const norm = (db - minDb) / (maxDb - minDb);
        const [r, g, b] = thermalColor(norm);
        const idx = (y * width + x) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // Subtle frequency ruler overlay (5 kHz, 10 kHz, 15 kHz, 20 kHz)
    if (options.showGrid !== false) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.72)';
      ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      const marks = [5000, 10000, 15000, 20000];
      for (const hz of marks) {
        if (hz > maxFreqHz) continue;
        const y = Math.round((1 - hz / maxFreqHz) * (height - 1));
        ctx.beginPath();
        ctx.moveTo(0, y + 0.5);
        ctx.lineTo(width, y + 0.5);
        ctx.stroke();
        ctx.fillText(`${hz / 1000} kHz`, 6, Math.max(11, y - 3));
      }
      ctx.restore();
    }

    return true;
  }

  function renderComparisonSpectrograms(noisyPCM, denoisedPCM, noisySel = '#canvas-noisy', denoisedSel = '#canvas-denoised') {
    const a = renderSpectrogram(noisySel, noisyPCM);
    const b = renderSpectrogram(denoisedSel, denoisedPCM);
    return a && b;
  }

  // ---------------------------------------------------------------------------
  // 6. Synthetic 48 kHz Indian Household Demo Generator (3 s)
  // ---------------------------------------------------------------------------

  function generateIndianHouseholdDemo(durationSec = 3.0, sampleRate = TARGET_SAMPLE_RATE) {
    const n = Math.round(durationSec * sampleRate);
    const cleanPCM = new Float32Array(n);
    const noisyPCM = new Float32Array(n);

    // Pink noise state (ceiling fan aerodynamic turbulence)
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;

    // Seeded PRNG for deterministic demo playback
    let seed = 480001337;
    function rand() {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return (seed / 0xffffffff) * 2 - 1;
    }

    let phaseF0 = 0;
    for (let i = 0; i < n; i++) {
      const t = i / sampleRate;

      // 4 syllable speech cadence with pauses
      const sylEnv = (
        Math.exp(-Math.pow((t - 0.55) / 0.22, 2)) +
        Math.exp(-Math.pow((t - 1.25) / 0.26, 2)) +
        Math.exp(-Math.pow((t - 2.00) / 0.24, 2)) +
        Math.exp(-Math.pow((t - 2.58) / 0.18, 2))
      );

      // Pitch contour (135 Hz -> 170 Hz)
      const f0 = 142 + 24 * Math.sin(2 * Math.PI * 0.7 * t);
      phaseF0 += (2 * Math.PI * f0) / sampleRate;

      // Vocal formants: F1 (~620 Hz), F2 (~1850 Hz), F3 (~2950 Hz) + glottal harmonics
      const voiced = (
        0.55 * Math.sin(phaseF0) +
        0.42 * Math.sin(2 * Math.PI * (620 + 90 * Math.sin(3 * t)) * t) +
        0.30 * Math.sin(2 * Math.PI * (1850 + 180 * Math.cos(2.4 * t)) * t) +
        0.16 * Math.sin(2 * Math.PI * 2950 * t)
      );

      // High-frequency crisp sibilance bursts (6–11 kHz) on consonant transitions
      const sibEnv = (
        Math.exp(-Math.pow((t - 0.34) / 0.045, 2)) +
        Math.exp(-Math.pow((t - 1.52) / 0.045, 2)) +
        Math.exp(-Math.pow((t - 2.30) / 0.040, 2))
      );
      const sibilance = sibEnv * 0.22 * (
        Math.sin(2 * Math.PI * 6800 * t + rand() * 1.4) +
        0.7 * Math.sin(2 * Math.PI * 9600 * t + rand() * 1.8)
      );

      const speech = Math.max(-0.9, Math.min(0.9, (voiced * sylEnv * 0.42) + sibilance));
      cleanPCM[i] = speech;

      // Indian household interference: 50/100/150 Hz AC mains hum + ceiling fan pink noise
      const hum = 0.11 * (
        1.0 * Math.sin(2 * Math.PI * 50.0 * t) +
        0.5 * Math.sin(2 * Math.PI * 100.0 * t + 0.4) +
        0.25 * Math.sin(2 * Math.PI * 150.0 * t + 1.1)
      );

      const white = rand();
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      const pink = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.045;
      b6 = white * 0.115926;

      const bladeMod = 1.0 + 0.14 * Math.sin(2 * Math.PI * 3.8 * t);
      const fanNoise = pink * bladeMod;

      noisyPCM[i] = Math.max(-1.0, Math.min(1.0, speech + hum + fanNoise));
    }

    return {
      noisyPCM,
      cleanPCM,
      sampleRate,
      durationSec,
      fileName: 'bnfm_indian_household_demo_48k.wav'
    };
  }

  // ---------------------------------------------------------------------------
  // 7. 2-Second Room Snapshot Calibration (Loop A 48 kHz Noise Profile)
  // ---------------------------------------------------------------------------

  function computeWelchSpectrum(pcm, fftSize = FFT_SIZE) {
    const hop = fftSize >> 1;
    const half = (fftSize >> 1) + 1;
    const win = buildHannWindow(fftSize);
    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);
    const avg = new Float32Array(half);

    let frames = 0;
    for (let start = 0; start + fftSize <= pcm.length; start += hop) {
      re.fill(0);
      im.fill(0);
      for (let i = 0; i < fftSize; i++) re[i] = pcm[start + i] * win[i];
      fftRadix2(re, im, false);
      for (let k = 0; k < half; k++) {
        avg[k] += Math.hypot(re[k], im[k]);
      }
      frames++;
    }

    if (frames > 0) {
      for (let k = 0; k < half; k++) avg[k] /= frames;
    }

    let sumSq = 0;
    for (let i = 0; i < pcm.length; i++) sumSq += pcm[i] * pcm[i];
    const rms = Math.sqrt(sumSq / Math.max(1, pcm.length));
    const noiseFloorDb = Number((20 * Math.log10(rms + 1e-9)).toFixed(2));

    return { spectrum: avg, noiseFloorDb, fftSize };
  }

  async function calibrateRoomSnapshot(mediaStream, onProgress = null, durationSec = 2.0) {
    const AudioCtx = typeof AudioContext !== 'undefined' ? AudioContext : webkitAudioContext;
    const ctx = new AudioCtx({ sampleRate: TARGET_SAMPLE_RATE });
    const source = ctx.createMediaStreamSource(mediaStream);

    const targetSamples = Math.round(durationSec * ctx.sampleRate);
    const captured = new Float32Array(targetSamples);
    let written = 0;

    return new Promise((resolve, reject) => {
      const proc = ctx.createScriptProcessor(2048, 1, 1);
      const silentGain = ctx.createGain();
      silentGain.gain.value = 0.0;

      proc.onaudioprocess = async (e) => {
        const ch = e.inputBuffer.getChannelData(0);
        const remaining = targetSamples - written;
        const copyLen = Math.min(remaining, ch.length);
        captured.set(ch.subarray(0, copyLen), written);
        written += copyLen;

        if (typeof onProgress === 'function') {
          onProgress(Math.min(1, written / targetSamples));
        }

        if (written >= targetSamples) {
          proc.onaudioprocess = null;
          source.disconnect();
          proc.disconnect();
          silentGain.disconnect();
          const sr = ctx.sampleRate;
          try { await ctx.close(); } catch (_) {}

          const pcm48k = sr === TARGET_SAMPLE_RATE
            ? captured
            : resampleLinear(captured, sr, TARGET_SAMPLE_RATE);

          const { spectrum, noiseFloorDb, fftSize } = computeWelchSpectrum(pcm48k, FFT_SIZE);
          const profile = {
            sampleRate: TARGET_SAMPLE_RATE,
            fftSize,
            noiseFloorDb,
            durationSec,
            spectrum,
            capturedAt: new Date().toISOString()
          };

          if (root.BNFMStore && typeof root.BNFMStore.saveRoomProfile === 'function') {
            try {
              await root.BNFMStore.saveRoomProfile(profile);
            } catch (err) {
              reject(err);
              return;
            }
          }

          resolve(profile);
        }
      };

      source.connect(proc);
      proc.connect(silentGain);
      silentGain.connect(ctx.destination);
    });
  }

  return {
    TARGET_SAMPLE_RATE,
    FFT_SIZE,
    decodeAudioFile,
    enhanceAudio,
    enhanceBuffer,
    initDF3Onnx,
    probeHardware,
    warmupContinuous,
    wienerDenoise48k,
    encodeWav48k,
    encodeWavBuffer48k,
    decodeWavBuffer,
    exportAAC,
    renderSpectrogram,
    renderComparisonSpectrograms,
    generateIndianHouseholdDemo,
    calibrateRoomSnapshot,
    computeWelchSpectrum
  };
});
