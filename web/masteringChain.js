/**
 * Browser Noise Free Mic — Browser Studio Mastering Chain (masteringChain.js)
 *
 * Dual-mode 48 kHz studio vocal mastering engine:
 *   1. Offline Float32Array DSP pipeline (`processOffline`)
 *   2. Zero-latency Web Audio API live graph (`createLiveChain`)
 *
 * Signal Chain:
 *   - 80 Hz High-Pass Filter (24 dB/oct 4th-order Butterworth cascaded biquad)
 *   - Optional Noise Gate (Heavy preset)
 *   - Studio Voice Parametric EQ:
 *       * -2.0 dB @ 280 Hz (Q = 1.2) boxiness cut
 *       * +1.5 dB @ 3.5 kHz (Q = 0.7) vocal presence
 *       * +1.5 dB @ 11 kHz high-shelf studio air
 *   - Soft-Knee Vocal Compressor + True-Peak Limiter (-1.0 dBFS)
 *   - Auto Loudness (-16 LUFS K-weighted) & Smooth tanh Soft De-Clipper
 *   - Toggleable -65 dBFS Comfort Noise Floor (shaped pink dither)
 *   - Smart Edge Silence Trim (-48 dBFS threshold, 40 ms smooth fade padding)
 *
 * 4 Studio Presets: podcast, natural, heavy, raw
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.BNFMMasteringChain = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const DEFAULT_SR = 48000;
  const TRUE_PEAK_CEILING_DB = -1.0;
  const TRUE_PEAK_CEILING_LIN = Math.pow(10, TRUE_PEAK_CEILING_DB / 20); // ~0.89125
  const COMFORT_NOISE_DB = -65.0;
  const COMFORT_NOISE_RMS = Math.pow(10, COMFORT_NOISE_DB / 20); // ~0.0005623
  const EDGE_TRIM_THRESHOLD_DB = -48.0;
  const EDGE_TRIM_PAD_MS = 40;

  // 4th-order Butterworth cascaded 2nd-order Q factors
  const BUTTERWORTH_4TH_Q1 = 0.54119610;
  const BUTTERWORTH_4TH_Q2 = 1.30656296;

  const PRESETS = Object.freeze({
    podcast: Object.freeze({
      id: 'podcast',
      label: 'Podcast Studio Cut',
      description: 'Broadcast-ready warmth, presence, -16 LUFS loudness & -1 dBFS ceiling.',
      dryWet: 1.0,
      bypassAll: false,
      hpfHz: 80,
      gateEnabled: false,
      gateThresholdDb: -50,
      gateFloorDb: -12,
      eq: {
        boxinessHz: 280,
        boxinessGainDb: -2.0,
        boxinessQ: 1.2,
        presenceHz: 3500,
        presenceGainDb: 1.5,
        presenceQ: 0.7,
        airHz: 11000,
        airGainDb: 1.5,
        airQ: 0.707
      },
      comp: {
        thresholdDb: -18.0,
        ratio: 3.0,
        kneeDb: 6.0,
        attackMs: 10.0,
        releaseMs: 120.0,
        makeupDb: 2.0
      },
      targetLufs: -16.0,
      limiterCeilingDb: TRUE_PEAK_CEILING_DB,
      comfortNoise: true,
      edgeTrim: true
    }),

    natural: Object.freeze({
      id: 'natural',
      label: 'Natural Voice',
      description: '82% wet blend with gentle EQ and light dynamics to preserve room intimacy.',
      dryWet: 0.82,
      bypassAll: false,
      hpfHz: 80,
      gateEnabled: false,
      gateThresholdDb: -52,
      gateFloorDb: -8,
      eq: {
        boxinessHz: 280,
        boxinessGainDb: -1.0,
        boxinessQ: 1.2,
        presenceHz: 3500,
        presenceGainDb: 0.8,
        presenceQ: 0.7,
        airHz: 11000,
        airGainDb: 0.8,
        airQ: 0.707
      },
      comp: {
        thresholdDb: -20.0,
        ratio: 2.0,
        kneeDb: 8.0,
        attackMs: 15.0,
        releaseMs: 140.0,
        makeupDb: 1.0
      },
      targetLufs: -16.0,
      limiterCeilingDb: TRUE_PEAK_CEILING_DB,
      comfortNoise: true,
      edgeTrim: true
    }),

    heavy: Object.freeze({
      id: 'heavy',
      label: 'Heavy Fan / Traffic',
      description: '100% wet with tighter 110 Hz HPF, soft noise gate, and firmer vocal compression.',
      dryWet: 1.0,
      bypassAll: false,
      hpfHz: 110,
      gateEnabled: true,
      gateThresholdDb: -44.0,
      gateFloorDb: -14.0,
      eq: {
        boxinessHz: 280,
        boxinessGainDb: -2.5,
        boxinessQ: 1.2,
        presenceHz: 3500,
        presenceGainDb: 2.0,
        presenceQ: 0.7,
        airHz: 11000,
        airGainDb: 1.2,
        airQ: 0.707
      },
      comp: {
        thresholdDb: -16.0,
        ratio: 3.5,
        kneeDb: 5.0,
        attackMs: 8.0,
        releaseMs: 110.0,
        makeupDb: 2.5
      },
      targetLufs: -16.0,
      limiterCeilingDb: TRUE_PEAK_CEILING_DB,
      comfortNoise: true,
      edgeTrim: true
    }),

    raw: Object.freeze({
      id: 'raw',
      label: 'Raw Neural',
      description: 'Pure neural denoiser output with EQ, compression, and loudness bypassed.',
      dryWet: 1.0,
      bypassAll: true,
      hpfHz: 0,
      gateEnabled: false,
      eq: null,
      comp: null,
      targetLufs: null,
      limiterCeilingDb: TRUE_PEAK_CEILING_DB,
      comfortNoise: false,
      edgeTrim: false
    })
  });

  // ---------------------------------------------------------------------------
  // Biquad Filter Design (RBJ Audio EQ Cookbook — Direct Form II Transposed)
  // ---------------------------------------------------------------------------

  function designHighpass(freq, Q, sampleRate) {
    const w0 = (2 * Math.PI * freq) / sampleRate;
    const cosW0 = Math.cos(w0);
    const sinW0 = Math.sin(w0);
    const alpha = sinW0 / (2 * Q);
    const a0 = 1 + alpha;

    return {
      b0: ((1 + cosW0) / 2) / a0,
      b1: (-(1 + cosW0)) / a0,
      b2: ((1 + cosW0) / 2) / a0,
      a1: (-2 * cosW0) / a0,
      a2: (1 - alpha) / a0
    };
  }

  function designPeaking(freq, gainDb, Q, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const w0 = (2 * Math.PI * freq) / sampleRate;
    const cosW0 = Math.cos(w0);
    const sinW0 = Math.sin(w0);
    const alpha = sinW0 / (2 * Q);
    const a0 = 1 + alpha / A;

    return {
      b0: (1 + alpha * A) / a0,
      b1: (-2 * cosW0) / a0,
      b2: (1 - alpha * A) / a0,
      a1: (-2 * cosW0) / a0,
      a2: (1 - alpha / A) / a0
    };
  }

  function designHighShelf(freq, gainDb, Q, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const w0 = (2 * Math.PI * freq) / sampleRate;
    const cosW0 = Math.cos(w0);
    const sinW0 = Math.sin(w0);
    const alpha = sinW0 / (2 * Q);
    const twoSqrtAAlpha = 2 * Math.sqrt(A) * alpha;
    const a0 = (A + 1) - (A - 1) * cosW0 + twoSqrtAAlpha;

    return {
      b0: (A * ((A + 1) + (A - 1) * cosW0 + twoSqrtAAlpha)) / a0,
      b1: (-2 * A * ((A - 1) + (A + 1) * cosW0)) / a0,
      b2: (A * ((A + 1) + (A - 1) * cosW0 - twoSqrtAAlpha)) / a0,
      a1: (2 * ((A - 1) - (A + 1) * cosW0)) / a0,
      a2: ((A + 1) - (A - 1) * cosW0 - twoSqrtAAlpha) / a0
    };
  }

  function applyBiquadInPlace(buf, c) {
    const { b0, b1, b2, a1, a2 } = c;
    let z1 = 0.0;
    let z2 = 0.0;
    for (let i = 0; i < buf.length; i++) {
      const x = buf[i];
      const y = b0 * x + z1;
      z1 = b1 * x - a1 * y + z2;
      z2 = b2 * x - a2 * y;
      buf[i] = y;
    }
  }

  // ---------------------------------------------------------------------------
  // DSP Stages (Offline Float32Array)
  // ---------------------------------------------------------------------------

  function applyNoiseGate(buf, thresholdDb, floorDb, sampleRate) {
    const threshLin = Math.pow(10, thresholdDb / 20);
    const floorLin = Math.pow(10, floorDb / 20);
    const attackCoeff = Math.exp(-1 / (0.005 * sampleRate));
    const releaseCoeff = Math.exp(-1 / (0.080 * sampleRate));
    const envCoeff = Math.exp(-1 / (0.012 * sampleRate));

    let env = 0.0;
    let gain = 1.0;
    for (let i = 0; i < buf.length; i++) {
      const absX = Math.abs(buf[i]);
      env = envCoeff * env + (1 - envCoeff) * absX;
      const target = env >= threshLin ? 1.0 : floorLin;
      const coeff = target > gain ? attackCoeff : releaseCoeff;
      gain = coeff * gain + (1 - coeff) * target;
      buf[i] *= gain;
    }
  }

  function applySoftKneeCompressor(buf, comp, sampleRate) {
    const { thresholdDb, ratio, kneeDb, attackMs, releaseMs, makeupDb } = comp;
    const attackCoeff = Math.exp(-1 / ((attackMs / 1000) * sampleRate));
    const releaseCoeff = Math.exp(-1 / ((releaseMs / 1000) * sampleRate));
    const makeupLin = Math.pow(10, (makeupDb || 0) / 20);
    const halfKnee = kneeDb / 2;
    const slope = 1 - 1 / ratio;

    let envDb = -96.0;
    for (let i = 0; i < buf.length; i++) {
      const x = buf[i];
      const levelDb = 20 * Math.log10(Math.abs(x) + 1e-9);
      const coeff = levelDb > envDb ? attackCoeff : releaseCoeff;
      envDb = coeff * envDb + (1 - coeff) * levelDb;

      let gainReductionDb = 0.0;
      const overDb = envDb - thresholdDb;
      if (overDb <= -halfKnee) {
        gainReductionDb = 0.0;
      } else if (overDb >= halfKnee) {
        gainReductionDb = slope * overDb;
      } else {
        const diff = overDb + halfKnee;
        gainReductionDb = (slope * diff * diff) / (2 * kneeDb);
      }

      const gainLin = Math.pow(10, -gainReductionDb / 20) * makeupLin;
      buf[i] = x * gainLin;
    }
  }

  /**
   * K-weighted integrated loudness approximation (ITU-R BS.1770-4):
   * Stage 1: +4 dB high-shelf @ 1681 Hz, Stage 2: 38 Hz HPF, gated at -70 LUFS.
   */
  function measureKWeightedLUFS(buf, sampleRate) {
    if (!buf || buf.length === 0) return -70.0;
    const weighted = new Float32Array(buf);
    applyBiquadInPlace(weighted, designHighShelf(1681.97, 4.0, 0.707, sampleRate));
    applyBiquadInPlace(weighted, designHighpass(38.13, 0.5003, sampleRate));

    const blockSize = Math.max(1, Math.round(0.4 * sampleRate));
    const stepSize = Math.max(1, Math.round(0.1 * sampleRate));
    const blockPowers = [];

    for (let start = 0; start < weighted.length; start += stepSize) {
      const end = Math.min(start + blockSize, weighted.length);
      let sumSq = 0.0;
      for (let i = start; i < end; i++) {
        sumSq += weighted[i] * weighted[i];
      }
      const ms = sumSq / (end - start);
      const blockLufs = -0.691 + 10 * Math.log10(ms + 1e-12);
      if (blockLufs > -70.0) {
        blockPowers.push(ms);
      }
    }

    if (blockPowers.length === 0) {
      let sumSq = 0.0;
      for (let i = 0; i < weighted.length; i++) sumSq += weighted[i] * weighted[i];
      const ms = sumSq / weighted.length;
      return -0.691 + 10 * Math.log10(ms + 1e-12);
    }

    const avgPower = blockPowers.reduce((a, b) => a + b, 0) / blockPowers.length;
    const relGate = -0.691 + 10 * Math.log10(avgPower + 1e-12) - 10.0;
    const gated = blockPowers.filter((p) => (-0.691 + 10 * Math.log10(p + 1e-12)) > relGate);
    const finalPower = gated.length > 0
      ? gated.reduce((a, b) => a + b, 0) / gated.length
      : avgPower;

    return -0.691 + 10 * Math.log10(finalPower + 1e-12);
  }

  /**
   * True-Peak Limiter (-1.0 dBFS) + smooth tanh soft de-clipper ceiling.
   */
  function applyTruePeakLimiterAndDeClipper(buf, ceilingDb, sampleRate) {
    const ceiling = Math.pow(10, ceilingDb / 20);
    const attackCoeff = Math.exp(-1 / (0.0005 * sampleRate)); // 0.5 ms fast attack
    const releaseCoeff = Math.exp(-1 / (0.060 * sampleRate)); // 60 ms release

    let gain = 1.0;
    for (let i = 0; i < buf.length; i++) {
      const x = buf[i];
      const absX = Math.abs(x);
      const desiredGain = absX > ceiling ? ceiling / (absX + 1e-9) : 1.0;
      const coeff = desiredGain < gain ? attackCoeff : releaseCoeff;
      gain = coeff * gain + (1 - coeff) * desiredGain;

      // Soft tanh de-clipper guarantees smooth saturation strictly below ceiling
      const limited = x * gain;
      buf[i] = ceiling * Math.tanh(limited / ceiling);
    }
  }

  /**
   * Shaped -65 dBFS comfort noise floor (Paul Kellet 7-state pink dither).
   */
  function addComfortNoiseInPlace(buf, targetRms = COMFORT_NOISE_RMS) {
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    const pinkScale = targetRms * 0.19; // calibrated to ~targetRms RMS after Kellet filter
    for (let i = 0; i < buf.length; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      const pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
      b6 = white * 0.115926;
      buf[i] += pink * pinkScale;
    }
  }

  /**
   * Smart Edge Silence Trim:
   * Trims leading/trailing frames below -48 dBFS while keeping 40 ms smooth fade padding.
   */
  function smartEdgeSilenceTrim(buf, sampleRate, thresholdDb = EDGE_TRIM_THRESHOLD_DB, padMs = EDGE_TRIM_PAD_MS) {
    const n = buf.length;
    if (n === 0) return { pcm: buf, trimmedStartMs: 0, trimmedEndMs: 0 };

    const threshLin = Math.pow(10, thresholdDb / 20);
    const frameLen = Math.max(1, Math.round(0.010 * sampleRate)); // 10 ms frame
    const padSamples = Math.round((padMs / 1000) * sampleRate);

    let firstActive = -1;
    let lastActive = -1;

    for (let start = 0; start < n; start += frameLen) {
      const end = Math.min(start + frameLen, n);
      let sumSq = 0.0;
      for (let i = start; i < end; i++) sumSq += buf[i] * buf[i];
      const rms = Math.sqrt(sumSq / (end - start));
      if (rms >= threshLin) {
        if (firstActive === -1) firstActive = start;
        lastActive = end;
      }
    }

    // If entire clip is silent, do not return an empty buffer
    if (firstActive === -1) {
      return { pcm: buf, trimmedStartMs: 0, trimmedEndMs: 0 };
    }

    const trimStart = Math.max(0, firstActive - padSamples);
    const trimEnd = Math.min(n, lastActive + padSamples);
    const out = buf.slice(trimStart, trimEnd);

    // Smooth raised-cosine fade-in / fade-out over the 40 ms padding
    const fadeLen = Math.min(padSamples, Math.floor(out.length / 4));
    for (let i = 0; i < fadeLen; i++) {
      const w = 0.5 * (1 - Math.cos((Math.PI * i) / fadeLen));
      if (trimStart > 0) out[i] *= w;
      if (trimEnd < n) out[out.length - 1 - i] *= w;
    }

    return {
      pcm: out,
      trimmedStartMs: Math.round((trimStart / sampleRate) * 1000),
      trimmedEndMs: Math.round(((n - trimEnd) / sampleRate) * 1000)
    };
  }

  function computePeakDb(buf) {
    let peak = 0.0;
    for (let i = 0; i < buf.length; i++) {
      const a = Math.abs(buf[i]);
      if (a > peak) peak = a;
    }
    return 20 * Math.log10(peak + 1e-9);
  }

  // ---------------------------------------------------------------------------
  // Public Offline API
  // ---------------------------------------------------------------------------

  function processOffline(inputPCM, options = {}) {
    const sampleRate = options.sampleRate || DEFAULT_SR;
    const presetKey = (options.preset && PRESETS[options.preset]) ? options.preset : 'podcast';
    const preset = PRESETS[presetKey];

    const comfortNoiseEnabled = typeof options.comfortNoise === 'boolean'
      ? options.comfortNoise
      : preset.comfortNoise;
    const edgeTrimEnabled = typeof options.edgeTrim === 'boolean'
      ? options.edgeTrim
      : preset.edgeTrim;
    const dryWet = typeof options.dryWet === 'number'
      ? Math.max(0, Math.min(1, options.dryWet))
      : preset.dryWet;

    let buf = new Float32Array(inputPCM);
    const inputLufs = measureKWeightedLUFS(buf, sampleRate);

    // Optional dry/wet blend with original noisy PCM (e.g., Natural Voice 82% wet)
    if (options.dryPCM && dryWet < 0.999) {
      const dry = options.dryPCM;
      const len = Math.min(buf.length, dry.length);
      const dryMix = 1.0 - dryWet;
      for (let i = 0; i < len; i++) {
        buf[i] = buf[i] * dryWet + dry[i] * dryMix;
      }
    }

    if (preset.bypassAll) {
      // Raw Neural: bypass EQ, compression, loudness normalization, and trim; keep safe ceiling
      applyTruePeakLimiterAndDeClipper(buf, preset.limiterCeilingDb, sampleRate);
      return {
        pcm: buf,
        stats: {
          preset: preset.id,
          sampleRate,
          inputLufs: Number(inputLufs.toFixed(2)),
          outputLufs: Number(measureKWeightedLUFS(buf, sampleRate).toFixed(2)),
          peakDb: Number(computePeakDb(buf).toFixed(2)),
          trimmedStartMs: 0,
          trimmedEndMs: 0,
          comfortNoise: false,
          edgeTrim: false
        }
      };
    }

    // 1. 24 dB/oct Butterworth High-Pass Filter (cascaded 2nd-order sections)
    if (preset.hpfHz > 0) {
      applyBiquadInPlace(buf, designHighpass(preset.hpfHz, BUTTERWORTH_4TH_Q1, sampleRate));
      applyBiquadInPlace(buf, designHighpass(preset.hpfHz, BUTTERWORTH_4TH_Q2, sampleRate));
    }

    // 2. Optional Noise Gate (Heavy Fan / Traffic preset)
    if (preset.gateEnabled) {
      applyNoiseGate(buf, preset.gateThresholdDb, preset.gateFloorDb, sampleRate);
    }

    // 3. Studio Voice Parametric EQ
    if (preset.eq) {
      const { eq } = preset;
      applyBiquadInPlace(buf, designPeaking(eq.boxinessHz, eq.boxinessGainDb, eq.boxinessQ, sampleRate));
      applyBiquadInPlace(buf, designPeaking(eq.presenceHz, eq.presenceGainDb, eq.presenceQ, sampleRate));
      applyBiquadInPlace(buf, designHighShelf(eq.airHz, eq.airGainDb, eq.airQ, sampleRate));
    }

    // 4. Soft-Knee Vocal Compressor
    if (preset.comp) {
      applySoftKneeCompressor(buf, preset.comp, sampleRate);
    }

    // 5. Auto Loudness Normalization (-16 LUFS)
    if (typeof preset.targetLufs === 'number') {
      const currentLufs = measureKWeightedLUFS(buf, sampleRate);
      if (currentLufs > -60.0) {
        // Clamp gain delta to [-18, +18] dB to prevent boosting pure room silence
        const gainDb = Math.max(-18.0, Math.min(18.0, preset.targetLufs - currentLufs));
        const gainLin = Math.pow(10, gainDb / 20);
        for (let i = 0; i < buf.length; i++) buf[i] *= gainLin;
      }
    }

    // 6. True-Peak Limiter (-1.0 dBFS) & Smooth tanh Soft De-Clipper
    applyTruePeakLimiterAndDeClipper(buf, preset.limiterCeilingDb, sampleRate);

    // 7. Toggleable -65 dBFS Comfort Noise Floor
    if (comfortNoiseEnabled) {
      addComfortNoiseInPlace(buf, COMFORT_NOISE_RMS);
    }

    // 8. Smart Edge Silence Trim (-48 dBFS, 40 ms fade padding)
    let trimmedStartMs = 0;
    let trimmedEndMs = 0;
    if (edgeTrimEnabled) {
      const trimmed = smartEdgeSilenceTrim(buf, sampleRate, EDGE_TRIM_THRESHOLD_DB, EDGE_TRIM_PAD_MS);
      buf = trimmed.pcm;
      trimmedStartMs = trimmed.trimmedStartMs;
      trimmedEndMs = trimmed.trimmedEndMs;
    }

    const outputLufs = measureKWeightedLUFS(buf, sampleRate);
    const peakDb = computePeakDb(buf);

    return {
      pcm: buf,
      stats: {
        preset: preset.id,
        sampleRate,
        inputLufs: Number(inputLufs.toFixed(2)),
        outputLufs: Number(outputLufs.toFixed(2)),
        peakDb: Number(peakDb.toFixed(2)),
        trimmedStartMs,
        trimmedEndMs,
        comfortNoise: comfortNoiseEnabled,
        edgeTrim: edgeTrimEnabled
      }
    };
  }

  // ---------------------------------------------------------------------------
  // Zero-Latency Web Audio API Live Mastering Graph
  // ---------------------------------------------------------------------------

  function makeSoftClipCurve(ceilingLin = TRUE_PEAK_CEILING_LIN, samples = 4096) {
    const curve = new Float32Array(samples);
    for (let i = 0; i < samples; i++) {
      const x = (i * 2) / (samples - 1) - 1;
      curve[i] = ceilingLin * Math.tanh(x / ceilingLin);
    }
    return curve;
  }

  function createComfortNoiseBuffer(audioCtx, durationSec = 4.0) {
    const length = Math.round(audioCtx.sampleRate * durationSec);
    const buffer = audioCtx.createBuffer(1, length, audioCtx.sampleRate);
    const data = buffer.getChannelData(0);
    addComfortNoiseInPlace(data, COMFORT_NOISE_RMS);
    return buffer;
  }

  function createLiveChain(audioCtx, presetKey = 'podcast', overrides = {}) {
    const input = audioCtx.createGain();
    const output = audioCtx.createGain();
    const dryGain = audioCtx.createGain();
    const wetGain = audioCtx.createGain();

    // Cascaded 24 dB/oct Butterworth HPF
    const hpf1 = audioCtx.createBiquadFilter();
    hpf1.type = 'highpass';
    hpf1.Q.value = BUTTERWORTH_4TH_Q1;

    const hpf2 = audioCtx.createBiquadFilter();
    hpf2.type = 'highpass';
    hpf2.Q.value = BUTTERWORTH_4TH_Q2;

    // 3-Band Studio Voice Parametric EQ
    const eqBoxiness = audioCtx.createBiquadFilter();
    eqBoxiness.type = 'peaking';

    const eqPresence = audioCtx.createBiquadFilter();
    eqPresence.type = 'peaking';

    const eqAir = audioCtx.createBiquadFilter();
    eqAir.type = 'highshelf';

    // Soft-Knee Vocal Compressor
    const compressor = audioCtx.createDynamicsCompressor();

    // Soft De-Clipper / -1.0 dBFS True-Peak Ceiling
    const shaper = audioCtx.createWaveShaper();
    shaper.curve = makeSoftClipCurve(TRUE_PEAK_CEILING_LIN);
    shaper.oversample = '2x';

    // Comfort Noise Floor (-65 dBFS pink dither)
    const noiseGain = audioCtx.createGain();
    const noiseSource = audioCtx.createBufferSource();
    noiseSource.buffer = createComfortNoiseBuffer(audioCtx, 4.0);
    noiseSource.loop = true;
    noiseSource.connect(noiseGain);
    noiseGain.connect(output);
    noiseSource.start();

    // Wire Wet Path
    input.connect(hpf1);
    hpf1.connect(hpf2);
    hpf2.connect(eqBoxiness);
    eqBoxiness.connect(eqPresence);
    eqPresence.connect(eqAir);
    eqAir.connect(compressor);
    compressor.connect(shaper);
    shaper.connect(wetGain);
    wetGain.connect(output);

    // Wire Dry Path (for Natural Voice 82% wet or live dry/wet)
    input.connect(dryGain);
    dryGain.connect(output);

    function applyPreset(nextPresetKey, nextOverrides = {}) {
      const p = PRESETS[nextPresetKey] || PRESETS.podcast;
      const now = audioCtx.currentTime;
      const ramp = 0.02;

      const dw = typeof nextOverrides.dryWet === 'number' ? nextOverrides.dryWet : p.dryWet;
      const comfort = typeof nextOverrides.comfortNoise === 'boolean' ? nextOverrides.comfortNoise : p.comfortNoise;

      if (p.bypassAll) {
        wetGain.gain.setTargetAtTime(0.0, now, ramp);
        dryGain.gain.setTargetAtTime(1.0, now, ramp);
        noiseGain.gain.setTargetAtTime(0.0, now, ramp);
        return;
      }

      wetGain.gain.setTargetAtTime(dw, now, ramp);
      dryGain.gain.setTargetAtTime(1.0 - dw, now, ramp);

      hpf1.frequency.setTargetAtTime(p.hpfHz || 80, now, ramp);
      hpf2.frequency.setTargetAtTime(p.hpfHz || 80, now, ramp);

      if (p.eq) {
        eqBoxiness.frequency.setTargetAtTime(p.eq.boxinessHz, now, ramp);
        eqBoxiness.Q.setTargetAtTime(p.eq.boxinessQ, now, ramp);
        eqBoxiness.gain.setTargetAtTime(p.eq.boxinessGainDb, now, ramp);

        eqPresence.frequency.setTargetAtTime(p.eq.presenceHz, now, ramp);
        eqPresence.Q.setTargetAtTime(p.eq.presenceQ, now, ramp);
        eqPresence.gain.setTargetAtTime(p.eq.presenceGainDb, now, ramp);

        eqAir.frequency.setTargetAtTime(p.eq.airHz, now, ramp);
        eqAir.gain.setTargetAtTime(p.eq.airGainDb, now, ramp);
      }

      if (p.comp) {
        compressor.threshold.setTargetAtTime(p.comp.thresholdDb, now, ramp);
        compressor.ratio.setTargetAtTime(p.comp.ratio, now, ramp);
        compressor.knee.setTargetAtTime(p.comp.kneeDb, now, ramp);
        compressor.attack.setTargetAtTime(p.comp.attackMs / 1000, now, ramp);
        compressor.release.setTargetAtTime(p.comp.releaseMs / 1000, now, ramp);
      }

      noiseGain.gain.setTargetAtTime(comfort ? 1.0 : 0.0, now, ramp);
    }

    applyPreset(presetKey, overrides);

    return {
      input,
      output,
      applyPreset,
      dispose() {
        try { noiseSource.stop(); } catch (_) {}
        input.disconnect();
        output.disconnect();
        noiseGain.disconnect();
      }
    };
  }

  return {
    DEFAULT_SR,
    TRUE_PEAK_CEILING_DB,
    COMFORT_NOISE_DB,
    EDGE_TRIM_THRESHOLD_DB,
    EDGE_TRIM_PAD_MS,
    PRESETS,
    processOffline,
    createLiveChain,
    measureKWeightedLUFS,
    smartEdgeSilenceTrim,
    computePeakDb
  };
});
