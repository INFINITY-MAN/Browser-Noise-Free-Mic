/**
 * Browser Noise Free Mic — 48 kHz Streaming AudioWorkletProcessor (streamWorklet.js)
 *
 * Processor Name: `bnfm-stream-processor`
 * - Frame Hop: 480 samples (10 ms @ 48 kHz)
 * - Ring Jitter Buffer: 3 hops (30 ms = 1,440 samples) target prefill
 * - Underrun Counter with smooth fade recovery (no clicks/pops)
 * - Instant Zero-Click Bypass via 64-sample (~1.33 ms) crossfade ramp
 */

class BnfmStreamProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [];
  }

  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this.hopSize = opts.hopSize || 480;             // 10 ms @ 48 kHz
    this.jitterHops = opts.jitterHops || 3;         // 3 hops = 30 ms jitter buffer
    this.jitterTarget = this.hopSize * this.jitterHops;

    // Input accumulator for 480-sample hops
    this.inAccum = new Float32Array(this.hopSize);
    this.inFill = 0;

    // Output ring jitter buffer (capacity = 12 hops = 120 ms)
    this.ringCapacity = this.hopSize * 12;
    this.outRing = new Float32Array(this.ringCapacity);
    this.ringRead = 0;
    this.ringWrite = 0;
    this.ringCount = 0;
    this.primed = false;

    // Dry delay line aligned to jitter buffer for phase-coherent dry/wet & zero-click bypass
    this.dryRing = new Float32Array(this.ringCapacity);
    this.dryWrite = 0;

    // Instant zero-click bypass state (64-sample linear ramp @ 48 kHz = 1.33 ms)
    this.bypass = Boolean(opts.bypass || false);
    this.bypassMix = this.bypass ? 1.0 : 0.0; // 1.0 = full dry bypass, 0.0 = wet active
    this.bypassStep = 1.0 / 64.0;
    this.dryWet = typeof opts.dryWet === 'number' ? opts.dryWet : 1.0;

    // Telemetry counters
    this.underruns = 0;
    this.seq = 0;
    this.statsCounter = 0;
    this.inSumSq = 0;
    this.outSumSq = 0;
    this.meterSamples = 0;

    this.port.onmessage = (event) => this._handleMessage(event.data);
  }

  _handleMessage(msg) {
    if (!msg || typeof msg !== 'object') return;

    if (msg.type === 'enhanced' && msg.pcm) {
      const pcm = msg.pcm instanceof Float32Array ? msg.pcm : new Float32Array(msg.pcm);
      this._pushEnhanced(pcm);
      return;
    }

    if (msg.type === 'config') {
      if (typeof msg.bypass === 'boolean') {
        this.bypass = msg.bypass;
      }
      if (typeof msg.dryWet === 'number') {
        this.dryWet = Math.max(0.0, Math.min(1.0, msg.dryWet));
      }
      if (typeof msg.jitterHops === 'number' && msg.jitterHops >= 1 && msg.jitterHops <= 8) {
        this.jitterHops = Math.round(msg.jitterHops);
        this.jitterTarget = this.hopSize * this.jitterHops;
      }
      return;
    }

    if (msg.type === 'reset') {
      this.inFill = 0;
      this.ringRead = 0;
      this.ringWrite = 0;
      this.ringCount = 0;
      this.primed = false;
      this.underruns = 0;
      this.outRing.fill(0);
      this.dryRing.fill(0);
    }
  }

  _pushEnhanced(pcm) {
    const cap = this.ringCapacity;
    for (let i = 0; i < pcm.length; i++) {
      if (this.ringCount >= cap) {
        // Drop oldest sample to prevent unbounded latency drift
        this.ringRead = (this.ringRead + 1) % cap;
        this.ringCount--;
      }
      this.outRing[this.ringWrite] = pcm[i];
      this.ringWrite = (this.ringWrite + 1) % cap;
      this.ringCount++;
    }

    if (!this.primed && this.ringCount >= this.jitterTarget) {
      this.primed = true;
    }
  }

  process(inputs, outputs) {
    const inputList = inputs[0];
    const outputList = outputs[0];
    if (!outputList || outputList.length === 0) return true;

    const outCh = outputList[0];
    const inCh = (inputList && inputList.length > 0) ? inputList[0] : null;
    const blockLen = outCh.length; // 128 samples per Web Audio render quantum
    const cap = this.ringCapacity;

    // 1. Collect incoming 48 kHz samples into 480-sample (10 ms) hops
    for (let i = 0; i < blockLen; i++) {
      const sample = inCh ? inCh[i] : 0.0;
      this.inSumSq += sample * sample;

      this.dryRing[this.dryWrite] = sample;
      this.dryWrite = (this.dryWrite + 1) % cap;

      this.inAccum[this.inFill++] = sample;
      if (this.inFill >= this.hopSize) {
        const frame = new Float32Array(this.inAccum);
        this.port.postMessage(
          {
            type: 'frame',
            pcm: frame,
            seq: this.seq++,
            sampleRate: sampleRate
          },
          [frame.buffer]
        );
        this.inFill = 0;
      }
    }

    // 2. Render output with 3-hop (30 ms) ring jitter buffer + zero-click bypass ramp
    const bypassTarget = this.bypass ? 1.0 : 0.0;
    let blockUnderrun = false;

    for (let i = 0; i < blockLen; i++) {
      // Advance smooth 64-sample bypass crossfade
      if (this.bypassMix < bypassTarget) {
        this.bypassMix = Math.min(bypassTarget, this.bypassMix + this.bypassStep);
      } else if (this.bypassMix > bypassTarget) {
        this.bypassMix = Math.max(bypassTarget, this.bypassMix - this.bypassStep);
      }

      const liveDry = inCh ? inCh[i] : 0.0;
      const delayIdx = (this.dryWrite - blockLen + i - this.jitterTarget + cap * 2) % cap;
      const alignedDry = this.dryRing[delayIdx];

      let wetSample = 0.0;
      if (this.primed && this.ringCount > 0) {
        wetSample = this.outRing[this.ringRead];
        this.ringRead = (this.ringRead + 1) % cap;
        this.ringCount--;
      } else if (this.primed && !this.bypass) {
        blockUnderrun = true;
        wetSample = 0.0;
      }

      // Blend wet enhanced signal with aligned dry (for dryWet < 1.0)
      const processed = wetSample * this.dryWet + alignedDry * (1.0 - this.dryWet);

      // Instant zero-click crossfade between processed and live bypass
      const finalSample = processed * (1.0 - this.bypassMix) + liveDry * this.bypassMix;
      outCh[i] = finalSample;
      this.outSumSq += finalSample * finalSample;
    }

    if (blockUnderrun) {
      this.underruns++;
      this.primed = false; // Re-prime jitter buffer to 3 hops cleanly
    }

    // Copy mono output to any additional output channels (stereo monitoring)
    for (let ch = 1; ch < outputList.length; ch++) {
      outputList[ch].set(outCh);
    }

    // 3. Emit periodic telemetry every ~100 ms (~38 render quanta @ 48 kHz)
    this.meterSamples += blockLen;
    this.statsCounter++;
    if (this.statsCounter >= 38) {
      const inRms = Math.sqrt(this.inSumSq / Math.max(1, this.meterSamples));
      const outRms = Math.sqrt(this.outSumSq / Math.max(1, this.meterSamples));
      this.port.postMessage({
        type: 'stats',
        underruns: this.underruns,
        bufferedSamples: this.ringCount,
        jitterTargetSamples: this.jitterTarget,
        primed: this.primed,
        bypass: this.bypass,
        dryWet: this.dryWet,
        inputRmsDb: Number((20 * Math.log10(inRms + 1e-9)).toFixed(2)),
        outputRmsDb: Number((20 * Math.log10(outRms + 1e-9)).toFixed(2))
      });
      this.statsCounter = 0;
      this.inSumSq = 0;
      this.outSumSq = 0;
      this.meterSamples = 0;
    }

    return true;
  }
}

registerProcessor('bnfm-stream-processor', BnfmStreamProcessor);
