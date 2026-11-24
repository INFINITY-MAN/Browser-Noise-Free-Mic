/**
 * NEON Browser Noise-Free Mic — AudioWorkletProcessor (Mode 2: Live Streaming)
 *
 * Captures 256-sample (16 ms @ 16 kHz) mono frames and posts transferable
 * Float32Arrays to the main thread/worker for 1-frame GTCRN streaming inference.
 * Plays back enhanced frames through a 3-hop (768-sample / 48 ms) circular
 * jitter buffer with underrun tracking, dry/wet blending, and instant bypass.
 */

class NeonStreamProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this.hopSize = opts.hopSize || 256; // 16 ms @ 16 kHz
    this.jitterTarget = opts.jitterTarget || 768; // 3 hops = 48 ms prefill

    // Capture accumulator (two 128-sample render quanta -> 256 samples)
    this.captureBuf = new Float32Array(this.hopSize);
    this.captureOffset = 0;

    // Circular playback jitter buffer (power-of-two capacity >= 8 hops)
    this.ringCapacity = 4096;
    this.ringMask = this.ringCapacity - 1;
    this.ringBuf = new Float32Array(this.ringCapacity);
    this.ringRead = 0;
    this.ringWrite = 0;
    this.ringCount = 0;
    this.isPrefilling = true;

    // Delay-matched dry ring for wet/dry blending during streaming
    this.dryRingBuf = new Float32Array(this.ringCapacity);
    this.dryRead = 0;
    this.dryWrite = 0;
    this.dryCount = 0;

    // Controls & telemetry
    this.bypass = false;
    this.wetMix = 1.0;
    this.underruns = 0;
    this.framesProcessed = 0;
    this.telemetryCounter = 0;

    this.port.onmessage = (event) => {
      const msg = event.data;
      if (!msg) return;

      if (msg.type === "enhanced" && msg.frame) {
        const samples = msg.frame instanceof Float32Array ? msg.frame : new Float32Array(msg.frame);
        this._pushRing(samples);
      } else if (msg.type === "config") {
        if (typeof msg.bypass === "boolean") {
          this.bypass = msg.bypass;
        }
        if (typeof msg.wetMix === "number") {
          this.wetMix = Math.max(0, Math.min(1, msg.wetMix));
        }
      } else if (msg.type === "reset") {
        this.ringRead = 0;
        this.ringWrite = 0;
        this.ringCount = 0;
        this.dryRead = 0;
        this.dryWrite = 0;
        this.dryCount = 0;
        this.captureOffset = 0;
        this.isPrefilling = true;
        this.underruns = 0;
      }
    };
  }

  _pushRing(samples) {
    const len = samples.length;
    // Prevent runaway latency buildup if worker bursts ahead (> 6 hops = 1536 samples)
    const maxBuffered = this.jitterTarget * 2;
    if (this.ringCount + len > maxBuffered) {
      const drop = (this.ringCount + len) - maxBuffered;
      this.ringRead = (this.ringRead + drop) & this.ringMask;
      this.ringCount -= drop;
    }

    for (let i = 0; i < len; i++) {
      this.ringBuf[this.ringWrite] = samples[i];
      this.ringWrite = (this.ringWrite + 1) & this.ringMask;
    }
    this.ringCount += len;

    if (this.isPrefilling && this.ringCount >= this.jitterTarget) {
      this.isPrefilling = false;
    }
  }

  _pushDry(samples) {
    const len = samples.length;
    const maxDry = this.jitterTarget * 2;
    if (this.dryCount + len > maxDry) {
      const drop = (this.dryCount + len) - maxDry;
      this.dryRead = (this.dryRead + drop) & this.ringMask;
      this.dryCount -= drop;
    }
    for (let i = 0; i < len; i++) {
      this.dryRingBuf[this.dryWrite] = samples[i];
      this.dryWrite = (this.dryWrite + 1) & this.ringMask;
    }
    this.dryCount += len;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const outCh0 = output[0];
    const blockLen = outCh0.length; // typically 128
    const inCh0 = input && input.length > 0 ? input[0] : null;

    if (!inCh0) {
      outCh0.fill(0);
      for (let ch = 1; ch < output.length; ch++) output[ch].fill(0);
      return true;
    }

    // 1. Capture input samples into 256-sample hop frames
    let peak = 0;
    for (let i = 0; i < blockLen; i++) {
      const s = inCh0[i];
      const absS = s < 0 ? -s : s;
      if (absS > peak) peak = absS;

      this.captureBuf[this.captureOffset++] = s;
      if (this.captureOffset === this.hopSize) {
        const frameCopy = new Float32Array(this.captureBuf);
        this.port.postMessage(
          { type: "capture", frame: frameCopy, timestamp: currentTime },
          [frameCopy.buffer]
        );
        this.captureOffset = 0;
        this.framesProcessed++;
      }
    }

    // Maintain dry ring buffer for wet/dry mixing
    this._pushDry(inCh0);

    // 2. Output stage: Instant Bypass vs. Jitter-Buffered Neural Output
    if (this.bypass) {
      for (let i = 0; i < blockLen; i++) {
        outCh0[i] = inCh0[i];
      }
    } else if (this.isPrefilling) {
      // Waiting for 3-hop (768-sample / 48 ms) jitter buffer prefill
      outCh0.fill(0);
    } else if (this.ringCount >= blockLen) {
      const wet = this.wetMix;
      const dry = 1.0 - wet;
      for (let i = 0; i < blockLen; i++) {
        const enhSample = this.ringBuf[this.ringRead];
        this.ringRead = (this.ringRead + 1) & this.ringMask;

        let drySample = inCh0[i];
        if (this.dryCount > 0) {
          drySample = this.dryRingBuf[this.dryRead];
          this.dryRead = (this.dryRead + 1) & this.ringMask;
          this.dryCount--;
        }
        outCh0[i] = wet * enhSample + dry * drySample;
      }
      this.ringCount -= blockLen;
    } else {
      // Buffer underrun: drain remaining samples cleanly, zero-fill tail, re-arm prefill
      this.underruns++;
      let i = 0;
      while (i < blockLen && this.ringCount > 0) {
        outCh0[i++] = this.ringBuf[this.ringRead];
        this.ringRead = (this.ringRead + 1) & this.ringMask;
        this.ringCount--;
      }
      while (i < blockLen) {
        outCh0[i++] = 0;
      }
      this.isPrefilling = true;
    }

    // Mirror mono output to additional output channels if stereo sink
    for (let ch = 1; ch < output.length; ch++) {
      output[ch].set(outCh0);
    }

    // Periodic status update (~every 16 render quanta ≈ 128 ms @ 16 kHz)
    this.telemetryCounter++;
    if (this.telemetryCounter >= 16) {
      this.telemetryCounter = 0;
      this.port.postMessage({
        type: "status",
        underruns: this.underruns,
        bufferedSamples: this.ringCount,
        isPrefilling: this.isPrefilling,
        peak
      });
    }

    return true;
  }
}

registerProcessor("neon-stream-processor", NeonStreamProcessor);
