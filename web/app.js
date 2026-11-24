/**
 * NEON Browser Noise-Free Mic — Main Studio Controller (Dual-Mode Frontend)
 *
 * Mode 1: Recording & File Enhancement Studio
 *   - Live Mic capture (echoCancellation: false, noiseSuppression: false, autoGainControl: false) + VU meter
 *   - Drag-and-drop file upload (.wav, .mp3, .m4a, .ogg) + OfflineAudioContext 16 kHz resampling
 *   - Synthetic Indian Household Demo (3s formants + 50/100/150 Hz transformer hum + ceiling fan pink noise)
 *   - Zero-latency timestamp-synced A (Noisy) / B (Wiener DSP) / C (GTCRN Neural) Web Audio switcher
 *   - Play/Pause, timeline scrubber, Dry/Wet mix slider (0%-100%)
 *   - Dual Inferno colormap spectrogram canvases (canvas-noisy & canvas-denoised)
 *   - 16-bit 16 kHz mono RIFF WAV exporter
 *
 * Mode 2: Continuous Live Mic -> Virtual Audio Cable
 *   - AudioWorkletNode (streamWorklet.js) + enumerateDevices (audioinput / audiooutput)
 *   - AudioContext.setSinkId with MediaStreamAudioDestinationNode -> <audio>.setSinkId fallback
 *   - Instant Bypass toggle, Dry/Wet mix, rolling 50-frame latency (ms), buffer underrun counter
 *
 * Telemetry + Adaptive Learning (Loop A IndexedDB + Loop B /api/telemetry, /api/feedback, /api/metrics-dashboard)
 */

(function () {
  "use strict";

  const SAMPLE_RATE = 16000;

  // ==================== INFERNO COLORMAP LUT (256 RGB STOPS) ====================
  const INFERNO_STOPS = [
    [0.0, 0, 0, 4],
    [0.14, 27, 12, 65],
    [0.29, 74, 12, 107],
    [0.43, 120, 28, 109],
    [0.57, 165, 44, 96],
    [0.71, 207, 68, 70],
    [0.86, 237, 105, 37],
    [0.95, 251, 155, 6],
    [1.0, 252, 255, 164]
  ];
  const INFERNO_LUT = new Uint8ClampedArray(256 * 3);
  (function buildInfernoLut() {
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let s0 = INFERNO_STOPS[0];
      let s1 = INFERNO_STOPS[INFERNO_STOPS.length - 1];
      for (let k = 0; k < INFERNO_STOPS.length - 1; k++) {
        if (t >= INFERNO_STOPS[k][0] && t <= INFERNO_STOPS[k + 1][0]) {
          s0 = INFERNO_STOPS[k];
          s1 = INFERNO_STOPS[k + 1];
          break;
        }
      }
      const f = s1[0] > s0[0] ? (t - s0[0]) / (s1[0] - s0[0]) : 0;
      INFERNO_LUT[i * 3] = Math.round(s0[1] + f * (s1[1] - s0[1]));
      INFERNO_LUT[i * 3 + 1] = Math.round(s0[2] + f * (s1[2] - s0[2]));
      INFERNO_LUT[i * 3 + 2] = Math.round(s0[3] + f * (s1[3] - s0[3]));
    }
  })();

  // ==================== APPLICATION STATE ====================
  const app = {
    worker: null,
    learner: null,
    backendLabel: "CPU: WASM SIMD (INT8)",
    provider: "wasm",
    precision: "INT8",
    modelBytes: 0,
    modelsReady: false,

    // Mode 1: Recording & File Studio
    isRecording: false,
    recStream: null,
    recCtx: null,
    recProcessor: null,
    recChunks: [],
    recAnalyser: null,
    vuRaf: null,

    pcmNoisy: null,
    pcmWiener: null,
    pcmGtcrn: null,
    noisyMag: null,
    wienerMag: null,
    gtcrnMag: null,
    specFrames: 0,
    specBins: 257,

    // Synced A/B/C Web Audio Player
    playCtx: null,
    bufA: null,
    bufB: null,
    bufC: null,
    srcA: null,
    srcB: null,
    srcC: null,
    gainA: null,
    gainB: null,
    gainC: null,
    activeTrack: "C", // 'A' | 'B' | 'C'
    wetMix: 1.0,
    isPlaying: false,
    playStartCtxTime: 0,
    playOffsetSec: 0,
    clipDurationSec: 0,
    scrubRaf: null,

    // Mode 2: Continuous Live Streaming
    streamActive: false,
    streamCtx: null,
    streamMic: null,
    streamSourceNode: null,
    streamWorkletNode: null,
    streamAudioEl: null,
    streamBypass: false,
    rollingLatencies: [],
    underruns: 0,

    // Loop B Feedback
    selectedFeedback: null,
    lastInferenceMs: 0,
    lastRtf: 0,
    lastSnrDb: 0
  };

  // ==================== DOM SHORTCUTS ====================
  const $ = (id) => document.getElementById(id);

  // ==================== INITIALIZATION ====================
  async function boot() {
    app.learner = new window.AdaptiveLearner();
    const summary = await app.learner.init();

    // Restore user's preferred Dry/Wet mix
    app.wetMix = summary.preferredWetMix;
    const wetPct = Math.round(app.wetMix * 100);
    if ($("slider-wet-mode1")) $("slider-wet-mode1").value = String(wetPct);
    if ($("label-wet-mode1")) $("label-wet-mode1").textContent = `${wetPct}%`;
    if ($("slider-wet-mode2")) $("slider-wet-mode2").value = String(wetPct);
    if ($("label-wet-mode2")) $("label-wet-mode2").textContent = `${wetPct}%`;

    updateLoopAUI(summary);
    bindUIEvents();
    initWorker();
    startMemoryPolling();
    refreshAudioDevices();
    refreshDashboard();

    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", refreshAudioDevices);
    }
  }

  function initWorker() {
    app.worker = new Worker("audioWorker.js");
    app.worker.onmessage = (event) => {
      const msg = event.data;
      if (!msg) return;

      if (msg.type === "ready") {
        app.modelsReady = true;
        app.backendLabel = msg.backend;
        app.provider = msg.provider;
        app.precision = msg.precision;
        app.modelBytes = msg.modelBytes || 0;
        updateBackendBadge(msg.backend, msg.provider, msg.modelsLoaded);

        // Push persisted IndexedDB room noise profile into worker
        if (app.learner && app.learner.hasProfile) {
          app.worker.postMessage({
            type: "load_profile",
            profile: Array.from(app.learner.noiseProfile),
            calibrationFrames: app.learner.calibrationFrames
          });
        }
        updateMemoryCard();
      } else if (msg.type === "batch_result") {
        handleBatchResult(msg);
      } else if (msg.type === "stream_result") {
        handleStreamWorkerResult(msg);
      } else if (msg.type === "error") {
        console.error("[NEON Worker Error]", msg.message);
        setStudioStatus(`Engine warning: ${msg.message}`, true);
      }
    };
    app.worker.postMessage({ type: "init" });
  }

  function updateBackendBadge(label, provider, modelsLoaded) {
    const badge = $("backend-badge");
    const status = $("model-status");
    const cardHw = $("card-hardware");

    if (badge) {
      badge.textContent = label;
      badge.className =
        provider === "webgpu"
          ? "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/40"
          : "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/15 text-indigo-300 border border-indigo-500/40";
    }
    if (status) {
      status.textContent = modelsLoaded ? "Ready" : "Ready (DSP Fallback)";
      status.className =
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30";
    }
    if (cardHw) {
      cardHw.textContent = label;
    }
  }

  function updateMemoryCard() {
    const el = $("card-memory");
    if (!el) return;
    const modelKb = Math.round(app.modelBytes / 1024);
    let heapStr = "N/A";
    if (performance && performance.memory && Number.isFinite(performance.memory.usedJSHeapSize)) {
      heapStr = `${(performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(1)} MB`;
    }
    el.textContent = `Model: ${modelKb} KB | Heap: ${heapStr}`;
  }

  function startMemoryPolling() {
    updateMemoryCard();
    setInterval(updateMemoryCard, 2000);
  }

  // ==================== UI EVENT BINDINGS ====================
  function bindUIEvents() {
    // Dual-Mode Tabs
    $("tab-mode-1").addEventListener("click", () => switchTab(1));
    $("tab-mode-2").addEventListener("click", () => switchTab(2));

    // Mode 1 Controls
    $("btn-record").addEventListener("click", toggleRecording);
    $("btn-demo-indian").addEventListener("click", loadIndianHouseholdDemo);

    const dropzone = $("file-dropzone");
    const fileInput = $("file-input");
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("border-cyan-400", "bg-cyan-500/10");
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("border-cyan-400", "bg-cyan-500/10");
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("border-cyan-400", "bg-cyan-500/10");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleUploadedFile(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleUploadedFile(e.target.files[0]);
      }
    });

    // A/B/C Switcher
    $("btn-mode-a").addEventListener("click", () => setActiveTrack("A"));
    $("btn-mode-b").addEventListener("click", () => setActiveTrack("B"));
    $("btn-mode-c").addEventListener("click", () => setActiveTrack("C"));

    // Playback & Scrubber
    $("btn-play-pause").addEventListener("click", togglePlayback);
    $("timeline-scrubber").addEventListener("input", (e) => {
      const pct = Number(e.target.value) / 1000;
      seekPlayback(pct * app.clipDurationSec);
    });

    // Dry/Wet Sliders (synced across Mode 1 & Mode 2 + persisted in IndexedDB)
    $("slider-wet-mode1").addEventListener("input", (e) => onWetMixChanged(Number(e.target.value) / 100));
    $("slider-wet-mode2").addEventListener("input", (e) => onWetMixChanged(Number(e.target.value) / 100));

    // Export WAV
    $("btn-export-wav").addEventListener("click", exportCurrentWav);

    // Mode 2 Controls
    $("btn-refresh-devices").addEventListener("click", refreshAudioDevices);
    $("btn-stream-toggle").addEventListener("click", toggleContinuousStream);
    $("btn-bypass-toggle").addEventListener("click", toggleStreamBypass);
    $("select-output-device").addEventListener("change", (e) => applyOutputSink(e.target.value));

    // Loop A & Loop B Controls
    $("btn-reset-profile").addEventListener("click", async () => {
      const summary = await app.learner.resetRoomProfile();
      if (app.worker) app.worker.postMessage({ type: "reset_profile" });
      updateLoopAUI(summary);
    });

    document.querySelectorAll("[data-feedback]").forEach((btn) => {
      btn.addEventListener("click", () => {
        app.selectedFeedback = btn.getAttribute("data-feedback");
        document.querySelectorAll("[data-feedback]").forEach((b) => {
          b.classList.remove("ring-2", "ring-cyan-400", "bg-cyan-500/20", "border-cyan-400");
        });
        btn.classList.add("ring-2", "ring-cyan-400", "bg-cyan-500/20", "border-cyan-400");
      });
    });

    $("btn-submit-feedback").addEventListener("click", submitLoopBFeedback);
    $("btn-refresh-dashboard").addEventListener("click", refreshDashboard);
  }

  function switchTab(mode) {
    const p1 = $("panel-mode-1");
    const p2 = $("panel-mode-2");
    const t1 = $("tab-mode-1");
    const t2 = $("tab-mode-2");

    if (mode === 1) {
      p1.classList.remove("hidden");
      p2.classList.add("hidden");
      t1.className = "px-4 py-2.5 rounded-lg text-sm font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 transition";
      t2.className = "px-4 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/70 border border-transparent transition";
    } else {
      stopPlayback();
      p1.classList.add("hidden");
      p2.classList.remove("hidden");
      t2.className = "px-4 py-2.5 rounded-lg text-sm font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 transition";
      t1.className = "px-4 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/70 border border-transparent transition";
      refreshAudioDevices();
    }
  }

  function setStudioStatus(text, isError = false) {
    const el = $("studio-status");
    if (!el) return;
    el.textContent = text;
    el.className = isError ? "text-xs text-rose-400" : "text-xs text-slate-400";
  }

  function onWetMixChanged(wet) {
    app.wetMix = Math.max(0, Math.min(1, wet));
    const pct = Math.round(app.wetMix * 100);
    if ($("slider-wet-mode1")) $("slider-wet-mode1").value = String(pct);
    if ($("label-wet-mode1")) $("label-wet-mode1").textContent = `${pct}%`;
    if ($("slider-wet-mode2")) $("slider-wet-mode2").value = String(pct);
    if ($("label-wet-mode2")) $("label-wet-mode2").textContent = `${pct}%`;

    applyPlaybackGains();
    if (app.streamWorkletNode) {
      app.streamWorkletNode.port.postMessage({ type: "config", wetMix: app.wetMix });
    }
    if (app.learner) {
      app.learner.setPreferredWetMix(app.wetMix).catch(() => {});
    }
  }

  // ==================== MODE 1: RECORDING, UPLOAD & SYNTHETIC DEMO ====================

  async function toggleRecording() {
    if (app.isRecording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  }

  async function startRecording() {
    try {
      stopPlayback();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1
        }
      });

      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      const processor = ctx.createScriptProcessor(4096, 1, 1);

      app.recChunks = [];
      processor.onaudioprocess = (e) => {
        if (!app.isRecording) return;
        app.recChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };

      src.connect(analyser);
      analyser.connect(processor);
      processor.connect(ctx.destination);

      app.recStream = stream;
      app.recCtx = ctx;
      app.recAnalyser = analyser;
      app.recProcessor = processor;
      app.isRecording = true;

      const btn = $("btn-record");
      btn.textContent = "Stop Recording";
      btn.classList.remove("bg-rose-600", "hover:bg-rose-500");
      btn.classList.add("bg-amber-500", "hover:bg-amber-400", "text-slate-950");
      setStudioStatus("Recording raw mic audio (DSP disabled)...");
      drawVuMeter();
    } catch (err) {
      setStudioStatus(`Mic access failed: ${err.message}`, true);
    }
  }

  async function stopRecording() {
    app.isRecording = false;
    if (app.vuRaf) cancelAnimationFrame(app.vuRaf);
    if ($("vu-bar")) $("vu-bar").style.width = "0%";

    const btn = $("btn-record");
    btn.textContent = "Record Mic";
    btn.classList.add("bg-rose-600", "hover:bg-rose-500");
    btn.classList.remove("bg-amber-500", "hover:bg-amber-400", "text-slate-950");

    const nativeSr = app.recCtx ? app.recCtx.sampleRate : 48000;
    if (app.recProcessor) app.recProcessor.disconnect();
    if (app.recStream) app.recStream.getTracks().forEach((t) => t.stop());
    if (app.recCtx) await app.recCtx.close();

    const totalLen = app.recChunks.reduce((acc, c) => acc + c.length, 0);
    if (totalLen < 256) {
      setStudioStatus("Recording too short.");
      return;
    }

    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of app.recChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    setStudioStatus("Resampling to 16 kHz & running GTCRN...");
    const pcm16k = await resampleTo16k(merged, nativeSr);
    runBatchClip(pcm16k, "Live Mic Recording");
  }

  function drawVuMeter() {
    if (!app.isRecording || !app.recAnalyser) return;
    const data = new Float32Array(app.recAnalyser.fftSize);
    app.recAnalyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    const rms = Math.sqrt(sum / data.length);
    const pct = Math.min(100, Math.round(rms * 320));
    if ($("vu-bar")) $("vu-bar").style.width = `${pct}%`;
    app.vuRaf = requestAnimationFrame(drawVuMeter);
  }

  async function handleUploadedFile(file) {
    try {
      stopPlayback();
      setStudioStatus(`Decoding ${file.name}...`);
      const arrayBuf = await file.arrayBuffer();
      const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuf = await tmpCtx.decodeAudioData(arrayBuf);
      await tmpCtx.close();

      // Downmix to mono
      const len = audioBuf.length;
      const mono = new Float32Array(len);
      const chCount = audioBuf.numberOfChannels;
      for (let c = 0; c < chCount; c++) {
        const chData = audioBuf.getChannelData(c);
        for (let i = 0; i < len; i++) mono[i] += chData[i] / chCount;
      }

      const pcm16k = await resampleTo16k(mono, audioBuf.sampleRate);
      runBatchClip(pcm16k, file.name);
    } catch (err) {
      setStudioStatus(`Could not decode audio file: ${err.message}`, true);
    }
  }

  async function resampleTo16k(monoPcm, srcRate) {
    if (srcRate === SAMPLE_RATE) return monoPcm;
    const duration = monoPcm.length / srcRate;
    const targetLen = Math.max(512, Math.round(duration * SAMPLE_RATE));
    const offline = new OfflineAudioContext(1, targetLen, SAMPLE_RATE);
    const srcBuf = offline.createBuffer(1, monoPcm.length, srcRate);
    srcBuf.copyToChannel(monoPcm, 0);
    const srcNode = offline.createBufferSource();
    srcNode.buffer = srcBuf;
    srcNode.connect(offline.destination);
    srcNode.start(0);
    const rendered = await offline.startRendering();
    return new Float32Array(rendered.getChannelData(0));
  }

  /**
   * Generates a 3.0-second synthetic Indian household acoustic test clip @ 16 kHz:
   * - Speech-like syllabic formants (F0 ~135-175 Hz + formant resonances F1/F2/F3)
   * - 50 / 100 / 150 Hz Indian mains transformer hum
   * - Ceiling fan pink noise (Paul Kellet filter + 4.5 Hz blade-pass modulation)
   */
  function loadIndianHouseholdDemo() {
    stopPlayback();
    const durationSec = 3.0;
    const N = Math.round(SAMPLE_RATE * durationSec);
    const pcm = new Float32Array(N);

    // Paul Kellet pink noise state
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    let phase0 = 0;

    // 4 speech syllables across 3 seconds (leaving noise-only pauses at 0..0.3s and 2.6..3.0s)
    const syllables = [
      { start: 0.35, end: 0.85, f1: 680, f2: 1250, f3: 2550 },
      { start: 1.00, end: 1.48, f1: 420, f2: 1950, f3: 2700 },
      { start: 1.62, end: 2.08, f1: 560, f2: 1500, f3: 2450 },
      { start: 2.20, end: 2.60, f1: 720, f2: 1180, f3: 2620 }
    ];

    for (let i = 0; i < N; i++) {
      const t = i / SAMPLE_RATE;

      // 1. Speech-like harmonic formants
      let speech = 0;
      for (let s = 0; s < syllables.length; s++) {
        const sy = syllables[s];
        if (t >= sy.start && t <= sy.end) {
          const u = (t - sy.start) / (sy.end - sy.start);
          const env = Math.sin(Math.PI * u) * (0.85 + 0.15 * Math.sin(2 * Math.PI * 5 * u));
          const f0 = 145 + 22 * Math.sin(2 * Math.PI * 1.2 * t);
          phase0 += (2 * Math.PI * f0) / SAMPLE_RATE;
          const glottal =
            0.55 * Math.sin(phase0) +
            0.30 * Math.sin(2 * phase0) +
            0.15 * Math.sin(3 * phase0);
          const formants =
            0.50 * Math.sin(2 * Math.PI * sy.f1 * t) +
            0.30 * Math.sin(2 * Math.PI * sy.f2 * t) +
            0.15 * Math.sin(2 * Math.PI * sy.f3 * t);
          speech = env * (0.28 * glottal + 0.24 * formants * (0.6 + 0.4 * Math.sin(phase0)));
          break;
        }
      }

      // 2. 50 Hz / 100 Hz / 150 Hz transformer hum
      const hum =
        0.095 * Math.sin(2 * Math.PI * 50 * t) +
        0.048 * Math.sin(2 * Math.PI * 100 * t + 0.4) +
        0.026 * Math.sin(2 * Math.PI * 150 * t + 1.1);

      // 3. Ceiling fan pink noise + 4.5 Hz blade-pass modulation
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      const pink = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.028;
      b6 = white * 0.115926;
      const fanMod = 0.82 + 0.18 * Math.sin(2 * Math.PI * 4.5 * t);

      pcm[i] = Math.max(-0.98, Math.min(0.98, speech + hum + pink * fanMod));
    }

    runBatchClip(pcm, "Fan + 50Hz Hum Demo (3.0s)");
  }

  function runBatchClip(pcm16k, label) {
    app.pcmNoisy = pcm16k;
    app.clipDurationSec = pcm16k.length / SAMPLE_RATE;
    setStudioStatus(`Processing "${label}" (${app.clipDurationSec.toFixed(2)}s @ 16 kHz)...`);

    const copy = new Float32Array(pcm16k);
    app.worker.postMessage({ type: "process_batch", requestId: Date.now(), pcm: copy }, [copy.buffer]);
  }

  function handleBatchResult(msg) {
    app.pcmWiener = msg.wienerPcm;
    app.pcmGtcrn = msg.gtcrnPcm;
    app.noisyMag = msg.noisyMag;
    app.wienerMag = msg.wienerMag;
    app.gtcrnMag = msg.gtcrnMag;
    app.specFrames = msg.numFrames;
    app.specBins = msg.numBins;

    app.lastInferenceMs = msg.inferenceMs;
    app.lastRtf = msg.rtf;
    app.lastSnrDb = msg.snrGainDb;

    // Update Telemetry Cards
    if ($("card-latency")) $("card-latency").textContent = `${msg.inferenceMs.toFixed(1)} ms`;
    if ($("card-rtf")) $("card-rtf").textContent = `${msg.rtf.toFixed(3)}x`;
    if ($("card-snr")) $("card-snr").textContent = `+${msg.snrGainDb.toFixed(1)} dB`;
    updateMemoryCard();

    // Prepare synchronized A/B/C audio buffers
    prepareSyncedBuffers();
    setActiveTrack("C");

    // Render Dual Inferno Spectrograms
    renderSpectrogram($("canvas-noisy"), app.noisyMag, app.specFrames, app.specBins);
    renderSpectrogram($("canvas-denoised"), app.gtcrnMag, app.specFrames, app.specBins);

    $("btn-play-pause").disabled = false;
    $("btn-export-wav").disabled = false;
    $("timeline-scrubber").disabled = false;
    $("time-total").textContent = formatTime(app.clipDurationSec);
    $("time-current").textContent = "0:00.0";
    setStudioStatus(`Ready — ${app.clipDurationSec.toFixed(2)}s clip enhanced in ${msg.inferenceMs.toFixed(1)} ms (${msg.rtf.toFixed(3)}x RTF).`);

    // Record Loop A & send Loop B telemetry
    if (app.learner) {
      app.learner.recordSessionMetrics(msg.rtf, msg.snrGainDb).then(() => {
        updateLoopAUI(app.learner.getSummary());
      });
      app.learner.sendTelemetry({
        mode: "batch",
        backend: app.backendLabel,
        inferenceMs: Number(msg.inferenceMs.toFixed(2)),
        rtf: Number(msg.rtf.toFixed(4)),
        snrGainDb: Number(msg.snrGainDb.toFixed(2)),
        clipDurationSec: Number(app.clipDurationSec.toFixed(2))
      });
    }
  }

  // ==================== TIMESTAMP-SYNCED A / B / C WEB AUDIO PLAYBACK ====================

  function prepareSyncedBuffers() {
    stopPlayback();
    if (!app.playCtx) {
      app.playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    }
    const len = app.pcmNoisy.length;
    app.bufA = app.playCtx.createBuffer(1, len, SAMPLE_RATE);
    app.bufB = app.playCtx.createBuffer(1, len, SAMPLE_RATE);
    app.bufC = app.playCtx.createBuffer(1, len, SAMPLE_RATE);
    app.bufA.copyToChannel(app.pcmNoisy, 0);
    app.bufB.copyToChannel(app.pcmWiener, 0);
    app.bufC.copyToChannel(app.pcmGtcrn, 0);
    app.playOffsetSec = 0;
  }

  function setActiveTrack(track) {
    app.activeTrack = track;
    const map = { A: "btn-mode-a", B: "btn-mode-b", C: "btn-mode-c" };
    Object.entries(map).forEach(([k, id]) => {
      const el = $(id);
      if (!el) return;
      if (k === track) {
        el.className = "px-3.5 py-2 rounded-lg text-xs font-semibold bg-cyan-500 text-slate-950 shadow shadow-cyan-500/30 transition";
      } else {
        el.className = "px-3.5 py-2 rounded-lg text-xs font-medium bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 transition";
      }
    });
    applyPlaybackGains();

    // Update enhanced spectrogram canvas to reflect B vs C if data is loaded
    if (app.specFrames > 0) {
      const mag = track === "B" ? app.wienerMag : track === "A" ? app.noisyMag : app.gtcrnMag;
      renderSpectrogram($("canvas-denoised"), mag, app.specFrames, app.specBins);
    }
  }

  function applyPlaybackGains() {
    if (!app.gainA || !app.gainB || !app.gainC || !app.playCtx) return;
    const now = app.playCtx.currentTime;
    const wet = app.wetMix;
    const dry = 1.0 - wet;

    let gA = 0, gB = 0, gC = 0;
    if (app.activeTrack === "A") {
      gA = 1.0;
    } else if (app.activeTrack === "B") {
      gA = dry;
      gB = wet;
    } else {
      gA = dry;
      gC = wet;
    }
    app.gainA.gain.setTargetAtTime(gA, now, 0.005);
    app.gainB.gain.setTargetAtTime(gB, now, 0.005);
    app.gainC.gain.setTargetAtTime(gC, now, 0.005);
  }

  function togglePlayback() {
    if (!app.bufA) return;
    if (app.isPlaying) {
      pausePlayback();
    } else {
      startPlayback(app.playOffsetSec >= app.clipDurationSec ? 0 : app.playOffsetSec);
    }
  }

  function startPlayback(offsetSec) {
    if (!app.playCtx || !app.bufA) return;
    if (app.playCtx.state === "suspended") app.playCtx.resume();
    teardownSources();

    const ctx = app.playCtx;
    app.srcA = ctx.createBufferSource();
    app.srcB = ctx.createBufferSource();
    app.srcC = ctx.createBufferSource();
    app.srcA.buffer = app.bufA;
    app.srcB.buffer = app.bufB;
    app.srcC.buffer = app.bufC;

    app.gainA = ctx.createGain();
    app.gainB = ctx.createGain();
    app.gainC = ctx.createGain();

    app.srcA.connect(app.gainA).connect(ctx.destination);
    app.srcB.connect(app.gainB).connect(ctx.destination);
    app.srcC.connect(app.gainC).connect(ctx.destination);

    applyPlaybackGains();

    const safeOffset = Math.max(0, Math.min(app.clipDurationSec - 0.01, offsetSec));
    const startAt = ctx.currentTime + 0.01;
    app.srcA.start(startAt, safeOffset);
    app.srcB.start(startAt, safeOffset);
    app.srcC.start(startAt, safeOffset);

    app.playStartCtxTime = startAt - safeOffset;
    app.playOffsetSec = safeOffset;
    app.isPlaying = true;
    $("btn-play-pause").textContent = "Pause";

    app.srcA.onended = () => {
      if (app.isPlaying && app.playCtx.currentTime - app.playStartCtxTime >= app.clipDurationSec - 0.03) {
        stopPlayback();
      }
    };
    tickScrubber();
  }

  function pausePlayback() {
    if (!app.isPlaying) return;
    app.playOffsetSec = Math.min(app.clipDurationSec, app.playCtx.currentTime - app.playStartCtxTime);
    teardownSources();
    app.isPlaying = false;
    $("btn-play-pause").textContent = "Play";
    if (app.scrubRaf) cancelAnimationFrame(app.scrubRaf);
  }

  function stopPlayback() {
    teardownSources();
    app.isPlaying = false;
    app.playOffsetSec = 0;
    if ($("btn-play-pause")) $("btn-play-pause").textContent = "Play";
    if ($("timeline-scrubber")) $("timeline-scrubber").value = "0";
    if ($("time-current")) $("time-current").textContent = "0:00.0";
    if (app.scrubRaf) cancelAnimationFrame(app.scrubRaf);
  }

  function seekPlayback(targetSec) {
    app.playOffsetSec = Math.max(0, Math.min(app.clipDurationSec, targetSec));
    $("time-current").textContent = formatTime(app.playOffsetSec);
    if (app.isPlaying) {
      startPlayback(app.playOffsetSec);
    }
  }

  function teardownSources() {
    [app.srcA, app.srcB, app.srcC].forEach((s) => {
      if (s) {
        s.onended = null;
        try { s.stop(); } catch (_) {}
        try { s.disconnect(); } catch (_) {}
      }
    });
    app.srcA = app.srcB = app.srcC = null;
  }

  function tickScrubber() {
    if (!app.isPlaying || !app.playCtx) return;
    const elapsed = Math.min(app.clipDurationSec, Math.max(0, app.playCtx.currentTime - app.playStartCtxTime));
    const pct = app.clipDurationSec > 0 ? Math.round((elapsed / app.clipDurationSec) * 1000) : 0;
    $("timeline-scrubber").value = String(pct);
    $("time-current").textContent = formatTime(elapsed);
    app.scrubRaf = requestAnimationFrame(tickScrubber);
  }

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1).padStart(4, "0");
    return `${m}:${s}`;
  }

  // ==================== INFERNO SPECTROGRAM RENDERER ====================

  function renderSpectrogram(canvas, mag, numFrames, numBins) {
    if (!canvas || !mag || numFrames <= 0) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const img = ctx.createImageData(w, h);
    const data = img.data;

    // Normalize dB range [-75 dB, +5 dB]
    const minDb = -75;
    const maxDb = 5;
    const invRange = 1.0 / (maxDb - minDb);

    for (let y = 0; y < h; y++) {
      // Top of canvas = Nyquist (8 kHz), bottom = DC (0 Hz)
      const bin = Math.min(numBins - 1, Math.floor(((h - 1 - y) / h) * numBins));
      const rowOffset = bin * numFrames;
      for (let x = 0; x < w; x++) {
        const frame = Math.min(numFrames - 1, Math.floor((x / w) * numFrames));
        const val = mag[rowOffset + frame] + 1e-9;
        const db = 20 * Math.log10(val);
        const norm = Math.max(0, Math.min(1, (db - minDb) * invRange));
        const lutIdx = Math.round(norm * 255) * 3;
        const px = (y * w + x) * 4;
        data[px] = INFERNO_LUT[lutIdx];
        data[px + 1] = INFERNO_LUT[lutIdx + 1];
        data[px + 2] = INFERNO_LUT[lutIdx + 2];
        data[px + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // ==================== 16-BIT 16 KHZ MONO RIFF WAV ENCODER ====================

  function encodeWav16k(pcm) {
    const numSamples = pcm.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeStr(0, "RIFF");
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true); // PCM chunk size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample
    writeStr(36, "data");
    view.setUint32(40, numSamples * 2, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
    return buffer;
  }

  function getCurrentMixedPcm() {
    if (!app.pcmNoisy) return null;
    if (app.activeTrack === "A") return app.pcmNoisy;
    const target = app.activeTrack === "B" ? app.pcmWiener : app.pcmGtcrn;
    const wet = app.wetMix;
    const dry = 1.0 - wet;
    const mixed = new Float32Array(app.pcmNoisy.length);
    for (let i = 0; i < mixed.length; i++) {
      mixed[i] = wet * target[i] + dry * app.pcmNoisy[i];
    }
    return mixed;
  }

  function exportCurrentWav() {
    const pcm = getCurrentMixedPcm();
    if (!pcm) return;
    const wavBuf = encodeWav16k(pcm);
    const blob = new Blob([wavBuf], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `neon_${app.activeTrack.toLowerCase()}_16khz.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ==================== MODE 2: CONTINUOUS LIVE MIC -> VIRTUAL AUDIO CABLE ====================

  async function refreshAudioDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inSel = $("select-input-device");
      const outSel = $("select-output-device");
      if (!inSel || !outSel) return;

      const prevIn = inSel.value;
      const prevOut = outSel.value;
      inSel.innerHTML = "";
      outSel.innerHTML = "";

      let inCount = 0;
      let outCount = 0;
      for (const dev of devices) {
        if (dev.kind === "audioinput") {
          inCount++;
          const opt = document.createElement("option");
          opt.value = dev.deviceId;
          opt.textContent = dev.label || `Microphone ${inCount}`;
          inSel.appendChild(opt);
        } else if (dev.kind === "audiooutput") {
          outCount++;
          const opt = document.createElement("option");
          opt.value = dev.deviceId;
          opt.textContent = dev.label || `Speaker / Output ${outCount}`;
          outSel.appendChild(opt);
        }
      }

      if (prevIn) inSel.value = prevIn;
      if (prevOut) outSel.value = prevOut;
    } catch (err) {
      console.warn("[NEON] Device enumeration warning:", err);
    }
  }

  async function toggleContinuousStream() {
    if (app.streamActive) {
      await stopContinuousStream();
    } else {
      await startContinuousStream();
    }
  }

  async function startContinuousStream() {
    try {
      const inputDeviceId = $("select-input-device").value;
      const outputDeviceId = $("select-output-device").value;

      const audioConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1
      };
      if (inputDeviceId) audioConstraints.deviceId = { exact: inputDeviceId };

      const micStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
      // Refresh labels now that permission is granted
      refreshAudioDevices();

      const ctx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE,
        latencyHint: "interactive"
      });
      await ctx.audioWorklet.addModule("streamWorklet.js");

      const sourceNode = ctx.createMediaStreamSource(micStream);
      const workletNode = new AudioWorkletNode(ctx, "neon-stream-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: { hopSize: 256, jitterTarget: 768 }
      });

      app.worker.postMessage({ type: "reset_stream" });
      workletNode.port.postMessage({
        type: "config",
        bypass: app.streamBypass,
        wetMix: app.wetMix
      });

      workletNode.port.onmessage = (event) => {
        const msg = event.data;
        if (!msg) return;
        if (msg.type === "capture" && app.streamActive) {
          app.worker.postMessage({ type: "stream_frame", frame: msg.frame }, [msg.frame.buffer]);
        } else if (msg.type === "status") {
          app.underruns = msg.underruns;
          if ($("stream-underruns")) $("stream-underruns").textContent = String(msg.underruns);
        }
      };

      sourceNode.connect(workletNode);

      // Output sink routing: AudioContext.setSinkId with MediaStreamAudioDestinationNode -> <audio>.setSinkId fallback
      app.streamCtx = ctx;
      app.streamMic = micStream;
      app.streamSourceNode = sourceNode;
      app.streamWorkletNode = workletNode;

      await routeStreamOutput(ctx, workletNode, outputDeviceId);

      app.streamActive = true;
      app.rollingLatencies = [];
      app.underruns = 0;

      const btn = $("btn-stream-toggle");
      btn.textContent = "Stop Continuous Mic";
      btn.classList.remove("bg-emerald-600", "hover:bg-emerald-500");
      btn.classList.add("bg-rose-600", "hover:bg-rose-500");
      $("stream-status-pill").textContent = "LIVE STREAMING (16 kHz)";
      $("stream-status-pill").className = "px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
    } catch (err) {
      alert(`Failed to start live mic stream: ${err.message}`);
    }
  }

  async function routeStreamOutput(ctx, workletNode, sinkId) {
    if (sinkId && typeof ctx.setSinkId === "function") {
      try {
        await ctx.setSinkId(sinkId);
        workletNode.connect(ctx.destination);
        return;
      } catch (err) {
        console.warn("[NEON] AudioContext.setSinkId failed, using MediaStreamAudioDestinationNode fallback:", err);
      }
    }

    if (sinkId) {
      const dest = ctx.createMediaStreamDestination();
      workletNode.connect(dest);
      const audioEl = new Audio();
      audioEl.srcObject = dest.stream;
      audioEl.autoplay = true;
      if (typeof audioEl.setSinkId === "function") {
        try {
          await audioEl.setSinkId(sinkId);
        } catch (err) {
          console.warn("[NEON] <audio>.setSinkId warning:", err);
        }
      }
      await audioEl.play().catch(() => {});
      app.streamAudioEl = audioEl;
    } else {
      workletNode.connect(ctx.destination);
    }
  }

  async function applyOutputSink(sinkId) {
    if (!app.streamActive || !app.streamCtx) return;
    if (typeof app.streamCtx.setSinkId === "function" && !app.streamAudioEl) {
      try {
        await app.streamCtx.setSinkId(sinkId);
        return;
      } catch (_) {}
    }
    if (app.streamAudioEl && typeof app.streamAudioEl.setSinkId === "function") {
      try {
        await app.streamAudioEl.setSinkId(sinkId);
      } catch (_) {}
    }
  }

  async function stopContinuousStream() {
    app.streamActive = false;
    if (app.streamSourceNode) app.streamSourceNode.disconnect();
    if (app.streamWorkletNode) app.streamWorkletNode.disconnect();
    if (app.streamAudioEl) {
      app.streamAudioEl.pause();
      app.streamAudioEl.srcObject = null;
      app.streamAudioEl = null;
    }
    if (app.streamMic) app.streamMic.getTracks().forEach((t) => t.stop());
    if (app.streamCtx) await app.streamCtx.close();

    app.streamCtx = null;
    app.streamMic = null;
    app.streamSourceNode = null;
    app.streamWorkletNode = null;

    const btn = $("btn-stream-toggle");
    btn.textContent = "Start Continuous Mic";
    btn.classList.add("bg-emerald-600", "hover:bg-emerald-500");
    btn.classList.remove("bg-rose-600", "hover:bg-rose-500");
    $("stream-status-pill").textContent = "IDLE";
    $("stream-status-pill").className = "px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-400 border border-slate-700";
  }

  function toggleStreamBypass() {
    app.streamBypass = !app.streamBypass;
    const btn = $("btn-bypass-toggle");
    if (app.streamBypass) {
      btn.textContent = "Bypass: ON (Raw Dry Mic)";
      btn.className = "px-4 py-2.5 rounded-lg text-sm font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/50 transition";
    } else {
      btn.textContent = "Bypass: OFF (Neural Active)";
      btn.className = "px-4 py-2.5 rounded-lg text-sm font-semibold bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-700 transition";
    }
    if (app.streamWorkletNode) {
      app.streamWorkletNode.port.postMessage({ type: "config", bypass: app.streamBypass });
    }
  }

  function handleStreamWorkerResult(msg) {
    if (!app.streamActive || !app.streamWorkletNode) return;
    app.streamWorkletNode.port.postMessage({ type: "enhanced", frame: msg.frame }, [msg.frame.buffer]);

    app.rollingLatencies.push(msg.latencyMs);
    if (app.rollingLatencies.length > 50) app.rollingLatencies.shift();
    const avgMs = app.rollingLatencies.reduce((a, b) => a + b, 0) / app.rollingLatencies.length;
    const rtf = avgMs / 16.0; // 16 ms hop

    if ($("stream-latency-ms")) $("stream-latency-ms").textContent = `${avgMs.toFixed(2)} ms`;
    if ($("card-latency")) $("card-latency").textContent = `${avgMs.toFixed(2)} ms`;
    if ($("card-rtf")) $("card-rtf").textContent = `${rtf.toFixed(3)}x`;

    if (msg.profileSync && app.learner) {
      app.learner.syncProfileFromWorker(msg.profileSync.profile, msg.profileSync.calibrationFrames);
      app.learner.saveState().catch(() => {});
      updateLoopAUI(app.learner.getSummary());
    }
  }

  // ==================== LOOP A & LOOP B UI HELPERS ====================

  function updateLoopAUI(summary) {
    const statusEl = $("loop-a-status");
    const framesEl = $("loop-a-frames");
    if (statusEl) {
      statusEl.textContent = summary.hasProfile
        ? `Calibrated (${summary.calibrationFrames} noise frames saved in NoiseFreeMicDB)`
        : "Uncalibrated — waiting for >= 150 ms ambient silence";
    }
    if (framesEl) {
      framesEl.textContent = `Sessions: ${summary.sessionCount}`;
    }
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  }

  async function submitLoopBFeedback() {
    const statusEl = $("feedback-status");
    if (!app.selectedFeedback) {
      statusEl.textContent = "Please select Clean, Over-Suppressed, or Residual Noise first.";
      statusEl.className = "text-xs text-amber-400";
      return;
    }

    const optIn = Boolean($("chk-optin-audio") && $("chk-optin-audio").checked);
    const payload = {
      rating: app.selectedFeedback,
      backend: app.backendLabel,
      rtf: app.lastRtf,
      snrGainDb: app.lastSnrDb,
      optInAudio: optIn
    };

    if (optIn && app.pcmNoisy) {
      // Cap opt-in snippet to first 3 seconds to keep payload compact
      const snippet = app.pcmNoisy.subarray(0, Math.min(app.pcmNoisy.length, SAMPLE_RATE * 3));
      payload.audioWavBase64 = arrayBufferToBase64(encodeWav16k(snippet));
    }

    statusEl.textContent = "Submitting feedback...";
    statusEl.className = "text-xs text-slate-400";
    const res = await app.learner.sendFeedback(payload);

    if (res.offline) {
      statusEl.textContent = "Feedback saved locally (API server offline).";
      statusEl.className = "text-xs text-amber-400";
    } else {
      statusEl.textContent = "Thank you — feedback recorded to NEON telemetry.";
      statusEl.className = "text-xs text-emerald-400";
      refreshDashboard();
    }
  }

  async function refreshDashboard() {
    const container = $("metrics-dashboard-body");
    if (!container || !app.learner) return;

    const res = await app.learner.fetchDashboard();
    if (!res.ok) {
      container.innerHTML =
        '<div class="text-xs text-slate-500">Live backend metrics unavailable (<code>/api/metrics-dashboard</code> offline). Local IndexedDB metrics remain active.</div>';
      return;
    }

    const d = res.data || {};
    const totalRuns = d.total_runs ?? d.totalRuns ?? 0;
    const avgRtf = d.avg_rtf ?? d.avgRtf ?? 0;
    const avgSnr = d.avg_snr_gain_db ?? d.avgSnrGainDb ?? 0;
    const cleanPct = d.clean_rate_pct ?? d.cleanRatePct ?? 0;

    container.innerHTML = `
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div class="bg-slate-900/70 border border-slate-800 rounded-lg p-3">
          <div class="text-slate-400">Total Runs</div>
          <div class="text-base font-bold text-slate-100 mt-0.5">${totalRuns}</div>
        </div>
        <div class="bg-slate-900/70 border border-slate-800 rounded-lg p-3">
          <div class="text-slate-400">Fleet Avg RTF</div>
          <div class="text-base font-bold text-cyan-300 mt-0.5">${Number(avgRtf).toFixed(3)}x</div>
        </div>
        <div class="bg-slate-900/70 border border-slate-800 rounded-lg p-3">
          <div class="text-slate-400">Fleet Avg SNR Gain</div>
          <div class="text-base font-bold text-emerald-300 mt-0.5">+${Number(avgSnr).toFixed(1)} dB</div>
        </div>
        <div class="bg-slate-900/70 border border-slate-800 rounded-lg p-3">
          <div class="text-slate-400">Clean Feedback Rate</div>
          <div class="text-base font-bold text-indigo-300 mt-0.5">${Number(cleanPct).toFixed(1)}%</div>
        </div>
      </div>
    `;
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
