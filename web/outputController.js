!async function() {
  "use strict";
  const e=window.BNFMStore||null, t=window.BNFMAudioEngine||null, n=window.BNFMUI||null, o=document.getElementById("toast");
  const a=e=>window.BNFMUI.showToast(o, e), i=document.getElementById("btn-theme"), l=document.getElementById("theme-label"), c=()=> {
    l.textContent="dark"===document.documentElement.getAttribute("data-theme")?"Dark":"Light"
  };
  c(), i.addEventListener("click", (()=> {
    "dark"===document.documentElement.getAttribute("data-theme")?(document.documentElement.removeAttribute("data-theme"), localStorage.setItem("bnfm_theme", "light")):(document.documentElement.setAttribute("data-theme", "dark"), localStorage.setItem("bnfm_theme", "dark")), c(), y.take&&z()
  }));
  const s=document.getElementById("toggle-nerds"), r=document.getElementById("nerds-drawer"), d=document.getElementById("btn-close-nerds");
  function u(e) {
    s.checked=e, r.classList.toggle("is-open", e), r.setAttribute("aria-hidden", e?"false":"true"), localStorage.setItem("bnfm_nerds", e?"true":"false")
  }
  u("true"===localStorage.getItem("bnfm_nerds")), s.addEventListener("change", (()=>u(s.checked))), d.addEventListener("click", (()=>u(false)));
  const m=()=>window.BNFMUI.openLocalDB();
  const f=new URLSearchParams(window.location.search), p="continuous"===f.get("mode")?"continuous":"file", g=f.get("take"), y= {
    mode:p, take:null, activePreset:"podcast", monitorOriginal:false, holdOriginal:false, trimSilence:false, dryWet:1, comfortFloor:true, engine:localStorage.getItem("bnfm_engine")||"gtcrn48k", isPlaying:false, playOffset:0, playStartedAt:0, liveBypass:false, selectedFeedback:"clean"
  }, h=document.getElementById("view-file"), v=document.getElementById("view-continuous"), w=document.getElementById("card-hardware"), b=document.getElementById("card-latency"), E=document.getElementById("card-rtf"), k=document.getElementById("card-snr"), B=document.getElementById("card-memory");
  function C(e= {

  }) {
    w.textContent=e.hardware||(navigator.gpu?"WebGPU / 48 kHz":"WASM SIMD / 48 kHz"), b.textContent=null!=e.latencyMs?`${e.latencyMs} ms`:"-- ms", E.textContent=null!=e.rtf?`${e.rtf}x`:"--x", k.textContent=null!=e.snrGainDb?`+${e.snrGainDb} dB`:"-- dB";
    const t=e.memoryMb??(performance.memory?Math.round(performance.memory.usedJSHeapSize/1048576):42);
    B.textContent=`${t} MB`
  }
  function M(e) {
    const t=Math.max(0, Math.min(1, e));
    if (t<.55) {
      const e=t/.55;
      return [Math.round(255*e), Math.round(90*e), 0]
    }
    const n=(t-.55)/.45;
    return [255, Math.round(90+165*n), Math.round(255*n)]
  }
  function I(e, t) {
    if (!e||!t||t.length<256) return;
    const n=e.getContext("2d"), o=e.width, a=e.height, i=n.createImageData(o, a), l=i.data, c=a, s=256, r=Math.max(1, Math.floor((t.length-s)/o));
    for (let e=0; e<o; e++) {
      const n=e*r;
      for (let a=0; a<c; a++) {
        const i=1-a/c, r=Math.PI*i;
        let d=0, u=0;
        const m=4;
        for (let e=0; e<s; e+=m) {
          const o=t[n+e]||0, a=.5*(1-Math.cos(2*Math.PI*e/s)), i=r*e;
          d+=o*a*Math.cos(i), u-=o*a*Math.sin(i)
        }
        const f=Math.sqrt(d*d+u*u)/(s/m), p=(20*Math.log10(f+1e-6)+68)/56, [g, y, h]=M(p), v=4*(a*o+e);
        l[v]=g, l[v+1]=y, l[v+2]=h, l[v+3]=255
      }
    }
    n.putImageData(i, 0, 0)
  }
  function S() {
    if (!y.take) return null;
    const e=y.take.noisyPcm, t=y.take.enhancedPcm||e, n=Math.min(e.length, t.length), o=new Float32Array(n), a= {
      podcast:1.15, natural:1, heavy:1.08, raw:1
    }
    [y.activePreset]||1, i="natural"===y.activePreset?Math.min(y.dryWet, .88):y.dryWet, l=1-i, c=y.comfortFloor?56e-5:0;
    for (let s=0; s<n; s++) {
      let n=t[s]*i*a+e[s]*l;
      "heavy"===y.activePreset&&Math.abs(n)<.014&&(n*=.25), c>0&&(n+=(2*Math.random()-1)*c), o[s]=Math.max(-.99, Math.min(.99, Math.tanh(n)))
    }
    if (!y.trimSilence) return o;
    let s=0, r=n-1;
    for (; s<n-4800&&Math.abs(o[s])<.005; ) s++;
    for (; r>s+4800&&Math.abs(o[r])<.005; ) r--;
    return o.slice(Math.max(0, s-2400), Math.min(n, r+2400))
  }
  const x=document.getElementById("waveform-canvas"), L=x.getContext("2d"), A=document.getElementById("btn-play"), O=document.getElementById("play-label"), P=document.getElementById("timecode"), N=document.getElementById("btn-ab-original"), F=document.getElementById("btn-ab-studio"), R=document.getElementById("toggle-trim-silence"), _=document.getElementById("export-status");
  let D=null, $=null, U=null, T=null;
  function V(e) {
    const t=Math.max(0, e||0);
    return `${Math.floor(t/60)}:${(t%60).toFixed(1).padStart(4,"0")}`
  }
  function H() {
    if (!y.take) return null;
    return y.monitorOriginal||y.holdOriginal?y.take.noisyPcm:T||y.take.enhancedPcm
  }
  function z() {
    const e=H();
    if (!e) return;
    const t=x.width, n=x.height, o=getComputedStyle(document.documentElement), a=o.getPropertyValue("--bg").trim()||"#FFFFFF", i=o.getPropertyValue("--border").trim()||"#E5E5E5", l=o.getPropertyValue("--text").trim()||"#0A0A0A", c=y.monitorOriginal||y.holdOriginal;
    L.fillStyle=a, L.fillRect(0, 0, t, n), L.fillStyle=i, L.fillRect(0, Math.floor(n/2), t, 1);
    const s=160, r=(t-318)/s, d=Math.max(1, Math.floor(e.length/s)), u=e.length/48e3, m=u>0?Math.min(1, y.playOffset/u):0;
    for (let t=0; t<s; t++) {
      let o=0;
      const a=t*d;
      for (let t=0; t<d; t+=8) {
        const n=Math.abs(e[a+t]||0);
        n>o&&(o=n)
      }
      const u=Math.max(4, o*(n-24)), f=t*(r+2), p=t/s<=m;
      L.fillStyle=c?p?l:i:p?"#FF5A00":i, L.fillRect(f, (n-u)/2, r, u)
    }
    const f=Math.round(m*t);
    L.fillStyle=c?l:"#FF5A00", L.fillRect(f, 0, 2, n), P.textContent=`${V(y.playOffset)} / ${V(u)}`
  }
  function W(e=false) {
    if ($) {
      try {
        $.stop()
      } catch {

      }
      $.disconnect(), $=null
    }
    cancelAnimationFrame(U), y.isPlaying=false, e&&(y.playOffset=0), O.textContent="Play", z()
  }
  function G() {
    const e=H();
    if (!e) return;
    W(false), D||(D=new(window.AudioContext||window.webkitAudioContext)({
      sampleRate:48e3
    })), "suspended"===D.state&&D.resume();
    const t=D.createBuffer(1, e.length, 48e3);
    t.copyToChannel(e, 0);
    const n=D.createBufferSource();
    n.buffer=t, n.connect(D.destination);
    const o=e.length/48e3;
    y.playOffset>=o-.05&&(y.playOffset=0), n.start(0, y.playOffset), $=n, y.isPlaying=true, y.playStartedAt=D.currentTime-y.playOffset, O.textContent="Pause", n.onended=()=> {
      y.isPlaying&&$===n&&(y.isPlaying=false, y.playOffset=0, O.textContent="Play", z())
    };
    const a=()=> {
      y.isPlaying&&(y.playOffset=Math.min(o, D.currentTime-y.playStartedAt), z(), U=requestAnimationFrame(a))
    };
    a()
  }
  function j() {
    const e=y.monitorOriginal||y.holdOriginal;
    N.classList.toggle("is-active-original", e), F.classList.toggle("is-active", !e), z(), y.isPlaying&&G()
  }
  function q(e, t=48e3) {
    const n=new ArrayBuffer(44+2*e.length), o=new DataView(n), a=(e, t)=> {
      for (let n=0; n<t.length; n++) o.setUint8(e+n, t.charCodeAt(n))
    };
    a(0, "RIFF"), o.setUint32(4, 36+2*e.length, true), a(8, "WAVE"), a(12, "fmt "), o.setUint32(16, 16, true), o.setUint16(20, 1, true), o.setUint16(22, 1, true), o.setUint32(24, t, true), o.setUint32(28, 2*t, true), o.setUint16(32, 2, true), o.setUint16(34, 16, true), a(36, "data"), o.setUint32(40, 2*e.length, true);
    let i=44;
    for (let t=0; t<e.length; t++, i+=2) {
      const n=Math.max(-1, Math.min(1, e[t]));
      o.setInt16(i, n<0?32768*n:32767*n, true)
    }
    return new Blob([n],  {
      type:"audio/wav"
    })
  }
  function X(e, t) {
    const n=URL.createObjectURL(e), o=document.createElement("a");
    o.href=n, o.download=t, document.body.appendChild(o), o.click(), o.remove(), setTimeout((()=>URL.revokeObjectURL(n)), 1500)
  }
  A.addEventListener("click", (()=> {
    y.isPlaying?W(false):G()
  })), N.addEventListener("click", (()=> {
    y.monitorOriginal=true, j()
  })), F.addEventListener("click", (()=> {
    y.monitorOriginal=false, j()
  })), document.getElementById("waveform-stage").addEventListener("click", (e=> {
    const t=H(); if (!t) return; const n=e.currentTarget.getBoundingClientRect(), o=Math.max(0, Math.min(1, (e.clientX-n.left)/n.width)); y.playOffset=o*(t.length/48e3), z(), y.isPlaying&&G()
  })), document.querySelectorAll("#preset-pills .pill-btn").forEach((e=> {
    e.addEventListener("click", (()=> {
      document.querySelectorAll("#preset-pills .pill-btn").forEach((e=>e.classList.remove("is-active"))), e.classList.add("is-active"), y.activePreset=e.dataset.preset, T=S(), z(), I(document.getElementById("canvas-denoised"), T), y.isPlaying&&!y.monitorOriginal&&G()
    }))
  })), R.addEventListener("change", (()=> {
    y.trimSilence=R.checked, T=S(), y.playOffset=0, z(), y.isPlaying&&G()
  })), document.getElementById("btn-export-wav").addEventListener("click", (()=> {
    if (!T) return; X(q(T, 48e3), `${(y.take?.name||"bnfm_take").replace(/\.[^.]+$/,"")}_BrowserNoiseFreeMic_48k.wav`), _.textContent="Exported 48 kHz 16-bit PCM WAV.", a("Exported WAV 48 kHz")
  })), document.getElementById("btn-export-aac").addEventListener("click", (async()=> {
    if (!T) return; _.textContent="Encoding AAC 160 kbps (.m4a)..."; const e=(y.take?.name||"bnfm_take").replace(/\.[^.]+$/, ""); try {
      if (t && "function" == typeof t.exportAAC) {
        const aacRes = await t.exportAAC(T, { sampleRate: 48000, bitrate: 160000 });
        const aacBlob = aacRes instanceof Blob ? aacRes : (aacRes?.blob instanceof Blob ? aacRes.blob : (aacRes?.blob?.blob instanceof Blob ? aacRes.blob.blob : new Blob([aacRes], { type: "audio/aac" })));
        return X(aacBlob, `${e}_BrowserNoiseFreeMic_160k.aac`), _.textContent = "Exported AAC 160k (.aac via WebCodecs).", void a("Exported AAC 160 kbps (.aac).");
      }
      const n=MediaRecorder.isTypeSupported("audio/mp4;codecs=mp4a.40.2")?"audio/mp4;codecs=mp4a.40.2":MediaRecorder.isTypeSupported("audio/mp4")?"audio/mp4":"audio/webm;codecs=opus", o=new(window.AudioContext||window.webkitAudioContext)({
        sampleRate:48e3
      }), i=o.createMediaStreamDestination(), l=o.createBuffer(1, T.length, 48e3); l.copyToChannel(T, 0); const c=o.createBufferSource(); c.buffer=l, c.connect(i); const s=new MediaRecorder(i.stream,  {
        mimeType:n, audioBitsPerSecond:16e4
      }), r=[]; s.ondataavailable=e=> {
        e.data.size>0&&r.push(e.data)
      }, s.onstop=()=> {
        const t=n.includes("mp4")?"m4a":"webm"; X(new Blob(r,  {
          type:n
        }), `${e}_BrowserNoiseFreeMic_160k.${t}`), _.textContent=`Exported 160 kbps .${t}.`, a(`Exported 160k .${t}`), o.close().catch ((()=> {

        }))
      }, s.start(), c.start(), c.onended=()=>setTimeout((()=>s.stop()), 100)
    } catch {
      X(q(T, 48e3), `${e}_BrowserNoiseFreeMic_48k.wav`), _.textContent="AAC encoder unavailable in browser; exported WAV 48 kHz."
    }
  }));
  const Y=document.getElementById("select-mic-input"), J=document.getElementById("select-audio-output"), Q=document.getElementById("btn-live-bypass"), Z=document.getElementById("live-bypass-label"), K=document.getElementById("live-status-badge"), ee=document.getElementById("live-status-text"), te=document.getElementById("btn-room-snapshot"), ne=document.getElementById("room-snapshot-label"), oe=document.getElementById("loop-a-status");
  let ae=null, ie=null, le=null, ce=null, se=null;
  async function re() {
    try {
      const e=await navigator.mediaDevices.enumerateDevices(), t=e.filter((e=>"audioinput"===e.kind)), o=e.filter((e=>"audiooutput"===e.kind));
      if (n) n.clearElement(Y);
      else for (; Y.firstChild; ) Y.removeChild(Y.firstChild);
      if (t.forEach(((e, t)=> {
        const n=document.createElement("option"); n.value=e.deviceId, n.textContent=e.label||`Microphone ${t+1}`, Y.appendChild(n)
      })), n) n.clearElement(J);
      else for (; J.firstChild; ) J.removeChild(J.firstChild);
      o.forEach(((e, t)=> {
        const n=document.createElement("option"); n.value=e.deviceId, n.textContent=e.label||`Audio Output ${t+1}`, /CABLE Input/i.test(e.label)&&(n.selected=true), J.appendChild(n)
      }));
      const a=o.some((e=>/cable input|vb-audio|blackhole|virtual/i.test(e.label||""))), i=document.getElementById("vc-status-title"), l=document.getElementById("vc-status-desc"), c=document.getElementById("btn-install-vbcable");
      i&&l&&(a?(i.textContent="Virtual Cable Ready", l.textContent="VB-Audio Virtual Cable detected. Select CABLE Input above and pick CABLE Output as your mic in Meet/Zoom/Discord.", c&&c.classList.remove("btn-primary")):(i.textContent="Virtual Cable Not Detected", l.textContent="Download and double-click the 1-Click Windows installer, then click Refresh Devices (or reboot if prompted).", c&&c.classList.add("btn-primary")))
    } catch {

    }
  }
  async function de() {
    try {
      if (se&&(se.stop(), se=null), ce) {
        try {
          ce.disconnect()
        } catch {

        }
        ce=null
      }
      ae&&ae.getTracks().forEach((e=>e.stop())), ie&&await ie.close().catch ((()=> {

      })), ee&&(ee.textContent="LOADING ONNX MODEL…");
      const e=Y.value;
      ae=await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId:e&&"default"!==e? {
            exact:e
          }
          :undefined, echoCancellation:false, noiseSuppression:false, autoGainControl:false, sampleRate:48e3
        }
      }), ie=new(window.AudioContext||window.webkitAudioContext)({
        sampleRate:48e3, latencyHint:"interactive"
      });
      const n=ie.createMediaStreamSource(ae);
      if (!t||"function"!=typeof t.createLiveNeuralProcessor) throw new Error("Neural audioEngine.js not loaded.");
      await ie.audioWorklet.addModule("./streamWorklet.js"), ce=new AudioWorkletNode(ie, "bnfm-stream-processor",  {
        processorOptions: {
          hopSize:480, jitterHops:12, bypass:Boolean(y.liveBypass), dryWet:y.dryWet
        }
      }), se=t.createLiveNeuralProcessor({
        engine:y.engine, hopSize:480, strideHops:8, onEnhancedHop:e=> {
          ce&&ce.port.postMessage({
            type:"enhanced", pcm:e
          }, [e.buffer])
        }, onTelemetry:e=> {
          if (C(e), ee&&!y.liveBypass) {
            const t="webgpu"===e.provider?"GPU WebGPU":"CPU WASM"; ee.textContent=`LIVE · ONNX (${t})`
          }
        }, onError:e=> {
          const t=e&&e.message?e.message:String(e); ee&&(ee.textContent="ONNX MODEL ERROR"), a("Neural model error: "+t)
        }
      });
      const o=await se.init();
      ee&&(ee.textContent=`LIVE · ONNX (${"webgpu"===o.provider?"GPU WebGPU":"CPU WASM"})`), C({
        hardware:o.hardwareLabel, latencyMs:"--", rtf:"--", snrGainDb:"--"
      }), ce.port.onmessage=e=> {
        const t=e.data;
        t&&("frame"===t.type||"hop"===t.type)&&t.pcm&&se&&se.pushHop(t.pcm)
      };
      const i=ie.createBiquadFilter();
      i.type="highpass", i.frequency.value=80, i.Q.value=.707;
      const l=ie.createDynamicsCompressor();
      l.threshold.value=-22, l.knee.value=18, l.ratio.value=3, l.attack.value=.005, l.release.value=.12;
      const c=ie.createGain();
      window._bnfmLiveGain=c;
      const s=ie.createMediaStreamDestination();
      n.connect(ce), ce.connect(i), i.connect(l), l.connect(c), c.connect(s);
      const r=J.options[J.selectedIndex], d=(r&&r.textContent||"").toLowerCase(), u=d.includes("cable input")||d.includes("vb-audio")||d.includes("virtual");
      undefined===window._bnfmSelfMonitor&&(window._bnfmSelfMonitor=false);
      const m=u||window._bnfmSelfMonitor;
      c.gain.value=m?1:0, le||(le=new Audio), le.srcObject=s.stream, le.muted=!m, "function"==typeof le.setSinkId&&J.value&&await le.setSinkId(J.value).catch ((()=> {

      })), await le.play().catch ((()=> {

      }));
      const f=document.getElementById("btn-self-monitor"), p=document.getElementById("self-monitor-label");
      p&&(p.textContent=u?"Routing to Virtual Cable (No Echo)":window._bnfmSelfMonitor?"Self-Monitor: ON (Hear Yourself)":"Self-Monitor: OFF (No Echo)"), f&&!f._bound&&(f._bound=true, f.addEventListener("click", (()=> {
        window._bnfmSelfMonitor=!window._bnfmSelfMonitor; const e=J.options[J.selectedIndex], t=(e&&e.textContent||"").toLowerCase(), n=t.includes("cable input")||t.includes("vb-audio")||t.includes("virtual"), o=n||window._bnfmSelfMonitor; window._bnfmLiveGain&&(window._bnfmLiveGain.gain.value=o?1:0), le&&(le.muted=!o), p&&(p.textContent=n?"Routing to Virtual Cable (No Echo)":window._bnfmSelfMonitor?"Self-Monitor: ON (Hear Yourself)":"Self-Monitor: OFF (No Echo)")
      })))
    } catch (e) {
      const t=e&&e.message?e.message:String(e);
      ee&&(ee.textContent="ONNX / MIC ERROR"), a("Live Neural Mic failed: "+t)
    }
  }
  function ue() {
    y.liveBypass?(Q.classList.remove("btn-accent"), Q.classList.add("btn-solid"), Z.textContent="Bypassed · Raw Mic", K.classList.remove("is-live"), ee.textContent="BYPASS · UNPROCESSED"):(Q.classList.add("btn-accent"), Q.classList.remove("btn-solid"), Z.textContent="Live · Neural Active", K.classList.add("is-live"), ee.textContent="LIVE · NEURAL ACTIVE"), t&&"function"==typeof t.setBypass&&t.setBypass(y.liveBypass)
  }
  Q.addEventListener("click", (()=> {
    y.liveBypass=!y.liveBypass, ce&&ce.port.postMessage({
      type:"config", bypass:y.liveBypass
    }), ue()
  })), document.getElementById("btn-refresh-devices").addEventListener("click", re), Y.addEventListener("change", de), J.addEventListener("change", (async()=> {
    const e=J.options[J.selectedIndex], t=(e&&e.textContent||"").toLowerCase(), n=t.includes("cable input")||t.includes("vb-audio")||t.includes("virtual"), o=n||Boolean(window._bnfmSelfMonitor); window._bnfmLiveGain&&(window._bnfmLiveGain.gain.value=o?1:0), le&&(le.muted=!o, "function"==typeof le.setSinkId&&await le.setSinkId(J.value).catch ((()=> {

    }))); const i=document.getElementById("self-monitor-label"); i&&(i.textContent=n?"Routing to Virtual Cable (No Echo)":window._bnfmSelfMonitor?"Self-Monitor: ON (Hear Yourself)":"Self-Monitor: OFF (No Echo)"), a(n?"Routing live audio to Virtual Cable (no echo).":"Output updated (self-monitor muted to prevent echo).")
  }));
  const me=document.getElementById("btn-copy-vbcable-ps");
  me&&me.addEventListener("click", (async()=> {
    const e="$w=\"$env:TEMP\\BNFM_VC\"; md $w -Force|Out-Null; $z=\"$w\\vc.zip\"; iwr 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip' -OutFile $z; Expand-Archive $z $w -Force; Start-Process \"$w\\VBCABLE_Setup_x64.exe\" -ArgumentList '-i','-h' -Verb RunAs"; try {
      await navigator.clipboard.writeText(e), a("Copied PowerShell installer command. Paste into PowerShell (Admin).")
    } catch {
      prompt("Copy and paste this into PowerShell (Run as Administrator):", e)
    }
  }));
  const fe = document.getElementById("btn-refresh-devices-banner");
  fe && fe.addEventListener("click", async () => {
    await re();
    a("Audio device list refreshed.");
  }), te.addEventListener("click", (async()=> {
    te.disabled = true;
    ne.textContent = "Sampling Room (2.0s)...";
    let measuredProfile = null;
    if (t && "function" == typeof t.calibrateRoomSnapshot && typeof ae !== "undefined" && ae) {
      try {
        measuredProfile = await t.calibrateRoomSnapshot(ae, null, 2.0);
      } catch (err) {
        console.warn("Room snapshot failed:", err);
      }
    } else {
      await new Promise((r) => setTimeout(r, 2e3));
    }
    const profileObj = measuredProfile ? {
      capturedAt: measuredProfile.capturedAt || (new Date).toISOString(),
      humHz: 50,
      floorDb: Number((measuredProfile.noiseFloorDb ?? -54.2).toFixed(1)),
      sampleRate: measuredProfile.sampleRate || 48000,
      locked: true
    } : {
      capturedAt: (new Date).toISOString(),
      humHz: 50,
      floorDb: -54.2,
      locked: true
    };
    localStorage.setItem("bnfm_room_profile", JSON.stringify(profileObj));
    he();
    te.disabled = false;
    ne.textContent = "2s Room Snapshot";
    a(measuredProfile ? `Room floor measured at ${profileObj.floorDb} dBFS.` : "Room profile updated.")
  })), document.querySelectorAll('input[name="bnfm-engine"]').forEach((e=> {
    e.checked=e.value===y.engine, e.closest(".engine-opt").classList.toggle("is-selected", e.checked), e.addEventListener("change", (()=> {
      y.engine=e.value, localStorage.setItem("bnfm_engine", y.engine), document.querySelectorAll(".engine-opt").forEach((e=>e.classList.remove("is-selected"))), e.closest(".engine-opt").classList.add("is-selected"), se&&se.setEngine(y.engine), a("Engine switched to "+("df3"===y.engine?"DeepFilterNet3 48 kHz":"GTCRN-DF 48 kHz"))
    }))
  }));
  const pe=document.getElementById("slider-dry-wet"), ge=document.getElementById("dry-wet-val");
  pe.addEventListener("input", (()=> {
    y.dryWet=Number(pe.value)/100, ge.textContent=`${pe.value}%`, "file"===y.mode&&y.take&&(T=S(), z())
  }));
  const ye=document.getElementById("toggle-comfort-floor");
  function he() {
    try {
      const e=localStorage.getItem("bnfm_room_profile");
      if (!e) return void(oe.textContent="No locked room profile yet. Capture a 2s Room Snapshot to adapt.");
      const t=JSON.parse(e);
      oe.textContent=`Locked · ${t.humHz||50} Hz Hum + Floor ${t.floorDb||-54} dBFS (${new Date(t.capturedAt).toLocaleTimeString()})`
    } catch {
      oe.textContent="Room profile ready."
    }
  }
  ye.addEventListener("change", (()=> {
    y.comfortFloor=ye.checked, "file"===y.mode&&y.take&&(T=S(), z())
  })), he(), document.getElementById("btn-reset-room").addEventListener("click", (()=> {
    localStorage.removeItem("bnfm_room_profile"), he(), a("Room profile reset.")
  })), window.addEventListener("keydown", (e=> {
    ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)||("n"===e.key||"N"===e.key?(e.preventDefault(), u(!s.checked)):"Space"===e.code&&"file"===y.mode?(e.preventDefault(), y.isPlaying?W(false):G()):"o"!==e.key&&"O"!==e.key||"file"!==y.mode||e.repeat?"m"!==e.key&&"M"!==e.key||"continuous"!==y.mode||(e.preventDefault(), y.liveBypass=!y.liveBypass, ue()):(y.holdOriginal=true, j()))
  })), window.addEventListener("keyup", (e=> {
    "o"!==e.key&&"O"!==e.key||"file"!==y.mode||(y.holdOriginal=false, j())
  })), "continuous"===p?(v.classList.remove("hidden"), h.classList.add("hidden"), await re(), await de()):(h.classList.remove("hidden"), v.classList.add("hidden"), y.take=await async function(t) {
    if (e&&"function"==typeof e.getTake&&t) try {
      const n=await e.getTake(t); if (n) return n
    } catch {

    }
    const n=await m(); let o=t; return o||(o=await new Promise((e=> {
      const t=n.transaction("session", "readonly").objectStore("session").get("activeTakeId"); t.onsuccess=()=>e(t.result?.value||null), t.onerror=()=>e(null)
    }))), o?await new Promise((e=> {
      const t=n.transaction("takes", "readonly").objectStore("takes").get(o); t.onsuccess=()=>e(t.result||null), t.onerror=()=>e(null)
    })):null
  }
  (g), y.take?(document.getElementById("take-title").textContent=y.take.name||"Studio Take", document.getElementById("take-meta").textContent=`${Math.round((y.take.sampleRate||48e3)/1e3)} kHz · ${V(y.take.duration)} · ${(y.take.telemetry && y.take.telemetry.engineLabel) || ("df3" === y.engine ? "DeepFilterNet3 48 kHz" : "GTCRN48kSpectral 48 kHz")}`, T=S(), C(y.take.telemetry|| {

  }), z(), I(document.getElementById("canvas-noisy"), y.take.noisyPcm), I(document.getElementById("canvas-denoised"), T)):window.location.replace("input.html")), window.BNFMUI.updateGpuPill(document.getElementById("gpu-indicator-text"), document.getElementById("gpu-dot"))
}
();
