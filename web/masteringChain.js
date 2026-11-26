!function(e, t) {
  "object"==typeof module&&module.exports?module.exports=t():(e.BNFMMasteringChain = e.BNFMMastering = t())
}
("undefined"!=typeof globalThis?globalThis:this, (function() {
  "use strict"; const e=48e3, t=-1, a=Math.pow(10, -.05), n=Math.pow(10, -3.25), r=-48, o=40, s=.5411961, i=1.30656296, c=Object.freeze({
    podcast:Object.freeze({
      id:"podcast", label:"Podcast Studio Cut", description:"Broadcast-ready warmth, presence, -16 LUFS loudness & -1 dBFS ceiling.", dryWet:1, bypassAll:false, hpfHz:80, gateEnabled:false, gateThresholdDb:-50, gateFloorDb:-12, eq: {
        boxinessHz:280, boxinessGainDb:-2, boxinessQ:1.2, presenceHz:3500, presenceGainDb:1.5, presenceQ:.7, airHz:11e3, airGainDb:1.5, airQ:.707
      }, comp: {
        thresholdDb:-18, ratio:3, kneeDb:6, attackMs:10, releaseMs:120, makeupDb:2
      }, targetLufs:-16, limiterCeilingDb:t, comfortNoise:true, edgeTrim:true
    }), natural:Object.freeze({
      id:"natural", label:"Natural Voice", description:"82% wet blend with gentle EQ and light dynamics to preserve room intimacy.", dryWet:.82, bypassAll:false, hpfHz:80, gateEnabled:false, gateThresholdDb:-52, gateFloorDb:-8, eq: {
        boxinessHz:280, boxinessGainDb:-1, boxinessQ:1.2, presenceHz:3500, presenceGainDb:.8, presenceQ:.7, airHz:11e3, airGainDb:.8, airQ:.707
      }, comp: {
        thresholdDb:-20, ratio:2, kneeDb:8, attackMs:15, releaseMs:140, makeupDb:1
      }, targetLufs:-16, limiterCeilingDb:t, comfortNoise:true, edgeTrim:true
    }), heavy:Object.freeze({
      id:"heavy", label:"Heavy Fan / Traffic", description:"100% wet with tighter 110 Hz HPF, soft noise gate, and firmer vocal compression.", dryWet:1, bypassAll:false, hpfHz:110, gateEnabled:true, gateThresholdDb:-44, gateFloorDb:-14, eq: {
        boxinessHz:280, boxinessGainDb:-2.5, boxinessQ:1.2, presenceHz:3500, presenceGainDb:2, presenceQ:.7, airHz:11e3, airGainDb:1.2, airQ:.707
      }, comp: {
        thresholdDb:-16, ratio:3.5, kneeDb:5, attackMs:8, releaseMs:110, makeupDb:2.5
      }, targetLufs:-16, limiterCeilingDb:t, comfortNoise:true, edgeTrim:true
    }), raw:Object.freeze({
      id:"raw", label:"Raw Neural", description:"Pure neural denoiser output with EQ, compression, and loudness bypassed.", dryWet:1, bypassAll:true, hpfHz:0, gateEnabled:false, eq:null, comp:null, targetLufs:null, limiterCeilingDb:t, comfortNoise:false, edgeTrim:false
    })
  }); function l(e, t, a) {
    const n=2*Math.PI*e/a, r=Math.cos(n), o=Math.sin(n)/(2*t), s=1+o; return {
      b0:(1+r)/2/s, b1:-(1+r)/s, b2:(1+r)/2/s, a1:-2*r/s, a2:(1-o)/s
    }
  }
  function h(e, t, a, n) {
    const r=Math.pow(10, t/40), o=2*Math.PI*e/n, s=Math.cos(o), i=Math.sin(o)/(2*a), c=1+i/r; return {
      b0:(1+i*r)/c, b1:-2*s/c, b2:(1-i*r)/c, a1:-2*s/c, a2:(1-i/r)/c
    }
  }
  function m(e, t, a, n) {
    const r=Math.pow(10, t/40), o=2*Math.PI*e/n, s=Math.cos(o), i=Math.sin(o)/(2*a), c=2*Math.sqrt(r)*i, l=r+1-(r-1)*s+c; return {
      b0:r*(r+1+(r-1)*s+c)/l, b1:-2*r*(r-1+(r+1)*s)/l, b2:r*(r+1+(r-1)*s-c)/l, a1:2*(r-1-(r+1)*s)/l, a2:(r+1-(r-1)*s-c)/l
    }
  }
  function p(e, t) {
    const {
      b0:a, b1:n, b2:r, a1:o, a2:s
    }
    =t; let i=0, c=0; for (let t=0; t<e.length; t++) {
      const l=e[t], h=a*l+i; i=n*l-o*h+c, c=r*l-s*h, e[t]=h
    }
  }
  function u(e, t) {
    if (!e||0===e.length) return -70; const a=new Float32Array(e); p(a, m(1681.97, 4, .707, t)), p(a, l(38.13, .5003, t)); const n=Math.max(1, Math.round(.4*t)), r=Math.max(1, Math.round(.1*t)), o=[]; for (let e=0; e<a.length; e+=r) {
      const t=Math.min(e+n, a.length); let r=0; for (let n=e; n<t; n++) r+=a[n]*a[n]; const s=r/(t-e); 10*Math.log10(s+1e-12)-.691>-70&&o.push(s)
    }
    if (0===o.length) {
      let e=0; for (let t=0; t<a.length; t++) e+=a[t]*a[t]; const t=e/a.length; return 10*Math.log10(t+1e-12)-.691
    }
    const s=o.reduce(((e, t)=>e+t), 0)/o.length, i=10*Math.log10(s+1e-12)-.691-10, c=o.filter((e=>10*Math.log10(e+1e-12)-.691>i)), h=c.length>0?c.reduce(((e, t)=>e+t), 0)/c.length:s; return 10*Math.log10(h+1e-12)-.691
  }
  function d(e, t, a) {
    const n=Math.pow(10, t/20), r=Math.exp(-1/(5e-4*a)), o=Math.exp(-1/(.06*a)); let s=1; for (let t=0; t<e.length; t++) {
      const a=e[t], i=Math.abs(a), c=i>n?n/(i+1e-9):1, l=c<s?r:o; s=l*s+(1-l)*c; const h=a*s; e[t]=n*Math.tanh(h/n)
    }
  }
  function b(e, t=n) {
    let a=0, r=0, o=0, s=0, i=0, c=0, l=0; const h=.19*t; for (let t=0; t<e.length; t++) {
      const n=2*Math.random()-1; a=.99886*a+.0555179*n, r=.99332*r+.0750759*n, o=.969*o+.153852*n, s=.8665*s+.3104856*n, i=.55*i+.5329522*n, c=-.7616*c-.016898*n; const m=a+r+o+s+i+c+l+.5362*n; l=.115926*n, e[t]+=m*h
    }
  }
  function f(e, t, a=r, n=o) {
    const s=e.length; if (0===s) return {
      pcm:e, trimmedStartMs:0, trimmedEndMs:0
    }; const i=Math.pow(10, a/20), c=Math.max(1, Math.round(.01*t)), l=Math.round(n/1e3*t); let h=-1, m=-1; for (let t=0; t<s; t+=c) {
      const a=Math.min(t+c, s); let n=0; for (let r=t; r<a; r++) n+=e[r]*e[r]; Math.sqrt(n/(a-t))>=i&&(-1===h&&(h=t), m=a)
    }
    if (-1===h) return {
      pcm:e, trimmedStartMs:0, trimmedEndMs:0
    }; const p=Math.max(0, h-l), u=Math.min(s, m+l), d=e.slice(p, u), b=Math.min(l, Math.floor(d.length/4)); for (let e=0; e<b; e++) {
      const t=.5*(1-Math.cos(Math.PI*e/b)); p>0&&(d[e]*=t), u<s&&(d[d.length-1-e]*=t)
    }
    return {
      pcm:d, trimmedStartMs:Math.round(p/t*1e3), trimmedEndMs:Math.round((s-u)/t*1e3)
    }
  }
  function g(e) {
    let t=0; for (let a=0; a<e.length; a++) {
      const n=Math.abs(e[a]); n>t&&(t=n)
    }
    return 20*Math.log10(t+1e-9)
  }
  return {
    DEFAULT_SR:e, TRUE_PEAK_CEILING_DB:t, COMFORT_NOISE_DB:-65, EDGE_TRIM_THRESHOLD_DB:r, EDGE_TRIM_PAD_MS:o, PRESETS:c, processOffline:function(t, a= {

    }) {
      const M=a.sampleRate||e, T=a.preset&&c[a.preset]?a.preset:"podcast", D=c[T], y="boolean"==typeof a.comfortNoise?a.comfortNoise:D.comfortNoise, x="boolean"==typeof a.edgeTrim?a.edgeTrim:D.edgeTrim, A="number"==typeof a.dryWet?Math.max(0, Math.min(1, a.dryWet)):D.dryWet; let z=new Float32Array(t); const H=u(z, M); if (a.dryPCM&&A<.999) {
        const e=a.dryPCM, t=Math.min(z.length, e.length), n=1-A; for (let a=0; a<t; a++) z[a]=z[a]*A+e[a]*n
      }
      if (D.bypassAll) return d(z, D.limiterCeilingDb, M),  {
        pcm:z, stats: {
          preset:D.id, sampleRate:M, inputLufs:Number(H.toFixed(2)), outputLufs:Number(u(z, M).toFixed(2)), peakDb:Number(g(z).toFixed(2)), trimmedStartMs:0, trimmedEndMs:0, comfortNoise:false, edgeTrim:false
        }
      }; if (D.hpfHz>0&&(p(z, l(D.hpfHz, s, M)), p(z, l(D.hpfHz, i, M))), D.gateEnabled&&function(e, t, a, n) {
        const r=Math.pow(10, t/20), o=Math.pow(10, a/20), s=Math.exp(-1/(.005*n)), i=Math.exp(-1/(.08*n)), c=Math.exp(-1/(.012*n)); let l=0, h=1; for (let t=0; t<e.length; t++) {
          l=c*l+(1-c)*Math.abs(e[t]); const a=l>=r?1:o, n=a>h?s:i; h=n*h+(1-n)*a, e[t]*=h
        }
      }
      (z, D.gateThresholdDb, D.gateFloorDb, M), D.eq) {
        const {
          eq:e
        }
        =D; p(z, h(e.boxinessHz, e.boxinessGainDb, e.boxinessQ, M)), p(z, h(e.presenceHz, e.presenceGainDb, e.presenceQ, M)), p(z, m(e.airHz, e.airGainDb, e.airQ, M))
      }
      if (D.comp&&function(e, t, a) {
        const {
          thresholdDb:n, ratio:r, kneeDb:o, attackMs:s, releaseMs:i, makeupDb:c
        }
        =t, l=Math.exp(-1/(s/1e3*a)), h=Math.exp(-1/(i/1e3*a)), m=Math.pow(10, (c||0)/20), p=o/2, u=1-1/r; let d=-96; for (let t=0; t<e.length; t++) {
          const a=e[t], r=20*Math.log10(Math.abs(a)+1e-9), s=r>d?l:h; d=s*d+(1-s)*r; let i=0; const c=d-n; if (c<=-p) i=0; else if (c>=p) i=u*c; else {
            const e=c+p; i=u*e*e/(2*o)
          }
          const b=Math.pow(10, -i/20)*m; e[t]=a*b
        }
      }
      (z, D.comp, M), "number"==typeof D.targetLufs) {
        const e=u(z, M); if (e>-60) {
          const t=Math.max(-18, Math.min(18, D.targetLufs-e)), a=Math.pow(10, t/20); for (let e=0; e<z.length; e++) z[e]*=a
        }
      }
      d(z, D.limiterCeilingDb, M), y&&b(z, n); let q=0, E=0; if (x) {
        const e=f(z, M, r, o); z=e.pcm, q=e.trimmedStartMs, E=e.trimmedEndMs
      }
      const F=u(z, M), w=g(z); return {
        pcm:z, stats: {
          preset:D.id, sampleRate:M, inputLufs:Number(H.toFixed(2)), outputLufs:Number(F.toFixed(2)), peakDb:Number(w.toFixed(2)), trimmedStartMs:q, trimmedEndMs:E, comfortNoise:y, edgeTrim:x
        }
      }
    }, createLiveChain:function(e, t="podcast", r= {

    }) {
      const o=e.createGain(), l=e.createGain(), h=e.createGain(), m=e.createGain(), p=e.createBiquadFilter(); p.type="highpass", p.Q.value=s; const u=e.createBiquadFilter(); u.type="highpass", u.Q.value=i; const d=e.createBiquadFilter(); d.type="peaking"; const f=e.createBiquadFilter(); f.type="peaking"; const g=e.createBiquadFilter(); g.type="highshelf"; const M=e.createDynamicsCompressor(), T=e.createWaveShaper(); T.curve=function(e=a, t=4096) {
        const n=new Float32Array(t); for (let a=0; a<t; a++) {
          const r=2*a/(t-1)-1; n[a]=e*Math.tanh(r/e)
        }
        return n
      }
      (a), T.oversample="2x"; const D=e.createGain(), y=e.createBufferSource(); function x(t, a= {

      }) {
        const n=c[t]||c.podcast, r=e.currentTime, o=.02, s="number"==typeof a.dryWet?a.dryWet:n.dryWet, i="boolean"==typeof a.comfortNoise?a.comfortNoise:n.comfortNoise; if (n.bypassAll) return m.gain.setTargetAtTime(0, r, o), h.gain.setTargetAtTime(1, r, o), void D.gain.setTargetAtTime(0, r, o); m.gain.setTargetAtTime(s, r, o), h.gain.setTargetAtTime(1-s, r, o), p.frequency.setTargetAtTime(n.hpfHz||80, r, o), u.frequency.setTargetAtTime(n.hpfHz||80, r, o), n.eq&&(d.frequency.setTargetAtTime(n.eq.boxinessHz, r, o), d.Q.setTargetAtTime(n.eq.boxinessQ, r, o), d.gain.setTargetAtTime(n.eq.boxinessGainDb, r, o), f.frequency.setTargetAtTime(n.eq.presenceHz, r, o), f.Q.setTargetAtTime(n.eq.presenceQ, r, o), f.gain.setTargetAtTime(n.eq.presenceGainDb, r, o), g.frequency.setTargetAtTime(n.eq.airHz, r, o), g.gain.setTargetAtTime(n.eq.airGainDb, r, o)), n.comp&&(M.threshold.setTargetAtTime(n.comp.thresholdDb, r, o), M.ratio.setTargetAtTime(n.comp.ratio, r, o), M.knee.setTargetAtTime(n.comp.kneeDb, r, o), M.attack.setTargetAtTime(n.comp.attackMs/1e3, r, o), M.release.setTargetAtTime(n.comp.releaseMs/1e3, r, o)), D.gain.setTargetAtTime(i?1:0, r, o)
      }
      return y.buffer=function(e, t=4) {
        const a=Math.round(e.sampleRate*t), r=e.createBuffer(1, a, e.sampleRate); return b(r.getChannelData(0), n), r
      }
      (e, 4), y.loop=true, y.connect(D), D.connect(l), y.start(), o.connect(p), p.connect(u), u.connect(d), d.connect(f), f.connect(g), g.connect(M), M.connect(T), T.connect(m), m.connect(l), o.connect(h), h.connect(l), x(t, r),  {
        input:o, output:l, applyPreset:x, dispose() {
          try {
            y.stop()
          } catch (e) {

          }
          o.disconnect(), l.disconnect(), D.disconnect()
        }
      }
    }, measureKWeightedLUFS:u, smartEdgeSilenceTrim:f, computePeakDb:g
  }
}));
