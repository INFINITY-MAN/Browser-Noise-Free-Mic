!function(t, e) {
  const n=function(t) {
    "use strict";
    const e=48e3, n=1024;
    function o(t, e, n=false) {
      const o=t.length;
      for (let n=1, a=0; n<o; n++) {
        let r=o>>1;
        for (; a&r; r>>=1) a^=r;
        if (a^=r, n<a) {
          const o=t[n];
          t[n]=t[a], t[a]=o;
          const r=e[n];
          e[n]=e[a], e[a]=r
        }
      }
      const a=n?1:-1;
      for (let n=2; n<=o; n<<=1) {
        const r=2*a*Math.PI/n, i=Math.cos(r), s=Math.sin(r), c=n>>1;
        for (let a=0; a<o; a+=n) {
          let n=1, o=0;
          for (let r=0; r<c; r++) {
            const l=t[a+r], f=e[a+r], u=t[a+r+c]*n-e[a+r+c]*o, h=t[a+r+c]*o+e[a+r+c]*n;
            t[a+r]=l+u, e[a+r]=f+h, t[a+r+c]=l-u, e[a+r+c]=f-h;
            const d=n*i-o*s;
            o=n*s+o*i, n=d
          }
        }
      }
      if (n) for (let n=0; n<o; n++) t[n]/=o, e[n]/=o
    }
    function a(t) {
      const e=new Float32Array(t);
      for (let n=0; n<t; n++) e[n]=.5*(1-Math.cos(2*Math.PI*n/(t-1)));
      return e
    }
    function r(t, n=e) {
      const o=t.length, a=2*o, r=new ArrayBuffer(44+a), i=new DataView(r);
      function s(t, e) {
        for (let n=0; n<e.length; n++) i.setUint8(t+n, e.charCodeAt(n))
      }
      s(0, "RIFF"), i.setUint32(4, 36+a, true), s(8, "WAVE"), s(12, "fmt "), i.setUint32(16, 16, true), i.setUint16(20, 1, true), i.setUint16(22, 1, true), i.setUint32(24, n, true), i.setUint32(28, 2*n, true), i.setUint16(32, 2, true), i.setUint16(34, 16, true), s(36, "data"), i.setUint32(40, a, true);
      let c=44;
      for (let e=0; e<o; e++) {
        const n=Math.max(-1, Math.min(1, t[e]));
        i.setInt16(c, n<0?32768*n:32767*n, true), c+=2
      }
      return r
    }
    function i(t, n=e) {
      const o=r(t, n);
      return new Blob([o],  {
        type:"audio/wav"
      })
    }
    function s(t) {
      const n=new DataView(t);
      if (t.byteLength<44) throw new Error("Invalid WAV buffer: too short.");
      let o=12, a=1, r=1, i=e, s=16, c=-1, l=0;
      for (; o+8<=t.byteLength; ) {
        const e=String.fromCharCode(n.getUint8(o), n.getUint8(o+1), n.getUint8(o+2), n.getUint8(o+3)), f=n.getUint32(o+4, true);
        if ("fmt "===e) a=n.getUint16(o+8, true), r=n.getUint16(o+10, true), i=n.getUint32(o+12, true), s=n.getUint16(o+22, true);
        else if ("data"===e) {
          c=o+8, l=Math.min(f, t.byteLength-c);
          break
        }
        o+=8+f+f%2
      }
      if (c<0) throw new Error("WAV data chunk not found.");
      const f=s/8, u=Math.floor(l/(f*r)), h=new Float32Array(u);
      let d=c;
      for (let t=0; t<u; t++) {
        let e=0;
        for (let t=0; t<r; t++) {
          if (3===a&&32===s) e+=n.getFloat32(d, true);
          else if (16===s) e+=n.getInt16(d, true)/32768;
          else if (24===s) {
            const t=n.getUint8(d), o=n.getUint8(d+1);
            e+=(n.getInt8(d+2)<<16|o<<8|t)/8388608
          } else 32===s&&(e+=n.getInt32(d, true)/2147483648);
          d+=f
        }
        h[t]=e/r
      }
      return {
        pcm:h, sampleRate:i
      }
    }
    function c(t, e, n) {
      if (e===n) return new Float32Array(t);
      const o=e/n, a=Math.max(1, Math.round(t.length/o)), r=new Float32Array(a);
      for (let e=0; e<a; e++) {
        const n=e*o, a=Math.floor(n), i=Math.min(a+1, t.length-1), s=n-a;
        r[e]=t[a]*(1-s)+t[i]*s
      }
      return r
    }
    async function l(t) {
      const n=t.numberOfChannels, o=t.length, a=new Float32Array(o);
      for (let e=0; e<n; e++) {
        const r=t.getChannelData(e);
        for (let t=0; t<o; t++) a[t]+=r[t]/n
      }
      if (t.sampleRate===e) return a;
      if ("undefined"!=typeof OfflineAudioContext) {
        const n=Math.ceil(o*e/t.sampleRate), r=new OfflineAudioContext(1, n, e), i=r.createBuffer(1, o, t.sampleRate);
        i.copyToChannel(a, 0);
        const s=r.createBufferSource();
        s.buffer=i, s.connect(r.destination), s.start(0);
        const c=await r.startRendering();
        return new Float32Array(c.getChannelData(0))
      }
      return c(a, t.sampleRate, e)
    }
    function f(t) {
      const n=[];
      let o=0;
      const a=[96e3, 88200, 64e3, 48e3, 44100, 32e3, 24e3, 22050, 16e3, 12e3, 11025, 8e3, 7350];
      for (; o+7<=t.length; ) {
        if (255!==t[o]||240!=(240&t[o+1])) {
          o++;
          continue
        }
        const r=t[o+2]>>2&15, i=(1&t[o+2])<<2|t[o+3]>>6&3, s=(3&t[o+3])<<11|t[o+4]<<3|t[o+5]>>5&7;
        if (s<7||o+s>t.length) break;
        n.push({
          data:t.subarray(o, o+s), sampleRate:a[r]||e, channels:i||1
        }), o+=s
      }
      return n
    }
    async function u(t) {
      if ("undefined"==typeof AudioDecoder||"undefined"==typeof EncodedAudioChunk) throw new Error("WebCodecs AudioDecoder unavailable");
      const n=f(new Uint8Array(t));
      if (0===n.length) throw new Error("Input is not a raw ADTS AAC stream");
      const o=n[0], a= {
        codec:"mp4a.40.2", sampleRate:o.sampleRate, numberOfChannels:o.channels
      };
      if (!(await AudioDecoder.isConfigSupported(a)).supported) throw new Error("WebCodecs mp4a.40.2 not supported");
      const r=[];
      let i=o.sampleRate;
      const s=new AudioDecoder({
        output:t=> {
          i=t.sampleRate; const e=t.numberOfFrames, n=t.numberOfChannels, o=new Float32Array(e), a=new Float32Array(e); for (let r=0; r<n; r++) {
            t.copyTo(a,  {
              planeIndex:r, format:"f32-planar"
            }); for (let t=0; t<e; t++) o[t]+=a[t]/n
          }
          r.push(o), t.close()
        }, error:t=> {
          throw t
        }
      });
      s.configure(a);
      const l=Math.round(1024/o.sampleRate*1e6);
      for (let t=0; t<n.length; t++) s.decode(new EncodedAudioChunk({
        type:"key", timestamp:t*l, duration:l, data:n[t].data
      }));
      await s.flush(), s.close();
      const u=r.reduce(((t, e)=>t+e.length), 0), h=new Float32Array(u);
      let d=0;
      for (const t of r) h.set(t, d), d+=t.length;
      return {
        pcm:c(h, i, e), sampleRate:e, decoderUsed:"webcodecs-aac"
      }
    }
    async function h(n, o="audio.wav") {
      let a;
      if (n instanceof ArrayBuffer) a=n;
      else if (ArrayBuffer.isView(n)) a=n.buffer.slice(n.byteOffset, n.byteOffset+n.byteLength);
      else {
        if (!n||"function"!=typeof n.arrayBuffer) throw new Error("Unsupported audio input type.");
        a=await n.arrayBuffer(), n.name&&(o=n.name)
      }
      if (/\.wav$/i.test(o)) try {
        const t=s(a), n=t.sampleRate===e?t.pcm:c(t.pcm, t.sampleRate, e);
        return {
          pcm:n, sampleRate:e, durationSec:n.length/e, decoder:"native-wav"
        }
      } catch {

      }
      const r=t.AudioContext||t.webkitAudioContext;
      if (r) {
        const t=new r;
        try {
          const n=await t.decodeAudioData(a.slice(0)), o=await l(n);
          return {
            pcm:o, sampleRate:e, durationSec:o.length/e, decoder:"web-audio"
          }
        } catch {

        } finally {
          t.close().catch ((()=> {

          }))
        }
      }
      try {
        return {
          ...await u(a), decoder:"webcodecs-aac"
        }
      } catch (t) {
        throw new Error(`Could not decode "${o}" in browser: ${t&&t.message?t.message:t}`)
      }
    }
    async function d(t, n= {

    }) {
      const o=n.sampleRate||e, a=n.bitrate||16e4;
      const encRes = await p(t, o, a);
      return {
        blob: encRes instanceof Blob ? encRes : encRes.blob,
        mimeType: "audio/aac",
        extension: "aac",
        encoder: "webcodecs-aac"
      }
    }
    async function p(t, n=e, o=16e4) {
      if ("undefined"==typeof AudioEncoder||"undefined"==typeof AudioData) throw new Error("WebCodecs AudioEncoder unavailable");
      const a= {
        codec:"mp4a.40.2", sampleRate:n, numberOfChannels:1, bitrate:o, aac: {
          format:"adts"
        }
      };
      if (!(await AudioEncoder.isConfigSupported(a)).supported) throw new Error("WebCodecs AAC-LC encoding not supported");
      const r=[], i=new AudioEncoder({
        output:t=> {
          const e=new Uint8Array(t.byteLength); t.copyTo(e), r.push(e)
        }, error:t=> {
          throw t
        }
      });
      i.configure(a);
      const s=1024;
      for (let e=0; e<t.length; e+=s) {
        const o=Math.min(s, t.length-e), a=new Float32Array(s);
        a.set(t.subarray(e, e+o));
        const r=new AudioData({
          format:"f32-planar", sampleRate:n, numberOfFrames:s, numberOfChannels:1, timestamp:Math.round(e/n*1e6), data:a
        });
        i.encode(r), r.close()
      }
      return await i.flush(), i.close(),  {
        blob:new Blob(r,  {
          type:"audio/aac"
        }), extension:".aac", encoderUsed:"webcodecs-aac"
      }
    }
    function m(t) {
      const e=Math.max(0, Math.min(1, t));
      if (e<.62) {
        const t=e/.62;
        return [Math.round(255*Math.pow(t, .85)), Math.round(90*Math.pow(t, 1.35)), 0]
      }
      const n=(e-.62)/.38;
      return [255, Math.round(90+165*n), Math.round(255*Math.pow(n, 1.15))]
    }
    function M(t, r, i= {

    }) {
      const s="string"==typeof t?"undefined"!=typeof document?document.querySelector(t):null:t;
      if (!s||!r||0===r.length) return false;
      const c=s.getContext("2d");
      if (!c) return false;
      const l=i.sampleRate||e, f=i.maxFreqHz||2e4, u=i.fftSize||n, h=i.minDb??-90, d=i.maxDb??-6, p=s.width||640, M=s.height||200, w=c.createImageData(p, M), y=w.data, g=a(u), A=new Float32Array(u), b=new Float32Array(u), x=l/2, C=Math.min(u>>1, Math.floor(f/x*(u>>1))), F=Math.max(1, r.length-u);
      for (let t=0; t<p; t++) {
        const e=Math.floor(t/Math.max(1, p-1)*F);
        A.fill(0), b.fill(0);
        for (let t=0; t<u; t++) {
          const n=e+t;
          A[t]=(n<r.length?r[n]:0)*g[t]
        }
        o(A, b, false);
        for (let e=0; e<M; e++) {
          const n=1-e/Math.max(1, M-1), o=Math.min(C, Math.max(0, Math.round(n*C))), a=Math.hypot(A[o], b[o])/(.5*u), r=(20*Math.log10(a+1e-9)-h)/(d-h), [i, s, c]=m(r), l=4*(e*p+t);
          y[l]=i, y[l+1]=s, y[l+2]=c, y[l+3]=255
        }
      }
      if (c.putImageData(w, 0, 0), false!==i.showGrid) {
        c.save(), c.strokeStyle="rgba(255, 255, 255, 0.14)", c.fillStyle="rgba(255, 255, 255, 0.72)", c.font="10px ui-monospace, SFMono-Regular, Menlo, monospace";
        const t=[5e3, 1e4, 15e3, 2e4];
        for (const e of t) {
          if (e>f) continue;
          const t=Math.round((1-e/f)*(M-1));
          c.beginPath(), c.moveTo(0, t+.5), c.lineTo(p, t+.5), c.stroke(), c.fillText(e/1e3+" kHz", 6, Math.max(11, t-3))
        }
        c.restore()
      }
      return true
    }
    function w(t, e, n="#canvas-noisy", o="#canvas-denoised") {
      const a=M(n, t), r=M(o, e);
      return a&&r
    }
    function y(t=3, n=e) {
      const o=Math.round(t*n), a=new Float32Array(o), r=new Float32Array(o);
      let i=0, s=0, c=0, l=0, f=0, u=0, h=0, d=480001337;
      function p() {
        return d=1664525*d+1013904223>>>0, d/4294967295*2-1
      }
      let m=0;
      for (let t=0; t<o; t++) {
        const e=t/n, o=Math.exp(-Math.pow((e-.55)/.22, 2))+Math.exp(-Math.pow((e-1.25)/.26, 2))+Math.exp(-Math.pow((e-2)/.24, 2))+Math.exp(-Math.pow((e-2.58)/.18, 2)), d=142+24*Math.sin(2*Math.PI*.7*e);
        m+=2*Math.PI*d/n;
        const M=.55*Math.sin(m)+.42*Math.sin(2*Math.PI*(620+90*Math.sin(3*e))*e)+.3*Math.sin(2*Math.PI*(1850+180*Math.cos(2.4*e))*e)+.16*Math.sin(2*Math.PI*2950*e), w=.22*(Math.exp(-Math.pow((e-.34)/.045, 2))+Math.exp(-Math.pow((e-1.52)/.045, 2))+Math.exp(-Math.pow((e-2.3)/.04, 2)))*(Math.sin(2*Math.PI*6800*e+1.4*p())+.7*Math.sin(2*Math.PI*9600*e+1.8*p())), y=Math.max(-.9, Math.min(.9, M*o*.42+w));
        a[t]=y;
        const g=.11*(1*Math.sin(2*Math.PI*50*e)+.5*Math.sin(2*Math.PI*100*e+.4)+.25*Math.sin(2*Math.PI*150*e+1.1)), A=p();
        i=.99886*i+.0555179*A, s=.99332*s+.0750759*A, c=.969*c+.153852*A, l=.8665*l+.3104856*A, f=.55*f+.5329522*A, u=-.7616*u-.016898*A;
        const b=.045*(i+s+c+l+f+u+h+.5362*A);
        h=.115926*A;
        const x=b*(1+.14*Math.sin(2*Math.PI*3.8*e));
        r[t]=Math.max(-1, Math.min(1, y+g+x))
      }
      return {
        noisyPCM:r, cleanPCM:a, sampleRate:n, durationSec:t, fileName:"bnfm_indian_household_demo_48k.wav"
      }
    }
    function g(t, e=n) {
      const r=e>>1, i=1+(e>>1), s=a(e), c=new Float32Array(e), l=new Float32Array(e), f=new Float32Array(i);
      let u=0;
      for (let n=0; n+e<=t.length; n+=r) {
        c.fill(0), l.fill(0);
        for (let o=0; o<e; o++) c[o]=t[n+o]*s[o];
        o(c, l, false);
        for (let t=0; t<i; t++) f[t]+=Math.hypot(c[t], l[t]);
        u++
      }
      if (u>0) for (let t=0; t<i; t++) f[t]/=u;
      let h=0;
      for (let e=0; e<t.length; e++) h+=t[e]*t[e];
      const d=Math.sqrt(h/Math.max(1, t.length));
      return {
        spectrum:f, noiseFloorDb:Number((20*Math.log10(d+1e-9)).toFixed(2)), fftSize:e
      }
    }
    async function A(o, a=null, r=2) {
      const i=new("undefined"!=typeof AudioContext?AudioContext:webkitAudioContext)({
        sampleRate:e
      }), s=i.createMediaStreamSource(o), l=Math.round(r*i.sampleRate), f=new Float32Array(l);
      let u=0;
      return new Promise(((o, h)=> {
        const d=i.createScriptProcessor(2048, 1, 1), p=i.createGain(); p.gain.value=0, d.onaudioprocess=async m=> {
          const M=m.inputBuffer.getChannelData(0), w=l-u, y=Math.min(w, M.length); if (f.set(M.subarray(0, y), u), u+=y, "function"==typeof a&&a(Math.min(1, u/l)), u>=l) {
            d.onaudioprocess=null, s.disconnect(), d.disconnect(), p.disconnect(); const a=i.sampleRate; try {
              await i.close()
            } catch (t) {

            }
            const l=a===e?f:c(f, a, e),  {
              spectrum:u, noiseFloorDb:m, fftSize:M
            }
            =g(l, n), w= {
              sampleRate:e, fftSize:M, noiseFloorDb:m, durationSec:r, spectrum:u, capturedAt:(new Date).toISOString()
            }; if (t.BNFMStore&&"function"==typeof t.BNFMStore.saveRoomProfile) try {
              await t.BNFMStore.saveRoomProfile(w)
            } catch (t) {
              return void h(t)
            }
            o(w)
          }
        }, s.connect(d), d.connect(p), p.connect(i.destination)
      }))
    }
    return {
      TARGET_SAMPLE_RATE:e, FFT_SIZE:n, fftRadix2:o, buildHannWindow:a, encodeWavBuffer48k:r, encodeWav48k:i, decodeWavBuffer:s, resampleLinear:c, resampleTo48kMono:l, decodeAudioFile:h, exportAAC:d, renderSpectrogram:M, renderComparisonSpectrograms:w, generateIndianHouseholdDemo:y, computeWelchSpectrum:g, calibrateRoomSnapshot:A
    }
  }
  (t);
  "object"==typeof module&&module.exports?module.exports=n:t.BNFMCodec=n
}
("undefined"!=typeof self?self:this);
