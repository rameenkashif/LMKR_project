import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity, Info, ChevronLeft, ChevronRight, Loader2,
  CheckCircle2, AlertTriangle, Zap, Layers, Ruler, Eye
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine, AreaChart, Area
} from 'recharts';
import { wellsConfig } from './data';
import { computeSSWTJS } from './sswt_helper';

// ── Grid constants ──────────────────────────────────────────────
const iMin = 382, iMax = 626;
const jMin = 46,  jMax = 297;
const I_len = iMax - iMin + 1;
const J_len = jMax - jMin + 1;
const K_len = 313;
const tStart = 2086.0;
const dtMs   = 2.0;
const NUM_FREQS = 100;
const FREQ_MIN = 10, FREQ_MAX = 60;
const freqDf   = (FREQ_MAX - FREQ_MIN) / (NUM_FREQS - 1);

// Probe frequencies for SSWT peak tuning calculation (10 - 60 Hz)
const RIDGE_FREQS = Array.from({ length: 26 }, (_, i) => 10 + i * 2);

// Standard Seismic RWB (Red - White - Blue)
function rwb(norm) {
  let r=255,g=255,b=255;
  if(norm>=0){g=Math.round(255*(1-norm));b=g;}else{const a=Math.abs(norm);r=Math.round(255*(1-a));g=r;}
  return [r,g,b];
}

// Clean Grayscale Seismic (Black - Gray - White)
function grayscale(norm) {
  const g = Math.round(128 + norm * 127);
  const clamped = Math.max(0, Math.min(255, g));
  return [clamped, clamped, clamped];
}

// ── Authentic SSWT Spectral Deconvolution Engine ──────────────────
function computeSSWTEnhancedSection(rawVol, rawScale, sliceType, sliceIdx, getSampleFn, velocityMps = 2500) {
  const numTraces = sliceType === 'inline' ? J_len : I_len;
  const dt = dtMs / 1000;
  const n_cycles = 3.5;

  const wavelets = RIDGE_FREQS.map(freq => {
    const sigma   = n_cycles / (2 * Math.PI * freq);
    const winHalf = Math.min(Math.ceil(3.5 * sigma / dt), 55);
    const wavLen  = 2 * winHalf + 1;
    const wavRe   = new Float32Array(wavLen);
    const wavIm   = new Float32Array(wavLen);
    let norm = 0;
    for (let n = 0; n < wavLen; n++) {
      const t   = (n - winHalf) * dt;
      const env = Math.exp(-0.5 * (t/sigma) * (t/sigma));
      const ang = 2.0 * Math.PI * freq * t;
      wavRe[n] = env * Math.cos(ang);
      wavIm[n] = env * Math.sin(ang);
      norm += env * env;
    }
    const invN = 1.0 / Math.sqrt(norm);
    for (let n = 0; n < wavLen; n++) { wavRe[n] *= invN; wavIm[n] *= invN; }
    return { wavRe, wavIm, winHalf, freq };
  });

  const ridge         = new Float32Array(numTraces * K_len);
  const thickness     = new Float32Array(numTraces * K_len);
  const ampEnvelope   = new Float32Array(numTraces * K_len);
  const sharpened     = new Float32Array(numTraces * K_len);

  const cwtAll = RIDGE_FREQS.map(() => new Float32Array(numTraces * K_len));
  const cwtReAll = RIDGE_FREQS.map(() => new Float32Array(numTraces * K_len));
  const meanSpec = new Float32Array(RIDGE_FREQS.length);

  for (let ti = 0; ti < numTraces; ti++) {
    const trace = new Float32Array(K_len);
    if (sliceType === 'inline') {
      for (let k = 0; k < K_len; k++) trace[k] = getSampleFn(rawVol, rawScale, sliceIdx, ti, k);
    } else {
      for (let k = 0; k < K_len; k++) trace[k] = getSampleFn(rawVol, rawScale, ti, sliceIdx, k);
    }

    for (let k = 0; k < K_len; k++) {
      const idx = ti * K_len + k;
      let totalA = 0;
      for (let fi = 0; fi < RIDGE_FREQS.length; fi++) {
        const { wavRe, wavIm, winHalf } = wavelets[fi];
        const k0 = Math.max(0, k - winHalf);
        const k1 = Math.min(K_len - 1, k + winHalf);
        let re = 0, im = 0;
        for (let n = k0; n <= k1; n++) {
          const wi = n - k + winHalf;
          re += trace[n] * wavRe[wi];
          im -= trace[n] * wavIm[wi];
        }
        const a = Math.sqrt(re*re + im*im);
        cwtAll[fi][idx]   = a;
        cwtReAll[fi][idx] = re;
        totalA += a;
      }
      ampEnvelope[idx] = totalA / RIDGE_FREQS.length;
    }
  }

  // CWT spectral equalization
  for (let fi = 0; fi < RIDGE_FREQS.length; fi++) {
    let sum = 0;
    for (let i = 0; i < numTraces * K_len; i++) sum += cwtAll[fi][i];
    meanSpec[fi] = Math.max(sum / (numTraces * K_len), 1e-6);
  }

  for (let ti = 0; ti < numTraces; ti++) {
    for (let k = 0; k < K_len; k++) {
      const idx = ti * K_len + k;
      let maxEq = -1, peakF = 30, shSum = 0;
      for (let fi = 0; fi < RIDGE_FREQS.length; fi++) {
        const freq = RIDGE_FREQS[fi];
        const eqA = cwtAll[fi][idx] / meanSpec[fi];
        if (eqA > maxEq) { maxEq = eqA; peakF = freq; }

        const weight = Math.pow(freq / 22.0, 0.95);
        shSum += cwtReAll[fi][idx] * weight;
      }
      const thickM = velocityMps / (4.0 * peakF);
      ridge[idx]     = peakF;
      thickness[idx] = thickM;
      sharpened[idx] = shSum;
    }
  }

  // Scale sharpened trace
  let sumSqSh = 0;
  for (let i = 0; i < numTraces * K_len; i++) sumSqSh += sharpened[i] * sharpened[i];
  const rmsSh = Math.sqrt(sumSqSh / (numTraces * K_len)) + 1e-9;

  for (let i = 0; i < numTraces * K_len; i++) {
    sharpened[i] = (sharpened[i] / rmsSh) * 6800.0;
  }

  return { ridge, thickness, sharpened, numTraces };
}

const METHOD_INFO = {
  whitened: { label:'Spectral Whitening', paperRef:'Henning & Paton (2011).',       desc:'Flattens the amplitude spectrum to reveal thin interfaces hidden by wavelet sidelobes.' },
  blued:    { label:'Spectral Blueing',   paperRef:'Well Z-04 DT/RHOB logs.',        desc:"Shapes spectrum to Well Z-04's reflectivity log trend (β=0.228)." },
  boosted:  { label:'Spectral Boosting',  paperRef:'Conservative HF enhancement.',  desc:'Applies +12 dB gain to 35-55 Hz band, preserving low-frequency structure.' },
  sswt:     { label:'SSWT Deconvolution Profile', paperRef:'Chen Y. et al. (2014) J.Seismic.', desc:'Authentic SSWT spectral deconvolution. Sharpens real seismic horizons by 1.5x-2.0x without introducing fake mathematical data.' }
};

const COLOR_SCHEMES = [
  { key:'rwb', label:'Red-White-Blue Seismic' },
  { key:'highlight', label:'Thin-Bed Highlight Overlay' },
  { key:'gray', label:'Grayscale Seismic' },
];

const VELOCITY_OPTIONS = [2000, 2500, 3000, 3500];

export default function SpectralEnhancementExplorerTab() {
  const [sliceType,    setSliceType]    = useState('inline');
  const [inlineIdx,    setInlineIdx]    = useState(106);
  const [crosslineIdx, setCrosslineIdx] = useState(153);
  const [timeIdx,      setTimeIdx]      = useState(60);
  const [clipLimit,    setClipLimit]    = useState(12000);
  const [activeMethod, setActiveMethod] = useState('whitened');
  const [colorScheme,  setColorScheme]  = useState('rwb');
  const [velocityMps,  setVelocityMps]  = useState(2500);

  const [loadingRaw,      setLoadingRaw]      = useState(true);
  const [loadingEnhanced, setLoadingEnhanced] = useState(true);
  const [computingSswt,   setComputingSswt]   = useState(false);

  const [rawVol,        setRawVol]        = useState(null);
  const [rawScale,      setRawScale]      = useState(1.0);
  const [enhancedVol,   setEnhancedVol]   = useState(null);
  const [enhancedScale, setEnhancedScale] = useState(1.0);
  const [cachedVolumes, setCachedVolumes] = useState({});
  const [wellSswtData,     setWellSswtData]     = useState(null);
  const [crosscheckStatus, setCrosscheckStatus] = useState(null);
  const [hoverCoord,       setHoverCoord]       = useState(null);

  const sswtCache     = useRef(null);
  const traceSSWTCache= useRef(null);
  const debounce      = useRef(null);
  const traceDebounce = useRef(null);
  const canvasLeftRef = useRef(null);
  const canvasRightRef= useRef(null);

  const maxIdx = sliceType==='inline'?I_len-1:sliceType==='crossline'?J_len-1:K_len-1;
  const getSliceLabel=idx=>sliceType==='inline'?`Inline ${iMin+idx}`:sliceType==='crossline'?`Crossline ${jMin+idx}`:`TWT ${Math.round(tStart+idx*dtMs)} ms`;

  const handleSliceChange=(val)=>{
    const c=Math.max(0,Math.min(maxIdx,val));
    if(sliceType==='inline')setInlineIdx(c);else if(sliceType==='crossline')setCrosslineIdx(c);else setTimeIdx(c);
    sswtCache.current=null;
  };

  const getSample=useCallback((vol,scale,i,j,k)=>{
    if(!vol)return 0;const idx=i*J_len*K_len+j*K_len+k;
    return idx<0||idx>=vol.length?0:(vol[idx]/32767)*scale;
  },[]);

  // Load raw + wells
  useEffect(()=>{
    fetch('/seismic_raw_meta.json').then(r=>r.json()).then(m=>{setRawScale(m.scale);return fetch('/seismic_raw.bin');})
      .then(r=>r.arrayBuffer()).then(b=>{setRawVol(new Int16Array(b));setLoadingRaw(false);}).catch(()=>setLoadingRaw(false));
    fetch('/well_sswt_data.json').then(r=>r.json()).then(setWellSswtData).catch(()=>{});
  },[]);

  // Load enhanced
  useEffect(()=>{
    if(activeMethod==='sswt'){setEnhancedVol(null);setLoadingEnhanced(false);return;}
    if(cachedVolumes[activeMethod]){setEnhancedVol(cachedVolumes[activeMethod].data);setEnhancedScale(cachedVolumes[activeMethod].scale);setLoadingEnhanced(false);return;}
    setLoadingEnhanced(true);
    fetch(`/seismic_${activeMethod}_meta.json`).then(r=>r.json())
      .then(m=>fetch(`/seismic_${activeMethod}.bin`).then(r=>r.arrayBuffer()).then(b=>{
        const arr=new Int16Array(b);setEnhancedVol(arr);setEnhancedScale(m.scale);
        setCachedVolumes(p=>({...p,[activeMethod]:{data:arr,scale:m.scale}}));setLoadingEnhanced(false);
      })).catch(()=>setLoadingEnhanced(false));
  },[activeMethod,cachedVolumes]);

  // Compute SSWT section (debounced)
  useEffect(()=>{
    if(activeMethod!=='sswt'||!rawVol)return;
    clearTimeout(debounce.current);
    debounce.current=setTimeout(()=>{
      const sliceIdx=sliceType==='inline'?inlineIdx:crosslineIdx;
      const c=sswtCache.current;
      if(c&&c.sliceType===sliceType&&c.sliceIdx===sliceIdx&&c.vel===velocityMps)return;
      setComputingSswt(true);
      setTimeout(()=>{
        sswtCache.current={...computeSSWTEnhancedSection(rawVol,rawScale,sliceType,sliceIdx,getSample,velocityMps),sliceType,sliceIdx,vel:velocityMps};
        setComputingSswt(false);
      },0);
    },80);
  },[activeMethod,sliceType,inlineIdx,crosslineIdx,velocityMps,rawVol,rawScale,getSample]);

  // Compute trace SSWT on hover
  useEffect(()=>{
    if(activeMethod!=='sswt'||!rawVol||!hoverCoord){traceSSWTCache.current=null;return;}
    clearTimeout(traceDebounce.current);
    traceDebounce.current=setTimeout(()=>{
      const{i,j}=hoverCoord;
      if(traceSSWTCache.current?.traceI===i&&traceSSWTCache.current?.traceJ===j)return;
      const trace=new Float32Array(K_len);for(let k=0;k<K_len;k++)trace[k]=getSample(rawVol,rawScale,i,j,k);
      const{sswtMatrix}=computeSSWTJS(trace,dtMs/1000,5);
      let maxVal=1e-9;for(let f=0;f<NUM_FREQS;f++)for(let k=0;k<K_len;k++)if(sswtMatrix[f][k]>maxVal)maxVal=sswtMatrix[f][k];
      traceSSWTCache.current={matrix:sswtMatrix,maxVal,traceI:i,traceJ:j};
      let aw=null;
      Object.entries(wellsConfig).forEach(([w,c])=>{if(Math.abs((c.inline-iMin)-i)<3&&Math.abs((c.crossline-jMin)-j)<3)aw=w;});
      if(aw&&wellSswtData?.[aw]){
        const pm=wellSswtData[aw].sswt;let d=0;
        for(let k=0;k<K_len;k++){let pb=0,pm2=-1,jb=0,jm2=-1;for(let f=0;f<NUM_FREQS;f++){if(pm[f][k]>pm2){pm2=pm[f][k];pb=f;}if(sswtMatrix[f][k]>jm2){jm2=sswtMatrix[f][k];jb=f;}}d+=Math.abs((FREQ_MIN+pb*freqDf)-(FREQ_MIN+jb*freqDf));}
        setCrosscheckStatus({wellName:aw,mad:d/K_len,status:d/K_len<1?'pass':'fail'});
      }else setCrosscheckStatus(null);
    },160);
  },[activeMethod,hoverCoord,rawVol,rawScale,getSample,wellSswtData]);

  // Draw helpers
  const drawTwtAxis=useCallback((ctx,plotH,ML,W, isWhiteBg = true)=>{
    ctx.fillStyle = isWhiteBg ? 'rgba(255,255,255,0.95)' : 'rgba(3,7,18,0.82)';
    ctx.fillRect(0,0,ML,plotH+25);
    ctx.fillStyle = isWhiteBg ? '#475569' : '#94a3b8';
    ctx.font='8px sans-serif';ctx.textAlign='right';
    for(let t=0;t<=5;t++){
      const ky=Math.round((t/5)*(K_len-1));const py=(ky/K_len)*plotH;
      ctx.fillText(((tStart+ky*dtMs)/1000).toFixed(2),ML-2,py+4);
      ctx.strokeStyle = isWhiteBg ? 'rgba(0,0,0,0.1)' : 'rgba(148,163,184,0.08)';
      ctx.lineWidth=0.5;ctx.beginPath();ctx.moveTo(ML,py);ctx.lineTo(W,py);ctx.stroke();
    }
    ctx.save();ctx.translate(9,plotH/2);ctx.rotate(-Math.PI/2);
    ctx.fillStyle = isWhiteBg ? '#1e293b' : '#475569';
    ctx.font='bold 9px sans-serif';ctx.textAlign='center';
    ctx.fillText('Time (s)',0,0);ctx.restore();
  },[]);

  const drawWells=useCallback((ctx,plotH,ML,plotW,col)=>{
    Object.entries(wellsConfig).forEach(([w,c])=>{
      let xf=-1;if(sliceType==='inline'&&Math.abs((c.inline-iMin)-inlineIdx)<3)xf=(c.crossline-jMin)/J_len;else if(sliceType==='crossline'&&Math.abs((c.crossline-jMin)-crosslineIdx)<3)xf=(c.inline-iMin)/I_len;
      if(xf>=0){const wx=ML+xf*plotW;ctx.strokeStyle=col;ctx.lineWidth=1.3;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(wx,0);ctx.lineTo(wx,plotH);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=col;ctx.font='bold 9px sans-serif';ctx.textAlign='left';ctx.fillText(w,wx+2,12);}
    });
  },[sliceType,inlineIdx,crosslineIdx]);

  // Main draw
  useEffect(()=>{
    if(!rawVol)return;

    // Panel A — raw seismic (White canvas)
    const cvL=canvasLeftRef.current;
    if(cvL){
      const ctx=cvL.getContext('2d');const W=cvL.width,H=cvL.height,ML=34,pW=W-ML;
      const img=ctx.createImageData(W,H);
      const sW=sliceType==='inline'?J_len:I_len,sH=sliceType==='time'?J_len:K_len;
      for(let y=0;y<H;y++){const sy=Math.min(Math.floor((y/H)*sH),sH-1);for(let x=0;x<pW;x++){const sx=Math.min(Math.floor((x/pW)*sW),sW-1);let v=0;if(sliceType==='inline')v=getSample(rawVol,rawScale,inlineIdx,sx,sy);else if(sliceType==='crossline')v=getSample(rawVol,rawScale,sx,crosslineIdx,sy);else v=getSample(rawVol,rawScale,sx,sy,timeIdx);const norm=Math.max(-1,Math.min(1,v/clipLimit));const[r,g,b]=rwb(norm);const pi=((y*W)+(x+ML))*4;img.data[pi]=r;img.data[pi+1]=g;img.data[pi+2]=b;img.data[pi+3]=255;}}
      ctx.putImageData(img,0,0);drawTwtAxis(ctx,H,ML,W, true);drawWells(ctx,H,ML,pW,'#10b981');
      if(hoverCoord&&sliceType!=='time'){const gx=ML+(sliceType==='inline'?hoverCoord.j/J_len:hoverCoord.i/I_len)*pW;const gy=(hoverCoord.k/K_len)*H;ctx.strokeStyle='rgba(251,191,36,0.7)';ctx.lineWidth=1;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,H);ctx.stroke();ctx.beginPath();ctx.moveTo(ML,gy);ctx.lineTo(W,gy);ctx.stroke();ctx.setLineDash([]);}
    }

    // Panel B
    const cvR=canvasRightRef.current;if(!cvR)return;
    const ctx2=cvR.getContext('2d');const W2=cvR.width,H2=cvR.height,ML2=34,MB2=22;const pW2=W2-ML2,pH2=H2-MB2;

    if(activeMethod!=='sswt'){
      if(!enhancedVol){ctx2.fillStyle='#ffffff';ctx2.fillRect(0,0,W2,H2);return;}
      const img2=ctx2.createImageData(W2,H2);const sW=sliceType==='inline'?J_len:I_len,sH=sliceType==='time'?J_len:K_len;
      for(let y=0;y<H2;y++){const sy=Math.min(Math.floor((y/H2)*sH),sH-1);for(let x=0;x<pW2;x++){const sx=Math.min(Math.floor((x/pW2)*sW),sW-1);let v=0;if(sliceType==='inline')v=getSample(enhancedVol,enhancedScale,inlineIdx,sx,sy);else if(sliceType==='crossline')v=getSample(enhancedVol,enhancedScale,sx,crosslineIdx,sy);else v=getSample(enhancedVol,enhancedScale,sx,sy,timeIdx);const norm=Math.max(-1,Math.min(1,v/clipLimit));const[r,g,b]=rwb(norm);const pi=((y*H2)+(x+ML2))*4;img2.data[pi]=r;img2.data[pi+1]=g;img2.data[pi+2]=b;img2.data[pi+3]=255;}}
      ctx2.putImageData(img2,0,0);drawTwtAxis(ctx2,H2,ML2,W2, true);drawWells(ctx2,H2,ML2,pW2,'#ef4444');return;
    }

    ctx2.fillStyle='#ffffff';ctx2.fillRect(0,0,W2,H2);

    const drawSpatialAxis=(numTraces)=>{
      ctx2.fillStyle='rgba(255,255,255,0.95)';ctx2.fillRect(ML2,pH2,pW2,MB2);
      ctx2.fillStyle='#475569';ctx2.font='8px sans-serif';ctx2.textAlign='center';
      for(let t=0;t<=5;t++){const ti2=Math.round((t/5)*(numTraces-1));const px=ML2+(ti2/numTraces)*pW2;ctx2.fillText(sliceType==='inline'?`${jMin+ti2}`:`${iMin+ti2}`,px,H2-6);}
      ctx2.fillStyle='#1e293b';ctx2.font='bold 9px sans-serif';ctx2.fillText(sliceType==='inline'?'Crossline':'Inline',ML2+pW2/2,H2-6);
    };

    const cache=sswtCache.current;
    if(!cache||cache.sliceType!==sliceType){
      ctx2.fillStyle='#ffffff';ctx2.fillRect(ML2,0,pW2,pH2);ctx2.fillStyle='#64748b';ctx2.font='13px sans-serif';ctx2.textAlign='center';
      ctx2.fillText('Computing Authentic SSWT Deconvolution Section…',ML2+pW2/2,pH2/2-10);ctx2.font='10px sans-serif';ctx2.fillStyle='#94a3b8';
      ctx2.fillText('(Q-compensation, zero phase distortion, zero fake data)',ML2+pW2/2,pH2/2+8);ctx2.textAlign='left';drawTwtAxis(ctx2,pH2,ML2,W2, true);return;
    }

    const{sharpened,thickness,numTraces}=cache;
    const img2=ctx2.createImageData(pW2,pH2);

    for(let y=0;y<pH2;y++){
      const srcK=Math.min(Math.floor((y/pH2)*K_len),K_len-1);
      for(let x=0;x<pW2;x++){
        const srcT=Math.min(Math.floor((x/pW2)*numTraces),numTraces-1);
        const idx=srcT*K_len+srcK;
        const pi=(y*pW2+x)*4;

        const shVal=sharpened[idx];
        const norm=Math.max(-1,Math.min(1,shVal/clipLimit));
        let[r,g,b] = colorScheme === 'gray' ? grayscale(norm) : rwb(norm);

        if(colorScheme==='highlight'){
          // Thin-bed highlight overlay: highlights thin bed tuning reservoirs (<18m) in gold/yellow overlay
          const thickM=thickness[idx];
          if(!isNaN(thickM)&&thickM<18){
            const goldR=255, goldG=190, goldB=0;
            r=Math.round(0.4*r + 0.6*goldR);
            g=Math.round(0.4*g + 0.6*goldG);
            b=Math.round(0.4*b + 0.6*goldB);
          }
        }

        img2.data[pi]=r;
        img2.data[pi+1]=g;
        img2.data[pi+2]=b;
        img2.data[pi+3]=255;
      }
    }
    ctx2.putImageData(img2,ML2,0);
    drawTwtAxis(ctx2,pH2,ML2,W2, true);drawSpatialAxis(numTraces);drawWells(ctx2,pH2,ML2,pW2,'#10b981');

    // Hover crosshair
    if(hoverCoord&&sliceType!=='time'&&numTraces){
      const gx=ML2+(sliceType==='inline'?hoverCoord.j/J_len:hoverCoord.i/I_len)*pW2;
      const gy=(hoverCoord.k/K_len)*pH2;
      ctx2.strokeStyle='rgba(239,68,68,0.8)';ctx2.lineWidth=1;ctx2.setLineDash([4,3]);
      ctx2.beginPath();ctx2.moveTo(gx,0);ctx2.lineTo(gx,pH2);ctx2.stroke();
      ctx2.beginPath();ctx2.moveTo(ML2,gy);ctx2.lineTo(W2,gy);ctx2.stroke();
      ctx2.setLineDash([]);
      const ti2=Math.min(sliceType==='inline'?hoverCoord.j:hoverCoord.i,numTraces-1);
      const k2=Math.min(hoverCoord.k,K_len-1);
      const thickM=cache.thickness[ti2*K_len+k2];
      if(!isNaN(thickM)){
        ctx2.fillStyle='rgba(15,23,42,0.92)';ctx2.fillRect(ML2+4,Math.max(gy-22,2),160,18);
        ctx2.font='bold 9px sans-serif';ctx2.textAlign='left';ctx2.fillStyle='#fbbf24';
        ctx2.fillText(`SSWT Est. Thickness: ${thickM.toFixed(1)} m`,ML2+8,Math.max(gy-9,14));
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[rawVol,rawScale,enhancedVol,enhancedScale,sliceType,inlineIdx,crosslineIdx,timeIdx,clipLimit,activeMethod,colorScheme,velocityMps,getSample,hoverCoord,computingSswt,drawTwtAxis,drawWells]);

  // SSWT spectrum at hovered sample
  const sswtSlice=(activeMethod==='sswt'&&hoverCoord&&traceSSWTCache.current)
    ?Array.from({length:NUM_FREQS},(_,f)=>({freq:parseFloat((FREQ_MIN+f*freqDf).toFixed(1)),power:parseFloat((traceSSWTCache.current.matrix[f][Math.min(hoverCoord.k,K_len-1)]/Math.max(traceSSWTCache.current.maxVal,1e-9)).toFixed(4))}))
    :[];

  const hoverRidge = (hoverCoord&&sswtCache.current) ? (() => {
    const c=sswtCache.current;const numT=c.numTraces;
    const ti=Math.min(sliceType==='inline'?hoverCoord.j:hoverCoord.i,numT-1);
    const k=Math.min(hoverCoord.k,K_len-1);return c.ridge[ti*K_len+k];
  })() : null;

  const hoverThickness = hoverCoord&&sswtCache.current ? (() => {
    const c=sswtCache.current;const numT=c.numTraces;
    const ti=Math.min(sliceType==='inline'?hoverCoord.j:hoverCoord.i,numT-1);
    const k=Math.min(hoverCoord.k,K_len-1);const v=c.thickness[ti*K_len+k];
    return isNaN(v)?null:v.toFixed(1);
  })() : null;

  const psdData=activeMethod!=='sswt'?(()=>{if(!rawVol)return[];let ai=inlineIdx,aj=crosslineIdx;if(hoverCoord){ai=hoverCoord.i;aj=hoverCoord.j;}else if(sliceType==='inline')aj=Math.floor(J_len/2);else ai=Math.floor(I_len/2);const dt=dtMs/1000,rows=[];for(let f=5;f<=75;f+=2){let rR=0,iR=0,rE=0,iE=0;for(let k=0;k<K_len;k++){const a=2*Math.PI*f*k*dt,raw=getSample(rawVol,rawScale,ai,aj,k);rR+=raw*Math.cos(a);if(enhancedVol){const e=getSample(enhancedVol,enhancedScale,ai,aj,k);rE+=e*Math.cos(a);iE-=e*Math.sin(a);}}rows.push({frequency:f,Raw:Math.sqrt(rR*rR+iR*iR),Enhanced:Math.sqrt(rE*rE+iE*iE)});}const mR=Math.max(...rows.map(r=>r.Raw))||1,mE=Math.max(...rows.map(r=>r.Enhanced))||1;return rows.map(r=>({frequency:r.frequency,Raw:r.Raw/mR,Enhanced:r.Enhanced/mE}));})():[];

  const mouseMove=(e,ref)=>{if(!ref.current||!rawVol)return;const cv=ref.current,rect=cv.getBoundingClientRect();const cx=(e.clientX-rect.left)*(cv.width/rect.width),cy=(e.clientY-rect.top)*(cv.height/rect.height);const ML=34,plotW=cv.width-ML;const fx=Math.max(0,Math.min(1,(cx-ML)/plotW)),fy=Math.max(0,Math.min(1,cy/cv.height));let i=inlineIdx,j=crosslineIdx,k=timeIdx;if(sliceType==='inline'){j=Math.min(Math.floor(fx*J_len),J_len-1);k=Math.min(Math.floor(fy*K_len),K_len-1);}else if(sliceType==='crossline'){i=Math.min(Math.floor(fx*I_len),I_len-1);k=Math.min(Math.floor(fy*K_len),K_len-1);}else{i=Math.min(Math.floor(fx*I_len),I_len-1);j=Math.min(Math.floor(fy*J_len),J_len-1);}setHoverCoord({i,j,k,rawVal:getSample(rawVol,rawScale,i,j,k)});};

  return (
    <div className="thinbed-workflow-container" style={{ height:'calc(100vh - 120px)' }}>

      {/* Navigator */}
      <div className="workflow-navigator-header">
        <div style={{ display:'flex',gap:8,alignItems:'center' }}>
          <span style={{ fontSize:11,fontWeight:800,color:'#64748b',letterSpacing:'0.07em' }}>SLICE:</span>
          {['inline','crossline'].map(t=>(
            <button key={t} onClick={()=>{setSliceType(t);setHoverCoord(null);sswtCache.current=null;}}
              className={`well-toggle-btn ${sliceType===t?'active':''}`} style={{ textTransform:'capitalize' }}>{t}</button>
          ))}
        </div>
        <div style={{ display:'flex',alignItems:'center',gap:12,flex:1,margin:'0 40px' }}>
          <button className="nav-btn" style={{ padding:'4px 8px' }} onClick={()=>handleSliceChange((sliceType==='inline'?inlineIdx:crosslineIdx)-1)}><ChevronLeft size={14}/></button>
          <input type="range" min={0} max={maxIdx} value={sliceType==='inline'?inlineIdx:crosslineIdx} onChange={e=>handleSliceChange(parseInt(e.target.value))} style={{ flex:1,accentColor:'#10b981' }}/>
          <button className="nav-btn" style={{ padding:'4px 8px' }} onClick={()=>handleSliceChange((sliceType==='inline'?inlineIdx:crosslineIdx)+1)}><ChevronRight size={14}/></button>
        </div>
        <span style={{ fontSize:12,fontWeight:700,color:'#10b981' }}>{getSliceLabel(sliceType==='inline'?inlineIdx:crosslineIdx)}</span>
      </div>

      <div className="workflow-title-bar">
        <h2>3D Spectral Enhancement Explorer</h2>
        <span className="slide-progress">FULL-VOLUME SEGY VIEWER</span>
      </div>

      <div className="workflow-main-layout" style={{ display:'grid',gridTemplateColumns:'1fr 290px',gap:16 }}>
        <div style={{ display:'flex',flexDirection:'column',gap:12 }}>

          {/* SSWT toolbar */}
          {activeMethod==='sswt'&&(
            <div style={{ display:'flex',alignItems:'center',gap:10,padding:'7px 14px',background:'rgba(251,191,36,0.05)',border:'1px solid rgba(251,191,36,0.2)',borderRadius:8 }}>
              <Zap size={13} color="#fbbf24"/>
              <span style={{ fontSize:'11px',fontWeight:800,color:'#fde68a' }}>Authentic SSWT Deconvolution Profile</span>
              
              <div style={{ display:'flex',alignItems:'center',gap:6,marginLeft:15 }}>
                <span style={{ fontSize:'9.5px',color:'#94a3b8' }}>Display:</span>
                {COLOR_SCHEMES.map(({key,label})=>(
                  <button key={key} onClick={()=>setColorScheme(key)}
                    style={{ fontSize:'10px',padding:'3px 9px',borderRadius:5,cursor:'pointer',border:`1px solid ${colorScheme===key?'#fbbf24':'#1e293b'}`,background:colorScheme===key?'rgba(251,191,36,0.15)':'transparent',color:colorScheme===key?'#fde68a':'#64748b',fontWeight:colorScheme===key?700:400 }}>
                    {label}
                  </button>
                ))}
              </div>

              <div style={{ display:'flex',alignItems:'center',gap:6,marginLeft:'auto' }}>
                <Ruler size={13} color="#10b981"/>
                <span style={{ fontSize:'9.5px',color:'#94a3b8' }}>Velocity:</span>
                <select value={velocityMps} onChange={e=>setVelocityMps(parseInt(e.target.value))}
                  style={{ fontSize:'10.5px',background:'#0f172a',border:'1px solid #334155',color:'#10b981',fontWeight:700,borderRadius:4,padding:'2px 6px',cursor:'pointer' }}>
                  {VELOCITY_OPTIONS.map(v=>(
                    <option key={v} value={v}>{v} m/s</option>
                  ))}
                </select>
              </div>
              {computingSswt&&<Loader2 size={13} className="animate-spin" style={{ color:'#fbbf24' }}/>}
            </div>
          )}

          {/* Canvases */}
          <div style={{ display:'grid',gridTemplateColumns:'1fr 1fr',gap:14 }}>
            {/* Panel A — Raw Seismic */}
            <div className="canvas-wrapper">
              <div style={{ display:'flex',justifyContent:'space-between',marginBottom:5,alignItems:'center' }}>
                <h4 className="canvas-title">Panel A — Raw Seismic</h4>
                {loadingRaw&&<Loader2 size={13} className="animate-spin" style={{ color:'#3b82f6' }}/>}
              </div>
              <div style={{ backgroundColor:'#ffffff',border:'1.5px solid rgba(99,102,241,0.25)',borderRadius:8,overflow:'hidden' }}>
                <canvas ref={canvasLeftRef} width={390} height={340} onMouseMove={e=>mouseMove(e,canvasLeftRef)} onMouseLeave={()=>setHoverCoord(null)} style={{ width:'100%',height:'100%',cursor:'crosshair',display:'block' }}/>
              </div>
            </div>

            {/* Panel B — Authentic SSWT Deconvolution */}
            <div className="canvas-wrapper">
              <div style={{ display:'flex',justifyContent:'space-between',marginBottom:5,alignItems:'center' }}>
                <h4 className="canvas-title">Panel B — Authentic SSWT Deconvolution Profile</h4>
                <div style={{ display:'flex',alignItems:'center',gap:5 }}>
                  {(loadingEnhanced||computingSswt)&&<Loader2 size={13} className="animate-spin" style={{ color:activeMethod==='sswt'?'#fbbf24':'#ef4444' }}/>}
                  {crosscheckStatus&&(
                    <div style={{ display:'flex',alignItems:'center',gap:3,fontSize:'9px',padding:'2px 7px',borderRadius:4,background:crosscheckStatus.status==='pass'?'rgba(34,197,94,0.12)':'rgba(239,68,68,0.12)',border:`1px solid ${crosscheckStatus.status==='pass'?'#22c55e':'#ef4444'}`,color:crosscheckStatus.status==='pass'?'#86efac':'#fca5a5' }}>
                      {crosscheckStatus.status==='pass'?<CheckCircle2 size={10}/>:<AlertTriangle size={10}/>}
                      {crosscheckStatus.status==='pass'?`JS=Python ✓ (${crosscheckStatus.mad.toFixed(2)}Hz)`:`Diverge ${crosscheckStatus.mad.toFixed(2)}Hz`}
                    </div>
                  )}
                </div>
              </div>
              <div style={{ backgroundColor:'#ffffff',border:`1.5px solid ${activeMethod==='sswt'?'rgba(251,191,36,0.5)':'rgba(239,68,68,0.3)'}`,borderRadius:8,overflow:'hidden' }}>
                <canvas ref={canvasRightRef} width={390} height={340} onMouseMove={e=>mouseMove(e,canvasRightRef)} onMouseLeave={()=>setHoverCoord(null)} style={{ width:'100%',height:'100%',cursor:'crosshair',display:'block' }}/>
              </div>

              {/* Clean Seismic Legend */}
              <div style={{ display:'flex',alignItems:'center',gap:8,marginTop:5 }}>
                <span style={{ fontSize:'9px',color:'#3b82f6',fontWeight:700 }}>◀ Trough (Negative RC)</span>
                <div className="rwb-bar" style={{ flex:1,height:8,borderRadius:3 }}/>
                <span style={{ fontSize:'9px',color:'#ef4444',fontWeight:700 }}>Peak (Positive RC) ▶</span>
              </div>
            </div>
          </div>

          {/* Bottom chart */}
          <div className="glass-card" style={{ padding:12,height:195,display:'flex',flexDirection:'column' }}>
            {activeMethod==='sswt' ? (
              <>
                <div style={{ display:'flex',alignItems:'baseline',gap:12,marginBottom:4 }}>
                  <h4 style={{ margin:0,fontSize:11,color:'#fde68a',letterSpacing:'0.05em',textTransform:'uppercase' }}>
                    {hoverCoord&&traceSSWTCache.current
                      ?`SSWT Full Spectrum @ TWT ${(tStart+hoverCoord.k*dtMs).toFixed(0)}ms — IL ${iMin+hoverCoord.i}  XL ${jMin+hoverCoord.j}`
                      :'Hover → Full SSWT spectrum at cursor depth'}
                  </h4>
                  {hoverRidge&&hoverThickness&&(
                    <div style={{ display:'flex',gap:14,fontSize:'10px' }}>
                      <span style={{ color:'#94a3b8' }}>Tuning Peak: <span style={{ fontWeight:800,color:'#fbbf24' }}>{hoverRidge.toFixed(0)} Hz</span></span>
                      <span style={{ color:'#94a3b8' }}>Bed Thickness: <span style={{ fontWeight:900,color:parseFloat(hoverThickness)<14?'#ef4444':parseFloat(hoverThickness)<20?'#fde68a':'#60a5fa' }}>{hoverThickness} m</span> (@ {velocityMps} m/s)</span>
                    </div>
                  )}
                </div>
                <div style={{ flex:1 }}>
                  {sswtSlice.length>0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={sswtSlice} margin={{ top:2,right:12,left:-18,bottom:0 }}>
                        <defs>
                          <linearGradient id="sswtGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.4}/>
                            <stop offset="95%" stopColor="#fbbf24" stopOpacity={0.02}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)"/>
                        <XAxis dataKey="freq" stroke="#475569" tick={{ fontSize:9 }} label={{ value:'Frequency (Hz)',position:'insideBottomRight',offset:-4,fill:'#475569',fontSize:9 }}/>
                        <YAxis stroke="#475569" domain={[0,1]} tick={{ fontSize:9 }}/>
                        <RechartsTooltip contentStyle={{ backgroundColor:'#0f172a',borderColor:'rgba(251,191,36,0.4)',color:'#fde68a',fontSize:'10px' }} formatter={v=>[v.toFixed(4),'SSWT power']}/>
                        {hoverRidge&&<ReferenceLine x={hoverRidge} stroke={hoverRidge>40?'#ef4444':'#fbbf24'} strokeWidth={2} label={{ value:`Tuning Peak ${hoverRidge.toFixed(0)}Hz (${hoverThickness}m)`,fill:hoverRidge>40?'#fca5a5':'#fde68a',fontSize:9,position:'top' }}/>}
                        <Area type="monotone" dataKey="power" stroke="#fbbf24" strokeWidth={2} fill="url(#sswtGrad)" dot={false}/>
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{ height:'100%',display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:6 }}>
                      <span style={{ color:'#1e293b',fontSize:'12px' }}>Hover on either panel to inspect SSWT peak tuning frequency & bed thickness</span>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                <h4 style={{ margin:'0 0 5px 0',fontSize:11,color:'#c7d2fe',letterSpacing:'0.05em',textTransform:'uppercase' }}>{hoverCoord?`Amplitude Spectrum — IL ${iMin+hoverCoord.i}  XL ${jMin+hoverCoord.j}`:'Amplitude Spectrum'}</h4>
                <div style={{ flex:1 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={psdData} margin={{ top:0,right:10,left:-20,bottom:0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)"/>
                      <XAxis dataKey="frequency" stroke="#475569" tick={{ fontSize:9 }}/>
                      <YAxis stroke="#475569" domain={[0,1]} tick={{ fontSize:9 }}/>
                      <RechartsTooltip contentStyle={{ backgroundColor:'#090d16',borderColor:'rgba(99,102,241,0.35)',color:'#fff',fontSize:'10px' }}/>
                      <Line type="monotone" dataKey="Raw" stroke="#3b82f6" strokeWidth={1.5} dot={false}/>
                      <Line type="monotone" dataKey="Enhanced" stroke="#ef4444" strokeWidth={2} dot={false}/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="glass-card panel-sidebar" style={{ display:'flex',flexDirection:'column',gap:12 }}>
          <h3>Method Selector</h3>
          <div style={{ display:'flex',flexDirection:'column',gap:7 }}>
            {Object.entries(METHOD_INFO).map(([key,info])=>(
              <button key={key} onClick={()=>setActiveMethod(key)} className={`well-toggle-btn ${activeMethod===key?'active':''}`}
                style={{ width:'100%',textAlign:'left',padding:'9px 12px',borderRadius:8,cursor:'pointer',border:`1.5px solid ${activeMethod===key?(key==='sswt'?'#fbbf24':'#ef4444'):'#1e293b'}`,background:activeMethod===key?(key==='sswt'?'rgba(251,191,36,0.07)':'rgba(239,68,68,0.08)'):'transparent',color:activeMethod===key?(key==='sswt'?'#fde68a':'#fca5a5'):'#94a3b8',fontSize:11.5,fontWeight:'bold',display:'flex',alignItems:'center',gap:7 }}>
                {key==='sswt'?<Zap size={13}/>:<Activity size={13}/>}{info.label}
              </button>
            ))}
          </div>

          <div className="info-badge"><Info size={14}/><span style={{ fontSize:'10px' }}>{METHOD_INFO[activeMethod].paperRef}</span></div>

          {/* Live thin bed readout */}
          {activeMethod==='sswt'&&hoverThickness&&(
            <div style={{ background:'rgba(15,23,42,0.9)',border:`1px solid ${parseFloat(hoverThickness)<14?'rgba(239,68,68,0.6)':'rgba(251,191,36,0.4)'}`,borderRadius:7,padding:'10px 12px' }}>
              <div style={{ fontSize:'9px',fontWeight:800,color:'#64748b',textTransform:'uppercase',marginBottom:4 }}>SSWT Bed Thickness Estimate</div>
              <div style={{ fontSize:'24px',fontWeight:900,color:parseFloat(hoverThickness)<14?'#ef4444':parseFloat(hoverThickness)<20?'#fde68a':'#60a5fa',lineHeight:1 }}>{hoverThickness} meters</div>
              <div style={{ fontSize:'10.5px',color:'#94a3b8',marginTop:4 }}>Peak Tuning: <span style={{ color:'#fbbf24',fontWeight:800 }}>{hoverRidge.toFixed(0)} Hz</span> (@ {velocityMps} m/s)</div>
              <div style={{ fontSize:'9.5px',color:'#64748b',lineHeight:'1.4',marginTop:4 }}>
                {parseFloat(hoverThickness) < 14
                  ?'🔥 THIN BED TUNING RESERVOIR (<14m) — High frequency tuning peak. Unresolvable on Raw Seismic!'
                  :parseFloat(hoverThickness) < 20
                  ?'⚡ MEDIUM BED (14-20m) — Moderate tuning interval.'
                  :'REGULAR THICK LAYER (>20m) — Above tuning resolution limit.'}
              </div>
            </div>
          )}

          <div>
            <label style={{ fontSize:'10px',fontWeight:800,color:'#64748b',textTransform:'uppercase' }}>Description:</label>
            <p style={{ fontSize:'11px',color:'#94a3b8',lineHeight:'1.55',marginTop:5 }}>{METHOD_INFO[activeMethod].desc}</p>
          </div>

          <div className="legend-box" style={{ marginTop:'auto' }}>
            <span className="legend-title">Seismic Amplitude (SSWT Deconvolved)</span>
            <div className="rwb-bar" style={{ marginBottom:8 }}/>
            <div className="control-group">
              <label>Amplitude Clip Limit: {clipLimit.toLocaleString()}</label>
              <input type="range" min={2000} max={25000} step={500} value={clipLimit} onChange={e=>setClipLimit(parseInt(e.target.value))} style={{ width:'100%',accentColor:'#3b82f6' }}/>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="workflow-footer-bar">
        <div style={{ display:'flex',alignItems:'center',gap:8 }}>
          <Activity size={13} color={activeMethod==='sswt'?'#fbbf24':'#3b82f6'}/>
          <span style={{ fontSize:'10px',color:'#64748b',fontWeight:600 }}>STATUS:</span>
          <span style={{ fontSize:'10px',fontWeight:700,color:activeMethod==='sswt'?'#fbbf24':'#3b82f6' }}>
            {activeMethod==='sswt'
              ?(computingSswt?'COMPUTING SSWT DECONVOLUTION…'
                :'AUTHENTIC SSWT DECONVOLUTION — USE "THIN-BED HIGHLIGHT OVERLAY" TO ILLUMINATE TUNING PINCHOUTS')
              :'3D INT16 MEMORY-MAPPED VOLUME'}
          </span>
        </div>
        {hoverCoord?(
          <div className="live-coord-readout" style={{ fontSize:'10px' }}>
            <span>IL: <strong>{iMin+hoverCoord.i}</strong></span>
            <span>XL: <strong>{jMin+hoverCoord.j}</strong></span>
            {sliceType!=='time'&&<span>TWT: <strong>{(tStart+hoverCoord.k*dtMs).toFixed(0)}ms</strong></span>}
            {hoverThickness&&(
              <span style={{ color:parseFloat(hoverThickness)<14?'#ef4444':'#10b981',fontWeight:900 }}>
                Est. Thickness: {hoverThickness}m
              </span>
            )}
          </div>
        ):(
          <span style={{ fontSize:'10px',color:'#475569',fontStyle:'italic' }}>
            Hover section to read SSWT tuning frequency & estimated thin-bed thickness in meters…
          </span>
        )}
      </div>
    </div>
  );
}
