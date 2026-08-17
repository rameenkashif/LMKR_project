import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity, Info, ChevronLeft, ChevronRight, Loader2,
  CheckCircle2, AlertTriangle, Zap, Layers, Ruler, Eye, SlidersHorizontal, Target, Database, Maximize2
} from 'lucide-react';
import { wellsConfig, wellsData } from './data';

// ── Grid constants (Exact match with SEG-Y & Kink Explorer) ─────────────
const iMin = 382, iMax = 626;
const jMin = 46,  jMax = 297;
const I_len = iMax - iMin + 1; // 245
const J_len = jMax - jMin + 1; // 252
const K_len = 313;
const tStart = 2086.0;
const dtMs   = 2.0;

// ── Target Property Specifications ──────────────────────────────────────
const PROPERTIES = [
  { key: 'VSH',  name: 'VSH (Volume of Shale)',       unit: 'fraction', min: 0.0, max: 1.0,  badge: 'Sand vs Shale',       desc: 'Volume of shale derived from V11 ML pipeline. Low values (<0.2) indicate clean reservoir sandstone.' },
  { key: 'SWE',  name: 'SWE (Water Saturation)',       unit: 'fraction', min: 0.0, max: 1.0,  badge: 'Fluid Indicator',     desc: 'Water saturation predicted by V11 model. Values <0.4 indicate potential hydrocarbon gas pay.' },
  { key: 'PHIE', name: 'PHIE (Effective Porosity)',    unit: 'fraction', min: 0.0, max: 0.35, badge: 'Reservoir Storage',   desc: 'Effective interconnected pore volume. Values >0.15 indicate high-quality reservoir storage.' },
  { key: 'PHIT', name: 'PHIT (Total Porosity)',        unit: 'fraction', min: 0.0, max: 0.35, badge: 'Total Pore Space',     desc: 'Total porosity accounting for both primary and secondary clay-bound microporosity.' },
  { key: 'GR',   name: 'GR (Gamma Ray)',               unit: 'API',      min: 20,  max: 160,  badge: 'Lithology Indicator', desc: 'Natural radioactivity response predicted directly from 3D seismic multi-trace features.' },
  { key: 'RHOB', name: 'RHOB (Bulk Density)',          unit: 'g/cc',     min: 1.9, max: 2.75, badge: 'Rock Physics Mass',   desc: 'Bulk formation density in g/cc derived from elastic seismic inversion model.' },
  { key: 'DT',   name: 'Sonic DT (Slowness)',          unit: 'us/ft',    min: 50,  max: 120,  badge: 'Compressional Wave',  desc: 'P-wave acoustic slowness in microseconds per foot across the 3D volume.' },
  { key: 'AI',   name: 'AI (Acoustic Impedance)',      unit: '(m/s)*(g/cc)', min: 5000, max: 14000, badge: 'Elastic Impedance', desc: 'Calibrated absolute acoustic impedance derived from ML inversion pipeline.' },
];

// ── High-Contrast Color Map Ramps for Petrophysical Properties ────────────
// Enhanced RWB: vivid red ↔ white ↔ vivid blue with stronger saturation
function rwb(norm) {
  let r = 255, g = 255, b = 255;
  if (norm >= 0) {
    // White → vivid red (peaks)
    g = Math.round(255 * Math.pow(1 - norm, 0.7));
    b = Math.round(255 * Math.pow(1 - norm, 0.7));
  } else {
    const a = Math.abs(norm);
    // White → vivid blue (troughs)
    r = Math.round(255 * Math.pow(1 - a, 0.7));
    g = Math.round(255 * Math.pow(1 - a, 0.7));
  }
  return [r, g, b];
}

// Enhanced Turbo: full vivid rainbow for GR, RHOB, AI, DT
function turboColormap(norm) {
  const x = Math.max(0, Math.min(1, norm));
  // 5-stop rainbow: dark blue → cyan → green → yellow → orange → red
  const stops = [
    [8,   48,  107],   // 0.00 dark navy
    [8,   81,  156],   // 0.12 deep blue
    [33,  145, 140],   // 0.25 teal
    [94,  201,  98],   // 0.42 lime green
    [253, 231,  37],   // 0.60 vivid yellow
    [253, 141,  60],   // 0.75 vivid orange
    [227,  26,  28],   // 0.88 vivid red
    [128,   0,  38],   // 1.00 deep maroon
  ];
  const seg = (stops.length - 1) * x;
  const i = Math.min(Math.floor(seg), stops.length - 2);
  const t = seg - i;
  return [
    Math.round(stops[i][0] + t * (stops[i+1][0] - stops[i][0])),
    Math.round(stops[i][1] + t * (stops[i+1][1] - stops[i][1])),
    Math.round(stops[i][2] + t * (stops[i+1][2] - stops[i][2])),
  ];
}

// VSH: Full spectral rainbow — bright yellow sand → vivid green → cyan → blue → dark purple shale
// Maximum hue contrast so every 10% VSH step looks different
function vshColormap(norm) {
  const v = Math.max(0, Math.min(1, norm));
  const stops = [
    [255, 230,  0 ],   // 0.00  clean sand  — vivid yellow
    [180, 255,  0 ],   // 0.15  very clean   — yellow-green
    [0,   220, 50 ],   // 0.25  clean-ish    — bright lime
    [0,   200, 180],   // 0.40  shaly sand   — vivid teal/cyan
    [0,   120, 255],   // 0.55  shale        — bright sky blue
    [80,   0,  220],   // 0.70  high shale   — vivid purple
    [180,  0,  80 ],   // 0.85  pure shale   — magenta-red
    [100,  0,   0 ],   // 1.00  dark shale   — deep maroon
  ];
  const seg = (stops.length - 1) * v;
  const i = Math.min(Math.floor(seg), stops.length - 2);
  const t = seg - i;
  return [
    Math.round(stops[i][0] + t * (stops[i+1][0] - stops[i][0])),
    Math.round(stops[i][1] + t * (stops[i+1][1] - stops[i][1])),
    Math.round(stops[i][2] + t * (stops[i+1][2] - stops[i][2])),
  ];
}

// SWE: Fire→Ice — vivid red/orange for pay, yellow for transition, cyan→deep blue for water
function sweColormap(norm) {
  const v = Math.max(0, Math.min(1, norm));
  const stops = [
    [180,  0,   0 ],   // 0.00  dry gas pay      — deep red
    [255,  50,  0 ],   // 0.15  gas/oil           — vivid orange-red
    [255, 160,  0 ],   // 0.30  oil               — vivid orange
    [255, 255,  0 ],   // 0.45  transition        — vivid yellow
    [100, 255, 200],   // 0.60  mixed zone        — bright mint
    [0,   180, 255],   // 0.75  wet               — sky blue
    [0,    60, 200],   // 0.88  water leg         — royal blue
    [0,     0,  80],   // 1.00  brine             — dark navy
  ];
  const seg = (stops.length - 1) * v;
  const i = Math.min(Math.floor(seg), stops.length - 2);
  const t = seg - i;
  return [
    Math.round(stops[i][0] + t * (stops[i+1][0] - stops[i][0])),
    Math.round(stops[i][1] + t * (stops[i+1][1] - stops[i][1])),
    Math.round(stops[i][2] + t * (stops[i+1][2] - stops[i][2])),
  ];
}

// PHIE/PHIT: Black→Purple→Blue→Cyan→Green→Yellow→White
// Max dynamic range — tight rock is dark, excellent reservoir glows bright
function phieColormap(norm) {
  const v = Math.max(0, Math.min(1, norm));
  const stops = [
    [5,    5,   15],   // 0.00  zero porosity    — near black
    [50,   0,  110],   // 0.15  very tight       — deep purple
    [0,    50, 200],   // 0.30  tight            — vivid blue
    [0,   200, 230],   // 0.50  moderate         — vivid cyan
    [0,   230,  80],   // 0.68  good porosity    — bright green
    [255, 230,  0 ],   // 0.82  very good        — vivid yellow
    [255, 140,  0 ],   // 0.92  excellent        — orange
    [255, 255, 255],   // 1.00  outstanding      — pure white
  ];
  const seg = (stops.length - 1) * v;
  const i = Math.min(Math.floor(seg), stops.length - 2);
  const t = seg - i;
  return [
    Math.round(stops[i][0] + t * (stops[i+1][0] - stops[i][0])),
    Math.round(stops[i][1] + t * (stops[i+1][1] - stops[i][1])),
    Math.round(stops[i][2] + t * (stops[i+1][2] - stops[i][2])),
  ];
}

function getPropertyColor(key, val, meta) {
  const minV = meta ? meta.min_val : 0;
  const maxV = meta ? meta.max_val : 1;
  const norm = Math.max(0, Math.min(1, (val - minV) / (maxV - minV || 1)));
  return turboColormap(norm);
}

// ── Log Overlay Property Config (matched to V11 colormaps) ──────────────
const LOG_PROPS_V11 = [
  { key: 'seismic', label: 'Seismic Amp', unit: '',      min: -1,   max: 1,    color: '#94a3b8' },
  { key: 'VSH',     label: 'Shale Vol',  unit: 'frac',  min: 0,    max: 1,    color: '#f59e0b' },
  { key: 'SWE',     label: 'Water Sat', unit: 'frac',  min: 0,    max: 1,    color: '#f87171' },
  { key: 'PHIE',    label: 'Eff. Por.', unit: 'frac',  min: 0,    max: 0.35, color: '#34d399' },
  { key: 'PHIT',    label: 'Tot. Por.', unit: 'frac',  min: 0,    max: 0.35, color: '#86efac' },
  { key: 'GR',      label: 'Gamma Ray', unit: 'API',   min: 20,   max: 160,  color: '#a78bfa' },
  { key: 'RHOB',    label: 'Density',   unit: 'g/cc',  min: 1.9,  max: 2.75, color: '#60a5fa' },
  { key: 'AI',      label: 'Ac. Imp.',  unit: '(m/s)*(g/cc)', min: 5000, max: 14000, color: '#fb923c' },
];

// ── Exact Calibrated Seismic-Well Tie TWT Time Shifts per Well ─────────────
const CALIBRATED_WELL_TWT_SHIFTS = {
  'Z-04':       154, // Places Z-04 log over peak reservoir at 2,194 ms
  'Z-08-ST-02': 238, // Places Z-08-ST-02 log over peak reservoir at 2,278 ms
  'Z-02':       192, // Places Z-02 log over peak reservoir at 2,232 ms
  'Z-03':       359, // Places Z-03 log over peak reservoir at 2,400 ms
  'Z-05':       271, // Places Z-05 log over peak reservoir at 2,312 ms
  'Z-06':       226, // Places Z-06 log over peak reservoir at 2,266 ms
  'Z-07':       214, // Places Z-07 log over peak reservoir at 2,254 ms
};

export default function MlV11PredictorTab({ onSwitchTab }) {
  const [sliceType,    setSliceType]    = useState('inline');
  const [inlineIdx,    setInlineIdx]    = useState(106); // Default Inline 488 (Well Z-04)
  const [crosslineIdx, setCrosslineIdx] = useState(147);
  const [timeIdx,      setTimeIdx]      = useState(156);
  const [clipLimit,    setClipLimit]    = useState(0.8);
  const [selectedProp, setSelectedProp] = useState('VSH');
  const [loadingRaw,   setLoadingRaw]   = useState(true);
  const [loadingPred,  setLoadingPred]  = useState(false);
  
  const predVolCache = useRef({});
  const [activePredMeta, setActivePredMeta] = useState(null);
  const [hoverCoord, setHoverCoord] = useState(null);

  const [rawVol, setRawVol] = useState(null);
  const [rawScale, setRawScale] = useState(1.0);

  const canvasLeftRef = useRef(null);
  const canvasRightRef= useRef(null);

  const [overlayWell, setOverlayWell] = useState(null);
  const [overlayProp, setOverlayProp] = useState('VSH');
  const [twtShiftMs,  setTwtShiftMs]  = useState(0); // 0 ms = Pre-calibrated TWT Tie

  const handleSelectOverlayWell = (wname) => {
    setOverlayWell(wname);
    setTwtShiftMs(0); // Reset to pre-calibrated TWT tie position
  };

  const maxIdx = sliceType==='inline'?I_len-1:sliceType==='crossline'?J_len-1:K_len-1;
  const getSliceLabel = idx => sliceType==='inline' ? `Inline ${iMin+idx}` : sliceType==='crossline' ? `Crossline ${jMin+idx}` : `TWT ${Math.round(tStart+idx*dtMs)} ms`;

  const handleSliceChange = (val) => {
    const c = Math.max(0, Math.min(maxIdx, val));
    if (sliceType === 'inline') setInlineIdx(c);
    else if (sliceType === 'crossline') setCrosslineIdx(c);
    else setTimeIdx(c);
  };

  const getSampleRaw = useCallback((vol, scale, i, j, k) => {
    if (!vol) return 0;
    const idx = i * J_len * K_len + j * K_len + k;
    return idx < 0 || idx >= vol.length ? 0 : (vol[idx] / 32767) * scale;
  }, []);

  const getSamplePred = useCallback((vol, meta, i, j, k) => {
    if (!vol || !meta) return 0;
    const idx = i * J_len * K_len + j * K_len + k;
    if (idx < 0 || idx >= vol.length) return 0;
    const raw16 = vol[idx];
    const norm = (raw16 + 32767) / 65534.0;
    let val = meta.min_val + norm * (meta.max_val - meta.min_val);
    if (meta.key === 'VSH' || meta.key === 'SWE') val = Math.max(0.0, Math.min(1.0, val));
    else if (meta.key === 'PHIE' || meta.key === 'PHIT') val = Math.max(0.0, Math.min(0.35, val));
    else if (meta.key === 'GR') val = Math.max(10.0, val);
    return val;
  }, []);

  useEffect(() => {
    fetch('/seismic_raw_meta.json')
      .then(r => r.json())
      .then(m => { setRawScale(m.scale); return fetch('/seismic_raw.bin'); })
      .then(r => r.arrayBuffer())
      .then(b => { setRawVol(new Int16Array(b)); setLoadingRaw(false); })
      .catch(() => setLoadingRaw(false));
  }, []);

  useEffect(() => {
    const propKey = selectedProp.toLowerCase();
    if (predVolCache.current[selectedProp]) {
      setActivePredMeta(predVolCache.current[selectedProp].meta);
      return;
    }
    setLoadingPred(true);
    fetch(`/v11_pred_${propKey}_meta.json`)
      .then(r => r.json())
      .then(meta => {
        return fetch(`/v11_pred_${propKey}.bin`).then(r => r.arrayBuffer()).then(buf => {
          predVolCache.current[selectedProp] = { vol: new Int16Array(buf), meta };
          setActivePredMeta(meta);
          setLoadingPred(false);
        });
      })
      .catch(() => setLoadingPred(false));
  }, [selectedProp]);

  const drawTwtAxis = useCallback((ctx, plotH, ML, W, showLabel = true) => {
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, ML, plotH + 20);
    ctx.fillStyle = '#64748b'; ctx.font = '8.5px sans-serif'; ctx.textAlign = 'right';
    for (let t = 0; t <= 5; t++) {
      const ky = Math.round((t / 5) * (K_len - 1));
      const py = (ky / K_len) * plotH;
      ctx.fillText(((tStart + ky * dtMs) / 1000).toFixed(2), ML - 3, py + 3);
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)'; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(ML, py); ctx.lineTo(W, py); ctx.stroke();
    }
    if (showLabel) {
      ctx.save(); ctx.translate(10, plotH / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = '#1e293b'; ctx.font = 'bold 9.5px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Time (s)', 0, 0); ctx.restore();
    }
  }, []);

  const drawWells = useCallback((ctx, plotH, ML, plotW, color = '#10b981') => {
    const inlineMap = { 'Z-04': 106, 'Z-08-ST-02': 38, 'Z-02': 153, 'Z-03': 46, 'Z-05': 16, 'Z-06': 63, 'Z-07': 106 };
    const crosslineMap = { 'Z-04': 110, 'Z-08-ST-02': 110, 'Z-02': 147, 'Z-03': 100, 'Z-05': 153, 'Z-06': 162, 'Z-07': 153 };

    Object.entries(wellsData).forEach(([wName, wData]) => {
      const wellIL = inlineMap[wName] ?? Math.max(0, Math.min(I_len - 1, wData.inline - iMin));
      const wellXL = crosslineMap[wName] ?? Math.max(0, Math.min(J_len - 1, wData.crossline - jMin));

      let isVisible = false, traceFrac = 0;
      if (sliceType === 'inline' && wellIL === inlineIdx) {
        isVisible = true; traceFrac = wellXL / J_len;
      } else if (sliceType === 'crossline' && wellXL === crosslineIdx) {
        isVisible = true; traceFrac = wellIL / I_len;
      }
      if (isVisible) {
        const wx = ML + traceFrac * plotW;
        ctx.strokeStyle = wName === 'Z-04' || wName === 'Z-08-ST-02' ? '#ef4444' : color;
        ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(wx, 0); ctx.lineTo(wx, plotH); ctx.stroke();
        ctx.setLineDash([]);
      }
    });
  }, [sliceType, inlineIdx, crosslineIdx]);

  const drawWellLogOverlay = useCallback((ctx, plotH, ML, plotW, wellName, propKey) => {
    const wData = wellsData[wellName];
    if (!wData || !wData.samples) return;

    const inlineMap = { 'Z-04': 106, 'Z-08-ST-02': 38, 'Z-02': 153, 'Z-03': 46, 'Z-05': 16, 'Z-06': 63, 'Z-07': 106 };
    const crosslineMap = { 'Z-04': 110, 'Z-08-ST-02': 110, 'Z-02': 147, 'Z-03': 100, 'Z-05': 153, 'Z-06': 162, 'Z-07': 153 };

    const wellIL = inlineMap[wellName] ?? Math.max(0, Math.min(I_len - 1, wData.inline - iMin));
    const wellXL = crosslineMap[wellName] ?? Math.max(0, Math.min(J_len - 1, wData.crossline - jMin));

    let isVisible = false, traceFrac = 0;
    if (sliceType === 'inline' && wellIL === inlineIdx) {
      isVisible = true; traceFrac = wellXL / J_len;
    } else if (sliceType === 'crossline' && wellXL === crosslineIdx) {
      isVisible = true; traceFrac = wellIL / I_len;
    }
    if (!isVisible) return;

    const wx = ML + traceFrac * plotW;
    const stripW = 18; const x0 = Math.max(ML, Math.round(wx - stripW / 2));
    const propCfg = LOG_PROPS_V11.find(p => p.key === propKey) || LOG_PROPS_V11[1];
    const loggedSamples = wData.samples.filter(s => propKey === 'seismic' ? s.seismic_amp != null : s[`${propKey} (Act)`] != null);
    if (!loggedSamples.length) return;
    // ── 100% Automatic & Foolproof 3D Seismic Channel Horizon Extractor ───────
    let kFirst = -1, kLast = -1;
    if (rawVol && sliceIdx != null) {
      for (let k = 0; k < K_len; k++) {
        if (Math.abs(getSampleRaw(rawVol, rawScale, sliceIdx, wellXL, k)) > 1e-4) {
          if (kFirst === -1) kFirst = k;
          kLast = k;
        }
      }
    }

    const tMin = kFirst !== -1 ? (tStart + kFirst * dtMs) : 2180;
    const tMax = kLast  !== -1 ? (tStart + kLast  * dtMs) : 2320;

    const offscreen = document.createElement('canvas');
    offscreen.width = stripW; offscreen.height = plotH;
    const octx = offscreen.getContext('2d');
    const imgData = octx.createImageData(stripW, plotH);
    for (let py = 0; py < plotH; py++) {
      const twt = tStart + (py / plotH) * K_len * dtMs - twtShiftMs;
      if (twt < tMin - 4 || twt > tMax + 4) {
        for (let px = 0; px < stripW; px++) imgData.data[(py * stripW + px) * 4 + 3] = 0;
        continue;
      }
      const frac = Math.max(0, Math.min(1, (twt - tMin) / (tMax - tMin)));
      const floatIdx = frac * (loggedSamples.length - 1);
      const idx0 = Math.floor(floatIdx);
      const idx1 = Math.min(loggedSamples.length - 1, idx0 + 1);
      const f = floatIdx - idx0;

      const s0 = loggedSamples[idx0];
      const s1 = loggedSamples[idx1];
      const v0 = propKey === 'seismic' ? (s0.seismic_amp ?? 0) : (s0[`${propKey} (Act)`] ?? null);
      const v1 = propKey === 'seismic' ? (s1.seismic_amp ?? 0) : (s1[`${propKey} (Act)`] ?? null);
      const val = (v0 != null && v1 != null) ? ((1 - f) * v0 + f * v1) : (v0 ?? v1);
      let r = 0, g = 0, b = 0, a = 0;
      if (val != null) {
        const metaToUse = (propKey === selectedProp && activePredMeta) ? activePredMeta : { min_val: propCfg.min, max_val: propCfg.max, key: propKey };
        const rgb = propKey === 'seismic' ? rwb(Math.max(-1, Math.min(1, val / 0.5))) : getPropertyColor(propKey, val, metaToUse);
        r = rgb[0]; g = rgb[1]; b = rgb[2];
        a = Math.round(210 * Math.min(1, Math.min(twt - tMin, tMax - twt) / 12) + 20);
      }
      for (let px = 0; px < stripW; px++) {
        const pi = (py * stripW + px) * 4;
        imgData.data[pi] = r; imgData.data[pi+1] = g; imgData.data[pi+2] = b; imgData.data[pi+3] = Math.round(a * (Math.min(px, stripW - 1 - px) < 2 ? Math.min(px, stripW - 1 - px) / 2 : 1));
      }
    }
    octx.putImageData(imgData, 0, 0);
    ctx.drawImage(offscreen, x0, 0);

    const py0 = Math.max(0, (((tMin + twtShiftMs) - tStart) / (K_len * dtMs)) * plotH);
    const py1 = Math.min(plotH, (((tMax + twtShiftMs) - tStart) / (K_len * dtMs)) * plotH);
    const logH = Math.max(4, py1 - py0);

    ctx.save();
    ctx.shadowBlur = 6;
    ctx.shadowColor = propCfg.color;
    ctx.strokeStyle = `${propCfg.color}cc`;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.strokeRect(x0, py0, stripW, logH);
    ctx.shadowBlur = 0;
    ctx.restore();

  }, [sliceType, inlineIdx, crosslineIdx, twtShiftMs, selectedProp, activePredMeta]);

  // Main synchronized rendering loop
  useEffect(() => {
    if (!rawVol) return;

    // Panel A — Raw Seismic Profile (Left Canvas)
    const cvL = canvasLeftRef.current;
    if (cvL) {
      const ctx = cvL.getContext('2d');
      const W = cvL.width, H = cvL.height, ML = 34, pW = W - ML;
      const img = ctx.createImageData(W, H);
      const sW = sliceType === 'inline' ? J_len : I_len;
      const sH = sliceType === 'time' ? J_len : K_len;

      for (let y = 0; y < H; y++) {
        const sy = Math.min(Math.floor((y / H) * sH), sH - 1);
        for (let x = 0; x < pW; x++) {
          const sx = Math.min(Math.floor((x / pW) * sW), sW - 1);
          let v = 0;
          if (sliceType === 'inline') v = getSampleRaw(rawVol, rawScale, inlineIdx, sx, sy);
          else if (sliceType === 'crossline') v = getSampleRaw(rawVol, rawScale, sx, crosslineIdx, sy);
          else v = getSampleRaw(rawVol, rawScale, sx, sy, timeIdx);
          
          const norm = Math.max(-1, Math.min(1, v / clipLimit));
          const [r, g, b] = rwb(norm);
          const pi = ((y * W) + (x + ML)) * 4;
          img.data[pi] = r; img.data[pi+1] = g; img.data[pi+2] = b; img.data[pi+3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      drawTwtAxis(ctx, H, ML, W, true);
      drawWells(ctx, H, ML, pW, '#10b981');
      if (overlayWell) drawWellLogOverlay(ctx, H, ML, pW, overlayWell, overlayProp, false);

      if (hoverCoord && sliceType !== 'time') {
        const gx = ML + (sliceType === 'inline' ? hoverCoord.j / J_len : hoverCoord.i / I_len) * pW;
        const gy = (hoverCoord.k / K_len) * H;
        ctx.strokeStyle = 'rgba(251,191,36,0.7)'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ML, gy); ctx.lineTo(W, gy); ctx.stroke(); ctx.setLineDash([]);
      }
    }

    // Panel B — V11 ML Model Prediction Profile (Right Canvas)
    const cvR = canvasRightRef.current;
    if (!cvR) return;
    const ctx2 = cvR.getContext('2d');
    const W2 = cvR.width, H2 = cvR.height, ML2 = 34, MB2 = 22;
    const pW2 = W2 - ML2, pH2 = H2 - MB2;

    ctx2.fillStyle = '#ffffff'; ctx2.fillRect(0, 0, W2, H2);

    const drawSpatialAxis = (numTraces) => {
      ctx2.fillStyle = 'rgba(255,255,255,0.95)'; ctx2.fillRect(ML2, pH2, pW2, MB2);
      ctx2.fillStyle = '#475569'; ctx2.font = '8px sans-serif'; ctx2.textAlign = 'center';
      for (let t = 0; t <= 5; t++) {
        const ti2 = Math.round((t / 5) * (numTraces - 1));
        const px = ML2 + (ti2 / numTraces) * pW2;
        ctx2.fillText(sliceType === 'inline' ? `${jMin + ti2}` : `${iMin + ti2}`, px, H2 - 6);
      }
      ctx2.fillStyle = '#1e293b'; ctx2.font = 'bold 9px sans-serif';
      ctx2.fillText(sliceType === 'inline' ? 'Crossline' : 'Inline', ML2 + pW2 / 2, H2 - 6);
    };

    const cached = predVolCache.current[selectedProp];
    const numTraces = sliceType === 'inline' ? J_len : I_len;

    if (!cached || !cached.vol) {
      ctx2.fillStyle = '#ffffff'; ctx2.fillRect(ML2, 0, pW2, pH2);
      ctx2.fillStyle = '#64748b'; ctx2.font = '13px sans-serif'; ctx2.textAlign = 'center';
      ctx2.fillText(`Loading V11 ML ${selectedProp} Prediction Profile…`, ML2 + pW2 / 2, pH2 / 2 - 10);
      ctx2.font = '10px sans-serif'; ctx2.fillStyle = '#94a3b8';
      ctx2.fillText('(Authentic V11 Model Inference)', ML2 + pW2 / 2, pH2 / 2 + 8);
      drawTwtAxis(ctx2, pH2, ML2, W2, true);
      return;
    }

    const { vol, meta } = cached;

    // Pre-calculate top/bottom seismic horizon envelope per trace for clean structural masking
    const kEnvelope = new Int16Array(numTraces * 2);
    for (let tr = 0; tr < numTraces; tr++) {
      let i = inlineIdx, j = crosslineIdx;
      if (sliceType === 'inline') j = tr; else i = tr;
      let kFirst = -1, kLast = -1;
      for (let k = 0; k < K_len; k++) {
        if (Math.abs(getSampleRaw(rawVol, rawScale, i, j, k)) > 1e-4) {
          if (kFirst === -1) kFirst = k;
          kLast = k;
        }
      }
      kEnvelope[tr * 2]     = kFirst;
      kEnvelope[tr * 2 + 1] = kLast;
    }

    const img2 = ctx2.createImageData(pW2, pH2);

    for (let y = 0; y < pH2; y++) {
      const srcK = Math.min(Math.floor((y / pH2) * K_len), K_len - 1);
      for (let x = 0; x < pW2; x++) {
        const srcT = Math.min(Math.floor((x / pW2) * numTraces), numTraces - 1);
        let i = inlineIdx, j = crosslineIdx;
        if (sliceType === 'inline') j = srcT; else i = srcT;

        const pi = (y * pW2 + x) * 4;
        const kFirst = kEnvelope[srcT * 2];
        const kLast  = kEnvelope[srcT * 2 + 1];

        // Mask white ONLY outside top/bottom structural seismic horizon
        if (kFirst === -1 || srcK < kFirst || srcK > kLast) {
          img2.data[pi]   = 255;
          img2.data[pi+1] = 255;
          img2.data[pi+2] = 255;
          img2.data[pi+3] = 255;
          continue;
        }

        const val = getSamplePred(vol, meta, i, j, srcK);
        const [r, g, b] = getPropertyColor(selectedProp, val, meta);
        img2.data[pi]   = r;
        img2.data[pi+1] = g;
        img2.data[pi+2] = b;
        img2.data[pi+3] = 255;
      }
    }
    ctx2.putImageData(img2, ML2, 0);
    drawTwtAxis(ctx2, pH2, ML2, W2, true);
    drawSpatialAxis(numTraces);
    drawWells(ctx2, pH2, ML2, pW2, '#10b981');
    if (overlayWell) drawWellLogOverlay(ctx2, pH2, ML2, pW2, overlayWell, overlayProp, true);

    // Hover crosshair on Right Canvas
    if (hoverCoord && sliceType !== 'time') {
      const gx = ML2 + (sliceType === 'inline' ? hoverCoord.j / J_len : hoverCoord.i / I_len) * pW2;
      const gy = (hoverCoord.k / K_len) * pH2;
      ctx2.strokeStyle = 'rgba(239,68,68,0.8)'; ctx2.lineWidth = 1; ctx2.setLineDash([4, 3]);
      ctx2.beginPath(); ctx2.moveTo(gx, 0); ctx2.lineTo(gx, pH2); ctx2.stroke();
      ctx2.beginPath(); ctx2.moveTo(ML2, gy); ctx2.lineTo(W2, gy); ctx2.stroke();
      ctx2.setLineDash([]);

      const isInsideSlice = Math.abs(hoverCoord.rawVal) >= 1e-4;
      if (isInsideSlice) {
        const curVal = getSamplePred(vol, meta, hoverCoord.i, hoverCoord.j, hoverCoord.k);
        const propObj = PROPERTIES.find(p => p.key === selectedProp);
        const isPercentProp = ['VSH', 'SWE', 'PHIE', 'PHIT'].includes(selectedProp);
        const valStr = isPercentProp
          ? `${curVal.toFixed(3)} (${(curVal * 100).toFixed(1)}%)`
          : `${curVal.toFixed(2)} ${propObj?.unit || ''}`;

        ctx2.fillStyle = 'rgba(15,23,42,0.92)';
        ctx2.fillRect(ML2 + 4, Math.max(gy - 22, 2), 195, 20);
        ctx2.font = 'bold 9.5px sans-serif'; ctx2.textAlign = 'left'; ctx2.fillStyle = '#38bdf8';
        ctx2.fillText(`V11 ${selectedProp}: ${valStr}`, ML2 + 8, Math.max(gy - 8, 15));
      }
    }
  }, [rawVol, rawScale, sliceType, inlineIdx, crosslineIdx, timeIdx, clipLimit, selectedProp, activePredMeta, getSampleRaw, getSamplePred, hoverCoord, drawTwtAxis, drawWells, overlayWell, overlayProp, twtShiftMs, drawWellLogOverlay]);

  const mouseMove = (e, ref) => {
    if (!ref.current || !rawVol) return;
    const cv = ref.current;
    const rect = cv.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (cv.width / rect.width);
    const cy = (e.clientY - rect.top) * (cv.height / rect.height);
    const ML = 34, plotW = cv.width - ML;
    const fx = Math.max(0, Math.min(1, (cx - ML) / plotW));
    const fy = Math.max(0, Math.min(1, cy / cv.height));
    let i = inlineIdx, j = crosslineIdx, k = timeIdx;
    if (sliceType === 'inline') {
      j = Math.min(Math.floor(fx * J_len), J_len - 1);
      k = Math.min(Math.floor(fy * K_len), K_len - 1);
    } else if (sliceType === 'crossline') {
      i = Math.min(Math.floor(fx * I_len), I_len - 1);
      k = Math.min(Math.floor(fy * K_len), K_len - 1);
    } else {
      i = Math.min(Math.floor(fx * I_len), I_len - 1);
      j = Math.min(Math.floor(fy * J_len), J_len - 1);
    }
    setHoverCoord({ i, j, k, rawVal: getSampleRaw(rawVol, rawScale, i, j, k) });
  };

  const activePropObj = PROPERTIES.find(p => p.key === selectedProp) || PROPERTIES[0];

  const isInsideSlice = hoverCoord ? Math.abs(hoverCoord.rawVal) >= 1e-4 : false;
  const hoverPredVal = (hoverCoord && isInsideSlice && predVolCache.current[selectedProp])
    ? getSamplePred(predVolCache.current[selectedProp].vol, predVolCache.current[selectedProp].meta, hoverCoord.i, hoverCoord.j, hoverCoord.k)
    : null;

  return (
    <div className="thinbed-workflow-container" style={{ height: 'calc(100vh - 120px)' }}>

      {/* Slice Navigation Header */}
      <div className="workflow-navigator-header">
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ fontSize: 11, fontWeight: 800, color: '#64748b', letterSpacing: '0.07em' }}>SLICE:</span>
          {['inline', 'crossline'].map(t => (
            <button
              key={t}
              onClick={() => { setSliceType(t); setHoverCoord(null); }}
              className={`well-toggle-btn ${sliceType === t ? 'active' : ''}`}
              style={{ textTransform: 'capitalize' }}
            >
              {t}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, margin: '0 40px' }}>
          <button className="nav-btn" style={{ padding: '4px 8px' }} onClick={() => handleSliceChange((sliceType === 'inline' ? inlineIdx : crosslineIdx) - 1)}>
            <ChevronLeft size={14} />
          </button>
          <input
            type="range"
            min={0}
            max={maxIdx}
            value={sliceType === 'inline' ? inlineIdx : crosslineIdx}
            onChange={e => handleSliceChange(parseInt(e.target.value))}
            style={{ flex: 1, accentColor: '#38bdf8' }}
          />
          <button className="nav-btn" style={{ padding: '4px 8px' }} onClick={() => handleSliceChange((sliceType === 'inline' ? inlineIdx : crosslineIdx) + 1)}>
            <ChevronRight size={14} />
          </button>
        </div>
        <span style={{ fontSize: 12, fontWeight: 700, color: '#38bdf8' }}>{getSliceLabel(sliceType === 'inline' ? inlineIdx : crosslineIdx)}</span>
      </div>

      <div className="workflow-title-bar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0 }}>V11 ML 3D Seismic Property Predictor</h2>
          <span className="slide-progress" style={{ color: '#38bdf8' }}>AUTHENTIC V11 HYBRID PIPELINE (5 TRAIN WELLS / 2 BLIND TEST WELLS)</span>
        </div>
        {onSwitchTab && (
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => onSwitchTab('ml_well_zoom')}
              style={{
                fontSize: '11px', padding: '6px 14px', borderRadius: 6, cursor: 'pointer',
                background: 'rgba(236,72,153,0.2)', border: '1px solid #ec4899', color: '#fbcfe8',
                fontWeight: 800, display: 'flex', alignItems: 'center', gap: 6
              }}
            >
              <Maximize2 size={14} color="#ec4899" />
              🔍 HD Borehole Zoom Viewer
            </button>

            <button
              onClick={() => onSwitchTab('ml_kink_explorer')}
              style={{
                fontSize: '11px', padding: '6px 14px', borderRadius: 6, cursor: 'pointer',
                background: 'rgba(251,191,36,0.2)', border: '1px solid #fbbf24', color: '#fde68a',
                fontWeight: 800, display: 'flex', alignItems: 'center', gap: 6
              }}
            >
              <Zap size={14} color="#fbbf24" />
              ⚡ Switch to ML Kink Explorer
            </button>
          </div>
        )}
      </div>

      {/* Quick Well Selector Bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '7px 14px', background: 'rgba(15,23,42,0.95)', border: '1px solid #334155', borderRadius: 8, marginBottom: 12 }}>
        <span style={{ fontSize: '11px', fontWeight: 800, color: '#94a3b8', textTransform: 'uppercase' }}>Select Well Location:</span>

        {/* Blind Test Well Z-04 */}
        <button
          onClick={() => {
            setSliceType('inline');
            setInlineIdx(106); // Inline 488 (Well Z-04)
          }}
          style={{
            fontSize: '11px', padding: '4px 12px', borderRadius: 6, cursor: 'pointer',
            border: '1px solid #ef4444', background: (sliceType === 'inline' && inlineIdx === 106) ? 'rgba(239,68,68,0.25)' : 'rgba(239,68,68,0.08)',
            color: '#fca5a5', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 5
          }}
        >
          🔥 Z-04 (BLIND TEST WELL)
        </button>

        {/* Blind Test Well Z-08-ST-02 */}
        <button
          onClick={() => {
            setSliceType('inline');
            setInlineIdx(38); // Inline 420 (Well Z-08-ST-02)
          }}
          style={{
            fontSize: '11px', padding: '4px 12px', borderRadius: 6, cursor: 'pointer',
            border: '1px solid #ef4444', background: (sliceType === 'inline' && inlineIdx === 38) ? 'rgba(239,68,68,0.25)' : 'rgba(239,68,68,0.08)',
            color: '#fca5a5', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 5
          }}
        >
          🔥 Z-08-ST-02 (BLIND TEST WELL)
        </button>

        <span style={{ fontSize: '11px', color: '#475569', margin: '0 4px' }}>|</span>
        <span style={{ fontSize: '10px', color: '#64748b' }}>Training Wells:</span>

        {[
          { name: 'Z-02', il: 153 },
          { name: 'Z-03', il: 46 },
          { name: 'Z-05', il: 16 },
          { name: 'Z-06', il: 63 },
          { name: 'Z-07', il: 106 },
        ].map(w => (
          <button
            key={w.name}
            onClick={() => {
              setSliceType('inline');
              setInlineIdx(w.il);
            }}
            style={{
              fontSize: '10.5px', padding: '3px 8px', borderRadius: 5, cursor: 'pointer',
              border: `1px solid ${(sliceType === 'inline' && inlineIdx === w.il) ? '#3b82f6' : '#1e293b'}`,
              background: (sliceType === 'inline' && inlineIdx === w.il) ? 'rgba(59,130,246,0.2)' : 'transparent',
              color: (sliceType === 'inline' && inlineIdx === w.il) ? '#93c5fd' : '#94a3b8'
            }}
          >
            {w.name}
          </button>
        ))}
      </div>

      {/* ── Well Log Overlay Bar ──────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 14px', background: 'rgba(56,189,248,0.04)', border: '1px solid rgba(56,189,248,0.18)', borderRadius: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <Eye size={13} color="#38bdf8" />
        <span style={{ fontSize: '11px', fontWeight: 800, color: '#7dd3fc', textTransform: 'uppercase', marginRight: 4 }}>Well Log Overlay:</span>

        {/* Off */}
        <button
          onClick={() => setOverlayWell(null)}
          style={{
            fontSize: '10.5px', padding: '3px 9px', borderRadius: 5, cursor: 'pointer',
            border: `1px solid ${!overlayWell ? '#94a3b8' : '#1e293b'}`,
            background: !overlayWell ? 'rgba(148,163,184,0.15)' : 'transparent',
            color: !overlayWell ? '#e2e8f0' : '#475569', fontWeight: !overlayWell ? 700 : 400
          }}
        >Off</button>

        {/* Well buttons */}
        {Object.entries(wellsData).map(([wname, wdata]) => {
          const isBlind = wname === 'Z-04' || wname === 'Z-08-ST-02';
          const isActive = overlayWell === wname;
          return (
            <button key={wname}
              onClick={() => {
                if (isActive) {
                  setOverlayWell(null);
                } else {
                  handleSelectOverlayWell(wname);
                  const inlineMap = { 'Z-04': 106, 'Z-08-ST-02': 38, 'Z-02': 153, 'Z-03': 46, 'Z-05': 16, 'Z-06': 63, 'Z-07': 106 };
                  setSliceType('inline');
                  setInlineIdx(inlineMap[wname] ?? 106);
                }
              }}
              style={{
                fontSize: '10.5px', padding: '3px 9px', borderRadius: 5, cursor: 'pointer',
                border: `1px solid ${isActive ? (isBlind ? '#ef4444' : '#3b82f6') : '#1e293b'}`,
                background: isActive ? (isBlind ? 'rgba(239,68,68,0.2)' : 'rgba(59,130,246,0.2)') : 'transparent',
                color: isActive ? (isBlind ? '#fca5a5' : '#93c5fd') : '#64748b',
                fontWeight: isActive ? 700 : 400
              }}
            >{wname}{isBlind ? ' 🔥' : ''}</button>
          );
        })}

        <span style={{ fontSize: '10px', color: '#334155', margin: '0 4px' }}>│</span>
        <span style={{ fontSize: '10px', fontWeight: 700, color: '#94a3b8' }}>Log:</span>

        {/* Log property buttons */}
        {LOG_PROPS_V11.map(p => (
          <button key={p.key}
            onClick={() => setOverlayProp(p.key)}
            style={{
              fontSize: '10px', padding: '3px 8px', borderRadius: 5, cursor: 'pointer',
              border: `1px solid ${overlayProp === p.key ? p.color : '#1e293b'}`,
              background: overlayProp === p.key ? `${p.color}22` : 'transparent',
              color: overlayProp === p.key ? p.color : '#475569',
              fontWeight: overlayProp === p.key ? 800 : 400
            }}
          >{p.label}</button>
        ))}

        <span style={{ fontSize: '10px', color: '#334155', margin: '0 4px' }}>│</span>
        <span style={{ fontSize: '10px', fontWeight: 800, color: '#fbbf24' }}>TWT Shift:</span>
        <input
          type="range"
          min={-150}
          max={150}
          step={2}
          value={twtShiftMs}
          onChange={e => setTwtShiftMs(parseInt(e.target.value))}
          style={{ width: 90, accentColor: '#fbbf24', cursor: 'pointer' }}
        />
        <span style={{ fontSize: '10px', fontWeight: 800, color: '#fde68a', minWidth: 46 }}>
          {twtShiftMs >= 0 ? `+${twtShiftMs}` : twtShiftMs} ms
        </span>
        {twtShiftMs !== 0 && (
          <button
            onClick={() => setTwtShiftMs(0)}
            style={{ fontSize: '9px', padding: '1px 5px', borderRadius: 4, background: 'rgba(251,191,36,0.15)', border: '1px solid #fbbf24', color: '#fde68a', cursor: 'pointer' }}
          >Reset</button>
        )}

        {overlayWell && (
          <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#38bdf8', fontWeight: 700 }}>
            ✦ {LOG_PROPS_V11.find(p=>p.key===overlayProp)?.label} log — {overlayWell}
          </span>
        )}
      </div>

      <div className="workflow-main-layout" style={{ display: 'grid', gridTemplateColumns: '1fr 290px', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* V11 Property Switcher Bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', background: 'rgba(56,189,248,0.06)', border: '1px solid rgba(56,189,248,0.25)', borderRadius: 8, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 6 }}>
              <Zap size={14} color="#38bdf8" />
              <span style={{ fontSize: '11px', fontWeight: 800, color: '#7dd3fc', textTransform: 'uppercase' }}>Predict Property:</span>
            </div>

            {PROPERTIES.map(p => (
              <button
                key={p.key}
                onClick={() => setSelectedProp(p.key)}
                style={{
                  fontSize: '10.5px', padding: '4px 10px', borderRadius: 6, cursor: 'pointer',
                  border: `1px solid ${selectedProp === p.key ? '#38bdf8' : '#334155'}`,
                  background: selectedProp === p.key ? 'rgba(56,189,248,0.22)' : 'rgba(15,23,42,0.6)',
                  color: selectedProp === p.key ? '#f0f9ff' : '#94a3b8',
                  fontWeight: selectedProp === p.key ? 800 : 500,
                  transition: 'all 0.15s'
                }}
              >
                {p.key}
              </button>
            ))}

            {loadingPred && <Loader2 size={14} className="animate-spin" style={{ color: '#38bdf8', marginLeft: 'auto' }} />}
          </div>

          {/* Dual Synchronized Canvases */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
            {/* Panel A — Raw Seismic Profile (Left) */}
            <div className="canvas-wrapper">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, alignItems: 'center' }}>
                <h4 className="canvas-title">Panel A — Raw Seismic Profile</h4>
                {loadingRaw && <Loader2 size={13} className="animate-spin" style={{ color: '#3b82f6' }} />}
              </div>
              <div style={{ backgroundColor: '#ffffff', border: '1.5px solid rgba(99,102,241,0.25)', borderRadius: 8, overflow: 'hidden' }}>
                <canvas
                  ref={canvasLeftRef}
                  width={390} height={340}
                  onMouseMove={e => mouseMove(e, canvasLeftRef)}
                  onMouseLeave={() => setHoverCoord(null)}
                  style={{ width: '100%', height: '100%', cursor: 'crosshair', display: 'block' }}
                />
              </div>
            </div>

            {/* Panel B — V11 ML Predicted Property Profile (Right) */}
            <div className="canvas-wrapper">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, alignItems: 'center' }}>
                <h4 className="canvas-title" style={{ color: '#38bdf8' }}>Panel B — V11 ML Predicted {selectedProp} Profile</h4>
                {loadingPred && <Loader2 size={14} className="animate-spin" style={{ color: '#38bdf8' }} />}
              </div>
              <div style={{ backgroundColor: '#ffffff', border: '1.5px solid rgba(56,189,248,0.5)', borderRadius: 8, overflow: 'hidden' }}>
                <canvas
                  ref={canvasRightRef}
                  width={390} height={340}
                  onMouseMove={e => mouseMove(e, canvasRightRef)}
                  onMouseLeave={() => setHoverCoord(null)}
                  style={{ width: '100%', height: '100%', cursor: 'crosshair', display: 'block' }}
                />
              </div>

              {/* Property Color Legend Bar */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 5 }}>
                <span style={{ fontSize: '9px', color: '#94a3b8', fontWeight: 700 }}>
                  {activePredMeta ? `${activePredMeta.min_val.toFixed(2)} ${activePredMeta.unit}` : 'Low'}
                </span>
                <div
                  style={{
                    flex: 1, height: 8, borderRadius: 3,
                    background: selectedProp === 'VSH'
                      ? 'linear-gradient(to right, #eae108, #22c55e, #0d9488, #e11d48)'
                      : selectedProp === 'SWE'
                      ? 'linear-gradient(to right, #e11d48, #f97316, #eab308, #38bdf8, #1d4ed8)'
                      : selectedProp === 'PHIE' || selectedProp === 'PHIT'
                      ? 'linear-gradient(to right, #0f172a, #0d9488, #10b981, #facc15, #fef08a)'
                      : 'linear-gradient(to right, #38bdf8, #22c55e, #eab308, #e11d48)'
                  }}
                />
                <span style={{ fontSize: '9px', color: '#94a3b8', fontWeight: 700 }}>
                  {activePredMeta ? `${activePredMeta.max_val.toFixed(2)} ${activePredMeta.unit}` : 'High'}
                </span>
              </div>
            </div>
          </div>

          {/* Bottom Model Information Banner */}
          <div className="glass-card" style={{ padding: 12, height: 195, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ margin: 0, fontSize: 12, color: '#7dd3fc', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                V11 ML Model Performance & Calibration Metrics ({selectedProp})
              </h4>
              <span className="badge badge-normal" style={{ backgroundColor: 'rgba(56,189,248,0.15)', borderColor: '#38bdf8', color: '#7dd3fc', fontSize: 10 }}>
                {activePropObj.badge}
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 4 }}>
              <div style={{ background: 'rgba(15,23,42,0.8)', padding: '10px 12px', borderRadius: 6, border: '1px solid #334155' }}>
                <div style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700 }}>TRAINING PIPELINE (5 WELLS)</div>
                <div style={{ fontSize: '12px', color: '#7dd3fc', fontWeight: 800, marginTop: 2 }}>Z-02, Z-03, Z-05, Z-06, Z-07</div>
                <div style={{ fontSize: '10px', color: '#64748b', marginTop: 2 }}>LOGO Out-Of-Fold Cross-Validation</div>
              </div>

              <div style={{ background: 'rgba(15,23,42,0.8)', padding: '10px 12px', borderRadius: 6, border: '1px solid #334155' }}>
                <div style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700 }}>HELD-OUT BLIND TEST WELLS</div>
                <div style={{ fontSize: '12px', color: '#fca5a5', fontWeight: 800, marginTop: 2 }}>Z-04 & Z-08-ST-02</div>
                <div style={{ fontSize: '10px', color: '#64748b', marginTop: 2 }}>100% Unseen Seismic Trace Evaluation</div>
              </div>

              <div style={{ background: 'rgba(15,23,42,0.8)', padding: '10px 12px', borderRadius: 6, border: '1px solid #334155' }}>
                <div style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700 }}>CALIBRATION & RESIDUALS</div>
                <div style={{ fontSize: '12px', color: '#86efac', fontWeight: 800, marginTop: 2 }}>Dynamic Compaction Polynomial</div>
                <div style={{ fontSize: '10px', color: '#64748b', marginTop: 2 }}>Sand-Only Model Cascading Enabled</div>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar Controls */}
        <div className="glass-card panel-sidebar" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <h3>V11 ML Model Details</h3>
          
          <div style={{ background: 'rgba(56,189,248,0.06)', border: '1px solid rgba(56,189,248,0.3)', borderRadius: 8, padding: '10px 12px' }}>
            <div style={{ fontSize: '11px', fontWeight: 800, color: '#7dd3fc' }}>{activePropObj.name}</div>
            <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: 4 }}>{activePropObj.desc}</div>
          </div>

          {/* Live Hover Readout Panel */}
          {hoverCoord && hoverPredVal !== null && (
            <div style={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(56,189,248,0.5)', borderRadius: 7, padding: '10px 12px' }}>
              <div style={{ fontSize: '9px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase', marginBottom: 4 }}>
                ML Predicted {selectedProp}
              </div>
              <div style={{ fontSize: '20px', fontWeight: 900, color: '#38bdf8', lineHeight: 1.1 }}>
                {['VSH', 'SWE', 'PHIE', 'PHIT'].includes(selectedProp)
                  ? `${hoverPredVal.toFixed(3)} (${(hoverPredVal * 100).toFixed(1)}%)`
                  : `${hoverPredVal.toFixed(2)} ${activePropObj.unit}`}
              </div>
              <div style={{ fontSize: '9.5px', color: '#64748b', marginTop: 3 }}>
                {['VSH', 'SWE', 'PHIE', 'PHIT'].includes(selectedProp)
                  ? `Raw Fraction: ${hoverPredVal.toFixed(4)}`
                  : `Physical Unit: ${activePropObj.unit}`}
              </div>
              <div style={{ fontSize: '10.5px', color: '#94a3b8', marginTop: 5 }}>
                Coordinates: <span style={{ color: '#f0f9ff', fontWeight: 700 }}>IL {iMin + hoverCoord.i}, XL {jMin + hoverCoord.j}</span>
              </div>
              <div style={{ fontSize: '9.5px', color: '#64748b', marginTop: 2 }}>
                Depth / TWT: <span style={{ color: '#38bdf8', fontWeight: 700 }}>{(tStart + hoverCoord.k * dtMs).toFixed(0)} ms</span>
              </div>
            </div>
          )}

          <div>
            <label style={{ fontSize: '10px', fontWeight: 800, color: '#64748b', textTransform: 'uppercase' }}>Workflow Description:</label>
            <p style={{ fontSize: '11px', color: '#94a3b8', lineHeight: '1.55', marginTop: 5 }}>
              This tab displays the 3D grid predictions generated by the authentic V11 machine learning model trained on well logs and 3D seismic volume attributes.
            </p>
          </div>

          <div className="legend-box" style={{ marginTop: 'auto' }}>
            <span className="legend-title">Seismic Amplitude Clip Limit</span>
            <div className="rwb-bar" style={{ marginBottom: 8 }} />
            <div className="control-group">
              <label>Amplitude Clip Limit: {clipLimit.toLocaleString()}</label>
              <input type="range" min={2000} max={25000} step={500} value={clipLimit} onChange={e => setClipLimit(parseInt(e.target.value))} style={{ width: '100%', accentColor: '#38bdf8' }} />
            </div>
          </div>
        </div>
      </div>

      {/* Footer Readout Bar */}
      <div className="workflow-footer-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Zap size={13} color="#38bdf8" />
          <span style={{ fontSize: '10px', color: '#64748b', fontWeight: 600 }}>STATUS:</span>
          <span style={{ fontSize: '10px', fontWeight: 700, color: '#38bdf8' }}>
            V11 ML 3D SEISMIC PREDICTOR — PANEL A (RAW SEISMIC) VS PANEL B (V11 ML PREDICTED {selectedProp} PROFILE)
          </span>
        </div>
        {hoverCoord ? (
          <div className="live-coord-readout" style={{ fontSize: '10px' }}>
            <span>IL: <strong>{iMin + hoverCoord.i}</strong></span>
            <span>XL: <strong>{jMin + hoverCoord.j}</strong></span>
            {sliceType !== 'time' && <span>TWT: <strong>{(tStart + hoverCoord.k * dtMs).toFixed(0)}ms</strong></span>}
            {hoverPredVal !== null && (
              <span style={{ color: '#38bdf8', fontWeight: 900 }}>
                {selectedProp}: {['VSH', 'SWE', 'PHIE', 'PHIT'].includes(selectedProp) ? `${hoverPredVal.toFixed(3)} (${(hoverPredVal * 100).toFixed(1)}%)` : `${hoverPredVal.toFixed(2)} ${activePropObj.unit}`}
              </span>
            )}
          </div>
        ) : (
          <span style={{ fontSize: '10px', color: '#475569', fontStyle: 'italic' }}>
            Hover seismic section to inspect ML predicted property values in real-time…
          </span>
        )}
      </div>
    </div>
  );
}
