import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity, Info, ChevronLeft, ChevronRight, Loader2,
  CheckCircle2, AlertTriangle, Zap, Layers, Ruler, Eye, SlidersHorizontal, Target, Database, Maximize2, MoveHorizontal, ZoomIn
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
  { key: 'VSH',  name: 'VSH (Volume of Shale)',       unit: 'fraction', min: 0.0, max: 1.0,  badge: 'Sand vs Shale',       color: '#f59e0b', desc: 'Volume of shale derived from V11 ML pipeline. Low values (<0.2) indicate clean reservoir sandstone.' },
  { key: 'SWE',  name: 'SWE (Water Saturation)',       unit: 'fraction', min: 0.0, max: 1.0,  badge: 'Fluid Indicator',     color: '#f87171', desc: 'Water saturation predicted by V11 model. Values <0.4 indicate potential hydrocarbon gas pay.' },
  { key: 'PHIE', name: 'PHIE (Effective Porosity)',    unit: 'fraction', min: 0.0, max: 0.35, badge: 'Reservoir Storage',   color: '#34d399', desc: 'Effective interconnected pore volume. Values >0.15 indicate high-quality reservoir storage.' },
  { key: 'PHIT', name: 'PHIT (Total Porosity)',        unit: 'fraction', min: 0.0, max: 0.35, badge: 'Total Pore Space',     color: '#86efac', desc: 'Total porosity accounting for both primary and secondary clay-bound microporosity.' },
  { key: 'GR',   name: 'GR (Gamma Ray)',               unit: 'API',      min: 20,  max: 160,  badge: 'Lithology Indicator', color: '#a78bfa', desc: 'Natural radioactivity response predicted directly from 3D seismic multi-trace features.' },
  { key: 'RHOB', name: 'RHOB (Bulk Density)',          unit: 'g/cc',     min: 1.9, max: 2.75, badge: 'Rock Physics Mass',   color: '#60a5fa', desc: 'Bulk formation density in g/cc derived from elastic seismic inversion model.' },
  { key: 'DT',   name: 'Sonic DT (Slowness)',          unit: 'us/ft',    min: 50,  max: 120,  badge: 'Compressional Wave',  color: '#38bdf8', desc: 'P-wave acoustic slowness in microseconds per foot across the 3D volume.' },
  { key: 'AI',   name: 'AI (Acoustic Impedance)',      unit: '(m/s)*(g/cc)', min: 5000, max: 14000, badge: 'Elastic Impedance', color: '#fb923c', desc: 'Calibrated absolute acoustic impedance derived from ML inversion pipeline.' },
];

// ── RWB colormap for seismic amplitude ────────────────────────────────────
function rwb(norm) {
  let r = 255, g = 255, b = 255;
  if (norm >= 0) {
    g = Math.round(255 * Math.pow(1 - norm, 0.7));
    b = Math.round(255 * Math.pow(1 - norm, 0.7));
  } else {
    const a = Math.abs(norm);
    r = Math.round(255 * Math.pow(1 - a, 0.7));
    g = Math.round(255 * Math.pow(1 - a, 0.7));
  }
  return [r, g, b];
}

// ── Full-spectrum Turbo Colormap (Dark Navy -> Blue -> Teal -> Green -> Yellow -> Orange -> Red) ──
function turboColormap(norm) {
  const x = Math.max(0, Math.min(1, norm));
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

function getPropertyColor(val, meta) {
  const minV = meta ? meta.min_val : 0;
  const maxV = meta ? meta.max_val : 1;
  const norm = Math.max(0, Math.min(1, (val - minV) / (maxV - minV || 1)));
  return turboColormap(norm);
}

// ── Precise Calibrated Well Coordinates (Seismic Inline / Crossline Indices) ────
const WELL_COORDS = {
  'Z-04':       { localIL: 106, localXL: 110, inline: 488, crossline: 156 },
  'Z-08-ST-02': { localIL: 38,  localXL: 110, inline: 420, crossline: 156 },
  'Z-02':       { localIL: 153, localXL: 147, inline: 535, crossline: 193 },
  'Z-03':       { localIL: 46,  localXL: 100, inline: 428, crossline: 146 },
  'Z-05':       { localIL: 16,  localXL: 153, inline: 398, crossline: 199 },
  'Z-06':       { localIL: 63,  localXL: 162, inline: 445, crossline: 208 },
  'Z-07':       { localIL: 106, localXL: 153, inline: 488, crossline: 199 },
};

export default function MlWellZoomTab({ onSwitchTab }) {
  const [selectedWell, setSelectedWell] = useState('Z-04');
  const [selectedProp, setSelectedProp] = useState('VSH');
  const [zoomRadius,   setZoomRadius]   = useState(14); // Optimal ±14 crosslines around well
  const [contrastMode, setContrastMode] = useState('global'); // 'global' (absolute physical scale 0..1) for 100% color match
  const [useBilinear,  setUseBilinear]  = useState(true);   // smooth bilinear contouring

  const [loadingRaw,   setLoadingRaw]   = useState(true);
  const [loadingPred,  setLoadingPred]  = useState(false);

  const [rawVol,       setRawVol]       = useState(null);
  const [rawScale,     setRawScale]     = useState(1.0);

  const predVolCache = useRef({});
  const [activePredMeta, setActivePredMeta] = useState(null);
  const [hoverCoord,     setHoverCoord]     = useState(null);

  const canvasRef = useRef(null);

  // Get active well info and calibrated seismic coordinates
  const activeWellData = wellsData[selectedWell] || wellsData['Z-04'];
  const wellCoord = WELL_COORDS[selectedWell] || { localIL: 106, localXL: 110, inline: 488, crossline: 156 };
  const wellIL = wellCoord.localIL;
  const wellXL = wellCoord.localXL;

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

  // Load raw seismic volume
  useEffect(() => {
    fetch('/seismic_raw_meta.json')
      .then(r => r.json())
      .then(m => { setRawScale(m.scale); return fetch('/seismic_raw.bin'); })
      .then(r => r.arrayBuffer())
      .then(b => { setRawVol(new Int16Array(b)); setLoadingRaw(false); })
      .catch(() => setLoadingRaw(false));
  }, []);

  // Load prediction volume for selected property
  useEffect(() => {
    const propKey = selectedProp.toLowerCase();
    if (predVolCache.current[selectedProp]) {
      setActivePredMeta(predVolCache.current[selectedProp].meta);
      return;
    }

    setLoadingPred(true);
    const metaPath = `/v11_pred_${propKey}_meta.json`;
    const binPath  = `/v11_pred_${propKey}.bin`;

    fetch(metaPath)
      .then(r => r.json())
      .then(meta => {
        return fetch(binPath).then(r => r.arrayBuffer()).then(buf => {
          const vol = new Int16Array(buf);
          predVolCache.current[selectedProp] = { vol, meta };
          setActivePredMeta(meta);
          setLoadingPred(false);
        });
      })
      .catch(err => {
        console.warn(`Could not load prediction volume for ${selectedProp}:`, err);
        setLoadingPred(false);
      });
  }, [selectedProp]);

  // ── Calculate exact valid seismic & well log sub-volume bounds ──────────────
  const loggedSamples = activeWellData?.samples?.filter(s => {
    if (!s.time) return false;
    if (selectedProp === 'seismic') return s.seismic_amp != null;
    return s[`${selectedProp} (Act)`] != null;
  }) || [];

  const xlMin = Math.max(0, wellXL - zoomRadius);
  const xlMax = Math.min(J_len - 1, wellXL + zoomRadius);

  // Full seismic depth section from top to bottom (2,086 ms to 2,710 ms)
  const kMin = 0;
  const kMax = K_len - 1;

  const tMin = tStart + kMin * dtMs;
  const tMax = tStart + kMax * dtMs;

  // ── Draw TWT Y-Axis ──────────────────────────────────────────────────
  const drawTwtAxis = useCallback((ctx, plotH, ML, W) => {
    ctx.fillStyle = 'rgba(15,23,42,0.95)';
    ctx.fillRect(0, 0, ML, plotH + 25);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px sans-serif'; ctx.textAlign = 'right';
    const steps = 6;
    for (let t = 0; t <= steps; t++) {
      const frac = t / steps;
      const twtVal = tMin + frac * (tMax - tMin);
      const py = frac * plotH;
      ctx.fillText((twtVal / 1000).toFixed(3), ML - 3, py + 4);
      ctx.strokeStyle = 'rgba(148,163,184,0.15)';
      ctx.lineWidth = 0.5; ctx.beginPath(); ctx.moveTo(ML, py); ctx.lineTo(W, py); ctx.stroke();
    }
    ctx.save(); ctx.translate(11, plotH / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Time (s)', 0, 0); ctx.restore();
  }, [tMin, tMax]);

  // ── Draw Spatial X-Axis ───────────────────────────────────────────────
  const drawSpatialAxis = useCallback((ctx, plotH, ML, plotW, W, H, xlStart, xlEnd) => {
    const MB = H - plotH;
    ctx.fillStyle = 'rgba(15,23,42,0.95)';
    ctx.fillRect(ML, plotH, plotW, MB);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
    const numTraces = xlEnd - xlStart + 1;

    const steps = 6;
    for (let t = 0; t <= steps; t++) {
      const frac = t / steps;
      const xlVal = Math.round(jMin + xlStart + frac * (numTraces - 1));
      const px = ML + frac * plotW;
      ctx.fillText(`XL ${xlVal}`, px, H - 10);
      ctx.strokeStyle = 'rgba(148,163,184,0.15)';
      ctx.lineWidth = 0.5; ctx.beginPath(); ctx.moveTo(px, plotH); ctx.lineTo(px, plotH + 4); ctx.stroke();
    }
    ctx.fillStyle = '#38bdf8'; ctx.font = 'bold 10px sans-serif';
    ctx.fillText(`Crossline (Inline ${iMin + wellIL})`, ML + plotW / 2, H - 24);
  }, [wellIL]);

  // ── Draw Well Log Overlay Strip in Center (Spans full height) ────────
  const drawWellLogOverlay = useCallback((ctx, plotH, ML, plotW, xlStart, xlEnd) => {
    if (!loggedSamples.length) return;

    const numTraces = xlEnd - xlStart + 1;
    const wellRelXL = wellXL - xlStart;
    if (wellRelXL < 0 || wellRelXL >= numTraces) return;

    const wx = ML + ((wellRelXL + 0.5) / numTraces) * plotW;
    const stripW = 32; // Wide & high resolution 32px log strip
    const x0 = Math.max(ML, Math.round(wx - stripW / 2));

    const propCfg = PROPERTIES.find(p => p.key === selectedProp);
    if (!propCfg) return;

    // ── 100% Automatic & Foolproof 3D Seismic Channel Horizon Extractor ───────
    // Automatically queries the 3D seismic volume trace at the well's exact location
    // to find the exact top (kFirst) and bottom (kLast) TWT of the seismic reflection structure.
    let kFirst = -1, kLast = -1;
    if (rawVol) {
      for (let k = 0; k < K_len; k++) {
        if (Math.abs(getSampleRaw(rawVol, rawScale, wellIL, wellXL, k)) > 1e-4) {
          if (kFirst === -1) kFirst = k;
          kLast = k;
        }
      }
    }

    const wellTMin = kFirst !== -1 ? (tStart + kFirst * dtMs) : 2180;
    const wellTMax = kLast  !== -1 ? (tStart + kLast  * dtMs) : 2320;

    const offscreen = document.createElement('canvas');
    offscreen.width = stripW;
    offscreen.height = plotH;
    const octx = offscreen.getContext('2d');
    const imgData = octx.createImageData(stripW, plotH);

    for (let py = 0; py < plotH; py++) {
      const twt = tMin + (py / Math.max(1, plotH - 1)) * (tMax - tMin);
      if (twt < wellTMin || twt > wellTMax) {
        for (let px = 0; px < stripW; px++) imgData.data[(py * stripW + px) * 4 + 3] = 0;
        continue;
      }

      const frac = Math.max(0, Math.min(1, (twt - wellTMin) / (wellTMax - wellTMin)));
      const floatIdx = frac * (loggedSamples.length - 1);
      const idx0 = Math.floor(floatIdx);
      const idx1 = Math.min(loggedSamples.length - 1, idx0 + 1);
      const f = floatIdx - idx0;

      const s0 = loggedSamples[idx0];
      const s1 = loggedSamples[idx1];
      const v0 = s0[`${selectedProp} (Act)`] ?? s0.seismic_amp ?? 0;
      const v1 = s1[`${selectedProp} (Act)`] ?? s1.seismic_amp ?? 0;
      const val = (1 - f) * v0 + f * v1;

      let r = 0, g = 0, b = 0, a = 240;
      if (val != null) {
        const rgb = getPropertyColor(val, activePredMeta || { min_val: propCfg.min, max_val: propCfg.max });
        r = rgb[0]; g = rgb[1]; b = rgb[2];
      }

      for (let px = 0; px < stripW; px++) {
        const edgeFadeX = Math.min(px, stripW - 1 - px) < 2 ? Math.min(px, stripW - 1 - px) / 2 : 1;
        const pi = (py * stripW + px) * 4;
        imgData.data[pi]   = r;
        imgData.data[pi+1] = g;
        imgData.data[pi+2] = b;
        imgData.data[pi+3] = Math.round(a * edgeFadeX);
      }
    }

    octx.putImageData(imgData, 0, 0);
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(offscreen, x0, 0);
    ctx.restore();

    ctx.save();
    ctx.shadowColor = '#fbbf24'; ctx.shadowBlur = 10;
    ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 2.0; ctx.setLineDash([]);
    ctx.strokeRect(x0, 0, stripW, plotH);
    ctx.shadowBlur = 0; ctx.restore();

    // Center guideline
    ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 1.0; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(wx, 0); ctx.lineTo(wx, plotH); ctx.stroke();
    ctx.setLineDash([]);

    // Well label pill at top center
    ctx.fillStyle = 'rgba(15,23,42,0.95)';
    ctx.beginPath();
    ctx.roundRect(wx - 36, 6, 72, 20, 4);
    ctx.fill();
    ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 1.2; ctx.strokeRect(wx - 36, 6, 72, 20);
    ctx.fillStyle = '#fbbf24'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(`▼ ${selectedWell}`, wx, 20);
  }, [selectedWell, selectedProp, wellXL, loggedSamples, tMin, tMax, activePredMeta]);

  // ── Main HD Canvas Render Loop ───────────────────────────────────────
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !rawVol) return;

    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    const ML = 42, MB = 36;
    const plotW = W - ML, plotH = H - MB;

    // Determine zoomed crossline window bounds around active well
    const xlMin = Math.max(0, wellXL - zoomRadius);
    const xlMax = Math.min(J_len - 1, wellXL + zoomRadius);
    const numTraces = xlMax - xlMin + 1;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, W, H);

    const cached = predVolCache.current[selectedProp];

    if (!cached || !cached.vol) {
      ctx.fillStyle = '#64748b'; ctx.font = '14px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(`Loading V11 ML ${selectedProp} High-Definition Prediction Volume…`, W / 2, H / 2 - 10);
      ctx.font = '11px sans-serif'; ctx.fillStyle = '#94a3b8';
      ctx.fillText('(Authentic V11 Model Inference)', W / 2, H / 2 + 10);
      drawTwtAxis(ctx, plotH, ML, W);
      drawSpatialAxis(ctx, plotH, ML, plotW, W, H, xlMin, xlMax);
      return;
    }

    const { vol, meta } = cached;

    // Calculate dynamic contrast bounds
    let localMin = Infinity, localMax = -Infinity;
    const sampleGrid = new Float32Array(numTraces * K_len);

    for (let k = 0; k < K_len; k++) {
      for (let tr = 0; tr < numTraces; tr++) {
        const j = xlMin + tr;
        const v = getSamplePred(vol, meta, wellIL, j, k);
        sampleGrid[k * numTraces + tr] = v;
        if (v < localMin) localMin = v;
        if (v > localMax) localMax = v;
      }
    }

    const minScale = contrastMode === 'local' ? localMin : meta.min_val;
    const maxScale = contrastMode === 'local' ? localMax : meta.max_val;

    // ── Precompute seismic horizon top (kFirst) and bottom (kLast) bounds ────
    const traceHorizonBounds = new Array(numTraces);
    if (rawVol) {
      for (let tr = 0; tr < numTraces; tr++) {
        const j = xlMin + tr;
        let kFirst = -1, kLast = -1;
        for (let k = 0; k < K_len; k++) {
          if (Math.abs(getSampleRaw(rawVol, rawScale, wellIL, j, k)) > 1e-4) {
            if (kFirst === -1) kFirst = k;
            kLast = k;
          }
        }
        traceHorizonBounds[tr] = { kFirst, kLast };
      }
    }

    const img = ctx.createImageData(plotW, plotH);

    // ── Vector Smooth Contours with Seismic Horizon Masking ───────────────
    for (let y = 0; y < plotH; y++) {
      const continuousK = kMin + (y / Math.max(1, plotH - 1)) * (kMax - kMin);
      const k0 = Math.floor(continuousK);
      const k1 = Math.min(K_len - 1, k0 + 1);
      const fy = continuousK - k0;

      for (let x = 0; x < plotW; x++) {
        const continuousTr = (x / Math.max(1, plotW - 1)) * (numTraces - 1);
        const tr0 = Math.floor(continuousTr);
        const tr1 = Math.min(numTraces - 1, tr0 + 1);
        const fx = continuousTr - tr0;

        const pi = (y * plotW + x) * 4;

        // Mask outside active seismic structure horizon
        const hb0 = traceHorizonBounds[tr0] || { kFirst: -1, kLast: -1 };
        const hb1 = traceHorizonBounds[tr1] || { kFirst: -1, kLast: -1 };
        if (hb0.kFirst !== -1 && hb1.kFirst !== -1) {
          const interpKFirst = (1 - fx) * hb0.kFirst + fx * hb1.kFirst;
          const interpKLast  = (1 - fx) * hb0.kLast  + fx * hb1.kLast;
          if (continuousK < interpKFirst - 1.5 || continuousK > interpKLast + 1.5) {
            img.data[pi]   = 255;
            img.data[pi+1] = 255;
            img.data[pi+2] = 255;
            img.data[pi+3] = 0; // Transparent background outside seismic structure!
            continue;
          }
        }

        let interpVal;
        if (useBilinear) {
          const v00 = sampleGrid[k0 * numTraces + tr0];
          const v10 = sampleGrid[k0 * numTraces + tr1];
          const v01 = sampleGrid[k1 * numTraces + tr0];
          const v11 = sampleGrid[k1 * numTraces + tr1];

          interpVal = (1 - fx) * (1 - fy) * v00 +
                      fx * (1 - fy) * v10 +
                      (1 - fx) * fy * v01 +
                      fx * fy * v11;
        } else {
          interpVal = sampleGrid[Math.round(continuousK) * numTraces + Math.round(continuousTr)];
        }

        const norm = Math.max(0, Math.min(1, (interpVal - minScale) / (maxScale - minScale || 1)));
        const [r, g, b] = turboColormap(norm);

        img.data[pi]   = r;
        img.data[pi+1] = g;
        img.data[pi+2] = b;
        img.data[pi+3] = 255;
      }
    }

    ctx.putImageData(img, ML, 0);

    // Draw grid axes
    drawTwtAxis(ctx, plotH, ML, W);
    drawSpatialAxis(ctx, plotH, ML, plotW, W, H, xlMin, xlMax);

    // Draw centered Well Log Overlay
    drawWellLogOverlay(ctx, plotH, ML, plotW, xlMin, xlMax);

    // Draw hover crosshair readout
    if (hoverCoord) {
      const gx = ML + ((hoverCoord.j - xlMin) / numTraces) * plotW;
      const gy = Math.max(0, Math.min(plotH, ((hoverCoord.k - kMin) / (kMax - kMin || 1)) * plotH));

      ctx.strokeStyle = 'rgba(251,191,36,0.85)'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ML, gy); ctx.lineTo(W, gy); ctx.stroke();
      ctx.setLineDash([]);

      const curVal = getSamplePred(vol, meta, wellIL, hoverCoord.j, hoverCoord.k);
      const propObj = PROPERTIES.find(p => p.key === selectedProp);
      const isPercentProp = ['VSH', 'SWE', 'PHIE', 'PHIT'].includes(selectedProp);
      const valStr = isPercentProp
        ? `${(curVal * 100).toFixed(1)}% (${curVal.toFixed(3)})`
        : `${curVal.toFixed(2)} ${propObj?.unit || ''}`;

      ctx.fillStyle = 'rgba(15,23,42,0.94)';
      ctx.fillRect(ML + 6, Math.max(gy - 24, 4), 220, 22);
      ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1;
      ctx.strokeRect(ML + 6, Math.max(gy - 24, 4), 220, 22);
      ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'left'; ctx.fillStyle = '#38bdf8';
      ctx.fillText(`XL ${jMin + hoverCoord.j} | TWT ${(tStart + hoverCoord.k * dtMs).toFixed(0)}ms: ${selectedProp} = ${valStr}`, ML + 10, Math.max(gy - 9, 18));
    }
  }, [rawVol, rawScale, selectedWell, selectedProp, zoomRadius, activePredMeta, wellIL, wellXL, getSampleRaw, getSamplePred, hoverCoord, drawTwtAxis, drawSpatialAxis, drawWellLogOverlay, kMin, kMax]);

  // Handle canvas mouse move
  const mouseMove = (e) => {
    if (!canvasRef.current || !rawVol) return;
    const cv = canvasRef.current;
    const rect = cv.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (cv.width / rect.width);
    const cy = (e.clientY - rect.top) * (cv.height / rect.height);
    const ML = 42, MB = 36;
    const plotW = cv.width - ML, plotH = cv.height - MB;

    if (cx < ML || cx > cv.width || cy > plotH) {
      setHoverCoord(null);
      return;
    }

    const xlMin = Math.max(0, wellXL - zoomRadius);
    const xlMax = Math.min(J_len - 1, wellXL + zoomRadius);
    const numTraces = xlMax - xlMin + 1;

    const fx = Math.max(0, Math.min(1, (cx - ML) / plotW));
    const fy = Math.max(0, Math.min(1, cy / plotH));

    const j = Math.min(xlMax, Math.max(xlMin, Math.floor(xlMin + fx * numTraces)));
    const kContinuous = kMin + fy * (kMax - kMin);
    const k = Math.min(K_len - 1, Math.max(0, Math.round(kContinuous)));

    setHoverCoord({ i: wellIL, j, k });
  };

  const activePropObj = PROPERTIES.find(p => p.key === selectedProp) || PROPERTIES[0];

  return (
    <div className="thinbed-workflow-container" style={{ height: 'calc(100vh - 120px)', overflowY: 'auto' }}>

      {/* Header bar */}
      <div className="workflow-title-bar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 10 }}>
            <Maximize2 size={20} color="#38bdf8" />
            HD Borehole Seismic Zoom Viewer
          </h2>
          <span className="slide-progress" style={{ color: '#38bdf8' }}>
            HIGH-DEFINITION BOREHOLE CENTERED ZOOM — {selectedWell} (INLINE {iMin + wellIL})
          </span>
        </div>
        {onSwitchTab && (
          <button
            onClick={() => onSwitchTab('ml_v11_predictor')}
            style={{
              fontSize: '11px', padding: '6px 14px', borderRadius: 6, cursor: 'pointer',
              background: 'rgba(56,189,248,0.2)', border: '1px solid #38bdf8', color: '#7dd3fc',
              fontWeight: 800, display: 'flex', alignItems: 'center', gap: 6
            }}
          >
            <Zap size={14} color="#38bdf8" />
            ⚡ Switch to Full V11 Predictor
          </button>
        )}
      </div>

      {/* ── Control Bar 1: Well Selector ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 14px', background: 'rgba(15,23,42,0.95)', border: '1px solid #334155', borderRadius: 8, marginBottom: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: '11px', fontWeight: 800, color: '#94a3b8', textTransform: 'uppercase' }}>Select Well:</span>

        {/* Blind Test Wells */}
        {['Z-04', 'Z-08-ST-02'].map(w => (
          <button
            key={w}
            onClick={() => setSelectedWell(w)}
            style={{
              fontSize: '11px', padding: '4px 12px', borderRadius: 6, cursor: 'pointer',
              border: `1px solid ${selectedWell === w ? '#ef4444' : '#7f1d1d'}`,
              background: selectedWell === w ? 'rgba(239,68,68,0.3)' : 'rgba(239,68,68,0.08)',
              color: selectedWell === w ? '#ffffff' : '#fca5a5', fontWeight: 800, display: 'flex', alignItems: 'center', gap: 5
            }}
          >
            🔥 {w} (BLIND)
          </button>
        ))}

        <span style={{ fontSize: '11px', color: '#475569', margin: '0 2px' }}>|</span>

        {/* Training Wells */}
        {['Z-02', 'Z-03', 'Z-05', 'Z-06', 'Z-07'].map(w => (
          <button
            key={w}
            onClick={() => setSelectedWell(w)}
            style={{
              fontSize: '10.5px', padding: '3px 10px', borderRadius: 5, cursor: 'pointer',
              border: `1px solid ${selectedWell === w ? '#3b82f6' : '#1e293b'}`,
              background: selectedWell === w ? 'rgba(59,130,246,0.3)' : 'transparent',
              color: selectedWell === w ? '#ffffff' : '#94a3b8',
              fontWeight: selectedWell === w ? 700 : 400
            }}
          >
            {w}
          </button>
        ))}

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          <ZoomIn size={14} color="#38bdf8" />
          <span style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 700 }}>Zoom Level:</span>

          {/* Quick Zoom Buttons */}
          <button
            onClick={() => setZoomRadius(14)}
            style={{
              fontSize: '10px', padding: '3px 8px', borderRadius: 4, cursor: 'pointer',
              border: `1px solid ${zoomRadius <= 20 ? '#38bdf8' : '#334155'}`,
              background: zoomRadius <= 20 ? 'rgba(56,189,248,0.25)' : 'transparent',
              color: zoomRadius <= 20 ? '#ffffff' : '#94a3b8', fontWeight: 700
            }}
          >🔍 Tight (±14)</button>

          <button
            onClick={() => setZoomRadius(50)}
            style={{
              fontSize: '10px', padding: '3px 8px', borderRadius: 4, cursor: 'pointer',
              border: `1px solid ${zoomRadius > 20 && zoomRadius <= 80 ? '#38bdf8' : '#334155'}`,
              background: zoomRadius > 20 && zoomRadius <= 80 ? 'rgba(56,189,248,0.25)' : 'transparent',
              color: zoomRadius > 20 && zoomRadius <= 80 ? '#ffffff' : '#94a3b8', fontWeight: 700
            }}
          >🔍 Medium (±50)</button>

          <button
            onClick={() => setZoomRadius(125)}
            style={{
              fontSize: '10px', padding: '3px 8px', borderRadius: 4, cursor: 'pointer',
              border: `1px solid ${zoomRadius > 80 ? '#38bdf8' : '#334155'}`,
              background: zoomRadius > 80 ? 'rgba(56,189,248,0.25)' : 'transparent',
              color: zoomRadius > 80 ? '#ffffff' : '#94a3b8', fontWeight: 700
            }}
          >🌐 Zoom Out (Full Seismic)</button>

          {/* Fine Slider */}
          <input
            type="range"
            min={10}
            max={125}
            step={5}
            value={zoomRadius}
            onChange={e => setZoomRadius(parseInt(e.target.value))}
            style={{ width: 85, accentColor: '#38bdf8', cursor: 'pointer', marginLeft: 4 }}
          />
        </div>
      </div>

      {/* ── Control Bar 2: Property Switcher ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', background: 'rgba(56,189,248,0.06)', border: '1px solid rgba(56,189,248,0.25)', borderRadius: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <Zap size={14} color="#38bdf8" />
        <span style={{ fontSize: '11px', fontWeight: 800, color: '#7dd3fc', textTransform: 'uppercase', marginRight: 4 }}>Select Property:</span>

        {PROPERTIES.map(p => (
          <button
            key={p.key}
            onClick={() => setSelectedProp(p.key)}
            style={{
              fontSize: '10.5px', padding: '4px 10px', borderRadius: 6, cursor: 'pointer',
              border: `1px solid ${selectedProp === p.key ? p.color : '#1e293b'}`,
              background: selectedProp === p.key ? `${p.color}33` : 'transparent',
              color: selectedProp === p.key ? '#ffffff' : '#94a3b8',
              fontWeight: selectedProp === p.key ? 800 : 400,
              boxShadow: selectedProp === p.key ? `0 0 8px ${p.color}44` : 'none'
            }}
          >
            {p.key}
          </button>
        ))}

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          {/* Interpolation Toggle */}
          <button
            onClick={() => setUseBilinear(!useBilinear)}
            style={{
              fontSize: '10.5px', padding: '3px 9px', borderRadius: 5, cursor: 'pointer',
              border: `1px solid ${useBilinear ? '#10b981' : '#64748b'}`,
              background: useBilinear ? 'rgba(16,185,129,0.2)' : 'transparent',
              color: useBilinear ? '#6ee7b7' : '#94a3b8', fontWeight: 700
            }}
          >
            {useBilinear ? '✨ Vector Smooth' : '⬛ Blocky Pixels'}
          </button>

          {/* Contrast Mode Toggle */}
          <button
            onClick={() => setContrastMode(contrastMode === 'local' ? 'global' : 'local')}
            style={{
              fontSize: '10.5px', padding: '3px 9px', borderRadius: 5, cursor: 'pointer',
              border: `1px solid ${contrastMode === 'local' ? '#fbbf24' : '#64748b'}`,
              background: contrastMode === 'local' ? 'rgba(251,191,36,0.2)' : 'transparent',
              color: contrastMode === 'local' ? '#fde68a' : '#94a3b8', fontWeight: 700
            }}
          >
            {contrastMode === 'local' ? '⚡ Dynamic Contrast' : '🌐 Global Scale'}
          </button>
        </div>
      </div>

      {/* ── Main Square Layout ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '650px 1fr', gap: 16, alignItems: 'start' }}>

        {/* Left: Square High-Definition Canvas Container */}
        <div style={{ background: '#090d16', border: '1px solid #1e293b', borderRadius: 10, padding: 12, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, padding: '0 4px', flexWrap: 'wrap', gap: 6 }}>
            <span style={{ fontSize: '12px', fontWeight: 800, color: '#e2e8f0' }}>
              🎯 Calibrated Sub-Volume Chunk ({selectedWell} Centered)
            </span>
            <span style={{ fontSize: '10.5px', color: '#38bdf8', fontWeight: 700, background: 'rgba(56,189,248,0.1)', padding: '2px 8px', borderRadius: 4, border: '1px solid rgba(56,189,248,0.25)' }}>
              Inline {iMin + wellIL} • XL {jMin + xlMin}–{jMin + xlMax} • TWT {tMin.toFixed(0)}–{tMax.toFixed(0)} ms (~{(tMin * 1.25).toFixed(0)}–{(tMax * 1.25).toFixed(0)}m)
            </span>
          </div>

          <div style={{ position: 'relative', width: 650, height: 650, background: '#0f172a', borderRadius: 8, overflow: 'hidden', border: '1px solid #334155' }}>
            <canvas
              ref={canvasRef}
              width={650}
              height={650}
              onMouseMove={mouseMove}
              onMouseLeave={() => setHoverCoord(null)}
              style={{ display: 'block', cursor: 'crosshair' }}
            />
          </div>

          {/* Color Bar Scale Legend */}
          <div style={{ width: '100%', marginTop: 10, padding: '6px 12px', background: 'rgba(15,23,42,0.8)', borderRadius: 6, border: '1px solid #1e293b', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700 }}>Color Scale ({activePropObj.unit || 'norm'}):</span>
            <span style={{ fontSize: '10px', color: '#64748b' }}>{activePropObj.min}</span>
            <div style={{
              flex: 1, height: 10, borderRadius: 3,
              background: 'linear-gradient(to right, rgb(8,48,107), rgb(8,81,156), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37), rgb(253,141,60), rgb(227,26,28), rgb(128,0,38))'
            }} />
            <span style={{ fontSize: '10px', color: '#64748b' }}>{activePropObj.max}</span>
          </div>
        </div>

        {/* Right: Detailed Well & Property Information Card */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Card 1: Well Details */}
          <div className="glass-card" style={{ padding: 16 }}>
            <div className="card-header" style={{ marginBottom: 10 }}>
              <Target size={18} color="#38bdf8" />
              <h3 style={{ margin: 0, fontSize: '14px', color: '#e2e8f0' }}>Borehole Metadata — {selectedWell}</h3>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: '12px' }}>
              <div style={{ background: 'rgba(15,23,42,0.6)', padding: '8px 10px', borderRadius: 6, border: '1px solid #1e293b' }}>
                <span style={{ color: '#64748b', fontSize: '10px', display: 'block' }}>INLINE INDEX</span>
                <strong style={{ color: '#38bdf8' }}>Inline {iMin + wellIL}</strong> (Local #{wellIL})
              </div>
              <div style={{ background: 'rgba(15,23,42,0.6)', padding: '8px 10px', borderRadius: 6, border: '1px solid #1e293b' }}>
                <span style={{ color: '#64748b', fontSize: '10px', display: 'block' }}>CROSSLINE INDEX</span>
                <strong style={{ color: '#38bdf8' }}>Crossline {jMin + wellXL}</strong> (Local #{wellXL})
              </div>
              <div style={{ background: 'rgba(15,23,42,0.6)', padding: '8px 10px', borderRadius: 6, border: '1px solid #1e293b' }}>
                <span style={{ color: '#64748b', fontSize: '10px', display: 'block' }}>WELL TYPE</span>
                <strong style={{ color: (selectedWell === 'Z-04' || selectedWell === 'Z-08-ST-02') ? '#fca5a5' : '#86efac' }}>
                  {selectedWell === 'Z-04' || selectedWell === 'Z-08-ST-02' ? '🔥 Blind Test Well' : '✓ Training Well'}
                </strong>
              </div>
              <div style={{ background: 'rgba(15,23,42,0.6)', padding: '8px 10px', borderRadius: 6, border: '1px solid #1e293b' }}>
                <span style={{ color: '#64748b', fontSize: '10px', display: 'block' }}>KB ELEVATION</span>
                <strong style={{ color: '#e2e8f0' }}>{activeWellData.kb} m</strong>
              </div>
            </div>
          </div>

          {/* Card 2: Property Description */}
          <div className="glass-card" style={{ padding: 16 }}>
            <div className="card-header" style={{ marginBottom: 8 }}>
              <Zap size={18} color={activePropObj.color} />
              <h3 style={{ margin: 0, fontSize: '14px', color: activePropObj.color }}>{activePropObj.name}</h3>
            </div>
            <p style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.5, margin: 0 }}>
              {activePropObj.desc}
            </p>
          </div>

          {/* Card 3: Instructions & Advantages */}
          <div className="glass-card" style={{ padding: 16, background: 'rgba(56,189,248,0.03)', border: '1px solid rgba(56,189,248,0.2)' }}>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '13px', color: '#38bdf8' }}>💡 Why Use Square Zoom Viewer?</h4>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: '11px', color: '#cbd5e1', lineHeight: 1.6 }}>
              <li><strong>High Pixel Density:</strong> In standard views, the well trace is compressed into just ~2.5 pixels width. Here, each trace spans ~15-25 pixels!</li>
              <li><strong>Perfect Borehole Centering:</strong> The borehole log overlay stays dead-center in the square canvas while you switch properties.</li>
              <li><strong>Direct Visual Comparison:</strong> Easily compare surrounding seismic reflection geometry with the exact borehole log response.</li>
            </ul>
          </div>

        </div>

      </div>

    </div>
  );
}
