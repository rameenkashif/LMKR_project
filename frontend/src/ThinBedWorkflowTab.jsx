import React, { useState, useEffect, useRef } from 'react';
import { Activity, Layers, Info, MapPin } from 'lucide-react';

// ── Ormsby Whitening algorithm in JS ──
const whitenTraceJS = (trace, dtMs) => {
  const N = trace.length;
  const dt = dtMs / 1000.0;
  const samplingFreq = 1 / dt;

  const X = [];
  for (let k = 0; k < N; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += trace[n] * Math.cos(angle);
      im -= trace[n] * Math.sin(angle);
    }
    X.push({ re, im });
  }

  const amp = X.map(c => Math.sqrt(c.re * c.re + c.im * c.im));
  const smoothAmp = new Array(N).fill(0);
  const w = 5;
  for (let i = 0; i < N; i++) {
    let sum = 0, count = 0;
    for (let j = -w; j <= w; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < N) {
        sum += amp[idx];
        count++;
      }
    }
    smoothAmp[i] = sum / count;
  }

  let maxSmooth = 0;
  for (let i = 0; i < N; i++) {
    if (smoothAmp[i] > maxSmooth) maxSmooth = smoothAmp[i];
  }
  const waterLevel = 0.05 * maxSmooth;

  const f1 = 8.0, f2 = 12.0, f3 = 55.0, f4 = 70.0;
  for (let k = 0; k < N; k++) {
    const freq = (k <= N / 2) ? (k * samplingFreq) / N : ((N - k) * samplingFreq) / N;
    let bp = 0.0;
    if (freq >= f1 && freq <= f4) {
      if (freq <= f2) {
        bp = 0.5 * (1 - Math.cos(Math.PI * (freq - f1) / (f2 - f1)));
      } else if (freq <= f3) {
        bp = 1.0;
      } else {
        bp = 0.5 * (1 + Math.cos(Math.PI * (freq - f3) / (f4 - f3)));
      }
    }
    const den = Math.max(smoothAmp[k], waterLevel);
    const gain = bp / den;
    X[k].re *= gain;
    X[k].im *= gain;
  }

  const filtered = [];
  for (let n = 0; n < N; n++) {
    let sum = 0;
    for (let k = 0; k < N; k++) {
      const angle = (2 * Math.PI * k * n) / N;
      sum += X[k].re * Math.cos(angle) - X[k].im * Math.sin(angle);
    }
    filtered.push(sum / N);
  }
  return filtered;
};

// ── Sliding STFT-based Mean Frequency in JS ──
const computeSlidingMeanFreq = (trace, dtMs) => {
  const N = trace.length;
  const dt = dtMs / 1000.0;
  const samplingFreq = 1.0 / dt;
  const meanFreqs = new Array(N).fill(0);
  
  const halfWin = 10; // 21-sample sliding window
  for (let i = 0; i < N; i++) {
    const start = Math.max(0, i - halfWin);
    const end = Math.min(N - 1, i + halfWin);
    const winLen = end - start + 1;
    const subTrace = trace.slice(start, end + 1);
    
    let num = 0, den = 0;
    for (let k = 0; k <= winLen / 2; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < winLen; n++) {
        const angle = (2 * Math.PI * k * n) / winLen;
        re += subTrace[n] * Math.cos(angle);
        im -= subTrace[n] * Math.sin(angle);
      }
      const freq = (k * samplingFreq) / winLen;
      const amp = Math.sqrt(re * re + im * im);
      num += freq * amp;
      den += amp;
    }
    meanFreqs[i] = den > 0 ? (num / den) : 0;
  }
  return meanFreqs;
};

// ── Frequency Rainbow Colormap: Blue (Low Freq) -> Red (High Freq) ──
const getFreqColor = (freq) => {
  const norm = Math.max(0, Math.min(1, (freq - 12) / 40)); // focus color range between 12-52Hz
  let r = 0, g = 0, b = 0;
  if (norm < 0.25) {
    const f = norm / 0.25;
    r = 0; g = Math.round(255 * f); b = 255;
  } else if (norm < 0.5) {
    const f = (norm - 0.25) / 0.25;
    r = 0; g = 255; b = Math.round(255 * (1 - f));
  } else if (norm < 0.75) {
    const f = (norm - 0.5) / 0.25;
    r = Math.round(255 * f); g = 255; b = 0;
  } else {
    const f = (norm - 0.75) / 0.25;
    r = 255; g = Math.round(255 * (1 - f)); b = 0;
  }
  return `rgb(${r}, ${g}, ${b})`;
};

export default function ThinBedWorkflowTab({ wellsData }) {
  const [arbLineWells, setArbLineWells] = useState(['Z-03', 'Z-07', 'Z-04']);
  const [showWiggles, setShowWiggles] = useState(true);
  const [hoveredData, setHoveredData] = useState(null);
  const [whitenedWells, setWhitenedWells] = useState({});

  const freqCanvasRef = useRef(null);
  const seismicCanvasRef = useRef(null);

  // ── 1. Precompute Whitened traces and Sliding Frequency profiles ──
  useEffect(() => {
    const updated = {};
    Object.keys(wellsData).forEach(wName => {
      const well = wellsData[wName];
      const rawTrace = well.samples.map(s => s.seismic_amp);
      const whitened = whitenTraceJS(rawTrace, 2.0);
      const slidingFreq = computeSlidingMeanFreq(rawTrace, 2.0);

      let maxRaw = 0, maxWhite = 0;
      for (let idx = 0; idx < rawTrace.length; idx++) {
        if (Math.abs(rawTrace[idx]) > maxRaw) maxRaw = Math.abs(rawTrace[idx]);
        if (Math.abs(whitened[idx]) > maxWhite) maxWhite = Math.abs(whitened[idx]);
      }
      const scale = maxWhite > 0 ? (maxRaw / maxWhite) : 1.0;

      updated[wName] = {
        ...well,
        samples: well.samples.map((s, idx) => ({
          ...s,
          seismic_amp_whitened: whitened[idx] * scale,
          sliding_freq: slidingFreq[idx]
        }))
      };
    });
    setWhitenedWells(updated);
  }, [wellsData]);

  // ── 2. Canvas Drawing logic ──
  useEffect(() => {
    if (!freqCanvasRef.current || !seismicCanvasRef.current || arbLineWells.length < 2) return;

    const fCanvas = freqCanvasRef.current;
    const fCtx = fCanvas.getContext('2d');
    const sCanvas = seismicCanvasRef.current;
    const sCtx = sCanvas.getContext('2d');

    const W = fCanvas.width;
    const H = fCanvas.height;

    // Reset transforms to avoid leaks
    fCtx.setTransform(1, 0, 0, 1, 0, 0);
    sCtx.setTransform(1, 0, 0, 1, 0, 0);

    // Margins for axes
    const offsetX = 55;
    const rightMargin = 20;
    const offsetY = 20;
    const bottomOffsetY = 40;
    const plotW = W - offsetX - rightMargin;
    const plotH = H - offsetY - bottomOffsetY;

    const nPanels = arbLineWells.length - 1;
    const colsPerPanel = Math.floor(plotW / nPanels);

    // Max trace length (TWT samples)
    let maxLen = 0;
    arbLineWells.forEach(w => {
      if (wellsData[w]) maxLen = Math.max(maxLen, wellsData[w].samples.length);
    });

    if (maxLen === 0) return;

    // ── DRAWING FUNCTION FOR EITHER CANVAS ──
    const drawSection = (ctx, drawType) => {
      // Background Clear
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);

      // Pass 1: Draw Color Density Background
      for (let p = 0; p < nPanels; p++) {
        const wellA = whitenedWells[arbLineWells[p]] || wellsData[arbLineWells[p]];
        const wellB = whitenedWells[arbLineWells[p+1]] || wellsData[arbLineWells[p+1]];
        if (!wellA || !wellB) continue;

        const startX = offsetX + p * colsPerPanel;
        const endX = offsetX + (p + 1) * colsPerPanel;

        for (let x = startX; x < endX; x++) {
          const frac = (x - startX) / colsPerPanel;
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sIdx = Math.floor(fracY * maxLen);
            const idx = Math.min(sIdx, wellA.samples.length - 1);

            const sA = wellA.samples[idx];
            const sB = wellB.samples[idx];
            if (!sA || !sB) continue;

            let color = '';
            if (drawType === 'frequency') {
              const valA = sA.sliding_freq || 25;
              const valB = sB.sliding_freq || 25;
              const val = (1 - frac) * valA + frac * valB;
              color = getFreqColor(val);
            } else {
              // Seismic Density (Red-White-Blue)
              const valA = sA.seismic_amp_whitened ?? sA.seismic_amp ?? 0;
              const valB = sB.seismic_amp_whitened ?? sB.seismic_amp ?? 0;
              const val = (1 - frac) * valA + frac * valB;
              const norm = Math.max(-1, Math.min(1, val / 15000));
              if (norm > 0) {
                color = `rgb(255, ${Math.round(255 * (1 - norm))}, ${Math.round(255 * (1 - norm))})`;
              } else {
                const absN = Math.abs(norm);
                color = `rgb(${Math.round(255 * (1 - absN))}, ${Math.round(255 * (1 - absN))}, 255)`;
              }
            }

            ctx.fillStyle = color;
            ctx.fillRect(x, y, 1, 1);
          }
        }
      }

      // Pass 2: Draw Seismic Wiggles (Right Canvas, or Left Canvas if showWiggles is forced)
      if (showWiggles || drawType === 'seismic') {
        const nWiggles = 24;
        const spacing = plotW / nWiggles;

        for (let w = 1; w < nWiggles; w++) {
          const wiggleX = offsetX + w * spacing;
          const frac = (wiggleX - offsetX) / plotW;
          const p = Math.floor(frac * nPanels);
          const pFrac = (frac - p / nPanels) * nPanels;

          const wellA = whitenedWells[arbLineWells[p]] || wellsData[arbLineWells[p]];
          const wellB = whitenedWells[arbLineWells[p+1]] || wellsData[arbLineWells[p+1]];
          if (!wellA || !wellB) continue;

          // Positive lobe shading
          ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
          ctx.beginPath();
          ctx.moveTo(wiggleX, offsetY);
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sIdx = Math.min(Math.floor(fracY * maxLen), wellA.samples.length - 1);
            const valA = wellA.samples[sIdx]?.seismic_amp_whitened ?? wellA.samples[sIdx]?.seismic_amp ?? 0;
            const valB = wellB.samples[sIdx]?.seismic_amp_whitened ?? wellB.samples[sIdx]?.seismic_amp ?? 0;
            const val = (1 - pFrac) * valA + pFrac * valB;
            const norm = val / 25000;
            const dx = Math.max(0, norm * (spacing * 0.65));
            ctx.lineTo(wiggleX + dx, y);
          }
          ctx.lineTo(wiggleX, offsetY + plotH);
          ctx.closePath();
          ctx.fill();

          // Waveform line
          ctx.strokeStyle = drawType === 'frequency' ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.7)';
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sIdx = Math.min(Math.floor(fracY * maxLen), wellA.samples.length - 1);
            const valA = wellA.samples[sIdx]?.seismic_amp_whitened ?? wellA.samples[sIdx]?.seismic_amp ?? 0;
            const valB = wellB.samples[sIdx]?.seismic_amp_whitened ?? wellB.samples[sIdx]?.seismic_amp ?? 0;
            const val = (1 - pFrac) * valA + pFrac * valB;
            const norm = val / 25000;
            const dx = norm * (spacing * 0.65);
            if (y === offsetY) ctx.moveTo(wiggleX + dx, y);
            else ctx.lineTo(wiggleX + dx, y);
          }
          ctx.stroke();
        }
      }

      // Pass 3: Draw Geological Horizons (Red Peak & Tensleep)
      const drawHorizon = (yFrac, name, color) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        for (let x = offsetX; x < offsetX + plotW; x++) {
          const frac = (x - offsetX) / plotW;
          const dip = Math.sin(frac * Math.PI * 2) * 11 + Math.cos(frac * Math.PI * 4) * 4;
          const y = offsetY + yFrac * plotH + dip;
          if (x === offsetX) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Label at the right edge
        ctx.fillStyle = '#fbbf24';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(name, offsetX + plotW - 12, offsetY + yFrac * plotH - 6);
        ctx.restore();
      };

      drawHorizon(0.42, "Red Peak", "rgba(251, 191, 36, 0.95)");
      drawHorizon(0.68, "Tensleep", "rgba(168, 85, 247, 0.95)");

      // Pass 4: Axes
      ctx.fillStyle = '#64748b';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      const tStart = wellsData[arbLineWells[0]].samples[0].time;
      const tEnd = wellsData[arbLineWells[0]].samples[wellsData[arbLineWells[0]].samples.length - 1].time;

      for (let t = Math.ceil(tStart / 50) * 50; t <= tEnd; t += 50) {
        const frac = (t - tStart) / (tEnd - tStart);
        const y = offsetY + frac * plotH;
        ctx.beginPath();
        ctx.moveTo(offsetX - 4, y);
        ctx.lineTo(offsetX, y);
        ctx.stroke();
        ctx.fillText(`${t} ms`, offsetX - 8, y);
      }

      // Vertical Borehole Lines
      arbLineWells.forEach((wName, idx) => {
        const xPos = offsetX + (idx / nPanels) * plotW;
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.moveTo(xPos, offsetY);
        ctx.lineTo(xPos, offsetY + plotH);
        ctx.stroke();

        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(wName, xPos, offsetY - 6);

        ctx.fillStyle = '#475569';
        ctx.fillText(`IL ${wellsData[wName].inline}`, xPos, offsetY + plotH + 12);
        ctx.fillText(`XL ${wellsData[wName].crossline}`, xPos, offsetY + plotH + 22);
      });
    };

    drawSection(fCtx, 'frequency');
    drawSection(sCtx, 'seismic');
  }, [arbLineWells, whitenedWells, showWiggles]);

  // ── 3. Hover Coordinate Tracking ──
  const handleMouseMove = (e, canvasRefObj, drawType) => {
    if (!canvasRefObj.current || arbLineWells.length < 2) return;
    const canvas = canvasRefObj.current;
    const rect = canvas.getBoundingClientRect();
    
    const offsetX = 55;
    const rightMargin = 20;
    const offsetY = 20;
    const bottomOffsetY = 40;
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;
    
    const plotW = canvas.width - offsetX - rightMargin;
    const plotH = canvas.height - offsetY - bottomOffsetY;
    
    if (clickX < offsetX || clickX > canvas.width - rightMargin || clickY < offsetY || clickY > canvas.height - bottomOffsetY) {
      setHoveredData(null);
      return;
    }
    
    const fracX = (clickX - offsetX) / plotW;
    const fracY = (clickY - offsetY) / plotH;
    
    const nPanels = arbLineWells.length - 1;
    const p = Math.min(nPanels - 1, Math.floor(fracX * nPanels));
    const pFrac = (fracX - p / nPanels) * nPanels;
    
    const wellA = wellsData[arbLineWells[p]];
    const wellB = wellsData[arbLineWells[p+1]];
    if (!wellA || !wellB) return;
    
    const il = Math.round(wellA.inline + pFrac * (wellB.inline - wellA.inline));
    const xl = Math.round(wellA.crossline + pFrac * (wellB.crossline - wellA.crossline));
    
    const tStart = wellA.samples[0].time;
    const tEnd = wellA.samples[wellA.samples.length - 1].time;
    const twt = tStart + fracY * (tEnd - tStart);
    
    // Retrieve value
    const wDataA = whitenedWells[wellA.name] || wellA;
    const wDataB = whitenedWells[wellB.name] || wellB;
    const sampleIdx = Math.floor(fracY * wellA.samples.length);
    const idx = Math.min(sampleIdx, wellA.samples.length - 1);
    
    let displayVal = '';
    if (drawType === 'frequency') {
      const valA = wDataA.samples[idx]?.sliding_freq || 0;
      const valB = wDataB.samples[idx]?.sliding_freq || 0;
      const val = (1 - pFrac) * valA + pFrac * valB;
      displayVal = `Mean Freq: ${val.toFixed(1)} Hz`;
    } else {
      const valA = wDataA.samples[idx]?.seismic_amp_whitened ?? wDataA.samples[idx]?.seismic_amp ?? 0;
      const valB = wDataB.samples[idx]?.seismic_amp_whitened ?? wDataB.samples[idx]?.seismic_amp ?? 0;
      const val = (1 - pFrac) * valA + pFrac * valB;
      displayVal = `Whitened Amp: ${val.toFixed(0)}`;
    }
    
    setHoveredData({ il, xl, twt, valueText: displayVal });
  };

  const toggleWellInPath = (wName) => {
    if (arbLineWells.includes(wName)) {
      if (arbLineWells.length > 2) {
        setArbLineWells(arbLineWells.filter(w => w !== wName));
      }
    } else {
      setArbLineWells([...arbLineWells, wName]);
    }
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'linear-gradient(160deg, #060b18 0%, #0d1526 50%, #060b18 100%)',
      color: '#e2e8f0', fontFamily: 'Inter, sans-serif', overflow: 'hidden'
    }}>
      
      {/* ── Header ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '14px',
        padding: '10px 20px', flexShrink: 0,
        background: 'rgba(6,11,24,0.95)',
        borderBottom: '1px solid rgba(99,102,241,0.2)',
      }}>
        <Layers size={20} color='#818cf8' />
        <div>
          <div style={{ fontWeight: 700, fontSize: '14px', letterSpacing: '0.05em', color: '#c7d2fe' }}>
            THIN BED WORKFLOW — FREQUENCY BASED ANALYSIS & WHITENING
          </div>
          <div style={{ fontSize: '10px', color: '#475569', marginTop: '1px' }}>
            Continuous sliding STFT frequency mapping vs Ormsby whitened seismic cross-section.
          </div>
        </div>

        {/* Path Selection indicator */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: '#94a3b8', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showWiggles}
              onChange={(e) => setShowWiggles(e.target.checked)}
              style={{ cursor: 'pointer', accentColor: '#10b981' }}
            />
            Show Wiggle Contours
          </label>
        </div>
      </div>

      {/* ── Active Path Controls ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '8px',
        padding: '8px 20px', flexShrink: 0,
        background: 'rgba(6,11,24,0.8)',
        borderBottom: '1px solid rgba(99,102,241,0.12)',
        flexWrap: 'wrap'
      }}>
        <span style={{ fontSize: '10px', color: '#475569', fontWeight: 600, letterSpacing: '0.07em' }}>CONSTRUCT LINE:</span>
        {Object.keys(wellsData).map(wName => {
          const selected = arbLineWells.includes(wName);
          const isBlind = wName === 'Z-04';
          return (
            <button
              key={wName}
              onClick={() => toggleWellInPath(wName)}
              style={{
                padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 600,
                cursor: 'pointer', border: `1px solid ${selected ? 'var(--accent-gold)' : '#334155'}`,
                background: selected ? 'rgba(251,191,36,0.12)' : 'rgba(15,23,42,0.7)',
                color: selected ? 'var(--accent-gold)' : '#64748b'
              }}
            >
              {wName}{isBlind ? ' ★' : ''}
            </button>
          );
        })}
        <div style={{ marginLeft: 'auto', fontSize: '10.5px', color: '#94a3b8' }}>
          Active Path: <strong>{arbLineWells.join(' ➔ ')}</strong>
        </div>
      </div>

      {/* ── Split-Screen Canvases ── */}
      <div style={{
        flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr',
        padding: '12px', gap: '12px', minHeight: 0
      }}>
        
        {/* Left: Frequency Tuning Analysis */}
        <div style={{ display: 'flex', flexDirection: 'column', position: 'relative' }}>
          <div style={{ position: 'absolute', top: '10px', left: '60px', zIndex: 10, background: 'rgba(15,23,42,0.85)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(99,102,241,0.3)', fontSize: '11px', fontWeight: 'bold', color: '#c7d2fe' }}>
            Part I: Frequency Based Analysis
          </div>
          <canvas
            ref={freqCanvasRef}
            width={600}
            height={390}
            onMouseMove={(e) => handleMouseMove(e, freqCanvasRef, 'frequency')}
            onMouseLeave={() => setHoveredData(null)}
            style={{ flex: 1, borderRadius: '8px', border: '1px solid rgba(99,102,241,0.2)', cursor: 'crosshair', display: 'block' }}
          />
        </div>

        {/* Right: Spectrally Whitened Seismic */}
        <div style={{ display: 'flex', flexDirection: 'column', position: 'relative' }}>
          <div style={{ position: 'absolute', top: '10px', left: '60px', zIndex: 10, background: 'rgba(15,23,42,0.85)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(99,102,241,0.3)', fontSize: '11px', fontWeight: 'bold', color: '#c7d2fe' }}>
            Part II: Whitened Seismic Section
          </div>
          <canvas
            ref={seismicCanvasRef}
            width={600}
            height={390}
            onMouseMove={(e) => handleMouseMove(e, seismicCanvasRef, 'seismic')}
            onMouseLeave={() => setHoveredData(null)}
            style={{ flex: 1, borderRadius: '8px', border: '1px solid rgba(99,102,241,0.2)', cursor: 'crosshair', display: 'block' }}
          />
        </div>

      </div>

      {/* ── Legends & Status Footer ── */}
      <div style={{
        padding: '8px 20px', flexShrink: 0,
        background: 'rgba(6,11,24,0.6)',
        borderTop: '1px solid rgba(30,58,95,0.4)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: '10px'
      }}>
        
        {/* Frequency Legend */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '9.5px', color: '#64748b', fontWeight: 700 }}>TUNING FREQUENCY:</span>
          <span style={{ fontSize: '9px', color: '#60a5fa' }}>Low Freq (12Hz)</span>
          <div style={{
            width: 140, height: 10, borderRadius: 2,
            background: 'linear-gradient(to right, rgb(0,0,255) 0%, rgb(0,255,255) 25%, rgb(0,255,0) 50%, rgb(255,255,0) 75%, rgb(255,0,0) 100%)',
            border: '1px solid rgba(255,255,255,0.1)'
          }} />
          <span style={{ fontSize: '9px', color: '#f87171' }}>High Freq (52Hz)</span>
        </div>

        {/* Geological Markers */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '10px', color: '#94a3b8' }}>
          <span style={{ fontWeight: 700, fontSize: '9.5px', color: '#64748b' }}>HORIZONS:</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '12px', height: '0px', borderTop: '2px dashed #fbbf24' }}></span> Red Peak
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '12px', height: '0px', borderTop: '2px dashed #a855f7' }}></span> Tensleep
          </div>
        </div>

        {/* Live Coordinate Overlay */}
        <div style={{ marginLeft: 'auto', minWidth: '240px', textAlign: 'right' }}>
          {hoveredData ? (
            <span style={{ fontSize: '11px', color: '#c7d2fe', fontWeight: 'bold' }}>
              IL: {hoveredData.il} · XL: {hoveredData.xl} · TWT: {hoveredData.twt.toFixed(1)} ms · {hoveredData.valueText}
            </span>
          ) : (
            <span style={{ fontSize: '10.5px', color: '#475569', fontStyle: 'italic' }}>
              Hover over either canvas to map coordinates and values...
            </span>
          )}
        </div>

      </div>

    </div>
  );
}
