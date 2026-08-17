import React, { useState, useEffect, useRef } from 'react';
import { AreaChart, Layers, Sliders, Cpu, Activity, Info, BarChart } from 'lucide-react';
import { wellsData } from './data';

export default function CwtSwtTab() {
  const [selectedWell, setSelectedWell] = useState('Z-04');
  const [waveletType, setWaveletType] = useState('Ricker');
  const [cwtFreqSteps, setCwtFreqSteps] = useState(25);
  const [swtThreshold, setSwtThreshold] = useState(0.08); // SWT denoising threshold
  const [activeSwtSubTab, setActiveSwtSubTab] = useState('decomposition'); // 'decomposition' or 'denoising'
  const [sliceFreq, setSliceFreq] = useState(30); // CWT frequency slice in Hz
  
  // Tuning calculator states
  const [velocity, setVelocity] = useState(3200); // m/s
  const [tuningFreq, setTuningFreq] = useState(30); // Hz

  const canvasRef = useRef(null);

  // Extract trace for selected well and normalize to range [-1.0, 1.0] for scaling
  const well = wellsData[selectedWell];
  const samples = well ? well.samples : [];
  const rawTraceValues = samples.map(s => s.seismic_amp || 0);
  const maxAmp = Math.max(...rawTraceValues.map(Math.abs), 0.0001);
  const rawTrace = rawTraceValues.map(v => v / maxAmp);
  const times = samples.map(s => s.time || 0);
  const dtMs = times.length > 1 ? times[1] - times[0] : 2;

  // 1. Interactive Tuning Thickness Calculation
  // Tuning thickness = wavelength / 4 = Velocity / (4 * Frequency)
  const wavelength = velocity / tuningFreq;
  const tuningThickness = wavelength / 4;

  // 2. JavaScript CWT & SSWT Calculations (Real-time Dual-Pane)
  useEffect(() => {
    if (!canvasRef.current || rawTrace.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Clear background
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, W, H);

    const N = rawTrace.length;
    const dtS = dtMs / 1000.0;

    const minFreq = 10;
    const maxFreq = 60;
    const freqs = [];
    for (let i = 0; i < cwtFreqSteps; i++) {
      freqs.push(minFreq + (maxFreq - minFreq) * (i / (cwtFreqSteps - 1)));
    }

    // 1. Compute CWT & SSWT
    const cwtAmp = [];
    const cwtReal = [];
    const cwtImag = [];
    
    freqs.forEach(freq => {
      const rowAmp = new Array(N).fill(0);
      const rowReal = new Array(N).fill(0);
      const rowImag = new Array(N).fill(0);
      
      const duration = 4.0 / freq;
      const tHalf = duration / 2;
      const waveletSamples = [];
      for (let t = -tHalf; t <= tHalf; t += dtS) {
        waveletSamples.push(t);
      }
      
      const sigma = 1.0 / (2.0 * Math.PI * freq) * 2.0;
      
      // Determine left pane wavelet (Ricker or Morlet)
      let wLeft = [];
      if (waveletType === 'Ricker') {
        wLeft = waveletSamples.map(t => {
          const term = Math.PI * freq * t;
          return (1 - 2 * term * term) * Math.exp(-term * term);
        });
      } else {
        wLeft = waveletSamples.map(t => Math.exp(-0.5 * (t / sigma) ** 2) * Math.cos(2.0 * Math.PI * freq * t));
      }
      const energyLeft = Math.sqrt(wLeft.reduce((sum, val) => sum + val * val, 0));
      const nwLeft = wLeft.map(w => w / (energyLeft || 1));
      
      // Complex Morlet for SSWT
      const wReal = waveletSamples.map(t => Math.exp(-0.5 * (t / sigma) ** 2) * Math.cos(2.0 * Math.PI * freq * t));
      const wImag = waveletSamples.map(t => Math.exp(-0.5 * (t / sigma) ** 2) * Math.sin(2.0 * Math.PI * freq * t));
      const energyRight = Math.sqrt(wReal.reduce((sum, val) => sum + val * val, 0));
      const nwReal = wReal.map(w => w / (energyRight || 1));
      const nwImag = wImag.map(w => w / (energyRight || 1));
      
      const halfM = Math.floor(nwLeft.length / 2);
      for (let n = 0; n < N; n++) {
        let sumLeft = 0;
        let sumReal = 0;
        let sumImag = 0;
        
        for (let k = 0; k < nwLeft.length; k++) {
          const idx = n - halfM + k;
          if (idx >= 0 && idx < N) {
            sumLeft += rawTrace[idx] * nwLeft[k];
            sumReal += rawTrace[idx] * nwReal[k];
            sumImag += rawTrace[idx] * nwImag[k];
          }
        }
        rowAmp[n] = Math.abs(sumLeft);
        rowReal[n] = sumReal;
        rowImag[n] = -sumImag; // Complex Conjugate (minus sign)
      }
      cwtAmp.push(rowAmp);
      cwtReal.push(rowReal);
      cwtImag.push(rowImag);
    });

    // 2. Compute CWT Time Derivative
    const cwtDiffReal = [];
    const cwtDiffImag = [];
    for (let f = 0; f < cwtFreqSteps; f++) {
      const rowDiffReal = new Array(N).fill(0);
      const rowDiffImag = new Array(N).fill(0);
      for (let n = 0; n < N; n++) {
        const prevIdx = n > 0 ? n - 1 : 0;
        const nextIdx = n < N - 1 ? n + 1 : N - 1;
        const stepS = dtS * (nextIdx - prevIdx);
        rowDiffReal[n] = (cwtReal[f][nextIdx] - cwtReal[f][prevIdx]) / (stepS || 0.001);
        rowDiffImag[n] = (cwtImag[f][nextIdx] - cwtImag[f][prevIdx]) / (stepS || 0.001);
      }
      cwtDiffReal.push(rowDiffReal);
      cwtDiffImag.push(rowDiffImag);
    }

    // 3. Compute Instantaneous Frequency
    const instFreq = [];
    for (let f = 0; f < cwtFreqSteps; f++) {
      const rowFreq = new Array(N).fill(0);
      for (let n = 0; n < N; n++) {
        const c = cwtReal[f][n];
        const d = cwtImag[f][n];
        const a = cwtDiffReal[f][n];
        const b = cwtDiffImag[f][n];
        const denom = c * c + d * d;
        const imPart = denom > 1e-9 ? (b * c - a * d) / denom : 0;
        rowFreq[n] = imPart / (2 * Math.PI);
      }
      instFreq.push(rowFreq);
    }

    // Find max value in CWT for normalization & masking
    let maxCwtVal = 0.001;
    for (let f = 0; f < cwtFreqSteps; f++) {
      for (let n = 0; n < N; n++) {
        if (cwtAmp[f][n] > maxCwtVal) maxCwtVal = cwtAmp[f][n];
      }
    }

    // 4. Perform Synchrosqueezing Reallocation (SSWT)
    const sswtMatrix = Array.from({ length: cwtFreqSteps }, () => new Array(N).fill(0));
    const df = freqs[1] - freqs[0];
    for (let f = 0; f < cwtFreqSteps; f++) {
      for (let n = 0; n < N; n++) {
        const c = cwtReal[f][n];
        const d = cwtImag[f][n];
        const amp = Math.sqrt(c * c + d * d);
        
        // Amplitude mask to block near-zero spikes
        if (amp > 0.03 * maxCwtVal) {
          const f_est = Math.abs(instFreq[f][n]);
          if (f_est >= minFreq && f_est <= maxFreq) {
            const binIdx = Math.round((f_est - minFreq) / df);
            if (binIdx >= 0 && binIdx < cwtFreqSteps) {
              sswtMatrix[binIdx][n] += amp;
            }
          }
        }
      }
    }

    // Apply soft thresholding to SSWT
    let maxSswtVal = 0.001;
    for (let f = 0; f < cwtFreqSteps; f++) {
      for (let n = 0; n < N; n++) {
        if (sswtMatrix[f][n] > maxSswtVal) maxSswtVal = sswtMatrix[f][n];
      }
    }
    for (let f = 0; f < cwtFreqSteps; f++) {
      for (let n = 0; n < N; n++) {
        if (sswtMatrix[f][n] < 0.05 * maxSswtVal) {
          sswtMatrix[f][n] = 0;
        }
      }
    }

    // Color maps
    const getJetColor = (v) => {
      const norm = Math.max(0, Math.min(1, v));
      let r = 0, g = 0, b = 0;
      if (norm < 0.25) {
        b = 255;
        g = Math.floor(norm * 4 * 255);
      } else if (norm < 0.5) {
        b = Math.floor((0.5 - norm) * 4 * 255);
        g = 255;
      } else if (norm < 0.75) {
        g = 255;
        r = Math.floor((norm - 0.5) * 4 * 255);
      } else {
        r = 255;
        g = Math.floor((1.0 - norm) * 4 * 255);
      }
      return `rgb(${r}, ${g}, ${b})`;
    };

    const getHotColor = (v) => {
      const norm = Math.max(0, Math.min(1, v));
      let r = 0, g = 0, b = 0;
      if (norm < 0.33) {
        r = Math.floor((norm / 0.33) * 255);
      } else if (norm < 0.66) {
        r = 255;
        g = Math.floor(((norm - 0.33) / 0.33) * 255);
      } else {
        r = 255;
        g = 255;
        b = Math.floor(((norm - 0.66) / 0.34) * 255);
      }
      return `rgb(${r}, ${g}, ${b})`;
    };

    // Draw parameters
    const paneW = 270;
    const dividerW = 60;
    const cellW = paneW / N;
    const cellH = H / cwtFreqSteps;

    // 1. Draw CWT (Left Pane)
    for (let f = 0; f < cwtFreqSteps; f++) {
      const y = H - (f + 1) * cellH;
      for (let n = 0; n < N; n++) {
        const val = cwtAmp[f][n] / maxCwtVal;
        ctx.fillStyle = getJetColor(val);
        ctx.fillRect(n * cellW, y, cellW + 0.5, cellH + 0.5);
      }
    }

    // 2. Draw SSWT (Right Pane)
    const xOffset = paneW + dividerW;
    for (let f = 0; f < cwtFreqSteps; f++) {
      const y = H - (f + 1) * cellH;
      for (let n = 0; n < N; n++) {
        const val = sswtMatrix[f][n] / maxSswtVal;
        ctx.fillStyle = getHotColor(val);
        ctx.fillRect(xOffset + n * cellW, y, cellW + 0.5, cellH + 0.5);
      }
    }

    // 3. Draw Divider and Shared Y-axis (x: 270 to 330)
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(paneW, 0, dividerW, H);
    
    // Draw grid/divider lines
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.moveTo(paneW, 0);
    ctx.lineTo(paneW, H);
    ctx.moveTo(xOffset, 0);
    ctx.lineTo(xOffset, H);
    ctx.stroke();

    // Draw central y-axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const tickSteps = 5;
    for (let i = 0; i < tickSteps; i++) {
      const pct = i / (tickSteps - 1);
      const freqVal = maxFreq - pct * (maxFreq - minFreq);
      const yPos = pct * H;
      ctx.fillText(`${Math.round(freqVal)}Hz`, paneW + dividerW / 2, yPos);
    }

    // 4. Draw dashed frequency slice line across both panes
    const y_slice = H - ((sliceFreq - minFreq) / (maxFreq - minFreq)) * H;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y_slice);
    ctx.lineTo(W, y_slice);
    ctx.stroke();
    ctx.setLineDash([]);

    // 5. Draw text overlay titles
    ctx.fillStyle = 'rgba(9, 13, 22, 0.7)';
    ctx.fillRect(10, 10, 110, 18);
    ctx.fillRect(xOffset + 10, 10, 120, 18);
    
    ctx.font = 'bold 9.5px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('CWT (Morlet)', 15, 13);
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('SSWT (Squeezed)', xOffset + 15, 13);

  }, [selectedWell, waveletType, cwtFreqSteps, rawTrace, sliceFreq]);

  // 3. Real 3-Level Stationary Wavelet Transform (SWT) in JavaScript
  // Decimation-free (undecimated) wavelet transform using Haar wavelets.
  // This is completely translation-invariant.
  const computeRealSWTHaar = () => {
    if (rawTrace.length === 0) return { d1: [], d2: [], d3: [], a3: [], denoised: [], noiseRemovedPct: 0 };

    const N = rawTrace.length;
    const a = [rawTrace.slice()];
    const d = [];

    // Forward Haar SWT (3 levels)
    for (let j = 0; j < 3; j++) {
      const step = Math.pow(2, j);
      const a_next = new Array(N).fill(0);
      const d_next = new Array(N).fill(0);
      for (let n = 0; n < N; n++) {
        const nextIdx = (n + step) % N;
        a_next[n] = (a[a.length - 1][n] + a[a.length - 1][nextIdx]) / Math.sqrt(2);
        d_next[n] = (-a[a.length - 1][n] + a[a.length - 1][nextIdx]) / Math.sqrt(2);
      }
      a.push(a_next);
      d.push(d_next);
    }

    const d1 = d[0];
    const d2 = d[1];
    const d3 = d[2];
    const a3 = a[3];

    // Denoising via Soft Thresholding on detail coefficients
    const softThreshold = (coefs, threshold) => {
      return coefs.map(v => {
        const abs = Math.abs(v);
        return abs > threshold ? Math.sign(v) * (abs - threshold) : 0;
      });
    };

    const d1_t = softThreshold(d1, swtThreshold);
    const d2_t = softThreshold(d2, swtThreshold);
    const d3_t = softThreshold(d3, swtThreshold);

    // Inverse Haar SWT Reconstruction
    let denoised = a3.slice();
    const threshDetails = [d1_t, d2_t, d3_t];
    for (let j = 2; j >= 0; j--) {
      const step = Math.pow(2, j);
      const a_prev = new Array(N).fill(0);
      for (let n = 0; n < N; n++) {
        let prevIdx = (n - step) % N;
        if (prevIdx < 0) prevIdx += N;
        a_prev[n] = 0.5 * ( (denoised[n] - threshDetails[j][n]) / Math.sqrt(2) + 
                            (denoised[prevIdx] + threshDetails[j][prevIdx]) / Math.sqrt(2) );
      }
      denoised = a_prev;
    }

    // Calculate Noise Energy Removed %
    const rawEnergy = rawTrace.reduce((sum, v) => sum + v * v, 0);
    const denoisedEnergy = denoised.reduce((sum, v) => sum + v * v, 0);
    const noiseRemovedPct = Math.max(0, Math.min(100, ((rawEnergy - denoisedEnergy) / (rawEnergy || 1)) * 100));

    return { d1, d2, d3, a3, denoised, noiseRemovedPct };
  };

  const { d1, d2, d3, a3, denoised, noiseRemovedPct } = computeRealSWTHaar();

  // Normalize each sub-band by its own maximum absolute value to fit within the SVGs
  const normSWT = (arr) => {
    const max = Math.max(...arr.map(Math.abs), 0.0001);
    return arr.map(v => v / max);
  };
  const d1_norm = normSWT(d1);
  const d2_norm = normSWT(d2);
  const d3_norm = normSWT(d3);
  const a3_norm = normSWT(a3);

  // Calculate CWT Frequency Slice Curve dynamically in JavaScript
  const getSliceCurve = () => {
    if (rawTrace.length === 0) return [];
    const N = rawTrace.length;
    const dtS = dtMs / 1000.0;
    const duration = 4.0 / sliceFreq;
    const tHalf = duration / 2;
    const waveletSamples = [];
    for (let t = -tHalf; t <= tHalf; t += dtS) {
      waveletSamples.push(t);
    }
    let wavelet = [];
    if (waveletType === 'Ricker') {
      wavelet = waveletSamples.map(t => {
        const term = Math.PI * sliceFreq * t;
        return (1 - 2 * term * term) * Math.exp(-term * term);
      });
    } else {
      const sigma = 1.0 / (2.0 * Math.PI * sliceFreq) * 2.0;
      wavelet = waveletSamples.map(t => {
        return Math.exp(-0.5 * (t / sigma) ** 2) * Math.cos(2.0 * Math.PI * sliceFreq * t);
      });
    }
    const energy = Math.sqrt(wavelet.reduce((sum, val) => sum + val * val, 0));
    const normWavelet = wavelet.map(w => w / (energy || 1));
    const sliceCurve = new Array(N).fill(0);
    const halfM = Math.floor(normWavelet.length / 2);
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < normWavelet.length; k++) {
        const idx = n - halfM + k;
        if (idx >= 0 && idx < N) {
          sum += rawTrace[idx] * normWavelet[k];
        }
      }
      sliceCurve[n] = Math.abs(sum);
    }
    const max = Math.max(...sliceCurve, 0.0001);
    return sliceCurve.map(v => v / max);
  };
  const sliceCurve = getSliceCurve();

  return (
    <div className="slide" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* Header and Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity size={24} style={{ color: 'var(--accent-gold)' }} />
            Geologist CWT / SWT Wavelet Analyst Workspace
          </h2>
          <p className="slide-subtitle">Interactive signal analysis tools to study thin-bed tuning, gas shadows, and shift-invariant ML feature decomposition.</p>
        </div>

        {/* Well Selector */}
        <div className="glass-card" style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 600 }}>Active Well Study:</span>
          <div style={{ display: 'flex', gap: '6px' }}>
            {['Z-02', 'Z-03', 'Z-04', 'Z-05', 'Z-06', 'Z-07'].map(wName => (
              <button
                key={wName}
                onClick={() => setSelectedWell(wName)}
                style={{
                  padding: '4px 10px',
                  borderRadius: '6px',
                  border: `1.5px solid ${selectedWell === wName ? 'var(--accent-blue)' : 'var(--border-color)'}`,
                  background: selectedWell === wName ? 'rgba(59,130,246,0.12)' : 'var(--bg-secondary)',
                  color: selectedWell === wName ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  fontWeight: selectedWell === wName ? 'bold' : 'normal',
                  fontSize: '12.5px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                {wName} study
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '20px' }}>
        
        {/* ========================================================
            LEFT COLUMN: INTERACTIVE CWT & SWT VISUALIZATIONS
            ======================================================== */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* CWT Real-time Spectrogram panel */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={18} style={{ color: 'var(--accent-blue)' }} />
                <h3 style={{ margin: 0, fontSize: '15px' }}>1. Real-Time CWT & SSWT Workstation Spectrograms</h3>
              </div>
              
              {/* Wavelet Controls */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Wavelet:</span>
                <select 
                  value={waveletType} 
                  onChange={(e) => setWaveletType(e.target.value)}
                  style={{ padding: '3px 8px', borderRadius: '4px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)', fontSize: '12px' }}
                >
                  <option value="Ricker">Ricker (Zero Phase Standard)</option>
                  <option value="Morlet">Morlet (Complex Packet)</option>
                </select>

                <span style={{ fontSize: '12px', color: 'var(--text-secondary)', marginLeft: '8px' }}>Steps:</span>
                <input 
                  type="range" 
                  min="10" 
                  max="50" 
                  value={cwtFreqSteps} 
                  onChange={(e) => setCwtFreqSteps(parseInt(e.target.value))}
                  style={{ width: '80px', height: '4px' }}
                />
                <span style={{ fontSize: '11px', minWidth: '20px', fontFamily: 'monospace' }}>{cwtFreqSteps}</span>
              </div>
            </div>

            {/* Time ticks for alignment */}
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px', paddingLeft: '5px', paddingRight: '5px' }}>
              <span>Left: CWT (Morlet spectral decomposition)</span>
              <span style={{ fontFamily: 'monospace' }}>{times[0]?.toFixed(0)} ms ──── TWT ──── {times[times.length - 1]?.toFixed(0)} ms</span>
              <span>Right: SSWT (Synchrosqueezed high-res)</span>
            </div>

            {/* Canvas Plot */}
            <div style={{ position: 'relative', width: '100%', height: '220px', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
              <canvas 
                ref={canvasRef} 
                width={600} 
                height={220} 
                style={{ width: '100%', height: '100%', display: 'block' }}
              />
            </div>

            {/* Real-time Frequency Slice Selector & Waveform */}
            <div style={{ marginTop: '16px', borderTop: '1px solid var(--border-color)', paddingTop: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--text-primary)', fontWeight: 'bold' }}>
                  <Info size={14} style={{ color: 'var(--accent-blue)' }} />
                  <span>Interactive Workstation: CWT spectral slice at {sliceFreq} Hz</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="range"
                    min="10"
                    max="60"
                    value={sliceFreq}
                    onChange={(e) => setSliceFreq(parseInt(e.target.value))}
                    style={{ width: '150px', height: '4px' }}
                  />
                  <span style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--accent-blue)', fontFamily: 'monospace' }}>{sliceFreq} Hz</span>
                </div>
              </div>

              {/* Slice Waveform Display */}
              <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr', gap: '10px', alignItems: 'center' }}>
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', textAlign: 'right' }}>Spectral Amp</div>
                <div style={{ backgroundColor: '#090d16', borderRadius: '6px', border: '1px solid var(--border-color)', height: '70px', padding: '6px 10px', position: 'relative' }}>
                  <svg width="100%" height="100%" viewBox="0 0 500 60" preserveAspectRatio="none">
                    <line x1="0" y1="30" x2="500" y2="30" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="3 3" />
                    <path
                      d={`M ${sliceCurve.map((v, i) => `${(i / (sliceCurve.length - 1)) * 500} ${55 - v * 45}`).join(' L ')}`}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="1.5"
                    />
                  </svg>
                  <div style={{ position: 'absolute', top: '4px', right: '8px', fontSize: '9.5px', color: 'var(--text-muted)' }}>
                    {sliceFreq <= 25 ? 'Low-Freq Gas Shadow Tracker' : sliceFreq >= 45 ? 'Thin-Bed Boundary Resolver' : 'Dominant Wavelength Track'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* SWT Interactive Filter Bank Plot */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu size={18} style={{ color: 'var(--accent-green)' }} />
                <h3 style={{ margin: 0, fontSize: '15px' }}>2. Stationary Wavelet Transform (SWT) Sandbox</h3>
              </div>

              {/* Sub-Tab Navigation */}
              <div style={{ display: 'flex', border: '1px solid var(--border-color)', borderRadius: '6px', overflow: 'hidden', background: 'var(--bg-dark)' }}>
                <button
                  onClick={() => setActiveSwtSubTab('decomposition')}
                  style={{
                    padding: '4px 12px',
                    fontSize: '11.5px',
                    border: 'none',
                    background: activeSwtSubTab === 'decomposition' ? 'rgba(16,185,129,0.12)' : 'transparent',
                    color: activeSwtSubTab === 'decomposition' ? 'var(--accent-green)' : 'var(--text-secondary)',
                    fontWeight: activeSwtSubTab === 'decomposition' ? 'bold' : 'normal',
                    cursor: 'pointer'
                  }}
                >
                  Detail Levels
                </button>
                <button
                  onClick={() => setActiveSwtSubTab('denoising')}
                  style={{
                    padding: '4px 12px',
                    fontSize: '11.5px',
                    border: 'none',
                    borderLeft: '1px solid var(--border-color)',
                    background: activeSwtSubTab === 'denoising' ? 'rgba(16,185,129,0.12)' : 'transparent',
                    color: activeSwtSubTab === 'denoising' ? 'var(--accent-green)' : 'var(--text-secondary)',
                    fontWeight: activeSwtSubTab === 'denoising' ? 'bold' : 'normal',
                    cursor: 'pointer'
                  }}
                >
                  Denoising Sandbox
                </button>
                <button
                  onClick={() => setActiveSwtSubTab('static_qc')}
                  style={{
                    padding: '4px 12px',
                    fontSize: '11.5px',
                    border: 'none',
                    borderLeft: '1px solid var(--border-color)',
                    background: activeSwtSubTab === 'static_qc' ? 'rgba(16,185,129,0.12)' : 'transparent',
                    color: activeSwtSubTab === 'static_qc' ? 'var(--accent-green)' : 'var(--text-secondary)',
                    fontWeight: activeSwtSubTab === 'static_qc' ? 'bold' : 'normal',
                    cursor: 'pointer'
                  }}
                >
                  Python QC Plot
                </button>
              </div>
            </div>

            {/* Content rendering based on active sub-tab */}
            {activeSwtSubTab === 'decomposition' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {/* Custom Stacked SVGs showing L1, L2, L3 details and a3 approx */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', backgroundColor: '#090d16', borderRadius: '8px', border: '1px solid var(--border-color)', padding: '10px' }}>
                  
                  {/* SWT L1 Detail */}
                  <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center' }}>
                    <div style={{ fontSize: '10.5px', color: '#f43f5e', fontWeight: 'bold' }}>L1 Detail (Noise)</div>
                    <svg width="100%" height="60" viewBox="0 0 500 60" preserveAspectRatio="none">
                      <line x1="0" y1="30" x2="500" y2="30" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="3 3" />
                      <path
                        d={`M ${d1_norm.map((v, i) => `${(i / (d1_norm.length - 1)) * 500} ${30 - v * 25}`).join(' L ')}`}
                        fill="none"
                        stroke="#f43f5e"
                        strokeWidth="1.2"
                      />
                    </svg>
                  </div>

                  {/* SWT L2 Detail */}
                  <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center' }}>
                    <div style={{ fontSize: '10.5px', color: '#f59e0b', fontWeight: 'bold' }}>L2 Detail (Sands)</div>
                    <svg width="100%" height="60" viewBox="0 0 500 60" preserveAspectRatio="none">
                      <line x1="0" y1="30" x2="500" y2="30" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="3 3" />
                      <path
                        d={`M ${d2_norm.map((v, i) => `${(i / (d2_norm.length - 1)) * 500} ${30 - v * 25}`).join(' L ')}`}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="1.2"
                      />
                    </svg>
                  </div>

                  {/* SWT L3 Detail */}
                  <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center' }}>
                    <div style={{ fontSize: '10.5px', color: '#8b5cf6', fontWeight: 'bold' }}>L3 Detail (Sequences)</div>
                    <svg width="100%" height="60" viewBox="0 0 500 60" preserveAspectRatio="none">
                      <line x1="0" y1="30" x2="500" y2="30" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="3 3" />
                      <path
                        d={`M ${d3_norm.map((v, i) => `${(i / (d3_norm.length - 1)) * 500} ${30 - v * 25}`).join(' L ')}`}
                        fill="none"
                        stroke="#8b5cf6"
                        strokeWidth="1.2"
                      />
                    </svg>
                  </div>

                  {/* SWT L3 Approx */}
                  <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center' }}>
                    <div style={{ fontSize: '10.5px', color: '#10b981', fontWeight: 'bold' }}>L3 Approx (Compaction)</div>
                    <svg width="100%" height="60" viewBox="0 0 500 60" preserveAspectRatio="none">
                      <line x1="0" y1="30" x2="500" y2="30" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="3 3" />
                      <path
                        d={`M ${a3_norm.map((v, i) => `${(i / (a3_norm.length - 1)) * 500} ${30 - v * 25}`).join(' L ')}`}
                        fill="none"
                        stroke="#10b981"
                      />
                    </svg>
                  </div>

                  {/* Stratigraphic Sequence Cycles Guide */}
                  <div style={{ marginTop: '10px', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid var(--border-color)', padding: '12px' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 'bold', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.05em' }}>Geologist Sequence Stratigraphy Guide:</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '12px' }}>
                      <div style={{ borderLeft: '3px solid #f59e0b', paddingLeft: '8px' }}>
                        <strong style={{ color: 'var(--text-primary)' }}>L2 (Parasequences):</strong>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '11px', marginTop: '2px' }}>Tracks individual reservoir sand bodies (10–25m thick). Peaks map top sand boundaries; troughs map sandstone bases.</div>
                      </div>
                      <div style={{ borderLeft: '3px solid #8b5cf6', paddingLeft: '8px' }}>
                        <strong style={{ color: 'var(--text-primary)' }}>L3 (Systems Tracts):</strong>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '11px', marginTop: '2px' }}>Tracks major 3rd-order depositional sequences. Ideal for tracking regional unconformities and maximum flooding surfaces.</div>
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            ) : activeSwtSubTab === 'denoising' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {/* Controls */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12.5px', background: 'var(--bg-dark)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Soft Threshold:</span>
                    <input 
                      type="range" 
                      min="0.01" 
                      max="0.25" 
                      step="0.01"
                      value={swtThreshold} 
                      onChange={(e) => setSwtThreshold(parseFloat(e.target.value))}
                      style={{ width: '130px' }}
                    />
                    <span style={{ fontWeight: 'bold', color: 'var(--accent-green)', fontFamily: 'monospace' }}>{swtThreshold.toFixed(2)}</span>
                  </div>
                  <div style={{ color: 'var(--text-muted)' }}>
                    Noise Energy Removed: <strong style={{ color: '#22c55e' }}>{noiseRemovedPct.toFixed(1)}%</strong>
                  </div>
                </div>

                {/* Overlaid Waveform SVG */}
                <div style={{ position: 'relative', width: '100%', height: '200px', backgroundColor: '#090d16', borderRadius: '8px', border: '1px solid var(--border-color)', padding: '10px' }}>
                  <svg width="100%" height="100%" viewBox="0 0 500 180" preserveAspectRatio="none">
                    <line x1="0" y1="90" x2="500" y2="90" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="3 3" />
                    
                    {/* Raw Trace in background */}
                    <path
                      d={`M ${rawTrace.map((v, i) => `${(i / (rawTrace.length - 1)) * 500} ${90 - v * 70}`).join(' L ')}`}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="1.2"
                      opacity="0.3"
                    />

                    {/* Denoised reconstructed trace */}
                    <path
                      d={`M ${denoised.map((v, i) => `${(i / (denoised.length - 1)) * 500} ${90 - v * 70}`).join(' L ')}`}
                      fill="none"
                      stroke="#22c55e"
                      strokeWidth="1.8"
                    />
                  </svg>
                  
                  {/* Legend overlay */}
                  <div style={{ position: 'absolute', bottom: '10px', left: '10px', display: 'flex', gap: '12px', fontSize: '10.5px', backgroundColor: 'rgba(9, 13, 22, 0.85)', padding: '2px 6px', borderRadius: '4px' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'rgba(56, 189, 248, 0.6)' }}>
                      <span style={{ width: '8px', height: '2px', backgroundColor: '#38bdf8', opacity: 0.6 }}></span> Raw Signal
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#22c55e' }}>
                      <span style={{ width: '8px', height: '2px', backgroundColor: '#22c55e' }}></span> Haar SWT Denoised
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', background: 'var(--bg-dark)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                  This panel displays the publication-quality QC image generated by the Python backend script 
                  <code style={{ marginLeft: '5px', color: 'var(--accent-blue)' }}>cwt_swt_analysis.py</code>.
                </div>
                <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#090d16', padding: '6px', display: 'flex', justifyContent: 'center' }}>
                  <img 
                    src="/well_images/cwt_swt_analysis.png" 
                    alt="SWT QC Plot" 
                    style={{ width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }} 
                  />
                </div>
              </div>
            )}
          </div>

        </div>

        {/* ========================================================
            RIGHT COLUMN: GEOLOGIST POCKET CALCULATORS & NOTES
            ======================================================== */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Tuning thickness calculator */}
          <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--accent-gold)' }}>
            <div className="card-header" style={{ marginBottom: '14px' }}>
              <Sliders size={18} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
              <h3>Interactive Tuning Thickness Calculator (λ / 4)</h3>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              
              {/* Asset Target Quick Presets */}
              <div style={{ background: 'rgba(251,191,36,0.03)', padding: '10px', borderRadius: '6px', border: '1px dashed rgba(251,191,36,0.2)' }}>
                <div style={{ fontSize: '11.5px', color: 'var(--accent-gold)', fontWeight: 'bold', marginBottom: '6px' }}>Zamzama Reservoir Target Presets:</div>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  <button 
                    onClick={() => { setVelocity(3100); setTuningFreq(28); }}
                    style={{ padding: '3px 8px', fontSize: '10.5px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: 'var(--text-secondary)', cursor: 'pointer', transition: 'all 0.2s' }}
                  >
                    Gas Sand B (12m Target)
                  </button>
                  <button 
                    onClick={() => { setVelocity(3450); setTuningFreq(40); }}
                    style={{ padding: '3px 8px', fontSize: '10.5px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: 'var(--text-secondary)', cursor: 'pointer', transition: 'all 0.2s' }}
                  >
                    Brine Sand B (21m Target)
                  </button>
                  <button 
                    onClick={() => { setVelocity(3900); setTuningFreq(55); }}
                    style={{ padding: '3px 8px', fontSize: '10.5px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: 'var(--text-secondary)', cursor: 'pointer', transition: 'all 0.2s' }}
                  >
                    Tight Deep Sands (17m Target)
                  </button>
                </div>
              </div>
              {/* Velocity slider */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12.5px', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Interval Velocity (Vp):</span>
                  <strong>{velocity} m/s</strong>
                </div>
                <input 
                  type="range" 
                  min="2000" 
                  max="5000" 
                  step="100"
                  value={velocity} 
                  onChange={(e) => setVelocity(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              {/* Frequency slider */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12.5px', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Dominant Wavelet Frequency:</span>
                  <strong>{tuningFreq} Hz</strong>
                </div>
                <input 
                  type="range" 
                  min="15" 
                  max="70" 
                  value={tuningFreq} 
                  onChange={(e) => setTuningFreq(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              {/* Calculations results */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', backgroundColor: 'var(--bg-dark)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)', marginTop: '8px' }}>
                <div>
                  <div style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>Wavelength (λ)</div>
                  <div style={{ fontSize: '16px', fontWeight: 800, color: 'var(--text-primary)' }}>{wavelength.toFixed(1)} m</div>
                </div>
                <div>
                  <div style={{ fontSize: '10.5px', color: 'var(--accent-gold)' }}>Tuning Limit (λ/4)</div>
                  <div style={{ fontSize: '16px', fontWeight: 800, color: 'var(--accent-gold)' }}>{tuningThickness.toFixed(1)} m</div>
                </div>
              </div>

              {/* Geological interpretation */}
              <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: '1.5', padding: '10px', background: 'rgba(251,191,36,0.04)', borderRadius: '6px' }}>
                <span style={{ fontWeight: 'bold', color: 'var(--accent-gold)' }}>Geological Impact: </span>
                {tuningThickness <= 15 ? (
                  <span>
                    Your tuning limit is <strong>{tuningThickness.toFixed(1)}m</strong>. The thin reservoir sand channels in Zamzama (10–18m thickness) sit exactly at this threshold. CWT spectral decomposition is <strong>required</strong> to resolve their boundaries!
                  </span>
                ) : (
                  <span>
                    Tuning limit is <strong>{tuningThickness.toFixed(1)}m</strong>. Thin beds below this limit will merge into a single reflection, hiding reservoir pinch-outs on standard amplitude sections.
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Geological Explanation & Physics Card */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="card-header" style={{ marginBottom: '12px' }}>
              <Info size={18} className="card-icon" style={{ color: 'var(--accent-blue)' }} />
              <h3>Geologist's Interpretation Guidelines</h3>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13px', lineHeight: '1.5' }}>
              <div style={{ borderLeft: '3px solid var(--accent-blue)', paddingLeft: '10px' }}>
                <h4 style={{ margin: '0 0 4px 0', fontSize: '13px', color: 'var(--text-primary)' }}>CWT: Hydrocarbon Indicator</h4>
                <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                  When gas sands are present, high frequencies are attenuated. By looking at the <strong>15–20 Hz slice</strong> on the CWT spectrogram, you can locate the "low-frequency shadow" directly beneath the gas accumulation.
                </p>
              </div>

              <div style={{ borderLeft: '3px solid var(--accent-green)', paddingLeft: '10px' }}>
                <h4 style={{ margin: '0 0 4px 0', fontSize: '13px', color: 'var(--text-primary)' }}>SWT: Shift-Invariant ML Features</h4>
                <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                  Traditional DWT breaks when the trace shifts, causing the ML model to output blocky, non-physical facies anomalies. SWT is <strong>translation-invariant</strong>, meaning it tracks continuous dipping beds smoothly, making it ideal as an ML feature input.
                </p>
              </div>

              <div style={{ borderLeft: '3px solid var(--accent-gold)', paddingLeft: '10px' }}>
                <h4 style={{ margin: '0 0 4px 0', fontSize: '13px', color: 'var(--text-primary)' }}>SSWT: Synchrosqueezing (Ultra-High Resolution)</h4>
                <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                  Standard CWT smears energy in the frequency domain due to Heisenberg uncertainty limits. SSWT extracts instantaneous frequencies and dynamically "squeezes" the CWT energy onto sharp, discrete horizons. This is the ultimate tool for resolving thin parasequence tops/bases.
                </p>
              </div>
            </div>
          </div>

          {/* Standalone study badge */}
          <div className="glass-card" style={{ padding: '14px', display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12.5px', color: 'var(--text-muted)' }}>
            <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-green)' }}></span>
            <span>Study is isolated in this panel and will not modify raw log calculations in other tabs.</span>
          </div>

        </div>

      </div>
    </div>
  );
}
