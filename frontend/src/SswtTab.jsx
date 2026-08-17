import React, { useState, useEffect, useRef } from 'react';
import { Layers, Info, Activity, ZoomIn, CheckCircle2, ChevronRight, Sliders, Database, MapPin } from 'lucide-react';
import { wellsData } from './data';

export default function SswtTab() {
  const [activeSubTab, setActiveSubTab] = useState('workstation'); // 'workstation' | 'profiler' | 'concept'
  const [selectedTrack, setSelectedTrack] = useState(3); // Default to SSWT track
  
  // Real well selection and cutoff parameters for the profiler
  const [selectedWell, setSelectedWell] = useState('Z-04');
  const [vshCutoff, setVshCutoff] = useState(0.17); // Default VSH threshold for sandstone definition
  const [cwtFreqSteps, setCwtFreqSteps] = useState(25);
  const [minFreq, setMinFreq] = useState(10);
  const [maxFreq, setMaxFreq] = useState(60);

  const profilerCanvasRef = useRef(null);

  const trackInfo = {
    1: {
      title: "Track 1: Raw Seismic Trace",
      desc: "This is the raw, unedited compressional wave reflection trace extracted from the 3D seismic volume at the coordinate nearest to well Z-04. It contains both stratigraphic reflections and noise.",
      usecase: "Used as the baseline signal to verify the performance of denoising and spectral decomposition algorithms."
    },
    2: {
      title: "Track 2: CWT Spectrogram (Continuous Wavelet)",
      desc: "Computed by convolving the seismic trace with scaled Morlet wavelets. It represents standard spectral decomposition. Notice how the energy packages are 'smeared' vertically across frequencies.",
      usecase: "Shows general frequency trends and broad energy packages, but cannot be used to pinpoint exact thin-bed boundaries due to frequency blurring."
    },
    3: {
      title: "Track 3: SSWT Spectrogram (Synchrosqueezed Wavelet)",
      desc: "Our most advanced time-frequency tool. By taking the time-derivative of the complex CWT phase, it calculates the true instantaneous frequency at every point and reallocates (squeezes) the energy into sharp frequency bins. This collapses the blurry CWT packages into sharp, horizontal horizons.",
      usecase: "Allows geologists to resolve thin-bed reservoirs (down to 10-15m) far below the standard tuning thickness limit, and trace precise sand boundary contacts."
    },
    4: {
      title: "Track 4: SWT L1 Detail (High-Frequency Noise)",
      desc: "The first detail level of our Stationary Wavelet Transform. It isolates the highest frequency components of the seismic trace, which do not correlate to coherent rock layers.",
      usecase: "Represents random mud noise, tool coupling jitter, and processing artifacts. We target this track for elimination during denoising."
    },
    5: {
      title: "Track 5: SWT L2 Detail (Reservoir Sand Cycle)",
      desc: "The middle-frequency band of the SWT. In the Zamzama field, this frequency band matches the thickness scale of the target sandstone channels (~10-25m).",
      usecase: "Highly critical for horizon picking: the peaks of this curve map the top boundaries of reservoir sands, and the troughs map the sandstone bases."
    },
    6: {
      title: "Track 6: SWT L3 Detail (Sequence Boundaries)",
      desc: "The low-frequency band of the SWT. It captures larger, third-order depositional sequences and regional geological cycles.",
      usecase: "Ideal for tracking maximum flooding surfaces (MFS), regional unconformities, and sequence boundaries across the basin."
    },
    7: {
      title: "Track 7: SWT L3 Approximation (Compaction Trend)",
      desc: "The lowest frequency baseline of the wavelet decomposition. It removes all high-frequency wiggles and sandstone reflections.",
      usecase: "Tracks the slow, regional geological compaction and basin-scale subsidence trend, which we use to calibrate our ML point prediction baseline."
    },
    8: {
      title: "Track 8: SWT Denoised Trace",
      desc: "The reconstructed seismic signal after applying soft-thresholding to details L1-L3 (removing the noise shown in Track 4). It is overlaid on the raw trace (faded blue) for direct visual QC.",
      usecase: "Provides a clean, jitter-free reflection trace that is ideal for structural interpretation and training stable machine learning models."
    }
  };

  // Real-Well Stacked Reservoir Profiler Calculations
  useEffect(() => {
    if (activeSubTab !== 'profiler' || !profilerCanvasRef.current) return;

    const canvas = profilerCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Clear background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, W, H);

    // Get real well data
    const well = wellsData[selectedWell];
    if (!well || !well.samples || well.samples.length === 0) return;

    const samples = well.samples;
    const N = samples.length;
    const times = samples.map(s => s.time || 0);
    const tStart = times[0];
    const tEnd = times[times.length - 1];

    const rawTraceValues = samples.map(s => s.seismic_amp || 0);
    const maxAmp = Math.max(...rawTraceValues.map(Math.abs), 0.0001);
    const trace = rawTraceValues.map(v => v / maxAmp);
    
    // Dynamic Lithology: Sand exists ONLY where seismic trace is active AND VSH is below cutoff
    const isSand = samples.map(s => {
      const active = Math.abs(s.seismic_amp || 0) > 0.005;
      const vsh = s['VSH (Pred)'] !== undefined ? s['VSH (Pred)'] : 1.0;
      return (active && vsh < vshCutoff) ? 1 : 0;
    });

    const dtMs = times.length > 1 ? times[1] - times[0] : 2.0;
    const dtS = dtMs / 1000.0;

    // 1. Setup target frequencies
    const freqs = [];
    for (let i = 0; i < cwtFreqSteps; i++) {
      freqs.push(minFreq + (maxFreq - minFreq) * (i / (cwtFreqSteps - 1)));
    }

    // 2. Compute CWT
    const cwtReal = [];
    const cwtImag = [];
    const cwtAmp = [];

    freqs.forEach(freq => {
      const rowReal = new Array(N).fill(0.0);
      const rowImag = new Array(N).fill(0.0);
      const rowAmp = new Array(N).fill(0.0);
      
      const duration = 4.0 / freq;
      const tHalf = duration / 2;
      const waveletSamples = [];
      for (let t = -tHalf; t <= tHalf; t += dtS) {
        waveletSamples.push(t);
      }
      
      const sigma = 1.0 / (2.0 * Math.PI * freq) * 2.0;
      const wReal = waveletSamples.map(t => Math.exp(-0.5 * (t / sigma) ** 2) * Math.cos(2.0 * Math.PI * freq * t));
      const wImag = waveletSamples.map(t => Math.exp(-0.5 * (t / sigma) ** 2) * Math.sin(2.0 * Math.PI * freq * t));
      
      const energy = Math.sqrt(wReal.reduce((sum, val) => sum + val * val, 0));
      const nwReal = wReal.map(w => w / (energy || 1));
      const nwImag = wImag.map(w => w / (energy || 1));
      
      const halfM = Math.floor(nwReal.length / 2);
      for (let n = 0; n < N; n++) {
        let sumReal = 0.0;
        let sumImag = 0.0;
        for (let k = 0; k < nwReal.length; k++) {
          const idx = n - halfM + k;
          if (idx >= 0 && idx < N) {
            sumReal += trace[idx] * nwReal[k];
            sumImag += trace[idx] * nwImag[k];
          }
        }
        rowReal[n] = sumReal;
        rowImag[n] = -sumImag; // Conjugated
        rowAmp[n] = Math.sqrt(sumReal * sumReal + sumImag * sumImag);
      }
      cwtReal.push(rowReal);
      cwtImag.push(rowImag);
      cwtAmp.push(rowAmp);
    });

    // Time Derivatives for SSWT
    const cwtDiffReal = [];
    const cwtDiffImag = [];
    for (let f = 0; f < cwtFreqSteps; f++) {
      const rowDiffReal = new Array(N).fill(0.0);
      const rowDiffImag = new Array(N).fill(0.0);
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

    // Instantaneous Freqs
    const instFreq = [];
    for (let f = 0; f < cwtFreqSteps; f++) {
      const rowFreq = new Array(N).fill(0.0);
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

    // Max CWT amplitude value
    let maxCwtVal = 0.001;
    for (let f = 0; f < cwtFreqSteps; f++) {
      for (let n = 0; n < N; n++) {
        if (cwtAmp[f][n] > maxCwtVal) maxCwtVal = cwtAmp[f][n];
      }
    }

    // Reallocate to SSWT matrix
    const sswtMatrix = Array.from({ length: cwtFreqSteps }, () => new Array(N).fill(0.0));
    const df = freqs[1] - freqs[0];
    for (let f = 0; f < cwtFreqSteps; f++) {
      for (let n = 0; n < N; n++) {
        const c = cwtReal[f][n];
        const d = cwtImag[f][n];
        const amp = Math.sqrt(c * c + d * d);
        
        // 3% amplitude soft mask to prevent division spikes
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

    // Normalize SSWT values
    let maxSswtVal = 0.001;
    for (let f = 0; f < cwtFreqSteps; f++) {
      for (let n = 0; n < N; n++) {
        if (sswtMatrix[f][n] > maxSswtVal) maxSswtVal = sswtMatrix[f][n];
      }
    }

    // Color mapping
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

    // Drawing layouts
    const trackH = H - 50;
    const getPixelY = (tVal) => 25 + ((tVal - tStart) / (tEnd - tStart)) * trackH;

    // Draw Shared Time axis on the left (x: 0 to 45)
    ctx.fillStyle = '#64748b';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    const timeSpan = tEnd - tStart;
    const stepSize = timeSpan > 400 ? 100 : 40;
    
    for (let t = Math.ceil(tStart / 10) * 10; t <= tEnd; t += stepSize) {
      ctx.fillText(`${Math.round(t)}ms`, 45, getPixelY(t));
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(48, getPixelY(t));
      ctx.lineTo(W, getPixelY(t));
      ctx.stroke();
    }

    const xTrack1 = 65; // Lithology (width: 40)
    const xTrack2 = 125; // Seismic wiggle (width: 80)
    const xTrack3 = 225; // CWT (width: 200)
    const xTrack4 = 445; // SSWT (width: 200)

    // A. Draw Track 1: Lithology
    ctx.fillStyle = '#1e293b'; // Shale background
    ctx.fillRect(xTrack1, 25, 40, trackH);
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(xTrack1, 25, 40, trackH);

    // Draw sand intervals based on dynamic filter
    ctx.fillStyle = '#fbbf24'; // Yellow for Sand
    const cellSizeH = trackH / N;
    for (let n = 0; n < N; n++) {
      if (isSand[n] === 1) {
        const yPos = getPixelY(times[n]);
        ctx.fillRect(xTrack1 + 1, yPos - cellSizeH/2, 38, cellSizeH + 0.5);
      }
    }

    // Track Titles at the top
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText("LITHOLOGY", xTrack1 + 20, 15);
    ctx.fillText("REAL SEISMIC", xTrack2 + 40, 15);
    ctx.fillText("CWT (Morlet)", xTrack3 + 100, 15);
    ctx.fillText("SSWT (Squeezed)", xTrack4 + 100, 15);

    // B. Draw Track 2: Real Seismic Wiggle Trace
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(xTrack2, 25, 80, trackH);
    ctx.strokeRect(xTrack2, 25, 80, trackH);
    
    // Zero Center line
    ctx.strokeStyle = '#334155';
    ctx.beginPath();
    ctx.moveTo(xTrack2 + 40, 25);
    ctx.lineTo(xTrack2 + 40, 25 + trackH);
    ctx.stroke();

    // Wiggle path
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const y = getPixelY(times[i]);
      const x = xTrack2 + 40 + trace[i] * 35;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // C. Draw Track 3: Real CWT Spectrogram
    const cwtW = 200;
    const cellW_cwt = cwtW / cwtFreqSteps;
    ctx.strokeRect(xTrack3, 25, cwtW, trackH);

    for (let f = 0; f < cwtFreqSteps; f++) {
      const xPos = xTrack3 + f * cellW_cwt;
      for (let n = 0; n < N; n++) {
        const val = cwtAmp[f][n] / maxCwtVal;
        ctx.fillStyle = getJetColor(val);
        const yPos = getPixelY(times[n]);
        ctx.fillRect(xPos, yPos - cellSizeH/2, cellW_cwt + 0.5, cellSizeH + 0.5);
      }
    }

    // D. Draw Track 4: Real SSWT Spectrogram
    const ssimW = 200;
    const cellW_sswt = ssimW / cwtFreqSteps;
    ctx.strokeRect(xTrack4, 25, ssimW, trackH);

    for (let f = 0; f < cwtFreqSteps; f++) {
      const xPos = xTrack4 + f * cellW_sswt;
      for (let n = 0; n < N; n++) {
        const val = swtThresholdMetric(sswtMatrix[f][n], maxSswtVal);
        ctx.fillStyle = getHotColor(val);
        const yPos = getPixelY(times[n]);
        ctx.fillRect(xPos, yPos - cellSizeH/2, cellW_sswt + 0.5, cellSizeH + 0.5);
      }
    }

    // Label frequency limits
    ctx.fillStyle = '#64748b';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${minFreq}Hz`, xTrack3 + 2, H - 10);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxFreq}Hz`, xTrack3 + cwtW - 2, H - 10);
    
    ctx.textAlign = 'left';
    ctx.fillText(`${minFreq}Hz`, xTrack4 + 2, H - 10);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxFreq}Hz`, xTrack4 + ssimW - 2, H - 10);

  }, [activeSubTab, selectedWell, vshCutoff, cwtFreqSteps, minFreq, maxFreq]);

  // Utility logic to apply soft visual threshold to SSWT
  const swtThresholdMetric = (val, maxVal) => {
    const rawRatio = val / maxVal;
    if (rawRatio < 0.05) return 0;
    return rawRatio;
  };

  // Compute stats for current selected well to replace editor sliders
  const getWellReservoirStats = () => {
    const well = wellsData[selectedWell];
    if (!well || !well.samples) return { totalSands: 0, thickSands: 0, thickAvg: 0, timeSpan: 0 };
    
    const samples = well.samples;
    const times = samples.map(s => s.time || 0);
    const dt = times.length > 1 ? times[1] - times[0] : 2;
    
    // Dynamic Lithology Filter
    const isSand = samples.map(s => {
      const active = Math.abs(s.seismic_amp || 0) > 0.005;
      const vsh = s['VSH (Pred)'] !== undefined ? s['VSH (Pred)'] : 1.0;
      return (active && vsh < vshCutoff) ? 1 : 0;
    });
    
    let totalSands = 0;
    let currentThick = 0;
    let sandThickSum = 0;
    let sandCount = 0;
    
    const velocity = 3200.0; // m/s
    const sampleThickness = (dt / 1000.0) * velocity / 2.0;

    for (let i = 0; i < isSand.length; i++) {
      if (isSand[i] === 1) {
        currentThick += sampleThickness;
        totalSands += sampleThickness;
      } else {
        if (currentThick > 0) {
          sandThickSum += currentThick;
          sandCount += 1;
          currentThick = 0;
        }
      }
    }
    if (currentThick > 0) {
      sandThickSum += currentThick;
      sandCount += 1;
    }

    return {
      totalSands: totalSands.toFixed(1),
      sandCount: sandCount,
      thickAvg: sandCount > 0 ? (sandThickSum / sandCount).toFixed(1) : 0,
      timeSpan: (times[times.length - 1] - times[0]).toFixed(0)
    };
  };

  const stats = getWellReservoirStats();

  return (
    <div className="slide" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* Title block */}
      <div className="slide-title-area" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Layers size={24} style={{ color: 'var(--accent-gold)' }} />
            Synchrosqueezed Wavelet Transform (SSWT) Workspace
          </h2>
          <p className="slide-subtitle">Workstation-grade spectral decomposition tool compressing frequency smearing into high-resolution stratigraphic horizons.</p>
        </div>

        {/* Sub-tab Switcher */}
        <div className="glass-card" style={{ padding: '4px', display: 'flex', gap: '4px' }}>
          <button
            onClick={() => setActiveSubTab('workstation')}
            style={{
              padding: '6px 14px',
              fontSize: '12.5px',
              borderRadius: '6px',
              border: 'none',
              background: activeSubTab === 'workstation' ? 'var(--accent-gold)' : 'transparent',
              color: activeSubTab === 'workstation' ? '#000000' : 'var(--text-secondary)',
              fontWeight: 'bold',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            Workstation Well Logs
          </button>
          <button
            onClick={() => setActiveSubTab('profiler')}
            style={{
              padding: '6px 14px',
              fontSize: '12.5px',
              borderRadius: '6px',
              border: 'none',
              background: activeSubTab === 'profiler' ? 'var(--accent-gold)' : 'transparent',
              color: activeSubTab === 'profiler' ? '#000000' : 'var(--text-secondary)',
              fontWeight: 'bold',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            Stacked Reservoir Profiler (Accurate)
          </button>
          <button
            onClick={() => setActiveSubTab('concept')}
            style={{
              padding: '6px 14px',
              fontSize: '12.5px',
              borderRadius: '6px',
              border: 'none',
              background: activeSubTab === 'concept' ? 'var(--accent-gold)' : 'transparent',
              color: activeSubTab === 'concept' ? '#000000' : 'var(--text-secondary)',
              fontWeight: 'bold',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            SSWT Core Concept (Chirp Test)
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {activeSubTab === 'workstation' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '20px' }}>
          
          {/* Left Card: Vertical 8-Track Image Visualizer */}
          <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '15px', margin: 0, color: 'var(--text-primary)' }}>Workstation 8-Track Composite Log Plot</h3>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Well Z-04 seismic profile (TWT 2030–2654 ms)</span>
            </div>

            <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#090d16', padding: '10px' }}>
              <img 
                src="/well_images/cwt_swt_analysis.png" 
                alt="Vertical 8-Track SSWT/CWT/SWT Analysis" 
                style={{ width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }} 
              />
            </div>

            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center', marginTop: '6px' }}>
              {Object.keys(trackInfo).map(num => (
                <button
                  key={num}
                  onClick={() => setSelectedTrack(parseInt(num))}
                  style={{
                    padding: '4px 10px',
                    fontSize: '11.5px',
                    borderRadius: '4px',
                    border: `1px solid ${selectedTrack === parseInt(num) ? 'var(--accent-gold)' : 'var(--border-color)'}`,
                    background: selectedTrack === parseInt(num) ? 'rgba(251,191,36,0.12)' : 'var(--bg-secondary)',
                    color: selectedTrack === parseInt(num) ? 'var(--accent-gold)' : 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontWeight: selectedTrack === parseInt(num) ? 'bold' : 'normal',
                    transition: 'all 0.2s'
                  }}
                >
                  Track {num} Info
                </button>
              ))}
            </div>
          </div>

          {/* Right Column: Workstation Controls & Explanations */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Active Track Inspector */}
            <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--accent-gold)' }}>
              <div className="card-header" style={{ marginBottom: '12px' }}>
                <ZoomIn size={18} style={{ color: 'var(--accent-gold)' }} />
                <h3>Workstation Track Inspector</h3>
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <h4 style={{ margin: 0, fontSize: '14px', color: 'var(--text-primary)' }}>
                  {trackInfo[selectedTrack].title}
                </h4>
                <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                  {trackInfo[selectedTrack].desc}
                </p>
                <div style={{ backgroundColor: 'var(--bg-dark)', padding: '10px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', fontSize: '12px' }}>
                  <strong style={{ color: 'var(--accent-gold)', display: 'block', marginBottom: '2px' }}>Geologist Application:</strong>
                  <span style={{ color: 'var(--text-secondary)' }}>{trackInfo[selectedTrack].usecase}</span>
                </div>
              </div>
            </div>

            {/* Geological Explanation Card */}
            <div className="glass-card" style={{ padding: '20px' }}>
              <div className="card-header" style={{ marginBottom: '12px' }}>
                <Info size={18} style={{ color: 'var(--accent-blue)' }} />
                <h3>The Geophysics of Synchrosqueezing (SSWT)</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px', lineHeight: '1.5', color: 'var(--text-secondary)' }}>
                <p style={{ margin: 0 }}>
                  Standard **Continuous Wavelet Transform (CWT)** convolved with wavelets suffers from frequency smearing due to the Heisenberg uncertainty principle.
                </p>
                <p style={{ margin: 0 }}>
                  **Synchrosqueezing** takes the CWT output, calculates the phase derivatives to estimate the true **instantaneous frequency** of each signal component, and reallocates (squeezes) the energy.
                </p>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                    <CheckCircle2 size={14} style={{ color: '#22c55e', marginTop: '3px', flexShrink: 0 }} />
                    <span><strong>Zero Smearing:</strong> Sharp lines represent actual reflection boundaries, not mathematical blur.</span>
                  </div>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                    <CheckCircle2 size={14} style={{ color: '#22c55e', marginTop: '3px', flexShrink: 0 }} />
                    <span><strong>Thin-Bed Resolution:</strong> Isolates sand channel thickness (10–15m) below standard seismic tuning limits.</span>
                  </div>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                    <CheckCircle2 size={14} style={{ color: '#22c55e', marginTop: '3px', flexShrink: 0 }} />
                    <span><strong>Instantaneous Attributes:</strong> Precise depth localization of low-frequency gas cap shadows.</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ACCURATE STACKED RESERVOIR PROFILER */}
      {activeSubTab === 'profiler' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
          
          {/* Left Panel: Real-time Canvas Rendering */}
          <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '15px', margin: 0, color: 'var(--text-primary)' }}>Real-Well Stacked Reservoir Profiler</h3>
              <span className="badge badge-normal" style={{ fontSize: '11px', backgroundColor: 'rgba(56,189,248,0.12)', color: '#38bdf8', border: '1px solid rgba(56,189,248,0.2)' }}>
                Accurate Log Database Mode
              </span>
            </div>

            <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#0f172a', padding: '10px', display: 'flex', justifyContent: 'center' }}>
              <canvas 
                ref={profilerCanvasRef} 
                width={660} 
                height={380} 
                style={{ width: '100%', height: '100%', display: 'block' }}
              />
            </div>
            
            {/* Legend & Help info */}
            <div style={{ display: 'flex', gap: '20px', backgroundColor: 'var(--bg-dark)', padding: '10px 15px', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '12px' }}>
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                <div style={{ width: '12px', height: '12px', backgroundColor: '#fbbf24', borderRadius: '2px' }} />
                <span style={{ color: 'var(--text-secondary)' }}>Reservoir Sandstone (Log)</span>
              </div>
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                <div style={{ width: '12px', height: '12px', backgroundColor: '#1e293b', borderRadius: '2px', border: '1px solid #475569' }} />
                <span style={{ color: 'var(--text-secondary)' }}>Tight Shale/Silt (Log)</span>
              </div>
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                <div style={{ width: '12px', height: '12px', backgroundColor: '#38bdf8', borderRadius: '6px' }} />
                <span style={{ color: 'var(--text-secondary)' }}>Real Seismic Trace</span>
              </div>
            </div>
          </div>

          {/* Right Panel: Well Selector & Database Stats (ACCURATE) */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Well Selector Card */}
            <div className="glass-card" style={{ padding: '16px 20px', borderLeft: '4px solid var(--accent-gold)' }}>
              <div className="card-header" style={{ marginBottom: '14px' }}>
                <Database size={18} style={{ color: 'var(--accent-gold)' }} />
                <h3 style={{ margin: 0 }}>Real Well Selector</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '12.5px' }}>
                
                {/* Selector Dropdown */}
                <div>
                  <label style={{ display: 'block', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: 'semibold' }}>
                    Select Well to Analyze:
                  </label>
                  <select
                    value={selectedWell}
                    onChange={(e) => setSelectedWell(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      backgroundColor: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      border: '1px solid var(--border-color)',
                      outline: 'none',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: 'bold'
                    }}
                  >
                    {Object.keys(wellsData).map(name => (
                      <option key={name} value={name}>Well {name}</option>
                    ))}
                  </select>
                </div>

                {/* VSH Cutoff Calibration Slider (Crucial!) */}
                <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-primary)', fontWeight: 'bold', marginBottom: '4px' }}>
                    <span>Sand VSH Cutoff:</span>
                    <span style={{ fontFamily: 'monospace', color: 'var(--accent-gold)' }}>{vshCutoff.toFixed(2)}</span>
                  </div>
                  <p style={{ margin: '0 0 8px 0', fontSize: '11px', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                    Calibrate VSH threshold to isolate stacked thin-beds from thick sands.
                  </p>
                  <input 
                    type="range" min="0.05" max="0.25" step="0.01" value={vshCutoff} 
                    onChange={(e) => setVshCutoff(parseFloat(e.target.value))} 
                    style={{ width: '100%', height: '4px', cursor: 'pointer' }}
                  />
                </div>

                {/* Spectral settings */}
                <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ fontWeight: 'semibold', color: 'var(--text-muted)' }}>Spectral Analysis Parameters:</div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '8px', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Frequency Range:</span>
                    <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                      <input 
                        type="number" min="5" max="20" value={minFreq} 
                        onChange={(e) => setMinFreq(parseInt(e.target.value) || 10)} 
                        style={{ width: '40px', padding: '3px', backgroundColor: 'var(--bg-dark)', color: '#fff', border: '1px solid var(--border-color)', borderRadius: '4px', textAlign: 'center', fontSize: '11px' }}
                      />
                      <span style={{ color: 'var(--text-muted)' }}>to</span>
                      <input 
                        type="number" min="40" max="90" value={maxFreq} 
                        onChange={(e) => setMaxFreq(parseInt(e.target.value) || 60)} 
                        style={{ width: '40px', padding: '3px', backgroundColor: 'var(--bg-dark)', color: '#fff', border: '1px solid var(--border-color)', borderRadius: '4px', textAlign: 'center', fontSize: '11px' }}
                      />
                      <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Hz</span>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '8px', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Frequency Bins:</span>
                    <input 
                      type="range" min="15" max="40" value={cwtFreqSteps} 
                      onChange={(e) => setCwtFreqSteps(parseInt(e.target.value))} 
                      style={{ width: '100%', height: '4px', cursor: 'pointer' }}
                    />
                  </div>
                </div>

              </div>
            </div>

            {/* Geological Stats Report (ACCURATE) */}
            <div className="glass-card" style={{ padding: '20px' }}>
              <div className="card-header" style={{ marginBottom: '12px' }}>
                <MapPin size={18} style={{ color: 'var(--accent-blue)' }} />
                <h3>Stratigraphic Report: {selectedWell}</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
                  <span>Total Log Time Span:</span>
                  <strong style={{ color: 'var(--text-primary)' }}>{stats.timeSpan} ms</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
                  <span>Total Net Sandstone:</span>
                  <strong style={{ color: '#fbbf24' }}>{stats.totalSands} meters</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
                  <span>Discrete Sand Layers:</span>
                  <strong style={{ color: 'var(--accent-gold)' }}>{stats.sandCount} thin beds</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '2px' }}>
                  <span>Average Bed Thickness:</span>
                  <strong style={{ color: 'var(--text-primary)' }}>{stats.thickAvg} meters</strong>
                </div>
                
                <div style={{ marginTop: '10px', backgroundColor: 'rgba(56,189,248,0.06)', border: '1px dashed rgba(56,189,248,0.2)', padding: '10px', borderRadius: '6px', fontSize: '11.5px', lineHeight: '1.4' }}>
                  <strong style={{ color: '#38bdf8', display: 'block', marginBottom: '3px' }}>SSWT Resolution Summary:</strong>
                  The actual well logs show {stats.sandCount} stacked sandstone thin-beds averaging {stats.thickAvg} meters. On the left canvas, see how the standard CWT blurs these beds together, while the SSWT collapses the energy to their precise stratigraphic contact depths.
                </div>
              </div>
            </div>

          </div>
        </div>
      )}

      {/* CONCEPT DEMONSTRATION SUB-TAB */}
      {activeSubTab === 'concept' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '20px' }}>
          
          {/* Left Panel: Signal comparison Image */}
          <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '15px', margin: 0, color: 'var(--text-primary)' }}>Chirp & Step Signal: Time-Frequency Benchmark</h3>
              <span className="badge badge-normal" style={{ fontSize: '11px' }}>Signal Processing Stack Exchange Benchmark</span>
            </div>

            <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden', backgroundColor: 'var(--bg-dark)', padding: '10px' }}>
              <img 
                src="/well_images/sswt_concept_comparison.png" 
                alt="SSWT Concept: Signal, CWT, and SST comparison" 
                style={{ width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }} 
              />
            </div>
            
            <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center' }}>
              Synthetic test signal convolved at sampling frequency Fs = 1000 Hz across frequencies from 10 to 100 Hz.
            </p>
          </div>

          {/* Right Panel: Theoretical Explanations */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Chirp test breakdown */}
            <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--accent-blue)' }}>
              <div className="card-header" style={{ marginBottom: '12px' }}>
                <Sliders size={18} style={{ color: 'var(--accent-blue)' }} />
                <h3>Benchmark Signal Interpretation</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13.5px', lineHeight: '1.6' }}>
                <div>
                  <strong style={{ color: 'var(--text-primary)' }}>1. The Input Signal (Top Track):</strong>
                  <p style={{ margin: '4px 0 0 0', color: 'var(--text-secondary)' }}>
                    Starts with a **linear chirp** (frequency ramping up from 20 Hz to 80 Hz over 1.0 second), followed by a series of sudden **frequency steps** alternating between 20 Hz and 80 Hz.
                  </p>
                </div>
                
                <div>
                  <strong style={{ color: 'var(--text-primary)' }}>2. CWT Response (Middle Track):</strong>
                  <p style={{ margin: '4px 0 0 0', color: 'var(--text-secondary)' }}>
                    Notice how the frequency ramp looks like a **thick, blurry diagonal package**. At the constant frequency blocks, the energy spills out (smears) vertically, making the exact frequencies look fuzzy.
                  </p>
                </div>

                <div>
                  <strong style={{ color: 'var(--text-primary)' }}>3. SST / SSWT Response (Bottom Track):</strong>
                  <p style={{ margin: '4px 0 0 0', color: 'var(--text-secondary)' }}>
                    Notice the contrast: the diagonal ramp has collapsed into a **pixel-thin, sharp line**. The frequency step transitions have zero boundary blur and are perfectly resolved as crisp, square blocks.
                  </p>
                </div>
              </div>
            </div>

            {/* Geological Relevance */}
            <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--accent-gold)' }}>
              <div className="card-header" style={{ marginBottom: '12px' }}>
                <Info size={18} style={{ color: 'var(--accent-gold)' }} />
                <h3>Why This Matters for Reservoir Mapping</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px', lineHeight: '1.5', color: 'var(--text-secondary)' }}>
                <p style={{ margin: 0 }}>
                  In seismology, rock layers behave exactly like these **frequency steps**. When a seismic wave hits a sandstone top, it undergoes a step contrast in acoustic properties.
                </p>
                <p style={{ margin: 0 }}>
                  Using standard **CWT** means the top and base reflections of the sand body will bleed into each other, creating a single blurry reflection package. 
                </p>
                <p style={{ margin: 0, fontWeight: 'bold', color: 'var(--text-primary)' }}>
                  By applying **SSWT (SST)**, we compress this bleeding energy back to the actual boundary depths, allowing the geologist to map sand tops and bases with sub-seismic, layer-by-layer precision.
                </p>
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
