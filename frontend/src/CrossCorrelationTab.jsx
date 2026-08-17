import React, { useState } from 'react';
import { AreaChart, Sliders, Activity, Info, Link, ShieldCheck } from 'lucide-react';
import { wellsData } from './data';

export default function CrossCorrelationTab() {
  const [manualShift, setManualShift] = useState(0); // in samples
  const [noiseLevel, setNoiseLevel] = useState(0.2); // noise standard deviation
  const [traceType, setTraceType] = useState('Synthetic'); // synthetic or real well Z-04

  // Generate Reference and Target signals
  // Length: 100 samples
  const N = 100;
  
  // 1. Generate base signal (3 main reflection events convolved with a Ricker wavelet)
  const generateBaseSignal = () => {
    const sig = new Array(N).fill(0);
    // Reflection coefficients
    const rc = { 25: 0.8, 50: -0.6, 75: 0.5 };
    
    // Ricker wavelet convolve helper
    const f = 25; // frequency
    const dt = 0.002; // sampling
    
    for (let i = 0; i < N; i++) {
      let val = 0;
      Object.keys(rc).forEach(pos => {
        const rcPos = parseInt(pos);
        const t = (i - rcPos) * dt;
        const term = Math.PI * f * t;
        const ricker = (1 - 2 * term * term) * Math.exp(-term * term);
        val += rc[pos] * ricker;
      });
      sig[i] = val;
    }
    return sig;
  };

  const baseSignal = generateBaseSignal();

  // 2. Generate target signal with a true shift of +8 samples and added noise
  const trueShift = 8;
  const generateTargetSignal = () => {
    const sig = new Array(N).fill(0);
    
    // Seeded pseudo-random noise helper
    const getNoise = (idx) => {
      const x = Math.sin(idx * 12.9898 + 78.233) * 43758.5453;
      return (x - Math.floor(x)) - 0.5; // range [-0.5, 0.5]
    };

    for (let i = 0; i < N; i++) {
      // Apply true shift of +8 samples
      const refIdx = i - trueShift;
      const val = (refIdx >= 0 && refIdx < N) ? baseSignal[refIdx] : 0;
      const noise = getNoise(i) * 2.0 * noiseLevel; // scaled noise
      sig[i] = val + noise;
    }
    return sig;
  };

  const targetSignal = generateTargetSignal();

  // 3. Compute cross-correlation at all lags (from -30 to +30 samples)
  const maxLag = 30;
  const computeCrossCorrelation = () => {
    const xcorr = [];
    
    // Normalize helper
    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = (arr, m) => Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
    
    const meanBase = mean(baseSignal);
    const stdBase = std(baseSignal, meanBase);

    for (let lag = -maxLag; lag <= maxLag; lag++) {
      let sum = 0;
      let count = 0;
      const matchedBase = [];
      const matchedTarget = [];

      for (let i = 0; i < N; i++) {
        const targetIdx = i + lag;
        if (targetIdx >= 0 && targetIdx < N) {
          matchedBase.push(baseSignal[i]);
          matchedTarget.push(targetSignal[targetIdx]);
        }
      }

      if (matchedBase.length > 0) {
        const mB = mean(matchedBase);
        const mT = mean(matchedTarget);
        const sB = std(matchedBase, mB);
        const sT = std(matchedTarget, mT);
        
        let cov = 0;
        for (let j = 0; j < matchedBase.length; j++) {
          cov += (matchedBase[j] - mB) * (matchedTarget[j] - mT);
        }
        cov /= matchedBase.length;
        
        const corr = (sB * sT > 0) ? cov / (sB * sT) : 0;
        xcorr.push({ lag, correlation: corr });
      } else {
        xcorr.push({ lag, correlation: 0 });
      }
    }
    return xcorr;
  };

  const xcorrData = computeCrossCorrelation();

  // Find optimal shift (the lag with the highest correlation)
  let bestLag = 0;
  let bestCorr = -1;
  xcorrData.forEach(d => {
    if (d.correlation > bestCorr) {
      bestCorr = d.correlation;
      bestLag = d.lag;
    }
  });

  // Calculate current correlation coefficient for user's manual shift
  const currentCorrItem = xcorrData.find(d => d.lag === manualShift);
  const currentCorrelation = currentCorrItem ? currentCorrItem.correlation : 0;

  return (
    <div className="slide" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* Header */}
      <div>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Activity size={24} style={{ color: 'var(--accent-gold)' }} />
          Cross-Correlation Pattern Matching Sandbox
        </h2>
        <p className="slide-subtitle">Interactive laboratory demonstrating normalized cross-correlation mapping for time-depth well ties.</p>
      </div>

      {/* Workspace Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '20px' }}>
        
        {/* ========================================================
            LEFT COLUMN: INTERACTIVE SIGNAL MATCHING
            ======================================================== */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Signal Plot Panel */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Link size={18} style={{ color: 'var(--accent-blue)' }} />
                <h3 style={{ margin: 0, fontSize: '15px' }}>Trace Alignment Panel</h3>
              </div>
              
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                Target True Offset: <strong style={{ color: 'var(--accent-gold)' }}>+8 ms</strong>
              </div>
            </div>

            {/* Trace SVG display */}
            <div style={{ position: 'relative', width: '100%', height: '200px', backgroundColor: '#090d16', borderRadius: '8px', border: '1px solid var(--border-color)', padding: '10px' }}>
              <svg width="100%" height="100%" viewBox="0 0 500 180" preserveAspectRatio="none">
                {/* Horizontal reference zero */}
                <line x1="0" y1="90" x2="500" y2="90" stroke="#1e293b" strokeWidth="1" strokeDasharray="3 3" />
                
                {/* 1. Base Signal - Synthetic Seismogram (Blue) */}
                <path
                  d={`M ${baseSignal.map((v, i) => `${(i / (N - 1)) * 500} ${90 - v * 80}`).join(' L ')}`}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="2.0"
                />

                {/* 2. Target Signal - Seismic Trace (Red, shifted manually by user slider) */}
                {(() => {
                  const shiftedPoints = [];
                  for (let i = 0; i < N; i++) {
                    const srcIdx = i - manualShift;
                    const val = (srcIdx >= 0 && srcIdx < N) ? targetSignal[i] : 0;
                    shiftedPoints.push(`${(i / (N - 1)) * 500} ${90 - val * 80}`);
                  }
                  return (
                    <path
                      d={`M ${shiftedPoints.join(' L ')}`}
                      fill="none"
                      stroke="#f59e0b"
                      strokeWidth="1.8"
                    />
                  );
                })()}
              </svg>

              {/* Legend overlay */}
              <div style={{ position: 'absolute', bottom: '10px', left: '10px', display: 'flex', gap: '12px', fontSize: '11px', backgroundColor: 'rgba(9, 13, 22, 0.8)', padding: '4px 8px', borderRadius: '4px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#38bdf8' }}>
                  <span style={{ width: '8px', height: '2.5px', backgroundColor: '#38bdf8' }}></span> Synthetic Seismogram (Reference)
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#f59e0b' }}>
                  <span style={{ width: '8px', height: '2.5px', backgroundColor: '#f59e0b' }}></span> Shifted Seismic Trace (Target)
                </span>
              </div>
            </div>

            {/* Slider controls */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '16px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Manual Alignment Shift:</span>
                  <strong style={{ color: 'var(--accent-gold)' }}>{manualShift} ms (samples)</strong>
                </div>
                <input 
                  type="range" 
                  min="-20" 
                  max="20" 
                  value={manualShift} 
                  onChange={(e) => setManualShift(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Seismic Noise Level (σ):</span>
                  <strong>{(noiseLevel * 100).toFixed(0)}%</strong>
                </div>
                <input 
                  type="range" 
                  min="0.0" 
                  max="0.8" 
                  step="0.05"
                  value={noiseLevel} 
                  onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          </div>

          {/* Cross-Correlation Curve Panel */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', justifyStyle: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <AreaChart size={18} style={{ color: 'var(--accent-gold)' }} />
                <h3 style={{ margin: 0, fontSize: '15px' }}>Cross-Correlation Function R_xy(τ)</h3>
              </div>
            </div>

            {/* SVG Plot of correlation coefficients vs Lags */}
            <div style={{ position: 'relative', width: '100%', height: '160px', backgroundColor: '#090d16', borderRadius: '8px', border: '1px solid var(--border-color)', padding: '10px' }}>
              <svg width="100%" height="100%" viewBox="0 0 500 140" preserveAspectRatio="none">
                {/* Horizontal reference zero */}
                <line x1="0" y1="70" x2="500" y2="70" stroke="#1e293b" strokeWidth="1" strokeDasharray="3 3" />
                
                {/* Correlation Curve */}
                <path
                  d={`M ${xcorrData.map((d, i) => `${(i / (xcorrData.length - 1)) * 500} ${70 - d.correlation * 60}`).join(' L ')}`}
                  fill="none"
                  stroke="var(--accent-blue)"
                  strokeWidth="2.0"
                />

                {/* Highlight True Shift peak (dot) */}
                {(() => {
                  const trueIdx = trueShift + maxLag;
                  const trueCorr = xcorrData[trueIdx] ? xcorrData[trueIdx].correlation : 0;
                  const cx = (trueIdx / (xcorrData.length - 1)) * 500;
                  const cy = 70 - trueCorr * 60;
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r="5"
                      fill="#22c55e"
                      stroke="#ffffff"
                      strokeWidth="1.5"
                    />
                  );
                })()}

                {/* Highlight Manual Shift indicator (vertical line) */}
                {(() => {
                  const userIdx = manualShift + maxLag;
                  const userX = (userIdx / (xcorrData.length - 1)) * 500;
                  return (
                    <line
                      x1={userX}
                      y1="0"
                      x2={userX}
                      y2="140"
                      stroke="#f59e0b"
                      strokeWidth="1.5"
                      strokeDasharray="4 2"
                    />
                  );
                })()}
              </svg>

              {/* Labels for Lags */}
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 5px', fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                <span>-30 ms</span>
                <span>0 ms (No Shift)</span>
                <span>+30 ms</span>
              </div>
            </div>

            {/* Scorecard indicators */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '14px' }}>
              <div style={{ padding: '10px', borderRadius: '6px', background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.2)' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Current Manual Correlation:</span>
                <div style={{ fontSize: '18px', fontWeight: 800, color: 'var(--accent-gold)' }}>
                  {currentCorrelation.toFixed(4)}
                </div>
              </div>
              <div style={{ padding: '10px', borderRadius: '6px', background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.2)' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Best Mathematical Correlation:</span>
                <div style={{ fontSize: '18px', fontWeight: 800, color: '#22c55e' }}>
                  {bestCorr.toFixed(4)} <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>at +{bestLag} ms</span>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* ========================================================
            RIGHT COLUMN: THEORY & DATABASE READOUT
            ======================================================== */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Theory Panel */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="card-header" style={{ marginBottom: '12px' }}>
              <Info size={18} className="card-icon" style={{ color: 'var(--accent-blue)' }} />
              <h3>The Mathematics of Well Ties</h3>
            </div>
            
            <div style={{ fontSize: '13px', lineHeight: '1.6', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <p>
                <strong>Cross-correlation</strong> measures the degree to which two waveforms series reflect each other. For a reflection synthetic $S(t)$ and real seismic trace $R(t)$, the correlation $r$ at time lag $\tau$ is calculated as:
              </p>
              
              <div style={{ padding: '8px 12px', background: 'var(--bg-dark)', borderRadius: '6px', fontSize: '11.5px', fontFamily: 'monospace', textAlign: 'center', borderLeft: '3px solid var(--accent-blue)' }}>
                R_xy(τ) = Σ [ (x_t - μ_x)(y_(t+τ) - μ_y) ] / [ σ_x * σ_y ]
              </div>

              <p>
                A high cross-correlation coefficient value (above 0.6) indicates that the synthetic peaks (which represent major stratigraphic boundaries like sandstone tops) align precisely with the seismic reflections.
              </p>

              <div style={{ padding: '10px', background: 'rgba(34,197,94,0.05)', borderRadius: '6px', borderLeft: '3px solid #22c55e', fontSize: '12.5px' }}>
                <strong style={{ color: '#22c55e' }}>Geologist takeaway:</strong> Sliding the manual slider to **+8 ms** aligns the reflection pulses perfectly, peaking the correlation curve. This is exactly what the automated grid search does in Python to tie your wells!
              </div>
            </div>
          </div>

          {/* database Readout Panel */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="card-header" style={{ marginBottom: '14px' }}>
              <ShieldCheck size={18} className="card-icon" style={{ color: 'var(--accent-green)' }} />
              <h3>Workstation Database: Well Tie Correlation</h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '13px' }}>
              {Object.keys(wellsData).map(w => {
                const corr = wellsData[w].tie.correlation;
                return (
                  <div key={w} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
                    <span>Well <strong>{w}</strong> {w === 'Z-04' ? <span className="badge badge-sweet">BLIND</span> : <span className="badge badge-normal">TRAIN</span>}</span>
                    <span style={{ fontFamily: 'monospace', fontWeight: 'bold', color: corr >= 0.7 ? '#22c55e' : 'var(--accent-gold)' }}>
                      R = {corr.toFixed(4)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
