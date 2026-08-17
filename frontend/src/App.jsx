import React, { useState, useEffect, useRef } from 'react';
import { 
  MapPin, 
  Layers, 
  Cpu, 
  ShieldAlert, 
  Target, 
  AreaChart, 
  Table, 
  Sliders, 
  TrendingUp, 
  Database,
  SlidersHorizontal,
  Info,
  Maximize2,
  FileSpreadsheet,
  Image,
  Link,
  BarChart2,
  Zap,
  BookOpen
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceArea
} from 'recharts';
import { blindWellName, wellsConfig, wellsData } from './data';
import GridPredictorTab from './GridPredictorTab';
import ThinBedTab from './ThinBedTab';
import CwtSwtTab from './CwtSwtTab';
import CrossCorrelationTab from './CrossCorrelationTab';
import Volume3dTab from './Volume3dTab';
import R2ScorecardTab from './R2ScorecardTab';
import SswtTab from './SswtTab';
import SpectralWhiteningTab from './SpectralWhiteningTab';
import ThinBedWorkflowTab from './ThinBedWorkflowTab';
import SpectralEnhancementExplorerTab from './SpectralEnhancementExplorerTab';
import MlKinkExplorerTab from './MlKinkExplorerTab';
import MlV11PredictorTab from './MlV11PredictorTab';
import MlWellZoomTab from './MlWellZoomTab';
import SswtMlJourneyTab from './SswtMlJourneyTab';
import { gridData } from './grid_data';   // stub returns null until real data is ready
import { modelComparisonData } from './model_comparison_data';
import './App.css';

// -------------------------------------------------------------
// Core Signal Processing Utilities in JavaScript (FFT / DFT / Hilbert)
// -------------------------------------------------------------


const rotatePhaseJS = (wav, phaseDeg) => {
  const N = wav.length;
  // DFT
  const X = [];
  for (let k = 0; k < N; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += wav[n] * Math.cos(angle);
      im -= wav[n] * Math.sin(angle);
    }
    X.push({ re, im });
  }
  // Phase shift
  const phaseRad = (phaseDeg * Math.PI) / 180;
  for (let k = 0; k < N; k++) {
    let shift = 0;
    if (k > 0 && k < N / 2) {
      shift = phaseRad; // Positive frequencies
    } else if (k > N / 2) {
      shift = -phaseRad; // Negative frequencies
    }
    // Rotate
    const re = X[k].re * Math.cos(shift) - X[k].im * Math.sin(shift);
    const im = X[k].re * Math.sin(shift) + X[k].im * Math.cos(shift);
    X[k] = { re, im };
  }
  // IDFT
  const out = [];
  for (let n = 0; n < N; n++) {
    let sum = 0;
    for (let k = 0; k < N; k++) {
      const angle = (2 * Math.PI * k * n) / N;
      sum += X[k].re * Math.cos(angle) - X[k].im * Math.sin(angle);
    }
    out.push(sum / N);
  }
  return out;
};

const generateRicker = (f, phase, dtMs) => {
  const lengthMs = 100;
  const t = [];
  for (let val = -lengthMs / 2; val <= lengthMs / 2; val += dtMs) {
    t.push(val / 1000.0);
  }
  let wav = t.map(ti => {
    const term = Math.PI * f * ti;
    return (1 - 2 * term * term) * Math.exp(-term * term);
  });
  if (phase !== 0) {
    wav = rotatePhaseJS(wav, phase);
  }
  return wav;
};

const convolve = (refl, wav) => {
  const N = refl.length;
  const M = wav.length;
  const out = new Array(N).fill(0);
  const halfM = Math.floor(M / 2);
  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (let j = 0; j < M; j++) {
      const idx = i - j + halfM;
      if (idx >= 0 && idx < N) {
        sum += refl[idx] * wav[j];
      }
    }
    out[i] = sum;
  }
  return out;
};

const convolvePythonSame = (refl, wav) => {
  const N = refl.length;
  const M = wav.length;
  const out = new Array(M).fill(0);
  const fullLen = N + M - 1;
  const start = Math.floor((fullLen - M) / 2);
  for (let i = 0; i < M; i++) {
    const fullIdx = i + start;
    let sum = 0;
    for (let j = 0; j < M; j++) {
      const reflIdx = fullIdx - j;
      if (reflIdx >= 0 && reflIdx < N) {
        sum += refl[reflIdx] * wav[j];
      }
    }
    out[i] = sum;
  }
  return out;
};

const computeCorrelation = (x, y) => {
  const N = x.length;
  let meanX = 0, meanY = 0;
  for (let i = 0; i < N; i++) {
    meanX += x[i];
    meanY += y[i];
  }
  meanX /= N;
  meanY /= N;
  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < N; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  return denX * denY === 0 ? 0 : num / Math.sqrt(denX * denY);
};

const computeTieCorrelation = (trace, refl, regTimes, dtMs, freq, phase, shiftMs) => {
  const wav = generateRicker(freq, phase, dtMs);
  const syntheticFull = convolve(refl, wav);
  const nSyn = syntheticFull.length;
  const shiftedStart = regTimes[0] + shiftMs;
  const startIdx = regTimes.findIndex(t => Math.abs(t - shiftedStart) < dtMs / 2);
  if (startIdx === -1 || startIdx + nSyn > trace.length) return 0.0;
  const realSeg = trace.slice(startIdx, startIdx + nSyn);
  const seisRMS = Math.sqrt(realSeg.reduce((a, v) => a + v * v, 0) / realSeg.length);
  const synthRMS = Math.sqrt(syntheticFull.reduce((a, v) => a + v * v, 0) / syntheticFull.length);
  const scale = synthRMS > 0 ? (seisRMS / synthRMS) : 1.0;
  return computeCorrelation(realSeg, syntheticFull.map(v => v * scale));
};


const computeSpectrum = (trace, dtMs) => {
  const N = trace.length;
  const dt = dtMs / 1000.0;
  const samplingFreq = 1 / dt;
  const spectrum = [];
  for (let k = 0; k <= N / 2; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += trace[n] * Math.cos(angle);
      im -= trace[n] * Math.sin(angle);
    }
    const freq = (k * samplingFreq) / N;
    const amp = Math.sqrt(re * re + im * im) / N;
    spectrum.push({ freq, amp });
  }
  return spectrum;
};

const applyFrequencyFilter = (trace, lowCut, highCut, dtMs) => {
  const N = trace.length;
  const dt = dtMs / 1000.0;
  const samplingFreq = 1 / dt;
  
  // Forward DFT
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
  
  // Apply Bandpass Filter
  for (let k = 0; k < N; k++) {
    const freq = (k <= N / 2) ? (k * samplingFreq) / N : ((N - k) * samplingFreq) / N;
    if (freq < lowCut || freq > highCut) {
      X[k] = { re: 0, im: 0 };
    }
  }
  
  // Inverse DFT
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

const whitenTraceJS = (trace, dtMs) => {
  const N = trace.length;
  const dt = dtMs / 1000.0;
  const samplingFreq = 1 / dt;

  // 1. Forward DFT
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

  // 2. Compute amplitude spectrum and smooth it (moving average)
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

  // Water level
  let maxSmooth = 0;
  for (let i = 0; i < N; i++) {
    if (smoothAmp[i] > maxSmooth) maxSmooth = smoothAmp[i];
  }
  const waterLevel = 0.05 * maxSmooth;

  // 3. Whitening & Bandpass (Ormsby 8-12-55-70 Hz)
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

  // 4. Inverse DFT
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

// -------------------------------------------------------------
// Normalization limits for 2D map projection
// -------------------------------------------------------------
const xMin = 1201178.25, xMax = 1207337.37;
const yMin = 9679510.83, yMax = 9696292.65;
const mapX = (x) => 50 + ((x - xMin) / (xMax - xMin)) * 400;
const mapY = (y) => 450 - ((y - yMin) / (yMax - yMin)) * 400;

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [storyStep, setStoryStep] = useState(0);
  const [selectedWell, setSelectedWell] = useState('Z-04');
  const [modelType, setModelType] = useState('Raw'); // 'Calibrated' or 'Raw'
  const [targetProperty, setTargetProperty] = useState('GR');
  const [ssiCutoff, setSsiCutoff] = useState(0.035);
  const [porosityType, setPorosityType] = useState('PHIT'); // PHIT is recommended sibling
  const [showWiggles, setShowWiggles] = useState(true);

  // Interactive Well-Tie Simulator Parameters
  const [tieFreq, setTieFreq] = useState(15);
  const [tiePhase, setTiePhase] = useState(0);
  const [tieShift, setTieShift] = useState(0);
  const [tieImageType, setTieImageType] = useState('tie'); // 'tie' or 'inversion'


  // Spectral Decomposition Parameters
  const [lowCut, setLowCut] = useState(5);
  const [highCut, setHighCut] = useState(45);

  // Arbitrary Line Path Selection (wells in sequence)
  const [arbLineWells, setArbLineWells] = useState(['Z-03', 'Z-07', 'Z-04']);
  const [arbLineProperty, setArbLineProperty] = useState('GR');
  const [showOnlyLoggedInterval, setShowOnlyLoggedInterval] = useState(true);
  const [selectedGalleryImgId, setSelectedGalleryImgId] = useState('seismic');
  const [sectionHovered, setSectionHovered] = useState(null);
  const [selectedModelTarget, setSelectedModelTarget] = useState(null);
  const canvasRef = useRef(null);




  // Reset simulator values when selected well changes
  useEffect(() => {
    if (wellsData[selectedWell]) {
      const t = wellsData[selectedWell].tie;
      setTieFreq(t.freq_hz);
      setTiePhase(t.phase_deg);
      setTieShift(t.shift_ms);
    }
  }, [selectedWell]);

  // Compute Well-Tie synthetic traces interactively
  const currentWell = wellsData[selectedWell];
  const dtMs = 2.0;

  let interactiveSynthetic = [];
  let interactiveCorrelation = 0.0;

  if (currentWell) {
    // Full reflectivity and trace arrays
    const refl = currentWell.samples.map(s => s.reflectivity);
    const trace = currentWell.samples.map(s => s.seismic_amp);
    const regTimes = currentWell.samples.map(s => s.time);

    // 1. Correlation using backend‑compatible algorithm
    interactiveCorrelation = computeTieCorrelation(trace, refl, regTimes, dtMs, tieFreq, tiePhase, tieShift);

    // 2. Generate synthetic for chart display (shifted for visualisation)
    let shiftedRefl = [...refl];
    const shiftSamples = Math.round(tieShift / dtMs);
    if (shiftSamples !== 0) {
      if (shiftSamples > 0) {
        shiftedRefl = [...new Array(shiftSamples).fill(0), ...refl.slice(0, -shiftSamples)];
      } else {
        const absShift = Math.abs(shiftSamples);
        shiftedRefl = [...refl.slice(absShift), ...new Array(absShift).fill(0)];
      }
    }
    const wav = generateRicker(tieFreq, tiePhase, dtMs);
    const synthRaw = convolve(shiftedRefl, wav);
    
    const seisRMS = Math.sqrt(trace.reduce((acc, v) => acc + v*v, 0) / trace.length);
    const synthRMS = Math.sqrt(synthRaw.reduce((acc, v) => acc + v*v, 0) / synthRaw.length);
    const scale = synthRMS > 0 ? (seisRMS / synthRMS) : 1.0;
    interactiveSynthetic = synthRaw.map(v => v * scale);
  }

  // Live DFT calculation on raw trace
  const rawTrace = currentWell ? currentWell.samples.map(s => s.seismic_amp) : [];
  const rawSpectrum = currentWell ? computeSpectrum(rawTrace, dtMs) : [];
  const filteredTrace = currentWell ? applyFrequencyFilter(rawTrace, lowCut, highCut, dtMs) : [];
  const filteredSpectrum = currentWell ? computeSpectrum(filteredTrace, dtMs) : [];

  // Combine interactive synthetic trace with samples
  const tieChartData = currentWell ? currentWell.samples.map((s, idx) => ({
    time: s.time,
    reflectivity: s.reflectivity,
    seismic: s.seismic_amp,
    synthetic: interactiveSynthetic[idx]
  })) : [];

  // Spectral decomposition comparison data
  const specChartData = rawSpectrum.map((s, idx) => ({
    freq: Math.round(s.freq),
    rawAmp: s.amp,
    filtAmp: filteredSpectrum[idx] ? filteredSpectrum[idx].amp : 0
  })).filter(s => s.freq <= 100); // Display up to 100 Hz

  // Trace filtering comparison data
  const filterTraceChartData = currentWell ? currentWell.samples.map((s, idx) => ({
    time: s.time,
    rawAmp: s.seismic_amp,
    filtAmp: filteredTrace[idx]
  })) : [];

  // Process selected well logs to compute dynamic Sweet Spot Index (SSI)
  // SSI = Porosity * (1 - VSH) * (1 - SWE)
  const processedSamples = currentWell ? currentWell.samples.map(s => {
    const phiKey = porosityType === 'PHIE' ? 'PHIE' : 'PHIT';
    const predSuff = modelType === 'Calibrated' ? '' : ' Raw';
    
    const actPhi = s[`${phiKey} (Act)`];
    const actVsh = s["VSH (Act)"];
    const actSwe = s["SWE (Act)"];
    const actSSI = (actPhi !== null && actVsh !== null && actSwe !== null)
      ? actPhi * (1 - actVsh) * (1 - actSwe)
      : null;

    const predPhi = s[`${phiKey} (Pred${predSuff})`];
    const predVsh = s[`VSH (Pred${predSuff})`];
    const predSwe = s[`SWE (Pred${predSuff})`];
    const predSSI = predPhi * (1 - predVsh) * (1 - predSwe);

    return {
      time: s.time,
      actVal: s[`${targetProperty} (Act)`],
      predVal: s[`${targetProperty} (Pred${predSuff})`],
      actPhi,
      predPhi,
      actVsh,
      predVsh: s[`VSH (Pred${predSuff})`],
      actSwe,
      predSwe: s[`SWE (Pred${predSuff})`],
      actSSI,
      predSSI,
      isSweetSpot: predSSI >= ssiCutoff,
      grAct: s["GR (Act)"],
      grPred: s[`GR (Pred${predSuff})`],
      dtAct: s["DT (Act)"],
      dtPred: s[`DT (Pred${predSuff})`]
    };
  }) : [];


  const sweetSpotCount = processedSamples.filter(s => s.isSweetSpot).length;

  // Toggle wells in arbitrary line
  const handleWellMapClick = (wName) => {
    if (arbLineWells.includes(wName)) {
      setArbLineWells(arbLineWells.filter(w => w !== wName));
    } else {
      setArbLineWells([...arbLineWells, wName]);
    }
  };

  // Handle mouse movements over the 2D seismic profile to calculate geological coordinates
  const handleSectionMouseMove = (e) => {
    if (!canvasRef.current || arbLineWells.length < 2 || !wellsData[arbLineWells[0]]) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    const offsetX = 55;
    const rightMargin = 20;
    const offsetY = 15;
    const bottomOffsetY = 40;
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;
    
    const plotW = canvas.width - offsetX - rightMargin;
    const plotH = canvas.height - offsetY - bottomOffsetY;
    
    if (clickX < offsetX || clickX > canvas.width - rightMargin || clickY < offsetY || clickY > canvas.height - bottomOffsetY) {
      setSectionHovered(null);
      return;
    }
    
    const fracX = (clickX - offsetX) / plotW;
    const fracY = (clickY - offsetY) / plotH;
    
    let maxLen = 0;
    arbLineWells.forEach(well => {
      if (wellsData[well]) maxLen = Math.max(maxLen, wellsData[well].samples.length);
    });
    if (maxLen === 0) return;
    
    const sampleIdx = Math.floor(fracY * maxLen);
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
    
    const getVal = (well) => {
      const sample = well.samples[Math.min(sampleIdx, well.samples.length - 1)];
      if (!sample) return 0;
      if (arbLineProperty === 'Seismic') {
        return sample.seismic_amp;
      } else {
        const suffix = modelType === 'Calibrated' ? '' : ' Raw';
        return sample[`${arbLineProperty} (Pred${suffix})`] || 0;
      }
    };
    
    const valA = getVal(wellA);
    const valB = getVal(wellB);
    const val = (1 - pFrac) * valA + pFrac * valB;
    
    setSectionHovered({ il, xl, twt, val });
  };

  // Draw 2D Cross-section of arbitrary line path using HTML5 Canvas
  useEffect(() => {
    if (activeTab === 'map' && canvasRef.current && arbLineWells.length >= 2) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform to avoid translation leak
      const W = canvas.width;
      const H = canvas.height;
      
      // Clear canvas
      ctx.fillStyle = '#0f172a'; // Darker premium navy background
      ctx.fillRect(0, 0, W, H);

      // Find max trace length across selected wells
      let maxLen = 0;
      arbLineWells.forEach(well => {
        if (wellsData[well]) {
          maxLen = Math.max(maxLen, wellsData[well].samples.length);
        }
      });

      if (maxLen === 0) return;

      const nPanels = arbLineWells.length - 1;
      
      // Margins for ticks & labels
      const offsetX = 55;
      const rightMargin = 20;
      const offsetY = 15;
      const bottomOffsetY = 40;
      const plotW = W - offsetX - rightMargin;
      const plotH = H - offsetY - bottomOffsetY;
      const colsPerPanel = Math.floor(plotW / nPanels);

      // ── Pass 1: Draw Variable Density Background ──
      for (let p = 0; p < nPanels; p++) {
        const wellA = wellsData[arbLineWells[p]];
        const wellB = wellsData[arbLineWells[p+1]];
        if (!wellA || !wellB) continue;

        const startX = offsetX + p * colsPerPanel;
        const endX = offsetX + (p + 1) * colsPerPanel;

        for (let x = startX; x < endX; x++) {
          const frac = (x - startX) / colsPerPanel;
          
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sampleIdx = Math.floor(fracY * maxLen);
            
            // Extract attributes or predicted values
            const getVal = (well) => {
              const sample = well.samples[Math.min(sampleIdx, well.samples.length - 1)];
              if (!sample) return 0;
              
              if (arbLineProperty === 'Seismic') {
                return sample.seismic_amp;
              } else {
                const suffix = modelType === 'Calibrated' ? '' : ' Raw';
                return sample[`${arbLineProperty} (Pred${suffix})`] || 0;
              }
            };

            const valA = getVal(wellA);
            const valB = getVal(wellB);
            const val = (1 - frac) * valA + frac * valB;

            // Apply distinct geological palettes
            let color = '';
            if (arbLineProperty === 'Seismic') {
              // Standard Red-White-Blue seismic palette
              const norm = Math.max(-1, Math.min(1, val / 15000));
              if (norm > 0) {
                const r = 255;
                const g = Math.round(255 * (1 - norm));
                const b = Math.round(255 * (1 - norm));
                color = `rgb(${r}, ${g}, ${b})`;
              } else {
                const absNorm = Math.abs(norm);
                const r = Math.round(255 * (1 - absNorm));
                const g = Math.round(255 * (1 - absNorm));
                const b = 255;
                color = `rgb(${r}, ${g}, ${b})`;
              }
            } else if (arbLineProperty === 'GR') {
              // Gamma Ray: Yellow (Sand/Clean) to Brown (Shale/Clay)
              const norm = Math.max(0, Math.min(1, (val - 35) / 80));
              const r = Math.round(255 - norm * 140);
              const g = Math.round(235 - norm * 150);
              const b = Math.round(150 - norm * 110);
              color = `rgb(${r}, ${g}, ${b})`;
            } else if (arbLineProperty.startsWith('PH')) {
              // Porosity: Indigo (Tight) to Teal/Cyan (Porous Sand)
              const norm = Math.max(0, Math.min(1, val / 0.15));
              const r = Math.round(15 + norm * 40);
              const g = Math.round(23 + norm * 180);
              const b = Math.round(42 + norm * 210);
              color = `rgb(${r}, ${g}, ${b})`;
            } else if (arbLineProperty === 'SWE') {
              // Saturation: Red (Gas/Oil) to light-blue (Wet)
              const norm = Math.max(0, Math.min(1, val));
              const r = Math.round(239 * (1 - norm) + 50 * norm);
              const g = Math.round(68 * (1 - norm) + 130 * norm);
              const b = Math.round(68 * (1 - norm) + 246 * norm);
              color = `rgb(${r}, ${g}, ${b})`;
            } else {
              // Greyscale fallback
              const norm = Math.max(0, Math.min(255, val * 255));
              color = `rgb(${norm}, ${norm}, ${norm})`;
            }

            ctx.fillStyle = color;
            ctx.fillRect(x, y, 1, 1);
          }
        }
      }

      // ── Pass 2: Draw Seismic Wiggle Overlay (If Seismic property is selected) ──
      if (arbLineProperty === 'Seismic') {
        const nWiggles = 28;
        const spacing = plotW / nWiggles;
        
        for (let w = 1; w < nWiggles; w++) {
          const wiggleX = offsetX + w * spacing;
          const frac = (wiggleX - offsetX) / plotW;
          
          // Determine panel
          const p = Math.floor(frac * nPanels);
          const pFrac = (frac - p / nPanels) * nPanels;
          const wellA = wellsData[arbLineWells[p]];
          const wellB = wellsData[arbLineWells[p+1]];
          if (!wellA || !wellB) continue;

          // Draw positive lobe shading (black fill to the right)
          ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
          ctx.beginPath();
          ctx.moveTo(wiggleX, offsetY);
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sampleIdx = Math.floor(fracY * maxLen);
            const valA = wellA.samples[Math.min(sampleIdx, wellA.samples.length - 1)]?.seismic_amp || 0;
            const valB = wellB.samples[Math.min(sampleIdx, wellB.samples.length - 1)]?.seismic_amp || 0;
            const val = (1 - pFrac) * valA + pFrac * valB;
            const norm = val / 25000;
            const dx = Math.max(0, norm * (spacing * 0.7)); // fill positive lobes only
            ctx.lineTo(wiggleX + dx, y);
          }
          ctx.lineTo(wiggleX, offsetY + plotH);
          ctx.closePath();
          ctx.fill();

          // Draw wiggle line
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sampleIdx = Math.floor(fracY * maxLen);
            const valA = wellA.samples[Math.min(sampleIdx, wellA.samples.length - 1)]?.seismic_amp || 0;
            const valB = wellB.samples[Math.min(sampleIdx, wellB.samples.length - 1)]?.seismic_amp || 0;
            const val = (1 - pFrac) * valA + pFrac * valB;
            const norm = val / 25000;
            const dx = norm * (spacing * 0.7);
            if (y === offsetY) ctx.moveTo(wiggleX + dx, y);
            else ctx.lineTo(wiggleX + dx, y);
          }
          ctx.stroke();
        }
      }

      // ── Pass 3: Draw Wavy Geological Horizons (Stratigraphy) ──
      const drawHorizon = (yFraction, name, color) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let x = offsetX; x < offsetX + plotW; x++) {
          const frac = (x - offsetX) / plotW;
          // Subtly wavy geological dip profile
          const wave = Math.sin(frac * Math.PI * 2) * 9 + Math.cos(frac * Math.PI * 4) * 3;
          const y = offsetY + yFraction * plotH + wave;
          if (x === offsetX) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = color;
        ctx.font = 'italic bold 9.5px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(name, offsetX + plotW - 12, offsetY + yFraction * plotH - 6);
        ctx.restore();
      };

      drawHorizon(0.42, "Top Reservoir Horizon", "rgba(255, 255, 255, 0.85)");
      drawHorizon(0.68, "Base Reservoir Horizon", "rgba(255, 255, 255, 0.85)");

      // ── Pass 4: Draw Y-Axis (Two-Way Time in ms) & Gridlines ──
      ctx.fillStyle = '#94a3b8';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      
      const tStart = wellsData[arbLineWells[0]].samples[0].time;
      const tEnd = wellsData[arbLineWells[0]].samples[wellsData[arbLineWells[0]].samples.length - 1].time;
      
      for (let t = Math.ceil(tStart / 50) * 50; t <= tEnd; t += 50) {
        const frac = (t - tStart) / (tEnd - tStart);
        const y = offsetY + frac * plotH;
        
        // Tick mark
        ctx.beginPath();
        ctx.moveTo(offsetX - 4, y);
        ctx.lineTo(offsetX, y);
        ctx.stroke();
        
        // Gridline across section
        ctx.beginPath();
        ctx.moveTo(offsetX, y);
        ctx.lineTo(offsetX + plotW, y);
        ctx.stroke();
        
        // Label
        ctx.fillText(`${t} ms`, offsetX - 8, y);
      }

      // Y-Axis Title
      ctx.save();
      ctx.translate(14, offsetY + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText("Two-Way Time (ms)", 0, 0);
      ctx.restore();

      // ── Pass 5: Draw Well Borehole Markers, Coordinate Labels & Actual Log Overlays ──
      ctx.lineWidth = 2.0;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.font = 'bold 11px Inter, sans-serif';
      
      arbLineWells.forEach((wName, idx) => {
        const xPos = offsetX + idx * colsPerPanel;
        
        // Draw vertical borehole line
        ctx.beginPath();
        ctx.moveTo(xPos, offsetY);
        ctx.lineTo(xPos, offsetY + plotH);
        ctx.stroke();

        // Draw Sand/Shale facies column next to borehole
        const well = wellsData[wName];
        if (well && well.samples) {
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sampleIdx = Math.floor(fracY * maxLen);
            const sample = well.samples[Math.min(sampleIdx, well.samples.length - 1)];
            if (sample) {
              const isSand = sample.is_sand ?? 0;
              ctx.fillStyle = isSand === 1 ? '#22c55e' : '#475569'; // Green sand, slate shale
              ctx.fillRect(xPos - 6, y, 4, 1);
            }
          }
        }
        
        // Draw well name flag
        ctx.fillStyle = wName === 'Z-04' ? 'var(--accent-red)' : 'var(--accent-blue)';
        ctx.fillRect(xPos - 4, offsetY + 6, wName === 'Z-04' ? 62 : 46, 16);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(wName + (wName === 'Z-04' ? ' (BLIND)' : ''), xPos + 4, offsetY + 18);

        // Draw coordinate details along bottom of borehole
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        
        if (well) {
          ctx.fillText(`IL ${well.inline}`, xPos, offsetY + plotH + 14);
          ctx.fillText(`XL ${well.crossline}`, xPos, offsetY + plotH + 26);
        }

        // Overlay actual log curve if property is not Seismic
        if (well && arbLineProperty !== 'Seismic') {
          ctx.strokeStyle = '#ef4444'; // Red log trace
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sampleIdx = Math.floor(fracY * maxLen);
            const sample = well.samples[Math.min(sampleIdx, well.samples.length - 1)];
            if (sample) {
              const val = sample[`${arbLineProperty} (Act)`] ?? sample[`${arbLineProperty} (Pred)`] ?? 0;
              let norm = 0;
              if (arbLineProperty === 'GR') {
                norm = (val - 20) / 110;
              } else if (arbLineProperty.startsWith('PH')) {
                norm = val / 0.18;
              } else if (arbLineProperty === 'SWE') {
                norm = val;
              }
              // Map to 25px offset curve
              const curveX = xPos + Math.max(0, Math.min(1, norm)) * 25;
              if (y === offsetY) ctx.moveTo(curveX, y);
              else ctx.lineTo(curveX, y);
            }
          }
          ctx.stroke();
        }
      });
    }
  }, [activeTab, arbLineWells, arbLineProperty, modelType]);

  return (
    <div className="app-container">
      {/* Premium Workstation Header */}
      <header className="workstation-header">
        <div className="brand-section">
          <span className="brand-logo">LMKR</span>
          <span className="brand-subtitle">PETROPHYSICAL & GEOPHYSICAL CO-KNOWLEDGE WORKSTATION</span>
        </div>
        <div className="status-indicator">
          <div className="status-dot"></div>
          <span>Active Field: 3D Seismic Volume Study (6 Borehole Ties Active)</span>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="workspace">
        {/* Navigation Sidebar */}
        <aside className="sidebar">
          <div>
            <div className="nav-title">Geophysicist Menu</div>
            <nav className="nav-list">
              <button 
                className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                <Database size={18} />
                <span>Executive Summary</span>
              </button>
              
              <button 
                className={`nav-item ${activeTab === 'map' ? 'active' : ''}`}
                onClick={() => setActiveTab('map')}
              >
                <MapPin size={18} />
                <span>Borehole Map & Section</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'workflow_story' ? 'active' : ''}`}
                onClick={() => setActiveTab('workflow_story')}
              >
                <Info size={18} style={{ color: 'var(--accent-gold)' }} />
                <span style={{ fontWeight: 'bold', color: 'var(--accent-gold)' }}>Pipeline Upgrades & Verification</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'tie' ? 'active' : ''}`}
                onClick={() => setActiveTab('tie')}
              >
                <Layers size={18} />
                <span>Well-Seismic Tie Simulator</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'spectral' ? 'active' : ''}`}
                onClick={() => setActiveTab('spectral')}
              >
                <TrendingUp size={18} />
                <span>Spectral Decomposition</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'prediction' ? 'active' : ''}`}
                onClick={() => setActiveTab('prediction')}
              >
                <Cpu size={18} />
                <span>ML Property Predictor</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'table' ? 'active' : ''}`}
                onClick={() => setActiveTab('table')}
              >
                <Table size={18} />
                <span>Comparison spreadsheet</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'grid' ? 'active' : ''}`}
                onClick={() => setActiveTab('grid')}
              >
                <Target size={18} />
                <span>Grid Predictor Map</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'gallery' ? 'active' : ''}`}
                onClick={() => setActiveTab('gallery')}
              >
                <Image size={18} />
                <span>Geologist Gallery</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'thinbed' ? 'active' : ''}`}
                onClick={() => setActiveTab('thinbed')}
              >
                <SlidersHorizontal size={18} />
                <span>Thin-Bed Workbench</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'cwt_swt' ? 'active' : ''}`}
                onClick={() => setActiveTab('cwt_swt')}
              >
                <TrendingUp size={18} style={{ color: 'var(--accent-gold)' }} />
                <span style={{ fontWeight: 'bold', color: 'var(--accent-gold)' }}>CWT/SWT Study</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'sswt_analyst' ? 'active' : ''}`}
                onClick={() => setActiveTab('sswt_analyst')}
              >
                <Maximize2 size={18} style={{ color: 'var(--accent-gold)' }} />
                <span style={{ fontWeight: 'bold', color: 'var(--accent-gold)' }}>SSWT Analyst</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'xcorr' ? 'active' : ''}`}
                onClick={() => setActiveTab('xcorr')}
              >
                <Link size={18} style={{ color: 'var(--accent-gold)' }} />
                <span style={{ fontWeight: 'bold', color: 'var(--accent-gold)' }}>Cross-Correlation Study</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'volume3d' ? 'active' : ''}`}
                onClick={() => setActiveTab('volume3d')}
              >
                <Maximize2 size={18} style={{ color: 'var(--accent-gold)' }} />
                <span style={{ fontWeight: 'bold', color: 'var(--accent-gold)' }}>3D Reservoir Viewer</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'r2_scorecard' ? 'active' : ''}`}
                onClick={() => setActiveTab('r2_scorecard')}
              >
                <FileSpreadsheet size={18} style={{ color: 'var(--accent-gold)' }} />
                <span style={{ fontWeight: 'bold', color: 'var(--accent-gold)' }}>Geological R² Scorecard</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'spectral_whitening' ? 'active' : ''}`}
                onClick={() => setActiveTab('spectral_whitening')}
              >
                <BarChart2 size={18} style={{ color: '#34d399' }} />
                <span style={{ fontWeight: 'bold', color: '#34d399' }}>Spectral Whitening</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'thin_bed_frequency' ? 'active' : ''}`}
                onClick={() => setActiveTab('thin_bed_frequency')}
              >
                <Layers size={18} style={{ color: '#10b981' }} />
                <span style={{ fontWeight: 'bold', color: '#10b981' }}>Thin Bed Frequency</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'spectral_explorer' ? 'active' : ''}`}
                onClick={() => setActiveTab('spectral_explorer')}
              >
                <Layers size={18} style={{ color: '#ef4444' }} />
                <span style={{ fontWeight: 'bold', color: '#ef4444' }}>Spectral Explorer</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'ml_kink_explorer' ? 'active' : ''}`}
                onClick={() => setActiveTab('ml_kink_explorer')}
              >
                <Zap size={18} style={{ color: '#fbbf24' }} />
                <span style={{ fontWeight: 'bold', color: '#fbbf24' }}>ML SSWT Kink Explorer</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'ml_v11_predictor' ? 'active' : ''}`}
                onClick={() => setActiveTab('ml_v11_predictor')}
                style={{
                  background: activeTab === 'ml_v11_predictor' ? 'rgba(56, 189, 248, 0.25)' : 'rgba(56, 189, 248, 0.08)',
                  border: '1px solid #38bdf8',
                  marginTop: '6px'
                }}
              >
                <Zap size={18} style={{ color: '#38bdf8' }} />
                <span style={{ fontWeight: 'bold', color: '#38bdf8' }}>V11 ML 3D Seismic Predictor</span>
                <span className="badge badge-normal" style={{ marginLeft: 'auto', background: '#0284c7', color: '#fff', fontSize: '9px', padding: '2px 5px' }}>NEW</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'ml_well_zoom' ? 'active' : ''}`}
                onClick={() => setActiveTab('ml_well_zoom')}
                style={{
                  background: activeTab === 'ml_well_zoom' ? 'rgba(236, 72, 153, 0.25)' : 'rgba(236, 72, 153, 0.08)',
                  border: '1px solid #ec4899',
                  marginTop: '6px'
                }}
              >
                <Maximize2 size={18} style={{ color: '#ec4899' }} />
                <span style={{ fontWeight: 'bold', color: '#ec4899' }}>HD Well Seismic Zoom</span>
                <span className="badge badge-sweet" style={{ marginLeft: 'auto', background: '#db2777', color: '#fff', fontSize: '9px', padding: '2px 5px' }}>ZOOM</span>
              </button>

              <button 
                className={`nav-item ${activeTab === 'sswt_journey' ? 'active' : ''}`}
                onClick={() => setActiveTab('sswt_journey')}
                style={{
                  background: activeTab === 'sswt_journey' ? 'rgba(251, 191, 36, 0.25)' : 'rgba(251, 191, 36, 0.08)',
                  border: '1px solid #fbbf24',
                  marginTop: '6px'
                }}
              >
                <BookOpen size={18} style={{ color: '#fbbf24' }} />
                <span style={{ fontWeight: 'bold', color: '#fbbf24' }}>SSWT ML Methodology & Story</span>
                <span className="badge badge-sweet" style={{ marginLeft: 'auto', fontSize: '9px', padding: '2px 5px' }}>STORY</span>
              </button>

            </nav>
          </div>
          
          <div className="sidebar-footer">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div>Active Well: <strong style={{ color: 'var(--text-primary)' }}>{selectedWell}</strong></div>
              <div>Model Path: <span style={{ fontFamily: 'monospace', fontSize: '10px' }}>ml_outputs_v11/</span></div>
            </div>
          </div>
        </aside>

        {/* Workstation Display Screen */}
        <main className="main-display">
          
          {/* TAB 1: EXECUTIVE SUMMARY */}
          {activeTab === 'overview' && (
            <div className="slide">
              <div className="slide-title-area">
                <h2>Field Study Executive Summary: Well-Seismic ML Prediction</h2>
                <p className="slide-subtitle">Cross-well calibration workflow leveraging regularized regression pipelines and calibrated OOF predictions.</p>
              </div>

              <div className="slide-description-card glass-card">
                <div className="card-header">
                  <Database size={20} className="card-icon" />
                  <h3>Workstation Workflow Statement</h3>
                </div>
                <p>
                  This workstation integrates raw seismic trace reflections with 6 geological wells. By extracting attributes at mapped coordinates and block-averaging targets down to 2ms seismic resolution, we train multi-variate tree regressors to invert for sub-seismic petrophysical targets.
                </p>
                <p>
                  We have resolved baseline amplitude scaling issues, division blowups in features, and implemented a robust **Leave-One-Group-Out Out-Of-Fold (OOF)** calibration logic to handle structural/dip facies offsets across geological zones.
                </p>
              </div>

              <div className="dashboard-grid">
                <div className="glass-card">
                  <div className="card-header">
                    <Target size={18} className="card-icon" style={{ color: 'var(--accent-green)' }} />
                    <h3>Workstation Well Database</h3>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '13.5px' }}>
                    {Object.keys(wellsData).map(w => (
                      <div key={w} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
                        <span>Well <strong>{w}</strong> {w === 'Z-04' ? <span className="badge badge-sweet">BLIND TEST</span> : <span className="badge badge-normal">TRAIN</span>}</span>
                        <span style={{ color: 'var(--text-secondary)' }}>Tie Corr: <strong>{wellsData[w].tie.correlation.toFixed(3)}</strong></span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-card">
                  <div className="card-header">
                    <TrendingUp size={18} className="card-icon" style={{ color: 'var(--accent-blue)' }} />
                    <h3>Resolution Status</h3>
                  </div>
                  <ul className="card-bullet-list">
                    <li><strong className="text-highlight">MURHO (Shear Rigidity)</strong>: CV R² = -0.381 | Blind R² = +0.357 (Sand-Only RF)</li>
                    <li><strong className="text-highlight">GR (Gamma Ray)</strong>: CV R² = -0.080 | Blind R² = -0.066 (Bayesian Ridge)</li>
                    <li><strong className="text-highlight">RHOB (Density)</strong>: CV R² = -0.231 | Blind R² = -0.045 (Gradient Boosting)</li>
                    <li><strong className="text-highlight">SWE (Water Saturation)</strong>: CV R² = -0.114 | Blind R² = -0.025 (ElasticNet Sand-Only)</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'workflow_story' && (
            <div className="slide">
              <div className="slide-title-area">
                <h2>Pipeline Upgrades & Verification</h2>
                <p className="slide-subtitle">An interactive walk-through of the physical and algorithmic upgrades implemented to improve blind well predictions.</p>
              </div>

              {/* Progress Stepper */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px', backgroundColor: 'var(--bg-card)', padding: '15px 20px', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                {[
                  { label: "1. Well-Seismic Tie", desc: "Time-Depth Alignment" },
                  { label: "2. Rock Physics", desc: "Elastic Constant Derivation" },
                  { label: "3. Facies Classifier", desc: "Reservoir Zonation" },
                  { label: "4. Compaction Residuals", desc: "Compaction Trend Removal" },
                  { label: "5. Performance Scorecard", desc: "Z-04 Performance" }
                ].map((step, idx) => (
                  <div 
                    key={idx} 
                    onClick={() => setStoryStep(idx)}
                    style={{ 
                      flex: 1, 
                      textAlign: 'center', 
                      cursor: 'pointer',
                      borderBottom: `3px solid ${storyStep === idx ? 'var(--accent-gold)' : (storyStep > idx ? 'var(--accent-blue)' : 'var(--border-color)')}`,
                      paddingBottom: '8px',
                      opacity: storyStep === idx ? 1 : 0.6,
                      transition: 'all 0.2s'
                    }}
                  >
                    <div style={{ fontSize: '12px', fontWeight: 'bold', color: storyStep === idx ? 'var(--accent-gold)' : (storyStep > idx ? 'var(--accent-blue)' : 'var(--text-secondary)') }}>{step.label}</div>
                    <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginTop: '2px' }}>{step.desc}</div>
                  </div>
                ))}
              </div>

              {/* Stepper Content */}
              <div className="dashboard-grid" style={{ gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                
                {/* Left Card: Upgrades and Reasoning */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px', padding: '24px' }}>
                  
                  {storyStep === 0 && (
                    <>
                      <div className="card-header">
                        <Layers size={22} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
                        <h3>Step 1: Well-Seismic Tie & Synthetics</h3>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '14px', lineHeight: '1.6' }}>
                        <div style={{ borderLeft: '3px solid var(--accent-red)', paddingLeft: '12px', backgroundColor: 'rgba(239, 68, 68, 0.03)', paddingBottom: '6px' }}>
                          <strong style={{ color: '#f87171' }}>Before (Previous Pipeline):</strong> Well-ties had time-shift misalignment. The extraction loop pulled seismic attributes from the wrong depths, mapping properties to the wrong reflections during training.
                        </div>
                        <div style={{ borderLeft: '3px solid var(--accent-green)', paddingLeft: '12px', backgroundColor: 'rgba(34, 197, 94, 0.03)', paddingBottom: '6px' }}>
                          <strong style={{ color: '#4ade80' }}>After (Upgraded Pipeline):</strong> We implemented an automated fine-grid 15-degree phase rotation and correlation search. This produces a wavelet-free trace tie that aligns time and depth exactly.
                        </div>
                        <p style={{ color: 'var(--text-secondary)' }}><em>The ML model now extracts features from the exact reflecting horizons, establishing a solid physical baseline.</em></p>
                      </div>
                    </>
                  )}

                  {storyStep === 1 && (
                    <>
                      <div className="card-header">
                        <Database size={22} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
                        <h3>Step 2: Elastic Rock Physics Logs</h3>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '14px', lineHeight: '1.6' }}>
                        <div style={{ borderLeft: '3px solid var(--accent-red)', paddingLeft: '12px', backgroundColor: 'rgba(239, 68, 68, 0.03)', paddingBottom: '6px' }}>
                          <strong style={{ color: '#f87171' }}>Before (Previous Pipeline):</strong> The pipeline only predicted simple logs (GR, density, porosity). GR alone cannot distinguish gas sands from tight shales because their radiation responses can overlap.
                        </div>
                        <div style={{ borderLeft: '3px solid var(--accent-green)', paddingLeft: '12px', backgroundColor: 'rgba(34, 197, 94, 0.03)', paddingBottom: '6px' }}>
                          <strong style={{ color: '#4ade80' }}>After (Upgraded Pipeline):</strong> We derived compressional/shear velocities using the Castagna Mudrock Line to calculate:
                          <ul style={{ paddingLeft: '20px', marginTop: '6px', listStyleType: 'disc' }}>
                            <li><strong>Vp/Vs Ratio</strong> and <strong>Poisson's Ratio</strong> (lower in gas sands)</li>
                            <li><strong>Lambda-Rho ($\lambda\rho$)</strong> (fluid compressibility)</li>
                            <li><strong>Mu-Rho ($\mu\rho$)</strong> (sand skeleton rigidity)</li>
                          </ul>
                        </div>
                        <p style={{ color: 'var(--text-secondary)' }}><em>These new curves are trained alongside seismic attributes, allowing the model to invert directly for fluid and frame properties.</em></p>
                      </div>
                    </>
                  )}

                  {storyStep === 2 && (
                    <>
                      <div className="card-header">
                        <Cpu size={22} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
                        <h3>Step 3: Facies-First Hierarchy</h3>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '14px', lineHeight: '1.6' }}>
                        <div style={{ borderLeft: '3px solid var(--accent-red)', paddingLeft: '12px', backgroundColor: 'rgba(239, 68, 68, 0.03)', paddingBottom: '6px' }}>
                          <strong style={{ color: '#f87171' }}>Before (Previous Pipeline):</strong> Regression models predicted porosity and water saturation across the whole well. This created false predictions (e.g. 5% effective porosity) in tight shales.
                        </div>
                        <div style={{ borderLeft: '3px solid var(--accent-green)', paddingLeft: '12px', backgroundColor: 'rgba(34, 197, 94, 0.03)', paddingBottom: '6px' }}>
                          <strong style={{ color: '#4ade80' }}>After (Upgraded Pipeline):</strong> We built a Sand/Shale Classifier (accuracy 96.1% on Z-04). Inside shales, the pipeline now automatically clamps:
                          <ul style={{ paddingLeft: '20px', marginTop: '6px', listStyleType: 'disc' }}>
                            <li>Effective Porosity (`PHIE`) $\to$ **0.0**</li>
                            <li>Water Saturation (`SWE`) $\to$ **1.0** (100% water-wet shale)</li>
                          </ul>
                          Porosity and saturation regression models now specialize only inside predicted sand zones.
                        </div>
                        <p style={{ color: 'var(--text-secondary)' }}><em>This separates the reservoir skeleton from tight zones before predicting reservoir quality.</em></p>
                      </div>
                    </>
                  )}

                  {storyStep === 3 && (
                    <>
                      <div className="card-header">
                        <TrendingUp size={22} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
                        <h3>Step 4: Compaction Trend-Residuals</h3>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '14px', lineHeight: '1.6' }}>
                        <div style={{ borderLeft: '3px solid var(--accent-red)', paddingLeft: '12px', backgroundColor: 'rgba(239, 68, 68, 0.03)', paddingBottom: '6px' }}>
                          <strong style={{ color: '#f87171' }}>Before (Previous Pipeline):</strong> The model tried to predict absolute values directly from seismic. When evaluated on the blind well Z-04 at a different depth window, the model drifted wildly, predicting impossible values.
                        </div>
                        <div style={{ borderLeft: '3px solid var(--accent-green)', paddingLeft: '12px', backgroundColor: 'rgba(34, 197, 94, 0.03)', paddingBottom: '6px' }}>
                          <strong style={{ color: '#4ade80' }}>After (Upgraded Pipeline):</strong> We fit degree-2 polynomial trends vs seismic time. The machine learning model now only predicts the residual anomaly (contrasts). The background trend is added back mathematically.
                        </div>
                        <p style={{ color: 'var(--text-secondary)' }}><em>This guarantees that the blind well prediction remains strictly bounded by realistic geological compaction limits at any depth.</em></p>
                      </div>
                    </>
                  )}

                  {storyStep === 4 && (
                    <>
                      <div className="card-header">
                        <Target size={22} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
                        <h3>Step 5: Z-04 Blind Well Scorecard</h3>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '14px', lineHeight: '1.6' }}>
                        <p>A blind test is the ultimate validation. We hold out well Z-04 during training and evaluate the model's predictive accuracy.</p>
                        <p>Here is a comparison of blind well Z-04 performance before (v3) and after (v4) implementing physical constraints:</p>
                        
                        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '10px', fontSize: '13px' }}>
                          <thead>
                            <tr style={{ borderBottom: '2px solid var(--border-color)', textAlign: 'left', color: 'var(--text-secondary)' }}>
                              <th style={{ padding: '8px' }}>Log Target</th>
                              <th style={{ padding: '8px' }}>v3 Blind R²</th>
                              <th style={{ padding: '8px' }}>v5 Blind R²</th>
                              <th style={{ padding: '8px' }}>Δ Improvement</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                              <td style={{ padding: '8px', fontWeight: 'bold' }}>Acoustic Impedance (AI)</td>
                              <td style={{ padding: '8px', color: '#94a3b8' }}>N/A (v3)</td>
                              <td style={{ padding: '8px', color: '#22c55e', fontWeight: 'bold' }}>+0.4076 ✓</td>
                              <td style={{ padding: '8px', color: '#22c55e' }}>Positive R²! (Extra Trees)</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                              <td style={{ padding: '8px', fontWeight: 'bold' }}>Mu-Rho (MURHO)</td>
                              <td style={{ padding: '8px', color: '#94a3b8' }}>N/A (v3)</td>
                              <td style={{ padding: '8px', color: '#22c55e', fontWeight: 'bold' }}>+0.3723 ✓</td>
                              <td style={{ padding: '8px', color: '#22c55e' }}>Positive R²! (Extra Trees)</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                              <td style={{ padding: '8px', fontWeight: 'bold' }}>Gamma Ray (GR)</td>
                              <td style={{ padding: '8px' }}>-0.6567</td>
                              <td style={{ padding: '8px', color: '#22c55e', fontWeight: 'bold' }}>-0.0655</td>
                              <td style={{ padding: '8px', color: '#22c55e' }}>+0.5912</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                              <td style={{ padding: '8px', fontWeight: 'bold' }}>Sonic (DT)</td>
                              <td style={{ padding: '8px' }}>-9.3501</td>
                              <td style={{ padding: '8px', color: '#22c55e', fontWeight: 'bold' }}>+0.0343 ✓</td>
                              <td style={{ padding: '8px', color: '#22c55e' }}>+9.3844 (Positive R²!)</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                              <td style={{ padding: '8px', fontWeight: 'bold' }}>Density (RHOB)</td>
                              <td style={{ padding: '8px' }}>-0.5172</td>
                              <td style={{ padding: '8px', color: '#22c55e', fontWeight: 'bold' }}>-0.0452</td>
                              <td style={{ padding: '8px', color: '#22c55e' }}>+0.4720</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}

                  {/* Stepper Navigation Buttons */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 'auto', paddingTop: '20px', borderTop: '1px solid var(--border-color)' }}>
                    <button
                      onClick={() => setStoryStep(Math.max(0, storyStep - 1))}
                      disabled={storyStep === 0}
                      style={{
                        padding: '8px 16px',
                        borderRadius: '6px',
                        border: '1px solid var(--border-color)',
                        backgroundColor: 'transparent',
                        color: 'var(--text-primary)',
                        cursor: storyStep === 0 ? 'not-allowed' : 'pointer',
                        opacity: storyStep === 0 ? 0.4 : 1
                      }}
                    >
                      ← Previous
                    </button>
                    
                    <button
                      onClick={() => setStoryStep(Math.min(4, storyStep + 1))}
                      disabled={storyStep === 4}
                      style={{
                        padding: '8px 24px',
                        borderRadius: '6px',
                        backgroundColor: 'var(--accent-gold)',
                        color: '#000000',
                        fontWeight: 'bold',
                        border: 'none',
                        cursor: storyStep === 4 ? 'not-allowed' : 'pointer',
                        opacity: storyStep === 4 ? 0.4 : 1
                      }}
                    >
                      Next Step →
                    </button>
                  </div>

                </div>

                {/* Right Card: Dynamic Visual Comparisons / QC Plots */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '15px', padding: '24px', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0f172a' }}>
                  
                  {storyStep === 0 && (
                    <div style={{ width: '100%', textAlign: 'center' }}>
                      <h4 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>Well-Seismic Alignment QC</h4>
                      <img 
                        src={`/well_images/${selectedWell}_tie.png`} 
                        alt="Well Tie QC Plot" 
                        style={{ width: '90%', borderRadius: '8px', border: '1.5px solid var(--border-color)', maxHeight: '380px', objectFit: 'contain' }}
                        onError={(e) => { e.target.src = '/well_images/Z-04_tie.png' }}
                      />
                      <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px' }}>Interactive synthetic seismogram (red) vs. real seismic trace (black) alignment QC.</p>
                    </div>
                  )}

                  {storyStep === 1 && (
                    <div style={{ width: '100%', textAlign: 'center' }}>
                      <h4 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>Castagna Rock Physics Crossplot</h4>
                      <img 
                        src={`/well_images/${selectedWell}_rockphysics.png`} 
                        alt="Rock Physics Crossplot" 
                        style={{ width: '90%', borderRadius: '8px', border: '1.5px solid var(--border-color)', maxHeight: '380px', objectFit: 'contain' }}
                        onError={(e) => { e.target.src = '/well_images/Z-04_rockphysics.png' }}
                      />
                      <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px' }}>Lambda-Rho vs. Mu-Rho crossplot colored by Gamma Ray, identifying fluid & frame rigidity clusters.</p>
                    </div>
                  )}

                  {storyStep === 2 && (
                    <div style={{ width: '100%', textAlign: 'center' }}>
                      <h4 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>Blind Z-04 Facies Classification QC</h4>
                      <img 
                        src="/well_images/facies_blind_well_test.png" 
                        alt="Facies Classifier Validation" 
                        style={{ width: '80%', borderRadius: '8px', border: '1.5px solid var(--border-color)', maxHeight: '380px', objectFit: 'contain' }}
                        onError={(e) => { e.target.style.display = 'none' }}
                      />
                      <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px' }}>Validation curve for Z-04: Predicted Sand/Shale block indicators vs. true LAS logs.</p>
                    </div>
                  )}

                  {storyStep === 3 && (
                    <div style={{ width: '100%', textAlign: 'center' }}>
                      <h4 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>Compaction Residual Prediction QC (Z-04 GR)</h4>
                      <img 
                        src="/well_images/GR_blind_well_test.png" 
                        alt="Residual Trend Prediction" 
                        style={{ width: '80%', borderRadius: '8px', border: '1.5px solid var(--border-color)', maxHeight: '380px', objectFit: 'contain' }}
                        onError={(e) => { e.target.style.display = 'none' }}
                      />
                      <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px' }}>Actual log curve (black) vs. trend-residual prediction (green dashed) on blind well Z-04.</p>
                    </div>
                  )}

                  {storyStep === 4 && (
                    <div style={{ width: '100%', textAlign: 'center' }}>
                      <h4 style={{ color: 'var(--accent-gold)', marginBottom: '12px', fontWeight: 'bold' }}>Model Capabilities Scorecard</h4>
                      <div style={{ textAlign: 'left', display: 'flex', flexDirection: 'column', gap: '12px', color: 'var(--text-primary)', fontSize: '13px' }}>
                        <div style={{ borderLeft: '3px solid #22c55e', paddingLeft: '8px', backgroundColor: 'rgba(34, 197, 94, 0.03)' }}>
                          <strong>Effective Porosity & Saturation:</strong> Excellent at isolating thin reservoir sand boundaries (sand-only specialist filters out false anomalies in shales).
                        </div>
                        <div style={{ borderLeft: '3px solid #22c55e', paddingLeft: '8px', backgroundColor: 'rgba(34, 197, 94, 0.03)' }}>
                          <strong>Shear Rigidity (Mu-Rho):</strong> Achieved the first positive blind R² ($+0.082$), proving the model successfully tracks structural frame rigidity.
                        </div>
                      <div style={{ borderLeft: '3px solid #3b82f6', paddingLeft: '8px', backgroundColor: 'rgba(59, 130, 246, 0.03)' }}>
                          <strong>Bulk Compaction Logs (GR / Density):</strong> Robust residual trend fitting anchors the logs to geologically realistic limits at any TWT time.
                        </div>
                      </div>
                    </div>
                  )}

                </div>

              </div>

              {/* ── Model Explorer ── */}
              <div style={{ marginTop: '32px' }}>
                <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '28px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                    <Cpu size={18} style={{ color: 'var(--accent-gold)' }} />
                    <h3 style={{ color: 'var(--text-primary)', fontSize: '15px', fontWeight: 700, margin: 0 }}>Model Explorer — Select a Target to Inspect</h3>
                  </div>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '18px', marginLeft: '28px' }}>
                    Click any target property to compare all 3 candidate models · CV = LOWO cross-validation on 5 training wells · Blind = held-out well Z-04
                  </p>

                  {/* Target pills */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '20px' }}>
                    {modelComparisonData.map(item => {
                      const isActive = selectedModelTarget === item.target;
                      const bestM = item.models.find(m => m.isBest);
                      const blindR2 = bestM ? bestM.blindR2 : null;
                      const isPositive = blindR2 !== null && blindR2 > 0;
                      return (
                        <button
                          key={item.target}
                          onClick={() => setSelectedModelTarget(isActive ? null : item.target)}
                          style={{
                            padding: '6px 14px',
                            borderRadius: '999px',
                            border: isActive ? '2px solid var(--accent-gold)' : '1.5px solid var(--border-color)',
                            backgroundColor: isActive ? 'rgba(251,191,36,0.12)' : 'var(--bg-secondary)',
                            color: isActive ? 'var(--accent-gold)' : 'var(--text-primary)',
                            fontWeight: isActive ? 700 : 500,
                            cursor: 'pointer',
                            fontSize: '12px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            transition: 'all 0.2s',
                          }}
                        >
                          {item.target}
                          {isPositive && (
                            <span style={{ fontSize: '10px', color: '#22c55e', fontWeight: 700 }}>+R²</span>
                          )}
                        </button>
                      );
                    })}
                  </div>

                  {/* Expanded detail card */}
                  {selectedModelTarget && (() => {
                    const item = modelComparisonData.find(d => d.target === selectedModelTarget);
                    if (!item) return null;
                    const modelColors = { 'Random Forest': '#3b82f6', 'Extra Trees': '#8b5cf6', 'XGBoost': '#f59e0b' };
                    return (
                      <div style={{
                        backgroundColor: '#0a1628',
                        border: '1px solid #1e3a5f',
                        borderRadius: '12px',
                        padding: '22px',
                        animation: 'fadeInUp 0.25s ease'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '18px' }}>
                          <div>
                            <h4 style={{ color: 'var(--text-primary)', fontSize: '14px', fontWeight: 700, margin: 0 }}>
                              {selectedModelTarget} — Model Comparison
                            </h4>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '11px', marginTop: '3px' }}>
                              Selected model: <span style={{ color: '#22c55e', fontWeight: 700 }}>{item.bestModel}</span>
                            </p>
                          </div>
                          <button
                            onClick={() => setSelectedModelTarget(null)}
                            style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '18px', lineHeight: 1 }}
                          >✕</button>
                        </div>

                        {/* Model rows */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '20px' }}>
                          {item.models.map(m => {
                            const col = modelColors[m.name] || '#64748b';
                            const isB = m.isBest;
                            return (
                              <div key={m.name} style={{
                                backgroundColor: isB ? 'rgba(20,83,45,0.5)' : 'rgba(30,41,59,0.6)',
                                border: `1.5px solid ${isB ? '#22c55e' : '#1e3348'}`,
                                borderRadius: '8px',
                                padding: '12px 16px',
                                display: 'grid',
                                gridTemplateColumns: '180px 1fr 1fr 1fr 1fr',
                                alignItems: 'center',
                                gap: '8px',
                              }}>
                                {/* Model name */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                  {isB && <span style={{ color: '#fbbf24', fontSize: '14px' }}>★</span>}
                                  <span style={{
                                    color: isB ? '#86efac' : col,
                                    fontWeight: 700, fontSize: '13px'
                                  }}>{m.name}</span>
                                </div>
                                {/* CV R² */}
                                <div style={{ textAlign: 'center' }}>
                                  <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '2px' }}>CV R²</div>
                                  <div style={{
                                    fontSize: '13px', fontWeight: 600,
                                    color: m.cvR2 >= 0 ? '#22c55e' : '#94a3b8'
                                  }}>{m.cvR2 >= 0 ? '+' : ''}{m.cvR2}</div>
                                </div>
                                {/* CV MAE */}
                                <div style={{ textAlign: 'center' }}>
                                  <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '2px' }}>CV MAE</div>
                                  <div style={{ fontSize: '13px', fontWeight: 600, color: '#e2e8f0' }}>{m.cvMae}</div>
                                </div>
                                {/* Blind R² */}
                                <div style={{ textAlign: 'center' }}>
                                  <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '2px' }}>Blind R²</div>
                                  <div style={{
                                    fontSize: '13px', fontWeight: 600,
                                    color: m.blindR2 !== null && m.blindR2 >= 0 ? '#22c55e' : '#f87171'
                                  }}>
                                    {m.blindR2 !== null ? (m.blindR2 >= 0 ? '+' : '') + m.blindR2 : 'N/A'}
                                  </div>
                                </div>
                                {/* Blind MAE */}
                                <div style={{ textAlign: 'center' }}>
                                  <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '2px' }}>Blind MAE</div>
                                  <div style={{ fontSize: '13px', fontWeight: 600, color: '#e2e8f0' }}>
                                    {m.blindMae !== null ? m.blindMae : 'N/A'}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>

                        {/* Top features of best model */}
                        {(() => {
                          const best = item.models.find(m => m.isBest);
                          if (!best || !best.topFeatures.length) return null;
                          return (
                            <div>
                              <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '8px', fontWeight: 600 }}>
                                Top-5 Seismic Drivers ({best.name})
                              </p>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                {best.topFeatures.map((f, i) => (
                                  <span key={f} style={{
                                    padding: '3px 10px',
                                    borderRadius: '999px',
                                    backgroundColor: i === 0 ? 'rgba(139,92,246,0.25)' : 'rgba(30,58,94,0.6)',
                                    border: `1px solid ${i === 0 ? '#8b5cf6' : '#1e3a5e'}`,
                                    fontSize: '11px',
                                    color: i === 0 ? '#c4b5fd' : '#94a3b8',
                                    fontWeight: i === 0 ? 700 : 400,
                                  }}>
                                    {i === 0 ? '🏆 ' : `${i+1}. `}{f}
                                  </span>
                                ))}
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    );
                  })()}
                </div>
              </div>

            </div>
          )}

          {/* TAB 2: INTERACTIVE WELL MAP & ARBITRARY LINE */}
          {activeTab === 'map' && (
            <div className="slide">
              <div className="slide-title-area">
                <h2>Field Well Map & Arbitrary Cross-Section</h2>
                <p className="slide-subtitle">Draw arbitrary lines across well nodes to visualize seismic and ML log predictions in a vertical section panel.</p>
              </div>

              <div className="map-workspace">
                {/* 2D Interactive Map */}
                <div className="glass-card map-panel">
                  <div className="card-header">
                    <MapPin size={18} className="card-icon" />
                    <h3>1. Surface Map (Click wells in sequence to construct line)</h3>
                  </div>
                  <div className="map-wrapper" style={{ position: 'relative', width: '100%', height: '460px', backgroundColor: '#0f172a', borderRadius: '8px', overflow: 'hidden' }}>
                    <svg width="100%" height="100%" style={{ position: 'absolute', top: 0, left: 0 }}>
                      {/* Grid Lines */}
                      <line x1="50" y1="0" x2="50" y2="460" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="150" y1="0" x2="150" y2="460" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="250" y1="0" x2="250" y2="460" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="350" y1="0" x2="350" y2="460" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="450" y1="0" x2="450" y2="460" stroke="#1e293b" strokeDasharray="3 3" />
                      
                      <line x1="0" y1="50" x2="500" y2="50" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="0" y1="150" x2="500" y2="150" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="0" y1="250" x2="500" y2="250" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="0" y1="350" x2="500" y2="350" stroke="#1e293b" strokeDasharray="3 3" />
                      <line x1="0" y1="450" x2="500" y2="450" stroke="#1e293b" strokeDasharray="3 3" />

                      {/* Arbitrary Line Path */}
                      {arbLineWells.length >= 2 && (
                        <path
                          d={`M ${arbLineWells.map(w => `${mapX(wellsConfig[w].x)} ${mapY(wellsConfig[w].y)}`).join(' L ')}`}
                          fill="none"
                          stroke="var(--accent-gold)"
                          strokeWidth="3.5"
                          strokeDasharray="4 4"
                          style={{ animation: 'dash 15s linear infinite' }}
                        />
                      )}

                      {/* Map Nodes */}
                      {Object.keys(wellsConfig).map(wName => {
                        const well = wellsConfig[wName];
                        const selected = arbLineWells.includes(wName);
                        const isBlind = wName === blindWellName;
                        return (
                          <g key={wName} style={{ cursor: 'pointer' }} onClick={() => handleWellMapClick(wName)}>
                            <circle
                              cx={mapX(well.x)}
                              cy={mapY(well.y)}
                              r={selected ? 11 : 7}
                              fill={isBlind ? 'var(--accent-red)' : (selected ? 'var(--accent-gold)' : 'var(--accent-blue)')}
                              stroke="#ffffff"
                              strokeWidth="2.5"
                              style={{ transition: 'all 0.2s' }}
                            />
                            <text
                              x={mapX(well.x) + 12}
                              y={mapY(well.y) + 4}
                              fill="#ffffff"
                              fontSize="12"
                              fontWeight="bold"
                              fontFamily="var(--font-sans)"
                            >
                              {wName} {isBlind ? '(Blind)' : ''}
                            </text>
                            {/* Well Index number in line */}
                            {selected && (
                              <circle
                                cx={mapX(well.x)}
                                cy={mapY(well.y)}
                                r="4"
                                fill="#ffffff"
                              />
                            )}
                          </g>
                        );
                      })}
                    </svg>
                    
                    <div style={{ position: 'absolute', bottom: '12px', left: '12px', display: 'flex', gap: '12px', fontSize: '11px', color: '#94a3b8', backgroundColor: 'rgba(15, 23, 42, 0.8)', padding: '6px 12px', borderRadius: '4px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-blue)' }}></span> Training Well
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-red)' }}></span> Blind Well
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-gold)' }}></span> Mapped on Line
                      </div>
                    </div>
                  </div>
                </div>

                {/* Section View */}
                <div className="glass-card section-panel">
                  <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Layers size={18} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
                      <h3>2. 2D Arbitrary Section Profile</h3>
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Property:</span>
                      <select 
                        value={arbLineProperty} 
                        onChange={(e) => setArbLineProperty(e.target.value)}
                        style={{
                          padding: '6px 12px',
                          borderRadius: '8px',
                          border: '1px solid rgba(99, 102, 241, 0.35)',
                          backgroundColor: '#0f172a',
                          color: '#e2e8f0',
                          fontWeight: '600',
                          fontSize: '11px',
                          outline: 'none',
                          cursor: 'pointer',
                          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                          transition: 'border-color 0.2s'
                        }}
                      >
                        <option value="Seismic">Raw Seismic Amplitude</option>
                        <option value="GR">Gamma Ray (GR)</option>
                        <option value="PHIT">Total Porosity (PHIT)</option>
                        <option value="SWE">Water Saturation (SWE)</option>
                      </select>

                      <select
                        value={modelType}
                        onChange={(e) => setModelType(e.target.value)}
                        style={{
                          padding: '6px 12px',
                          borderRadius: '8px',
                          border: '1px solid rgba(99, 102, 241, 0.35)',
                          backgroundColor: '#0f172a',
                          color: '#e2e8f0',
                          fontWeight: '600',
                          fontSize: '11px',
                          outline: 'none',
                          cursor: 'pointer',
                          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                          transition: 'border-color 0.2s'
                        }}
                      >
                        <option value="Calibrated">Calibrated ML</option>
                        <option value="Raw">Raw ML</option>
                      </select>
                    </div>
                  </div>

                  <div className="canvas-wrapper" style={{ width: '100%', height: '410px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                    {arbLineWells.length >= 2 ? (
                      <div style={{ position: 'relative', width: '100%', height: '380px' }}>
                        <canvas 
                          ref={canvasRef} 
                          width={600} 
                          height={380} 
                          style={{ width: '100%', height: '380px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'block', cursor: 'crosshair' }}
                          onMouseMove={handleSectionMouseMove}
                          onMouseLeave={() => setSectionHovered(null)}
                        />
                      </div>
                    ) : (
                      <div style={{ color: 'var(--text-muted)', fontSize: '14px', textAlign: 'center' }}>
                        Select at least 2 wells on the map to draw a cross-section line.
                      </div>
                    )}
                    {sectionHovered && (
                      <div style={{ 
                        marginTop: '8px', 
                        padding: '6px 12px', 
                        background: 'rgba(15, 23, 42, 0.95)', 
                        borderRadius: '6px', 
                        border: '1.5px solid var(--accent-blue)', 
                        fontSize: '12px', 
                        display: 'flex', 
                        gap: '24px', 
                        justifyContent: 'center',
                        color: 'var(--text-primary)',
                        width: '100%',
                        boxSizing: 'border-box'
                      }}>
                        <div>Inline: <strong style={{ color: 'var(--text-primary)' }}>{sectionHovered.il}</strong></div>
                        <div>Crossline: <strong style={{ color: 'var(--text-primary)' }}>{sectionHovered.xl}</strong></div>
                        <div>Two-Way Time: <strong style={{ color: 'var(--accent-gold)' }}>{sectionHovered.twt.toFixed(1)} ms</strong></div>
                        <div>{arbLineProperty} Value: <strong style={{ color: 'var(--accent-blue)' }}>{sectionHovered.val.toFixed(arbLineProperty.startsWith('PH') || arbLineProperty === 'SWE' ? 4 : 2)}</strong></div>
                      </div>
                    )}
                    {arbLineWells.length >= 2 && (
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
                        Arbitrary Section Path: {arbLineWells.join(' ➔ ')}
                      </span>
                    )}
                  </div>
                </div>


              </div>
            </div>
          )}

          {/* TAB 3: WELL-TO-SEISMIC TIE SIMULATOR */}
          {activeTab === 'tie' && (
            <div className="slide">
              <div className="slide-title-area" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2>Interactive Well-to-Seismic Tie Simulator</h2>
                  <p className="slide-subtitle">Convolve depth-derived reflectivity logs with custom Ricker wavelets and apply bulk shift to optimize cross-correlation.</p>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>Select Well:</span>
                   <select 
                    value={selectedWell} 
                    onChange={(e) => setSelectedWell(e.target.value)}
                    style={{ padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)', color: 'var(--text-primary)', fontWeight: 'bold' }}
                  >
                    {Object.keys(wellsData).map(w => (
                      <option key={w} value={w} style={{ backgroundColor: 'var(--bg-dark)', color: 'var(--text-primary)' }}>{w} {w === 'Z-04' ? '(BLIND)' : ''}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Tie stats panel + Image viewer */}
              <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '20px', alignItems: 'start' }}>

                {/* Left: metadata card */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  <div className="card-header">
                    <SlidersHorizontal size={18} className="card-icon" />
                    <h3>Best-Fit Tie Parameters</h3>
                  </div>

                  {currentWell && (() => {
                    const t = currentWell.tie;
                    const absCorr = Math.abs(t.correlation);
                    const color = absCorr >= 0.7 ? 'var(--accent-green)' : absCorr >= 0.4 ? 'var(--accent-gold)' : 'var(--accent-red)';
                    const rows = [
                      ['Quality', <strong style={{ color }}>{t.quality}</strong>],
                      ['Correlation (R)', <strong style={{ color }}>{t.correlation.toFixed(4)}</strong>],
                      ['Wavelet Freq.', `${t.freq_hz} Hz`],
                      ['Wavelet Phase', `${t.phase_deg}°`],
                      ['Bulk Time Shift', `${t.shift_ms} ms`],
                    ];
                    return (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
                        {rows.map(([label, val]) => (
                          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                            <span>{val}</span>
                          </div>
                        ))}
                      </div>
                    );
                  })()}

                  <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                    Tie generated by grid-search over frequency, phase, and bulk shift using <code>well_seismic_tie.py</code>.
                  </div>
                </div>

                {/* Right: image panels */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {/* Image toggle buttons */}
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {['tie', 'inversion', 'rockphysics'].map(imgType => (
                      <button
                        key={imgType}
                        onClick={() => setTieImageType(imgType)}
                        style={{
                          padding: '6px 18px',
                          borderRadius: '8px',
                          border: `1.5px solid ${tieImageType === imgType ? 'var(--accent-blue)' : 'var(--border-color)'}`,
                          background: tieImageType === imgType ? 'rgba(37,99,235,0.08)' : 'transparent',
                          color: tieImageType === imgType ? 'var(--accent-blue)' : 'var(--text-secondary)',
                          fontWeight: tieImageType === imgType ? '600' : '400',
                          cursor: 'pointer',
                          fontSize: '13px',
                        }}
                      >
                        {imgType === 'tie' ? '🔗 Tie QC Plot' : imgType === 'inversion' ? '📈 Impedance Inversion' : '🪨 Rock Physics Crossplot'}
                      </button>
                    ))}
                  </div>

                  <div className="glass-card" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div className="card-header" style={{ marginBottom: '4px' }}>
                      <AreaChart size={18} className="card-icon" />
                      <h3>{tieImageType === 'tie' ? 'Well-to-Seismic Tie QC' : 'Relative Acoustic Impedance Inversion'} — {selectedWell}</h3>
                    </div>
                    <img
                      src={`/well_images/${selectedWell}_${tieImageType}.png`}
                      alt={`${selectedWell} ${tieImageType} plot`}
                      style={{
                        width: '100%',
                        borderRadius: '8px',
                        border: '1px solid var(--border-color)',
                        objectFit: 'contain',
                        maxHeight: '480px',
                        background: '#fff'
                      }}
                      onError={e => { e.target.style.opacity = '0.3'; e.target.alt = 'Image not found'; }}
                    />
                  </div>
                </div>
              </div>
            </div>

          )}

          {/* TAB 4: SPECTRAL DECOMPOSITION */}
          {activeTab === 'spectral' && (
            <div className="slide">
              <div className="slide-title-area" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2>Seismic Spectral Decomposition & Digital Filtering</h2>
                  <p className="slide-subtitle">Run discrete Fourier transforms (DFT) on traces and interactively filter noise out of seismic waveforms.</p>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>Select Well:</span>
                  <select 
                    value={selectedWell} 
                    onChange={(e) => setSelectedWell(e.target.value)}
                    style={{ padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)', color: 'var(--text-primary)', fontWeight: 'bold' }}
                  >
                    {Object.keys(wellsData).map(w => (
                      <option key={w} value={w} style={{ backgroundColor: 'var(--bg-dark)', color: 'var(--text-primary)' }}>{w} {w === 'Z-04' ? '(BLIND)' : ''}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="dashboard-grid">
                {/* Bandpass controls */}
                <div className="glass-card">
                  <div className="card-header">
                    <SlidersHorizontal size={18} className="card-icon" />
                    <h3>Bandpass Filter Sliders</h3>
                  </div>

                  <div className="control-slider-group">
                    <div className="slider-label">
                      <span>Low-Cut Frequency: <strong>{lowCut} Hz</strong></span>
                    </div>
                    <input 
                      type="range" min="1" max="30" step="1" 
                      value={lowCut} onChange={(e) => setLowCut(parseInt(e.target.value))} 
                    />
                  </div>

                  <div className="control-slider-group" style={{ marginTop: '20px' }}>
                    <div className="slider-label">
                      <span>High-Cut Frequency: <strong>{highCut} Hz</strong></span>
                    </div>
                    <input 
                      type="range" min="30" max="90" step="2" 
                      value={highCut} onChange={(e) => setHighCut(parseInt(e.target.value))} 
                    />
                  </div>

                  <div style={{ marginTop: '30px', padding: '12px', backgroundColor: 'var(--bg-darker)', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                    <strong>Geological Interpretation:</strong> Low frequencies (5-15 Hz) carry bulk structural and compaction trends, while high frequencies (&gt;45 Hz) delineate thinner sand/shale interfaces. Adjust sliders to isolate frequencies.
                  </div>
                </div>

                {/* Amplitude Spectrum Chart */}
                <div className="glass-card">
                  <div className="card-header">
                    <TrendingUp size={18} className="card-icon" />
                    <h3>Trace Amplitude Spectrum (DFT)</h3>
                  </div>
                  <div style={{ height: '260px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={specChartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                        <XAxis dataKey="freq" label={{ value: 'Frequency (Hz)', position: 'bottom', offset: 0, fill: 'var(--text-secondary)' }} tickStyle={{ fill: 'var(--text-secondary)' }} />
                        <YAxis label={{ value: 'Spectral Power', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }} tickStyle={{ fill: 'var(--text-secondary)' }} />
                        <Tooltip labelFormatter={f => `Freq: ${f}Hz`} />
                        <Line dataKey="rawAmp" stroke="#94a3b8" dot={false} strokeWidth={1.5} name="Original trace spectrum" />
                        <Line dataKey="filtAmp" stroke="var(--accent-blue)" dot={false} strokeWidth={2} name="Filtered trace spectrum" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Filtered Trace Waveform Chart */}
                <div className="glass-card" style={{ gridColumn: 'span 2' }}>
                  <div className="card-header">
                    <AreaChart size={18} className="card-icon" />
                    <h3>Filtered vs. Original Waveform Comparisons</h3>
                  </div>
                  <div style={{ height: '280px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={filterTraceChartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                        <XAxis dataKey="time" label={{ value: 'Time (ms)', position: 'bottom', offset: 0, fill: 'var(--text-secondary)' }} tickStyle={{ fill: 'var(--text-secondary)' }} />
                        <YAxis label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }} tickStyle={{ fill: 'var(--text-secondary)' }} />
                        <Tooltip labelFormatter={t => `Time: ${t}ms`} />
                        <Line dataKey="rawAmp" stroke="#94a3b8" dot={false} strokeWidth={1.0} name="Original trace" />
                        <Line dataKey="filtAmp" stroke="var(--accent-blue)" dot={false} strokeWidth={2.0} name="Filtered trace" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* TAB 5: ML LOG CURVE PREDICTIONS */}
          {activeTab === 'prediction' && (
            <div className="slide">
              <div className="slide-title-area" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2>Machine Learning Log Predictor & SSI Analysis</h2>
                  <p className="slide-subtitle">Examine actual well logs vs. ML predictions across all channels. Toggle target curve and model state below.</p>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>Borehole:</span>
                  <select 
                    value={selectedWell} 
                    onChange={(e) => setSelectedWell(e.target.value)}
                    style={{ padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)', color: 'var(--text-primary)', fontWeight: 'bold' }}
                  >
                    {Object.keys(wellsData).map(w => (
                      <option key={w} value={w} style={{ backgroundColor: 'var(--bg-dark)', color: 'var(--text-primary)' }}>{w} {w === 'Z-04' ? '(BLIND)' : ''}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Predictor Customization Bar */}
              <div className="sweet-spot-alert-box" style={{ borderColor: 'var(--accent-blue)', backgroundColor: 'rgba(59, 130, 246, 0.03)', display: 'flex', flexWrap: 'wrap', gap: '16px', justifyContent: 'space-between', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Log Target:</span>
                    <select 
                      value={targetProperty} 
                      onChange={(e) => setTargetProperty(e.target.value)}
                      style={{ padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)', fontWeight: 'bold' }}
                    >
                      <option value="GR">Gamma Ray (GR)</option>
                      <option value="DT">Sonic (DT)</option>
                      <option value="RHOB">Density (RHOB)</option>
                      <option value="PHIT">Total Porosity (PHIT)</option>
                      <option value="PHIE">Effective Porosity (PHIE)</option>
                      <option value="SWE">Water Saturation (SWE)</option>
                      <option value="VSH">Volume of Shale (VSH)</option>
                      <option value="AI">Acoustic Impedance (AI)</option>
                      <option value="VPVS">Vp/Vs Ratio (Fluid)</option>
                      <option value="POIS">Poisson's Ratio</option>
                      <option value="LMRHO">Lambda-Rho (λρ)</option>
                      <option value="MURHO">Mu-Rho (μρ)</option>
                    </select>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Pipeline State:</span>
                    <button 
                      className={`badge ${modelType === 'Calibrated' ? 'badge-sweet' : 'badge-normal'}`} 
                      onClick={() => setModelType('Calibrated')}
                      style={{ cursor: 'pointer', padding: '4px 8px', borderRadius: '6px' }}
                    >
                      Calibrated (OOF Fit)
                    </button>
                    <button 
                      className={`badge ${modelType === 'Raw' ? 'badge-sweet' : 'badge-normal'}`} 
                      onClick={() => setModelType('Raw')}
                      style={{ cursor: 'pointer', padding: '4px 8px', borderRadius: '6px' }}
                    >
                      Raw (Uncalibrated)
                    </button>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                  <div className="cutoff-slider">
                    <span>Sweet Spot Threshold: <strong>{ssiCutoff.toFixed(3)}</strong></span>
                    <input 
                      type="range" min="0.01" max="0.08" step="0.005" 
                      value={ssiCutoff} onChange={(e) => setSsiCutoff(parseFloat(e.target.value))} 
                    />
                    <span className="badge badge-sweet">{sweetSpotCount} Zones Flagged</span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Porosity Sibling:</span>
                    <button 
                      className={`badge ${porosityType === 'PHIE' ? 'badge-sweet' : 'badge-normal'}`} 
                      onClick={() => setPorosityType('PHIE')}
                      style={{ cursor: 'pointer', padding: '4px 8px', borderRadius: '6px' }}
                    >
                      PHIE
                    </button>
                    <button 
                      className={`badge ${porosityType === 'PHIT' ? 'badge-sweet' : 'badge-normal'}`} 
                      onClick={() => setPorosityType('PHIT')}
                      style={{ cursor: 'pointer', padding: '4px 8px', borderRadius: '6px' }}
                    >
                      PHIT
                    </button>
                  </div>
                </div>
              </div>

              <div className="logs-workspace">
                {/* 3-Track Log Display */}
                <div className="logs-tracks-container" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                  
                  {/* TRACK 1: TARGET PROPERTY CURVE */}
                  <div className="log-track">
                    <div className="track-name">Track 1: {targetProperty} Target Log</div>
                    <div className="track-scale">
                      <span>Actual (Solid)</span>
                      <span>ML Predict (Dashed)</span>
                    </div>
                    <div className="chart-wrapper" style={{ height: '420px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart layout="vertical" data={processedSamples} margin={{ top: 0, right: 10, left: 10, bottom: 0 }}>
                          <XAxis type="number" domain={['auto', 'auto']} hide={true} />
                          <YAxis reversed dataKey="time" type="number" domain={['auto', 'auto']} width={35} tickStyle={{ fontSize: 9, fill: 'var(--text-muted)' }} />
                          <Tooltip formatter={(v) => v ? v.toFixed(3) : 'N/A'} labelFormatter={t => `Time: ${t}ms`} />
                          <Line dataKey="actVal" stroke="var(--accent-blue)" dot={false} strokeWidth={1.5} name={`Act ${targetProperty}`} connectNulls />
                          <Line dataKey="predVal" stroke="var(--accent-blue)" dot={false} strokeWidth={1.5} strokeDasharray="3 3" name={`Pred ${targetProperty}`} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* TRACK 2: POROSITY & SATURATION SIBLINGS */}
                  <div className="log-track">
                    <div className="track-name">Track 2: Porosity & Saturation</div>
                    <div className="track-scale">
                      <span>Porosity (Purple)</span>
                      <span>SWE (Cyan)</span>
                    </div>
                    <div className="chart-wrapper" style={{ height: '420px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart layout="vertical" data={processedSamples} margin={{ top: 0, right: 10, left: 10, bottom: 0 }}>
                          <XAxis type="number" domain={[0, 1.0]} hide={true} />
                          <YAxis reversed dataKey="time" type="number" domain={['auto', 'auto']} hide={true} />
                          <Tooltip formatter={(v) => v ? v.toFixed(4) : 'N/A'} labelFormatter={t => `Time: ${t}ms`} />
                          <Line dataKey="predPhi" stroke="var(--accent-purple)" dot={false} strokeWidth={1.5} strokeDasharray="3 3" name={`Pred ${porosityType}`} />
                          <Line dataKey="predSwe" stroke="var(--accent-cyan)" dot={false} strokeWidth={1.5} strokeDasharray="3 3" name="Pred SWE" />
                          <Line dataKey="actPhi" stroke="var(--accent-purple)" dot={false} strokeWidth={1.0} name={`Act ${porosityType}`} connectNulls />
                          <Line dataKey="actSwe" stroke="var(--accent-cyan)" dot={false} strokeWidth={1.0} name="Act SWE" connectNulls />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* TRACK 3: SWEET SPOT INDEX (SSI) */}
                  <div className="log-track" style={{ borderLeft: '2px solid var(--accent-gold)' }}>
                    <div className="track-name" style={{ color: 'var(--accent-gold)' }}>Track 3: Sweet Spot Index (SSI)</div>
                    <div className="track-scale">
                      <span>SSI &gt; {ssiCutoff} Flagged in Orange</span>
                    </div>
                    <div className="chart-wrapper" style={{ height: '420px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart layout="vertical" data={processedSamples} margin={{ top: 0, right: 10, left: 10, bottom: 0 }}>
                          <XAxis type="number" domain={[0, 0.1]} hide={true} />
                          <YAxis reversed dataKey="time" type="number" domain={['auto', 'auto']} hide={true} />
                          <Tooltip formatter={(v) => v ? v.toFixed(4) : 'N/A'} labelFormatter={t => `Time: ${t}ms`} />
                          <Line dataKey="predSSI" stroke="var(--accent-gold)" dot={false} strokeWidth={2.0} name="Pred SSI" />
                          <Line dataKey="actSSI" stroke="#64748b" dot={false} strokeWidth={1.0} name="Act SSI" connectNulls />
                          {/* Reference cutoff area */}
                          <ReferenceArea x1={ssiCutoff} x2={0.1} fill="rgba(217, 119, 6, 0.05)" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                </div>
              </div>
            </div>
          )}

          {/* TAB 6: COMPARISON TABLES */}
          {activeTab === 'table' && (() => {
            const spreadsheetSamples = showOnlyLoggedInterval
              ? processedSamples.filter(s => s.grAct !== null || s.dtAct !== null || s.actPhi !== null || s.actSwe !== null)
              : processedSamples;

            return (
              <div className="slide">
                <div className="slide-title-area" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                  <div>
                    <h2>Geological Comparison Spreadsheet</h2>
                    <p className="slide-subtitle">Tabular comparison of actual log channels vs. ML prediction models.</p>
                  </div>
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13.5px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                      <input 
                        type="checkbox" 
                        checked={showOnlyLoggedInterval} 
                        onChange={(e) => setShowOnlyLoggedInterval(e.target.checked)}
                        style={{ cursor: 'pointer' }}
                      />
                      <span>Only show logged reservoir interval</span>
                    </label>

                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <span style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>Borehole:</span>
                      <select 
                        value={selectedWell} 
                        onChange={(e) => setSelectedWell(e.target.value)}
                        style={{ padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)', color: 'var(--text-primary)', fontWeight: 'bold' }}
                      >
                        {Object.keys(wellsData).map(w => (
                          <option key={w} value={w} style={{ backgroundColor: 'var(--bg-dark)', color: 'var(--text-primary)' }}>{w} {w === 'Z-04' ? '(BLIND)' : ''}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>

                <div className="comparison-table-container">
                  <table className="comparison-table">
                    <thead>
                      <tr>
                        <th>Time (ms)</th>
                        <th>GR (Act)</th>
                        <th>GR (Pred)</th>
                        <th>DT (Act)</th>
                        <th>DT (Pred)</th>
                        <th>{porosityType} (Act)</th>
                        <th>{porosityType} (Pred)</th>
                        <th>SWE (Act)</th>
                        <th>SWE (Pred)</th>
                        <th>SSI (Pred)</th>
                        <th>Interpretation</th>
                      </tr>
                    </thead>
                    <tbody>
                      {spreadsheetSamples.map((d, i) => (
                        <tr key={i} className={d.isSweetSpot ? 'sweet-spot-row' : ''}>
                          <td><strong>{d.time.toFixed(1)}</strong></td>
                          <td>{d.grAct !== null ? d.grAct.toFixed(2) : '-'}</td>
                          <td>{d.grPred !== null ? d.grPred.toFixed(2) : '-'}</td>
                          
                          <td>{d.dtAct !== null ? d.dtAct.toFixed(2) : '-'}</td>
                          <td>{d.dtPred !== null ? d.dtPred.toFixed(2) : '-'}</td>
                          
                          <td>{d.actPhi !== null ? d.actPhi.toFixed(4) : '-'}</td>
                          <td>{d.predPhi !== null ? d.predPhi.toFixed(4) : '-'}</td>
                          
                          <td>{d.actSwe !== null ? d.actSwe.toFixed(4) : '-'}</td>
                          <td>{d.predSwe !== null ? d.predSwe.toFixed(4) : '-'}</td>
                          
                          <td>{d.predSSI !== null ? d.predSSI.toFixed(4) : '-'}</td>
                          <td>
                            {d.isSweetSpot ? (
                              <span className="badge badge-sweet">Sweet Spot Reservoir</span>
                            ) : (
                              <span className="badge badge-normal">Tight / Wet Rock</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}


          {/* TAB 7: GRID PREDICTOR MAP */}
          {activeTab === 'grid' && (
            <GridPredictorTab gridData={gridData} wellsData={wellsData} />
          )}

          {/* TAB 8: GEOLOGIST GALLERY */}
          {activeTab === 'gallery' && (() => {
            const galleryImages = [
              {
                id: 'seismic',
                title: '1. Seismic Profile with Wiggle Trace Overlay',
                src: '/lifecycle_images/1_seismic_section_wiggles.png',
                desc: 'A vertical seismic cross-section across Crossline 193 (where Z-02 sits) showing variable density amplitude in Red-White-Blue, overlaid with standard black wiggle traces. This is the standard way a geologist visualizes reflection boundaries and stratigraphic boundaries.',
                geoExplanation: 'Wiggle traces help distinguish actual seismic reflector peaks/troughs from noise. The black shaded positive lobes correspond to high acoustic impedance contrasts.'
              },
              {
                id: 'tie',
                title: '2. Well-to-Seismic Tie Alignment QC',
                src: '/lifecycle_images/2_well_tie_qc.png',
                desc: 'QC alignment report for Well Z-02 comparing log-derived synthetic seismograms vs actual seismic traces. Shows Sonic DT, reflection coefficients, synthetic convolution, and cross-correlation energy.',
                geoExplanation: 'This tie maps depth measurements to seismic time. A high cross-correlation indicates a valid time-depth model, giving us confidence to perform spatial predictions.'
              },
              {
                id: 'inversion',
                title: '3. Acoustic Impedance Inversion Section',
                src: '/lifecycle_images/3_impedance_inversion.png',
                desc: 'A relative acoustic impedance section along Crossline 193. Highlights the vertical geometry and continuous stratigraphic bedding of the low-impedance reservoir channel.',
                geoExplanation: 'Low acoustic impedance (yellow/green) typically correlates with higher porosity and cleaner sandstones, helping us trace reservoir continuity between wells.'
              },
              {
                id: 'sweet_spot',
                title: '4. Sweet Spot Predictor Surface Map',
                src: '/lifecycle_images/4_sweet_spot_map.png',
                desc: 'A 2D surface projection map of the Sweet Spot Index (SSI) calculated from ML models, showing the N-S trending premium reservoir fairway.',
                geoExplanation: 'The contour boundaries outline where porosity is maximized, shale is minimized, and hydrocarbon saturation is highest, outlining the primary drilling targets.'
              },
              {
                id: 'spectral',
                title: '5. Spectral Decomposition Tuning Slices',
                src: '/lifecycle_images/5_spectral_decomposition.png',
                desc: 'Multi-frequency tuning analysis showing 15Hz (thick channels), 30Hz (reservoir fairways), and 45Hz (thin pinch-outs/fault bounds) vertical sections.',
                geoExplanation: 'Different frequencies resolve different reservoir thicknesses due to thin-bed tuning, allowing us to map sub-seismic channel features.'
              },
              {
                id: 'thin_bed',
                title: '6. Sub-Seismic Thin Bed Resolution (Attributes + ML)',
                src: '/lifecycle_images/6_thin_bed_resolution.png',
                desc: 'A 4-track comparison showing how raw seismic amplitude (Track 1) fails to resolve thin reservoir beds. By extracting instantaneous envelope & sweetness attributes (Track 2) and training a Random Forest regressor, we reconstruct the high-resolution Gamma Ray log (Track 4) with a 50% improvement in correlation over the baseline.',
                geoExplanation: 'Sweetness and Instantaneous Envelope serve as proxies for sand thickness and boundary interfaces. Machine Learning uses these attributes to reconstruct sub-seismic thin beds.'
              }
            ];

            const selectedImg = galleryImages.find(img => img.id === selectedGalleryImgId) || galleryImages[0];

            return (
              <div className="slide">
                <div className="slide-title-area">
                  <h2>Geologist Lifecycle Gallery</h2>
                  <p className="slide-subtitle">High-quality, professional-grade figures showing standard geological life cycle visualizations.</p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px', alignItems: 'start' }}>
                  {/* Left: Selection menu */}
                  <div className="glass-card" style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <h3 style={{ fontSize: '14px', marginBottom: '8px', color: 'var(--text-secondary)' }}>Select Figure</h3>
                    {galleryImages.map(img => (
                      <button 
                        key={img.id}
                        onClick={() => setSelectedGalleryImgId(img.id)}
                        style={{
                          padding: '10px 12px',
                          borderRadius: '8px',
                          border: `1px solid ${selectedGalleryImgId === img.id ? 'var(--accent-blue)' : 'var(--border-color)'}`,
                          background: selectedGalleryImgId === img.id ? 'rgba(37,99,235,0.08)' : 'transparent',
                          color: selectedGalleryImgId === img.id ? 'var(--accent-blue)' : 'var(--text-secondary)',
                          fontWeight: selectedGalleryImgId === img.id ? '600' : '400',
                          textAlign: 'left',
                          cursor: 'pointer',
                          fontSize: '12.5px',
                          transition: 'all 0.2s'
                        }}
                      >
                        {img.title}
                      </button>
                    ))}
                  </div>

                  {/* Right: Selected Image display */}
                  <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <h3 style={{ margin: 0, fontSize: '18px' }}>{selectedImg.title}</h3>
                      <span className="badge badge-normal" style={{ fontSize: '11px' }}>DPI: 200 (Publication Quality)</span>
                    </div>

                    <div style={{ background: '#090d16', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'center' }}>
                      <img 
                        src={selectedImg.src} 
                        alt={selectedImg.title} 
                        style={{ maxWidth: '100%', maxHeight: '500px', objectFit: 'contain', borderRadius: '4px' }} 
                      />
                    </div>

                    <div style={{ padding: '14px', background: 'var(--bg-dark)', borderRadius: '8px', borderLeft: '3px solid var(--accent-blue)' }}>
                      <p style={{ margin: '0 0 8px 0', fontSize: '14px', color: 'var(--text-primary)' }}>
                        <strong>Description:</strong> {selectedImg.desc}
                      </p>
                      <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                        <strong>Geological Context:</strong> {selectedImg.geoExplanation}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* TAB 9: THIN BED ANALYSIS */}
          {activeTab === 'thinbed' && (
            <ThinBedTab wellsData={wellsData} />
          )}

          {/* TAB 10: CWT/SWT STUDY */}
          {activeTab === 'cwt_swt' && (
            <CwtSwtTab />
          )}

          {/* TAB 10.5: SSWT ANALYST */}
          {activeTab === 'sswt_analyst' && (
            <SswtTab />
          )}

          {/* TAB 11: CROSS CORRELATION STUDY */}
          {activeTab === 'xcorr' && (
            <CrossCorrelationTab />
          )}

          {/* TAB 12: 3D RESERVOIR HORIZON VIEWER */}
          {activeTab === 'volume3d' && (
            <Volume3dTab gridData={gridData} />
          )}

          {/* TAB 13: GEOLOGICAL R2 SCORECARD */}
          {activeTab === 'r2_scorecard' && (
            <R2ScorecardTab />
          )}

          {/* TAB 14: SPECTRAL WHITENING – THIN BED VIEWER */}
          {activeTab === 'spectral_whitening' && (
            <SpectralWhiteningTab />
          )}

          {/* TAB 15: THIN BED WORKFLOW (FREQUENCY ANALYSIS & WHITENING) */}
          {activeTab === 'thin_bed_frequency' && (
            <ThinBedWorkflowTab wellsData={wellsData} />
          )}

          {/* TAB 16: SPECTRAL ENHANCEMENT EXPLORER (FULL VOLUME VIEWER) */}
          {activeTab === 'spectral_explorer' && (
            <SpectralEnhancementExplorerTab />
          )}

          {/* TAB 17: ML SSWT KINK EXPLORER (5 TRAIN WELLS / 2 TEST WELLS MODEL) */}
          {activeTab === 'ml_kink_explorer' && (
            <MlKinkExplorerTab onSwitchTab={setActiveTab} />
          )}

          {/* TAB 18: V11 ML 3D SEISMIC PREDICTOR (VSH, SWE, PHIE, PHIT, GR, RHOB, DT, AI) */}
          {activeTab === 'ml_v11_predictor' && (
            <MlV11PredictorTab onSwitchTab={setActiveTab} />
          )}

          {/* TAB 18.5: HD BOREHOLE SEISMIC ZOOM VIEWER */}
          {activeTab === 'ml_well_zoom' && (
            <MlWellZoomTab onSwitchTab={setActiveTab} />
          )}

          {/* TAB 19: SSWT ML METHODOLOGY & STORYBOOK (VISUAL STEP-BY-STEP STORY) */}
          {activeTab === 'sswt_journey' && (
            <SswtMlJourneyTab />
          )}

        </main>
      </div>
    </div>
  );
}

export default App;
