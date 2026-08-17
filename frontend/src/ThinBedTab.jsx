import React, { useState, useEffect, useRef } from 'react';
import { ChevronLeft, ChevronRight, Layers, Info, Activity, Sliders, Play, MapPin, Database } from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend as RechartsLegend, 
  ResponsiveContainer 
} from 'recharts';
import { thinBedWellsData, frequencySpectrum, geobodiesMetadata } from './thin_bed_data';
import { wellsConfig } from './data';

// --- Frequency Color Interpolation for Bed Form + Instantaneous Freq Overlay ---
const getFreqColor = (freq) => {
  // Map negative instantaneous frequency spikes to bright green, others to grey
  if (freq < 0) {
    const norm = Math.min(1.0, Math.abs(freq) / 50.0);
    return `rgba(34, 197, 94, ${0.4 + 0.6 * norm})`; // Green spikes
  }
  return 'rgba(255, 255, 255, 0.05)';
};

export default function ThinBedTab() {
  const [slideIdx, setSlideIdx] = useState(0);
  const [arbLineWells, setArbLineWells] = useState(['Z-03', 'Z-07', 'Z-04']);
  const [terraceMethod, setTerraceMethod] = useState('zero_crossing'); // 'zero_crossing' or 'inflection_point'
  const [terraceAttr, setTerraceAttr] = useState('amplitude'); // 'amplitude', 'thickness', 'arc_length', 'event'
  const [hoveredData, setHoveredData] = useState(null);
  const [hoveredGeobody, setHoveredGeobody] = useState(null);

  // Canvases refs
  const canvasRefs = {
    slide1: useRef(null),
    slide2_left: useRef(null),
    slide2_right: useRef(null),
    slide4_left: useRef(null),
    slide4_right: useRef(null),
    slide5_left: useRef(null),
    slide5_right: useRef(null),
    slide6_left: useRef(null),
    slide6_right: useRef(null),
    slide7_left: useRef(null),
    slide7_right: useRef(null),
    slide8_left: useRef(null),
    slide8_right: useRef(null),
    slide9_left: useRef(null),
    slide9_right: useRef(null),
    slide10_left: useRef(null),
    slide10_right: useRef(null)
  };

  // Slides names & titles
  const slides = [
    { id: 'intro', title: '1. Raw Seismic Profile' },
    { id: 'sof', title: '2. Structurally-Oriented Filter (SOF)' },
    { id: 'spectrum', title: '3. Spectral Enhancement - Power Spectrum' },
    { id: 'enhancement', title: '4. Spectral Enhancement - Section View' },
    { id: 'terrace', title: '5. Terrace Wavelet Blocking Attributes' },
    { id: 'doublet', title: '6. Doublet Attribute Output' },
    { id: 'neg_ifreq', title: '7. Negative Instantaneous Frequency' },
    { id: 'bedform', title: '8. Skeletonized Bed Form Lineaments' },
    { id: 'combined', title: '9. Combined Bed Form & Neg Freq' },
    { id: 'geobody', title: '10. Doublet-Derived 3D Geobodies' }
  ];

  const handleNext = () => setSlideIdx((prev) => Math.min(prev + 1, slides.length - 1));
  const handlePrev = () => setSlideIdx((prev) => Math.max(prev - 1, 0));

  // --- Core Rendering Engine ---
  useEffect(() => {
    // 1. Determine active canvases based on slideIdx
    const activeCanvases = [];
    const stepId = slides[slideIdx].id;

    if (stepId === 'intro') activeCanvases.push({ ref: canvasRefs.slide1, type: 'raw' });
    else if (stepId === 'sof') {
      activeCanvases.push({ ref: canvasRefs.slide2_left, type: 'raw' });
      activeCanvases.push({ ref: canvasRefs.slide2_right, type: 'filtered' });
    } else if (stepId === 'enhancement') {
      activeCanvases.push({ ref: canvasRefs.slide4_left, type: 'filtered' });
      activeCanvases.push({ ref: canvasRefs.slide4_right, type: 'enhanced' });
    } else if (stepId === 'terrace') {
      activeCanvases.push({ ref: canvasRefs.slide5_left, type: 'enhanced' });
      activeCanvases.push({ ref: canvasRefs.slide5_right, type: 'terrace' });
    } else if (stepId === 'doublet') {
      activeCanvases.push({ ref: canvasRefs.slide6_left, type: 'enhanced' });
      activeCanvases.push({ ref: canvasRefs.slide6_right, type: 'doublet' });
    } else if (stepId === 'neg_ifreq') {
      activeCanvases.push({ ref: canvasRefs.slide7_left, type: 'enhanced' });
      activeCanvases.push({ ref: canvasRefs.slide7_right, type: 'neg_ifreq' });
    } else if (stepId === 'bedform') {
      activeCanvases.push({ ref: canvasRefs.slide8_left, type: 'enhanced' });
      activeCanvases.push({ ref: canvasRefs.slide8_right, type: 'bedform' });
    } else if (stepId === 'combined') {
      activeCanvases.push({ ref: canvasRefs.slide9_left, type: 'enhanced' });
      activeCanvases.push({ ref: canvasRefs.slide9_right, type: 'combined' });
    } else if (stepId === 'geobody') {
      activeCanvases.push({ ref: canvasRefs.slide10_left, type: 'enhanced' });
      activeCanvases.push({ ref: canvasRefs.slide10_right, type: 'geobody' });
    }

    activeCanvases.forEach(({ ref, type }) => {
      if (!ref.current || arbLineWells.length < 2) return;
      const canvas = ref.current;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0); // avoid transform leaks
      
      const W = canvas.width;
      const H = canvas.height;
      
      // Clear canvas
      ctx.fillStyle = '#090d16';
      ctx.fillRect(0, 0, W, H);

      const offsetX = 55;
      const rightMargin = 20;
      const offsetY = 20;
      const bottomOffsetY = 40;
      const plotW = W - offsetX - rightMargin;
      const plotH = H - offsetY - bottomOffsetY;

      const nPanels = arbLineWells.length - 1;
      const colsPerPanel = Math.floor(plotW / nPanels);

      // Max trace samples (TWT)
      let maxLen = 0;
      arbLineWells.forEach(w => {
        if (thinBedWellsData[w]) maxLen = Math.max(maxLen, thinBedWellsData[w].samples.length);
      });
      if (maxLen === 0) return;

      // ── Pass 1: Draw Pixel Background ──
      for (let p = 0; p < nPanels; p++) {
        const wellA = thinBedWellsData[arbLineWells[p]];
        const wellB = thinBedWellsData[arbLineWells[p+1]];
        if (!wellA || !wellB) continue;

        const startX = offsetX + p * colsPerPanel;
        const endX = offsetX + (p + 1) * colsPerPanel;

        for (let x = startX; x < endX; x++) {
          const frac = (x - startX) / colsPerPanel;
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sIdx = Math.min(Math.floor(fracY * maxLen), wellA.samples.length - 1);
            
            const sampleA = wellA.samples[sIdx];
            const sampleB = wellB.samples[sIdx];
            if (!sampleA || !sampleB) continue;

            let color = '';
            // Determine attribute values based on canvas type
            const getVal = (s) => {
              if (type === 'raw') return s.seismic_raw;
              if (type === 'filtered') return s.seismic_filtered;
              if (type === 'enhanced') return s.seismic_enhanced;
              if (type === 'terrace') {
                const suffix = terraceMethod === 'zero_crossing' ? 'zc' : 'ip';
                return s[`terrace_${terraceAttr}_${suffix}`];
              }
              if (type === 'doublet') return s.doublet;
              if (type === 'neg_ifreq') return s.neg_ifreq;
              if (type === 'bedform') return s.bed_form_cos;
              if (type === 'combined') return s.bed_form_cos;
              if (type === 'geobody') return s.seismic_enhanced; // geobodies are drawn on enhanced seismic
              return 0;
            };

            const valA = getVal(sampleA);
            const valB = getVal(sampleB);
            const val = (1 - frac) * valA + frac * valB;

            // Apply specific visual scale
            if (type === 'raw' || type === 'filtered' || type === 'enhanced' || type === 'geobody') {
              // Red-White-Blue seismic scale
              const norm = Math.max(-1, Math.min(1, val / 15000));
              if (norm > 0) {
                color = `rgb(255, ${Math.round(255 * (1 - norm))}, ${Math.round(255 * (1 - norm))})`;
              } else {
                const absN = Math.abs(norm);
                color = `rgb(${Math.round(255 * (1 - absN))}, ${Math.round(255 * (1 - absN))}, 255)`;
              }
            } else if (type === 'terrace') {
              if (terraceAttr === 'amplitude') {
                // Square seismic wiggles
                const norm = Math.max(-1, Math.min(1, val / 15000));
                if (norm > 0) {
                  color = `rgb(255, ${Math.round(255 * (1 - norm))}, ${Math.round(255 * (1 - norm))})`;
                } else {
                  const absN = Math.abs(norm);
                  color = `rgb(${Math.round(255 * (1 - absN))}, ${Math.round(255 * (1 - absN))}, 255)`;
                }
              } else if (terraceAttr === 'thickness') {
                // Hot colormap: Black (low) to Orange/White (high thickness)
                const norm = Math.max(0, Math.min(1, val / 45.0)); // max thickness ~45 samples
                color = `rgb(${Math.round(100 + 155 * norm)}, ${Math.round(70 * norm)}, ${Math.round(20 * norm)})`;
              } else if (terraceAttr === 'arc_length') {
                // Arc length: Jet-like colormap
                const norm = Math.max(-1, Math.min(1, val * 3.0));
                if (norm > 0) {
                  color = `rgb(${Math.round(255 * norm)}, ${Math.round(255 * (1 - norm))}, 50)`;
                } else {
                  color = `rgb(50, ${Math.round(255 * (1 - Math.abs(norm)))}, ${Math.round(255 * Math.abs(norm))})`;
                }
              } else {
                // Event labeling: Staircase color band
                const norm = (val % 8) / 8.0;
                color = `hsl(${norm * 360}, 65%, 40%)`;
              }
            } else if (type === 'doublet') {
              // Doublet: Yellow/Green (Low) to Red (High doublet response)
              const norm = Math.max(0, Math.min(1, val * 10.0));
              color = `rgb(${Math.round(150 + 105 * norm)}, ${Math.round(200 * (1 - norm))}, ${Math.round(50 * (1 - norm))})`;
            } else if (type === 'neg_ifreq') {
              // Highlight negative values in green/yellow, zero/positive in grey
              if (val < 0) {
                const norm = Math.min(1.0, Math.abs(val) / 45.0);
                color = `rgb(${Math.round(220 * norm)}, 255, ${Math.round(30 * (1 - norm))})`; // Green to Yellow
              } else {
                color = '#1e293b';
              }
            } else if (type === 'bedform' || type === 'combined') {
              // Bedform cosine: sharp lines along max/min phase (independent of amp)
              const norm = Math.max(-1, Math.min(1, val));
              if (norm > 0.8) {
                color = 'rgb(239, 68, 68)'; // Peaks = Red lineaments
              } else if (norm < -0.8) {
                color = 'rgb(59, 130, 246)'; // Troughs = Blue lineaments
              } else {
                color = 'rgba(15, 23, 42, 0.4)'; // transparent grey background
              }
            }

            ctx.fillStyle = color;
            ctx.fillRect(x, y, 1, 1);
          }
        }
      }

      // ── Pass 2: Overlays for geobodies, negative IF, or wiggles ──
      for (let p = 0; p < nPanels; p++) {
        const wellA = thinBedWellsData[arbLineWells[p]];
        const wellB = thinBedWellsData[arbLineWells[p+1]];
        if (!wellA || !wellB) continue;

        const startX = offsetX + p * colsPerPanel;
        const endX = offsetX + (p + 1) * colsPerPanel;

        for (let x = startX; x < endX; x++) {
          const frac = (x - startX) / colsPerPanel;
          for (let y = offsetY; y < offsetY + plotH; y++) {
            const fracY = (y - offsetY) / plotH;
            const sIdx = Math.min(Math.floor(fracY * maxLen), wellA.samples.length - 1);
            
            const sampleA = wellA.samples[sIdx];
            const sampleB = wellB.samples[sIdx];
            
            if (type === 'combined') {
              // Overlay negative instantaneous frequency spikes in bright green
              const valA = sampleA.neg_ifreq;
              const valB = sampleB.neg_ifreq;
              const val = (1 - frac) * valA + frac * valB;
              if (val < 0) {
                ctx.fillStyle = 'rgba(34, 197, 94, 0.7)'; // translucent green overlay
                ctx.fillRect(x, y, 1, 1);
              }
            } else if (type === 'geobody') {
              // Highlight connected doublet geobodies
              const valA = sampleA.is_geobody;
              const valB = sampleB.is_geobody;
              const val = (1 - frac) * valA + frac * valB;
              if (val > 0.5) {
                ctx.fillStyle = 'rgba(239, 68, 68, 0.6)'; // Translucent Geobody Red
                ctx.fillRect(x, y, 1, 1);
              }
            }
          }
        }
      }

      // ── Pass 3: Horizon highlights (Red Peak & Tensleep) ──
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
        
        ctx.fillStyle = '#fbbf24';
        ctx.font = 'bold 9.5px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(name, offsetX + plotW - 12, offsetY + yFrac * plotH - 6);
        ctx.restore();
      };

      drawHorizon(0.42, "Red Peak", "rgba(251, 191, 36, 0.85)");
      drawHorizon(0.68, "Tensleep", "rgba(168, 85, 247, 0.85)");

      // ── Pass 4: Borehole Lines & Axes Ticks ──
      ctx.fillStyle = '#64748b';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      const firstWell = thinBedWellsData[arbLineWells[0]];
      const tStart = firstWell.samples[0].time;
      const tEnd = firstWell.samples[firstWell.samples.length - 1].time;

      for (let t = Math.ceil(tStart / 50) * 50; t <= tEnd; t += 50) {
        const frac = (t - tStart) / (tEnd - tStart);
        const y = offsetY + frac * plotH;
        ctx.beginPath();
        ctx.moveTo(offsetX - 4, y);
        ctx.lineTo(offsetX, y);
        ctx.stroke();
        ctx.fillText(`${t} ms`, offsetX - 8, y);
      }

      arbLineWells.forEach((wName, idx) => {
        const xPos = offsetX + (idx / nPanels) * plotW;
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.moveTo(xPos, offsetY);
        ctx.lineTo(xPos, offsetY + plotH);
        ctx.stroke();

        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(wName, xPos, offsetY - 6);

        ctx.fillStyle = '#475569';
        ctx.fillText(`IL ${wellsConfig[wName].inline}`, xPos, offsetY + plotH + 12);
        ctx.fillText(`XL ${wellsConfig[wName].crossline}`, xPos, offsetY + plotH + 22);
      });
    });

  }, [slideIdx, arbLineWells, terraceMethod, terraceAttr]);

  // --- Hover interaction handler ---
  const handleMouseMove = (e, canvasRef, type) => {
    if (!canvasRef.current || arbLineWells.length < 2) return;
    const canvas = canvasRef.current;
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
      setHoveredGeobody(null);
      return;
    }
    
    const fracX = (clickX - offsetX) / plotW;
    const fracY = (clickY - offsetY) / plotH;
    
    const nPanels = arbLineWells.length - 1;
    const p = Math.min(nPanels - 1, Math.floor(fracX * nPanels));
    const pFrac = (fracX - p / nPanels) * nPanels;
    
    const wellA = thinBedWellsData[arbLineWells[p]];
    const wellB = thinBedWellsData[arbLineWells[p+1]];
    if (!wellA || !wellB) return;
    
    const il = Math.round(wellA.inline + pFrac * (wellB.inline - wellA.inline));
    const xl = Math.round(wellA.crossline + pFrac * (wellB.crossline - wellA.crossline));
    
    const tStart = wellA.samples[0].time;
    const tEnd = wellA.samples[wellA.samples.length - 1].time;
    const twt = tStart + fracY * (tEnd - tStart);
    
    const sampleIdx = Math.floor(fracY * wellA.samples.length);
    const idx = Math.min(sampleIdx, wellA.samples.length - 1);
    
    const valA = wellA.samples[idx];
    const valB = wellB.samples[idx];
    
    let displayVal = '';
    let geobodyId = 0;
    
    const getInterpolatedValue = (key) => {
      const vA = valA[key] || 0;
      const vB = valB[key] || 0;
      return (1 - pFrac) * vA + pFrac * vB;
    };
    
    if (type === 'raw') {
      displayVal = `Raw Amp: ${getInterpolatedValue('seismic_raw').toFixed(0)}`;
    } else if (type === 'filtered') {
      displayVal = `Filtered Amp: ${getInterpolatedValue('seismic_filtered').toFixed(0)}`;
    } else if (type === 'difference') {
      displayVal = `Difference (Noise): ${getInterpolatedValue('seismic_difference').toFixed(0)}`;
    } else if (type === 'enhanced') {
      displayVal = `Enhanced Amp: ${getInterpolatedValue('seismic_enhanced').toFixed(0)}`;
    } else if (type === 'terrace') {
      const suffix = terraceMethod === 'zero_crossing' ? 'zc' : 'ip';
      const key = `terrace_${terraceAttr}_${suffix}`;
      displayVal = `Terrace ${terraceAttr}: ${getInterpolatedValue(key).toFixed(1)}`;
    } else if (type === 'doublet') {
      displayVal = `Doublet: ${getInterpolatedValue('doublet').toFixed(3)}`;
    } else if (type === 'neg_ifreq') {
      displayVal = `Inst. Freq: ${getInterpolatedValue('neg_ifreq').toFixed(1)} Hz`;
    } else if (type === 'bedform') {
      displayVal = `Cosine Phase: ${getInterpolatedValue('bed_form_cos').toFixed(2)}`;
    } else if (type === 'geobody') {
      const idA = valA.geobody_id || 0;
      const idB = valB.geobody_id || 0;
      geobodyId = pFrac > 0.5 ? idB : idA;
      displayVal = geobodyId > 0 ? `Intersecting Geobody #${geobodyId}` : 'No Geobody';
    }

    setHoveredData({ il, xl, twt, valueText: displayVal });
    
    if (geobodyId > 0) {
      const meta = geobodiesMetadata.find(g => g.id === geobodyId);
      if (meta) setHoveredGeobody(meta);
    } else {
      setHoveredGeobody(null);
    }
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

  // --- Rendering UI layouts per slide step ---
  const renderSlideContent = () => {
    const stepId = slides[slideIdx].id;

    if (stepId === 'intro') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'flex', flexDirection: 'column', position: 'relative' }}>
            <h4 className="canvas-title">Raw Seismic Cross-Section Profile</h4>
            <canvas
              ref={canvasRefs.slide1}
              width={650}
              height={400}
              onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide1, 'raw')}
              onMouseLeave={() => setHoveredData(null)}
              className="slide-canvas-main"
            />
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Methodology Introduction</h3>
            <p>We implement the 3D thin-bed resolution workflow outlined in <strong>Henning & Paton (2011)</strong>.</p>
            <div className="info-badge">
              <Info size={16} />
              <span><strong>Data Source:</strong> Real Teapot Dome Seismic & Wells only.</span>
            </div>
            <p className="description">
              In raw seismic sections, resolution is bounded by the wave's dominant wavelength (lambda/4 approx 30m). Thin sands and bed boundaries below tuning thickness interfere, creating doublets or unresolved reflections.
            </p>
            <div className="legend-box">
              <span className="legend-title">Seismic Amplitude</span>
              <div className="color-bar rwb-bar"></div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9.5px', marginTop: '3px' }}>
                <span>Trough (Blue)</span>
                <span>Peak (Red)</span>
              </div>
            </div>
          </div>
        </div>
      );
    }

    if (stepId === 'sof') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel A: Raw Seismic Section</h4>
              <canvas
                ref={canvasRefs.slide2_left}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide2_left, 'raw')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel B: Structurally-Oriented Filter (SOF)</h4>
              <canvas
                ref={canvasRefs.slide2_right}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide2_right, 'filtered')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 2: Structurally-Oriented Filter</h3>
            <p className="description">
              We construct a <strong>3D Gradient Structure Tensor (GST)</strong> across the volume to extract the true 3D spatial dip vectors v = (vx, vy, vz) at each voxel.
            </p>
            <p className="description">
              Amplitudes are smoothed in a 3 x 3 neighborhood along the structural dip planes. This cancels random/coherent noise (like footprints) while preserving reflector edges.
            </p>
            <div className="formula-box">
              T_smooth = Smooth(grad(I) * grad(I)^T)
            </div>
          </div>
        </div>
      );
    }

    if (stepId === 'spectrum') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', height: '100%' }}>
          <div className="glass-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column' }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '13px', color: '#c7d2fe', letterSpacing: '0.05em' }}>
              Trace Amplitude Frequency Spectrum (Welch PSD)
            </h4>
            <div style={{ flex: 1, minHeight: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={frequencySpectrum} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="frequency" label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5, fill: '#64748b' }} stroke="#475569" />
                  <YAxis label={{ value: 'Normalized Power', angle: -90, position: 'insideLeft', fill: '#64748b' }} stroke="#475569" />
                  <RechartsTooltip contentStyle={{ backgroundColor: '#090d16', borderColor: 'rgba(99,102,241,0.35)', color: '#fff' }} />
                  <RechartsLegend />
                  <Line type="monotone" dataKey="powerBefore" name="Before Enhancement (SOF)" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="powerAfter" name="After Spectral Enhancement" stroke="#ef4444" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 3: Power Spectrum Tuning</h3>
            <p className="description">
              Because velocity is a fixed material property, the only way to improve vertical resolution is to expand the bandwidth and boost higher frequencies.
            </p>
            <p className="description">
              The plot displays the average trace power spectrum. Spectral enhancement boosts frequencies between 30Hz and 55Hz, flattening the bandwidth to flatten/whiten the spectrum.
            </p>
          </div>
        </div>
      );
    }

    if (stepId === 'enhancement') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel A: SOF Noise-Filtered</h4>
              <canvas
                ref={canvasRefs.slide4_left}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide4_left, 'filtered')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel B: Spectrally Enhanced Section</h4>
              <canvas
                ref={canvasRefs.slide4_right}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide4_right, 'enhanced')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 4: Spectral Enhancement</h3>
            <p className="description">
              Comparing Panel A (noise-filtered) and Panel B (spectrally enhanced).
            </p>
            <p className="description">
              Notice how the thick reflectors decompose into sharp, distinct bed boundaries. Sub-seismic pinch-outs and thin sand beds become visible.
            </p>
          </div>
        </div>
      );
    }

    if (stepId === 'terrace') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel A: Spectrally Enhanced</h4>
              <canvas
                ref={canvasRefs.slide5_left}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide5_left, 'enhanced')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel B: Terrace Blocking Attribute</h4>
              <canvas
                ref={canvasRefs.slide5_right}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide5_right, 'terrace')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 5: Terrace Attributes</h3>
            
            <div className="control-group">
              <label>Delineation Boundary:</label>
              <select value={terraceMethod} onChange={(e) => setTerraceMethod(e.target.value)}>
                <option value="zero_crossing">Zero Crossings</option>
                <option value="inflection_point">Points of Inflection</option>
              </select>
            </div>

            <div className="control-group">
              <label>Terrace Variant:</label>
              <select value={terraceAttr} onChange={(e) => setTerraceAttr(e.target.value)}>
                <option value="amplitude">Terrace Amplitude</option>
                <option value="thickness">Terrace Thickness</option>
                <option value="arc_length">Terrace Arc Length</option>
                <option value="event">Event Labeling</option>
              </select>
            </div>

            <p className="description" style={{ marginTop: '10px' }}>
              Blocks trace waveforms into discrete square intervals between zero-crossings or inflection points. This allows clean auto-tracking of thin reflectors without bleeding.
            </p>
          </div>
        </div>
      );
    }

    if (stepId === 'doublet') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel A: Enhanced Seismic</h4>
              <canvas
                ref={canvasRefs.slide6_left}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide6_left, 'enhanced')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel B: Doublet Attribute</h4>
              <canvas
                ref={canvasRefs.slide6_right}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide6_right, 'doublet')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 6: Doublet Attribute</h3>
            <p className="description">
              Identifies closely spaced thin-bed reflections that have not resolved.
            </p>
            <p className="description">
              Calculates the sample-by-sample difference between Zero Crossing and Inflection Point arc lengths. High response values (red bands) highlight unresolved doublets.
            </p>
          </div>
        </div>
      );
    }

    if (stepId === 'neg_ifreq') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel A: Enhanced Seismic</h4>
              <canvas
                ref={canvasRefs.slide7_left}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide7_left, 'enhanced')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel B: Negative Instantaneous Freq</h4>
              <canvas
                ref={canvasRefs.slide7_right}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide7_right, 'neg_ifreq')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 7: Negative Instantaneous Frequency</h3>
            <p className="description">
              Instantaneous frequency is calculated as the time derivative of instantaneous phase 1 / (2pi) * dtheta/dt using Hilbert transforms.
            </p>
            <p className="description">
              Anomalous negative spikes (yellow/green lines) occur due to thin-bed destructive wave interference, outlining sub-seismic sand bodies.
            </p>
          </div>
        </div>
      );
    }

    if (stepId === 'bedform') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel A: Enhanced Seismic</h4>
              <canvas
                ref={canvasRefs.slide8_left}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide8_left, 'enhanced')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel B: Bed Form skeleton</h4>
              <canvas
                ref={canvasRefs.slide8_right}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide8_right, 'bedform')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 8: Skeletonized Bed Form</h3>
            <p className="description">
              Extracts lineaments along the maximum phase (cos(phase) &gt; 0.8, Red) and minimum phase (cos(phase) &lt; -0.8, Blue).
            </p>
            <p className="description">
              This amplitude-independent skeletonization isolates stratigraphic boundaries and clinoforms, revealing layer geometry without frequency bias.
            </p>
          </div>
        </div>
      );
    }

    if (stepId === 'combined') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel A: Enhanced Seismic</h4>
              <canvas
                ref={canvasRefs.slide9_left}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide9_left, 'enhanced')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel B: Bed Form + Neg Inst Freq</h4>
              <canvas
                ref={canvasRefs.slide9_right}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide9_right, 'combined')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 9: Combined Bed Form & Neg Freq</h3>
            <p className="description">
              We overlay the anomalous negative frequency events (green translucent bands) on top of the bed form phase skeleton.
            </p>
            <p className="description">
              This multi-attribute fusion enhances stratigraphic tracking in low-amplitude zones where reflectors otherwise pinch out visually.
            </p>
          </div>
        </div>
      );
    }

    if (stepId === 'geobody') {
      return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '16px', height: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel A: Enhanced Seismic</h4>
              <canvas
                ref={canvasRefs.slide10_left}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide10_left, 'enhanced')}
                onMouseLeave={() => setHoveredData(null)}
                className="slide-canvas"
              />
            </div>
            <div className="canvas-wrapper">
              <h4 className="canvas-title">Panel B: Extracted Doublet Geobodies</h4>
              <canvas
                ref={canvasRefs.slide10_right}
                width={360}
                height={400}
                onMouseMove={(e) => handleMouseMove(e, canvasRefs.slide10_right, 'geobody')}
                onMouseLeave={() => { setHoveredData(null); setHoveredGeobody(null); }}
                className="slide-canvas"
              />
            </div>
          </div>
          <div className="glass-card panel-sidebar">
            <h3>Step 10: 3D Geobodies</h3>
            <p className="description">
              3D connected components segmented from the doublet volume. Red overlays indicate active geobody intersections along the section profile.
            </p>
            
            {hoveredGeobody ? (
              <div className="geobody-card">
                <h4>{hoveredGeobody.name}</h4>
                <div className="stat-row">
                  <span>Voxel Count:</span>
                  <strong>{hoveredGeobody.voxelCount}</strong>
                </div>
                <div className="stat-row">
                  <span>3D Volume:</span>
                  <strong>{hoveredGeobody.volumeM3.toLocaleString()} m³</strong>
                </div>
                <div className="stat-row">
                  <span>Depth (TWT):</span>
                  <strong>{hoveredGeobody.depthRangeTWT[0].toFixed(0)} - {hoveredGeobody.depthRangeTWT[1].toFixed(0)} ms</strong>
                </div>
                <div className="stat-row">
                  <span>Avg Porosity:</span>
                  <strong>{(hoveredGeobody.avgPorosity * 100).toFixed(1)}%</strong>
                </div>
                <div className="stat-row">
                  <span>Avg Saturation:</span>
                  <strong>{(hoveredGeobody.avgSaturation * 100).toFixed(1)}%</strong>
                </div>
                <div className="stat-row">
                  <span>Ties Wells:</span>
                  <strong>{hoveredGeobody.intersectedWells.join(', ') || 'None'}</strong>
                </div>
              </div>
            ) : (
              <div className="geobody-card empty">
                <p>Hover over the red shaded geobodies on Panel B to load true 3D physical metrics...</p>
              </div>
            )}
          </div>
        </div>
      );
    }
  };

  return (
    <div className="thinbed-workflow-container">
      
      {/* --- WORKSTATION SLIDEPATH BAR --- */}
      <div className="workflow-navigator-header">
        <button className="nav-btn" onClick={handlePrev} disabled={slideIdx === 0}>
          <ChevronLeft size={16} /> Prev Step
        </button>
        
        <div className="step-dots-container">
          {slides.map((s, idx) => (
            <button
              key={s.id}
              onClick={() => setSlideIdx(idx)}
              className={`step-dot-btn ${slideIdx === idx ? 'active' : ''}`}
              title={s.title}
            >
              {idx + 1}
            </button>
          ))}
        </div>

        <button className="nav-btn" onClick={handleNext} disabled={slideIdx === slides.length - 1}>
          Next Step <ChevronRight size={16} />
        </button>
      </div>

      {/* --- PANEL TITLE --- */}
      <div className="workflow-title-bar">
        <h2>{slides[slideIdx].title}</h2>
        <span className="slide-progress">STAGE {slideIdx + 1} OF 10</span>
      </div>

      {/* --- ACTIVE SECTION CONTROL --- */}
      <div className="well-path-controls-bar">
        <span style={{ fontSize: '10.5px', color: '#475569', fontWeight: 700 }}>SELECT PATH:</span>
        {Object.keys(thinBedWellsData).map(wName => {
          const selected = arbLineWells.includes(wName);
          const isBlind = wName === 'Z-04';
          return (
            <button
              key={wName}
              onClick={() => toggleWellInPath(wName)}
              className={`well-toggle-btn ${selected ? 'active' : ''}`}
            >
              {wName}{isBlind ? ' (Blind)' : ''}
            </button>
          );
        })}
        <div style={{ marginLeft: 'auto', fontSize: '11px', color: '#94a3b8' }}>
          Active Section: <strong>{arbLineWells.join(' ➔ ')}</strong>
        </div>
      </div>

      {/* --- INTERACTIVE CANVAS WORKSPACE --- */}
      <div className="workflow-main-layout">
        {renderSlideContent()}
      </div>

      {/* --- FOOTER STATUS & LIVE READOUT --- */}
      <div className="workflow-footer-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={14} color="#10b981" />
          <span style={{ fontSize: '10px', color: '#64748b', fontWeight: 600 }}>STATUS:</span>
          <span style={{ fontSize: '10px', color: '#10b981', fontWeight: 700 }}>REAL-TIME 3D VOLUMETRIC METRICS LIVE</span>
        </div>
        
        {hoveredData ? (
          <div className="live-coord-readout">
            <span>Inline: <strong>{hoveredData.il}</strong></span>
            <span>Crossline: <strong>{hoveredData.xl}</strong></span>
            <span>TWT: <strong>{hoveredData.twt.toFixed(1)} ms</strong></span>
            <span style={{ color: '#fbbf24' }}>{hoveredData.valueText}</span>
          </div>
        ) : (
          <span style={{ fontSize: '10px', color: '#475569', fontStyle: 'italic' }}>
            Hover cursor over canvas data points for live coordinate and attribute readouts...
          </span>
        )}
      </div>

    </div>
  );
}
