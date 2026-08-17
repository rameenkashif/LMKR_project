import React, { useState, useEffect, useRef } from 'react';
import { Target, Sliders, Layers, HelpCircle } from 'lucide-react';

const RAMPS = {
  GR:   [[255,255,180],[230,200,100],[180,130,60],[130,80,30],[80,40,10]],
  DT:   [[20,40,120],[40,100,200],[80,180,240],[160,220,250],[220,245,255]],
  RHOB: [[220,245,255],[150,210,240],[80,160,210],[40,100,170],[10,40,100]],
  VSH:  [[20,180,60],[120,220,80],[240,230,60],[230,130,40],[200,40,20]],
  PHIE: [[10,10,60],[20,60,160],[40,140,220],[100,200,250],[200,240,255]],
  SWE:  [[200,30,20],[230,120,40],[240,210,60],[80,160,230],[20,60,200]],
  LMRHO:[[220,245,255],[150,210,240],[80,160,210],[40,100,170],[10,40,100]],
  SSI:  [[60,60,70],[120,120,140],[80,200,120],[40,230,80],[20,255,60]],
};

function lerpColor(ramp, t) {
  const n = ramp.length - 1;
  const scaled = Math.max(0, Math.min(1, t)) * n;
  const lo = Math.floor(scaled);
  const hi = Math.min(lo + 1, n);
  const f = scaled - lo;
  return `rgb(${Math.round(ramp[lo][0] + (ramp[hi][0] - ramp[lo][0]) * f)}, ${Math.round(ramp[lo][1] + (ramp[hi][1] - ramp[lo][1]) * f)}, ${Math.round(ramp[lo][2] + (ramp[hi][2] - ramp[lo][2]) * f)})`;
}

export default function Volume3dTab({ gridData }) {
  const canvasRef = useRef(null);
  const [property, setProperty] = useState('SSI');
  const [yaw, setYaw] = useState(45); 
  const [pitch, setPitch] = useState(30); 
  const [inlineCut, setInlineCut] = useState(100); 
  const [crosslineCut, setCrosslineCut] = useState(100); 
  const [blockHeight, setBlockHeight] = useState(80); // Height of the vertical cube side panels

  useEffect(() => {
    if (!gridData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Dark workstation background
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, W, H);

    const ils = gridData.inlines;
    const xls = gridData.crosslines;
    const points = gridData.points;

    // 2D Array Lookup
    const grid = {};
    let valMin = Infinity;
    let valMax = -Infinity;

    points.forEach(p => {
      const v = p[property] ?? 0;
      if (v < valMin) valMin = v;
      if (v > valMax) valMax = v;
      if (!grid[p.il]) grid[p.il] = {};
      grid[p.il][p.xl] = p;
    });

    const valRange = valMax - valMin || 1;

    // camera rotation angles
    const yawRad = (yaw * Math.PI) / 180;
    const pitchRad = (pitch * Math.PI) / 180;

    // Center coordinates
    const cx = W / 2;
    const cy = H / 2 - 30; // Move up to fit the vertical drop block

    const scaleX = 160 / xls.length;
    const scaleY = 160 / ils.length;

    // 3D projection function onto screen (z is depth, going DOWNWARDS on block)
    const project = (row, col, zOffset = 0) => {
      // Normalize centered grid coordinates
      const x = (col - xls.length / 2) * scaleX;
      const y = (row - ils.length / 2) * scaleY;

      // Rotation around Z axis (yaw)
      const rotX = x * Math.cos(yawRad) - y * Math.sin(yawRad);
      const rotY = x * Math.sin(yawRad) + y * Math.cos(yawRad);

      // Perspective projection with vertical drop zOffset
      const projX = rotX;
      const projY = rotY * Math.sin(pitchRad) + zOffset * Math.cos(pitchRad);

      return {
        x: cx + projX * 2.0,
        y: cy + projY * 2.0,
        zDepth: rotY * Math.cos(pitchRad) + zOffset * Math.sin(pitchRad) // depth sorting
      };
    };

    // Limits based on cutaway sliders
    const maxRow = Math.max(2, Math.floor((ils.length * inlineCut) / 100));
    const maxCol = Math.max(2, Math.floor((xls.length * crosslineCut) / 100));

    const polygons = [];
    const ramp = RAMPS[property] || RAMPS.SSI;

    // 1. Generate TOP SURFACE polygons
    for (let r = 0; r < maxRow - 1; r++) {
      for (let c = 0; c < maxCol - 1; c++) {
        const il0 = ils[r];
        const il1 = ils[r + 1];
        const xl0 = xls[c];
        const xl1 = xls[c + 1];

        const p00 = grid[il0]?.[xl0];
        const p01 = grid[il0]?.[xl1];
        const p10 = grid[il1]?.[xl0];
        const p11 = grid[il1]?.[xl1];

        if (!p00 || !p01 || !p10 || !p11) continue;

        const val00 = p00[property] ?? 0;
        const val01 = p01[property] ?? 0;
        const val10 = p10[property] ?? 0;
        const val11 = p11[property] ?? 0;

        const pt00 = project(r, c, 0);
        const pt01 = project(r, c + 1, 0);
        const pt10 = project(r + 1, c, 0);
        const pt11 = project(r + 1, c + 1, 0);

        const avgDepth = (pt00.zDepth + pt01.zDepth + pt10.zDepth + pt11.zDepth) / 4;
        const avgVal = (val00 + val01 + val10 + val11) / 4;
        const t = (avgVal - valMin) / valRange;

        polygons.push({
          type: 'top',
          pts: [pt00, pt01, pt11, pt10],
          depth: avgDepth,
          color: lerpColor(ramp, t)
        });
      }
    }

    // 2. Generate VERTICAL LEFT CUTAWAY FACE (along Inline row boundary)
    // This represents the vertical well log profile going downward
    const rCut = maxRow - 1;
    if (rCut >= 0 && rCut < ils.length) {
      const il = ils[rCut];
      for (let c = 0; c < maxCol - 1; c++) {
        const xl0 = xls[c];
        const xl1 = xls[c + 1];
        const p0 = grid[il]?.[xl0];
        const p1 = grid[il]?.[xl1];
        if (!p0 || !p1) continue;

        const val0 = p0[property] ?? 0;
        const val1 = p1[property] ?? 0;

        // Subdivide vertically into 8 time steps (depth blocks)
        const vSteps = 8;
        for (let v = 0; v < vSteps; v++) {
          const z0 = (v / vSteps) * blockHeight;
          const z1 = ((v + 1) / vSteps) * blockHeight;

          const ptTop0 = project(rCut, c, z0);
          const ptTop1 = project(rCut, c + 1, z0);
          const ptBot1 = project(rCut, c + 1, z1);
          const ptBot0 = project(rCut, c, z1);

          // Simulate vertical geology: sand channel sits in middle (v=3,4,5), shales at top/bottom.
          // Adjust vertical color based on the horizontal trace value.
          let vertT = (val0 + val1) / 2;
          let factor = 1.0;
          if (v < 2 || v > 5) {
            // Shale baseline layers (low SSI, low Porosity, high VSH)
            factor = property === 'VSH' ? 1.2 : 0.15;
          } else {
            // Channel sands (high SSI, high Porosity, low LMRHO)
            factor = property === 'LMRHO' ? 0.35 : 1.1;
          }
          const t = Math.min(1.0, Math.max(0.0, ((vertT - valMin) / valRange) * factor));

          polygons.push({
            type: 'left-face',
            pts: [ptTop0, ptTop1, ptBot1, ptBot0],
            depth: (ptTop0.zDepth + ptTop1.zDepth) / 2 - 2, // offset slightly forward
            color: lerpColor(ramp, t)
          });
        }
      }
    }

    // 3. Generate VERTICAL RIGHT CUTAWAY FACE (along Crossline col boundary)
    const cCut = maxCol - 1;
    if (cCut >= 0 && cCut < xls.length) {
      for (let r = 0; r < maxRow - 1; r++) {
        const il0 = ils[r];
        const il1 = ils[r + 1];
        const p0 = grid[il0]?.[xls[cCut]];
        const p1 = grid[il1]?.[xls[cCut]];
        if (!p0 || !p1) continue;

        const val0 = p0[property] ?? 0;
        const val1 = p1[property] ?? 0;

        const vSteps = 8;
        for (let v = 0; v < vSteps; v++) {
          const z0 = (v / vSteps) * blockHeight;
          const z1 = ((v + 1) / vSteps) * blockHeight;

          const ptTop0 = project(r, cCut, z0);
          const ptTop1 = project(r + 1, cCut, z0);
          const ptBot1 = project(r + 1, cCut, z1);
          const ptBot0 = project(r, cCut, z1);

          let vertT = (val0 + val1) / 2;
          let factor = 1.0;
          if (v < 2 || v > 5) {
            factor = property === 'VSH' ? 1.2 : 0.15;
          } else {
            factor = property === 'LMRHO' ? 0.35 : 1.1;
          }
          const t = Math.min(1.0, Math.max(0.0, ((vertT - valMin) / valRange) * factor));

          polygons.push({
            type: 'right-face',
            pts: [ptTop0, ptTop1, ptBot1, ptBot0],
            depth: (ptTop0.zDepth + ptTop1.zDepth) / 2 - 2,
            color: lerpColor(ramp, t)
          });
        }
      }
    }

    // Sort all polygons (painter's algorithm)
    polygons.sort((a, b) => b.depth - a.depth);

    // Draw
    polygons.forEach(poly => {
      ctx.beginPath();
      ctx.moveTo(poly.pts[0].x, poly.pts[0].y);
      ctx.lineTo(poly.pts[1].x, poly.pts[1].y);
      ctx.lineTo(poly.pts[2].x, poly.pts[2].y);
      ctx.lineTo(poly.pts[3].x, poly.pts[3].y);
      ctx.closePath();

      ctx.fillStyle = poly.color;
      ctx.fill();

      // Subtle border for mesh visual
      ctx.strokeStyle = poly.type === 'top' ? 'rgba(15, 23, 42, 0.12)' : 'rgba(15, 23, 42, 0.2)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });

    // Draw axis guidelines
    const drawAxis = (len, color, label, angle) => {
      const rad = (angle * Math.PI) / 180 + yawRad;
      ctx.beginPath();
      ctx.moveTo(cx - 160, H - 40);
      ctx.lineTo(cx - 160 - Math.cos(rad) * len, H - 40 + Math.sin(rad) * len * Math.sin(pitchRad));
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      ctx.fillStyle = color;
      ctx.font = 'bold 9px monospace';
      ctx.fillText(label, cx - 160 - Math.cos(rad) * (len + 7), H - 40 + Math.sin(rad) * len * Math.sin(pitchRad) + 3);
    };

    drawAxis(30, '#38bdf8', 'XL (E)', 0);
    drawAxis(30, '#f43f5e', 'IL (N)', 90);

    // Vertical Depth axis indicator
    ctx.beginPath();
    ctx.moveTo(cx - 160, H - 40);
    ctx.lineTo(cx - 160, H - 40 + 30);
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#a855f7';
    ctx.fillText('Depth (Z)', cx - 160 + 5, H - 40 + 20);

  }, [gridData, property, yaw, pitch, inlineCut, crosslineCut, blockHeight]);

  const labels = {
    SSI: 'Sweet Spot Index',
    PHIE: 'Effective Porosity',
    LMRHO: 'Lambda-Rho (Gas Indicator)',
    VSH: 'Shale Volume',
    SWE: 'Water Saturation',
  };

  return (
    <div className="slide">
      <div className="slide-title-area">
        <h2>3D Reservoir Volume Block Viewer</h2>
        <p className="slide-subtitle">Interactive 3D seismic block simulator showing property maps on the top surface and vertical logging profiles on side cutaways.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '20px', alignItems: 'start' }}>
        
        {/* Left Column: 3D Canvas */}
        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0, fontSize: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} style={{ color: 'var(--accent-gold)' }} />
              3D Solid Block Model: {labels[property]}
            </h3>
            <span className="badge badge-normal" style={{ fontSize: '11px' }}>Solid Voxel Projection</span>
          </div>

          <div style={{ background: '#090d16', padding: '10px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'center' }}>
            <canvas 
              ref={canvasRef} 
              width={520} 
              height={380} 
              style={{ width: '100%', height: '380px', display: 'block', borderRadius: '4px' }}
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-secondary)' }}>
            <span>Rotate/tilt using camera sliders on the right</span>
            <span style={{ color: 'var(--accent-gold)', fontWeight: 'bold' }}>Grid Size: {gridData?.inlines.length} × {gridData?.crosslines.length} Traces</span>
          </div>
        </div>

        {/* Right Column: Controls & Legend */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Property Selector */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="card-header" style={{ marginBottom: '14px' }}>
              <Target size={18} className="card-icon" />
              <h3>Visualized Reservoir Property</h3>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {Object.keys(labels).map(key => (
                <button
                  key={key}
                  onClick={() => setProperty(key)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '8px',
                    border: `1.5px solid ${property === key ? 'var(--accent-blue)' : 'var(--border-color)'}`,
                    background: property === key ? 'rgba(59,130,246,0.08)' : 'transparent',
                    color: property === key ? 'var(--accent-blue)' : 'var(--text-secondary)',
                    fontWeight: property === key ? '600' : '400',
                    fontSize: '12.5px',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                >
                  {labels[key]}
                </button>
              ))}
            </div>
          </div>

          {/* 3D Controls */}
          <div className="glass-card" style={{ padding: '20px' }}>
            <div className="card-header" style={{ marginBottom: '16px' }}>
              <Sliders size={18} className="card-icon" style={{ color: 'var(--accent-gold)' }} />
              <h3>3D Camera & Slicing Controls</h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13px' }}>
              
              {/* Yaw Rotation */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Camera Yaw (Rotate):</span>
                  <strong>{yaw}°</strong>
                </div>
                <input 
                  type="range" 
                  min="0" 
                  max="360" 
                  value={yaw} 
                  onChange={(e) => setYaw(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              {/* Pitch Tilt */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Camera Pitch (Tilt):</span>
                  <strong>{pitch}°</strong>
                </div>
                <input 
                  type="range" 
                  min="10" 
                  max="70" 
                  value={pitch} 
                  onChange={(e) => setPitch(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              {/* blockHeight */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Vertical Block Depth (TWT):</span>
                  <strong>{blockHeight} px</strong>
                </div>
                <input 
                  type="range" 
                  min="30" 
                  max="120" 
                  value={blockHeight} 
                  onChange={(e) => setBlockHeight(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <hr style={{ border: 'none', borderBottom: '1px solid var(--border-color)', margin: '4px 0' }} />

              {/* Inline Cutaway */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Inline Cutaway:</span>
                  <strong>{inlineCut}%</strong>
                </div>
                <input 
                  type="range" 
                  min="10" 
                  max="100" 
                  value={inlineCut} 
                  onChange={(e) => setInlineCut(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              {/* Crossline Cutaway */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Crossline Cutaway:</span>
                  <strong>{crosslineCut}%</strong>
                </div>
                <input 
                  type="range" 
                  min="10" 
                  max="100" 
                  value={crosslineCut} 
                  onChange={(e) => setCrosslineCut(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

            </div>
          </div>

          {/* Help Info Box */}
          <div className="glass-card" style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-gold)', fontWeight: 'bold' }}>
              <HelpCircle size={15} />
              Geologist Guide: 3D Chair Slicing
            </div>
            <p style={{ margin: 0 }}>
              Adjust the **Inline** and **Crossline Cutaway** sliders to slice open the cube. The side faces display how the property changes vertically (going down from top to bottom of the reservoir window, just like a well log track).
            </p>
          </div>

        </div>

      </div>
    </div>
  );
}
