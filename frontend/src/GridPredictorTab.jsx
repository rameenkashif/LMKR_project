import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Target, SlidersHorizontal, Info } from 'lucide-react';

// ─── colour ramps ────────────────────────────────────────────────────────────
const RAMPS = {
  GR:   [[255,255,180],[230,200,100],[180,130,60],[130,80,30],[80,40,10]],
  DT:   [[20,40,120],[40,100,200],[80,180,240],[160,220,250],[220,245,255]],
  RHOB: [[220,245,255],[150,210,240],[80,160,210],[40,100,170],[10,40,100]],
  VSH:  [[20,180,60],[120,220,80],[240,230,60],[230,130,40],[200,40,20]],
  PHIE: [[10,10,60],[20,60,160],[40,140,220],[100,200,250],[200,240,255]],
  PHIT: [[10,10,60],[20,60,160],[40,140,220],[100,200,250],[200,240,255]],
  SWE:  [[200,30,20],[230,120,40],[240,210,60],[80,160,230],[20,60,200]],
  SSI:  [[60,60,70],[120,120,140],[80,200,120],[40,230,80],[20,255,60]],
};

function lerpColor(ramp, t) {
  const n = ramp.length - 1;
  const scaled = Math.max(0, Math.min(1, t)) * n;
  const lo = Math.floor(scaled);
  const hi = Math.min(lo + 1, n);
  const f = scaled - lo;
  return [
    Math.round(ramp[lo][0] + (ramp[hi][0] - ramp[lo][0]) * f),
    Math.round(ramp[lo][1] + (ramp[hi][1] - ramp[lo][1]) * f),
    Math.round(ramp[lo][2] + (ramp[hi][2] - ramp[lo][2]) * f),
  ];
}

const TARGETS = ['GR','DT','RHOB','VSH','PHIE','SWE','PHIT','SSI'];
const LABELS  = {
  GR:'Gamma Ray', DT:'Sonic DT', RHOB:'Density',
  VSH:'Vol. Shale', PHIE:'Eff. Porosity', SWE:'Water Sat.',
  PHIT:'Tot. Porosity', SSI:'Sweet Spot Index',
};

export default function GridPredictorTab({ gridData, wellsData }) {
  const canvasRef  = useRef(null);
  const legendRef  = useRef(null);
  const [property, setProperty]  = useState('SSI');
  const [hovered,  setHovered]   = useState(null);
  const [clicked,  setClicked]   = useState(null);
  const [cellSize, setCellSize]  = useState(3);
  const [lookup,   setLookup]    = useState(null);

  useEffect(() => {
    if (!gridData) return;
    const m = new Map();
    for (const p of gridData.points) m.set(`${p.il}_${p.xl}`, p);
    setLookup(m);
  }, [gridData]);

  const paint = useCallback(() => {
    if (!gridData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext('2d');
    const ils    = gridData.inlines;
    const xls    = gridData.crosslines;
    const cs     = cellSize;
    
    // Layout Offsets/Margins for Grid Axes Labels
    const offsetX = 52;
    const offsetY = 16;
    const bottomY = 38;
    const rightX  = 16;

    // Create offscreen canvas for fast pixel rendering
    const offscreen = document.createElement('canvas');
    offscreen.width  = xls.length * cs;
    offscreen.height = ils.length * cs;
    const oCtx = offscreen.getContext('2d');

    const vals = gridData.points.map(p => p[property] ?? 0);
    const vMin = Math.min(...vals);
    const vMax = Math.max(...vals);
    const range = vMax - vMin || 1;
    const ramp  = RAMPS[property] || RAMPS.SSI;

    const img  = oCtx.createImageData(offscreen.width, offscreen.height);
    const px   = img.data;
    const ilIdx = Object.fromEntries(ils.map((il,i) => [il,i]));
    const xlIdx = Object.fromEntries(xls.map((xl,i) => [xl,i]));

    for (const pt of gridData.points) {
      const row = ilIdx[pt.il];
      const col = xlIdx[pt.xl];
      if (row === undefined || col === undefined) continue;
      const t = ((pt[property] ?? vMin) - vMin) / range;
      const [r,g,b] = lerpColor(ramp, t);
      for (let dr = 0; dr < cs; dr++) {
        for (let dc = 0; dc < cs; dc++) {
          const off = ((row*cs+dr)*offscreen.width + (col*cs+dc))*4;
          px[off]=r; px[off+1]=g; px[off+2]=b; px[off+3]=255;
        }
      }
    }
    oCtx.putImageData(img, 0, 0);

    // Size main canvas to include axes padding
    canvas.width  = offscreen.width + offsetX + rightX;
    canvas.height = offscreen.height + offsetY + bottomY;

    // Clear main canvas with dark workstation navy
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw heatmap
    ctx.drawImage(offscreen, offsetX, offsetY);

    // Draw grid tick marks, coordinate lines, and axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, sans-serif';

    // 1. Y-Axis (Inlines)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const ilStep = Math.max(1, Math.floor(ils.length / 8));
    for (let i = 0; i < ils.length; i += ilStep) {
      const il = ils[i];
      const y = offsetY + i * cs + cs/2;
      
      // Tick line
      ctx.beginPath();
      ctx.moveTo(offsetX - 4, y);
      ctx.lineTo(offsetX, y);
      ctx.stroke();

      // Gridline
      ctx.beginPath();
      ctx.moveTo(offsetX, y);
      ctx.lineTo(offsetX + offscreen.width, y);
      ctx.stroke();

      // Text
      ctx.fillText(il.toString(), offsetX - 8, y);
    }

    // Y-Axis Title
    ctx.save();
    ctx.translate(14, offsetY + offscreen.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText("Inline Coordinate (Y)", 0, 0);
    ctx.restore();

    // 2. X-Axis (Crosslines)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xlStep = Math.max(1, Math.floor(xls.length / 8));
    for (let j = 0; j < xls.length; j += xlStep) {
      const xl = xls[j];
      const x = offsetX + j * cs + cs/2;

      // Tick line
      ctx.beginPath();
      ctx.moveTo(x, offsetY + offscreen.height);
      ctx.lineTo(x, offsetY + offscreen.height + 4);
      ctx.stroke();

      // Gridline
      ctx.beginPath();
      ctx.moveTo(x, offsetY);
      ctx.lineTo(x, offsetY + offscreen.height);
      ctx.stroke();

      // Text
      ctx.fillText(xl.toString(), x, offsetY + offscreen.height + 8);
    }

    // X-Axis Title
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText("Crossline Coordinate (X)", offsetX + offscreen.width / 2, offsetY + offscreen.height + 24);

    // Well pins
    if (gridData.wells) {
      const pinR = Math.max(5, cs*2);
      ctx.font = `bold ${Math.max(9, cs*3)}px sans-serif`;
      ctx.textAlign = 'center';
      for (const w of gridData.wells) {
        const row = ilIdx[w.il]; const col = xlIdx[w.xl];
        if (row===undefined || col===undefined) continue;
        const cx = offsetX + col*cs + cs/2;
        const cy = offsetY + row*cs + cs/2;
        ctx.beginPath(); ctx.arc(cx,cy,pinR,0,2*Math.PI);
        ctx.fillStyle='#ef4444'; ctx.fill();
        ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();
        ctx.fillStyle='#fff';
        ctx.fillText(w.name, cx, cy - pinR - 2);
      }
    }

    // Selected cell outline
    if (clicked) {
      const row = ilIdx[clicked.il]; const col = xlIdx[clicked.xl];
      if (row!==undefined && col!==undefined) {
        ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2;
        ctx.strokeRect(offsetX + col*cs, offsetY + row*cs, cs, cs);
      }
    }

    // Legend
    if (legendRef.current) {
      const lc = legendRef.current;
      const lCtx = lc.getContext('2d');
      lc.width=20; lc.height=200;
      for (let y=0;y<200;y++){
        const [r,g,b] = lerpColor(ramp, 1-y/199);
        lCtx.fillStyle=`rgb(${r},${g},${b})`;
        lCtx.fillRect(0,y,20,1);
      }
    }
  }, [gridData, property, cellSize, clicked]);

  useEffect(() => { paint(); }, [paint]);

  const hitTest = useCallback((e) => {
    if (!gridData || !canvasRef.current) return null;
    const rect  = canvasRef.current.getBoundingClientRect();
    
    const offsetX = 52;
    const offsetY = 16;
    
    const scaleX = canvasRef.current.width  / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    
    const pxX = (e.clientX - rect.left)*scaleX - offsetX;
    const pxY = (e.clientY - rect.top )*scaleY - offsetY;
    
    const ils  = gridData.inlines;
    const xls  = gridData.crosslines;
    const cs   = cellSize;
    const row  = Math.floor(pxY/cs);
    const col  = Math.floor(pxX/cs);
    if (row<0||row>=ils.length||col<0||col>=xls.length) return null;
    return lookup?.get(`${ils[row]}_${xls[col]}`) ?? {il:ils[row],xl:xls[col]};
  }, [gridData, cellSize, lookup]);

  const vals = gridData ? gridData.points.map(p=>p[property]??0) : [];
  const vMin = vals.length ? Math.min(...vals) : 0;
  const vMax = vals.length ? Math.max(...vals) : 1;

  if (!gridData) {
    return (
      <div className="slide" style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:20,minHeight:400}}>
        <div style={{width:56,height:56,borderRadius:'50%',border:'4px solid var(--accent-blue)',borderTopColor:'transparent',animation:'spin 1s linear infinite'}}/>
        <h2 style={{color:'var(--text-secondary)',fontWeight:400}}>Generating Grid Predictions…</h2>
        <p style={{color:'var(--text-muted)',maxWidth:480,textAlign:'center'}}>
          Running ML models over 61,740 seismic traces. This takes ~5-10 minutes.
          Refresh the page once the script finishes.
        </p>
        <code style={{fontSize:12,color:'var(--text-muted)',background:'var(--bg-darker)',padding:'8px 18px',borderRadius:8}}>
          python ml_training/generate_grid_predictions.py
        </code>
        <style>{`@keyframes spin{to{transform:rotate(360deg);}}`}</style>
      </div>
    );
  }

  return (
    <div className="slide">
      <div className="slide-title-area" style={{display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:12}}>
        <div>
          <h2>3-D Grid ML Predictor Map</h2>
          <p className="slide-subtitle">
            Averaged ML prediction over reservoir window ({gridData.reservoir.t_start}–{gridData.reservoir.t_end} ms TWT) across {gridData.points.length.toLocaleString()} IL×XL traces.
          </p>
        </div>
        <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
          {TARGETS.map(t=>(
            <button key={t} onClick={()=>setProperty(t)} style={{
              padding:'5px 12px', borderRadius:8, fontSize:12, cursor:'pointer',
              border:`1.5px solid ${property===t?'var(--accent-blue)':'var(--border-color)'}`,
              background: property===t?'rgba(37,99,235,0.1)':'transparent',
              color: property===t?'var(--accent-blue)':'var(--text-secondary)',
              fontWeight: property===t?700:400,
            }}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 300px',gap:20,alignItems:'start'}}>
        {/* Map panel */}
        <div className="glass-card" style={{padding:12}}>
          <div className="card-header" style={{marginBottom:8}}>
            <Target size={18} className="card-icon"/>
            <h3>{LABELS[property]} — Reservoir Average</h3>
            <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:8,fontSize:12,color:'var(--text-secondary)'}}>
              <SlidersHorizontal size={14}/>
              <span>Cell:</span>
              {[2,3,4].map(s=>(
                <button key={s} onClick={()=>setCellSize(s)} style={{
                  padding:'2px 7px', borderRadius:5, fontSize:11, cursor:'pointer',
                  border:`1px solid ${cellSize===s?'var(--accent-blue)':'var(--border-color)'}`,
                  background: cellSize===s?'rgba(37,99,235,0.1)':'transparent',
                  color: cellSize===s?'var(--accent-blue)':'var(--text-secondary)',
                }}>{s}px</button>
              ))}
            </div>
          </div>

          <div style={{display:'flex',gap:12,alignItems:'flex-start'}}>
            {/* Legend */}
            <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:4,flexShrink:0}}>
              <span style={{fontSize:10,color:'var(--text-muted)'}}>{vMax.toFixed(3)}</span>
              <canvas ref={legendRef} style={{width:20,height:200,borderRadius:4,border:'1px solid var(--border-color)'}}/>
              <span style={{fontSize:10,color:'var(--text-muted)'}}>{vMin.toFixed(3)}</span>
            </div>
            {/* Heatmap */}
            <div style={{flex:1,overflow:'auto',maxHeight:520}}>
              <canvas
                ref={canvasRef}
                style={{display:'block',cursor:'crosshair',imageRendering:'pixelated',maxWidth:'100%'}}
                onMouseMove={e=>setHovered(hitTest(e))}
                onMouseLeave={()=>setHovered(null)}
                onClick={e=>{const pt=hitTest(e);if(pt)setClicked(pt);}}
              />
            </div>
          </div>

          {hovered && (
            <div style={{marginTop:8,fontSize:12,color:'var(--text-secondary)',padding:'4px 10px',background:'var(--bg-darker)',borderRadius:6}}>
              IL <strong>{hovered.il}</strong> / XL <strong>{hovered.xl}</strong>
              {hovered[property]!==undefined && (
                <> — <strong style={{color:'var(--accent-blue)'}}>{property}: {hovered[property]?.toFixed(4)}</strong></>
              )}
              <span style={{marginLeft:10,color:'var(--text-muted)'}}>Click to inspect all properties</span>
            </div>
          )}

          <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'var(--text-muted)',marginTop:4,paddingLeft:32}}>
            <span>↑ IL {gridData.inlines[0]}</span>
            <span style={{textAlign:'center'}}>XL {gridData.crosslines[0]} → {gridData.crosslines[gridData.crosslines.length-1]}</span>
            <span>IL {gridData.inlines[gridData.inlines.length-1]} ↓</span>
          </div>
        </div>

        {/* Right panel */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          {/* Wells */}
          <div className="glass-card" style={{padding:14}}>
            <div className="card-header" style={{marginBottom:10}}>
              <Info size={16} className="card-icon"/>
              <h3>Well Locations</h3>
            </div>
            <div style={{display:'flex',flexDirection:'column',gap:6}}>
              {gridData.wells.map(w=>(
                <div key={w.name} style={{display:'flex',alignItems:'center',gap:8,fontSize:13}}>
                  <div style={{width:12,height:12,borderRadius:'50%',background:'#ef4444',border:'2px solid #fff',flexShrink:0}}/>
                  <strong>{w.name}</strong>
                  <span style={{color:'var(--text-muted)',fontSize:11}}>IL {w.il} / XL {w.xl}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Clicked inspection */}
          <div className="glass-card" style={{padding:14}}>
            <div className="card-header" style={{marginBottom:10}}>
              <Target size={16} className="card-icon"/>
              <h3>{clicked ? <>IL {clicked.il} / XL {clicked.xl}</> : 'Click a cell to inspect'}</h3>
            </div>
            {clicked ? (
              <div style={{display:'flex',flexDirection:'column',gap:7}}>
                {TARGETS.map(t=>{
                  const v = clicked[t];
                  if (v===undefined) return null;
                  const isSweet = t==='SSI' && v>=0.0045;
                  return (
                    <div key={t} style={{display:'flex',justifyContent:'space-between',fontSize:13,borderBottom:'1px solid var(--border-color)',paddingBottom:5}}>
                      <span style={{color:'var(--text-secondary)'}}>{LABELS[t]}</span>
                      <strong style={{color: t===property?'var(--accent-blue)': isSweet?'var(--accent-green)':'var(--text-primary)'}}>
                        {v.toFixed(4)}
                        {isSweet && <span style={{marginLeft:5,fontSize:10,color:'var(--accent-green)'}}>★ Sweet</span>}
                      </strong>
                    </div>
                  );
                })}
                {gridData.wells.length>0 && (()=>{
                  const near = gridData.wells.reduce((b,w)=>{
                    const d=Math.hypot(w.il-clicked.il, w.xl-clicked.xl);
                    return d<b.d?{w,d}:b;
                  },{w:null,d:Infinity});
                  return <div style={{marginTop:6,fontSize:11,color:'var(--text-muted)'}}>
                    Nearest: <strong style={{color:'var(--text-secondary)'}}>{near.w.name}</strong> ({Math.round(near.d)} grid units)
                  </div>;
                })()}
              </div>
            ):(
              <p style={{color:'var(--text-muted)',fontSize:13}}>
                Click any cell on the heatmap to see averaged ML predictions for all petrophysical properties at that IL/XL location.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
