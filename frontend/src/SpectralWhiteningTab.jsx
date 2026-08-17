import React, { useState } from 'react';
import { Activity, BarChart2, Star, ChevronLeft, ChevronRight, Info } from 'lucide-react';

const WELLS = [
  { name: 'Z-02', img: '/seismic_images/spectral_Z02.png', crossline: 193, blind: false,
    domOrig: 24, mfOrig: 31, mfWhite: 42, snr: '16.8x', note: 'Training well — northeast flank' },
  { name: 'Z-03', img: '/seismic_images/spectral_Z03.png', crossline: 146, blind: false,
    domOrig: 24, mfOrig: 30, mfWhite: 41, snr: '15.2x', note: 'Training well — western edge' },
  { name: 'Z-04', img: '/seismic_images/spectral_Z04.png', crossline: 193, blind: true,
    domOrig: 24, mfOrig: 31, mfWhite: 42, snr: '17.9x', note: 'BLIND test well — central structure' },
  { name: 'Z-05', img: '/seismic_images/spectral_Z05.png', crossline: 199, blind: false,
    domOrig: 24, mfOrig: 30, mfWhite: 40, snr: '14.9x', note: 'Training well — southern dip' },
  { name: 'Z-06', img: '/seismic_images/spectral_Z06.png', crossline: 208, blind: false,
    domOrig: 24, mfOrig: 31, mfWhite: 41, snr: '15.7x', note: 'Training well — eastern margin' },
  { name: 'Z-07', img: '/seismic_images/spectral_Z07.png', crossline: 199, blind: false,
    domOrig: 24, mfOrig: 29, mfWhite: 40, snr: '13.4x', note: 'Training well — deep section' },
];

export default function SpectralWhiteningTab() {
  const [selectedWell, setSelectedWell]   = useState(2); // Z-04 default
  const [showInfo, setShowInfo]           = useState(false);

  const well = WELLS[selectedWell];

  const prev = () => setSelectedWell(i => (i - 1 + WELLS.length) % WELLS.length);
  const next = () => setSelectedWell(i => (i + 1) % WELLS.length);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'linear-gradient(160deg, #060b18 0%, #0d1526 50%, #060b18 100%)',
      color: '#e2e8f0', fontFamily: 'Inter, sans-serif', overflow: 'hidden'
    }}>

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '14px',
        padding: '10px 20px', flexShrink: 0,
        background: 'rgba(6,11,24,0.95)',
        borderBottom: '1px solid rgba(99,102,241,0.2)',
      }}>
        <Activity size={20} color='#818cf8' />
        <div>
          <div style={{ fontWeight: 700, fontSize: '14px', letterSpacing: '0.05em', color: '#c7d2fe' }}>
            SPECTRAL WHITENING — THIN BED RESOLUTION ANALYSIS
          </div>
          <div style={{ fontSize: '10px', color: '#475569', marginTop: '1px' }}>
            Geoteric-style workflow · Ormsby BP 8–70 Hz · Trace-by-Trace AGC · Amplitude Mask · V2 Pipeline
          </div>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{
            padding: '3px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: 600,
            background: 'rgba(16,185,129,0.12)', border: '1px solid rgba(16,185,129,0.35)',
            color: '#34d399'
          }}>SNR {well.snr}</span>

          <span style={{
            padding: '3px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: 600,
            background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.35)',
            color: '#818cf8'
          }}>XL {well.crossline}</span>

          <button
            onClick={() => setShowInfo(v => !v)}
            style={{
              padding: '5px 12px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer',
              border: '1px solid rgba(99,102,241,0.4)',
              background: showInfo ? 'rgba(99,102,241,0.25)' : 'transparent',
              color: '#818cf8', display: 'flex', alignItems: 'center', gap: '5px'
            }}
          >
            <Info size={13} /> Metrics
          </button>
        </div>
      </div>

      {/* ── Well Toggle Bar ──────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '6px',
        padding: '8px 20px', flexShrink: 0,
        background: 'rgba(6,11,24,0.8)',
        borderBottom: '1px solid rgba(99,102,241,0.12)',
        flexWrap: 'wrap'
      }}>
        <span style={{ fontSize: '10px', color: '#475569', fontWeight: 600,
                        letterSpacing: '0.07em', marginRight: '4px' }}>WELL:</span>

        {WELLS.map((w, i) => (
          <button
            key={w.name}
            onClick={() => setSelectedWell(i)}
            style={{
              padding: '5px 14px', borderRadius: '6px', fontSize: '12px',
              fontWeight: 600, cursor: 'pointer', transition: 'all 0.18s',
              border: `1px solid ${i === selectedWell
                ? (w.blind ? 'rgba(239,68,68,0.8)' : 'rgba(99,102,241,0.8)')
                : 'rgba(71,85,105,0.5)'}`,
              background: i === selectedWell
                ? (w.blind
                    ? 'linear-gradient(135deg,rgba(239,68,68,0.25),rgba(185,28,28,0.15))'
                    : 'linear-gradient(135deg,rgba(99,102,241,0.25),rgba(79,70,229,0.15))')
                : 'rgba(15,23,42,0.7)',
              color: i === selectedWell
                ? (w.blind ? '#fca5a5' : '#a5b4fc')
                : '#64748b',
              boxShadow: i === selectedWell
                ? `0 0 12px ${w.blind ? 'rgba(239,68,68,0.2)' : 'rgba(99,102,241,0.2)'}`
                : 'none',
            }}
          >
            {w.name}{w.blind ? ' ★' : ''}
          </button>
        ))}

        {/* Prev / Next */}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '4px' }}>
          <button onClick={prev} style={{
            padding: '5px 10px', borderRadius: '6px', cursor: 'pointer',
            border: '1px solid rgba(71,85,105,0.5)', background: 'rgba(15,23,42,0.7)',
            color: '#64748b', display: 'flex', alignItems: 'center'
          }}><ChevronLeft size={14} /></button>
          <button onClick={next} style={{
            padding: '5px 10px', borderRadius: '6px', cursor: 'pointer',
            border: '1px solid rgba(71,85,105,0.5)', background: 'rgba(15,23,42,0.7)',
            color: '#64748b', display: 'flex', alignItems: 'center'
          }}><ChevronRight size={14} /></button>
        </div>
      </div>

      {/* ── Metrics Panel (collapsible) ──────────────────────────────────────── */}
      {showInfo && (
        <div style={{
          display: 'flex', gap: '16px', padding: '8px 20px', flexShrink: 0,
          background: 'rgba(13,21,38,0.95)',
          borderBottom: '1px solid rgba(99,102,241,0.12)',
          flexWrap: 'wrap', alignItems: 'center'
        }}>
          {[
            { label: 'DOMINANT FREQ',      val: `${well.domOrig} Hz`,  note: 'unchanged (phase-zero)' },
            { label: 'MEAN FREQ (ORIG)',    val: `${well.mfOrig} Hz`,   note: 'before whitening' },
            { label: 'MEAN FREQ (WHITE)',   val: `${well.mfWhite} Hz`,  note: 'after whitening', highlight: true },
            { label: 'FREQ UPLIFT',         val: `+${well.mfWhite - well.mfOrig} Hz`, note: 'mean frequency gain', highlight: true },
            { label: 'POST-WHITE SNR',      val: well.snr,              note: 'vs 2.1x in V1', highlight: true },
            { label: 'BANDPASS',            val: '8/12/55/70 Hz',      note: 'Ormsby raised-cosine' },
            { label: 'AGC GATE',            val: '50 ms',              note: 'temporal normalization' },
            { label: 'WATER LEVEL',         val: '5% trace peak',      note: 'prevents blowup' },
          ].map(({ label, val, note, highlight }) => (
            <div key={label} style={{
              display: 'flex', flexDirection: 'column', gap: '2px',
              padding: '6px 12px', borderRadius: '6px',
              background: highlight ? 'rgba(16,185,129,0.07)' : 'rgba(15,23,42,0.6)',
              border: `1px solid ${highlight ? 'rgba(16,185,129,0.2)' : 'rgba(30,58,95,0.5)'}`,
              minWidth: '110px'
            }}>
              <span style={{ fontSize: '8.5px', color: '#475569', letterSpacing: '0.08em', fontWeight: 600 }}>
                {label}
              </span>
              <span style={{ fontSize: '13px', fontWeight: 700,
                             color: highlight ? '#34d399' : '#e2e8f0' }}>{val}</span>
              <span style={{ fontSize: '9px', color: '#334155', fontStyle: 'italic' }}>{note}</span>
            </div>
          ))}

          <div style={{
            marginLeft: 'auto', padding: '6px 12px', borderRadius: '6px',
            background: 'rgba(15,23,42,0.6)',
            border: '1px solid rgba(30,58,95,0.5)'
          }}>
            <div style={{ fontSize: '8.5px', color: '#475569', letterSpacing: '0.07em', fontWeight: 600, marginBottom: '3px' }}>
              WELL NOTE
            </div>
            <div style={{ fontSize: '11px', color: '#94a3b8', fontStyle: 'italic' }}>
              {well.note}
            </div>
          </div>
        </div>
      )}

      {/* ── GR Legend ────────────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', gap: '16px', padding: '5px 20px', flexShrink: 0,
        background: 'rgba(6,11,24,0.6)',
        borderBottom: '1px solid rgba(30,58,95,0.4)',
        alignItems: 'center', flexWrap: 'wrap'
      }}>
        <span style={{ fontSize: '9.5px', color: '#334155', fontWeight: 700, letterSpacing: '0.08em' }}>
          GR LITHOLOGY LEGEND:
        </span>
        {[
          { color: '#1fc03a', label: 'Clean Sand  GR < 40' },
          { color: '#f5e014', label: 'Silty Sand  40–75' },
          { color: '#ff7300', label: 'Shaly Sand  75–120' },
          { color: '#2e65e0', label: 'Shale  > 120' },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{
              width: 10, height: 10, borderRadius: 2, background: color,
              border: '1px solid rgba(255,255,255,0.15)'
            }} />
            <span style={{ fontSize: '10px', color: '#64748b' }}>{label}</span>
          </div>
        ))}

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '9.5px', color: '#334155', fontWeight: 700 }}>SEISMIC AMPLITUDE:</span>
          <div style={{
            width: 100, height: 10, borderRadius: 2,
            background: 'linear-gradient(to right, #2563eb 0%, #ffffff 50%, #ef4444 100%)',
            border: '1px solid rgba(255,255,255,0.15)'
          }} />
          <span style={{ fontSize: '9px', color: '#475569' }}>Blue (−) → White (0) → Red (+)</span>
        </div>
      </div>

      {/* ── Main Image ──────────────────────────────────────────────────────── */}
      <div style={{
        flex: 1, display: 'flex', alignItems: 'stretch',
        padding: '10px', minHeight: 0, position: 'relative'
      }}>
        {/* Left arrow */}
        <button onClick={prev} style={{
          position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)',
          zIndex: 10, padding: '10px 8px', borderRadius: '8px',
          border: '1px solid rgba(99,102,241,0.3)',
          background: 'rgba(6,11,24,0.85)', color: '#818cf8',
          cursor: 'pointer', display: 'flex', alignItems: 'center',
          backdropFilter: 'blur(4px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          transition: 'all 0.18s'
        }}>
          <ChevronLeft size={20} />
        </button>

        {/* Image container */}
        <div style={{
          flex: 1, position: 'relative', borderRadius: '10px',
          overflow: 'hidden',
          border: `1px solid ${well.blind ? 'rgba(239,68,68,0.3)' : 'rgba(99,102,241,0.25)'}`,
          boxShadow: well.blind
            ? '0 0 40px rgba(239,68,68,0.08), inset 0 0 60px rgba(0,0,0,0.3)'
            : '0 0 40px rgba(99,102,241,0.06), inset 0 0 60px rgba(0,0,0,0.3)',
          mx: '40px'
        }}>
          <img
            key={well.img}
            src={well.img}
            alt={`Spectral Whitening – Well ${well.name}`}
            style={{
              width: '100%', height: '100%',
              objectFit: 'contain',
              objectPosition: 'center',
              display: 'block',
              background: '#0a0f1e'
            }}
          />

          {/* Well badge overlay */}
          <div style={{
            position: 'absolute', top: '12px', right: '12px',
            padding: '5px 12px', borderRadius: '6px', fontSize: '12px',
            fontWeight: 700, backdropFilter: 'blur(8px)',
            background: well.blind
              ? 'rgba(239,68,68,0.85)'
              : 'rgba(99,102,241,0.85)',
            color: '#fff',
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
            border: '1px solid rgba(255,255,255,0.15)'
          }}>
            {well.name} {well.blind ? '★ BLIND' : ''}
          </div>

          {/* Step indicator dots */}
          <div style={{
            position: 'absolute', bottom: '14px', left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex', gap: '6px'
          }}>
            {WELLS.map((_, i) => (
              <button
                key={i}
                onClick={() => setSelectedWell(i)}
                style={{
                  width: i === selectedWell ? 24 : 8,
                  height: 8, borderRadius: 4,
                  background: i === selectedWell
                    ? (WELLS[i].blind ? '#ef4444' : '#818cf8')
                    : 'rgba(255,255,255,0.2)',
                  border: 'none', cursor: 'pointer',
                  transition: 'all 0.25s',
                  padding: 0
                }}
              />
            ))}
          </div>
        </div>

        {/* Right arrow */}
        <button onClick={next} style={{
          position: 'absolute', right: '16px', top: '50%', transform: 'translateY(-50%)',
          zIndex: 10, padding: '10px 8px', borderRadius: '8px',
          border: '1px solid rgba(99,102,241,0.3)',
          background: 'rgba(6,11,24,0.85)', color: '#818cf8',
          cursor: 'pointer', display: 'flex', alignItems: 'center',
          backdropFilter: 'blur(4px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
          transition: 'all 0.18s'
        }}>
          <ChevronRight size={20} />
        </button>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────────── */}
      <div style={{
        padding: '5px 20px', flexShrink: 0,
        borderTop: '1px solid rgba(30,58,95,0.4)',
        display: 'flex', gap: '20px', alignItems: 'center',
        fontSize: '9.5px', color: '#334155'
      }}>
        <span style={{ color: '#1e3a5f' }}>
          Well {selectedWell + 1} of {WELLS.length}  |  XL {well.crossline}
        </span>
        <span>Geoteric Thin-Bed Workflow: Spectral Whitening (V2)</span>
        <span style={{ marginLeft: 'auto', color: '#1e3a5f' }}>
          Mean Freq: {well.mfOrig} Hz → {well.mfWhite} Hz  |  SNR: 2.1x (V1) → {well.snr} (V2)
        </span>
      </div>
    </div>
  );
}
