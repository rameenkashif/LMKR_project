import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, AlertTriangle, X, MessageSquare, Compass } from 'lucide-react';

const SUGGESTED_PROMPTS = [
  'Which wells are there, and which one is the blind test well?',
  'How accurate is the VSH prediction on the blind well?',
  'Compare predicted PHIE between Z-04 and Z-07',
  'Show me the GR log for Z-02',
];

const TAB_LABELS = {
  overview: 'Executive Summary', map: 'Borehole Map & Section', tie: 'Well-Seismic Tie Simulator',
  spectral: 'Spectral Decomposition', prediction: 'ML Property Predictor', table: 'Comparison spreadsheet',
  grid: 'Grid Predictor Map', gallery: 'Geologist Gallery', thinbed: 'Thin-Bed Workbench',
  cwt_swt: 'CWT/SWT Study', sswt_analyst: 'SSWT Analyst', xcorr: 'Cross-Correlation Study',
  volume3d: '3D Reservoir Viewer', r2_scorecard: 'Geological R² Scorecard',
  spectral_whitening: 'Spectral Whitening', thin_bed_frequency: 'Thin Bed Frequency',
  spectral_explorer: 'Spectral Explorer', ml_kink_explorer: 'ML SSWT Kink Explorer',
  ml_v11_predictor: 'V11 ML 3D Seismic Predictor', ml_well_zoom: 'HD Well Seismic Zoom',
  sswt_journey: 'SSWT ML Methodology & Story',
};

export default function ChatWidget({ onNavigate }) {
  const [open, setOpen] = useState(false);
  const [hovering, setHovering] = useState(false);
  const [displayMessages, setDisplayMessages] = useState([]); // [{role, text}]
  const [apiHistory, setApiHistory] = useState([]); // raw Anthropic-format conversation
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [displayMessages, loading, open]);

  const send = async (text) => {
    const trimmed = (text ?? input).trim();
    if (!trimmed || loading) return;

    setError(null);
    setInput('');
    setDisplayMessages((prev) => [...prev, { role: 'user', text: trimmed }]);
    setLoading(true);

    const newHistory = [...apiHistory, { role: 'user', content: trimmed }];

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newHistory }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);

      setApiHistory(data.messages);

      // Detect navigate_to tool calls in the turns this request produced
      // (everything after what we sent) and actually switch the dashboard.
      const newTurns = data.messages.slice(newHistory.length);
      let navNote = null;
      for (const turn of newTurns) {
        if (turn.role !== 'assistant' || !Array.isArray(turn.content)) continue;
        for (const block of turn.content) {
          if (block.type === 'tool_use' && block.name === 'navigate_to') {
            onNavigate?.(block.input);
            const label = TAB_LABELS[block.input.tab] || block.input.tab;
            navNote = `Navigated to "${label}"${block.input.well ? ` — ${block.input.well}` : ''}${block.input.property ? ` (${block.input.property})` : ''}`;
          }
        }
      }

      setDisplayMessages((prev) => {
        const next = [...prev, { role: 'assistant', text: data.reply }];
        if (navNote) next.push({ role: 'system', text: navNote });
        return next;
      });
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (e) => {
    e.preventDefault();
    send();
  };

  return (
    <div style={{ position: 'fixed', right: '24px', bottom: '24px', zIndex: 9999 }}>
      {open && (
        <div
          style={{
            position: 'absolute',
            right: 0,
            bottom: '68px',
            width: '380px',
            maxWidth: 'calc(100vw - 48px)',
            height: '540px',
            maxHeight: 'calc(100vh - 140px)',
            display: 'flex',
            flexDirection: 'column',
            background: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            borderRadius: '14px',
            boxShadow: '0 12px 40px rgba(0,0,0,0.25)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '12px 14px', borderBottom: '1px solid var(--border-color)',
              background: '#0284c7', color: '#fff',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, fontSize: '13.5px' }}>
              <Bot size={16} />
              AI Field Assistant
            </div>
            <button
              onClick={() => setOpen(false)}
              aria-label="Close chat"
              style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', display: 'flex' }}
            >
              <X size={18} />
            </button>
          </div>

          <div
            ref={scrollRef}
            style={{ flex: 1, overflowY: 'auto', padding: '14px', display: 'flex', flexDirection: 'column', gap: '12px' }}
          >
            {displayMessages.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ color: 'var(--text-secondary, #64748b)', fontSize: '12.5px' }}>
                  Ask about wells, tie quality, and V11 property predictions - grounded via tool calls against real dashboard data. Try:
                </div>
                {SUGGESTED_PROMPTS.map((p) => (
                  <button
                    key={p}
                    onClick={() => send(p)}
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      borderRadius: '8px',
                      border: '1px solid var(--border-color)',
                      background: 'var(--bg-darker)',
                      color: 'var(--text-primary)',
                      cursor: 'pointer',
                      fontSize: '12.5px',
                    }}
                  >
                    {p}
                  </button>
                ))}
              </div>
            )}

            {displayMessages.map((m, i) => (
              m.role === 'system' ? (
                <div key={i} style={{ display: 'flex', justifyContent: 'center' }}>
                  <div
                    style={{
                      display: 'flex', alignItems: 'center', gap: '6px',
                      fontSize: '11px', fontWeight: 600, color: '#0284c7',
                      background: 'rgba(2, 132, 199, 0.1)', border: '1px solid rgba(2, 132, 199, 0.25)',
                      borderRadius: '999px', padding: '4px 10px',
                    }}
                  >
                    <Compass size={11} />
                    {m.text}
                  </div>
                </div>
              ) : (
              <div key={i} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                <div
                  style={{
                    flexShrink: 0,
                    width: 24, height: 24, borderRadius: '50%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: m.role === 'user' ? 'var(--bg-darker)' : '#0284c7',
                    color: m.role === 'user' ? 'var(--text-primary)' : '#fff',
                  }}
                >
                  {m.role === 'user' ? <User size={12} /> : <Bot size={12} />}
                </div>
                <div
                  style={{
                    whiteSpace: 'pre-wrap',
                    fontSize: '12.5px',
                    lineHeight: 1.5,
                    padding: '8px 10px',
                    borderRadius: '8px',
                    background: m.role === 'user' ? 'var(--bg-darker)' : 'rgba(2, 132, 199, 0.08)',
                    border: '1px solid var(--border-color)',
                    maxWidth: '82%',
                  }}
                >
                  {m.text}
                </div>
              </div>
              )
            ))}

            {loading && (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', color: 'var(--text-secondary, #64748b)', fontSize: '12.5px' }}>
                <Loader2 size={13} className="animate-spin" />
                Thinking…
              </div>
            )}

            {error && (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start', color: '#dc2626', fontSize: '12.5px' }}>
                <AlertTriangle size={13} style={{ marginTop: 2, flexShrink: 0 }} />
                <span>{error}</span>
              </div>
            )}
          </div>

          <form
            onSubmit={onSubmit}
            style={{ display: 'flex', gap: '6px', padding: '10px', borderTop: '1px solid var(--border-color)' }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question…"
              style={{
                flex: 1,
                padding: '9px 10px',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                background: 'var(--bg-card)',
                color: 'var(--text-primary)',
                fontSize: '12.5px',
              }}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              style={{
                display: 'flex', alignItems: 'center', gap: '5px',
                padding: '9px 13px', borderRadius: '8px', border: 'none',
                background: loading || !input.trim() ? 'var(--border-color)' : '#0284c7',
                color: '#fff', fontWeight: 600, fontSize: '12.5px',
                cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              }}
            >
              <Send size={13} />
            </button>
          </form>
        </div>
      )}

      <div style={{ position: 'relative', display: 'flex', justifyContent: 'flex-end' }}>
        {hovering && !open && (
          <div
            style={{
              position: 'absolute',
              right: '64px',
              bottom: '14px',
              whiteSpace: 'nowrap',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              padding: '7px 12px',
              borderRadius: '8px',
              fontSize: '12.5px',
              fontWeight: 600,
              boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
            }}
          >
            Ask the Assistant
          </div>
        )}
        <button
          onClick={() => setOpen((o) => !o)}
          onMouseEnter={() => setHovering(true)}
          onMouseLeave={() => setHovering(false)}
          aria-label="Ask the Assistant"
          style={{
            width: '52px', height: '52px', borderRadius: '50%',
            border: 'none', cursor: 'pointer',
            background: '#0284c7', color: '#fff',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 6px 20px rgba(2, 132, 199, 0.45)',
          }}
        >
          {open ? <X size={22} /> : <MessageSquare size={22} />}
        </button>
      </div>
    </div>
  );
}
