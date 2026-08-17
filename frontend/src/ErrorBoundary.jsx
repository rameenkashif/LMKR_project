import React from 'react';
import { AlertTriangle, RotateCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Surfaced in the browser console for diagnosis - no telemetry backend exists yet.
    console.error('Dashboard crashed:', error, info?.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '16px',
          padding: '24px',
          textAlign: 'center',
          background: '#f8fafc',
          color: '#0f172a',
          fontFamily: 'system-ui, sans-serif',
        }}
      >
        <AlertTriangle size={40} color="#dc2626" />
        <h2 style={{ margin: 0 }}>Something went wrong rendering this page</h2>
        <p style={{ margin: 0, maxWidth: '480px', color: '#64748b', fontSize: '14px' }}>
          This is usually a transient loading issue, not lost data. Reloading almost always fixes it.
        </p>
        <pre
          style={{
            maxWidth: '600px',
            overflow: 'auto',
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '10px 14px',
            fontSize: '11px',
            color: '#dc2626',
            textAlign: 'left',
          }}
        >
          {String(this.state.error?.message || this.state.error)}
        </pre>
        <button
          onClick={() => window.location.reload()}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '10px 20px', borderRadius: '8px', border: 'none',
            background: '#0284c7', color: '#fff', fontWeight: 600, fontSize: '13px',
            cursor: 'pointer',
          }}
        >
          <RotateCw size={14} />
          Reload
        </button>
      </div>
    );
  }
}
