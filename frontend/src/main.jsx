import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'

// A stale service-worker-less cache can briefly reference a JS chunk from the
// previous deploy right after a new one goes live; reload once automatically
// rather than showing the error boundary for something a refresh just fixes.
window.addEventListener('vite:preloadError', () => {
  if (!sessionStorage.getItem('reloaded-after-chunk-error')) {
    sessionStorage.setItem('reloaded-after-chunk-error', '1');
    window.location.reload();
  }
});

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
