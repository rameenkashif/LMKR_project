/**
 * Synchrosqueezing Wavelet Transform (SSWT) helper in JavaScript.
 * Ported and verified against the custom Python implementation.
 */

export function computeSSWTJS(trace, dt_s = 0.002, n_cycles = 5.0) {
  const N = trace.length;
  
  // Frequency grid: 100 bins from 10 to 60 Hz
  const numFreqs = 100;
  const fMin = 10.0;
  const fMax = 60.0;
  const df = (fMax - fMin) / (numFreqs - 1);
  const freqs = new Float32Array(numFreqs);
  for (let i = 0; i < numFreqs; i++) {
    freqs[i] = fMin + i * df;
  }

  // 1. Compute complex CWT
  const cwtRe = Array.from({ length: numFreqs }, () => new Float32Array(N));
  const cwtIm = Array.from({ length: numFreqs }, () => new Float32Array(N));

  for (let i = 0; i < numFreqs; i++) {
    const freq = freqs[i];
    
    // Scale sigma with frequency
    const sigma = n_cycles / (2.0 * Math.PI * freq);
    const duration = 7.0 * sigma;
    
    // Generate wavelet time support points
    const waveletTimes = [];
    let t = -duration / 2.0;
    while (t <= duration / 2.0 + 1e-9) {
      waveletTimes.push(t);
      t += dt_s;
    }
    const lenW = waveletTimes.length;

    // Form complex Morlet wavelet
    const wavRe = new Float32Array(lenW);
    const wavIm = new Float32Array(lenW);
    let sumMag2 = 0.0;
    
    for (let k = 0; k < lenW; k++) {
      const wt = waveletTimes[k];
      const env = Math.exp(-0.5 * Math.pow(wt / sigma, 2));
      const phase = 2.0 * Math.PI * freq * wt;
      wavRe[k] = env * Math.cos(phase);
      wavIm[k] = env * Math.sin(phase);
      sumMag2 += wavRe[k] * wavRe[k] + wavIm[k] * wavIm[k];
    }
    
    // Normalize wavelet energy
    const norm = Math.sqrt(sumMag2);
    for (let k = 0; k < lenW; k++) {
      wavRe[k] /= norm;
      wavIm[k] /= norm;
    }

    // Convolution: CWT_complex = signal * wavelet (mode='same')
    const halfW = Math.floor(lenW / 2);
    for (let n = 0; n < N; n++) {
      let sumRe = 0.0;
      let sumIm = 0.0;
      for (let k = 0; k < lenW; k++) {
        const traceIdx = n - k + halfW;
        if (traceIdx >= 0 && traceIdx < N) {
          sumRe += trace[traceIdx] * wavRe[k];
          sumIm += trace[traceIdx] * wavIm[k];
        }
      }
      cwtRe[i][n] = sumRe;
      cwtIm[i][n] = sumIm;
    }
  }

  // 2. Time derivative of CWT (np.gradient along axis=1)
  const cwtDiffRe = Array.from({ length: numFreqs }, () => new Float32Array(N));
  const cwtDiffIm = Array.from({ length: numFreqs }, () => new Float32Array(N));

  for (let i = 0; i < numFreqs; i++) {
    for (let n = 0; n < N; n++) {
      if (n === 0) {
        cwtDiffRe[i][n] = (cwtRe[i][1] - cwtRe[i][0]) / dt_s;
        cwtDiffIm[i][n] = (cwtIm[i][1] - cwtIm[i][0]) / dt_s;
      } else if (n === N - 1) {
        cwtDiffRe[i][n] = (cwtRe[i][N - 1] - cwtRe[i][N - 2]) / dt_s;
        cwtDiffIm[i][n] = (cwtIm[i][N - 1] - cwtIm[i][N - 2]) / dt_s;
      } else {
        cwtDiffRe[i][n] = (cwtRe[i][n + 1] - cwtRe[i][n - 1]) / (2.0 * dt_s);
        cwtDiffIm[i][n] = (cwtIm[i][n + 1] - cwtIm[i][n - 1]) / (2.0 * dt_s);
      }
    }
  }

  // 3. Compute instantaneous frequency & squeeze energy
  const sswtMatrix = Array.from({ length: numFreqs }, () => new Float32Array(N));
  const eps = 1e-9;

  for (let i = 0; i < numFreqs; i++) {
    for (let n = 0; n < N; n++) {
      const cre = cwtRe[i][n];
      const cim = cwtIm[i][n];
      const denom = cre * cre + cim * cim;
      
      const safeDenom = denom < eps ? eps : denom;
      const dre = cwtDiffRe[i][n];
      const dim = cwtDiffIm[i][n];

      // Imaginary part of cwtDiff / cwtComplex
      const instFreqVal = (dim * cre - dre * cim) / safeDenom;
      const fEst = instFreqVal / (2.0 * Math.PI);

      if (fEst >= fMin && fEst <= fMax) {
        const binIdx = Math.round((fEst - fMin) / df);
        if (binIdx >= 0 && binIdx < numFreqs) {
          sswtMatrix[binIdx][n] += Math.sqrt(cre * cre + cim * cim);
        }
      }
    }
  }

  return {
    freqs,
    sswtMatrix
  };
}
