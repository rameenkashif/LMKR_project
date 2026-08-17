import os
import json
import numpy as np
import scipy.ndimage
import scipy.signal
import segyio
import lasio
from joblib import Parallel, delayed

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
segy_path = os.path.join(analysis_dir, "segy", "origional.segy")
las_path = os.path.join(analysis_dir, "las", "Z-04.las")
public_dir = os.path.join(analysis_dir, "frontend", "public")

if not os.path.exists(public_dir):
    os.makedirs(public_dir)

# Grid bounds
i_min, i_max = 382, 626
j_min, j_max = 46, 297
I_len = i_max - i_min + 1  # 245
J_len = j_max - j_min + 1  # 252
K_len = 313

# --- 1. Compute target reflectivity spectrum from Z-04 well log ---
def compute_reflectivity_trend():
    print("Loading LAS file Z-04.las...")
    try:
        las = lasio.read(las_path)
        # Check DT and RHOB curves
        dt = las["DT"]
        rhob = las["RHOB"]
        
        # Filter out NaNs/zeros
        valid = (~np.isnan(dt)) & (~np.isnan(rhob)) & (dt > 0) & (rhob > 0)
        dt = dt[valid]
        rhob = rhob[valid]
        
        # Calculate Acoustic Impedance (AI)
        # Velocity in m/s = 10^6 / DT (since DT is in us/ft or us/m; Teapot UTM is in feet, DT is in us/ft)
        # Let's just do: AI = RHOB / DT (proportional to impedance)
        ai = rhob / (dt + 1e-8)
        
        # Reflectivity series
        r = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])
        r = np.nan_to_num(r, nan=0.0)
        
        # Power spectrum of reflectivity
        # Sample interval in depth is typically 0.5 feet.
        # Compute FFT and fit power-law trend
        n_fft = 512
        r_fft = np.fft.rfft(r, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=0.5) # cycle/ft
        
        amp = np.abs(r_fft)
        
        # Fit log-log trend: log(amp) = beta * log(freq) + c
        # Fit over reasonable range (excluding DC and highest freqs)
        idx_fit = (freqs > 0.02) & (freqs < 0.3)
        if np.sum(idx_fit) > 5:
            log_f = np.log(freqs[idx_fit])
            log_a = np.log(amp[idx_fit])
            beta, c = np.polyfit(log_f, log_a, 1)
        else:
            beta = 1.2 # default geophysically common value
            
        print(f"Well Z-04 reflectivity blue slope fitted: beta = {beta:.3f}")
        return float(beta)
    except Exception as e:
        print(f"Error computing reflectivity trend: {e}. Defaulting to beta = 1.2")
        return 1.2

# --- 2. Load 3D SEGY raw volume ---
def load_segy_volume():
    print("Loading SEGY volume...")
    volume = np.zeros((I_len, J_len, K_len), dtype=np.float32)
    with segyio.open(segy_path, "r", ignore_geometry=True) as f_segy:
        inlines = f_segy.attributes(segyio.TraceField.FieldRecord)[:]
        crosslines = f_segy.attributes(segyio.TraceField.TraceNumber)[:]
        for k in range(len(f_segy.trace)):
            il = inlines[k]
            xl = crosslines[k]
            if i_min <= il <= i_max and j_min <= xl <= j_max:
                volume[il - i_min, xl - j_min, :] = f_segy.trace[k]
    return volume

# --- 3. Filter Banks and DSP Operations ---
def ormsby_taper(freqs, f1, f2, f3, f4):
    """Symmetric Ormsby bandpass filter frequency response."""
    taper = np.zeros_like(freqs)
    
    # Low taper
    idx1 = (freqs >= f1) & (freqs < f2)
    taper[idx1] = (freqs[idx1] - f1) / (f2 - f1)
    
    # Pass band
    idx2 = (freqs >= f2) & (freqs <= f3)
    taper[idx2] = 1.0
    
    # High taper
    idx3 = (freqs > f3) & (freqs <= f4)
    taper[idx3] = 1.0 - (freqs[idx3] - f3) / (f4 - f3)
    
    return taper

# Vectorized process for single inline slice (J_len, K_len)
def process_inline_slice(raw_slice, dt_ms, beta):
    # raw_slice shape: (J_len, K_len)
    J, K = raw_slice.shape
    n_fft = 512 # pad trace from 313 to 512 for FFT
    fs = 1000.0 / dt_ms # sample frequency: 500 Hz for 2ms
    
    # FFT along Axis 1 (TWT)
    spec = np.fft.rfft(raw_slice, n=n_fft, axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
    
    amp = np.abs(spec)
    phase = np.angle(spec)
    
    # Smooth trace spectrum (envelope) using 1D Gaussian filter
    # Smoothed along the frequency axis
    amp_smoothed = scipy.ndimage.gaussian_filter1d(amp, sigma=15.0, axis=1)
    
    # Ormsby bandpass taper response (5-10-60-75 Hz)
    taper = ormsby_taper(freqs, 4, 8, 62, 75)
    taper = taper[np.newaxis, :] # broadcast across traces
    
    # A. Spectral Whitening: Divide by smoothed envelope and apply taper
    amp_whitened = (amp / (amp_smoothed + 1e-8)) * taper
    # Scale to match original RMS amplitude order
    rms_raw = np.sqrt(np.mean(raw_slice**2, axis=1, keepdims=True))
    spec_w = amp_whitened * np.exp(1j * phase)
    slice_w = np.fft.irfft(spec_w, n=n_fft, axis=1)[:, :K]
    rms_w = np.sqrt(np.mean(slice_w**2, axis=1, keepdims=True))
    slice_w = slice_w * (rms_raw / (rms_w + 1e-8))
    
    # B. Spectral Blueing: Multiply by f^beta trend, divide by smoothed envelope, and apply taper
    # Ensure freqs are non-zero to avoid division-by-zero or power anomalies
    freqs_safe = np.maximum(freqs, 0.1)
    blue_trend = freqs_safe**beta
    blue_trend = blue_trend[np.newaxis, :]
    
    amp_blued = (amp / (amp_smoothed + 1e-8)) * blue_trend * taper
    spec_b = amp_blued * np.exp(1j * phase)
    slice_b = np.fft.irfft(spec_b, n=n_fft, axis=1)[:, :K]
    rms_b = np.sqrt(np.mean(slice_b**2, axis=1, keepdims=True))
    slice_b = slice_b * (rms_raw / (rms_b + 1e-8))
    
    # C. Spectral Boosting: Gain boost in high band (35-55 Hz)
    # Apply bandpass filter in time domain
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(4, [35.0 / nyq, 55.0 / nyq], btype='band')
    # Filter along time axis
    slice_boost_band = scipy.signal.filtfilt(b, a, raw_slice, axis=1)
    # Apply +12 dB gain (approx 3.98 factor) and add back to raw
    slice_boosted = raw_slice + 2.5 * slice_boost_band
    
    return slice_w, slice_b, slice_boosted

# Save flat volume array as scale Int16
def save_volume_bin(vol, path):
    print(f"Compressing and saving to {path}...")
    # Scale so that maximum absolute value is 32767
    max_val = np.max(np.abs(vol))
    scale = float(max_val) if max_val > 0 else 1.0
    
    scaled_vol = (vol / scale) * 32767.0
    scaled_vol = np.clip(scaled_vol, -32767, 32767).astype(np.int16)
    
    # Save scale factor in metadata
    meta_path = path.replace(".bin", "_meta.json")
    with open(meta_path, "w") as f_meta:
        json.dump({"scale": scale, "shape": vol.shape}, f_meta)
        
    with open(path, "wb") as f_bin:
        scaled_vol.tofile(f_bin)

def main():
    beta = compute_reflectivity_trend()
    volume = load_segy_volume()
    dt_ms = 2.0 # sample rate
    
    # Parallel processing of inlines
    print("Processing 3D volume slices in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(process_inline_slice)(volume[i], dt_ms, beta) for i in range(I_len)
    )
    
    vol_w = np.zeros_like(volume)
    vol_b = np.zeros_like(volume)
    vol_boost = np.zeros_like(volume)
    
    for i in range(I_len):
        vol_w[i], vol_b[i], vol_boost[i] = results[i]
        
    # Save bin files
    save_volume_bin(volume, os.path.join(public_dir, "seismic_raw.bin"))
    save_volume_bin(vol_w, os.path.join(public_dir, "seismic_whitened.bin"))
    save_volume_bin(vol_b, os.path.join(public_dir, "seismic_blued.bin"))
    save_volume_bin(vol_boost, os.path.join(public_dir, "seismic_boosted.bin"))
    
    print("All volumes precomputed and saved successfully!")

if __name__ == "__main__":
    main()
