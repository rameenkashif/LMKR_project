import os
import json
import numpy as np
import scipy.signal as signal

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
public_dir = os.path.join(analysis_dir, "frontend", "public")

raw_bin_path = os.path.join(public_dir, "seismic_raw.bin")
raw_meta_path = os.path.join(public_dir, "seismic_raw_meta.json")
output_json_path = os.path.join(public_dir, "well_sswt_data.json")

# 1. Load raw volume metadata
with open(raw_meta_path, "r") as f:
    meta = json.load(f)
scale = meta["scale"]
shape = meta["shape"] # [245, 252, 313]
I_len, J_len, K_len = shape

# Load binary raw volume and restore float values
with open(raw_bin_path, "rb") as f:
    flat_int16 = np.fromfile(f, dtype=np.int16)
raw_volume = (flat_int16.astype(np.float32) / 32767.0) * scale
raw_volume = raw_volume.reshape(shape)

# Well coordinates matching data.js
wells_coords = {
    "Z-02": {"inline": 535, "crossline": 193},
    "Z-03": {"inline": 428, "crossline": 146},
    "Z-04": {"inline": 569, "crossline": 193},
    "Z-05": {"inline": 398, "crossline": 199},
    "Z-06": {"inline": 445, "crossline": 208},
    "Z-07": {"inline": 488, "crossline": 199},
    "Z-08-ST-02": {"inline": 613, "crossline": 156}
}

# Parameters matching CWT/SSWT validation
dt_s = 0.002
freqs_target = np.linspace(10, 60, 100)
n_cycles = 5.0

def compute_sswt(trace, dt_s, n_cycles=5.0):
    N = len(trace)
    freqs = np.linspace(10, 60, 100)
    cwt_complex = np.zeros((len(freqs), N), dtype=complex)
    
    # Wavelet analysis loop
    for i, freq in enumerate(freqs):
        sigma = n_cycles / (2.0 * np.pi * freq)
        duration = 7.0 * sigma
        t_wavelet = np.arange(-duration / 2.0, duration / 2.0 + dt_s, dt_s)
        
        # Complex Morlet formula
        wavelet = np.exp(-0.5 * (t_wavelet / sigma) ** 2) * np.exp(1j * 2.0 * np.pi * freq * t_wavelet)
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2)) # normalize energy
        
        cwt_complex[i, :] = signal.convolve(trace, wavelet, mode='same')
        
    cwt_diff = np.gradient(cwt_complex, dt_s, axis=1)
    
    eps = 1e-9
    denom = cwt_complex.copy()
    denom[np.abs(denom) < eps] = eps
    inst_freq = np.imag(cwt_diff / denom) / (2.0 * np.pi)
    
    # Synchrosqueezing reassignment
    sswt_matrix = np.zeros_like(cwt_complex, dtype=float)
    df = freqs[1] - freqs[0]
    
    for i_f, freq in enumerate(freqs):
        for idx in range(N):
            f_est = inst_freq[i_f, idx]
            if freqs[0] <= f_est <= freqs[-1]:
                bin_idx = int(round((f_est - freqs[0]) / df))
                if 0 <= bin_idx < len(freqs):
                    sswt_matrix[bin_idx, idx] += np.abs(cwt_complex[i_f, idx])
                    
    return sswt_matrix.tolist()

# Compute SSWT for each well and save
well_sswt_data = {}
for wname, coords in wells_coords.items():
    print(f"Precomputing Python SSWT for well {wname} (IL {coords['inline']}, XL {coords['crossline']})...")
    i_idx = coords["inline"] - 382
    j_idx = coords["crossline"] - 46
    
    trace = raw_volume[i_idx, j_idx, :]
    sswt_matrix = compute_sswt(trace, dt_s, n_cycles)
    well_sswt_data[wname] = {
        "inline": coords["inline"],
        "crossline": coords["crossline"],
        "sswt": sswt_matrix # list of 100 lists (rows) representing frequencies
    }

# Save to public directory
with open(output_json_path, "w") as f:
    json.dump(well_sswt_data, f)
    
print("Successfully saved precomputed well SSWT data to well_sswt_data.json!")
