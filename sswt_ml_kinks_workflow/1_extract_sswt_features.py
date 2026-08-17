import os
import glob
import numpy as np
import lasio
import json

WORK_DIR = r"d:\Internship@LMKR\Analysis\sswt_ml_kinks_workflow"
PLOTS_DIR = os.path.join(WORK_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

LAS_DIR = r"d:\Internship@LMKR\Analysis\las_cleaned"
las_files = sorted(glob.glob(os.path.join(LAS_DIR, "*.las")))

# SSWT / CWT Morlet Parameters (10 Hz to 60 Hz across 26 frequency bins)
RIDGE_FREQS = np.linspace(10, 60, 26)
dt = 0.002
n_cycles = 3.5

wavelets = []
for freq in RIDGE_FREQS:
    sigma = n_cycles / (2 * np.pi * freq)
    win_len = int(np.ceil(6 * sigma / dt))
    if win_len % 2 == 0: win_len += 1
    t_w = np.arange(-win_len//2, win_len//2 + 1) * dt
    wav_re = np.exp(-0.5 * (t_w / sigma)**2) * np.cos(2 * np.pi * freq * t_w)
    wav_im = np.exp(-0.5 * (t_w / sigma)**2) * np.sin(2 * np.pi * freq * t_w)
    norm = np.sqrt(np.sum(wav_re**2 + wav_im**2))
    wavelets.append((wav_re / norm, wav_im / norm, freq))

all_features = []
all_ai_targets = []
all_kink_targets = []
all_well_names = []
all_depths = []

for filepath in las_files:
    well_name = os.path.splitext(os.path.basename(filepath))[0]
    las = lasio.read(filepath)
    df = las.df().dropna(subset=['DT', 'RHOB'])
    
    dt_log = df['DT'].values
    rhob_log = df['RHOB'].values
    depth = df.index.values
    
    vp = 1e6 / dt_log # m/s
    ai = vp * rhob_log # (m/s)*(g/cm3)
    
    # 2nd derivative of AI for sharp layer boundary "kinks"
    d2_ai = np.gradient(np.gradient(ai))
    kink_labels = (np.abs(d2_ai) > np.percentile(np.abs(d2_ai), 80)).astype(int)
    
    # Synthetic seismic trace
    rc = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1] + 1e-9)
    rc = np.pad(rc, (0, 1), 'constant')
    
    t_rick = np.arange(-0.04, 0.04, dt)
    ricker = (1 - 2 * (np.pi * 30 * t_rick)**2) * np.exp(-(np.pi * 30 * t_rick)**2)
    seismic_trace = np.convolve(rc, ricker, mode='same')
    
    N = len(seismic_trace)
    freq_matrix = np.zeros((N, len(RIDGE_FREQS)))
    
    for fi, (wav_re, wav_im, f) in enumerate(wavelets):
        cwt_re = np.convolve(seismic_trace, wav_re, mode='same')
        cwt_im = np.convolve(seismic_trace, wav_im, mode='same')
        cwt_mag = np.sqrt(cwt_re**2 + cwt_im**2)
        freq_matrix[:, fi] = cwt_mag
        
    mean_spec = np.mean(freq_matrix, axis=0) + 1e-9
    freq_matrix_eq = freq_matrix / mean_spec
    
    # Include normalized depth (0 to 1) as physical feature alongside 26 frequency bins
    norm_depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    feat_matrix = np.column_stack([freq_matrix_eq, norm_depth.reshape(-1, 1)])
    
    all_features.append(feat_matrix)
    all_ai_targets.append(ai)
    all_kink_targets.append(kink_labels)
    all_well_names.extend([well_name] * N)
    all_depths.append(depth)

X_all = np.vstack(all_features)
y_ai = np.concatenate(all_ai_targets)
y_kinks = np.concatenate(all_kink_targets)
well_names_arr = np.array(all_well_names)
depths_arr = np.concatenate(all_depths)

output_file = os.path.join(WORK_DIR, "sswt_kinks_dataset.npz")
np.savez_compressed(
    output_file,
    X=X_all,
    y_ai=y_ai,
    y_kinks=y_kinks,
    wells=well_names_arr,
    depths=depths_arr,
    freqs=RIDGE_FREQS
)

print(f"Features updated with normalized depth! Shape: {X_all.shape}")
