import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

WORK_DIR = r"d:\Internship@LMKR\Analysis\sswt_ml_kinks_workflow"
PLOTS_DIR = os.path.join(WORK_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load Trained Models and Dataset Meta
ai_model = joblib.load(os.path.join(WORK_DIR, "sswt_ai_regressor.joblib"))
kink_model = joblib.load(os.path.join(WORK_DIR, "sswt_kink_classifier.joblib"))

data_file = os.path.join(WORK_DIR, "sswt_kinks_dataset.npz")
data = np.load(data_file)
y_ai_mean = np.mean(data['y_ai'])
y_ai_std = np.std(data['y_ai'])

# Load 3D SEGY Inline 106
raw_data = np.fromfile(r"d:\Internship@LMKR\Analysis\frontend\public\seismic_raw.bin", dtype=np.int16).astype(np.float32) / 32767.0
with open(r"d:\Internship@LMKR\Analysis\frontend\public\seismic_raw_meta.json") as f:
    scale = json.load(f)["scale"]
raw_vol = (raw_data * scale).reshape((245, 252, 313))
slice_raw = raw_vol[106, :, :].T # shape (313, 252)

num_traces = slice_raw.shape[1]
K_len = slice_raw.shape[0]

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

ai_map = np.zeros_like(slice_raw)
kink_prob_map = np.zeros_like(slice_raw)

# Compute 2D ML Inversion Section
for ti in range(num_traces):
    trace = slice_raw[:, ti]
    freq_matrix = np.zeros((K_len, len(RIDGE_FREQS)))
    
    for fi, (wav_re, wav_im, f) in enumerate(wavelets):
        cwt_re = np.convolve(trace, wav_re, mode='same')
        cwt_im = np.convolve(trace, wav_im, mode='same')
        freq_matrix[:, fi] = np.sqrt(cwt_re**2 + cwt_im**2)
        
    mean_spec = np.mean(freq_matrix, axis=0) + 1e-9
    freq_matrix_eq = freq_matrix / mean_spec
    
    norm_depth = np.linspace(0, 1, K_len).reshape(-1, 1)
    lf_hf = (np.mean(freq_matrix_eq[:, :8], axis=1) / (np.mean(freq_matrix_eq[:, 16:], axis=1) + 1e-6)).reshape(-1, 1)
    
    feat_matrix = np.hstack([freq_matrix_eq, norm_depth, lf_hf])
    
    # Predict relative AI from ML model
    norm_ai_pred = ai_model.predict(feat_matrix)
    ai_map[:, ti] = norm_ai_pred
    
    # Predict Kink Probability
    kink_prob_map[:, ti] = kink_model.predict_proba(feat_matrix)[:, 1]

# Apply smooth spatial 2x2 Gaussian filter for clean continuous horizon lines
ai_smooth = gaussian_filter(ai_map, sigma=(1.2, 1.2))

# Mask non-data background outside reflection package
trace_envelope = np.abs(slice_raw)
mask = trace_envelope < 1200.0

raw_masked = np.copy(slice_raw)
raw_masked[mask] = np.nan

ai_masked = np.copy(ai_smooth)
ai_masked[mask] = np.nan

# ── Red-White-Blue Seismic with Distinct Thin White Horizon Lines ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='white')
fig.suptitle("SSWT ML Inversion: Red-White-Blue Seismic with Distinct Thin Bed Horizon Lines", fontsize=13, fontweight='bold')

# Panel A: Raw Seismic Profile
ax0 = axes[0]
im0 = ax0.imshow(raw_masked, cmap='seismic', vmin=-14000, vmax=14000, aspect='auto')
ax0.set_title("Panel A — Raw Seismic Profile", fontsize=11, fontweight='bold')
ax0.set_xlabel("Crossline Index")
ax0.set_ylabel("TWT Sample Depth")
fig.colorbar(im0, ax=ax0, label="Amplitude")

# Panel B: SSWT ML Inversion with Continuous Thin White Horizon Lines
ax1 = axes[1]
im1 = ax1.imshow(raw_masked, cmap='seismic', vmin=-14000, vmax=14000, aspect='auto')

# Draw continuous distinct thin white horizon boundary lines (kinks)
contours = ax1.contour(ai_masked, levels=14, colors='white', linewidths=1.1, alpha=0.92)
ax1.clabel(contours, inline=True, fontsize=7, fmt='%.1f')

# Overlay peak boundary kink lines in crisp dashed yellow
kink_contour = ax1.contour(gaussian_filter(kink_prob_map, sigma=(1.0, 1.0)), levels=[0.52], colors='gold', linestyles='--', linewidths=1.3)

ax1.set_title("Panel B — SSWT ML Inversion (Distinct Thin Bed Boundary Lines)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Crossline Index")
ax1.set_ylabel("TWT Sample Depth")
fig.colorbar(im1, ax=ax1, label="Seismic Amplitude + Thin Bed Horizons")

plt.tight_layout()
distinct_lines_path = os.path.join(PLOTS_DIR, "seismic_2d_distinct_horizon_lines.png")
plt.savefig(distinct_lines_path, dpi=250)
plt.close()

print(f"[OK] Saved Distinct Horizon Lines Section to: {distinct_lines_path}")
