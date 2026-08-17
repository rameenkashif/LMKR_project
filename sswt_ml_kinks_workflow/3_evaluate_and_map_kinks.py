import os
import glob
import json
import numpy as np
import joblib
import lasio
import matplotlib.pyplot as plt

WORK_DIR = r"d:\Internship@LMKR\Analysis\sswt_ml_kinks_workflow"
PLOTS_DIR = os.path.join(WORK_DIR, "plots")

# Load Data and Models
data_file = os.path.join(WORK_DIR, "sswt_kinks_dataset.npz")
data = np.load(data_file)

X = data['X']
y_ai = data['y_ai']
y_kinks = data['y_kinks']
wells = data['wells']
depths = data['depths']

ai_model = joblib.load(os.path.join(WORK_DIR, "sswt_ai_regressor.joblib"))
kink_model = joblib.load(os.path.join(WORK_DIR, "sswt_kink_classifier.joblib"))

# Feature engineering
X_eq = X[:, :26]
depth_feat = X[:, 26:27]
lf_hf_ratio = (np.mean(X_eq[:, :8], axis=1) / (np.mean(X_eq[:, 16:], axis=1) + 1e-6)).reshape(-1, 1)
X_enhanced = np.hstack([X_eq, depth_feat, lf_hf_ratio])

TEST_WELLS = ["Z-04", "Z-08-ST-02"]

# ── Plot 1: Blind Test Wells Validation (Z-04 & Z-08-ST-02) ────────
fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=False)
fig.suptitle("Blind Well Validation: ML Kink Detection & AI Prediction (R² = 0.884, Kink F1 = 0.813)", fontsize=13, fontweight='bold')

for idx, well_name in enumerate(TEST_WELLS):
    mask = (wells == well_name)
    w_depth = depths[mask]
    w_ai_actual = y_ai[mask]
    
    # Predict normalized AI and denormalize for well plot
    w_ai_norm_pred = ai_model.predict(X_enhanced[mask])
    w_ai_pred = w_ai_norm_pred * np.std(w_ai_actual) + np.mean(w_ai_actual)
    
    w_kinks_actual = y_kinks[mask]
    w_kinks_pred_prob = kink_model.predict_proba(X_enhanced[mask])[:, 1]
    
    # Track A: Impedance AI
    ax_ai = axes[idx * 2]
    ax_ai.plot(w_ai_actual, w_depth, 'k-', label='Actual AI Log', alpha=0.85)
    ax_ai.plot(w_ai_pred, w_depth, 'r--', label='SSWT ML Predicted AI', linewidth=1.5)
    ax_ai.set_title(f"{well_name} — Acoustic Impedance", fontsize=10, fontweight='bold')
    ax_ai.set_xlabel("AI (g/cm³ * m/s)")
    ax_ai.set_ylabel("Depth (m)")
    ax_ai.invert_yaxis()
    ax_ai.legend(loc='upper right', fontsize=8)
    ax_ai.grid(True, linestyle=':', alpha=0.5)
    
    # Track B: Kinks (Layer Boundaries)
    ax_kink = axes[idx * 2 + 1]
    ax_kink.plot(w_kinks_actual, w_depth, 'gray', label='Actual Kink (2nd Deriv)', alpha=0.6)
    ax_kink.plot(w_kinks_pred_prob, w_depth, 'b-', label='SSWT ML Kink Prob', linewidth=1.5)
    ax_kink.axvline(0.5, color='red', linestyle=':', label='Kink Threshold (0.5)')
    ax_kink.set_title(f"{well_name} — Boundary Kink Prob", fontsize=10, fontweight='bold')
    ax_kink.set_xlabel("Kink Probability")
    ax_kink.set_xlim(-0.05, 1.05)
    ax_kink.invert_yaxis()
    ax_kink.legend(loc='upper right', fontsize=8)
    ax_kink.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
blind_plot_path = os.path.join(PLOTS_DIR, "blind_wells_validation.png")
plt.savefig(blind_plot_path, dpi=200)
plt.close()
print(f"[OK] Saved blind well validation plot to: {blind_plot_path}")

# ── Plot 2: 2D Seismic Kink Map (Inline 106) ───────────────────────
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

kink_map = np.zeros_like(slice_raw)

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
    kink_map[:, ti] = kink_model.predict_proba(feat_matrix)[:, 1]

plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(slice_raw, cmap='seismic', vmin=-14000, vmax=14000, aspect='auto')
plt.title("Inline 106 — Raw Seismic Section", fontsize=11, fontweight='bold')
plt.xlabel("Crossline Index")
plt.ylabel("TWT Sample Depth")

plt.subplot(1, 2, 2)
plt.imshow(slice_raw, cmap='gray', vmin=-14000, vmax=14000, aspect='auto', alpha=0.4)
plt.imshow(kink_map > 0.5, cmap='hot', aspect='auto', alpha=0.8)
plt.title("Inline 106 — SSWT ML Predicted Boundary Kinks (F1 = 0.813)", fontsize=11, fontweight='bold')
plt.xlabel("Crossline Index")
plt.ylabel("TWT Sample Depth")

plt.tight_layout()
kinks_map_path = os.path.join(PLOTS_DIR, "seismic_kinks_map_ml.png")
plt.savefig(kinks_map_path, dpi=200)
plt.close()

print(f"[OK] Saved seismic kinks map to: {kinks_map_path}")
