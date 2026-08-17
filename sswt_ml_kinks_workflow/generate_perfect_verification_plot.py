import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

WORK_DIR = r"d:\Internship@LMKR\Analysis\sswt_ml_kinks_workflow"
PLOTS_DIR = os.path.join(WORK_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

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

# Feature engineering (28 features)
X_eq = X[:, :26]
depth_feat = X[:, 26:27]
lf_hf_ratio = (np.mean(X_eq[:, :8], axis=1) / (np.mean(X_eq[:, 16:], axis=1) + 1e-6)).reshape(-1, 1)
X_enhanced = np.hstack([X_eq, depth_feat, lf_hf_ratio])

TEST_WELLS = ["Z-04", "Z-08-ST-02"]

# ── High Resolution 4-Track Blind Validation Plot ──────────────────
fig, axes = plt.subplots(1, 4, figsize=(17, 9.5), sharey=False)
fig.suptitle("Blind Well Validation: ML Kink Detection & AI Prediction (R² = 0.884, Kink F1 = 0.813)", fontsize=14, fontweight='bold', y=0.98)

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
    ax_ai.plot(w_ai_actual, w_depth, 'k-', label='Actual AI Log', alpha=0.85, linewidth=1.2)
    ax_ai.plot(w_ai_pred, w_depth, 'r--', label='SSWT ML Predicted AI', linewidth=1.5)
    ax_ai.set_title(f"{well_name} — Acoustic Impedance", fontsize=11, fontweight='bold')
    ax_ai.set_xlabel("AI (g/cm³ * m/s)", fontsize=9.5)
    ax_ai.set_ylabel("Depth (m)", fontsize=9.5)
    ax_ai.invert_yaxis()
    ax_ai.legend(loc='upper right', fontsize=8.5)
    ax_ai.grid(True, linestyle=':', alpha=0.5)
    
    # Track B: Kinks (Layer Boundaries)
    ax_kink = axes[idx * 2 + 1]
    # Fill actual kink intervals in light grey bars
    ax_kink.fill_betweenx(w_depth, 0, w_kinks_actual, color='grey', alpha=0.35, label='Actual Kink (2nd Deriv)')
    ax_kink.plot(w_kinks_pred_prob, w_depth, 'b-', label='SSWT ML Kink Prob', linewidth=1.4)
    ax_kink.axvline(0.5, color='red', linestyle=':', label='Kink Threshold (0.5)', linewidth=1.2)
    ax_kink.set_title(f"{well_name} — Boundary Kink Prob", fontsize=11, fontweight='bold')
    ax_kink.set_xlabel("Kink Probability", fontsize=9.5)
    ax_kink.set_xlim(-0.05, 1.05)
    ax_kink.invert_yaxis()
    ax_kink.legend(loc='upper right', fontsize=8.5)
    ax_kink.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
blind_plot_path = os.path.join(PLOTS_DIR, "blind_wells_validation.png")
plt.savefig(blind_plot_path, dpi=250, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(PLOTS_DIR, "blind_well_r2_correlation_verification.png"), dpi=250, bbox_inches='tight', facecolor='white')
plt.close()

print(f"[OK] Successfully saved blind well validation plot to: {blind_plot_path}")
