"""
Feature-to-Target Correlation Heatmap & Feature Importance Analysis
====================================================================
Computes and visualizes:
  1. Feature-to-Target Pearson Correlation with Acoustic Impedance (AI) and Kinks
  2. Full Inter-Feature Correlation Heatmap across all 28 engineered SSWT features
  3. Feature Importance ranking from the trained XGBoost / Random Forest models
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

WORK_DIR = r"d:\Internship@LMKR\Analysis\sswt_ml_kinks_workflow"
PLOTS_DIR = os.path.join(WORK_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load Dataset and Models ─────────────────────────────────────────
data = np.load(os.path.join(WORK_DIR, "sswt_kinks_dataset.npz"))
X_raw    = data['X']
y_ai     = data['y_ai']
y_kinks  = data['y_kinks']
freqs    = data['freqs'] # 26 frequencies: 10 to 60 Hz

ai_model   = joblib.load(os.path.join(WORK_DIR, "sswt_ai_regressor.joblib"))
kink_model = joblib.load(os.path.join(WORK_DIR, "sswt_kink_classifier.joblib"))

# Feature Engineering (28 features total)
X_eq        = X_raw[:, :26]
depth_feat  = X_raw[:, 26:27]
lf_hf_ratio = (np.mean(X_eq[:, :8], axis=1) / (np.mean(X_eq[:, 16:], axis=1) + 1e-6)).reshape(-1, 1)
X_enhanced  = np.hstack([X_eq, depth_feat, lf_hf_ratio])

feature_names = [f"F_{int(f)}Hz" for f in freqs] + ["Depth_Norm", "LF_HF_Ratio"]

df = pd.DataFrame(X_enhanced, columns=feature_names)
df['Target_AI']    = y_ai
df['Target_Kinks'] = y_kinks

# Calculate correlation matrix
corr_matrix = df.corr()

# ── Figure 1: Comprehensive Correlation Heatmap & Feature Ranking ────
fig = plt.figure(figsize=(18, 12), facecolor='white')
fig.suptitle(
    "SSWT / CWT Feature Correlation Analysis & ML Feature Importance\n"
    "(26 Frequency Bins [10-60 Hz] + Depth_Norm + LF/HF Ratio)",
    fontsize=14, fontweight='bold', y=0.98
)

# Panel 1: Full Inter-Feature Correlation Heatmap
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, 
    ax=ax1, 
    cmap='coolwarm', 
    vmax=1.0, 
    vmin=-1.0, 
    center=0,
    square=True, 
    linewidths=0.5, 
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
    annot=False
)
ax1.set_title("Panel A: 28-Feature Inter-Correlation Matrix", fontsize=11, fontweight='bold')
ax1.tick_params(labelsize=8)

# Panel 2: Feature Correlation with Target AI
ax2 = plt.subplot2grid((2, 2), (0, 1))
ai_corrs = corr_matrix['Target_AI'].drop(['Target_AI', 'Target_Kinks'])
ai_corrs_sorted = ai_corrs.sort_values()

colors_ai = ['#ef4444' if x < 0 else '#10b981' for x in ai_corrs_sorted]
ax2.barh(ai_corrs_sorted.index, ai_corrs_sorted.values, color=colors_ai, alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.axvline(0, color='black', lw=1, ls='--')
ax2.set_xlabel("Pearson Correlation (r) with Acoustic Impedance (AI)", fontsize=9.5)
ax2.set_title("Panel B: Feature Correlation with Target AI", fontsize=11, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.tick_params(labelsize=8)

for i, (val, name) in enumerate(zip(ai_corrs_sorted.values, ai_corrs_sorted.index)):
    offset = 0.01 if val >= 0 else -0.05
    ax2.text(val + offset, i, f"{val:.2f}", va='center', fontsize=7.5, fontweight='bold')

# Panel 3: ML Feature Importance from Trained Model
ax3 = plt.subplot2grid((2, 2), (1, 1))
importances = ai_model.feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

ax3.barh(imp_series.index, imp_series.values, color='#3b82f6', alpha=0.85, edgecolor='black', linewidth=0.5)
ax3.set_xlabel("Random Forest Feature Importance (MDI)", fontsize=9.5)
ax3.set_title("Panel C: Trained ML Model Feature Importance Ranking", fontsize=11, fontweight='bold')
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.tick_params(labelsize=8)

for i, (val, name) in enumerate(zip(imp_series.values, imp_series.index)):
    ax3.text(val + 0.001, i, f"{val*100:.1f}%", va='center', fontsize=7.5, fontweight='bold', color='#1e293b')

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "feature_correlation_matrix.png")
plt.savefig(out_path, dpi=230, bbox_inches='tight', facecolor='white')
plt.close()

print(f"[OK] Saved Feature Correlation Analysis Plot to: {out_path}")
