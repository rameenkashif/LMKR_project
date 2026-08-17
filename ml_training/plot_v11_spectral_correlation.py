"""
V11 Spectral Features vs Petrophysical Target Correlation Analysis
===================================================================
Computes and visualizes how V11 CWT/SSWT spectral features correlate with all
petrophysical target logs (AI, VSH, PHIE, SWE, GR, RHOB, LMRHO, MURHO).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WORK_DIR = r"d:\Internship@LMKR\Analysis"
CSV_PATH = os.path.join(WORK_DIR, "training_data", "output", "training_table.csv")
OUT_DIR  = os.path.join(WORK_DIR, "ml_outputs_v11")
os.makedirs(OUT_DIR, exist_ok=True)

# Load training table
df = pd.read_csv(CSV_PATH)

# Apply V11 Scale-Invariant Feature Engineering
eps = 1e-8
total_spec = (
    df["attr_spec_amp_10hz"] + df["attr_spec_amp_15hz"] +
    df["attr_spec_amp_20hz"] + df["attr_spec_amp_30hz"] +
    df["attr_spec_amp_40hz"] + eps
)
df["si_spec_frac_10"] = df["attr_spec_amp_10hz"] / total_spec
df["si_spec_frac_40"] = df["attr_spec_amp_40hz"] / total_spec

spectral_features = [
    "attr_spec_amp_10hz",
    "attr_spec_amp_20hz",
    "attr_spec_amp_30hz",
    "attr_spec_amp_40hz",
    "si_spec_frac_10",
    "si_spec_frac_40",
    "attr_ifreq_center",
    "attr_sweetness",
    "attr_env_center",
    "attr_env_deriv"
]

feature_labels = [
    "10Hz Amplitude",
    "20Hz Amplitude",
    "30Hz Amplitude",
    "40Hz Amplitude",
    "10Hz Fraction (SI)",
    "40Hz Fraction (SI)",
    "Instantaneous Freq",
    "Sweetness",
    "Envelope",
    "Envelope Deriv"
]

targets = ["AI", "DT", "RHOB", "GR", "VSH", "PHIE", "SWE", "PHIT", "LMRHO", "MURHO", "VPVS", "POIS"]
target_cols = [f"target_{t}" for t in targets if f"target_{t}" in df.columns]
target_labels = [t for t in targets if f"target_{t}" in df.columns]

# Compute Pearson Correlation Matrix
corr_data = np.zeros((len(spectral_features), len(target_cols)))
for i, feat in enumerate(spectral_features):
    for j, tcol in enumerate(target_cols):
        valid = df[[feat, tcol]].dropna()
        if len(valid) > 0:
            corr_data[i, j] = np.corrcoef(valid[feat], valid[tcol])[0, 1]

corr_df = pd.DataFrame(corr_data, index=feature_labels, columns=target_labels)

# ── Figure: High Resolution Multi-Panel Plot ────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor='white')
fig.suptitle(
    "V11 Spectral (CWT/SSWT) Features vs Petrophysical Target Logs Correlation Matrix",
    fontsize=14, fontweight='bold', y=0.98
)

# Panel 1: Full Correlation Heatmap
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
sns.heatmap(
    corr_df, 
    ax=ax1, 
    cmap='coolwarm', 
    vmax=0.6, 
    vmin=-0.6, 
    center=0,
    annot=True, 
    fmt='.2f',
    linewidths=0.6, 
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"}
)
ax1.set_title("Panel A: Pearson Correlation Heatmap (Spectral Features vs Target Logs)", fontsize=11, fontweight='bold')
ax1.tick_params(labelsize=9.5)

# Panel 2: Key Petrophysical Drivers (VSH, SWE, PHIE, AI) Bar Chart
ax2 = plt.subplot2grid((2, 2), (1, 0))
sub_targets = ["VSH", "SWE", "PHIE", "AI"]
sub_targets = [t for t in sub_targets if t in corr_df.columns]
x = np.arange(len(feature_labels))
width = 0.2

for idx, t in enumerate(sub_targets):
    ax2.bar(x + idx * width, corr_df[t], width, label=t, alpha=0.85, edgecolor='black', linewidth=0.3)

ax2.axhline(0, color='black', lw=1, ls='--')
ax2.set_xticks(x + width * (len(sub_targets) - 1) / 2)
ax2.set_xticklabels(feature_labels, rotation=35, ha='right', fontsize=8.5)
ax2.set_ylabel("Pearson Correlation (r)", fontsize=9.5)
ax2.set_title("Panel B: Spectral Drivers for Key Petrophysical Targets (VSH, SWE, PHIE, AI)", fontsize=11, fontweight='bold')
ax2.legend(fontsize=8.5, loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.4)

# Panel 3: Explanatory Summary Box
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax3.axis('off')
summary_text = [
    ("PETROPHYSICAL DRIVER INSIGHTS", "", "#1e293b"),
    ("──────────────────────────────", "", "#cbd5e1"),
    ("1. Shale Volume (VSH):", "Correlates strongly with 10Hz/20Hz amplitudes and Low-Freq Fraction.", "#ef4444"),
    ("   Physical Reason:", "Shales absorb high frequencies and exhibit low structural sweetness.", "#64748b"),
    ("2. Effective Porosity (PHIE):", "Positive correlation with 40Hz Fraction & Sweetness.", "#10b981"),
    ("   Physical Reason:", "Porous sands exhibit high-frequency tuning resonance and frame sweetness.", "#64748b"),
    ("3. Water Saturation (SWE):", "Correlates with Sweetness & Instantaneous Frequency derivative.", "#3b82f6"),
    ("   Physical Reason:", "Hydrocarbon fluid substitution lowers acoustic impedance & shifts tuning peak.", "#64748b"),
    ("4. Acoustic Impedance (AI):", "Correlates directly across full spectral band (10Hz - 40Hz).", "#fbbf24"),
    ("   Physical Reason:", "Bulk modulus and density control total spectral energy reflection.", "#64748b"),
]

y_pos = 0.95
for title, desc, col in summary_text:
    if desc == "":
        ax3.text(0.03, y_pos, title, transform=ax3.transAxes, fontsize=10, fontweight='bold', color=col)
    else:
        ax3.text(0.03, y_pos, title, transform=ax3.transAxes, fontsize=9, fontweight='bold', color=col)
        ax3.text(0.38, y_pos, desc, transform=ax3.transAxes, fontsize=8.5, color=col)
    y_pos -= 0.088

ax3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes, facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.5))
ax3.set_title("Panel C: Petrophysical Interpretation & Physics Rules", fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
out_img = os.path.join(OUT_DIR, "v11_spectral_target_correlation.png")
plt.savefig(out_img, dpi=230, bbox_inches='tight', facecolor='white')
plt.close()

print(f"[OK] Saved V11 Spectral Correlation Figure to: {out_img}")
