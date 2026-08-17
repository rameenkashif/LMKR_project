"""
Experiment: What Happens If We Drop Negative Correlation Features?
===================================================================
Compares:
  Model A: All 28 Features
  Model B: Only Positive Correlation Features (Negative Ones Dropped)
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, f1_score, mean_absolute_error

WORK_DIR = r"d:\Internship@LMKR\Analysis\sswt_ml_kinks_workflow"
data = np.load(os.path.join(WORK_DIR, "sswt_kinks_dataset.npz"))

X_raw    = data['X']
y_ai     = data['y_ai']
y_kinks  = data['y_kinks']
wells    = data['wells']

TRAIN_WELLS = ["Z-02", "Z-03", "Z-05", "Z-06", "Z-07"]
TEST_WELLS  = ["Z-04", "Z-08-ST-02"]

train_mask = np.isin(wells, TRAIN_WELLS)
test_mask  = np.isin(wells, TEST_WELLS)

# Feature engineering (28 features total)
X_eq        = X_raw[:, :26]
depth_feat  = X_raw[:, 26:27]
lf_hf_ratio = (np.mean(X_eq[:, :8], axis=1) / (np.mean(X_eq[:, 16:], axis=1) + 1e-6)).reshape(-1, 1)
X_all       = np.hstack([X_eq, depth_feat, lf_hf_ratio])

freq_names = [f"F_{int(f)}Hz" for f in np.linspace(10, 60, 26)]
feat_names = freq_names + ["Depth_Norm", "LF_HF_Ratio"]

# Standardize AI per well
y_ai_norm = np.zeros_like(y_ai)
for w in np.unique(wells):
    wm = (wells == w)
    w_std = np.std(y_ai[wm]) + 1e-6
    y_ai_norm[wm] = (y_ai[wm] - np.mean(y_ai[wm])) / w_std

# Smooth kinks
y_kinks_smooth = np.copy(y_kinks)
for i in range(1, len(y_kinks) - 1):
    if wells[i] == wells[i-1] == wells[i+1]:
        if y_kinks[i-1] == 1 or y_kinks[i+1] == 1:
            y_kinks_smooth[i] = 1

# Calculate correlation of each feature with training target AI
corrs = [np.corrcoef(X_all[train_mask, j], y_ai_norm[train_mask])[0, 1] for j in range(28)]
pos_mask = np.array(corrs) > 0

pos_features = [feat_names[j] for j in range(28) if pos_mask[j]]
neg_features = [feat_names[j] for j in range(28) if not pos_mask[j]]

print("==========================================================")
print(" FEATURE SELECTION EXPERIMENT: DROPPING NEGATIVE CORRELATIONS")
print("==========================================================")
print(f"Positive Correlation Features ({len(pos_features)}): {pos_features}")
print(f"Negative Correlation Features ({len(neg_features)}): {neg_features}")
print("----------------------------------------------------------")

# Model A: All 28 Features
rf_ai_all = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
rf_ai_all.fit(X_all[train_mask], y_ai_norm[train_mask])
pred_all = rf_ai_all.predict(X_all[test_mask])

rf_kink_all = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
rf_kink_all.fit(X_all[train_mask], y_kinks_smooth[train_mask])
pred_kink_all = rf_kink_all.predict(X_all[test_mask])

# Model B: Only Positive Features
X_pos = X_all[:, pos_mask]
rf_ai_pos = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
rf_ai_pos.fit(X_pos[train_mask], y_ai_norm[train_mask])
pred_pos = rf_ai_pos.predict(X_pos[test_mask])

rf_kink_pos = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
rf_kink_pos.fit(X_pos[train_mask], y_kinks_smooth[train_mask])
pred_kink_pos = rf_kink_pos.predict(X_pos[test_mask])

# Compare Results
r2_all  = r2_score(y_ai_norm[test_mask], pred_all)
r2_pos  = r2_score(y_ai_norm[test_mask], pred_pos)
mae_all = mean_absolute_error(y_ai_norm[test_mask], pred_all)
mae_pos = mean_absolute_error(y_ai_norm[test_mask], pred_pos)

f1_all  = f1_score(y_kinks_smooth[test_mask], pred_kink_all)
f1_pos  = f1_score(y_kinks_smooth[test_mask], pred_kink_pos)

print("\n=== PERFORMANCE COMPARISON ON BLIND TEST WELLS ===")
print(f"Model A (All 28 Features):")
print(f"  - AI Prediction R² Score : {r2_all:.4f}")
print(f"  - AI Prediction MAE Error: {mae_all:.4f}")
print(f"  - Kink Detection F1-Score: {f1_all:.4f} ({f1_all*100:.1f}%)")

print(f"\nModel B (Only Positive Features - Dropped {len(neg_features)} Features):")
print(f"  - AI Prediction R² Score : {r2_pos:.4f}")
print(f"  - AI Prediction MAE Error: {mae_pos:.4f}")
print(f"  - Kink Detection F1-Score: {f1_pos:.4f} ({f1_pos*100:.1f}%)")
print("==========================================================")
