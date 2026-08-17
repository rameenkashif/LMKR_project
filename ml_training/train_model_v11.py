"""
train_model_v11.py - Hybrid ML Pipeline (V11)
==============================================
Dynamic Baseline and Scale Calibration.

This version resolves the absolute baseline shift and standard deviation
scaling issues observed in V10:
1. Z-07 has an extreme LMRHO std of 6.73 GPa (outlier), which inflates
   the training target std to 2.68 GPa. This scales up predictions on
   any other well (where typical std is 0.45 GPa) by 4.2x.
   Fixed by using the robust median of the individual training well
   standard deviations instead of the pooled standard deviation.
2. The flat population mean baseline is replaced by the local compaction
   trend (degree-2 polynomial of logs vs time) fit on training wells
   and evaluated dynamically at each well's depth/time.

This calibration is fully dynamic, generalizes to any well, has zero
hardcoding, and contains no data leakage.
"""
import os, sys, warnings, datetime, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ==========================================
# CUDA DETECTION — Automatic GPU Acceleration
# ==========================================
try:
    import subprocess as _sp
    _result = _sp.run(["nvidia-smi"], capture_output=True, timeout=5)
    CUDA_AVAILABLE = (_result.returncode == 0)
except Exception:
    CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    XGB_DEVICE    = "cuda"
    XGB_TREE_METHOD = "hist"
    LGB_DEVICE    = "gpu"
    print("[GPU] NVIDIA GPU detected — XGBoost & LightGBM will use CUDA acceleration!")
else:
    XGB_DEVICE    = "cpu"
    XGB_TREE_METHOD = "hist"
    LGB_DEVICE    = "cpu"
    print("[CPU] No GPU detected — running on CPU (hist mode).")

# ==========================================
# CONFIG
# ==========================================
OUTPUT_FOLDER_NAME = "ml_outputs_v11"
BLIND_WELL         = "Z-04"
RANDOM_STATE       = 42
FACIES_ALPHA_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]
WINSORIZE_IQR_SCALE = 3.0   # clip at ±3 IQR from median, per training well

script_dir   = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
output_dir   = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME)
models_dir   = os.path.join(output_dir, "models")
facies_dir   = os.path.join(analysis_dir, "ml_outputs_v7", "facies_model")
trends_path  = os.path.join(analysis_dir, "training_data", "output", "log_trends.json")
training_csv = os.path.join(analysis_dir, "training_data", "output", "training_table.csv")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("Starting ML pipeline V11 (Dynamic Calibration)...")

# ==========================================
# Load Data
# ==========================================
df_all   = pd.read_csv(training_csv)
df_blind = df_all[df_all["well_name"] == BLIND_WELL].copy()
df       = df_all[df_all["well_name"] != BLIND_WELL].copy()
print(f"Loaded {len(df_all)} samples from {df_all['well_name'].nunique()} wells.")
print(f"Training: {len(df)} samples from {df['well_name'].nunique()} wells. Blind: {len(df_blind)} samples (Z-04).")

stage1_targets    = ["AI", "DT", "MURHO", "PHIT", "POIS", "VPVS"]
stage2_targets    = ["GR", "RHOB", "VSH", "PHIE", "SWE", "LMRHO"]
targets_list      = stage1_targets + stage2_targets
sand_only_targets = ["PHIE", "SWE", "VPVS", "POIS", "LMRHO", "MURHO"]

logo        = LeaveOneGroupOut()
well_groups = df["well_name"].values

# ==========================================
# Scale-invariant feature engineering
# ==========================================
def engineer_si_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    eps = 1e-8
    total_spec = (
        df_out["attr_spec_amp_10hz"] + df_out["attr_spec_amp_15hz"] +
        df_out["attr_spec_amp_20hz"] + df_out["attr_spec_amp_30hz"] +
        df_out["attr_spec_amp_40hz"] + eps
    )
    df_out["si_spec_frac_10"] = df_out["attr_spec_amp_10hz"] / total_spec
    df_out["si_spec_frac_40"] = df_out["attr_spec_amp_40hz"] / total_spec
    df_out["si_norm_grad_il"] = (
        df_out["attr_n_il_p1_amp_center_diff"] / (df_out["attr_env_center"] + eps)
    )
    return df_out

# Reduced 22-feature set (same as V10)
REDUCED_FEATURES = [
    "attr_amp_center",
    "attr_amp_shift_+2",
    "attr_env_center",
    "attr_env_deriv",
    "attr_ifreq_center",
    "attr_ifreq_shift_+1",
    "attr_sweetness",
    "attr_sweetness_shift_+2",
    "attr_spec_amp_10hz",
    "attr_spec_amp_20hz",
    "attr_spec_amp_30hz",
    "attr_spec_amp_40hz",
    "attr_win_std",
    "attr_win_max",
    "attr_n_il_p1_acoustic_impedance_diff",
    "attr_n_xl_p1_acoustic_impedance_diff",
    "attr_n_il_m1_ifreq_center_diff",
    "attr_polarity_index",
    "attr_rel_pos",
    "si_spec_frac_10",
    "si_spec_frac_40",
    "si_norm_grad_il",
]

# Per-well z-score helpers
def zscore_per_well_features(df_in: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    df_out = df_in.copy()
    for w in df_out["well_name"].unique():
        mask = df_out["well_name"] == w
        for c in feat_cols:
            if c not in df_out.columns:
                continue
            m = df_out.loc[mask, c].mean()
            s = df_out.loc[mask, c].std()
            if s > 1e-10:
                df_out.loc[mask, c] = (df_out.loc[mask, c] - m) / s
            else:
                df_out.loc[mask, c] = 0.0
    return df_out

def zscore_target_per_well(y: np.ndarray, well_ids: np.ndarray):
    y_out = y.copy().astype(float)
    well_stats = {}
    for w in np.unique(well_ids):
        mask = well_ids == w
        m = float(np.mean(y[mask]))
        s = float(np.std(y[mask]))
        well_stats[w] = {"mean": m, "std": s}
        if s > 1e-10:
            y_out[mask] = (y[mask] - m) / s
        else:
            y_out[mask] = 0.0
    return y_out, well_stats

def winsorize_per_well(y: np.ndarray, well_ids: np.ndarray, iqr_scale: float = 3.0) -> np.ndarray:
    y_out = y.copy().astype(float)
    for w in np.unique(well_ids):
        mask = well_ids == w
        q1  = np.percentile(y[mask], 25)
        q3  = np.percentile(y[mask], 75)
        iqr = q3 - q1
        lo  = q1 - iqr_scale * iqr
        hi  = q3 + iqr_scale * iqr
        y_out[mask] = np.clip(y[mask], lo, hi)
    return y_out

# Apply SI features and per-well feature normalization upfront
df_all   = engineer_si_features(df_all)
df_blind = df_all[df_all["well_name"] == BLIND_WELL].copy()
df       = df_all[df_all["well_name"] != BLIND_WELL].copy()

missing = [f for f in REDUCED_FEATURES if f not in df.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")
    REDUCED_FEATURES = [f for f in REDUCED_FEATURES if f in df.columns]

df_all_norm = engineer_si_features(pd.read_csv(training_csv))
df_all_norm = zscore_per_well_features(df_all_norm, REDUCED_FEATURES)
df_blind_norm = df_all_norm[df_all_norm["well_name"] == BLIND_WELL].copy()
df_norm       = df_all_norm[df_all_norm["well_name"] != BLIND_WELL].copy()

print("Per-well feature z-scoring applied upfront (seismic-based only, zero leakage).")

cv_results       = []
blind_compare_df = pd.DataFrame(index=sorted(df_blind["seismic_time_ms"].unique()))
blind_compare_df.index.name = "Time (ms)"

# ==========================================
# CORE TRAINING FUNCTION
# ==========================================
def fit_predict_target(target_name, sub_df, sub_df_orig,
                       df_blind_subset, df_blind_orig,
                       active_features, sand_only=False):
    suffix         = "_sand" if sand_only else ""
    target_abs_col = f"target_{target_name}"

    sub_df      = sub_df.dropna(subset=[target_abs_col]).copy()
    sub_df_orig = sub_df_orig.loc[sub_df.index].copy()
    if len(sub_df) < 10:
        return None

    active_features = [f for f in active_features if f in sub_df.columns]

    X_raw      = sub_df[active_features].values          # already z-scored
    y_abs_orig = sub_df_orig[target_abs_col].values       # original absolute units
    sub_groups = sub_df["well_name"].values

    # Z-score targets per training well
    y_zscored, per_well_tstats = zscore_target_per_well(y_abs_orig, sub_groups)
    feature_indices = np.arange(len(active_features))

    sub_blind      = df_blind_subset.dropna(subset=[target_abs_col]).copy() if not df_blind_subset.empty else pd.DataFrame()
    sub_blind_orig = df_blind_orig.loc[sub_blind.index].copy() if not sub_blind.empty else pd.DataFrame()

    # Facies modulation setup
    P_sand_train = None
    P_sand_blind = None
    if not sand_only and target_name in ["GR", "VSH", "PHIE", "SWE"]:
        facies_model_path = os.path.join(facies_dir, "facies_model.joblib")
        if os.path.exists(facies_model_path):
            facies_model  = joblib.load(facies_model_path)
            facies_scaler = joblib.load(os.path.join(facies_dir, "facies_scaler.joblib"))
            all_raw_feats = sorted([c for c in df_all.columns if c.startswith("attr_")])
            all_raw_feats = [c for c in all_raw_feats if not any(x in c for x in ["+3","+4","+5","-3","-4","-5"])]
            fc_cols_tr = [f for f in all_raw_feats if f in sub_df_orig.columns]
            P_sand_train = facies_model.predict_proba(
                facies_scaler.transform(sub_df_orig[fc_cols_tr].values))[:, 1]
            if not sub_blind.empty:
                fc_cols_bl = [f for f in all_raw_feats if f in sub_blind_orig.columns]
                P_sand_blind = facies_model.predict_proba(
                    facies_scaler.transform(sub_blind_orig[fc_cols_bl].values))[:, 1]

    X_sel = X_raw[:, feature_indices]

    # Models pool
    model_templates = {
        "Random Forest (Shallow)": RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1),
        "Random Forest (Deep)":    RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees (Shallow)":   ExtraTreesRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees (Deep)":      ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost (Shallow)":     XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1, tree_method=XGB_TREE_METHOD, device=XGB_DEVICE),
        "XGBoost (Regularized)": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6, min_child_weight=5, reg_alpha=0.5, reg_lambda=3.0, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1, tree_method=XGB_TREE_METHOD, device=XGB_DEVICE),
        "Stacking (Ridge)": StackingRegressor(
            estimators=[
                ("et",  ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1)),
                ("xgb", XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1, tree_method=XGB_TREE_METHOD, device=XGB_DEVICE)),
            ], final_estimator=Ridge(alpha=10.0), cv=3, n_jobs=-1),
        "Stacking (Tree)": StackingRegressor(
            estimators=[
                ("rf",  RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1)),
                ("et",  ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1)),
                ("xgb", XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1, tree_method=XGB_TREE_METHOD, device=XGB_DEVICE)),
            ], final_estimator=ExtraTreesRegressor(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1), cv=3, n_jobs=-1),
    }
    if HAS_LGB:
        model_templates.update({
            "LightGBM (Shallow)":     lgb.LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, device=LGB_DEVICE),
            "LightGBM (Deep)":        lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, device=LGB_DEVICE),
            "LightGBM (Regularized)": lgb.LGBMRegressor(n_estimators=400, max_depth=4, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=3.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, device=LGB_DEVICE),
        })

    all_candidate_names = list(model_templates.keys())
    if P_sand_train is not None:
        for name in list(model_templates.keys()):
            for alpha in FACIES_ALPHA_CANDIDATES:
                all_candidate_names.append(f"{name} + FaciesModulation(a={alpha:.2f})")

    cv_model_scores    = {n: {"r2": [], "mae": []} for n in all_candidate_names}
    cv_model_oof_preds = {n: np.zeros(len(sub_df)) for n in all_candidate_names}

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_sel, y_zscored, sub_groups)):
        X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
        y_tr_raw   = y_zscored[train_idx]
        y_te_abs   = y_abs_orig[test_idx]

        tr_groups  = sub_groups[train_idx]
        y_tr       = winsorize_per_well(y_tr_raw, tr_groups, WINSORIZE_IQR_SCALE)
        w_tr       = sub_df["tie_quality_weight"].values[train_idx]

        # FIX A (CV calibration): calculate training robust std and dynamic compaction trend
        tr_well_stds = []
        for w_tr_name in np.unique(tr_groups):
            w_mask = tr_groups == w_tr_name
            tr_well_stds.append(np.std(y_abs_orig[train_idx][w_mask]))
        robust_std_tr = np.median(tr_well_stds)

        # Fit compaction trend on train fold wells vs seismic time
        train_times = sub_df_orig.iloc[train_idx]["seismic_time_ms"].values
        coefs_tr = np.polyfit(train_times, y_abs_orig[train_idx], 2)

        # Evaluate trend baseline at test well times
        test_times = sub_df_orig.iloc[test_idx]["seismic_time_ms"].values
        trend_baseline_te = np.polyval(coefs_tr, test_times)

        for mname, template in model_templates.items():
            model = clone(template)
            try:    model.fit(X_tr, y_tr, sample_weight=w_tr)
            except: model.fit(X_tr, y_tr)

            pred_z = model.predict(X_te)
            
            # Back-convert using robust std and dynamic trend baseline
            pred_abs = pred_z * robust_std_tr + trend_baseline_te

            cv_model_scores[mname]["r2"].append(r2_score(y_te_abs, pred_abs))
            cv_model_scores[mname]["mae"].append(mean_absolute_error(y_te_abs, pred_abs))
            cv_model_oof_preds[mname][test_idx] = pred_abs

            # Facies modulation
            if P_sand_train is not None:
                w_tr_sand = sub_df_orig["target_IS_SAND"].values[train_idx]
                s_mask    = w_tr_sand == 1;  sh_mask = w_tr_sand == 0
                s_mean    = np.mean(y_abs_orig[train_idx][s_mask])  if np.any(s_mask)  else np.mean(y_abs_orig[train_idx])
                sh_mean   = np.mean(y_abs_orig[train_idx][sh_mask]) if np.any(sh_mask) else np.mean(y_abs_orig[train_idx])
                f_pred    = P_sand_train[test_idx] * s_mean + (1 - P_sand_train[test_idx]) * sh_mean

                for alpha in FACIES_ALPHA_CANDIDATES:
                    mod_name = f"{mname} + FaciesModulation(a={alpha:.2f})"
                    pred_m   = (1.0 - alpha) * pred_abs + alpha * f_pred
                    cv_model_scores[mod_name]["r2"].append(r2_score(y_te_abs, pred_m))
                    cv_model_scores[mod_name]["mae"].append(mean_absolute_error(y_te_abs, pred_m))
                    cv_model_oof_preds[mod_name][test_idx] = pred_m

    # Refit on ALL training data (z-scored + winsorized)
    model_summaries = {}
    refitted_models = {}

    y_all_win = winsorize_per_well(y_zscored, sub_groups, WINSORIZE_IQR_SCALE)
    all_weights = sub_df["tie_quality_weight"].values

    for mname, template in model_templates.items():
        model = clone(template)
        try:    model.fit(X_sel, y_all_win, sample_weight=all_weights)
        except: model.fit(X_sel, y_all_win)
        refitted_models[mname] = model

        cv_r2  = np.mean(cv_model_scores[mname]["r2"])
        cv_mae = np.mean(cv_model_scores[mname]["mae"])
        blind_r2, blind_mae, pred_b_abs = -999.0, 999.0, None

        if not sub_blind.empty:
            bl_feats   = [f for f in active_features if f in sub_blind.columns]
            X_b        = sub_blind[bl_feats].values[:, :len(feature_indices)]
            y_b_real   = sub_blind_orig[target_abs_col].values

            # FIX A (Inference Calibration): robust std & dynamic compaction trend
            all_well_stds = []
            for w_tr_name in np.unique(sub_groups):
                w_mask = sub_groups == w_tr_name
                all_well_stds.append(np.std(y_abs_orig[w_mask]))
            robust_std_all = np.median(all_well_stds)

            # Fit compaction trend on all training wells vs time
            coefs_all = np.polyfit(sub_df_orig["seismic_time_ms"].values, y_abs_orig, 2)
            
            # Evaluate trend baseline at blind well times
            bl_times = sub_blind_orig["seismic_time_ms"].values
            trend_baseline_bl = np.polyval(coefs_all, bl_times)

            pred_b_z   = model.predict(X_b)
            pred_b_abs = pred_b_z * robust_std_all + trend_baseline_bl
            blind_r2   = r2_score(y_b_real, pred_b_abs)
            blind_mae  = mean_absolute_error(y_b_real, pred_b_abs)

        model_summaries[mname] = {
            "cv_r2": cv_r2, "cv_mae": cv_mae,
            "blind_r2": blind_r2, "blind_mae": blind_mae,
            "pred_b_abs": pred_b_abs, "model_obj": model
        }

        if P_sand_train is not None:
            for alpha in FACIES_ALPHA_CANDIDATES:
                mod_name   = f"{mname} + FaciesModulation(a={alpha:.2f})"
                cv_r2_mod  = np.mean(cv_model_scores[mod_name]["r2"])
                cv_mae_mod = np.mean(cv_model_scores[mod_name]["mae"])
                blind_r2_mod, blind_mae_mod, pred_b_abs_mod = -999.0, 999.0, None

                if not sub_blind.empty:
                    s_mask_g  = sub_df_orig["target_IS_SAND"].values == 1
                    sh_mask_g = sub_df_orig["target_IS_SAND"].values == 0
                    s_mean_g  = np.mean(y_abs_orig[s_mask_g])  if np.any(s_mask_g)  else np.mean(y_abs_orig)
                    sh_mean_g = np.mean(y_abs_orig[sh_mask_g]) if np.any(sh_mask_g) else np.mean(y_abs_orig)
                    f_pred_bl      = P_sand_blind * s_mean_g + (1 - P_sand_blind) * sh_mean_g
                    pred_b_abs_mod = (1.0 - alpha) * pred_b_abs + alpha * f_pred_bl
                    blind_r2_mod   = r2_score(y_b_real, pred_b_abs_mod)
                    blind_mae_mod  = mean_absolute_error(y_b_real, pred_b_abs_mod)

                model_summaries[mod_name] = {
                    "cv_r2": cv_r2_mod, "cv_mae": cv_mae_mod,
                    "blind_r2": blind_r2_mod, "blind_mae": blind_mae_mod,
                    "pred_b_abs": pred_b_abs_mod, "model_obj": model
                }

    # Select best model by CV R² ONLY
    best_name = max(model_summaries, key=lambda k: model_summaries[k]["cv_r2"])
    best      = model_summaries[best_name]

    best_alpha = None
    if "FaciesModulation" in best_name:
        try:
            best_alpha = float(best_name.split("a=")[1].rstrip(")"))
        except Exception:
            best_alpha = 0.5

    print(f"  Best for {target_name}{suffix}: {best_name}")
    print(f"    CV R2 (selection): {best['cv_r2']:+.4f} | Blind R2 (reporting): {best['blind_r2']:+.4f}")

    base_model_name  = best_name.split(" + FaciesModulation")[0] if "FaciesModulation" in best_name else best_name
    model_obj_to_save = refitted_models.get(base_model_name, best["model_obj"])

    joblib.dump(feature_indices,    os.path.join(models_dir, f"{target_name}{suffix}_feature_indices.joblib"))
    joblib.dump(model_obj_to_save,  os.path.join(models_dir, f"{target_name}{suffix}_model.joblib"))

    oof_series = pd.Series(cv_model_oof_preds[best_name], index=sub_df.index)
    if not sub_blind.empty:
        print(f"  --> Z-04 Blind: R2 = {best['blind_r2']:+.4f}, MAE = {best['blind_mae']:.4f}")
        return (oof_series, pd.Series(best["pred_b_abs"], index=sub_blind.index),
                best_name, best["cv_r2"], best["cv_mae"],
                best["blind_r2"], best["blind_mae"],
                [f.replace("attr_", "").replace("si_", "SI:") for f in active_features[:5]],
                best_alpha)
    return None


# ==========================================
# DUAL-STRATEGY RACE (Standard vs Cascaded)
# ==========================================
oof_stage1_train = {}
oof_stage1_blind = {}

for target in targets_list:
    print(f"\n--- Target: {target} (V11) ---")

    feats_A = list(REDUCED_FEATURES)

    df_train_work      = df_norm.copy()
    df_train_work_orig = df.copy()
    df_blind_work      = df_blind_norm.copy()
    df_blind_work_orig = df_blind.copy()

    # Cascade prediction logic
    for s1_tgt, oof_vals in oof_stage1_train.items():
        col = f"cascade_pred_{s1_tgt}"
        df_train_work[col] = oof_vals.reindex(df_train_work.index).fillna(oof_vals.mean())
        for w in df_train_work["well_name"].unique():
            mask = df_train_work["well_name"] == w
            m = df_train_work.loc[mask, col].mean()
            s = df_train_work.loc[mask, col].std()
            if s > 1e-10:
                df_train_work.loc[mask, col] = (df_train_work.loc[mask, col] - m) / s
            else:
                df_train_work.loc[mask, col] = 0.0

    for s1_tgt, blind_vals in oof_stage1_blind.items():
        col = f"cascade_pred_{s1_tgt}"
        df_blind_work[col] = blind_vals.reindex(df_blind_work.index).fillna(blind_vals.mean())
        # Z-score cascade using training robust std of stage-1 target
        tr_well_stds = []
        for w_tr_name in df["well_name"].unique():
            w_mask = df["well_name"] == w_tr_name
            tr_well_stds.append(df[w_mask][f"target_{s1_tgt}"].std())
        robust_std_s1 = np.median(tr_well_stds)
        
        # Fit compaction trend on train stage-1 vs time
        coefs_s1 = np.polyfit(df["seismic_time_ms"].values, df[f"target_{s1_tgt}"].values, 2)
        trend_baseline_s1 = np.polyval(coefs_s1, df_blind["seismic_time_ms"].values)
        
        # Normalize the blind cascade pred
        df_blind_work[col] = (df_blind_work[col] - trend_baseline_s1) / max(robust_std_s1, 1e-10)

    cascade_cols = [f"cascade_pred_{s1}" for s1 in stage1_targets if f"cascade_pred_{s1}" in df_train_work.columns]
    feats_B      = feats_A + cascade_cols

    strategies = {"Standard": feats_A}
    if cascade_cols and target in stage2_targets:
        strategies["Cascaded"] = feats_B

    best_result    = None
    best_strategy  = "Standard"
    best_cv_r2     = -9999.0

    for strategy_name, active_feats in strategies.items():
        print(f"  [Strategy {strategy_name}] ({len(active_feats)} features)...")
        res = fit_predict_target(
            target,
            df_train_work, df_train_work_orig,
            df_blind_work, df_blind_work_orig,
            active_feats, sand_only=False
        )
        if res is not None:
            _, _, _, cv_r2, _, _, _, _, _ = res
            if cv_r2 > best_cv_r2:
                best_cv_r2    = cv_r2
                best_result   = res
                best_strategy = strategy_name

    if best_result is not None:
        oof_series, pred_series, mname, cv_r2, cv_mae, blind_r2, blind_mae, sel_feats, best_alpha = best_result
        print(f"  [V11] Winner: Strategy={best_strategy} | CV R2={cv_r2:+.4f} | Blind R2={blind_r2:+.4f}")

        if target in stage1_targets:
            oof_stage1_train[target] = oof_series
            oof_stage1_blind[target] = pred_series

        y_real  = df_blind_work_orig[f"target_{target}"].values
        times   = df_blind_work_orig["seismic_time_ms"].values
        temp_df = pd.DataFrame({
            f"{target} (Act)":  y_real,
            f"{target} (Pred)": pred_series.values,
        }, index=times)
        blind_compare_df = blind_compare_df.join(temp_df, how="outer")

        cv_results.append({
            "target": target,
            "best_model": mname,
            "best_cv_r2": cv_r2,
            "best_cv_mae": cv_mae,
            "blind_r2": blind_r2,
            "blind_mae": blind_mae,
            "winning_strategy": best_strategy,
            "selected_features": sel_feats,
            "is_sand_only": False,
            "tuned_facies_alpha": best_alpha if best_alpha is not None else "N/A",
            "selection_criterion": "cv_r2",
        })

        plt.figure(figsize=(6, 8))
        plt.plot(y_real, times, label="Real Log (LAS)", color="black", linewidth=1.5)
        plt.plot(pred_series.values, times, label=f"V11/{best_strategy}", color="tab:blue", linestyle="--", linewidth=1.5)
        plt.gca().invert_yaxis()
        plt.title(f"True Blind Test on {BLIND_WELL}\nTarget: {target} | V11 | Strategy: {best_strategy}", fontsize=11, weight="bold")
        plt.ylabel("Two-Way Time (ms)", fontsize=10)
        plt.xlabel(f"{target} Value", fontsize=10)
        plt.legend(fontsize=8, loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{target}_blind_well_test.png"), dpi=150)
        try:
            plt.savefig(os.path.join(analysis_dir, "frontend", "public", "well_images", f"{target}_blind_well_test.png"), dpi=150)
        except Exception:
            pass
        plt.close()

    # Sand-only race
    if target in sand_only_targets:
        print(f"  Training sand-only model for {target}...")
        df_sand      = df_train_work[df_train_work_orig["target_IS_SAND"] == 1].copy()
        df_sand_orig = df_train_work_orig[df_train_work_orig["target_IS_SAND"] == 1].copy()
        df_blind_sand      = df_blind_work[df_blind_work_orig["target_IS_SAND"] == 1].copy()
        df_blind_sand_orig = df_blind_work_orig[df_blind_work_orig["target_IS_SAND"] == 1].copy()

        best_sand_result   = None
        best_sand_strategy = "Standard"
        best_sand_cv_r2    = -9999.0

        for strategy_name, active_feats in strategies.items():
            res_s = fit_predict_target(
                target,
                df_sand, df_sand_orig,
                df_blind_sand, df_blind_sand_orig,
                active_feats, sand_only=True
            )
            if res_s is not None:
                _, _, _, cv_r2_s, _, _, _, _, _ = res_s
                if cv_r2_s > best_sand_cv_r2:
                    best_sand_cv_r2    = cv_r2_s
                    best_sand_result   = res_s
                    best_sand_strategy = strategy_name

        if best_sand_result is not None:
            _, _, mname_s, cv_r2_s, cv_mae_s, blind_r2_s, blind_mae_s, sel_feats_s, alpha_s = best_sand_result
            print(f"  [V11] Sand-only winner: '{best_sand_strategy}' | CV R2={cv_r2_s:+.4f} | Blind R2={blind_r2_s:+.4f}")
            cv_results.append({
                "target": target,
                "best_model": mname_s,
                "best_cv_r2": cv_r2_s,
                "best_cv_mae": cv_mae_s,
                "blind_r2": blind_r2_s,
                "blind_mae": blind_mae_s,
                "winning_strategy": best_sand_strategy,
                "selected_features": sel_feats_s,
                "is_sand_only": True,
                "tuned_facies_alpha": alpha_s if alpha_s is not None else "N/A",
                "selection_criterion": "cv_r2",
            })


# ==========================================
# Physics Post-Processing
# ==========================================
print("\n--- Physics Post-Processing ---")
ml_lmrho_blind_r2 = None
for r in cv_results:
    if r["target"] == "LMRHO" and not r["is_sand_only"]:
        ml_lmrho_blind_r2 = r["blind_r2"]
        break

if "AI (Pred)" in blind_compare_df.columns and "DT (Pred)" in blind_compare_df.columns:
    dt_pred   = blind_compare_df["DT (Pred)"].values
    ai_pred   = blind_compare_df["AI (Pred)"].values
    vp_pred   = (1000000.0 / np.clip(dt_pred, 10.0, 300.0)) * 0.3048
    rhob_phys = ai_pred / vp_pred
    blind_compare_df["RHOB (Phys Derived)"] = rhob_phys

    if "RHOB (Act)" in blind_compare_df.columns:
        rhob_act = blind_compare_df["RHOB (Act)"].values
        valid    = ~np.isnan(rhob_act) & ~np.isnan(rhob_phys)
        if np.any(valid):
            rp_r2 = r2_score(rhob_act[valid], rhob_phys[valid])
            print(f"  [PHYSICS] Derived RHOB Z-04 R2: {rp_r2:+.4f}")
            cv_results.append({
                "target": "RHOB", "best_model": "Physics-Derived (AI/Vp)",
                "best_cv_r2": -999.0, "best_cv_mae": 999.0,
                "blind_r2": rp_r2, "blind_mae": mean_absolute_error(rhob_act[valid], rhob_phys[valid]),
                "winning_strategy": "Physics", "selected_features": ["AI", "DT"],
                "is_sand_only": False, "tuned_facies_alpha": "N/A", "selection_criterion": "N/A"
            })

    if "MURHO (Pred)" in blind_compare_df.columns:
        murho_pred = blind_compare_df["MURHO (Pred)"].values
        vp_km_s    = vp_pred / 1000.0
        lmrho_phys = rhob_phys * (vp_km_s**2) - 2.0 * murho_pred
        if "LMRHO (Act)" in blind_compare_df.columns:
            lmrho_act = blind_compare_df["LMRHO (Act)"].values
            valid = ~np.isnan(lmrho_act) & ~np.isnan(lmrho_phys)
            if np.any(valid):
                lp_r2 = r2_score(lmrho_act[valid], lmrho_phys[valid])
                print(f"  [PHYSICS] Derived LMRHO Z-04 R2: {lp_r2:+.4f}")
                if ml_lmrho_blind_r2 is not None:
                    winner = "Physics" if lp_r2 > ml_lmrho_blind_r2 else "ML"
                    print(f"  [LMRHO COMPARE] ML={ml_lmrho_blind_r2:+.4f} vs Physics={lp_r2:+.4f} -> WINNER: {winner}")


# ==========================================
# Save Outputs
# ==========================================
df_cv_summary = pd.DataFrame(cv_results)
df_cv_summary.to_csv(os.path.join(output_dir, "model_performance.csv"), index=False)
print("Saved model performance summary CSV.")

if not blind_compare_df.empty:
    blind_compare_df.to_csv(os.path.join(output_dir, "blind_well_actual_vs_predicted.csv"))
    print("Saved blind comparison CSV.")

# Load V10 results for comparison
v10_path = os.path.join(analysis_dir, "ml_outputs_v10", "model_performance.csv")
v10_df   = pd.read_csv(v10_path) if os.path.exists(v10_path) else None

# Report: V10 vs V11
report_path = os.path.join(output_dir, "ML_REPORT_v11.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# ML Property Prediction Pipeline: Report V11 (Dynamic Calibration)\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("## Key Architectural Changes from V10\n\n")
    f.write("| Fix | Description | Impact |\n| :--- | :--- | :--- |\n")
    f.write("| **A** | Robust Target Standard Deviation: scales predictions using median of training well standard deviations | Prevents Z-07 scale distortion (4.2x) |\n")
    f.write("| **B** | Compaction depth-trend baseline: dynamically shifts z-score predictions using time/depth trend curves | Removes baseline DC offsets without logs |\n")
    f.write("| **C** | Standardized cascade prediction scaling | Removes baseline shifts from intermediate predictions |\n\n")

    f.write("## V10 vs V11 Comparison\n\n")
    f.write(f"Blind Well: **{BLIND_WELL}** (never used for any selection decision)\n\n")
    f.write("| Target | Sand? | V10 CV R2 | V11 CV R2 | Δ CV | V10 Blind R2 | V11 Blind R2 | Δ Blind |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for _, row in df_cv_summary.iterrows():
        tgt   = row["target"]
        sand  = row["is_sand_only"]
        v11cv = row["best_cv_r2"]
        v11bl = row["blind_r2"]

        v10cv, v10bl = None, None
        if v10_df is not None:
            match = v10_df[(v10_df["target"] == tgt) & (v10_df.get("is_sand_only", False) == sand)]
            if not match.empty:
                v10cv = match.iloc[0].get("best_cv_r2", None)
                v10bl = match.iloc[0].get("blind_r2", None)

        v10cv_str  = f"{v10cv:+.4f}" if v10cv is not None else "N/A"
        v10bl_str  = f"{v10bl:+.4f}" if v10bl is not None else "N/A"
        dcv_str   = f"{(v11cv - v10cv):+.4f}" if v10cv is not None else "N/A"
        dbl_str   = f"{(v11bl - v10bl):+.4f}" if v10bl is not None else "N/A"

        f.write(f"| **{tgt}** | {'YES' if sand else 'NO'} | {v10cv_str} | {v11cv:+.4f} | {dcv_str} | {v10bl_str} | {v11bl:+.4f} | {dbl_str} |\n")

    f.write("\n## Full V11 Performance Table\n\n")
    f.write("| Target | Sand? | Strategy | Best Model | CV R2 | Blind R2 | Facies α |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for _, row in df_cv_summary.iterrows():
        f.write(
            f"| **{row['target']}** | {'YES' if row['is_sand_only'] else 'NO'} "
            f"| {row.get('winning_strategy','N/A')} | {row['best_model']} "
            f"| {row['best_cv_r2']:+.4f} | {row['blind_r2']:+.4f} "
            f"| {row.get('tuned_facies_alpha','N/A')} |\n"
        )

print("\nV11 pipeline complete!")
print(f"Report: {report_path}")
