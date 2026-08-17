import os
import re
import json
import numpy as np
import pandas as pd
import segyio
import lasio
import joblib
from scipy.signal import hilbert

# ==========================================
# USER CONFIGURATION: Set your output directory name here
# ==========================================
OUTPUT_FOLDER_NAME = "ml_outputs_v11"

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
segy_path = os.path.join(analysis_dir, "segy", "origional.segy")
las_dir = os.path.join(analysis_dir, "las_cleaned")  # Use cleaned LAS files
tie_summary_path = os.path.join(analysis_dir, "well_seismic", "output", "impedance_tie_summary.csv")
models_dir = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME, "models")
facies_dir = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME, "facies_model")
trends_json_path = os.path.join(analysis_dir, "training_data", "output", "log_trends.json")
output_js_path = os.path.join(analysis_dir, "frontend", "src", "data.js")

FEET_TO_METERS = 0.3048
MAX_ENV_RATIO = 50.0

# ── Checkshot TDS calibration data (Zamzama-TDS.xlsx) ─────────────────────
# Each entry: list of (TWT_ms, Depth_m) picks — physically measured ground truth
# Applied as piecewise-linear drift correction to the sonic DPTM log.
CHECKSHOT_TDS = {
    'Z-02':       [(2045, 3354.80), (2101, 3409.01), (2172, 3630.71)],
    'Z-03':       [(2200, 3651.50), (2242, 3700.77), (2336, 3916.83)],
    'Z-04':       [(2012, 3343.10), (2071, 3393.90), (2136, 3610.56)],
    'Z-05':       [(2125, 3488.55), (2155, 3541.41), (2247, 3757.67)],
    'Z-06':       [(2081, 3425.27), (2114, 3479.12), (2207, 3701.44)],
    'Z-07':       [(2066, 3399.30), (2109, 3452.90), (2199, 3677.34)],
    'Z-08-ST-02': [(2076, 3460.55), (2104, 3506.91)],
}

def apply_checkshot_correction(depth_arr, dptm_arr, well_name):
    """Apply piecewise-linear drift correction using checkshot TDS control points.
    
    The checkshot picks define the true TWT at known measured depths.
    We compute the drift (CS_TWT - DPTM) at each control depth and
    interpolate the correction across the full log depth range.
    """
    if well_name not in CHECKSHOT_TDS:
        return dptm_arr.copy()  # No checkshots available — return unchanged
    
    cs_points = CHECKSHOT_TDS[well_name]
    cs_depths = np.array([pt[1] for pt in cs_points])
    cs_twts   = np.array([pt[0] for pt in cs_points])
    
    # Compute DPTM values at checkshot control depths (via interpolation)
    valid = ~np.isnan(dptm_arr)
    if valid.sum() < 2:
        return dptm_arr.copy()
    
    dptm_at_cs = np.interp(cs_depths, depth_arr[valid], dptm_arr[valid])
    drifts = cs_twts - dptm_at_cs  # positive drift = DPTM runs too slow (too-late)
    
    # Interpolate drift correction across the full depth range
    correction = np.interp(depth_arr, cs_depths, drifts,
                           left=drifts[0], right=drifts[-1])
    
    dptm_cs = dptm_arr.copy()
    dptm_cs[valid] = dptm_arr[valid] + correction[valid]
    return dptm_cs

WELLS = {
    "Z-02": {"x": 1205859.09, "y": 9692966.31, "kb": 146.46},
    "Z-03": {"x": 1201178.25, "y": 9682452.00, "kb": 147.64},
    "Z-04": {"x": 1205820.18, "y": 9696292.65, "kb": 147.64},
    "Z-05": {"x": 1206404.17, "y": 9679510.83, "kb": 144.36},
    "Z-06": {"x": 1207337.37, "y": 9684145.64, "kb": 146.98},
    "Z-07": {"x": 1206364.34, "y": 9688320.18, "kb": 147.97},
    "Z-08-ST-02": {"x": 1202225.66, "y": 9700693.11, "kb": 152.20}
}

target_curves = ["GR", "DT", "RHOB", "VSH", "PHIE", "SWE", "PHIT", "AI", "VPVS", "POIS", "LMRHO", "MURHO"]
sand_only_targets = ["PHIE", "SWE", "VPVS", "POIS", "LMRHO", "MURHO"]

def _pad_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if shift == 0:
        return arr.copy()
    if shift > 0:
        out = np.zeros_like(arr)
        out[:-shift] = arr[shift:]
        return out
    shift = -shift
    out = np.zeros_like(arr)
    out[shift:] = arr[:-shift]
    return out

def shift_trace(arr: np.ndarray, shift_ms: float, dt_ms: float) -> np.ndarray:
    shift_samples = int(round(shift_ms / dt_ms))
    if shift_samples == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if shift_samples > 0:
        out[shift_samples:] = arr[:-shift_samples]
    else:
        out[:shift_samples] = arr[-shift_samples:]
    return out

def _rolling_stats(arr: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    half = window // 2
    maxv = np.zeros(n)
    meanv = np.zeros(n)
    minv = np.zeros(n)
    stdv = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = arr[lo:hi]
        maxv[i] = float(np.max(window_vals))
        meanv[i] = float(np.mean(window_vals))
        minv[i] = float(np.min(window_vals))
        stdv[i] = float(np.std(window_vals))
    return maxv, meanv, minv, stdv

def bandpass_envelope(trace, low_hz, high_hz, dt_ms):
    n = len(trace)
    freqs = np.fft.fftfreq(n, d=dt_ms/1000.0)
    f_trace = np.fft.fft(trace)
    
    f_filtered = f_trace.copy()
    f_filtered[np.abs(freqs) < low_hz] = 0
    f_filtered[np.abs(freqs) > high_hz] = 0
    
    filtered_trace = np.fft.ifft(f_filtered)
    env = np.abs(hilbert(np.real(filtered_trace)))
    return np.nan_to_num(env, nan=0.0)

def compute_trace_attributes(trace, times_ms, dt_ms, global_ai_min, global_ai_max, inv_scale=None, inv_shift=None) -> dict[str, np.ndarray]:
    trace = np.asarray(trace, dtype=float)
    n_samples = len(trace)
    amp = np.nan_to_num(trace, nan=0.0)
    analytic = hilbert(amp)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dt_s = dt_ms / 1000.0
    ifreq = np.gradient(phase, dt_s) / (2.0 * np.pi)
    ifreq = np.nan_to_num(ifreq, nan=0.0)
    rel_pos = np.linspace(0.0, 1.0, n_samples)
    env_deriv = np.nan_to_num(np.gradient(envelope), nan=0.0)
    
    cumsum_ai = np.cumsum(np.clip(amp, -1.0, 1.0))
    cumsum_ai = np.nan_to_num(cumsum_ai, nan=0.0)
    impedance = (cumsum_ai - global_ai_min) / (global_ai_max - global_ai_min + 1e-8)

    # Calibrated absolute impedance (matches training feature)
    if inv_scale is not None and inv_shift is not None:
        calibrated_ai = inv_scale * cumsum_ai + inv_shift
    else:
        calibrated_ai = impedance
    calibrated_ai = np.nan_to_num(calibrated_ai, nan=0.0)

    max_abs_env = np.max(envelope)
    eps = 1e-6 * max_abs_env
    env_ratio = np.divide(envelope, np.abs(amp) + eps)
    env_ratio = np.clip(env_ratio, 0.0, MAX_ENV_RATIO)

    # Sweetness = envelope / sqrt(max(ifreq, 0.1))
    sweetness = envelope / np.sqrt(np.maximum(ifreq, 0.1))

    # Phase 2a: Spectral decomposition envelopes at 5 bands
    spec_amp_10hz = bandpass_envelope(amp, 8, 12, dt_ms)
    spec_amp_15hz = bandpass_envelope(amp, 13, 17, dt_ms)
    spec_amp_20hz = bandpass_envelope(amp, 18, 22, dt_ms)
    spec_amp_30hz = bandpass_envelope(amp, 27, 33, dt_ms)
    spec_amp_40hz = bandpass_envelope(amp, 37, 43, dt_ms)

    attrs = {
        "acoustic_impedance": impedance,
        "calibrated_ai": calibrated_ai,
        "amp_center": amp,
        "amp_shift_+1": _pad_shift(amp, 1),
        "amp_shift_+2": _pad_shift(amp, 2),
        "amp_shift_-1": _pad_shift(amp, -1),
        "amp_shift_-2": _pad_shift(amp, -2),
        "env_center": envelope,
        "env_deriv": env_deriv,
        "env_ratio": env_ratio,
        "env_shift_+1": _pad_shift(envelope, 1),
        "env_shift_+2": _pad_shift(envelope, 2),
        "env_shift_-1": _pad_shift(envelope, -1),
        "env_shift_-2": _pad_shift(envelope, -2),
        "ifreq_center": ifreq,
        "ifreq_shift_+1": _pad_shift(ifreq, 1),
        "ifreq_shift_+2": _pad_shift(ifreq, 2),
        "ifreq_shift_-1": _pad_shift(ifreq, -1),
        "ifreq_shift_-2": _pad_shift(ifreq, -2),
        "sweetness": sweetness,
        "sweetness_shift_+1": _pad_shift(sweetness, 1),
        "sweetness_shift_+2": _pad_shift(sweetness, 2),
        "sweetness_shift_-1": _pad_shift(sweetness, -1),
        "sweetness_shift_-2": _pad_shift(sweetness, -2),
        "polarity_index": np.clip(amp / (envelope + eps), -1.0, 1.0),
        "polarity_index_shift_+1": _pad_shift(np.clip(amp / (envelope + eps), -1.0, 1.0), 1),
        "polarity_index_shift_+2": _pad_shift(np.clip(amp / (envelope + eps), -1.0, 1.0), 2),
        "polarity_index_shift_-1": _pad_shift(np.clip(amp / (envelope + eps), -1.0, 1.0), -1),
        "polarity_index_shift_-2": _pad_shift(np.clip(amp / (envelope + eps), -1.0, 1.0), -2),
        "rel_pos": rel_pos,
        "spec_amp_10hz": spec_amp_10hz,
        "spec_amp_15hz": spec_amp_15hz,
        "spec_amp_20hz": spec_amp_20hz,
        "spec_amp_30hz": spec_amp_30hz,
        "spec_amp_40hz": spec_amp_40hz,
    }
    win_max, win_mean, win_min, win_std = _rolling_stats(amp, window=5)
    
    attrs.update({
        "win_max": win_max,
        "win_mean": win_mean,
        "win_min": win_min,
        "win_std": win_std,
    })
    return {f"attr_{k}": v for k, v in attrs.items()}

def ricker_wavelet(freq_hz, length_ms, dt_ms):
    t = np.arange(-length_ms / 2, length_ms / 2 + dt_ms, dt_ms) / 1000.0
    y = (1 - 2 * (np.pi * freq_hz * t) ** 2) * np.exp(-(np.pi * freq_hz * t) ** 2)
    return y

def rotate_phase(wavelet, phase_deg):
    analytic = hilbert(wavelet)
    phase_rad = np.deg2rad(phase_deg)
    rotated = np.real(analytic * np.exp(1j * phase_rad))
    return rotated

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

V11_ALPHAS = {
    "VSH": 0.5,
    "PHIE": 1.0,
    "SWE": 0.5,
}

def main():
    print("Generating comprehensive frontend dataset...")
    df_ties = pd.read_csv(tie_summary_path)
    
    # Load trends
    with open(trends_json_path, "r") as f:
        log_trends = json.load(f)

    # Load performance summary to check for facies-modulated models
    perf_path = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME, "model_performance.csv")
    perf_df = pd.read_csv(perf_path) if os.path.exists(perf_path) else None
    
    def check_modulation(tgt):
        if perf_df is not None:
            row = perf_df[(perf_df["target"] == tgt) & (perf_df["is_sand_only"] == False)]
            if not row.empty:
                return "+ FaciesModulation" in str(row.iloc[0]["best_model"])
        return False

    # Load ML models and feature indices (V11)
    pipelines = {}
    for target in target_curves:
        feature_indices = joblib.load(os.path.join(models_dir, f"{target}_feature_indices.joblib"))
        model = joblib.load(os.path.join(models_dir, f"{target}_model.joblib"))
        
        pipelines[target] = {
            "indices": feature_indices,
            "model": model,
        }
        
        if target in sand_only_targets:
            sand_model = joblib.load(os.path.join(models_dir, f"{target}_sand_model.joblib"))
            sand_feature_indices = joblib.load(os.path.join(models_dir, f"{target}_sand_feature_indices.joblib"))
            
            pipelines[target + "_sand"] = {
                "indices": sand_feature_indices,
                "model": sand_model,
            }

    # Load facies classifier model
    facies_model_path = os.path.join(facies_dir, "facies_model.joblib")
    if not os.path.exists(facies_model_path):
        print("Facies classifier model not found — training automatically via train_facies_model.py...")
        from ml_training.train_facies_model import main as train_facies_main
        train_facies_main()

    print("Loading Facies classifier model...")
    facies_model = joblib.load(facies_model_path)
    facies_scaler = joblib.load(os.path.join(facies_dir, "facies_scaler.joblib"))
    facies_features = joblib.load(os.path.join(facies_dir, "facies_features.joblib"))
    
    # Define attribute order to match ML model features
    df_train_tbl = pd.read_csv(os.path.join(analysis_dir, "training_data", "output", "training_table.csv"))
    attr_cols = sorted([c for c in df_train_tbl.columns if c.startswith("attr_")])
    attr_cols = [c for c in attr_cols if not any(x in c for x in ["+3", "+4", "+5", "-3", "-4", "-5"])]
    
    # Get global acoustic impedance normalization bounds
    global_ai_min = -11.0
    global_ai_max = 14.0

    wells_out = {}

    with segyio.open(segy_path, "r", ignore_geometry=True) as f_segy:
        seis_times = f_segy.samples.astype(float)
        dt_ms = float(seis_times[1] - seis_times[0])
        inlines = f_segy.attributes(segyio.TraceField.FieldRecord)[:]
        crosslines = f_segy.attributes(segyio.TraceField.TraceNumber)[:]

        for _, row in df_ties.iterrows():
            well_name = row["well"]
            shift_ms = float(row["std_shift"])       # refined shift from impedance tie
            polarity_flag = row.get("polarity", "Normal")
            polarity = -1.0 if polarity_flag == "Reversed" else 1.0
            inv_scale = float(row.get("inv_scale", 1.0))
            inv_shift_val = float(row.get("inv_shift", 0.0))
            mapped_inline = int(row["inline"])
            mapped_crossline = int(row["crossline"])
            freq_hz = float(row["std_freq"])
            phase_deg = float(row["std_phase"])

            trace_idx = np.where((inlines == mapped_inline) & (crosslines == mapped_crossline))[0]
            if len(trace_idx) == 0:
                continue
            trace_idx = int(trace_idx[0])

            # Get raw seismic trace and apply polarity correction
            raw_trace = f_segy.trace[trace_idx].copy()
            trace = raw_trace * polarity

            # Build trace attributes along the entire trace (with calibrated AI)
            attrs_dict = compute_trace_attributes(trace, seis_times, dt_ms, global_ai_min, global_ai_max,
                                                  inv_scale=inv_scale, inv_shift=inv_shift_val)
            
            # Fetch neighboring traces
            neighbors_cfg = [
                ("il_p1", mapped_inline + 1, mapped_crossline),
                ("il_m1", mapped_inline - 1, mapped_crossline),
                ("xl_p1", mapped_inline, mapped_crossline + 1),
                ("xl_m1", mapped_inline, mapped_crossline - 1)
            ]
            
            for tag, il_n, xl_n in neighbors_cfg:
                n_trace_idx = np.where((inlines == il_n) & (crosslines == xl_n))[0]
                if len(n_trace_idx) > 0:
                    n_raw_trace = f_segy.trace[int(n_trace_idx[0])].copy()
                    n_trace = n_raw_trace * polarity
                    n_attrs = compute_trace_attributes(n_trace, seis_times, dt_ms, global_ai_min, global_ai_max,
                                                       inv_scale=inv_scale, inv_shift=inv_shift_val)
                    for att_key in ["amp_center", "env_center", "acoustic_impedance", "ifreq_center"]:
                        attrs_dict[f"attr_n_{tag}_{att_key}_diff"] = n_attrs[f"attr_{att_key}"] - attrs_dict[f"attr_{att_key}"]
                else:
                    for att_key in ["amp_center", "env_center", "acoustic_impedance", "ifreq_center"]:
                        attrs_dict[f"attr_n_{tag}_{att_key}_diff"] = np.zeros_like(trace)

            # Form raw feature matrix X for Facies classifier (uses original raw features)
            X_facies_raw = np.column_stack([attrs_dict[c] for c in attr_cols])
            X_facies_scaled = facies_scaler.transform(X_facies_raw)
            is_sand_prob = facies_model.predict_proba(X_facies_scaled)[:, 1]
            is_sand_flag = np.where(is_sand_prob > 0.5, 1, 0)

            # Pre-compute predictions for all targets in V11 style
            preds = {}
            stage1_targets = ["AI", "DT", "MURHO", "PHIT", "POIS", "VPVS"]
            
            # Form clean well dataframe for scale-invariant features & per-well feature z-scoring
            df_well = pd.DataFrame(attrs_dict)
            df_well = engineer_si_features(df_well)
            
            # Normalise features per-well (using current well's own mean and standard deviation)
            for c in REDUCED_FEATURES:
                m = df_well[c].mean()
                s = df_well[c].std()
                if s > 1e-10:
                    df_well[c] = (df_well[c] - m) / s
                else:
                    df_well[c] = 0.0
            
            X_current = df_well[REDUCED_FEATURES].values
            
            # Predict in targets_list order
            target_order = ["AI", "DT", "MURHO", "PHIT", "POIS", "VPVS", "GR", "RHOB", "VSH", "PHIE", "SWE", "LMRHO"]
            
            for target in target_order:
                pipe = pipelines[target]
                
                # Predict standard model z-score
                X_sel = X_current[:, pipe["indices"]]
                pred_z = pipe["model"].predict(X_sel)
                
                # Calculate training-set robust stats and trend baseline dynamically (LOGO style)
                df_tr = df_train_tbl[df_train_tbl["well_name"] != well_name]
                
                # 1. Fit compaction trend (degree-2 polynomial) of training wells vs time
                df_tr_clean = df_tr.dropna(subset=[f"target_{target}"])
                coefs = np.polyfit(df_tr_clean["seismic_time_ms"].values, df_tr_clean[f"target_{target}"].values, 2)
                trend_vals = np.polyval(coefs, seis_times)
                
                # 2. Get training robust standard deviation
                tr_well_stds = []
                for w_tr_name in df_tr["well_name"].unique():
                    y_w = df_tr[df_tr["well_name"] == w_tr_name][f"target_{target}"].values
                    y_w = y_w[~np.isnan(y_w)]
                    if len(y_w) > 5:
                        tr_well_stds.append(np.std(y_w))
                robust_std = np.median(tr_well_stds) if tr_well_stds else 1.0
                
                # 3. Dynamic Back-Conversion
                pred_abs = pred_z * robust_std + trend_vals
                
                # Apply Facies Modulation if best model has it (using V11_ALPHAS parameters)
                alpha = V11_ALPHAS.get(target, 0.0)
                if alpha > 0.0:
                    s_mask = df_tr_clean["target_IS_SAND"] == 1
                    sh_mask = df_tr_clean["target_IS_SAND"] == 0
                    
                    s_mean = df_tr_clean[s_mask][f"target_{target}"].mean()
                    sh_mean = df_tr_clean[sh_mask][f"target_{target}"].mean()
                    if np.isnan(s_mean): s_mean = df_tr_clean[f"target_{target}"].mean()
                    if np.isnan(sh_mean): sh_mean = df_tr_clean[f"target_{target}"].mean()
                    
                    facies_pred = is_sand_prob * s_mean + (1.0 - is_sand_prob) * sh_mean
                    pred_abs = (1.0 - alpha) * pred_abs + alpha * facies_pred
                
                preds[target] = pred_abs
                
                # If this target is in Stage 1, z-score it per-well and append it to X_current
                if target in stage1_targets:
                    p_mean = pred_abs.mean()
                    p_std = pred_abs.std()
                    pred_zscore = (pred_abs - p_mean) / p_std if p_std > 1e-10 else np.zeros_like(pred_abs)
                    X_current = np.column_stack([X_current, pred_zscore])
                
                # Apply sand-only standard logic if needed
                if target in sand_only_targets and (target + "_sand") in pipelines:
                    pipe_sand = pipelines[target + "_sand"]
                    X_sel_s = X_current[:, pipe_sand["indices"]]
                    pred_z_sand = pipe_sand["model"].predict(X_sel_s)
                    
                    # Back-convert sand model prediction
                    pred_abs_sand = pred_z_sand * robust_std + trend_vals
                    
                    if target == "PHIE":
                        baseline_val = 0.0
                    elif target == "SWE":
                        baseline_val = 1.0
                    else:
                        baseline_val = trend_vals
                        
                    is_sand_mask = is_sand_flag == 1
                    
                    combined_pred_abs = np.zeros_like(pred_abs)
                    if target in ["PHIE", "SWE"]:
                        combined_pred_abs.fill(baseline_val)
                        combined_pred_abs[is_sand_mask] = pred_abs_sand[is_sand_mask]
                    else:
                        combined_pred_abs = trend_vals.copy()
                        combined_pred_abs[is_sand_mask] = pred_abs_sand[is_sand_mask]
                    
                    # Update preds[target] with the sand-combined version
                    preds[target] = combined_pred_abs

            # Post-processed physical derived curves (addon)
            vp_pred = (1000000.0 / np.clip(preds["DT"], 10.0, 300.0)) * 0.3048
            rhob_phys = preds["AI"] / vp_pred
            preds["RHOB_phys"] = rhob_phys
            
            vp_km_s = vp_pred / 1000.0
            preds["LMRHO_phys"] = rhob_phys * (vp_km_s**2) - 2.0 * preds["MURHO"]

            # Load actual logs from LAS file (using cleaned LAS)
            las_path = os.path.join(las_dir, f"{well_name}.las")
            las = lasio.read(las_path)
            df_las = las.df()

            # Step 2: Apply checkshot drift correction to DPTM before time mapping
            dptm_ms = df_las["DPTM"].values.astype(float) if "DPTM" in df_las.columns else None
            well_times_mapped = dptm_ms + shift_ms if dptm_ms is not None else None

            # Calculate raw unshifted reflectivity curve
            refl_reg_raw = np.zeros(len(seis_times))
            if "DT" in df_las.columns and "RHOB" in df_las.columns and dptm_ms is not None:
                vp_ms = (1.0e6 / df_las["DT"].values.astype(float)) * 0.3048
                impedance_log = vp_ms * df_las["RHOB"].values.astype(float)
                refl = (impedance_log[1:] - impedance_log[:-1]) / (impedance_log[1:] + impedance_log[:-1])
                refl_time_raw = (dptm_ms[1:] + dptm_ms[:-1]) / 2.0
                
                bin_idx = np.clip(((refl_time_raw - seis_times[0]) / dt_ms).round().astype(int), 0, len(seis_times) - 1)
                counts = np.zeros(len(seis_times))
                for b, r in zip(bin_idx, refl):
                    refl_reg_raw[b] += r
                    counts[b] += 1
                counts[counts == 0] = 1
                refl_reg_raw = refl_reg_raw / counts
                
            refl_shifted = shift_trace(refl_reg_raw, shift_ms, dt_ms)
            wav = ricker_wavelet(freq_hz, 100, dt_ms)
            wav_rotated = rotate_phase(wav, phase_deg)
            synthetic = np.convolve(refl_shifted, wav_rotated, mode="same")
            
            rms_seis = np.std(trace)
            rms_syn = np.std(synthetic)
            scale_val = rms_syn / rms_seis if rms_seis != 0 else 1.0
            synthetic = synthetic / scale_val if scale_val != 0 else synthetic

            well_samples = []
            for i, t in enumerate(seis_times):
                sample_data = {
                    "time": float(t),
                    "seismic_amp": float(trace[i]),
                    "synthetic_amp": float(synthetic[i]),
                    "reflectivity": float(refl_reg_raw[i]),
                    "is_sand": int(is_sand_flag[i]),
                    "sand_prob": round(float(is_sand_prob[i]), 4)
                }
                
                if well_times_mapped is not None:
                    idx_in_bin = np.where((well_times_mapped >= t - dt_ms/2) & (well_times_mapped < t + dt_ms/2))[0]
                    if len(idx_in_bin) > 0:
                        for target in target_curves:
                            col_name = "PHIE" if target == "PHIE" else target
                            if col_name in df_las.columns:
                                val = float(np.mean(df_las[col_name].values[idx_in_bin]))
                                sample_data[f"{target} (Act)"] = None if np.isnan(val) else val
                    else:
                        for target in target_curves:
                            sample_data[f"{target} (Act)"] = None

                # ML Predictions
                for target in target_curves:
                    sample_data[f"{target} (Pred)"] = float(preds[target][i])
                    sample_data[f"{target} (Pred Raw)"] = float(preds[target][i]) # both identical
                
                # Physics Derived Predictions Addon
                sample_data["RHOB (Phys Derived)"] = float(preds["RHOB_phys"][i])
                sample_data["LMRHO (Phys Derived)"] = float(preds["LMRHO_phys"][i])
                
                well_samples.append(sample_data)

            # Keep the complete seismic trace range (from start to bottom of the well) for the geologist
            pass

            wells_out[well_name] = {
                "name": well_name,
                "x": WELLS[well_name]["x"],
                "y": WELLS[well_name]["y"],
                "kb": WELLS[well_name]["kb"],
                "inline": mapped_inline,
                "crossline": mapped_crossline,
                "tie": {
                    "correlation": float(row["std_corr"]),
                    "shift_ms": float(shift_ms),
                    "freq_hz": float(freq_hz),
                    "phase_deg": float(phase_deg),
                    "polarity": str(polarity_flag),
                    "wf_correlation": float(row.get("wf_corr", row["std_corr"]))
                },
                "samples": well_samples
            }

    # Write JS file
    output_js_content = f"""// Automatic generated database for LMKR Geologist Dashboard
export const blindWellName = "Z-04";

export const wellsConfig = {json.dumps(WELLS, indent=2)};

export const wellsData = {json.dumps(wells_out, indent=2)};
"""

    with open(output_js_path, "w", encoding="utf-8") as f:
        f.write(output_js_content)
    
    print(f"[OK] Successfully generated {output_js_path} for all 6 wells!")

if __name__ == "__main__":
    main()
