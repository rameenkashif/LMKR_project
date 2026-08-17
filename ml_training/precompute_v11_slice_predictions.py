import os, sys, json, time
import numpy as np
import segyio
import joblib
import pandas as pd
from scipy.signal import hilbert

OUTPUT_FOLDER_NAME = "ml_outputs_v11"

script_dir   = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
segy_path    = os.path.join(analysis_dir, "segy", "origional.segy")
public_dir   = os.path.join(analysis_dir, "frontend", "public")
models_dir   = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME, "models")
trends_json_path = os.path.join(analysis_dir, "training_data", "output", "log_trends.json")
training_csv     = os.path.join(analysis_dir, "training_data", "output", "training_table.csv")

os.makedirs(public_dir, exist_ok=True)

i_min, i_max = 382, 626
j_min, j_max = 46, 297
I_len = i_max - i_min + 1  # 245
J_len = j_max - j_min + 1  # 252
K_len = 313
tStart = 2086.0
dt_ms = 2.0
MAX_ENV_RATIO = 50.0

stage1_targets = ["AI", "DT", "MURHO", "PHIT", "POIS", "VPVS"]
stage2_targets = ["GR", "RHOB", "VSH", "PHIE", "SWE", "LMRHO"]
target_order   = stage1_targets + stage2_targets

print(f"Loading V11 ML models from {models_dir} ...")
pipelines = {}
for target in target_order:
    m_path = os.path.join(models_dir, f"{target}_model.joblib")
    i_path = os.path.join(models_dir, f"{target}_feature_indices.joblib")
    if os.path.exists(m_path):
        model = joblib.load(m_path)
        indices = joblib.load(i_path) if os.path.exists(i_path) else None
        if hasattr(model, "n_jobs"):
            model.n_jobs = -1
        pipelines[target] = {"model": model, "indices": indices}
        print(f"  [OK] Loaded {target}")

with open(trends_json_path, "r") as f:
    log_trends = json.load(f)

df_train_tbl = pd.read_csv(training_csv)

# Compute training-set robust standard deviations and trend polyfits for each target upfront
target_robust_stds = {}
target_trend_coefs = {}

for target in target_order:
    df_tr_clean = df_train_tbl.dropna(subset=[f"target_{target}"])
    coefs = np.polyfit(df_tr_clean["seismic_time_ms"].values, df_tr_clean[f"target_{target}"].values, 2)
    target_trend_coefs[target] = coefs
    
    tr_well_stds = []
    for w_name in df_train_tbl["well_name"].unique():
        y_w = df_train_tbl[df_train_tbl["well_name"] == w_name][f"target_{target}"].dropna().values
        if len(y_w) > 5:
            tr_well_stds.append(np.std(y_w))
    target_robust_stds[target] = float(np.median(tr_well_stds)) if tr_well_stds else 1.0

tie_summary_path = os.path.join(analysis_dir, "well_seismic", "output", "impedance_tie_summary.csv")
global_inv_scale = 0.075
global_inv_shift = 12000.0
if os.path.exists(tie_summary_path):
    df_ties = pd.read_csv(tie_summary_path)
    if "inv_scale" in df_ties.columns and "inv_shift" in df_ties.columns:
        global_inv_scale = float(df_ties["inv_scale"].mean())
        global_inv_shift = float(df_ties["inv_shift"].mean())

def pad_shift_2d(arr, shift):
    if shift == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if shift > 0:
        out[:, :-shift] = arr[:, shift:]
    else:
        s = -shift
        out[:, s:] = arr[:, :-s]
    return out

def bandpass_envelope_2d(amp_2d, low_hz, high_hz, dt_ms):
    n_traces, n_samples = amp_2d.shape
    freqs = np.fft.fftfreq(n_samples, d=dt_ms/1000.0)
    f_trace = np.fft.fft(amp_2d, axis=1)
    
    f_filtered = f_trace.copy()
    f_filtered[:, np.abs(freqs) < low_hz] = 0
    f_filtered[:, np.abs(freqs) > high_hz] = 0
    
    filtered_trace = np.fft.ifft(f_filtered, axis=1)
    env = np.abs(hilbert(np.real(filtered_trace), axis=1))
    return np.nan_to_num(env, nan=0.0)

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

print("\nReading SEG-Y volume into 3D array (I_len, J_len, K_len)...")
volume_raw = np.zeros((I_len, J_len, K_len), dtype=np.float32)
with segyio.open(segy_path, "r", ignore_geometry=True) as f_segy:
    inlines_all = f_segy.attributes(segyio.TraceField.FieldRecord)[:]
    xlines_all = f_segy.attributes(segyio.TraceField.TraceNumber)[:]
    n_traces_total = len(inlines_all)
    seis_times = f_segy.samples.astype(float)
    
    for k in range(n_traces_total):
        il = inlines_all[k]
        xl = xlines_all[k]
        if i_min <= il <= i_max and j_min <= xl <= j_max:
            volume_raw[il - i_min, xl - j_min, :] = f_segy.trace[k]

print(f"Volume loaded. Shape: {volume_raw.shape}")

# Precompute global AI bounds
sample_traces = volume_raw.reshape(-1, K_len)[::100]
ai_vals = np.cumsum(np.clip(sample_traces, -1.0, 1.0), axis=1).flatten()
global_ai_min = float(np.percentile(ai_vals, 1))
global_ai_max = float(np.percentile(ai_vals, 99))

predictions_3d = {target: np.zeros((I_len, J_len, K_len), dtype=np.float32) for target in target_order}

print("\nExecuting authentic V11 ML predictions inline-by-inline ...")
t0_all = time.time()

for i in range(I_len):
    amp = np.nan_to_num(volume_raw[i, :, :], nan=0.0)
    analytic = hilbert(amp, axis=1)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=1)
    dt_s = dt_ms / 1000.0
    ifreq = np.gradient(phase, dt_s, axis=1) / (2.0 * np.pi)
    ifreq = np.nan_to_num(ifreq, nan=0.0)
    env_deriv = np.nan_to_num(np.gradient(envelope, axis=1), nan=0.0)

    rel_pos = np.linspace(0.0, 1.0, K_len)
    rel_pos_2d = np.tile(rel_pos, (J_len, 1))

    cumsum_ai = np.cumsum(np.clip(amp, -1.0, 1.0), axis=1)
    cumsum_ai = np.nan_to_num(cumsum_ai, nan=0.0)
    impedance = (cumsum_ai - global_ai_min) / (global_ai_max - global_ai_min + 1e-8)
    calibrated_ai = global_inv_scale * cumsum_ai + global_inv_shift
    calibrated_ai = np.nan_to_num(calibrated_ai, nan=0.0)

    max_abs_env = np.max(envelope, axis=1, keepdims=True)
    eps = 1e-6 * max_abs_env
    env_ratio = np.clip(envelope / (np.abs(amp) + eps), 0.0, MAX_ENV_RATIO)
    sweetness = envelope / np.sqrt(np.maximum(ifreq, 0.1))

    s0 = amp
    s1 = pad_shift_2d(amp, 1)
    s2 = pad_shift_2d(amp, 2)
    s3 = pad_shift_2d(amp, -1)
    s4 = pad_shift_2d(amp, -2)
    win_max = np.maximum.reduce([s0, s1, s2, s3, s4])
    win_min = np.minimum.reduce([s0, s1, s2, s3, s4])
    win_mean = (s0 + s1 + s2 + s3 + s4) / 5.0
    win_std = np.sqrt(((s0 - win_mean)**2 + (s1 - win_mean)**2 + (s2 - win_mean)**2 + (s3 - win_mean)**2 + (s4 - win_mean)**2) / 5.0)

    spec_amp_10hz = bandpass_envelope_2d(amp, 8, 12, dt_ms)
    spec_amp_15hz = bandpass_envelope_2d(amp, 13, 17, dt_ms)
    spec_amp_20hz = bandpass_envelope_2d(amp, 18, 22, dt_ms)
    spec_amp_30hz = bandpass_envelope_2d(amp, 27, 33, dt_ms)
    spec_amp_40hz = bandpass_envelope_2d(amp, 37, 43, dt_ms)

    total_spec = spec_amp_10hz + spec_amp_15hz + spec_amp_20hz + spec_amp_30hz + spec_amp_40hz + 1e-8
    si_spec_frac_10 = spec_amp_10hz / total_spec
    si_spec_frac_40 = spec_amp_40hz / total_spec

    # XL diffs
    grad_p = np.zeros_like(amp)
    grad_p[:-1, :] = amp[1:, :] - amp[:-1, :]
    si_norm_grad_il = grad_p / (envelope + 1e-8)

    grad_ai = np.zeros_like(calibrated_ai)
    grad_ai[:-1, :] = calibrated_ai[1:, :] - calibrated_ai[:-1, :]

    grad_ifreq = np.zeros_like(ifreq)
    grad_ifreq[:-1, :] = ifreq[1:, :] - ifreq[:-1, :]

    attrs_map = {
        "attr_amp_center": amp,
        "attr_amp_shift_+2": s2,
        "attr_env_center": envelope,
        "attr_env_deriv": env_deriv,
        "attr_ifreq_center": ifreq,
        "attr_ifreq_shift_+1": pad_shift_2d(ifreq, 1),
        "attr_sweetness": sweetness,
        "attr_sweetness_shift_+2": pad_shift_2d(sweetness, 2),
        "attr_spec_amp_10hz": spec_amp_10hz,
        "attr_spec_amp_20hz": spec_amp_20hz,
        "attr_spec_amp_30hz": spec_amp_30hz,
        "attr_spec_amp_40hz": spec_amp_40hz,
        "attr_win_std": win_std,
        "attr_win_max": win_max,
        "attr_n_il_p1_acoustic_impedance_diff": grad_ai,
        "attr_n_xl_p1_acoustic_impedance_diff": grad_ai,
        "attr_n_il_m1_ifreq_center_diff": grad_ifreq,
        "attr_polarity_index": np.clip(amp / (envelope + eps), -1.0, 1.0),
        "attr_rel_pos": rel_pos_2d,
        "si_spec_frac_10": si_spec_frac_10,
        "si_spec_frac_40": si_spec_frac_40,
        "si_norm_grad_il": si_norm_grad_il,
    }

    # Stack reduced features: shape (J_len * K_len, 22)
    X_base = np.column_stack([attrs_map[feat].flatten() for feat in REDUCED_FEATURES])

    # Z-score normalize per slice (zero mean, unit std per feature column)
    means = np.mean(X_base, axis=0)
    stds = np.std(X_base, axis=0)
    stds = np.where(stds > 1e-10, stds, 1.0)
    X_base_norm = (X_base - means) / stds

    stage1_z_preds = {}

    # 1. Stage 1 Predictions
    for target in stage1_targets:
        pipe = pipelines[target]
        indices = pipe["indices"]
        X_sel = X_base_norm[:, indices]
        pred_z = pipe["model"].predict(X_sel)
        stage1_z_preds[target] = pred_z

        # Dynamic Back-Conversion to Absolute Physical Units
        trend_vals = np.polyval(target_trend_coefs[target], seis_times)
        trend_tile = np.tile(trend_vals, J_len)
        robust_std = target_robust_stds[target]
        pred_abs = pred_z * robust_std + trend_tile

        if target == "PHIT":
            pred_abs = np.clip(pred_abs, 0.0, 0.35)
        elif target == "POIS":
            pred_abs = np.clip(pred_abs, 0.05, 0.45)
        elif target == "VPVS":
            pred_abs = np.clip(pred_abs, 1.2, 2.5)

        predictions_3d[target][i, :, :] = pred_abs.reshape((J_len, K_len))

    # 2. Stage 2 Predictions with Cascaded Stage-1 Features
    X_cascaded = np.column_stack([stage1_z_preds[s1] for s1 in stage1_targets])
    X_full_norm = np.hstack([X_base_norm, X_cascaded])

    for target in stage2_targets:
        pipe = pipelines[target]
        indices = pipe["indices"]
        X_sel = X_full_norm[:, indices]
        pred_z = pipe["model"].predict(X_sel)

        trend_vals = np.polyval(target_trend_coefs[target], seis_times)
        trend_tile = np.tile(trend_vals, J_len)
        robust_std = target_robust_stds[target]
        pred_abs = pred_z * robust_std + trend_tile

        if target == "VSH":
            pred_abs = np.clip(pred_abs, 0.0, 1.0)
        elif target in ["PHIE", "PHIT"]:
            pred_abs = np.clip(pred_abs, 0.0, 0.35)
        elif target == "SWE":
            pred_abs = np.clip(pred_abs, 0.0, 1.0)
        elif target == "GR":
            pred_abs = np.clip(pred_abs, 10.0, 200.0)

        predictions_3d[target][i, :, :] = pred_abs.reshape((J_len, K_len))

    if (i + 1) % 25 == 0 or i == I_len - 1:
        print(f"  Inline slice {i + 1}/{I_len} completed ({time.time() - t0_all:.1f}s)")

print(f"\nAll 3D V11 predictions finished in {time.time() - t0_all:.1f}s.")

# Save Int16 Binary Volume Files in frontend/public
EXPORT_TARGETS = {
    "VSH":  {"name": "Volume of Shale (VSH)", "unit": "fraction"},
    "SWE":  {"name": "Water Saturation (SWE)", "unit": "fraction"},
    "PHIE": {"name": "Effective Porosity (PHIE)", "unit": "fraction"},
    "PHIT": {"name": "Total Porosity (PHIT)", "unit": "fraction"},
    "GR":   {"name": "Gamma Ray (GR)", "unit": "API"},
    "RHOB": {"name": "Bulk Density (RHOB)", "unit": "g/cc"},
    "DT":   {"name": "Sonic DT", "unit": "us/ft"},
    "AI":   {"name": "Acoustic Impedance (AI)", "unit": "(m/s)*(g/cc)"},
}

for key, meta_cfg in EXPORT_TARGETS.items():
    if key in predictions_3d:
        vol = predictions_3d[key]
        min_val = float(np.percentile(vol, 0.5))
        max_val = float(np.percentile(vol, 99.5))
        if max_val <= min_val:
            max_val = min_val + 1.0
            
        norm = np.clip((vol - min_val) / (max_val - min_val), 0.0, 1.0)
        int16_vol = np.round(norm * 65534.0 - 32767.0).astype(np.int16)
        
        bin_filename = f"v11_pred_{key.lower()}.bin"
        meta_filename = f"v11_pred_{key.lower()}_meta.json"
        
        bin_path = os.path.join(public_dir, bin_filename)
        meta_path = os.path.join(public_dir, meta_filename)
        
        int16_vol.tofile(bin_path)
        
        meta_info = {
            "key": key,
            "name": meta_cfg["name"],
            "unit": meta_cfg["unit"],
            "min_val": min_val,
            "max_val": max_val,
            "shape": [I_len, J_len, K_len],
            "i_min": i_min, "i_max": i_max,
            "j_min": j_min, "j_max": j_max,
            "tStart": tStart, "dt_ms": dt_ms
        }
        with open(meta_path, "w") as mf:
            json.dump(meta_info, mf, indent=2)
            
        print(f"  [SAVED] {bin_filename} ({os.path.getsize(bin_path)/1e6:.1f} MB) | Range: [{min_val:.3f}, {max_val:.3f}]")

print("\n[SUCCESS] Precomputed V11 ML 3D seismic volume predictions complete!")
