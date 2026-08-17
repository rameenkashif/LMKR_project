import os, sys, json, time
import numpy as np
import segyio
import joblib
import pandas as pd
from scipy.signal import hilbert

# ==========================================
# USER CONFIGURATION: Set your output directory name here
# ==========================================
OUTPUT_FOLDER_NAME = "ml_outputs_v8"

# Paths
script_dir   = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
segy_path    = os.path.join(analysis_dir, "segy",          "origional.segy")
models_dir   = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME, "models")
facies_dir   = os.path.join(analysis_dir, OUTPUT_FOLDER_NAME, "facies_model")
trends_json_path = os.path.join(analysis_dir, "training_data", "output", "log_trends.json")
output_js    = os.path.join(analysis_dir, "frontend", "src", "grid_data.js")

RESERVOIR_T_START = 2086.0
RESERVOIR_T_END   = 2154.0
MAX_ENV_RATIO = 50.0

POLARITY_REF_T_START = 1900.0
POLARITY_REF_T_END   = 2000.0

WELLS = {
    "Z-02": {"x": 1205859.09, "y": 9692966.31, "kb": 146.46},
    "Z-03": {"x": 1201178.25, "y": 9682452.00, "kb": 147.64},
    "Z-04": {"x": 1205820.18, "y": 9696292.65, "kb": 147.64},
    "Z-05": {"x": 1206404.17, "y": 9679510.83, "kb": 144.36},
    "Z-06": {"x": 1207337.37, "y": 9684145.64, "kb": 146.98},
    "Z-07": {"x": 1206364.34, "y": 9688320.18, "kb": 147.97},
    "Z-08-ST-02": {"x": 1202225.66, "y": 9700693.11, "kb": 152.20}
}

stage1_targets = ["AI", "DT", "MURHO", "PHIT", "POIS", "VPVS"]
stage2_targets = ["GR", "RHOB", "VSH", "PHIE", "SWE", "LMRHO"]
TARGETS = stage1_targets + stage2_targets
sand_only_targets = ["PHIE", "SWE", "VPVS", "POIS", "LMRHO", "MURHO"]

print("Loading ML models (v5.0 residual, Extra Trees, and sand-only models)...")
pipelines = {}
for target in TARGETS:
    try:
        # Load standard model
        model = joblib.load(os.path.join(models_dir, f"{target}_model.joblib"))
        scaler = joblib.load(os.path.join(models_dir, f"{target}_scaler.joblib"))
        sel_idx = joblib.load(os.path.join(models_dir, f"{target}_selected_indices.joblib"))
        pipelines[target] = {"model": model, "scaler": scaler, "indices": sel_idx}
        
        # Load sand-only model if available
        if target in sand_only_targets:
            sand_model = joblib.load(os.path.join(models_dir, f"{target}_sand_model.joblib"))
            sand_scaler = joblib.load(os.path.join(models_dir, f"{target}_sand_scaler.joblib"))
            sand_sel_idx = joblib.load(os.path.join(models_dir, f"{target}_sand_selected_indices.joblib"))
            pipelines[target + "_sand"] = {"model": sand_model, "scaler": sand_scaler, "indices": sand_sel_idx}
            
        print(f"  [OK] {target}")
    except Exception as e:
        print(f"  [ERR] {target}: {e}")

# Load facies model
print("Loading Facies classifier model...")
try:
    facies_model = joblib.load(os.path.join(facies_dir, "facies_model.joblib"))
    facies_scaler = joblib.load(os.path.join(facies_dir, "facies_scaler.joblib"))
    facies_features = joblib.load(os.path.join(facies_dir, "facies_features.joblib"))
    print("  [OK] Facies Classifier")
except Exception as e:
    print(f"  [ERR] Facies Classifier: {e}")
    facies_model = None

# Load trends
with open(trends_json_path, "r") as f:
    log_trends = json.load(f)

# Load impedance calibration
tie_summary_path = os.path.join(analysis_dir, "well_seismic", "output", "impedance_tie_summary.csv")
global_inv_scale = 0.075
global_inv_shift = 12000.0
if os.path.exists(tie_summary_path):
    df_ties = pd.read_csv(tie_summary_path)
    if "inv_scale" in df_ties.columns and "inv_shift" in df_ties.columns:
        global_inv_scale = float(df_ties["inv_scale"].mean())
        global_inv_shift = float(df_ties["inv_shift"].mean())
        print(f"Loaded calibration: scale={global_inv_scale:.4f}, shift={global_inv_shift:.1f}")

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
    """Vectorized frequency filter + envelope over all traces in grid."""
    n_traces, n_samples = amp_2d.shape
    freqs = np.fft.fftfreq(n_samples, d=dt_ms/1000.0)
    f_trace = np.fft.fft(amp_2d, axis=1)
    
    f_filtered = f_trace.copy()
    f_filtered[:, np.abs(freqs) < low_hz] = 0
    f_filtered[:, np.abs(freqs) > high_hz] = 0
    
    filtered_trace = np.fft.ifft(f_filtered, axis=1)
    env = np.abs(hilbert(np.real(filtered_trace), axis=1))
    return np.nan_to_num(env, nan=0.0)

print("\nOpening SEG-Y ...")
f_segy = segyio.open(segy_path, "r", ignore_geometry=True)
seis_times = f_segy.samples.astype(float)
dt_ms = float(seis_times[1] - seis_times[0])
inlines_all = f_segy.attributes(segyio.TraceField.FieldRecord)[:]
xlines_all = f_segy.attributes(segyio.TraceField.TraceNumber)[:]
n_traces = len(inlines_all)

print(f"Collecting all {n_traces:,} traces into memory ...")
all_traces = segyio.tools.collect(f_segy.trace)[:]
f_segy.close()

# Polarity Correction
print("\nRunning automatic polarity check ...")
ref_mask = (seis_times >= POLARITY_REF_T_START) & (seis_times <= POLARITY_REF_T_END)
ref_idx = np.where(ref_mask)[0]
if len(ref_idx) == 0:
    polarity_flags = np.ones(n_traces, dtype=float)
else:
    ref_window_traces = all_traces[:, ref_idx]
    ref_mean_amp = np.mean(ref_window_traces, axis=1)
    polarity_flags = np.where(ref_mean_amp >= 0, 1.0, -1.0)
    n_reversed = int(np.sum(polarity_flags == -1.0))
    print(f"  Auto-corrected polarity: {n_reversed:,} reversed traces flipped.")

all_traces = all_traces * polarity_flags[:, np.newaxis]

# Global AI normalisation bounds
print("Computing global AI bounds (sample) ...")
sample_step = max(1, n_traces // 500)
ai_vals = []
for ti in range(0, n_traces, sample_step):
    tr = all_traces[ti]
    ai_vals.extend(np.cumsum(np.clip(tr, -1.0, 1.0)).tolist())
global_ai_min = float(np.percentile(ai_vals, 1))
global_ai_max = float(np.percentile(ai_vals, 99))
print(f"  AI range: {global_ai_min:.3f} – {global_ai_max:.3f}")

res_mask = (seis_times >= RESERVOIR_T_START) & (seis_times <= RESERVOIR_T_END)
res_idx = np.where(res_mask)[0]
print(f"Reservoir window: {RESERVOIR_T_START}–{RESERVOIR_T_END} ms ({len(res_idx)} samples)")

# Vectorized Feature Generation
print("\nComputing features in parallel ...")
t_start_feat = time.time()
amp = np.nan_to_num(all_traces, nan=0.0)

analytic = hilbert(amp, axis=1)
envelope = np.abs(analytic)
phase = np.unwrap(np.angle(analytic), axis=1)
dt_s = dt_ms / 1000.0
ifreq = np.gradient(phase, dt_s, axis=1) / (2.0 * np.pi)
ifreq = np.nan_to_num(ifreq, nan=0.0)
env_deriv = np.nan_to_num(np.gradient(envelope, axis=1), nan=0.0)

rel_pos = np.linspace(0.0, 1.0, amp.shape[1])
rel_pos_2d = np.tile(rel_pos, (n_traces, 1))

cumsum_ai = np.cumsum(np.clip(amp, -1.0, 1.0), axis=1)
cumsum_ai = np.nan_to_num(cumsum_ai, nan=0.0)
impedance = (cumsum_ai - global_ai_min) / (global_ai_max - global_ai_min + 1e-8)

calibrated_ai = global_inv_scale * cumsum_ai + global_inv_shift
calibrated_ai = np.nan_to_num(calibrated_ai, nan=0.0)

max_abs_env = np.max(envelope, axis=1, keepdims=True)
eps = 1e-6 * max_abs_env
env_ratio = np.clip(envelope / (np.abs(amp) + eps), 0.0, MAX_ENV_RATIO)
sweetness = envelope / np.sqrt(np.maximum(ifreq, 0.1))

# Rolling stats
s0 = amp
s1 = pad_shift_2d(amp, 1)
s2 = pad_shift_2d(amp, 2)
s3 = pad_shift_2d(amp, -1)
s4 = pad_shift_2d(amp, -2)
win_max = np.maximum.reduce([s0, s1, s2, s3, s4])
win_min = np.minimum.reduce([s0, s1, s2, s3, s4])
win_mean = (s0 + s1 + s2 + s3 + s4) / 5.0
win_std = np.sqrt(((s0 - win_mean)**2 + (s1 - win_mean)**2 + (s2 - win_mean)**2 + (s3 - win_mean)**2 + (s4 - win_mean)**2) / 5.0)

# Spectral decomposition envelopes
spec_amp_10hz = bandpass_envelope_2d(amp, 8, 12, dt_ms)
spec_amp_15hz = bandpass_envelope_2d(amp, 13, 17, dt_ms)
spec_amp_20hz = bandpass_envelope_2d(amp, 18, 22, dt_ms)
spec_amp_30hz = bandpass_envelope_2d(amp, 27, 33, dt_ms)
spec_amp_40hz = bandpass_envelope_2d(amp, 37, 43, dt_ms)

attrs_dict = {
    "acoustic_impedance": impedance,
    "calibrated_ai": calibrated_ai,
    "amp_center": amp,
    "amp_shift_+1": pad_shift_2d(amp, 1),
    "amp_shift_+2": pad_shift_2d(amp, 2),
    "amp_shift_-1": pad_shift_2d(amp, -1),
    "amp_shift_-2": pad_shift_2d(amp, -2),
    "env_center": envelope,
    "env_deriv": env_deriv,
    "env_ratio": env_ratio,
    "env_shift_+1": pad_shift_2d(envelope, 1),
    "env_shift_+2": pad_shift_2d(envelope, 2),
    "env_shift_-1": pad_shift_2d(envelope, -1),
    "env_shift_-2": pad_shift_2d(envelope, -2),
    "ifreq_center": ifreq,
    "ifreq_shift_+1": pad_shift_2d(ifreq, 1),
    "ifreq_shift_+2": pad_shift_2d(ifreq, 2),
    "ifreq_shift_-1": pad_shift_2d(ifreq, -1),
    "ifreq_shift_-2": pad_shift_2d(ifreq, -2),
    "sweetness": sweetness,
    "sweetness_shift_+1": pad_shift_2d(sweetness, 1),
    "sweetness_shift_+2": pad_shift_2d(sweetness, 2),
    "sweetness_shift_-1": pad_shift_2d(sweetness, -1),
    "sweetness_shift_-2": pad_shift_2d(sweetness, -2),
    "polarity_index": np.clip(amp / (envelope + eps), -1.0, 1.0),
    "polarity_index_shift_+1": pad_shift_2d(np.clip(amp / (envelope + eps), -1.0, 1.0), 1),
    "polarity_index_shift_+2": pad_shift_2d(np.clip(amp / (envelope + eps), -1.0, 1.0), 2),
    "polarity_index_shift_-1": pad_shift_2d(np.clip(amp / (envelope + eps), -1.0, 1.0), -1),
    "polarity_index_shift_-2": pad_shift_2d(np.clip(amp / (envelope + eps), -1.0, 1.0), -2),
    "rel_pos": rel_pos_2d,
    "win_max": win_max,
    "win_mean": win_mean,
    "win_min": win_min,
    "win_std": win_std,
    "spec_amp_10hz": spec_amp_10hz,
    "spec_amp_15hz": spec_amp_15hz,
    "spec_amp_20hz": spec_amp_20hz,
    "spec_amp_30hz": spec_amp_30hz,
    "spec_amp_40hz": spec_amp_40hz,
}

# Extract and inject Multi-Trace Neighbor features
# Pre-build coordinate trace lookup
print("Building inline/crossline trace lookup table...")
trace_lookup = {}
for idx in range(n_traces):
    il = int(inlines_all[idx])
    xl = int(xlines_all[idx])
    trace_lookup[(il, xl)] = idx

# 4 neighbors
neighbors_cfg = [
    ("il_p1", 1, 0),
    ("il_m1", -1, 0),
    ("xl_p1", 0, 1),
    ("xl_m1", 0, -1)
]

print("Computing spatial multi-trace attributes...")
t_start_spatial = time.time()
for tag, dil, dxl in neighbors_cfg:
    # We will construct the shifted attributes directly
    shifted_idx = np.zeros(n_traces, dtype=int)
    valid_mask = np.zeros(n_traces, dtype=bool)
    
    for idx in range(n_traces):
        il = int(inlines_all[idx])
        xl = int(xlines_all[idx])
        target_coords = (il + dil, xl + dxl)
        if target_coords in trace_lookup:
            shifted_idx[idx] = trace_lookup[target_coords]
            valid_mask[idx] = True
            
    # For each key, extract neighbor slice and subtract center slice
    for att_key in ["amp_center", "env_center", "acoustic_impedance", "ifreq_center"]:
        # Pre-allocate gradient array
        grad = np.zeros_like(attrs_dict[att_key])
        # Vectorized extract for valid neighbors
        if np.any(valid_mask):
            neighbor_vals = attrs_dict[att_key][shifted_idx[valid_mask]]
            center_vals = attrs_dict[att_key][valid_mask]
            grad[valid_mask] = neighbor_vals - center_vals
            
        # Add directly to attrs_dict
        feat_name = f"n_{tag}_{att_key}_diff"
        attrs_dict[feat_name] = grad

# Prefix features and sort alphabetically
feature_names = sorted([f"attr_{k}" for k in attrs_dict.keys()])

# Stack features: shape (n_traces, n_samples, n_features)
X_all_features = np.stack([attrs_dict[name.replace("attr_", "")] for name in feature_names], axis=2)

# Slice to reservoir window
X_res = X_all_features[:, res_idx, :]
n_features = len(feature_names)
X_res_2d = X_res.reshape(-1, n_features)

print(f"Features computed in {time.time() - t_start_feat:.1f}s. Input shape: {X_res_2d.shape}")

# ── Run Facies Classifier ──────────────────────────────────────────────────
print("\nRunning Sand/Shale classifier...")
is_sand_prob_res_2d = np.zeros((n_traces, len(res_idx)))
if facies_model is not None:
    # Prepare features for facies classifier
    X_facies_scaled = facies_scaler.transform(X_res_2d)
    is_sand_prob_res = facies_model.predict_proba(X_facies_scaled)[:, 1]
    is_sand_prob_res_2d = is_sand_prob_res.reshape(n_traces, len(res_idx))
    print(f"  Facies prediction complete. Average sand sample ratio: {np.mean(is_sand_prob_res > 0.5):.2%}")

# ── Run ML Inference ────────────────────────────────────────────────────────
print("\nRunning ML model predictions...")
t_start_inf = time.time()
predictions_averaged = {}
predictions_2d = {}

# Precompute compaction trends for the reservoir window times
res_times = seis_times[res_idx]

cascade_preds = {}

for target in TARGETS:
    t0 = time.time()
    
    # Build the exact feature matrix that was used during training.
    # At any point, the available cascade features are the predictions of any PRECEDING Stage 1 targets.
    available_cascades = []
    for s1 in stage1_targets:
        if s1 == target:
            break
        if f"cascade_pred_{s1}" in cascade_preds:
            available_cascades.append(cascade_preds[f"cascade_pred_{s1}"])
            
    if len(available_cascades) > 0:
        X_cascades = np.column_stack(available_cascades)
        X_input = np.hstack([X_res_2d, X_cascades])
    else:
        X_input = X_res_2d
        
    # 1. Standard (all-samples) prediction
    pipe = pipelines[target]
    X_sel = X_input[:, pipe["indices"]]
    X_scaled = pipe["scaler"].transform(X_sel)
    pred_res_samples = pipe["model"].predict(X_scaled)
    
    # Save predictions to serve as cascaded features for subsequent targets
    if target in stage1_targets:
        cascade_preds[f"cascade_pred_{target}"] = pred_res_samples
        
    # Map back to 2D
    pred_res_2d = pred_res_samples.reshape(n_traces, len(res_idx))
    
    # Reconstruct absolute values by adding trend
    coefs = log_trends[target]
    trend_vals = np.polyval(coefs, res_times)
    pred_abs_2d = pred_res_2d + trend_vals[np.newaxis, :]
    
    # 2. Hierarchical Sand-Only Model (where predicted Sand probability > 0.5)
    if target in sand_only_targets and (target + "_sand") in pipelines:
        # Build features for sand-only model (incorporates predictions of Stage 1 targets up to and including the current target)
        available_cascades_sand = []
        for s1 in stage1_targets:
            if s1 == target:
                if f"cascade_pred_{s1}" in cascade_preds:
                    available_cascades_sand.append(cascade_preds[f"cascade_pred_{s1}"])
                break
            if f"cascade_pred_{s1}" in cascade_preds:
                available_cascades_sand.append(cascade_preds[f"cascade_pred_{s1}"])
                
        if len(available_cascades_sand) > 0:
            X_cascades_s = np.column_stack(available_cascades_sand)
            X_input_sand = np.hstack([X_res_2d, X_cascades_s])
        else:
            X_input_sand = X_res_2d
            
        pipe_sand = pipelines[target + "_sand"]
        X_sel_s = X_input_sand[:, pipe_sand["indices"]]
        X_scaled_s = pipe_sand["scaler"].transform(X_sel_s)
        pred_res_sand_samples = pipe_sand["model"].predict(X_scaled_s)
        
        pred_res_sand_2d = pred_res_sand_samples.reshape(n_traces, len(res_idx))
        pred_abs_sand_2d = pred_res_sand_2d + trend_vals[np.newaxis, :]
        
        # Apply sand-only predictions to samples where is_sand_prob > 0.5, else use shale baseline
        if target == "PHIE":
            baseline_val = 0.0
        elif target == "SWE":
            baseline_val = 1.0
        else:
            baseline_val = trend_vals[np.newaxis, :]
            
        is_sand_mask = is_sand_prob_res_2d > 0.5
        
        # Combine
        combined_pred_abs_2d = np.zeros_like(pred_abs_2d)
        if target in ["PHIE", "SWE"]:
            combined_pred_abs_2d.fill(baseline_val)
            combined_pred_abs_2d[is_sand_mask] = pred_abs_sand_2d[is_sand_mask]
        else:
            combined_pred_abs_2d = np.tile(trend_vals, (n_traces, 1))
            combined_pred_abs_2d[is_sand_mask] = pred_abs_sand_2d[is_sand_mask]
            
        pred_abs_2d = combined_pred_abs_2d
        print(f"  [OK] {target} (Sand-Only calibrated)")
    else:
        print(f"  [OK] {target} (Standard absolute)")
        
    # Average along reservoir window axis
    pred_mean = np.mean(pred_abs_2d, axis=1)
    predictions_averaged[target] = pred_mean
    predictions_2d[target] = pred_abs_2d
    print(f"    Completed in {time.time() - t0:.1f}s")

# Average sand probability along reservoir window
is_sand_prob_mean = np.mean(is_sand_prob_res_2d, axis=1)
is_sand_mean = np.where(is_sand_prob_mean > 0.5, 1.0, 0.0)

# Compute Sweet Spot Index sample-by-sample
# Using LMRHO (fluid indicator) as the proxy instead of noisy SWE to avoid zero-out anomalies
print("\nComputing Sweet Spot Index (SSI) sample-by-sample vertically...")
phie_2d = np.clip(predictions_2d.get("PHIE", np.zeros_like(is_sand_prob_res_2d)), 0, 1)
vsh_2d = np.clip(predictions_2d.get("VSH", np.zeros_like(is_sand_prob_res_2d)), 0, 1)
lmrho_2d = predictions_2d.get("LMRHO", np.zeros_like(is_sand_prob_res_2d))

# Normalize LMRHO: lower LMRHO = higher gas likelihood (scaled 0 to 1)
lmrho_min = np.min(lmrho_2d)
lmrho_max = np.max(lmrho_2d)
lmrho_range = lmrho_max - lmrho_min if (lmrho_max - lmrho_min) > 0.001 else 1.0
gas_index_2d = (lmrho_max - lmrho_2d) / lmrho_range

ssi_2d = phie_2d * (1.0 - vsh_2d) * gas_index_2d
predictions_averaged["SSI"] = np.mean(ssi_2d, axis=1)

# ── Assemble Grid Data ──────────────────────────────────────────────────────
print("\nAssembling grid points ...")
grid_points = []
for ti in range(n_traces):
    il = int(inlines_all[ti])
    xl = int(xlines_all[ti])
    point = {"il": il, "xl": xl}
    for target in TARGETS:
        point[target] = round(float(predictions_averaged[target][ti]), 4)
        
    point["IS_SAND"] = int(is_sand_mean[ti])
    point["SSI"] = round(float(predictions_averaged["SSI"][ti]), 5)
    grid_points.append(point)

# ── well inline/crossline from tie summary ──────────────────────────────────
well_grid = {}
try:
    with open(tie_summary_path, newline="") as fh:
        reader = csv = pd.read_csv(tie_summary_path)
        for _, row in reader.iterrows():
            wname = row["well"]
            well_grid[wname] = {
                "name": wname,
                "il":   int(row["inline"]),
                "xl":   int(row["crossline"]),
                "x":    WELLS[wname]["x"],
                "y":    WELLS[wname]["y"],
            }
except Exception as e:
    print(f"Warning: could not read tie CSV: {e}")

# ── export JS ───────────────────────────────────────────────────────────────
print(f"\nExporting {len(grid_points):,} grid points to {output_js} ...")
all_inlines   = [int(x) for x in sorted(list(set(inlines_all.astype(int))))]
all_crosslines = [int(x) for x in sorted(list(set(xlines_all.astype(int))))]

export = {
    "inlines":    all_inlines,
    "crosslines": all_crosslines,
    "targets":    TARGETS + ["SSI", "IS_SAND"],
    "reservoir":  {"t_start": RESERVOIR_T_START, "t_end": RESERVOIR_T_END},
    "wells":      list(well_grid.values()),
    "points":     grid_points,
}

js_content = f"export const gridData = {json.dumps(export, separators=(',', ':'))};\n"

with open(output_js, "w", encoding="utf-8") as fh:
    fh.write(js_content)

size_mb = os.path.getsize(output_js) / 1e6
print(f"[OK] Written {output_js}  ({size_mb:.1f} MB)")
