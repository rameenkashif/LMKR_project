import os
import json
import numpy as np
import pandas as pd
import segyio
import lasio
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
segy_path = os.path.abspath(os.path.join(script_dir, "..", "segy", "origional.segy"))
las_dir = os.path.abspath(os.path.join(script_dir, "..", "las_cleaned"))  # Points to las_cleaned!
tie_summary_path = os.path.abspath(os.path.join(script_dir, "..", "well_seismic", "output", "impedance_tie_summary.csv"))
output_dir = os.path.abspath(os.path.join(script_dir, "output"))

# Create output folder
os.makedirs(output_dir, exist_ok=True)

FEET_TO_METERS = 0.3048
MAX_ENV_RATIO = 50.0  # Sane geological limit to prevent divide-by-zero blowout

def _pad_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    """Shift a 1D trace by one or more samples, padding with zeros at the edges."""
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

def _rolling_stats(arr: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling max/mean/min/std around each sample."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    half = window // 2
    maxv = np.zeros(n, dtype=float)
    meanv = np.zeros(n, dtype=float)
    minv = np.zeros(n, dtype=float)
    stdv = np.zeros(n, dtype=float)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = arr[lo:hi]
        if len(window_vals) == 0:
            continue
        maxv[i] = float(np.max(window_vals))
        meanv[i] = float(np.mean(window_vals))
        minv[i] = float(np.min(window_vals))
        stdv[i] = float(np.std(window_vals))

    return maxv, meanv, minv, stdv

def bandpass_envelope(trace, low_hz, high_hz, dt_ms):
    """Compute envelope of a bandpass filtered trace to extract spectral components."""
    n = len(trace)
    freqs = np.fft.fftfreq(n, d=dt_ms/1000.0)
    f_trace = np.fft.fft(trace)
    
    # Bandpass filter in frequency domain
    f_filtered = f_trace.copy()
    f_filtered[np.abs(freqs) < low_hz] = 0
    f_filtered[np.abs(freqs) > high_hz] = 0
    
    filtered_trace = np.fft.ifft(f_filtered)
    # Hilbert envelope of real component
    env = np.abs(hilbert(np.real(filtered_trace)))
    return np.nan_to_num(env, nan=0.0)

def compute_trace_attributes(trace, times_ms, dt_ms, global_ai_min=None, global_ai_max=None, inv_scale=None, inv_shift=None) -> dict[str, np.ndarray]:
    """Build a comprehensive attribute feature dictionary from a seismic trace including spectral attributes."""
    trace = np.asarray(trace, dtype=float)
    n_samples = len(trace)
    if n_samples == 0:
        return {}

    amp = np.nan_to_num(trace.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    analytic = hilbert(amp)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dt_s = dt_ms / 1000.0
    ifreq = np.gradient(phase, dt_s) / (2.0 * np.pi)
    ifreq = np.nan_to_num(ifreq, nan=0.0, posinf=0.0, neginf=0.0)

    rel_pos = np.linspace(0.0, 1.0, n_samples, dtype=float)
    env_deriv = np.nan_to_num(np.gradient(envelope), nan=0.0, posinf=0.0, neginf=0.0)

    # Relative impedance proxy (cumulative sum of clipped amplitude)
    cumsum_ai = np.cumsum(np.clip(amp, -1.0, 1.0))
    cumsum_ai = np.nan_to_num(cumsum_ai, nan=0.0, posinf=0.0, neginf=0.0)
    impedance = cumsum_ai.copy()
    
    if global_ai_min is not None and global_ai_max is not None:
        impedance = (impedance - global_ai_min) / (global_ai_max - global_ai_min + 1e-8)
    else:
        impedance = impedance - np.min(impedance)
        if np.max(impedance) > 0:
            impedance = impedance / np.max(impedance)

    if inv_scale is not None and inv_shift is not None:
        calibrated_ai = inv_scale * cumsum_ai + inv_shift
    else:
        calibrated_ai = impedance
    calibrated_ai = np.nan_to_num(calibrated_ai, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs_env = np.max(envelope)
    eps = 1e-6 * max_abs_env
    env_ratio = np.divide(envelope, np.abs(amp) + eps)

    polarity_eps = max(eps, 1e-9)
    polarity_index = np.clip(amp / (envelope + polarity_eps), -1.0, 1.0)
    env_ratio = np.clip(env_ratio, 0.0, MAX_ENV_RATIO)

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
        "amp_shift_+3": _pad_shift(amp, 3),
        "amp_shift_+4": _pad_shift(amp, 4),
        "amp_shift_+5": _pad_shift(amp, 5),
        "amp_shift_-1": _pad_shift(amp, -1),
        "amp_shift_-2": _pad_shift(amp, -2),
        "amp_shift_-3": _pad_shift(amp, -3),
        "amp_shift_-4": _pad_shift(amp, -4),
        "amp_shift_-5": _pad_shift(amp, -5),
        "env_center": envelope,
        "env_deriv": env_deriv,
        "env_ratio": env_ratio,
        "env_shift_+1": _pad_shift(envelope, 1),
        "env_shift_+2": _pad_shift(envelope, 2),
        "env_shift_+3": _pad_shift(envelope, 3),
        "env_shift_+4": _pad_shift(envelope, 4),
        "env_shift_+5": _pad_shift(envelope, 5),
        "env_shift_-1": _pad_shift(envelope, -1),
        "env_shift_-2": _pad_shift(envelope, -2),
        "env_shift_-3": _pad_shift(envelope, -3),
        "env_shift_-4": _pad_shift(envelope, -4),
        "env_shift_-5": _pad_shift(envelope, -5),
        "ifreq_center": ifreq,
        "ifreq_shift_+1": _pad_shift(ifreq, 1),
        "ifreq_shift_+2": _pad_shift(ifreq, 2),
        "ifreq_shift_+3": _pad_shift(ifreq, 3),
        "ifreq_shift_+4": _pad_shift(ifreq, 4),
        "ifreq_shift_+5": _pad_shift(ifreq, 5),
        "ifreq_shift_-1": _pad_shift(ifreq, -1),
        "ifreq_shift_-2": _pad_shift(ifreq, -2),
        "ifreq_shift_-3": _pad_shift(ifreq, -3),
        "ifreq_shift_-4": _pad_shift(ifreq, -4),
        "ifreq_shift_-5": _pad_shift(ifreq, -5),
        "sweetness": sweetness,
        "sweetness_shift_+1": _pad_shift(sweetness, 1),
        "sweetness_shift_+2": _pad_shift(sweetness, 2),
        "sweetness_shift_+3": _pad_shift(sweetness, 3),
        "sweetness_shift_+4": _pad_shift(sweetness, 4),
        "sweetness_shift_+5": _pad_shift(sweetness, 5),
        "sweetness_shift_-1": _pad_shift(sweetness, -1),
        "sweetness_shift_-2": _pad_shift(sweetness, -2),
        "sweetness_shift_-3": _pad_shift(sweetness, -3),
        "sweetness_shift_-4": _pad_shift(sweetness, -4),
        "sweetness_shift_-5": _pad_shift(sweetness, -5),
        "polarity_index": polarity_index,
        "polarity_index_shift_+1": _pad_shift(polarity_index, 1),
        "polarity_index_shift_+2": _pad_shift(polarity_index, 2),
        "polarity_index_shift_-1": _pad_shift(polarity_index, -1),
        "polarity_index_shift_-2": _pad_shift(polarity_index, -2),
        "rel_pos": rel_pos,
        # Spectral features
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

    prefixed_attrs = {f"attr_{k}": v for k, v in attrs.items()}
    return prefixed_attrs

def main():
    print("Starting training dataset generation with Block-Averaging using CLEANED LAS files...")
    if not os.path.exists(tie_summary_path):
        raise FileNotFoundError(f"Tie summary not found at {tie_summary_path}. Run well-seismic tie first!")

    df_ties = pd.read_csv(tie_summary_path)

    with segyio.open(segy_path, "r", ignore_geometry=True) as f_segy:
        seis_times = f_segy.samples.astype(float)
        dt_ms = float(seis_times[1] - seis_times[0])
        
        all_data_rows = []
        
        inlines = f_segy.attributes(segyio.TraceField.FieldRecord)[:]
        crosslines = f_segy.attributes(segyio.TraceField.TraceNumber)[:]
        
        train_impedance_samples = []
        
        for _, row in df_ties.iterrows():
            well_name = row["well"]
            polarity_flag = row.get("polarity", "Normal")
            polarity = -1.0 if polarity_flag == "Reversed" else 1.0
            mapped_inline = int(row["inline"])
            mapped_crossline = int(row["crossline"])
            
            trace_idx = np.where((inlines == mapped_inline) & (crosslines == mapped_crossline))[0]
            if len(trace_idx) == 0:
                continue
            trace_idx = int(trace_idx[0])
            
            raw_trace = f_segy.trace[trace_idx].copy()
            trace = raw_trace * polarity
            
            amp = np.nan_to_num(trace.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            impedance_raw = np.cumsum(np.clip(amp, -1.0, 1.0))
            impedance_raw = np.nan_to_num(impedance_raw, nan=0.0, posinf=0.0, neginf=0.0)
            
            if well_name != "Z-04":
                train_impedance_samples.extend(impedance_raw)
                
        global_ai_min = np.min(train_impedance_samples)
        global_ai_max = np.max(train_impedance_samples)
        print(f"[GLOBAL SCALE] Acoustic Impedance training min: {global_ai_min:.4f}, max: {global_ai_max:.4f} (excluding Z-04)")
        
        for _, row in df_ties.iterrows():
            well_name = row["well"]
            shift_ms = float(row["std_shift"])
            polarity_flag = row.get("polarity", "Normal")
            polarity = -1.0 if polarity_flag == "Reversed" else 1.0
            tie_quality = float(row.get("std_corr", 1.0))
            inv_scale = float(row.get("inv_scale", 1.0))
            inv_shift = float(row.get("inv_shift", 0.0))
            mapped_inline = int(row["inline"])
            mapped_crossline = int(row["crossline"])
            
            trace_idx = np.where((inlines == mapped_inline) & (crosslines == mapped_crossline))[0]
            if len(trace_idx) == 0:
                continue
            trace_idx = int(trace_idx[0])
            
            # center trace
            raw_trace = f_segy.trace[trace_idx].copy()
            trace = raw_trace * polarity
            
            # Generate attributes for center trace
            attrs = compute_trace_attributes(trace, seis_times, dt_ms,
                                             global_ai_min=global_ai_min, global_ai_max=global_ai_max,
                                             inv_scale=inv_scale, inv_shift=inv_shift)
            
            # Lateral Multi-Trace Neighborhood Gradient Features
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
                    # extract key attributes from neighbor
                    n_attrs = compute_trace_attributes(n_trace, seis_times, dt_ms,
                                                       global_ai_min=global_ai_min, global_ai_max=global_ai_max,
                                                       inv_scale=inv_scale, inv_shift=inv_shift)
                    # compute gradients
                    for att_key in ["attr_amp_center", "attr_env_center", "attr_acoustic_impedance", "attr_ifreq_center"]:
                        attrs[f"attr_n_{tag}_{att_key.replace('attr_', '')}_diff"] = n_attrs[att_key] - attrs[att_key]
                else:
                    for att_key in ["attr_amp_center", "attr_env_center", "attr_acoustic_impedance", "attr_ifreq_center"]:
                        attrs[f"attr_n_{tag}_{att_key.replace('attr_', '')}_diff"] = np.zeros_like(trace)

            # Naming correction for Z-08
            lookup_name = "Z-08-ST-02" if well_name == "Z-08" else well_name
            las_path = os.path.join(las_dir, f"{lookup_name}.las")
            if not os.path.exists(las_path):
                print(f"[SKIP] Cleaned LAS file not found for well {well_name} at {las_path}")
                continue
                
            las = lasio.read(las_path)
            df_las = las.df()
            
            target_curves = ["GR", "DT", "RHOB", "VSH", "PHIE", "SWE", "PHIT", "AI", "VPVS", "POIS", "LMRHO", "MURHO"]
            available_targets = [c for c in target_curves if c in df_las.columns]
            
            if "DPTM" not in df_las.columns:
                continue
                
            dptm_ms = df_las["DPTM"].values.astype(float)
            well_time_ms = dptm_ms + shift_ms
            
            bin_indices = np.clip(
                ((well_time_ms - seis_times[0]) / dt_ms).round().astype(int),
                0, len(seis_times) - 1
            )
            
            well_blocked_rows = []
            for bin_idx in np.unique(bin_indices):
                mask = bin_indices == bin_idx
                t_ms = seis_times[bin_idx]
                
                row_dict = {
                    "well_name": well_name,
                    "seismic_time_ms": float(t_ms),
                    "inline": mapped_inline,
                    "crossline": mapped_crossline,
                    "tie_quality_weight": tie_quality,
                }
                
                has_valid_target = False
                for curve_name in available_targets:
                    vals = df_las[curve_name].values[mask].astype(float)
                    mean_val = np.nanmean(vals)
                    if not np.isnan(mean_val):
                        row_dict[f"target_{curve_name}"] = float(mean_val)
                        has_valid_target = True
                    else:
                        row_dict[f"target_{curve_name}"] = float("nan")
                
                if "target_VSH" in row_dict and not np.isnan(row_dict["target_VSH"]):
                    row_dict["target_IS_SAND"] = int(row_dict["target_VSH"] < 0.4)
                else:
                    row_dict["target_IS_SAND"] = 0
                        
                if has_valid_target:
                    for attr_name, attr_arr in attrs.items():
                        row_dict[attr_name] = float(attr_arr[bin_idx])
                    well_blocked_rows.append(row_dict)
                    
            df_well_blocked = pd.DataFrame(well_blocked_rows)
            all_data_rows.append(df_well_blocked)
            print(f"[SUCCESS] Prepared {len(df_well_blocked)} block-averaged samples for well {well_name}")

    df_training_table = pd.concat(all_data_rows, ignore_index=True).dropna().reset_index(drop=True)
    print(f"\nFinal training dataset contains {len(df_training_table)} block-averaged samples.")

    # Fitting compaction trends
    print("\nFitting degree-2 geological compaction trends...")
    log_trends = {}
    train_df = df_training_table[df_training_table["well_name"] != "Z-04"]
    
    numerical_targets = [c for c in target_curves]
    for target in numerical_targets:
        target_col = f"target_{target}"
        if target_col in df_training_table.columns:
            x_train = train_df["seismic_time_ms"].values
            y_train = train_df[target_col].values
            
            coefs = np.polyfit(x_train, y_train, 2)
            log_trends[target] = list(coefs)
            
            all_times = df_training_table["seismic_time_ms"].values
            trend_vals = np.polyval(coefs, all_times)
            df_training_table[f"{target_col}_trend"] = trend_vals
            df_training_table[f"{target_col}_residual"] = df_training_table[target_col] - trend_vals
            
            print(f"  Target '{target}': Trend fit coefficients = {[f'{c:.4e}' for c in coefs]}")
            
    # Save trends JSON
    trends_json_path = os.path.join(output_dir, "log_trends.json")
    with open(trends_json_path, "w") as f:
        json.dump(log_trends, f, indent=4)
    print(f"Saved trend coefficients to: {trends_json_path}")

    # Save training table CSV
    training_table_path = os.path.join(output_dir, "training_table.csv")
    df_training_table.to_csv(training_table_path, index=False)
    print(f"Saved training table to: {training_table_path}")

if __name__ == "__main__":
    main()
