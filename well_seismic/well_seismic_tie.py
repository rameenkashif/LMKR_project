import os
import csv
import json
import datetime
import numpy as np
import pandas as pd
import segyio
import lasio
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
segy_path = os.path.abspath(os.path.join(script_dir, "..", "segy", "origional.segy"))
las_dir = os.path.abspath(os.path.join(script_dir, "..", "las"))
output_dir = os.path.abspath(os.path.join(script_dir, "output"))

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Conversion factors and settings
FEET_TO_METERS = 0.3048
INLINE_FIELD = segyio.TraceField.FieldRecord
CROSSLINE_FIELD = segyio.TraceField.TraceNumber

# Tie quality thresholds
TIE_QUALITY_GOOD = 0.60
TIE_QUALITY_FAIR = 0.40

# Grid search parameters for finding the best tie
WAVELET_FREQS_HZ = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
WAVELET_PHASES_DEG = [0, 90, 180, 270]
BULK_SHIFT_SAMPLES = range(-20, 21)  # -40ms to +40ms at 2ms sampling

# Well data in original feet coordinate system
WELLS = {
    "Z-02": {"x": 1205859.09, "y": 9692966.31, "kb": 146.46},
    "Z-03": {"x": 1201178.25, "y": 9682452.00, "kb": 147.64},
    "Z-04": {"x": 1205820.18, "y": 9696292.65, "kb": 147.64},
    "Z-05": {"x": 1206404.17, "y": 9679510.83, "kb": 144.36},
    "Z-06": {"x": 1207337.37, "y": 9684145.64, "kb": 146.98},
    "Z-07": {"x": 1206364.34, "y": 9688320.18, "kb": 147.97},
    "Z-08-ST-02": {"x": 1202225.66, "y": 9700693.11, "kb": 152.20}
}

class SeismicVolume:
    """Wrapper to handle SEG-Y loading and nearest trace searching."""
    def __init__(self, path):
        self.path = path
        self._f = segyio.open(path, "r", ignore_geometry=True)
        self.inline = self._f.attributes(INLINE_FIELD)[:]
        self.crossline = self._f.attributes(CROSSLINE_FIELD)[:]
        self.src_x = self._f.attributes(segyio.TraceField.SourceX)[:].astype(float)
        self.src_y = self._f.attributes(segyio.TraceField.SourceY)[:].astype(float)
        self.times_ms = self._f.samples.astype(float)
        self.dt_ms = float(self.times_ms[1] - self.times_ms[0])
        self.n_traces = self._f.tracecount

    def trace(self, index):
        return self._f.trace[int(index)]

    def bounding_box(self):
        return (self.src_x.min(), self.src_x.max(),
                self.src_y.min(), self.src_y.max())

    def nearest_trace(self, x, y):
        dist = np.sqrt((self.src_x - x) ** 2 + (self.src_y - y) ** 2)
        idx = int(np.argmin(dist))
        return idx, float(dist[idx]), int(self.inline[idx]), int(self.crossline[idx])

    def close(self):
        self._f.close()


class WellLogError(Exception):
    pass


def load_well_logs(las_path):
    """Load LAS logs and return necessary curves with simple validation."""
    las = lasio.read(las_path)
    df = las.df()

    required = ["DT", "RHOB"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise WellLogError(
            f"LAS file is missing required curves: {missing}."
        )

    depth = df.index.values.astype(float)
    dt_log = df["DT"].values.astype(float)
    rhob = df["RHOB"].values.astype(float)
    dptm = df["DPTM"].values.astype(float) if "DPTM" in df.columns else None

    return {
        "depth": depth,
        "dt_us_ft": dt_log,
        "rhob_gcc": rhob,
        "dptm_ms": dptm,
        "curves_available": list(df.columns),
    }


def get_time_depth_curve(well_logs):
    """Return depth and time-depth relationship."""
    if well_logs["dptm_ms"] is None:
        raise WellLogError("No DPTM (depth-time) curve found in this LAS file.")
    return well_logs["depth"], well_logs["dptm_ms"]


def build_reflectivity(well_logs):
    """Calculate Acoustic Impedance and Reflectivity Coefficients in depth and time."""
    depth, t_ms = get_time_depth_curve(well_logs)

    dt_us_ft = well_logs["dt_us_ft"]
    rhob = well_logs["rhob_gcc"]

    # Convert sonic log (us/ft) to velocity (m/s)
    # Velocity (ft/s) = 1e6 / DT
    # Velocity (m/s) = (1e6 / DT) * 0.3048
    vp_ms = (1.0e6 / dt_us_ft) * 0.3048
    impedance = vp_ms * rhob

    # Calculate reflection coefficients (RC)
    refl = (impedance[1:] - impedance[:-1]) / (impedance[1:] + impedance[:-1])
    refl_time = (t_ms[1:] + t_ms[:-1]) / 2.0

    return refl, refl_time, impedance


def resample_reflectivity(refl, refl_time, dt_ms):
    """Resample reflectivity onto a regular time grid matching the seismic sample rate."""
    t_start = np.ceil(refl_time.min() / dt_ms) * dt_ms
    t_end = np.floor(refl_time.max() / dt_ms) * dt_ms
    reg_time = np.arange(t_start, t_end + dt_ms, dt_ms)

    refl_reg = np.zeros(len(reg_time))
    counts = np.zeros(len(reg_time))
    bin_idx = np.clip(((refl_time - t_start) / dt_ms).round().astype(int), 0, len(reg_time) - 1)
    
    for b, r in zip(bin_idx, refl):
        refl_reg[b] += r
        counts[b] += 1
        
    counts[counts == 0] = 1
    refl_reg = refl_reg / counts

    return reg_time, refl_reg


def ricker_wavelet(freq_hz, length_ms, dt_ms):
    """Generate a Ricker wavelet."""
    t = np.arange(-length_ms / 2, length_ms / 2 + dt_ms, dt_ms) / 1000.0
    y = (1 - 2 * (np.pi * freq_hz * t) ** 2) * np.exp(-(np.pi * freq_hz * t) ** 2)
    return y


def rotate_phase(wavelet, phase_deg):
    """Apply phase shift rotation to a wavelet using Hilbert transform."""
    analytic = hilbert(wavelet)
    phase_rad = np.deg2rad(phase_deg)
    rotated = np.real(analytic * np.exp(1j * phase_rad))
    return rotated


def _normalize(x):
    """Zero-mean, unit-variance normalization."""
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def find_best_tie(refl_reg, reg_time, seismic, seis_times, dt_ms):
    """Grid-search wavelet parameters and bulk shift to optimize correlation."""
    best = None

    for freq in WAVELET_FREQS_HZ:
        base_wav = ricker_wavelet(freq, 100, dt_ms)
        for phase in WAVELET_PHASES_DEG:
            wav = rotate_phase(base_wav, phase)
            syn = np.convolve(refl_reg, wav, mode="same")
            n_syn = len(syn)

            for shift in BULK_SHIFT_SAMPLES:
                shifted_start = reg_time[0] + shift * dt_ms
                idx = int(np.searchsorted(seis_times, shifted_start))
                if idx < 0 or idx + n_syn > len(seis_times):
                    continue
                real_seg = seismic[idx: idx + n_syn]
                if len(real_seg) != n_syn:
                    continue

                corr = float(np.corrcoef(_normalize(real_seg), _normalize(syn))[0, 1])

                if best is None or abs(corr) > abs(best["correlation"]):
                    best = {
                        "freq_hz": freq,
                        "phase_deg": phase,
                        "shift_ms": shift * dt_ms,
                        "correlation": corr,
                        "synthetic": syn,
                        "real_window": real_seg,
                        "window_times": seis_times[idx: idx + n_syn],
                    }

    if best is None:
        raise RuntimeError("No overlapping tie window found.")

    return best


def tie_quality_label(abs_corr):
    if abs_corr >= TIE_QUALITY_GOOD:
        return "GOOD"
    elif abs_corr >= TIE_QUALITY_FAIR:
        return "FAIR"
    else:
        return "POOR"


def recursive_inversion(seismic_trace, seis_times, tie_result, well_impedance, well_time_ms):
    """Extract band-limited relative impedance from seismic trace, calibrated with well log."""
    synth = tie_result["synthetic"]
    real_window = tie_result["real_window"]

    rms_refl_equivalent = np.std(synth)
    rms_seismic = np.std(real_window)
    scale = rms_refl_equivalent / rms_seismic if rms_seismic != 0 else 1.0

    r_series = np.clip(seismic_trace * scale, -0.45, 0.45)

    seed_idx = np.argmin(np.abs(well_time_ms - tie_result["window_times"][0]))
    z0 = well_impedance[seed_idx] if len(well_impedance) else 1.0

    z = np.zeros(len(r_series))
    z[0] = z0
    for i in range(len(r_series) - 1):
        z[i + 1] = z[i] * (1 + r_series[i]) / (1 - r_series[i])

    return z


def plot_tie(well_name, refl_reg, reg_time, tie_result, out_path):
    """Generate and save the tie plot comparing synthetic and real seismic wiggles."""
    synth = tie_result["synthetic"]
    real_window = tie_result["real_window"]
    times = tie_result["window_times"]
    corr = tie_result["correlation"]

    display_synth = -synth if corr < 0 else synth
    polarity_note = "inverted" if corr < 0 else "normal"

    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    # Plot 1: Well reflectivity
    axes[0].plot(refl_reg, reg_time, color="black", linewidth=1.0)
    axes[0].invert_yaxis()
    axes[0].set_title("Well Reflectivity\n(Depth derived)")
    axes[0].set_ylabel("Two-Way Time (ms)")
    axes[0].set_xlabel("Reflectivity Coeff.")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Tie comparison
    axes[1].plot(_normalize(real_window), times, color="tab:blue", label="Real Seismic Trace", linewidth=1.5)
    axes[1].plot(_normalize(display_synth), times, color="tab:red", linestyle="--", 
                 label=f"Synthetic ({polarity_note})", linewidth=1.5)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_xlabel("Normalized Amplitude")
    
    quality = tie_quality_label(abs(corr))
    axes[1].set_title(
        f"{well_name} Tie QC - {quality}\n"
        f"Corr: {corr:.3f} | Freq: {tie_result['freq_hz']}Hz\n"
        f"Phase: {tie_result['phase_deg']}° | Shift: {tie_result['shift_ms']:.0f}ms"
    )
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_inversion(well_name, seis_times, seismic_trace, inv_impedance, well_impedance, well_time_ms, out_path):
    """Generate and save relative impedance inversion plot."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    # Plot 1: Seismic trace
    axes[0].plot(seismic_trace, seis_times, color="gray", linewidth=1.0)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Seismic Trace at Well\n(Tied Location)")
    axes[0].set_xlabel("Amplitude")
    axes[0].set_ylabel("Two-Way Time (ms)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Relative Impedance
    axes[1].plot(inv_impedance, seis_times, color="tab:purple", label="Recursive Inversion", linewidth=1.5)
    ax2 = axes[1].twinx()
    ax2.plot(well_impedance, well_time_ms, color="tab:green", alpha=0.6, label="Well Log Impedance")
    axes[1].invert_yaxis()
    axes[1].set_title("Relative Acoustic Impedance")
    axes[1].set_xlabel("Impedance Value")
    
    # Combined legend
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def write_well_report(well_name, out_path, location_info, tie_result, curves):
    """Save an individual detailed Markdown report for the well."""
    quality = tie_quality_label(abs(tie_result["correlation"]))
    polarity_desc = "reversed phase (reversed polarity)" if tie_result["correlation"] < 0 else "normal phase (same polarity)"

    lines = [
        f"# Well-to-Seismic Tie Report - {well_name}",
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Well Coordinate & Seismic Mapping QC",
        f"- **Well Location**: X = {location_info['x']:.2f} m, Y = {location_info['y']:.2f} m",
        f"- **Kelly Bushing (KB) Elevation**: {location_info['kb']:.2f} m",
        f"- **Mapped Trace**: Inline **{location_info['inline']}**, Crossline **{location_info['crossline']}**",
        f"- **Well-to-Trace Distance (Offset)**: {location_info['distance_m']:.2f} meters",
        "",
        "## 2. Well-to-Seismic Tie Analysis",
        f"- **Tie Quality**: **{quality}**",
        f"- **Correlation Coefficient ($R$)**: {tie_result['correlation']:.3f}",
        f"- **Optimal Ricker Wavelet Frequency**: {tie_result['freq_hz']} Hz",
        f"- **Optimal Wavelet Phase**: {tie_result['phase_deg']}°",
        f"- **Bulk Time Shift Applied**: {tie_result['shift_ms']:.1f} ms",
        f"- **Polarity Convention**: Seismic {polarity_desc}",
        "",
        "## 3. Geophysical Interpretation",
    ]

    if quality == "GOOD":
        lines.append("The synthetic trace demonstrates a robust correlation with the actual seismic reflections. "
                     "This indicates the time-depth relationship (DPTM curve) and logging measurements are highly reliable. "
                     "This well-tie is safe to use for reservoir modeling and horizontal mapping.")
    elif quality == "FAIR":
        lines.append("A reasonable match is observed, though there are minor differences. This could be due to "
                     "slight log washouts or noise in the seismic dataset. Use this tie with caution for fine-scale mapping.")
    else:
        lines.append("Poor match detected. Do not use this tie for depth conversion without further manual calibration.")

    lines.extend([
        "",
        "## 4. Input Well Logs Info",
        f"- **Available Curves in LAS**: `{', '.join(curves)}`",
        "",
        "---",
        "*Report generated automatically as part of the understanding pipeline.*"
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_project_summary(out_path, results):
    """Write project summary report in Markdown."""
    lines = [
        "# Well-to-Seismic Tie Project Summary Report",
        f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This project summary contains the results of matching the geology from well logs to 3D seismic reflections.",
        "",
        "## Tie Results Table",
        "",
        "| Well ID | Inline | Crossline | Offset (m) | Correlation | Quality | Freq (Hz) | Phase (°) | Shift (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in results:
        lines.append(
            f"| **{r['well']}** | {r['inline']} | {r['crossline']} | {r['distance_m']:.1f} | "
            f"{r['correlation']:.3f} | **{r['quality']}** | {r['freq_hz']} | {r['phase_deg']} | {r['shift_ms']:.1f} |"
        )

    lines.extend([
        "",
        "## Summary Details",
        f"- Total Wells Tied: **{len(results)}**",
        f"- Average Correlation Coefficient: **{np.mean([abs(r['correlation']) for r in results]):.3f}**",
        "",
        "Refer to each well's individual report file and plots inside this folder for specific details."
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("Initializing seismic data...")
    seismic = SeismicVolume(segy_path)
    print(f"Loaded SEG-Y: {seismic.n_traces} traces, dt = {seismic.dt_ms} ms, time range: {seismic.times_ms[0]} to {seismic.times_ms[-1]} ms.")

    results_summary = []

    for well_name, well_cfg in WELLS.items():
        print(f"\nProcessing well: {well_name}...")
        
        # 1. Convert well coordinates to meters to search
        x_m = well_cfg["x"] * FEET_TO_METERS
        y_m = well_cfg["y"] * FEET_TO_METERS
        kb_m = well_cfg["kb"] * FEET_TO_METERS

        # 2. Get closest trace
        trace_idx, dist_m, inline, crossline = seismic.nearest_trace(x_m, y_m)
        real_trace = seismic.trace(trace_idx)

        # 3. Load well logs and build reflectivity
        las_path = os.path.join(las_dir, f"{well_name}.las")
        try:
            well_logs = load_well_logs(las_path)
        except Exception as e:
            print(f"[SKIP] Well {well_name} failed log load: {e}")
            continue

        refl, refl_time, impedance = build_reflectivity(well_logs)
        _, well_time_ms = get_time_depth_curve(well_logs)

        # 4. Resample onto seismic regular time grid (2ms)
        reg_time, refl_reg = resample_reflectivity(refl, refl_time, seismic.dt_ms)

        # 5. Grid search for optimal well tie
        tie_result = find_best_tie(refl_reg, reg_time, real_trace, seismic.times_ms, seismic.dt_ms)

        # 6. Recursive inversion
        inv_impedance = recursive_inversion(real_trace, seismic.times_ms, tie_result, impedance, well_time_ms)

        # 7. Generate outputs
        well_output_prefix = os.path.join(output_dir, well_name)
        os.makedirs(os.path.dirname(well_output_prefix), exist_ok=True)

        tie_plot_path = f"{well_output_prefix}_tie.png"
        inv_plot_path = f"{well_output_prefix}_inversion.png"
        report_path = f"{well_output_prefix}_report.md"

        plot_tie(well_name, refl_reg, reg_time, tie_result, tie_plot_path)
        plot_inversion(well_name, seismic.times_ms, real_trace, inv_impedance, impedance, well_time_ms, inv_plot_path)
        
        location_info = {"x": x_m, "y": y_m, "kb": kb_m, "inline": inline, "crossline": crossline, "distance_m": dist_m}
        write_well_report(well_name, report_path, location_info, tie_result, well_logs["curves_available"])

        print(f"[SUCCESS] Well {well_name} tied: corr = {tie_result['correlation']:.3f} | shift = {tie_result['shift_ms']:.1f}ms")

        results_summary.append({
            "well": well_name,
            "inline": inline,
            "crossline": crossline,
            "distance_m": dist_m,
            "correlation": tie_result["correlation"],
            "quality": tie_quality_label(abs(tie_result["correlation"])),
            "freq_hz": tie_result["freq_hz"],
            "phase_deg": tie_result["phase_deg"],
            "shift_ms": tie_result["shift_ms"]
        })

    # Close volume
    seismic.close()

    # Save summary files
    csv_path = os.path.join(output_dir, "tie_summary.csv")
    if results_summary:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results_summary[0].keys()))
            writer.writeheader()
            writer.writerows(results_summary)

    write_project_summary(os.path.join(output_dir, "PROJECT_SUMMARY.md"), results_summary)
    print("\nWell-to-Seismic tie processing complete for all wells!")


if __name__ == "__main__":
    main()
