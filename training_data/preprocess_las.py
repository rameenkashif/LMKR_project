"""
preprocess_las.py - Data Quality Control for LAS log files
==========================================================
1. Reads all raw LAS files from `las/`.
2. Flags DT values outside [35, 140] us/ft as invalid (NaN).
3. Linearly interpolates flagged DT depths.
4. Recomputes derived rock physics properties:
   - Vp (Compressional velocity, m/s)
   - Vs (Shear velocity, m/s)
   - AI (Acoustic Impedance, m/s * g/cc)
   - VPVS (Vp/Vs ratio)
   - POIS (Poisson's ratio)
   - LMRHO (Lambda-Rho, g/cc * km^2/s^2)
   - MURHO (Mu-Rho, g/cc * km^2/s^2)
5. Saves cleaned LAS files to `las_cleaned/` directory.
"""
import os
import numpy as np
import pandas as pd
import lasio

script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, ".."))
las_dir = os.path.join(analysis_dir, "las")
las_cleaned_dir = os.path.join(analysis_dir, "las_cleaned")
os.makedirs(las_cleaned_dir, exist_ok=True)

# List of LAS files
las_files = [f for f in os.listdir(las_dir) if f.endswith(".las")]

print("=== Starting LAS Data Quality Control Preprocessing ===")

for filename in las_files:
    well_name = filename.replace(".las", "")
    las_path = os.path.join(las_dir, filename)
    cleaned_path = os.path.join(las_cleaned_dir, filename)

    print(f"\nProcessing well: {well_name}")
    las = lasio.read(las_path)
    df = las.df()

    if "DT" not in df.columns or "RHOB" not in df.columns:
        print(f"  [SKIP] Required DT/RHOB curves missing. Copying original file to cleaned/.")
        las.write(cleaned_path)
        continue

    # Extract curves
    dt_raw = df["DT"].values.astype(float)
    rhob = df["RHOB"].values.astype(float)

    # 1. Identify flagged points
    # Sane range: [35, 140] us/ft
    dt_invalid_mask = (dt_raw < 35.0) | (dt_raw > 140.0) | np.isnan(dt_raw)
    num_flagged = np.sum(dt_invalid_mask & ~np.isnan(dt_raw))
    print(f"  Flagged {num_flagged} invalid DT samples out of {len(dt_raw)} total samples.")

    # 2. Check for other simultaneous anomalies (Z-07 specific logging/audit)
    if well_name == "Z-07" and num_flagged > 0:
        print("  [AUDIT] Z-07 Anomaly Zone Curve Inspection:")
        anom_idx = np.where(dt_invalid_mask)[0]
        # Inspect other columns at these index depths
        inspect_cols = [c for c in ["RHOB", "VSH", "PHIE", "SWE", "GR"] if c in df.columns]
        for idx in anom_idx:
            depth = df.index[idx]
            vals = {c: df.iloc[idx][c] for c in inspect_cols}
            print(f"    Depth: {depth:.4f}m | DT: {dt_raw[idx]:.2f} | " + " | ".join(f"{k}: {v:.4f}" for k, v in vals.items()))

    # 3. Clean DT using linear interpolation
    df_clean = df.copy()
    # Replace invalid values with NaN
    df_clean.loc[dt_invalid_mask, "DT"] = np.nan
    # Interpolate
    df_clean["DT"] = df_clean["DT"].interpolate(method="linear").ffill().bfill()
    dt_clean = df_clean["DT"].values

    # 4. Recompute derived rock physics properties
    # Vp calculation (m/s)
    vp = (1000000.0 / np.clip(dt_clean, 35.0, 140.0)) * 0.3048
    
    # Vs estimation from Castagna mudrock line (m/s)
    vs = (vp - 1360.0) / 1.16
    vs = np.clip(vs, 100.0, None)
    
    # VPVS
    vpvs = np.divide(vp, vs, out=np.zeros_like(vp), where=vs>0)
    
    # POIS
    vp2 = vp ** 2
    vs2 = vs ** 2
    den = 2.0 * (vp2 - vs2)
    pois = np.divide(vp2 - 2.0 * vs2, den, out=np.zeros_like(vp), where=den>0)
    pois = np.clip(pois, -1.0, 0.5)

    # LMRHO and MURHO
    rhob_g_cc = np.clip(rhob, 0.1, 5.0)
    vp_km_s = vp / 1000.0
    vs_km_s = vs / 1000.0
    lmrho = rhob_g_cc * (vp_km_s**2 - 2.0 * vs_km_s**2)
    murho = rhob_g_cc * (vs_km_s**2)

    # AI
    ai = vp * rhob

    # Update LAS curve helper
    def update_curve(las_obj, name, values, descr):
        if name in las_obj.keys():
            las_obj.curves[name].data = values
        else:
            las_obj.append_curve(name, values, unit="", descr=descr)

    update_curve(las, "DT", dt_clean, "Cleaned sonic transit time (us/ft)")
    update_curve(las, "VPVS", vpvs, "Recomputed Vp/Vs ratio from cleaned DT")
    update_curve(las, "POIS", pois, "Recomputed Poisson ratio from cleaned DT")
    update_curve(las, "LMRHO", lmrho, "Recomputed Lambda-Rho from cleaned DT and RHOB")
    update_curve(las, "MURHO", murho, "Recomputed Mu-Rho from cleaned DT and RHOB")
    update_curve(las, "AI", ai, "Recomputed Acoustic Impedance from cleaned DT and RHOB")

    # Add VP and VS curves to the LAS file for reference
    update_curve(las, "VP", vp, "Recomputed P-velocity (m/s)")
    update_curve(las, "VS", vs, "Recomputed S-velocity (m/s)")

    # Save to the cleaned folder
    las.write(cleaned_path)
    print(f"  [SUCCESS] Wrote cleaned LAS file to {cleaned_path}")

    # Confirm Z-07 LMRHO Standard Deviation drop
    if well_name == "Z-07":
        orig_std = np.std(df["LMRHO"].values)
        new_std = np.std(lmrho)
        print(f"\n[CONFIRMATION] Z-07 LMRHO Standard Deviation:")
        print(f"  Original: {orig_std:.4f} GPa")
        print(f"  Cleaned : {new_std:.4f} GPa")
        print(f"  Outlier removal success: {new_std < 1.0}")
