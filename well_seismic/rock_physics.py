import os
import lasio
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
las_dir = os.path.abspath(os.path.join(script_dir, "..", "las"))
frontend_dir = os.path.abspath(os.path.join(script_dir, "..", "frontend", "public", "well_images"))
os.makedirs(frontend_dir, exist_ok=True)

# Well lists
wells = ["Z-02", "Z-03", "Z-04", "Z-05", "Z-06", "Z-07", "Z-08"]

def process_well_rock_physics(well_name):
    las_path = os.path.join(las_dir, f"{well_name}.las")
    if not os.path.exists(las_path):
        print(f"[SKIP] Well {well_name} LAS file not found at {las_path}")
        return

    print(f"Processing Rock Physics for Well: {well_name}")
    las = lasio.read(las_path)
    df = las.df()

    # Verify DT and RHOB exist
    if "DT" not in df.columns or "RHOB" not in df.columns:
        print(f"[SKIP] Well {well_name} does not have required DT or RHOB curves.")
        return

    # Extract curves
    dt = df["DT"].values.astype(float) # us/ft
    rhob = df["RHOB"].values.astype(float) # g/cc
    gr = df["GR"].values.astype(float) if "GR" in df.columns else np.zeros_like(dt)

    # Vp calculation (m/s)
    # Vp (m/s) = (1,000,000 / DT_us_ft) * 0.3048
    vp = (1000000.0 / np.clip(dt, 10.0, 300.0)) * 0.3048
    vp = np.nan_to_num(vp, nan=0.0)

    # Vs estimation from Castagna mudrock line (m/s)
    # Vs = (Vp - 1360) / 1.16
    vs = (vp - 1360.0) / 1.16
    vs = np.clip(vs, 100.0, None) # Floor shear velocity to 100 m/s
    vs = np.nan_to_num(vs, nan=0.0)

    # Compute rock physics curves
    # 1. Vp/Vs ratio
    vpvs = np.divide(vp, vs, out=np.zeros_like(vp), where=vs>0)
    vpvs = np.nan_to_num(vpvs, nan=0.0)

    # 2. Poisson's ratio
    # pois = (vp^2 - 2*vs^2) / (2*(vp^2 - vs^2))
    vp2 = vp ** 2
    vs2 = vs ** 2
    den = 2.0 * (vp2 - vs2)
    pois = np.divide(vp2 - 2.0 * vs2, den, out=np.zeros_like(vp), where=den>0)
    pois = np.clip(pois, -1.0, 0.5)
    pois = np.nan_to_num(pois, nan=0.0)

    # 3. Lambda-Rho (LMRHO)
    # lambda_rho = rhob * (vp^2 - 2*vs^2)
    # Convert rhob to g/cc (already is) and velocities to km/s to prevent huge numbers
    rhob_g_cc = np.clip(rhob, 0.1, 5.0)
    vp_km_s = vp / 1000.0
    vs_km_s = vs / 1000.0
    lmrho = rhob_g_cc * (vp_km_s**2 - 2.0 * vs_km_s**2)
    lmrho = np.nan_to_num(lmrho, nan=0.0)

    # 4. Mu-Rho (MURHO)
    # mu_rho = rhob * vs^2 (velocities in km/s)
    murho = rhob_g_cc * (vs_km_s**2)
    murho = np.nan_to_num(murho, nan=0.0)

    # Update LAS file
    def update_las_curve(name, values, descr=""):
        if name in las.keys():
            las[name].data = values
        else:
            las.append_curve(name, values, unit="", descr=descr)

    update_las_curve("VPVS", vpvs, "Vp/Vs ratio derived from DT via Castagna Mudrock line")
    update_las_curve("POIS", pois, "Poisson ratio derived from DT via Castagna Mudrock line")
    update_las_curve("LMRHO", lmrho, "Lambda-Rho derived from DT and RHOB (g/cc * km^2/s^2)")
    update_las_curve("MURHO", murho, "Mu-Rho derived from DT and RHOB (g/cc * km^2/s^2)")

    # Save LAS file
    las.write(las_path)
    print(f"  --> Updated LAS file: {las_path} with 4 new curves.")

    # Create Rock Physics Crossplot (Lambda-Rho vs Mu-Rho) colored by GR
    valid_mask = (lmrho > 0) & (murho > 0) & (gr > 0)
    if np.any(valid_mask):
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(lmrho[valid_mask], murho[valid_mask], c=gr[valid_mask], cmap="jet", alpha=0.7, s=15, vmin=30, vmax=110)
        plt.colorbar(sc, label="Gamma Ray (GR)")
        
        # Add bounds / labels for geologist reference
        plt.title(f"{well_name} Rock Physics Crossplot\n(Lambda-Rho vs Mu-Rho, colored by GR)")
        plt.xlabel(r"Lambda-Rho ($\lambda\rho$ - Fluid indicator)")
        plt.ylabel(r"Mu-Rho ($\mu\rho$ - Frame stiffness)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim(0, max(15, np.max(lmrho[valid_mask])))
        plt.ylim(0, max(15, np.max(murho[valid_mask])))
        
        # Add typical Gas Sand, Brine Sand and Shale zone annotations
        plt.text(2.0, 10.0, "Gas Sand Zone (Low LMRHO)", color="red", fontsize=8, weight="bold")
        plt.text(8.0, 12.0, "Brine Sand Zone", color="blue", fontsize=8, weight="bold")
        plt.text(12.0, 5.0, "Shale Zone", color="black", fontsize=8, weight="bold")
        
        # Save figure to frontend public images
        plot_path = os.path.join(frontend_dir, f"{well_name}_rockphysics.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  --> Saved Rock Physics plot to {plot_path}")

def main():
    print("--- Phase 1: Deriving Rock Physics Logs ---")
    for w in wells:
        try:
            process_well_rock_physics(w)
        except Exception as e:
            print(f"[ERROR] Failed processing well {w}: {e}")

if __name__ == "__main__":
    main()
