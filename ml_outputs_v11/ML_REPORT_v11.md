# ML Property Prediction Pipeline: Report V11 (Dynamic Calibration)
Generated: 2026-08-03 11:41:08

## Key Architectural Changes from V10

| Fix | Description | Impact |
| :--- | :--- | :--- |
| **A** | Robust Target Standard Deviation: scales predictions using median of training well standard deviations | Prevents Z-07 scale distortion (4.2x) |
| **B** | Compaction depth-trend baseline: dynamically shifts z-score predictions using time/depth trend curves | Removes baseline DC offsets without logs |
| **C** | Standardized cascade prediction scaling | Removes baseline shifts from intermediate predictions |

## V10 vs V11 Comparison

Blind Well: **Z-04** (never used for any selection decision)

| Target | Sand? | V10 CV R2 | V11 CV R2 | Δ CV | V10 Blind R2 | V11 Blind R2 | Δ Blind |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AI** | NO | N/A | -0.0092 | N/A | N/A | -0.1039 | N/A |
| **DT** | NO | N/A | +0.0637 | N/A | N/A | -0.3874 | N/A |
| **MURHO** | NO | N/A | +0.0382 | N/A | N/A | +0.0026 | N/A |
| **MURHO** | YES | N/A | +0.0458 | N/A | N/A | +0.0797 | N/A |
| **PHIT** | NO | N/A | -0.5354 | N/A | N/A | +0.0855 | N/A |
| **POIS** | NO | N/A | +0.0747 | N/A | N/A | -0.3280 | N/A |
| **POIS** | YES | N/A | +0.0512 | N/A | N/A | -0.1103 | N/A |
| **VPVS** | NO | N/A | +0.0556 | N/A | N/A | -0.6020 | N/A |
| **VPVS** | YES | N/A | +0.0201 | N/A | N/A | -0.3089 | N/A |
| **GR** | NO | N/A | +0.0159 | N/A | N/A | +0.0591 | N/A |
| **RHOB** | NO | N/A | -0.1321 | N/A | N/A | -0.3901 | N/A |
| **VSH** | NO | N/A | -0.1162 | N/A | N/A | -0.0423 | N/A |
| **PHIE** | NO | N/A | -0.3819 | N/A | N/A | +0.0432 | N/A |
| **PHIE** | YES | N/A | -0.3458 | N/A | N/A | +0.0395 | N/A |
| **SWE** | NO | N/A | -0.1530 | N/A | N/A | -0.1711 | N/A |
| **SWE** | YES | N/A | -0.2147 | N/A | N/A | -0.2303 | N/A |
| **LMRHO** | NO | N/A | -0.0256 | N/A | N/A | -0.2948 | N/A |
| **LMRHO** | YES | N/A | -0.0292 | N/A | N/A | -0.6050 | N/A |
| **RHOB** | NO | N/A | -999.0000 | N/A | N/A | -0.9022 | N/A |

## Full V11 Performance Table

| Target | Sand? | Strategy | Best Model | CV R2 | Blind R2 | Facies α |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AI** | NO | Standard | Stacking (Tree) | -0.0092 | -0.1039 | N/A |
| **DT** | NO | Standard | Stacking (Tree) | +0.0637 | -0.3874 | N/A |
| **MURHO** | NO | Standard | Stacking (Ridge) | +0.0382 | +0.0026 | N/A |
| **MURHO** | YES | Standard | Stacking (Ridge) | +0.0458 | +0.0797 | N/A |
| **PHIT** | NO | Standard | Stacking (Ridge) | -0.5354 | +0.0855 | N/A |
| **POIS** | NO | Standard | Stacking (Tree) | +0.0747 | -0.3280 | N/A |
| **POIS** | YES | Standard | Stacking (Ridge) | +0.0512 | -0.1103 | N/A |
| **VPVS** | NO | Standard | Stacking (Tree) | +0.0556 | -0.6020 | N/A |
| **VPVS** | YES | Standard | Stacking (Ridge) | +0.0201 | -0.3089 | N/A |
| **GR** | NO | Standard | Random Forest (Shallow) | +0.0159 | +0.0591 | N/A |
| **RHOB** | NO | Standard | Random Forest (Shallow) | -0.1321 | -0.3901 | N/A |
| **VSH** | NO | Standard | Extra Trees (Shallow) | -0.1162 | -0.0423 | N/A |
| **PHIE** | NO | Standard | Stacking (Tree) | -0.3819 | +0.0432 | N/A |
| **PHIE** | YES | Cascaded | Stacking (Tree) | -0.3458 | +0.0395 | N/A |
| **SWE** | NO | Standard | Random Forest (Shallow) | -0.1530 | -0.1711 | N/A |
| **SWE** | YES | Standard | Random Forest (Shallow) | -0.2147 | -0.2303 | N/A |
| **LMRHO** | NO | Cascaded | Random Forest (Shallow) | -0.0256 | -0.2948 | N/A |
| **LMRHO** | YES | Cascaded | Random Forest (Deep) | -0.0292 | -0.6050 | N/A |
| **RHOB** | NO | Physics | Physics-Derived (AI/Vp) | -999.0000 | -0.9022 | N/A |
