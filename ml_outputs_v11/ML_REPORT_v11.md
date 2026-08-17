# ML Property Prediction Pipeline: Report V11 (Dynamic Calibration)
Generated: 2026-08-17 05:42:43

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
| **AI** | NO | N/A | -0.0185 | N/A | N/A | -0.1135 | N/A |
| **DT** | NO | N/A | +0.0858 | N/A | N/A | -0.3540 | N/A |
| **MURHO** | NO | N/A | +0.0557 | N/A | N/A | -0.0353 | N/A |
| **MURHO** | YES | N/A | +0.0524 | N/A | N/A | +0.1084 | N/A |
| **PHIT** | NO | N/A | -0.5392 | N/A | N/A | +0.0759 | N/A |
| **POIS** | NO | N/A | +0.0900 | N/A | N/A | -0.2603 | N/A |
| **POIS** | YES | N/A | +0.0630 | N/A | N/A | -0.4599 | N/A |
| **VPVS** | NO | N/A | +0.0789 | N/A | N/A | -0.4818 | N/A |
| **VPVS** | YES | N/A | +0.0641 | N/A | N/A | -0.7560 | N/A |
| **GR** | NO | N/A | -0.0051 | N/A | N/A | +0.0401 | N/A |
| **RHOB** | NO | N/A | -0.1476 | N/A | N/A | -0.4981 | N/A |
| **VSH** | NO | N/A | -0.1332 | N/A | N/A | -0.2063 | N/A |
| **PHIE** | NO | N/A | -0.3928 | N/A | N/A | +0.0454 | N/A |
| **PHIE** | YES | N/A | -0.3698 | N/A | N/A | +0.0648 | N/A |
| **SWE** | NO | N/A | -0.1916 | N/A | N/A | -0.0320 | N/A |
| **SWE** | YES | N/A | -0.2684 | N/A | N/A | -0.0972 | N/A |
| **LMRHO** | NO | N/A | -0.0719 | N/A | N/A | -0.2715 | N/A |
| **LMRHO** | YES | N/A | -0.0973 | N/A | N/A | -0.2877 | N/A |
| **RHOB** | NO | N/A | -999.0000 | N/A | N/A | -0.8697 | N/A |

## Full V11 Performance Table

| Target | Sand? | Strategy | Best Model | CV R2 | Blind R2 | Facies α |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AI** | NO | Standard | Stacking (Tree) | -0.0185 | -0.1135 | N/A |
| **DT** | NO | Standard | Stacking (Tree) | +0.0858 | -0.3540 | N/A |
| **MURHO** | NO | Standard | Stacking (Tree) | +0.0557 | -0.0353 | N/A |
| **MURHO** | YES | Standard | Stacking (Ridge) | +0.0524 | +0.1084 | N/A |
| **PHIT** | NO | Standard | Stacking (Ridge) | -0.5392 | +0.0759 | N/A |
| **POIS** | NO | Standard | Stacking (Tree) | +0.0900 | -0.2603 | N/A |
| **POIS** | YES | Standard | Extra Trees (Deep) | +0.0630 | -0.4599 | N/A |
| **VPVS** | NO | Standard | Stacking (Tree) | +0.0789 | -0.4818 | N/A |
| **VPVS** | YES | Standard | Extra Trees (Deep) | +0.0641 | -0.7560 | N/A |
| **GR** | NO | Standard | Random Forest (Shallow) | -0.0051 | +0.0401 | N/A |
| **RHOB** | NO | Standard | Extra Trees (Shallow) | -0.1476 | -0.4981 | N/A |
| **VSH** | NO | Cascaded | Stacking (Tree) | -0.1332 | -0.2063 | N/A |
| **PHIE** | NO | Standard | Stacking (Ridge) | -0.3928 | +0.0454 | N/A |
| **PHIE** | YES | Cascaded | Stacking (Tree) | -0.3698 | +0.0648 | N/A |
| **SWE** | NO | Standard | Extra Trees (Shallow) | -0.1916 | -0.0320 | N/A |
| **SWE** | YES | Cascaded | Extra Trees (Deep) | -0.2684 | -0.0972 | N/A |
| **LMRHO** | NO | Standard | Stacking (Tree) | -0.0719 | -0.2715 | N/A |
| **LMRHO** | YES | Cascaded | Stacking (Tree) | -0.0973 | -0.2877 | N/A |
| **RHOB** | NO | Physics | Physics-Derived (AI/Vp) | -999.0000 | -0.8697 | N/A |
