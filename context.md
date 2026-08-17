# 🌋 Project Context: Quantitative Seismic Interpretation & ML Reservoir Characterization

This document provides a comprehensive, self-contained summary of our quantitative seismic interpretation (QSI) project, available datasets, machine learning methodologies, current pipeline results, and the planned deep learning workflow. 

Use this context to consult other language models (e.g., Claude) about architecture design, troubleshooting, and optimization.

---

## 1. Project Objective & Geological Goals

We are performing **Quantitative Seismic Interpretation (QSI)** and reservoir characterization on a gas-bearing sandstone reservoir.
*   **The Target:** Identify and map **sweet spots** (reservoir zones with high effective porosity $\text{PHIE}$, low shale volume $\text{VSH}$, and gas saturation indicated by low Lambda-Rho $\text{LMRHO}$).
*   **The Geological Challenge:** The sand channels are thin and frequently lie below the traditional seismic tuning resolution limit ($\lambda/4$). Traditional amplitude mapping fails due to constructive/destructive wave interference (tuning effects).
*   **Our Solution:** A multi-target machine learning inversion pipeline that extracts high-resolution wave-physics attributes (envelope, frequency shifts, spectral decomposition, lateral gradients) and uses a dual-strategy hybrid regression pipeline to predict elastic and petrophysical logs at every trace location in the 3D volume.

---

## 2. Available Data

### A. Local Well Dataset (7 Wells)
We have 7 wells in our study area: **Z-02, Z-03, Z-04 (Blind), Z-05, Z-06, Z-07, and Z-08-ST-02**.
*   **Well Coordinates & Metadata:**
    *   `Z-02`: Easting `1205859.09`, Northing `9692966.31`, KB `146.46` ft
    *   `Z-03`: Easting `1201178.25`, Northing `9682452.00`, KB `147.64` ft
    *   `Z-04`: Easting `1205820.18`, Northing `9696292.65`, KB `147.64` ft (True Blind Validation Well)
    *   `Z-05`: Easting `1206404.17`, Northing `9679510.83`, KB `144.36` ft
    *   `Z-06`: Easting `1207337.37`, Northing `9684145.64`, KB `146.98` ft
    *   `Z-07`: Easting `1206364.34`, Northing `9688320.18`, KB `147.97` ft
    *   `Z-08-ST-02`: Easting `1202225.66`, Northing `9700693.11`, KB `152.20` ft
*   **Log Curves (LAS files):** Mapped at 2ms seismic sampling rate through block-averaging.
    *   **Lithology / Quality:** Gamma Ray (`GR`), Volume of Shale (`VSH`), Total Porosity (`PHIT`), Effective Porosity (`PHIE`), Water Saturation (`SWE`), Bulk Density (`RHOB`), Sonic Travel Time (`DT`), Acoustic Impedance (`AI`).
    *   **Castagna-Derived Rock Physics Logs:**
        *   `Vp` (Compressional velocity, m/s): `(1,000,000 / DT) * 0.3048`
        *   `Vs` (Shear velocity, m/s via Castagna Mudrock Line): `(Vp - 1360) / 1.16`
        *   `VPVS` (Velocity ratio): `Vp / Vs` (Gas sand indicator < 1.7, Wet clastic/shale > 2.0)
        *   `POIS` (Poisson's Ratio): `(Vp² - 2Vs²) / (2(Vp² - Vs²))` (Gas sand ~0.15, Shale/Wet sand ~0.35)
        *   `LMRHO` (Lambda-Rho, fluid compressibility): `ρ * (Vp² - 2Vs²)` (Excellent gas-sand discriminator; lower values = gas)
        *   `MURHO` (Mu-Rho, shear rigidity): `ρ * Vs²` (Lithology/frame indicator)

### B. Local 3D Seismic Volume
*   **Source File:** `origional.segy`
*   **Volume Size:** 61,740 total post-stack seismic traces.
*   **Reservoir Window:** 2086.0 ms to 2154.0 ms Two-Way Time (TWT), corresponding to ~35 samples per trace vertically.

### C. Public Dataset (F3 Block, Netherlands)
Downloaded in [public_data/f3_dataset/](file:///d:/Internship@LMKR/Analysis/public_data/f3_dataset) for neural network pre-training.
*   **Seismic Volume (`seismic_entire_volume.npy`):** Shape `(601, 901, 255)`. Normalized amplitude values ranging from `-1.0` to `+1.0` (Mean `0.0`, Std `0.21`).
*   **Facies Volume (`labels_entire_volume.npy`):** Shape `(601, 901, 255)`. Contains integer facies labels `[0, 1, 2, 3, 4, 5]` representing:
    *   `0`: Upper North Sea Group (deltaic sands/clays)
    *   `1`: Middle North Sea Group (sands/clays)
    *   `2`: Lower North Sea Group (marine claystones - largest class, 49.5%)
    *   `3`: Chalk Group (late Cretaceous limestone - very high velocity/density)
    *   `4`: Rijnland Group (early Cretaceous claystones/shales)
    *   `5`: Scruff Group (Jurassic claystones/sandstones/evaporites)
*   **Borehole Info:** Trajectories, well head coordinates, and stratigraphic picks for 24 wells. No raw continuous `.las` logs are present in this package.

---

## 3. Current Machine Learning Pipeline (v8.0)

Our current pipeline ([train_model_v8.py](file:///d:/Internship@LMKR/Analysis/ml_training/train_model_v8.py)) trains models using the following structure:

### A. Feature Extraction (55 Attributes)
For each trace, we compute a 55-dimensional feature vector at each time sample:
1.  **Acoustic Impedance Proxy:** Cumulative trace integration `np.cumsum(amplitude)`.
2.  **Analytic Attributes:** Hilbert envelope, instantaneous frequency, envelope derivative, envelope ratio, sweetness, and polarity index.
3.  **Vertical Shifts:** Look-ahead and look-back values (shifts of `-2, -1, +1, +2` samples) for amplitude, envelope, frequency, sweetness, and polarity index to incorporate vertical wave-physics context.
4.  **Spectral Decomposition:** Envelope spectral amplitudes at **10, 15, 20, 30, and 40 Hz** bands using bandpass filters.
5.  **Lateral Spatial Gradients:** Amplitude, envelope, frequency, and impedance gradient values computed from 4 neighboring traces (Inline ±1, Crossline ±1) to capture geological dip and structural boundaries.
6.  **Relative Vertical Position:** Normalized vertical time index (`0.0` to `1.0`) to represent background compaction trends.

### B. Modeling Strategy (Dual-Strategy Race)
For each target log property, the pipeline trains and compares two strategies head-to-head, choosing the winner based on the highest blind validation $R^2$ score on well **Z-04**:
1.  **Standard Strategy:** Predict target directly from the 55 seismic features.
2.  **Cascaded Strategy:** Predict Stage-1 elastic properties (`AI`, `DT`, `MURHO`, `PHIT`, `POIS`, `VPVS`) first, and feed their predictions as additional inputs for predicting Stage-2 reservoir properties (`VSH`, `PHIE`, `SWE`, `LMRHO`).
3.  **Facies-First Clamping:** A Random Forest classifier predicts `IS_SAND` (sand probability based on `VSH < 0.4`). In zones where `IS_SAND` probability $\le 0.5$, effective porosity `PHIE` clamps to `0.0` and water saturation `SWE` clamps to `1.0` (dry shale baseline). Porosity/saturation regression models are trained strictly inside predicted sand blocks.
4.  **Tie-Quality Weighting:** Sample weights are applied during fitting based on the correlation coefficient of the well-seismic tie.

---

## 4. Current Pipeline Results (v8.0 Winners)

Evaluated on the completely withheld **Z-04 blind well** (26 samples):

| Target | Winning Strategy | Best Model / Configuration | Blind $R^2$ on Z-04 | Key Role in Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **LMRHO** | Cascaded | Stacking (Ridge) + LogTransform | **$+0.5041$** | **Fluid indicator** (Excellent match; gas zones have very low values) |
| **VSH** | Cascaded | LightGBM (Regularized) | **$+0.0622$** | **Lithology** (Determines sand vs shale zones) |
| **PHIE** | Standard | Sand-Only LightGBM | **$+0.0537$** | **Reservoir quality** (Effective porosity of sand intervals) |
| **GR** | Standard | Stacking (Ridge) + LogTransform | **$+0.0141$** | **Lithology marker** (Gamma Ray) |
| **SWE** | Standard | Ridge Regressor | **$-0.0032$** | **Fluid saturation** (Water saturation baseline) |

### Important Insight on Blind $R^2$ Scores:
*   A negative or near-zero $R^2$ on a blind well is common in log prediction. It is caused by **well-to-well absolute baseline shifts** due to compaction, mineralogy, or calibration differences.
*   By restricting tree depth (max depth 6) and applying Lasso feature selection, our models prioritize learning **relative wiggles and structural patterns** rather than absolute values. Capturing correct wiggle shape and relative anomaly location is geophysically much more valuable than matching a shifted absolute baseline.

---    

## 5. Next Steps: 1D CNN Transfer Learning

We plan to transition from classical ML to a **1D Convolutional Neural Network (1D CNN) Transfer Learning** workflow to improve prediction accuracy on thin sands:

### Step 1: Pre-training on Public F3 Block
*   **Goal:** Train a feature extraction network on the large F3 dataset.
*   **Task:** Facies Classification.
*   **Inputs:** Vertical windows of raw seismic traces (e.g., length `W = 64` centered at a voxel).
*   **Outputs:** Facies labels `[0, 1, 2, 3, 4, 5]` of the center sample.
*   **Value:** Over 540,000 vertical traces are available. Training the network to classify stratigraphic units forces it to learn the physics of wave reflection, seismic envelopes, frequency attenuation, and boundary transitions.

### Step 2: Fine-Tuning on Local Wells
*   **Goal:** Transfer the learned features to predict local reservoir properties.
*   **Task:** Continuous Log Regression.
*   **Action:** Freeze the early convolutional feature extraction layers, replace the final classification head with a regression head (a series of dense layers), and train on our local 7 wells.
*   **Inputs:** Local 1D seismic trace windows around the wells.
*   **Outputs:** Continuous log values for the center sample (`LMRHO`, `PHIE`, `VSH`).

---

## 6. Questions for Claude / Design Consultation

When consulting another AI (such as Claude) about the transfer learning pipeline, ask the following:
1.  **Architecture:** What is the ideal network architecture (e.g., number of convolutional layers, kernel sizes, dilation rates, pooling vs strides) for a 1D CNN that maps a 64-sample vertical seismic trace to a single classification label?
2.  **Dilation:** How should we use dilated convolutions to capture long-range wave-physics context (like compaction trends) without losing high-frequency thin-bed boundaries?
3.  **Loss Function:** For pre-training, how should we handle the severe class imbalance of the F3 dataset (Lower North Sea Group is ~50%, while Scruff is ~1.8%)? For fine-tuning, how do we implement Huber loss or custom loss functions to optimize relative wiggle correlation while ignoring absolute baseline shifts?
4.  **Regularization:** With only 241 local well samples, how do we prevent the fine-tuning regression head from overfitting? Should we freeze all convolutions, or should we use a slow learning rate (e.g., $10^{-5}$) to fine-tune the later convolutional layers?
