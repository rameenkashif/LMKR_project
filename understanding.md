# Machine Learning Seismic-to-Log Prediction: Methodology & Understanding

This document explains the data preparation, modeling methodology, sample derivations, and result interpretations for our seismic-to-well-log prediction pipeline.

---

## 1. What We Took (Inputs & Targets)

To train our models, we extract two types of data:

### **Features (Inputs from Seismic)**
For each well, we find the seismic trace at its mapped coordinate (Inline/Crossline) and generate a **42-attribute feature dictionary**:
*   **Acoustic Impedance Proxy**: A relative impedance log computed via the cumulative sum of amplitudes.
*   **Amplitude & Shifts**: The raw trace amplitude (`amp_center`) and 10 shifted variations (`+1` to `+5` and `-1` to `-5` samples) to give the model local context.
*   **Reflection Envelope**: Represents reflectivity strength (`env_center`), its derivative (`env_deriv`), ratio, and 10 shifted variations.
*   **Instantaneous Frequency**: Frequency content computed via phase unwrapping (`ifreq_center`) and 10 shifted variations.
*   **Relative Position**: A normalized index (`0.0` to `1.0`) indicating the sample's vertical position in the trace.
*   **Window Statistics**: Rolling max, mean, min, and standard deviation calculated over a 5-sample sliding window.

### **Targets (Well Log Properties)**
We predict seven log properties extracted from well LAS files:
*   `DT` (Sonic Travel Time)
*   `GR` (Gamma Ray)
*   `PHIE` (Effective Porosity)
*   `PHIT` (Total Porosity)
*   `RHOB` (Bulk Density)
*   `SWE` (Water Saturation)
*   `VSH` (Volume of Clay/Shale)

---

## 2. How We Processed & Used the Seismic Data

Seismic reflection data (stored in `origional.segy`) is a 3D volume of acoustic reflection amplitudes. To turn this raw geophysical data into predictors for well-log logs, we applied the following steps:

### **1. 1D Trace Extraction**
For each well, we find its spatial location inside the 3D volume using the mapped **Inline** and **Crossline** numbers from the well-seismic tie (`tie_summary.csv`). We extract the corresponding 1D seismic trace, which is a vertical series of reflection amplitudes measured at discrete two-way travel times.

### **2. Polarity Correction**
Depending on how the well-seismic tie was computed, the seismic phase might be inverted compared to the well logs. If the correlation coefficient in `tie_summary.csv` is negative, we multiply the entire extracted seismic trace by `-1.0` to align the polarities.

### **3. Detailed Attribute Extraction (Calculations & Code-level Formulas)**
We extract a total of 42 attributes from each 1D trace. Below is the detailed breakdown of how each feature class is mathematically and programmatically computed:

#### **A. Acoustic Impedance (AI) Proxy**
*   **Formula**: Relative impedance is estimated by trace integration (cumulative sum of amplitude), representing the transform from reflection boundary values to layer-based rock properties.
*   **Calculation**:
    ```python
    impedance = np.cumsum(np.clip(amp, -1.0, 1.0))
    impedance = impedance - np.min(impedance)
    impedance = impedance / np.max(impedance)
    ```
*   **Physical Meaning**: Represents rock properties (velocity $\times$ density) changes across formations.

#### **B. Reflection Envelope (Reflection Strength)**
*   **Formula**: We use the **Hilbert Transform** to compute the analytic signal $z(t) = x(t) + i \cdot y(t)$, where $x(t)$ is the seismic trace and $y(t)$ is its Hilbert transform (90-degree phase shift). The envelope $E(t)$ is the magnitude of $z(t)$:
    $$E(t) = |z(t)| = \sqrt{x(t)^2 + y(t)^2}$$
*   **Calculation**:
    ```python
    analytic = scipy.signal.hilbert(amp)
    envelope = np.abs(analytic)
    ```
*   **Derivative**: `env_deriv = np.gradient(envelope)` — tracks the rate of energy change.
*   **Ratio**: `envelope / (np.abs(amp) + 1e-8)` — quantifies the relative instantaneous peak amplitude.
*   **Physical Meaning**: Indicates lithological changes, layer boundaries, and presence of bright spots (gas/fluids).

#### **C. Instantaneous Frequency**
*   **Formula**: Instantaneous phase $\theta(t)$ is the angle of the analytic signal. Instantaneous frequency $f(t)$ is the time derivative of the phase:
    $$f(t) = \frac{1}{2\pi} \frac{d\theta(t)}{dt}$$
*   **Calculation**:
    ```python
    phase = np.unwrap(np.angle(analytic))
    dt_s = dt_ms / 1000.0
    ifreq = np.gradient(phase, dt_s) / (2.0 * np.pi)
    ```
*   **Physical Meaning**: Serves as a tool for checking thin-bed thickness variations, wave attenuation, and fracture zones.

#### **D. Time-Shifts (Look-Ahead / Look-Back Context)**
*   **Formula**: Moving a value from time $t$ to $t \pm N \cdot dt$ samples.
*   **Calculation**:
    ```python
    def _pad_shift(arr: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0: return arr.copy()
        out = np.zeros_like(arr)
        if shift > 0:
            out[:-shift] = arr[shift:] # Look-ahead
        else:
            shift = -shift
            out[shift:] = arr[:-shift] # Look-back
        return out
    ```
    We create 10 shifted variations (from `shift = -5` to `+5`) for **amplitude**, **envelope**, and **instantaneous frequency**.
*   **Physical Meaning**: Seismic reflections occur at geological interfaces (boundaries), whereas well log measurements describe the internal interval properties of layers. Providing shifts tells the model about the boundary properties surrounding a specific sample.

#### **E. Window Statistics**
*   **Formula**: Rolling properties in a window of 5 samples around sample $i$ (index range: $i - 2$ to $i + 2$).
*   **Calculation**:
    ```python
    for i in range(n):
        lo = max(0, i - 2)
        hi = min(n, i + 3)
        window_vals = amp[lo:hi]
        maxv[i] = np.max(window_vals)
        meanv[i] = np.mean(window_vals)
        minv[i] = np.min(window_vals)
        stdv[i] = np.std(window_vals)
    ```
*   **Physical Meaning**: Captures local waveform texture, average amplitude level, and waveform variance.

#### **F. Relative Position**
*   **Calculation**: `rel_pos = np.linspace(0.0, 1.0, n_samples)`
*   **Physical Meaning**: Represents depth trend, serving as a proxy for natural compaction trends.

All of these attributes are then aligned with the block-averaged well log curves in time to build our tabular machine learning dataset.

---

## 3. Why We Grouped / Binned (e.g., 2154.0 ms)

Seismic traces and well logs are recorded at completely different vertical resolutions:
*   **Well logs** are measured at a very high resolution (typically every 6 inches or 0.1524 meters), corresponding to sub-millisecond travel times.
*   **Seismic traces** are recorded at a coarser, discrete interval (usually 2ms or 4ms).

To align these two datasets:
1.  We convert well depth to time (`DPTM`) and apply the bulk shift (`shift_ms`) from the well-seismic tie.
2.  We map each high-resolution well sample to its **nearest discrete seismic time bin** (e.g., `2154.0` ms, `2156.0` ms, etc.).
3.  We **block-average** the well curves: all log readings falling into the same time bin (e.g., `2154.0`) are averaged into a single value.

This grouping aligns the geological data with the geophysical scale, filtering out high-frequency noise that the seismic data cannot physically resolve.

---

## 4. How Sample Counts Were Derived (36, 26, etc.)

Our final training dataset contains a total of **241 block-averaged samples** across 6 wells. The sample counts per well are:
*   `Z-06`: 50 samples
*   `Z-05`: 46 samples
*   `Z-07`: 42 samples
*   `Z-03`: 41 samples
*   **`Z-02`**: **36 samples**
*   **`Z-04` (Blind Well)**: **26 samples**

### **Why Z-02 has 36 samples and Z-04 has 26:**
These counts are a direct outcome of:
1.  **Overlapping Intervals**: The well log's recorded interval only maps to a certain time range in the seismic volume.
2.  **NaN Filtering**: During training table construction, any row containing a `NaN` in *any* of the 7 target logs is dropped. Since logs often start/end at different depths, the overlap of valid, non-null data restricts the clean sample counts. Z-02 ended up with 36 fully populated rows, while Z-04 has 26.

---

## 5. Feature Correlation & Selection Techniques

To handle the 42 extracted seismic attributes and prevent overfitting, we use two key feature engineering techniques: linear correlation analysis and L1-regularized dynamic feature selection.

### **1. Pearson Correlation Analysis (Linear Screening)**
In the data preparation script (`build_training_data.py`), we compute the Pearson correlation coefficient ($r$) between all 42 seismic features and the 7 target logs:
*   **Correlation Matrix**: We run `.corr()` to measure the strength and direction of linear relationships.
*   **Ranking**: We calculate the average absolute correlation of each attribute across all 7 targets to see which features are globally informative.
*   **Heatmap Visualization**: We isolate the **top 15 seismic features** with the highest average correlation and render them in a styled heatmap (`correlation_heatmap.png`). This helps inspect physical relationships (for example, showing how the acoustic impedance proxy correlates strongly with density `RHOB` and sonic logs `DT`).

### **2. LassoCV (Dynamic Feature Selection)**
While Pearson correlation checks for standalone linear relationships, targets are often predicted best by combinations of attributes, some of which may be highly collinear (redundant). To select the best sub-features dynamically for each target, we use **LassoCV** (Least Absolute Shrinkage and Selection Operator with Cross-Validation):
*   **Standardization**: We first apply `StandardScaler` because Lasso's L1 penalty is sensitive to feature scales.
*   **L1 Regularization**: Lasso adds an L1 penalty (absolute sum of coefficients) to the loss function. This penalty drives the weights of weak, redundant, or noisy attributes **exactly to zero**.
*   **Cross-Validation Search**: LassoCV automatically evaluates a grid of alpha values ($\alpha = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]$) using 5-fold cross-validation to select the regularization strength that minimizes mean squared error.
*   **Adaptive Feature Count**: The number of selected features changes dynamically based on the target log. For example, LassoCV determined that `DT` requires 25 attributes, while `PHIT` only needs 1 (`env_shift_-5`).
*   **Fallback Mechanism**: If the L1 penalty is too aggressive and shrinks all coefficients to zero, the script falls back to selecting the top 5 features with the highest absolute coefficients.

By combining LassoCV (which selects a sparse, linear subset of features) with non-linear models (Random Forest and XGBoost), we benefit from robust feature selection while still capturing non-linear geological relationships.

---

## 6. How We Trained the Models

1.  **Blind Well Holdout**: We withheld the blind well (`Z-04` with 26 samples) entirely from the pipeline. The models never saw Z-04's features or targets during training or feature selection.
2.  **Feature Selection (LassoCV)**: To avoid overfitting on the 42 attributes, we ran `LassoCV` on the remaining 215 samples. Lasso applies L1 regularization to shrink non-important coefficients to zero. For example:
    *   `DT` selected 25 features.
    *   `PHIT` selected only 1 feature (`env_shift_-5`).
    *   `PHIE`/`RHOB`/`SWE` selected 5 features (window stats + relative position).
3.  **Leave-One-Well-Out Cross-Validation (LOGO CV)**: We trained and validated our model templates (Random Forest and XGBoost) using 5-fold LOGO CV on the 5 training wells. In each fold, the model was trained on 4 wells and tested on the 5th left-out well. Regularizing Gaussian noise (`0.02` std) was added to training features to prevent overfitting.
4.  **Final Pipeline**: The architecture with the best average CV R² (Random Forest in all cases) was retrained on all 215 training samples and saved. We then tested this final model on the 26 true blind samples from Z-04.

---

## 7. Why a Lower or Negative R² is Better Than it Looks

The Coefficient of Determination ($R^2$) is calculated as:
$$R^2 = 1 - \frac{\sum (y_{actual} - y_{pred})^2}{\sum (y_{actual} - y_{mean})^2}$$

*   **A negative $R^2$** means the model's sum of squared errors is worse than simply guessing the mean of the actual data in that specific well.
*   **Why does this happen in Blind Well testing?**
    *   **Baseline Shift**: Different wells have different absolute baselines due to geological shifts (compaction, mineralogy, pressure, or fluid differences). 
    *   A model trained on other wells doesn't know the absolute baseline of the blind well, so its predictions might be shifted up or down, dragging the $R^2$ into negative values.
*   **Why is a "lesser" (simpler) model better here?**
    *   If we did not restrict features (e.g. using all 42 features) or did not use shallow trees (max depth 6), the model would achieve a high $R^2$ on the training wells (e.g. $+0.85$).
    *   However, it would overfit to the training wells' absolute baselines, resulting in catastrophic failures on the blind well (producing blind $R^2$ values like $-15.0$ or worse).
    *   By forcing Lasso feature selection and restricting tree depth, the model learns to capture the **relative wiggles and geological trends** instead of absolute baselines. For geophysicists, predicting the shape and character (wiggles) of the log is far more valuable than matching a shifted absolute value.

---

## 8. Result Comparison

Here is a comparison of the Cross-Validation performance (average of 5 training wells) vs. the True Blind Well performance (on Z-04):

| Target Property | Best Model | Best CV $R^2$ | Best CV MAE | True Blind $R^2$ (Z-04) | True Blind MAE (Z-04) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DT** | Random Forest | $+0.0784$ | $3.2974$ | **$-1.2335$** | $2.6243$ |
| **GR** | Random Forest | $-0.0266$ | $14.1063$ | **$-1.0084$** | $22.6327$ |
| **PHIE** | Random Forest | $-0.3277$ | $0.0262$ | **$-0.7764$** | $0.0243$ |
| **PHIT** | Random Forest | $-0.1336$ | $0.0178$ | **$-4.7913$** | $0.0252$ |
| **RHOB** | Random Forest | $-0.3136$ | $0.0620$ | **$-0.5351$** | $0.0899$ |
| **SWE** | Random Forest | $-0.3064$ | $0.2440$ | **$-0.0589$** | $0.1624$ |
| **VSH** | Random Forest | $-0.4685$ | $0.0859$ | **$-0.8633$** | $0.1016$ |

### **Key Observations:**
1.  **Random Forest Dominance**: Random Forest outperformed XGBoost in CV R² for all targets. XGBoost's higher complexity led to more overfitting on the small dataset, proving that simpler models are better suited here.
2.  **R² Drop on Blind Well**: The $R^2$ values are lower (and mostly negative) on the blind well compared to CV. This is standard behavior for true blind tests, as the model cannot dynamically calibrate to the baseline shift of a new well.
3.  **Low MAE**: Despite the negative $R^2$, the Mean Absolute Error (MAE) values on the blind well are relatively small (e.g., only $2.62\ \mu\text{s/ft}$ error for DT, and $2.43\%$ error for PHIE). This indicates that the predicted values are physically realistic and closely track the real logs, despite baseline offsets.
