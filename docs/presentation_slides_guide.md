# 🎨 Presentation Deck Visual Guide: Slide 12 & Slide 13
## High-Impact UI Visuals & Dashboard Screenshot Capture Guide

This guide provides exact recommendations and capture instructions for **Slide 12 (Thin-Bed Resolution)** and **Slide 13 (Spectral Explorer)** to build a compelling visual presentation deck.

---

## 🎬 SLIDE 12 — "Helping Geoscientists See Below Seismic Resolution"
> **Focus Tab**: `ThinBedTab.jsx` (Guided Walkthrough & 3D Geobody Extraction)

Rather than showing every step of the 5-stage guided workflow, select these **3 high-impact visual panels**:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ PANEL 1: Noise Filtering (SOF)       │ PANEL 2: Thin-Bed Doublet Attribute  │
│ Raw vs. Structurally-Oriented Filter │ Highlighted Sub-Seismic Channel Bed  │
├──────────────────────────────────────┴──────────────────────────────────────┤
│ PANEL 3: 3D Geobody Extraction Result                                       │
│ 2D Slice Mask + Geobody Metrics Card (Volume m³, Bounding Box, Wells)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 📸 Recommended Screenshots to Capture for Slide 12:

1. **Panel 1: Before / After Structurally-Oriented Filter (SOF) Comparison**
   - **View**: Two-panel section view (Raw amplitude on left vs. SOF-filtered amplitude on right).
   - **Key Visual Detail**: Highlights how 3D Gradient Structure Tensor (GST) filtering cleans random noise while preserving sharp reflector terminations across faults.

2. **Panel 2: Doublet / Terrace Attribute Thin-Bed Detection**
   - **View**: Terrace Zero-Crossing / Inflection Point arc-length difference output section.
   - **Key Visual Detail**: A thin channel sand that appears as a single smeared wave cycle in raw seismic splits into distinct top/base boundary events. *(Add a red callout circle on your slide highlighting the thin-bed event)*.

3. **Panel 3: 3D Geobody Extraction Result (Technically Distinctive Output)**
   - **View**: 2D Crossline slice overlay showing the extracted geobody binary mask + the **Geobody Metrics Card**.
   - **Key Visual Detail**: Ensures the metrics card is clearly visible:
     - **Calculated Volume ($m^3$)**: e.g., $1.42 \times 10^6\text{ m}^3$
     - **Bounding Box**: Inline, Crossline, and TWT depth ranges.
     - **Intersected Wells**: List of boreholes penetrating the geobody (e.g. Z-02, Z-04).

---

## 🎬 SLIDE 13 — "Compare Raw vs. Enhanced Seismic Side-by-Side"
> **Focus Tab**: `SpectralWhiteningTab.jsx` / `SpectralEnhancementExplorerTab.jsx`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MAIN PANEL: Interactive Side-by-Side Section View                            │
│ [Left: Raw Seismic Section]  |  [Right: Enhanced Spectral Section]          │
│ (Interactive Method Selector Pill Bar visible at top)                        │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ INSET 1: Quantitative Frequency Spectrum│ INSET 2: SSWT Time-Frequency Heatmap │
│ Power (dB) vs. Frequency (Hz) Plot   │ Spectrogram-Style Reassigned Ridge  │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### 📸 Recommended Screenshots to Capture for Slide 13:

1. **Main Panel: Full Side-by-Side Interactive View**
   - **View**: Raw seismic amplitude section (left) vs. Enhanced section (right).
   - **Key Visual Detail**: Keep the **Method Selector Pill Bar** visible at the top (showing *CWT Morlet Envelopes*, *SSWT Phase Reassignment*, *Spectral Whitening*, *Dominant Frequency Shift*) so reviewers see it is a live interactive tool.

2. **Inset 1: Quantitative Power Spectrum Plot (Power vs. Frequency)**
   - **View**: Overlaid power spectrum curve ($10\text{--}90\text{ Hz}$).
   - **Key Visual Detail**: Shows raw spectrum (peaking at $30\text{ Hz}$ and rolling off fast) vs. enhanced spectrum (broadened high-frequency bandwidth up to $65\text{ Hz}$).

3. **Inset 2: SSWT Time-Frequency Heatmap (Spectrogram-Style)**
   - **View**: Single-trace SSWT time-frequency energy heatmap.
   - **Key Visual Detail**: Displays instant frequency ridges sharp down to time samples, contrasting visually with broad-band section views.

---

## 💡 Practical Screenshot & Presentation Capture Tips

1. **Use Real, Populated Reservoir Views**:
   - Do **not** capture default/empty states (e.g., no well selected, unpopulated slice).
   - Navigate to **Inline 535 / Crossline 193** or near blind well **Z-04** where active reservoir channels show crisp features before capturing.

2. **Maintain Strict Color Scale & Zoom Consistency**:
   - Keep identical zoom levels ($\pm 50\text{ XL}$) and matching color maps (Seismic Red-White-Blue) across before/after pairs on the same slide.
   - Avoid inconsistent color limits, which undercut visual comparisons.

3. **Capture at High Native Resolution**:
   - Expand your browser window to **1080p or 4K fullscreen** before taking screenshots to prevent blurriness when projected on large presentation displays.

4. **Clean UI Chrome**:
   - Close browser developer tools, console logs, and bookmark bars before capturing to give your screenshots a polished, enterprise-grade software look.
