# Bioimpedance COMSOL Analysis

Analysis and visualization of bioimpedance data from COMSOL simulations comparing **normal** and **cancer** tissue using a 2-electrode configuration.

## Overview

This project analyzes electrical impedance and potential distribution data exported from COMSOL Multiphysics 6.2 simulations. The analysis compares bioelectrical properties between normal and cancerous tissue at 50 kHz excitation frequency.

## Data

The `Data/` folder contains COMSOL export files:

| File | Description |
|------|-------------|
| `Normal.csv` | Simulation results for normal tissue |
| `Cancer.csv` | Simulation results for cancerous tissue |

Each file contains **13,552 mesh nodes** with:
- 3D coordinates (x, y, z) in meters
- Complex impedance Z₁₁ (Ω) at 50 kHz
- Complex electric potential V (V) at 50 kHz

## Notebook: `bioimpedance_analysis.ipynb`

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy
```

### Contents

#### 1. Data Loading & Preprocessing
- Custom parser for COMSOL complex number format
- Handles scientific notation (e.g., `1.23E-8i`)
- Extracts magnitude, phase, real/imaginary components

#### 2. Data Exploration
- Statistical summaries (mean, std, min, max)
- Geometry bounds visualization
- Sample data inspection

#### 3. Impedance Analysis
- **Magnitude/Phase distributions** - Histograms comparing normal vs cancer
- **Nyquist plot (Cole-Cole)** - Complex impedance visualization
- **Box plots** - Statistical comparison of impedance values

#### 4. Electric Potential Visualization
- **3D scatter plots** - Full spatial distribution of potential
- **2D contour slices** - Cross-sections at Z = -2, -4.5, -7 m

#### 5. Signal Processing
- **Line profiles** - Potential variation along electrode axis
- **Spatial gradients** - Electric field approximation (dV/dx)
- **FFT analysis** - Spatial frequency content

#### 6. Normal vs Cancer Comparison
- Side-by-side bar charts and histograms
- Complex phasor diagrams
- Regional analysis by quadrant
- Summary statistics table

#### 7. Statistical Analysis
- **t-test** - Parametric comparison
- **Mann-Whitney U test** - Non-parametric comparison
- **Cohen's d** - Effect size calculation
- **Correlation heatmaps** - Variable relationships

### Generated Figures

The notebook saves the following figures:

| Figure | Description |
|--------|-------------|
| `impedance_analysis.png` | Impedance magnitude, phase, Nyquist plot |
| `3d_potential.png` | 3D electric potential distribution |
| `2d_slices.png` | 2D contour plots at different depths |
| `signal_processing.png` | Line profiles and gradients |
| `fft_analysis.png` | Spatial FFT analysis |
| `comparison_summary.png` | Comprehensive comparison figure |
| `correlation_analysis.png` | Correlation heatmaps |

## Key Findings

| Metric | Normal Tissue | Cancer Tissue | Ratio |
|--------|---------------|---------------|-------|
| Impedance \|Z\| | 0.179 Ω | 0.0072 Ω | ~25x |
| Phase ∠Z | -3.98° | -0.00016° | - |
| Mean \|V\| | 35.7 µV | 1.44 µV | ~25x |

### Interpretation

Cancer tissue exhibits **significantly lower impedance** compared to normal tissue. This is consistent with established bioimpedance research:

- **Altered membrane properties** - Cancer cells have different membrane capacitance
- **Increased water content** - Higher intracellular and extracellular fluid
- **Modified extracellular matrix** - Structural changes in tissue architecture
- **Higher ionic conductivity** - Increased ion mobility

These differences make bioimpedance a promising technique for cancer detection and tissue characterization.

## COMSOL Model Details

- **Software**: COMSOL Multiphysics 6.2.0.290
- **Dimension**: 3D
- **Mesh nodes**: 13,552
- **Frequency**: 50 kHz
- **Physics**: Electric Currents (ec)
- **Electrode configuration**: 2-electrode system

## Usage

1. Clone the repository
2. Ensure data files are in `Data/` folder
3. Open `bioimpedance_analysis.ipynb` in Jupyter
4. Run all cells to generate analysis and figures

```bash
jupyter notebook bioimpedance_analysis.ipynb
```

## License

MIT License
