# SuperDRTtools V1.2

<div align="center">

**A modern GUI toolbox for DRT analysis, K-K validation, EIS masking, comparison, mapping, and batch processing**

*Designed for practical electrochemical impedance spectroscopy workflows with multi-file analysis, project management, and flexible data export.*

![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Python-blue)
![GUI](https://img.shields.io/badge/interface-PyQt5-2C7BE5)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

</div>

---

## Overview

**SuperDRTtools** is an enhanced graphical toolbox for analyzing **electrochemical impedance spectroscopy (EIS)** data using the **distribution of relaxation times (DRT)** framework. It extends the original pyDRTtools workflow with multi-file import, Kramers-Kronig validation, masking, multiprocessing-based batch fitting, project-level saving and loading, interactive DRT comparison, DRT maps, and configurable EIS export.

The software is intended to make routine EIS analysis more efficient by combining data-quality inspection, fitting, visualization, comparison, and export in a single desktop workflow.

---

## Highlights

<table>
<tr>
<td width="50%" valign="top">

### DRT and fitting
- Simple, Bayesian, and Hilbert-transform workflows
- Tikhonov-based DRT computation
- Automatic or user-selected peak deconvolution with Peak One / Peak All
- Peak position, frequency, resistance contribution, fraction, and FWHM results
- Fit One and multiprocessing-based Fit All
- Live fitting progress in the status bar

</td>
<td width="50%" valign="top">

### EIS quality control
- K-K validation using `linKK`
- K-K residual visualization
- Manual, box-selection, and rule-based masking
- Apply masking criteria to one file or all files
- Residual-aware EIS inspection

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Multi-file workflow
- Multi-file `.csv` / `.txt` import
- Drag-and-drop support for EIS data and `.sdrtp` project files
- Independent multi-window workspaces
- Automatic handling of common impedance-sign conventions
- File reordering, status tracking, and renaming
- Project open, save, overwrite, and save-as workflows

</td>
<td width="50%" valign="top">

### Visualization and export
- EIS, magnitude, phase, real, and imaginary views
- DRT residual, DRT comparison, DRT Map, and Peak Comparison
- Linked file/curve selection in DRT comparison
- Configurable DRT and EIS data export
- Separate or merged CSV output

</td>
</tr>
</table>

---

## Screenshots

### Main interface
![Main UI](docs/images/main-ui-placeholder.png)

### EIS data, K-K fitting, residual inspection, and masking
![EIS View](docs/images/eis-view-placeholder.png)

### DRT comparison and DRT Map
![DRT Comparison](docs/images/drt-views-comparison.png)
![DRT Map](docs/images/drt-views-MAP.png)

---

## Main features

### EIS import and file management

- Import multiple EIS files in one session.
- Support common `.csv` and `.txt` layouts.
- Drag files directly into the interface.
- Automatically normalize common `Z''` / `-Z''` input conventions.
- Reorder spectra in the Files panel.
- Rename the displayed file name from the right-click menu without changing the source file on disk.

### Lin-KK validation and masking

- Run Kramers-Kronig validation using `linKK`.
- Inspect K-K fitted data and residuals.
- Mask individual points or select multiple points from the EIS plot.
- Apply frequency-range and K-K residual-threshold rules to the current file or all imported files.
- Clear masks for one file or all files.

### DRT fitting

- Run Simple DRT, Bayesian DRT, Hilbert Transform, and peak-analysis workflows.
- Fit one selected spectrum or fit multiple spectra in parallel.
- Fit All uses multiprocessing so independent spectra can be processed concurrently.
- Solver output is suppressed during batch fitting to keep the console readable.
- The bottom status bar reports completed, remaining, skipped, and failed fitting tasks.

### DRT comparison

- Overlay DRT results from multiple files.
- Click a curve or select its file in the Files panel to highlight the corresponding result.
- The selected curve is shown in `#67001F`, while the remaining curves are faded without changing line width.

### Peak analysis

- Select `Auto` to estimate the peak count from the DRT, or choose a count manually.
- Run peak deconvolution for the selected file or all files in background processes.
- Review peak relaxation time, characteristic frequency, resistance contribution, percentage contribution, and FWHM.
- Peak name, relaxation time, and resistance contribution are annotated beside each fitted peak in the DRT plot.
- Double-click a peak annotation to rename only that peak or every matching peak in the project.
- After two or more files have peak results, use **Peak Comparison** to compare named peak resistance contributions in Files-panel order.
- Hover a Peak Comparison point to inspect its file, peak name, and resistance contribution.
- Right-click Peak Comparison to configure an automatic or manual y-axis range.

### DRT Map

- Display multi-file DRT results as a two-dimensional map.
- Preserve the ordering of spectra shown in the Files panel.
- Values above the configured maximum use the maximum color.
- Values below the configured minimum use the minimum color instead of white.

### Project management

The Project card provides a simple save/load workflow:

- **Open Project** loads a saved SuperDRTtools project.
- Dropping a `.sdrtp` file asks whether to open it in this window or a new window.
- **New Window** opens an independent blank workspace without changing the current project.
- **Save Project** creates a project when the current session has not yet been saved.
- When a project is already open, **Save Project** overwrites the current project.
- Right-click the Save button to use **Save Project As...** and preserve a different analysis state.
- Unsaved changes are indicated in the window title and are checked before closing or opening another project.

### EIS export

The EIS export dialog supports configurable output:

- Select exported columns: `Freq`, `Z'`, and `Z''`.
- Select data types: initial data, masked data, Lin-KK fitted data, and DRT reconstructed data.
- Optionally export Lin-KK and DRT residuals.
- Use **Separate** mode to create one CSV per spectrum.
- Use **Merged** mode to combine multiple spectra into one CSV using the displayed file names from the Files panel.
- Exported arrays are aligned to the original frequency sequence; unavailable or masked values are left blank.

### Peak export

- Export the selected file's complete peak metrics as CSV.
- Export all analyzed files to one Excel workbook, with one worksheet per file.
- Reports include peak name, relaxation time, characteristic frequency, height, resistance contribution, percentage contribution, FWHM, and component index.

---

## Available views

- EIS Data
- K-K Residual
- Magnitude
- Phase
- Re Part
- Im Part
- DRT Residual
- DRT
- DRT comparison
- DRT Map
- EIS Score

---

## Recommended workflow

1. Import one or more EIS datasets.
2. Check the inductance setting and input convention.
3. Run K-K validation.
4. Inspect K-K residuals and suspicious points in **EIS Data**.
5. Mask abnormal points when necessary.
6. Re-run K-K validation after changing the retained data.
7. Perform DRT fitting on the retained EIS data.
8. Compare spectra using DRT comparison or DRT Map.
9. Save the project and export the required DRT or EIS results.

> **Important:** DRT fitting uses the current retained EIS data. Lin-KK fitting is used for validation and inspection; it does not replace the DRT input by default.

---

## Installation

### Requirements

- Python >= 3
- A desktop environment capable of running PyQt5

### Recommended Anaconda environment

```bash
conda create --name SuperDRTtools python=3.10 pip ipython pandas matplotlib scikit-learn spyder
conda activate SuperDRTtools
pip install cvxopt PyQt5 impedance pyinstaller
```

Install any additional dependencies required by the original pyDRTtools modules in the same environment.

---

## Quick start

Run the application from the project root:

```bash
python launch.py
```

---

## Project structure

```text
SuperDRTtools/
|-- launch.py
|-- pyDRTtools/
|   |-- app/                 # application bootstrap and main-window composition
|   |-- algorithms/          # numerical DRT, BHT, sampling, and peak algorithms
|   |-- controllers/         # feature-specific GUI workflows
|   |-- infrastructure/      # multiprocessing Fit All and Peak Analysis backends
|   |-- services/            # EIS state, peak metrics, and project storage
|   |-- ui/                  # layout, canvas, widgets, and theme
|   `-- __init__.py          # lazy package-level aliases
|-- docs/
|-- tutorial/
`-- README.md
```

The GUI uses a feature-oriented architecture while the established numerical
algorithm modules remain isolated and unchanged. See
[Architecture](docs/ARCHITECTURE.md) for module responsibilities and dependency
rules.

---

## Packaging to EXE

SuperDRTtools can be packaged as a Windows executable with PyInstaller:

```bash
python -m PyInstaller --noconfirm --clean --windowed --add-data "launchImg/launch.png;launchImg" --splash launchImg\launch.png --onefile --name SuperDRTtools launch.py
```

For the smaller production build, use the separately maintained optimized
specification. The original specification remains available as a conservative
fallback:

```powershell
.\build_optimized.ps1
```

When building from a PyCharm PyInstaller run configuration, select the same
Conda interpreter and use the spec file directly:

```text
SuperDRTtools_optimized.spec --noconfirm --clean
```

The spec locates Conda runtime DLLs from the active interpreter automatically;
the PowerShell helper is optional.

The optimized build removes development/notebook packages, unused Matplotlib
backends, unused Qt modules/plugins, translations, and package source/test
resources. It does not modify numerical algorithms.

When using multiprocessing-based Fit All in a packaged Windows application, keep the application entry point protected by the standard `if __name__ == "__main__":` block.

---

## User manual

For detailed usage guidance, installation notes, and background information, see:

- [SuperDRTtools manual](docs/manual/pyDRTtools_manual.pdf)

---

## Why use SuperDRTtools?

SuperDRTtools is designed for researchers who need to inspect, fit, compare, organize, save, and export real EIS datasets within one interface. It is especially useful for multi-spectrum experiments where data validation, masking, batch DRT fitting, consistent visualization, and reproducible project states are required.

---

## How to cite this work?

[1] Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499.*

Link: https://doi.org/10.1016/j.electacta.2015.09.097

if you want to add more details about standard regularization methods for computing the regularization parameter used in ridge regression, you should also cite the following references:

[2] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the Regularization Parameter in the Distribution of Relaxation Times, Journal of the Electrochemical Society, 170 (2023) 030502.

Link: https://doi.org/10.1149/1945-7111/acbca4

if you are presenting the *Bayesian credible intervals* generated by the pyDRTtools in any of your academic works, you should cite the following references also:

[3] Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454.

Link: https://doi.org/10.1016/j.electacta.2015.03.123

[4] Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129.

Link: https://doi.org/10.1016/j.electacta.2017.07.050

if you are using the pyDRTtools to compute the *Hilbert Transform*, you should cite:

[5] Liu, J., Wan, T. H., & Ciucci, F. (2020).A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores. Electrochimica Acta, 357, 136864.

Link: https://doi.org/10.1016/j.electacta.2020.136864


# References:
1. Ciucci, F. (2020). The Gaussian process Hilbert transform (GP-HT): Testing the Consistency of electrochemical impedance spectroscopy data. Journal of The Electrochemical Society, 167, 12, 126503. [https://doi.org/10.1149/1945-7111/aba937](https://doi.org/10.1149/1945-7111/aba937)
2. Liu, J., Wan, T. H., & Ciucci, F. (2020).A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores. Electrochimica Acta, 357, 136864. [https://doi.org/10.1016/j.electacta.2020.136864](https://doi.org/10.1016/j.electacta.2020.136864)
3. Ciucci, F. (2019). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry, 13, 132-139. [doi.org/10.1016/j.coelec.2018.12.003](https://doi.org/10.1016/j.coelec.2018.12.003)
4. Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482. [doi.org/10.1016/j.electacta.2014.09.058](https://doi.org/10.1016/j.electacta.2014.09.058)
5. Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499. [doi.org/10.1016/j.electacta.2015.09.097](https://doi.org/10.1016/j.electacta.2015.09.097)
6. Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: a Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454. [doi.org/10.1016/j.electacta.2015.03.123](https://doi.org/10.1016/j.electacta.2015.03.123)
7. Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129. [doi.org/10.1016/j.electacta.2017.07.050](https://doi.org/10.1016/j.electacta.2017.07.050)
8. Liu, J., & Ciucci, F. (2019). The Gaussian process distribution of relaxation times: a machine learning tool for the analysis and prediction of electrochemical impedance spectroscopy data. Electrochimica Acta, 135316. [doi.org/10.1016/j.electacta.2019.135316](https://doi.org/10.1016/j.electacta.2019.135316)
9. Liu, J., & Ciucci, F. (2020). The deep-prior distribution of relaxation times. Journal of The Electrochemical Society, 167(2), 026506. [10.1149/1945-7111/ab631a](https://iopscience.iop.org/article/10.1149/1945-7111/ab631a/meta)
10. A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the Regularization Parameter in the Distribution of Relaxation Times, Journal of the Electrochemical Society, 170 (2023) 030502.
Link: https://doi.org/10.1149/1945-7111/acbca4


**How to get support?**

Just write to francesco.ciucci@ust.hk or francesco.ciucci@uni-bayreuth.de
