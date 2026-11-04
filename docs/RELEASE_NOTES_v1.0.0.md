# Release notes — v1.0.0

Initial research-software release accompanying the Pacific abalone nursery-growth manuscript.

## Included

- complete single-file analysis implementation;
- monotone GP fitting, LOO density stacking, and residual calibration;
- predictive-mixture median point estimates;
- Figures 1–6 and Tables 1–5 under their manuscript numbers, each with its own entry-point script, and `reproduce_all.py` to regenerate them and compare them with the published values;
- the numbers quoted in the Results text (Section 3) in `results/text/Section3_values.csv`, with its own entry-point script;
- environment specifications and reproducibility notes;
- machine-readable citation metadata and an MIT software license.

## Configuration

- response: log → Yeo–Johnson → standardization;
- point summary: stacked-mixture median;
- Monte Carlo predictive draws: 2,000, including Gaussian observation noise;
- calibration weights: one full-LOO stacking-weight vector shared by all rows;
- calibration center: median of each LOO predictive mixture;
- additive-kernel prior: global-scale × relative-block-scale parameterization.

## Data

The raw Excel workbooks are not included. Authorized users place `전복데이터.xlsx` and `전복 4개세트통합데이터.xlsx` in `input_data/` (see `input_data/README.md`).
