# Reproducibility notes

## Reference configuration

The source comments identify this reference configuration:

- Intel Xeon E5-1660 v4, AVX2, no AVX-512
- Python 3.12 environment
- PyTorch 2.5.1 with CPU computation forced by the script
- GPyTorch 1.15.2
- `OMP_NUM_THREADS=1`
- float64 calculations

The repository environment files pin PyTorch and GPyTorch because they are central to the GP implementation. scikit-learn (1.9.1) and XGBoost (3.4.1) are pinned as well, because the Random forest and XGBoost rows of Table 3 depend on their releases. Python (3.12.16, in `environment.yml`), NumPy (2.5.4), pandas (3.1.0) and SciPy (1.18.1) are pinned to the release run below. matplotlib, seaborn and openpyxl remain flexibly specified.

## Release run

The bundled `results/` were generated and verified in this environment:

- Windows 10 (10.0.19045), x86_64 (Intel Core i5-7400)
- Python 3.12.16
- PyTorch 2.5.1+cpu (the CPU build of PyTorch 2.5.1, as installed from PyPI on Windows) and GPyTorch 1.15.2
- NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1
- `OMP_NUM_THREADS=1`

In this environment every value pinned from the reference configuration reproduces at the printed precision. The Random forest and XGBoost rows of Table 3 were computed in this environment; other operating systems, CPU architectures and builds have not been checked for these two rows. `results/validation/run_checks.csv` records the library versions of the run.

## Expected numerical variation

All fits use recorded seeds, but exact last-bit equality is platform dependent. CPU instructions, BLAS implementations, PyTorch builds, and optimizer tie-breaking can change floating-point accumulation. The source describes agreement around `1e-6` as a portable expectation for the GP calculations. The group-additive kernel has a comparatively flat objective near its optimum, so very small optimizer differences may select a nearby iteration.

Predictive means, standard deviations, and quantiles are Monte Carlo estimates based on 2,000 draws. The script places the draw count in the checkpoint-directory name so checkpoints from different draw counts are not mixed.

## Reproduction workflow

For the included April–November 2024 synthetic inputs, run `python reproduce_all.py --example --workers 3`. This runs the same fitting stages and numbered outputs, requires input and internal-consistency checks, and records manuscript-value differences as informational. Outputs and checkpoints are isolated under `abalone_paper_results/example/<key>/`; the key records both input hashes, analysis and shared-runner hashes, the requirements hash, platform, Python and numerical-package versions. A custom example `--work-dir` is bound to the same metadata and rejects mismatched or unidentified checkpoints. Released `results/` are checked for changes.

The steps below apply to strict manuscript reproduction. Obtain authorized original workbooks and pass their locations through `--data` and `--eda-data` when retaining the bundled synthetic files in `input_data/`.

1. Create the environment from `environment.yml`.
2. Put `전복데이터.xlsx` and `전복 4개세트통합데이터.xlsx` in `input_data/` (see [`input_data/README.md`](../input_data/README.md)).
3. Run `python reproduce_all.py --workers 3`, or a single `scripts/figures/FigureN.py`, `scripts/tables/TableN.py` or `scripts/text/Section3.py`.
4. Read `results/validation/` and the exit status rather than relying only on console output.

`reproduce_all.py` regenerates Figures 1–6, Tables 1–5 and the Section 3 values and then verifies them:

- **Published values.** Every number printed in the manuscript's Tables 1–5 and in Figures 2, 4 and 5 is compared with the regenerated value rounded to the printed digits (`results/validation/manuscript_comparison.csv`). The published values are listed in `PUBLISHED_TABLES` and `PUBLISHED_FIGURES` in `reproduce_all.py`.
- **Internal consistency.** The SHA-256 of `abalone_median.py` before and after the run; the Figure 1–3 design counts; the Figure 4–6 drawn values against the interval calculations and the stacked mixture median; and every written table and text CSV against its builder.
- **Separate processes.** Every numbered script is run again as a separate process from a temporary working directory and compared with the combined run (numbers at `rtol = 0`, `atol = 1e-10`).

`--compare LABEL=DIR` adds a comparison with another result folder. The search for the labels `conformal` and `gamma_0` in PDF text uses `pypdf` when it is installed and is reported as skipped otherwise. The exit status is 1 if any check fails.

Running `python abalone_median.py --point median --workers 3` directly (with `전복데이터.xlsx` in `input_data/` or the repository root) produces the complete internal result set under `abalone_paper_results/point_median/`, including `reproduction_check.csv`, which compares every number the Results section quotes with the published values in `MANUSCRIPT_VALUES` of the script.

## Included outputs

GitHub Actions runs release checks followed by the full example analysis using Python 3.12.16, `requirements.txt`, CPU PyTorch and `pypdf`. Both jobs must pass for a successful workflow. The full job validates the Excel inputs, regenerates Figures 1–6, Tables 1–5 and the Section 3 values, verifies the separate-process scripts, and uploads `synthetic-example-results`. Completed fits can be reused for matching inputs and model code; the numerical analysis source, fitting stages, seeds and 2,000-draw setting are unchanged.

The bundled figures and tables were generated by `reproduce_all.py` from `abalone_median.py` and its checkpointed fits; [`FIGURE_PROVENANCE.md`](FIGURE_PROVENANCE.md) lists the source function of each. They use the global-scale × relative-block-scale prior parameterization of the group-additive kernel, the stacked-mixture median as the point summary, one common LOO stacking-weight vector, LOO mixture medians for the calibration scores, and Monte Carlo predictive moments computed directly from noisy predictive draws. `results/manifest.csv` records the SHA-256 of every file.
