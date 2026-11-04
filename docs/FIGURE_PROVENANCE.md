# Figure and table provenance

Each manuscript figure and table has one entry-point script. The scripts contain no analysis code: Figures 1–3 call `scripts/eda_common.py`, and Figures 4–6 and Tables 1–5 call `scripts/paper_outputs.py`, which imports `abalone_median.py`.

| Output | Script | Source | Input | Fits |
|---|---|---|---|---|
| Figure 1 | `scripts/figures/Figure1.py` | `plot_environment_by_site` in `scripts/eda_common.py` | `전복 4개세트통합데이터.xlsx` [`dataset_d_spat_2024`] | none |
| Figure 2 | `scripts/figures/Figure2.py` | `plot_correlation_heatmap` in `scripts/eda_common.py` | same | none |
| Figure 3 | `scripts/figures/Figure3.py` | `plot_monthly_log_weight_distributions` in `scripts/eda_common.py` | same | none |
| Figure 4 | `scripts/figures/Figure4.py` | `residual_normality`, `figure_qq` | `전복데이터.xlsx` | base |
| Figure 5 | `scripts/figures/Figure5.py` | `assemble_figures`, `table_intervals_methods`, `figure_intervals` | `전복데이터.xlsx` | base, figure state |
| Figure 6 | `scripts/figures/Figure6.py` | `assemble_figures`, `growth_panels` | `전복데이터.xlsx` | base, figure state |
| Table 1 | `scripts/tables/Table1.py` | `table1_facilities` | `전복데이터.xlsx` | none |
| Table 2 | `scripts/tables/Table2.py` | `assemble`, `table_models` | `전복데이터.xlsx` | base |
| Table 3 | `scripts/tables/Table3.py` | `growth_curve_run`, `table_models` | `전복데이터.xlsx` | base |
| Table 4 | `scripts/tables/Table4.py` | `assemble`, `table_ablation` | `전복데이터.xlsx` | base |
| Table 5 | `scripts/tables/Table5.py` | `gather`, `table_e1`, `table_e2`, `table_e3` | `전복데이터.xlsx` | base, ablation |
| Section 3 values | `scripts/text/Section3.py` | `table_models`, `growth_curve_run`, `residual_normality`, `table_intervals`, `table_ablation`, `table_e1`, `table_e2`, `table_e3` | `전복데이터.xlsx` | base, ablation |

Functions without a file name are in `abalone_median.py`. `scripts/paper_outputs.py` assembles the calculated results into the five table structures used in the manuscript and into the list of numbers the Results text quotes (`results/text/Section3_values.csv`); `reproduce_all.py` checks every numbered table against the corresponding in-memory builder and every printed number against the published value.

## Figures 1–3

The three plotting functions use a fixed jitter seed (2024) and `gaussian_kde` for the monthly distributions. Each figure is written as PDF (font type 42) and 300-dpi PNG. The figures carry no overall title, use hyphenated facility names in the fixed order Wando-Geumhae, Wando-Mingug, Jindo-Gaban, Jindo-Daru, and use the font sizes of Figures 4 and 6. The Figure 2 correlations are recomputed from the data each time.

## Figures 4–6

The figures are drawn by `abalone_median.py` under its internal names (`abalone_median_fig_residual_qq`, `abalone_median_fig_intervals_p80`, `abalone_median_fig_growth_curves_shared_p80`) and copied to their numbered names. Their labels are `Native 80%` and `Residual-calibrated 80%` in Figure 5, whose mean width reduction, coverage marks and interval endpoints are computed in the run; `80% residual-calibrated interval`, `Stacking (median)`, `Train / test boundary`, `Train (observed)` and `Test (observed)` in Figure 6, whose line and interval centre are the same stacked predictive-mixture median.

