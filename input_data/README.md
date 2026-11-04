# Input data

The original research workbooks are not included. To request access to the raw Pacific abalone data used in this study, please contact the corresponding author.

The following two files are reviewed synthetic examples, included under the filenames expected by the unchanged analysis code. Other Excel files remain excluded. The release checker permits only these two paths with their approved SHA-256 hashes; replacing either file with other data fails validation.

| Workbook | Used by |
|---|---|
| `전복데이터.xlsx` | Figures 4–6 and Tables 1–5 (`abalone_median.py`) |
| `전복 4개세트통합데이터.xlsx`, sheet `dataset_d_spat_2024` | Figures 1–3 |

Run `python reproduce_all.py --example --workers 3`. The examples contain four synthetic facility categories, eight monthly occasions (April–November 2024) and 30 individuals per facility-date: 32 aggregate rows and 960 individual rows. Measurement dates are month ends and stocking is dated 1 April 2024. Other-year records and additional worksheets are absent.

Weights are in 5–300 g and increase across measurement occasions within each category. Temperature is in 10–25, DO in 10–30, pH in 0–10 and salinity in 1–30. Width and height are missing; aggregate count, density and feed amount are also missing. Individual IDs are 1–30 within each facility-date, and the culture-type column is absent. Mean weights, log weights, ages and environmental values agree between the two files. The values were generated independently of the original observations, statistics and distributions; the four facility labels are compatibility codes already present in the analysis source.

Example outputs and checkpoints are isolated under `abalone_paper_results/example/<key>/`. The key records input hashes and environment versions. Manuscript-value differences are informational for examples; required input and internal-consistency failures still cause a nonzero exit status. Released manuscript outputs remain in `results/`.

The numbered scripts and `reproduce_all.py` look in this folder and then in the repository root, or take `--data`. `python abalone_median.py` looks for `전복데이터.xlsx` beside the script and then in this folder.

## `전복데이터.xlsx`

The analysis selects 2024 records and requires four facilities with eight ordered measurement occasions per facility. It uses the first six occasions for training and the final two for testing.

Required columns:

- `로그무게`: log of the facility-date arithmetic mean wet weight
- `전중(g)`: arithmetic mean total wet weight in grams
- `품종`: facility/cohort
- `측정일`: date
- `일령`: age in days
- `월평균PH`: monthly mean pH
- `월평균염분`: monthly mean salinity
- `월평균DO`: monthly mean dissolved oxygen
- `월평균수온`: monthly mean water temperature

## `전복 4개세트통합데이터.xlsx`

Sheet `dataset_d_spat_2024` holds the individual measurements of the four facilities (`완도_금해`, `완도_민국`, `진도_가반`, `진도_달우`).

Required columns: `날짜` (date), `품종` (facility), `전중(g)` (individual total wet weight, g), `수온` (temperature), `DO`, `PH`, `염분` (salinity).

If a workbook, sheet or required column is missing, the scripts print what they need and stop; they never generate substitute data. The blank width, height, count, density and feed-amount columns are not required GP inputs.

To reproduce the manuscript values, specify authorized original data with `--data` and `--eda-data`, omit `--example` and select a separate `--work-dir`. Do not commit or redistribute the original workbooks. They are not covered by the software license.
