#!/usr/bin/env python3
"""Reproduce manuscript Figures 1-6, Tables 1-5 and the Section 3 values, verify them.

    python reproduce_all.py --workers 3
    python reproduce_all.py --example --workers 3

Order: Figures 1-3, Table 1, base GP fits and Figures 4-6, Tables 2-4,
ablation fits, Table 5 and the Section 3 values, verification, manifest.  One
abalone_median.py session serves every step, so each fit is read (or made) once.

Verification compares every number printed in the manuscript's Tables 1-5 and
Figures 2, 4 and 5 (PUBLISHED_TABLES, PUBLISHED_FIGURES) with the regenerated
value rounded to the printed digits, and checks the outputs for internal
consistency.  Verification results are written to results/validation/ and the
manifest to results/manifest.csv; the exit status is 1 if any check fails.

With --example, outputs/checkpoints are isolated under abalone_paper_results/example/
and fixed manuscript-value comparisons are informational. Internal checks remain required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts import eda_common  # noqa: E402
from scripts import paper_outputs as po  # noqa: E402

FIGURE_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
TEXT_DIR = ROOT / "results" / "text"
VALIDATION_DIR = ROOT / "results" / "validation"
MANIFEST = ROOT / "results" / "manifest.csv"
ATOL = 1e-10
FORBIDDEN = re.compile(r"conformal|gamma_0", re.IGNORECASE)
# What abalone_median.py prints when it fits or backfills instead of reading a checkpoint.
NEW_FIT = re.compile(r"fitting \d+ of|fitting Matern|backfilling|적합 — 전체 \d+개 중 [1-9]|LOO 분포 보충")
FIGURE6_LEGEND = ["80% residual-calibrated interval", "Stacking (median)",
                  "Train / test boundary", "Train (observed)", "Test (observed)"]
EDA_SOURCES = {1: "plot_environment_by_site", 2: "plot_correlation_heatmap",
               3: "plot_monthly_log_weight_distributions"}


def configure_run(arguments, eda_data: Path, individual) -> list[dict]:
    """Validate example interfaces and isolate their outputs and checkpoints."""
    global FIGURE_DIR, TABLE_DIR, TEXT_DIR, VALIDATION_DIR, MANIFEST
    result_root = ROOT / "results"
    checks = []
    if arguments.example:
        import numpy as np
        import pandas as pd

        data = po.resolve_data(arguments.data)
        frame = pd.read_excel(data)
        required = ["품종", "측정일", "입식일", "측정월(YYYY-MM)", "일령", "전중(g)", "로그무게",
                    "각폭(mm)", "각고(mm)", "월평균미수", "월평균밀도", "개체당월평균공급량(g)",
                    "월평균수온", "월평균DO", "월평균PH", "월평균염분"]
        missing = sorted(set(required) - set(frame.columns))
        missing_individual = sorted(set([*eda_common.REQUIRED_COLUMNS, "개체번호", "각폭(mm)",
                                         "각고(mm)"]) - set(individual.columns))
        if missing or missing_individual:
            raise SystemExit(f"Example inputs lack columns: analysis={missing}, "
                             f"individual={missing_individual}")
        frame["측정일"] = pd.to_datetime(frame["측정일"], errors="coerce")
        frame["입식일"] = pd.to_datetime(frame["입식일"], errors="coerce")

        def check(item, expected, observed, passed):
            checks.append({"check": item, "expected": str(expected), "observed": str(observed),
                           "passed": bool(passed)})

        check("example row counts", "32 aggregate, 960 individual",
              f"{len(frame)}, {len(individual)}", len(frame) == 32 and len(individual) == 960)
        check("example facility order", eda_common.TARGET_SITES,
              frame["품종"].drop_duplicates().tolist(),
              frame["품종"].drop_duplicates().tolist() == eda_common.TARGET_SITES
              and individual["품종"].drop_duplicates().tolist() == eda_common.TARGET_SITES)
        check("example year", 2024, "measurement and stocking dates",
              frame["측정일"].dt.year.eq(2024).all() and frame["입식일"].dt.year.eq(2024).all()
              and individual["날짜"].dt.year.eq(2024).all())
        check("example monthly occasions", "April..November, once per facility",
              "aggregate dates", all(block["측정일"].dt.month.tolist() == list(range(4, 12))
                                     for _, block in frame.groupby("품종", sort=False)))
        grouped = individual.groupby(["품종", "날짜"], sort=True)
        counts = grouped.size()
        check("example individuals per facility-date", 30, sorted(counts.unique()),
              len(counts) == 32 and counts.eq(30).all())
        check("example individual IDs", "1..30", "each facility-date",
              all(block["개체번호"].tolist() == list(range(1, 31)) for _, block in grouped))
        for name in ["각폭(mm)", "각고(mm)", "월평균미수", "월평균밀도", "개체당월평균공급량(g)"]:
            check(f"example {name} missing", True, int(frame[name].notna().sum()),
                  frame[name].isna().all())
        check("example individual width and height missing", True, "all rows",
              individual[["각폭(mm)", "각고(mm)"]].isna().all().all())
        check("example culture-type column removed", True, "양식유형" not in individual,
              "양식유형" not in individual)
        check("example weight range", "5..300 g", "both workbooks",
              frame["전중(g)"].between(5, 300).all()
              and individual["전중(g)"].between(5, 300).all())
        check("example weights increase with date", True, "each facility",
              all(np.all(np.diff(block.sort_values("측정일")["전중(g)"]) > 0)
                  for _, block in frame.groupby("품종")))
        check("example age and month agree with dates", True, "all aggregate rows",
              (frame["측정일"] - frame["입식일"]).dt.days.eq(frame["일령"]).all()
              and frame["측정일"].dt.strftime("%Y-%m").eq(frame["측정월(YYYY-MM)"]).all())
        aggregate = frame.set_index(["품종", "측정일"])
        means = grouped[["전중(g)", "수온", "DO", "PH", "염분"]].mean()
        means.index.names = aggregate.index.names
        means = means.reindex(aggregate.index)
        check("example mean weights agree between workbooks", True, "all facility-dates",
              np.allclose(aggregate["전중(g)"], means["전중(g)"], rtol=0, atol=1e-10))
        check("example log weights are log(arithmetic mean)", True, "all aggregate rows",
              np.allclose(frame["로그무게"], np.log(frame["전중(g)"]), rtol=0, atol=1e-12))
        for name, (low, high) in {"수온": (10, 25), "DO": (10, 30), "PH": (0, 10),
                                  "염분": (1, 30)}.items():
            check(f"example {name} range and aggregate agreement", f"{low}..{high}", "both files",
                  individual[name].between(low, high).all()
                  and frame["월평균" + name].between(low, high).all()
                  and np.allclose(aggregate["월평균" + name], means[name], rtol=0, atol=1e-10))
        failed = [r["check"] for r in checks if not r["passed"]]
        if failed:
            raise SystemExit("Example input checks failed:\n  " + "\n  ".join(failed))
        versions = {"python": platform.python_version()}
        for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "gpytorch",
                        "linear-operator", "xgboost"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not installed"
        inputs = {"analysis_sha256": po.sha256(data), "eda_sha256": po.sha256(eda_data),
                  "source_sha256": po.sha256(po.ANALYSIS_SOURCE),
                  "runner_sha256": po.sha256(Path(po.__file__)), "sheet": arguments.sheet,
                  "requirements_sha256": po.sha256(ROOT / "requirements.txt"),
                  "platform": [platform.system(), platform.machine()], "versions": versions}
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
        work_dir = (Path(arguments.work_dir).expanduser().resolve() if arguments.work_dir else
                    po.DEFAULT_WORK_DIR / "example" / key)
        if (work_dir == ROOT or work_dir == po.DEFAULT_WORK_DIR
                or work_dir.is_relative_to(ROOT / "results")):
            raise SystemExit("--example requires its own work directory outside released results "
                             "and the original checkpoint root.")
        marker = work_dir / "example_inputs.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) != inputs:
            raise SystemExit("Example inputs or environment differ from these checkpoints. "
                             "Choose another --work-dir; existing fits were not changed.")
        if not marker.exists() and work_dir.is_dir() and any(work_dir.glob("_checkpoints_S*/*")):
            raise SystemExit("Unidentified checkpoints exist. Choose another --work-dir.")
        work_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        arguments.data, arguments.work_dir = str(data), str(work_dir)
        result_root = work_dir / "results"
        print(f"Example input checks: {len(checks)}/{len(checks)} passed", flush=True)
        print(f"Example outputs: {result_root}", flush=True)
        print("Manuscript comparisons are informational; internal checks remain required.",
              flush=True)
    FIGURE_DIR, TABLE_DIR = result_root / "figures", result_root / "tables"
    TEXT_DIR, VALIDATION_DIR = result_root / "text", result_root / "validation"
    MANIFEST = result_root / "manifest.csv"
    return checks


def required_failures(published, run, figures, tables, *, example=False):
    return ([f"{r['output']}: {r['item']} (published {r['published']}, "
             f"regenerated {r['computed']})" for r in published
             if not example and not r["passed"]]
            + [r["check"] for r in run if not r["passed"]]
            + [f"{r['figure']}: {r['check']}" for r in figures if not r["passed"]]
            + [f"{r['table']}: {r['comparison']}" for r in tables if not r["passed"]])

# -----------------------------------------------------------------------------
# Published values
# -----------------------------------------------------------------------------
# Every number printed in the manuscript tables, keyed by table number, as
# (first-column entry, {column: printed text}).  A regenerated number matches
# when it rounds to the printed text; any other cell must be equal as text.
# Group headings without numbers (Table 3) are not listed.
PUBLISHED_TABLES = {
    1: [
        ("Wando-Geumhae", {"Location": "Wando", "Stocked": "17 Feb 2024",
                           "Sampling period (n)": "Apr–Nov 2024 (8)", "Individuals/date": "30"}),
        ("Wando-Mingug", {"Location": "Wando", "Stocked": "17 Feb 2024",
                          "Sampling period (n)": "Apr–Nov 2024 (8)", "Individuals/date": "30"}),
        ("Jindo-Gaban", {"Location": "Jindo", "Stocked": "10 Mar 2024",
                         "Sampling period (n)": "Apr–Nov 2024 (8)", "Individuals/date": "30"}),
        ("Jindo-Daru", {"Location": "Jindo", "Stocked": "18 Mar 2024",
                        "Sampling period (n)": "Apr–Nov 2024 (8)", "Individuals/date": "30"}),
    ],
    2: [
        ("Stacking", {"Weight": "—", "RMSE (g)": "1.6363", "MAE (g)": "1.2852",
                      "MAPE (%)": "14.2875", "R2_pred": "0.9552"}),
        ("Group-additive SE", {"Weight": "0.4361", "RMSE (g)": "1.6891", "MAE (g)": "1.3807",
                               "MAPE (%)": "15.0548", "R2_pred": "0.9522"}),
        ("Neural network", {"Weight": "0.4074", "RMSE (g)": "1.8292", "MAE (g)": "1.4622",
                            "MAPE (%)": "16.8381", "R2_pred": "0.9440"}),
        ("Matérn(ν = 1/2)", {"Weight": "0.1565", "RMSE (g)": "3.7423", "MAE (g)": "2.8679",
                             "MAPE (%)": "27.3997", "R2_pred": "0.7655"}),
    ],
    3: [
        ("Stacking", {"Inputs": "full", "RMSE (g)": "1.6363", "MAE (g)": "1.2852",
                      "MAPE (%)": "14.2875", "R2_pred": "0.9552"}),
        ("von Bertalanffyᵃ", {"Inputs": "day-age", "RMSE (g)": "2.5365", "MAE (g)": "2.3094",
                              "MAPE (%)": "23.2569", "R2_pred": "0.8923"}),
        ("Gompertz", {"Inputs": "day-age", "RMSE (g)": "3.7218", "MAE (g)": "3.4059",
                      "MAPE (%)": "36.5780", "R2_pred": "0.7680"}),
        ("Bayesian ridge", {"Inputs": "full", "RMSE (g)": "4.8324", "MAE (g)": "3.8225",
                            "MAPE (%)": "36.1085", "R2_pred": "0.6089"}),
        ("Random forest", {"Inputs": "full", "RMSE (g)": "5.7878", "MAE (g)": "5.3725",
                           "MAPE (%)": "53.3253", "R2_pred": "0.4390"}),
        ("XGBoost", {"Inputs": "full", "RMSE (g)": "5.7462", "MAE (g)": "5.1809",
                     "MAPE (%)": "50.6192", "R2_pred": "0.4470"}),
    ],
    4: [
        ("Stacking (all three)", {"RMSE (g)": "1.6363", "MAE (g)": "1.2852",
                                  "MAPE (%)": "14.2875", "R2_pred": "0.9552"}),
        ("Without Matérn(ν = 1/2)", {"RMSE (g)": "1.6812", "MAE (g)": "1.3297",
                                     "MAPE (%)": "15.3060", "R2_pred": "0.9527"}),
        ("Without group-additive SE", {"RMSE (g)": "1.6886", "MAE (g)": "1.3266",
                                       "MAPE (%)": "15.2493", "R2_pred": "0.9523"}),
        ("Without neural network", {"RMSE (g)": "1.9189", "MAE (g)": "1.6180",
                                    "MAPE (%)": "16.8461", "R2_pred": "0.9383"}),
    ],
    5: [
        ("Stacking (proposed)", {"RMSE (g)": "1.6363", "MAE (g)": "1.2852",
                                 "MAPE (%)": "14.2875", "R2_pred": "0.9552"}),
        ("Without environmental covariates", {"RMSE (g)": "2.2362", "MAE (g)": "1.8544",
                                              "MAPE (%)": "18.9343", "R2_pred": "0.9163"}),
        ("With equal weights", {"RMSE (g)": "1.8181", "MAE (g)": "1.4762",
                                "MAPE (%)": "15.5634", "R2_pred": "0.9446"}),
        ("Without the monotone construction", {"RMSE (g)": "5.2308", "MAE (g)": "4.8017",
                                               "MAPE (%)": "48.5878", "R2_pred": "0.5418"}),
    ],
}

# Numbers printed in the manuscript figures and their captions.  The Figure 2
# correlations are eda_common.REFERENCE_CORRELATIONS, printed at three decimals.
PUBLISHED_FIGURES = {
    4: {  # Shapiro-Wilk W and p in each panel (kernel label -> (W, p))
        "Matérn(ν = 1/2)": ("0.964", "0.534"),
        "Group-additive SE": ("0.956", "0.365"),
        "Neural network": ("0.950", "0.270"),
    },
    5: {  # caption and lower-panel legend
        "native 80% mean width (g)": "6.5722",
        "residual-calibrated 80% mean width (g)": "4.8244",
        "mean width reduction (%)": "26.6",
        "native 80% coverage": "8/8",
        "residual-calibrated 80% coverage": "7/8",
    },
}


def banner(text: str) -> None:
    print("\n" + "#" * 78 + f"\n# {text}\n" + "#" * 78, flush=True)


def relative(path: Path) -> str:
    path = Path(path)
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


# -----------------------------------------------------------------------------
# File comparison
# -----------------------------------------------------------------------------
def as_float(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def read_cells(path: Path):
    import pandas as pd

    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def compare_csv(table: str, comparison: str, left: Path, right: Path):
    """Column names, row order, strings exactly; numbers with rtol = 0, atol = ATOL."""
    a, b = read_cells(left), read_cells(right)
    details = []

    def mismatch(row, column, x, y, kind, difference=""):
        details.append({"table": table, "comparison": comparison, "row": row,
                        "column": column, "left": x, "right": y, "kind": kind,
                        "abs_difference": difference})

    columns_match = list(a.columns) == list(b.columns)
    rows_match = len(a) == len(b)
    order_match = rows_match and a.iloc[:, 0].tolist() == b.iloc[:, 0].tolist()
    if not columns_match:
        mismatch("", "", "|".join(a.columns), "|".join(b.columns), "column names")
    if not rows_match:
        mismatch("", "", len(a), len(b), "row count")
    elif not order_match:
        mismatch("", a.columns[0], "|".join(a.iloc[:, 0]), "|".join(b.iloc[:, 0]), "row order")
    largest, strings, numbers = 0.0, 0, 0
    if columns_match and rows_match:
        for column in a.columns:
            for row, (x, y) in enumerate(zip(a[column], b[column])):
                fx, fy = as_float(x), as_float(y)
                if fx is not None and fy is not None:
                    if math.isnan(fx) and math.isnan(fy):
                        continue
                    difference = abs(fx - fy) if not (math.isnan(fx) or math.isnan(fy)) \
                        else math.inf
                    largest = max(largest, difference)
                    if not difference <= ATOL:
                        numbers += 1
                        mismatch(row, column, x, y, "number", difference)
                elif x != y:
                    strings += 1
                    mismatch(row, column, x, y, "string")
    summary = {"table": table, "comparison": comparison, "left": relative(left),
               "right": relative(right), "byte_identical": left.read_bytes() == right.read_bytes(),
               "columns_match": columns_match, "row_count_match": rows_match,
               "row_order_match": order_match, "string_mismatches": strings,
               "number_mismatches": numbers, "max_abs_difference": largest,
               "passed": columns_match and rows_match and order_match
               and strings == 0 and numbers == 0}
    return summary, details


def same_png(left: Path, right: Path) -> bool:
    import matplotlib.image as mpimg
    import numpy as np

    a, b = mpimg.imread(left), mpimg.imread(right)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def same_pdf(left: Path, right: Path) -> bool:
    """Byte equality apart from the creation timestamp matplotlib writes."""
    def strip(path):
        return re.sub(rb"/CreationDate \(D:[^)]*\)", b"", Path(path).read_bytes())
    return strip(left) == strip(right)


def pdf_text(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    return "\n".join(page.extract_text() for page in PdfReader(str(path)).pages)


# -----------------------------------------------------------------------------
# Figure 4-6 checks on the drawn artists
# -----------------------------------------------------------------------------
def check_figures_456(run, scratch: Path) -> list[dict]:
    import numpy as np
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.text import Text

    am = run.am
    rows = []

    def add(figure, check, expected, observed, passed):
        rows.append({"figure": figure, "check": check, "expected": str(expected),
                     "observed": str(observed), "passed": bool(passed)})

    redraw = scratch / "redraw"
    redraw.mkdir(parents=True, exist_ok=True)
    with run.captured_figures(redraw) as captured:
        for number in (4, 5, 6):
            run.draw_figure(number)
    for number, (internal, final) in po.FIGURES.items():
        identical = same_png(redraw / f"{am.FIGURE_PREFIX}_{internal}.png",
                             FIGURE_DIR / f"{final}.png")
        add(f"Figure {number}", "redraw equals the published PNG (pixels)", True, identical,
            identical)
        figure = captured[internal]
        add(f"Figure {number}", "no overall figure title", "",
            figure._suptitle.get_text() if figure._suptitle else "", figure._suptitle is None
            or not figure._suptitle.get_text())
        texts = [t.get_text() for t in figure.findobj(Text) if t.get_text()]
        found = [t for t in texts if FORBIDDEN.search(t)]
        add(f"Figure {number}", "no 'conformal' or 'gamma_0' in drawn text", "none",
            found or "none", not found)
        text = pdf_text(FIGURE_DIR / f"{final}.pdf")
        if text is None:
            add(f"Figure {number}", "no 'conformal' or 'gamma_0' in PDF text", "none",
                "skipped: pypdf not installed", True)
        else:
            found = sorted(set(FORBIDDEN.findall(text)))
            add(f"Figure {number}", "no 'conformal' or 'gamma_0' in PDF text", "none",
                found or "none", not found)

    # Figure 4: residuals and Shapiro-Wilk results of residual_normality
    normality, residuals = run.normality()
    figure = captured[po.FIGURES[4][0]]
    for axis, kernel in zip(figure.axes, am.KERNELS):
        row = normality.set_index("kernel").loc[am.KERNEL_LABEL[kernel]]
        points = np.concatenate([c.get_offsets()[:, 1] for c in axis.collections
                                 if isinstance(c, PathCollection)])
        gap = float(np.max(np.abs(np.sort(points) - np.sort(residuals[kernel]))))
        add("Figure 4", f"{kernel}: plotted points = standardised residuals",
            f"{len(residuals[kernel])} points", f"{len(points)} points, max gap {gap:.1e}",
            len(points) == len(residuals[kernel]) and gap <= 1e-12)
        expected = f"W = {row.shapiro_W:.3f}, p = {row.shapiro_p:.3f}"
        annotation = " ".join(t.get_text() for t in axis.texts)
        add("Figure 4", f"{kernel}: Shapiro-Wilk annotation", expected,
            annotation.replace("\n", " "), expected in annotation)
        add("Figure 4", f"{kernel}: Shapiro-Wilk W, p (full precision)", "",
            f"W = {row.shapiro_W:.10f}, p = {row.shapiro_p:.10f}, n = {int(row.n)}", True)

    # Figure 5: the drawn endpoints agree with the internal interval calculation.
    intervals = run.intervals()
    test_rows = intervals[~intervals.Facility.isin(["mean t7", "mean t8", "ALL"])]
    figure = captured[po.FIGURES[5][0]]
    upper, lower = figure.axes[:2]
    for label, low, high in ((f"Native {am.NOMINAL:.0%}", "Native_low_g", "Native_high_g"),
                             (f"Residual-calibrated {am.NOMINAL:.0%}", "Conformal_low_g",
                              "Conformal_high_g")):
        bars = [c for c in upper.collections
                if isinstance(c, LineCollection) and c.get_label() == label]
        if len(bars) != 1:
            add("Figure 5", f"'{label}' endpoints = interval calculation",
                "one bar series", len(bars), False)
            continue
        segments = np.asarray(bars[0].get_segments(), float)
        gap = float(np.max(np.abs(np.r_[segments[:, 0, 1] - test_rows[low].to_numpy(float),
                                        segments[:, 1, 1] - test_rows[high].to_numpy(float)])))
        add("Figure 5", f"'{label}' endpoints = interval calculation ({low}, {high})",
            f"{len(test_rows)} rows, |gap| <= {ATOL:g}", f"{len(segments)} rows, max gap {gap:.1e}",
            len(segments) == len(test_rows) and gap <= ATOL)
    total = intervals[intervals.Facility.eq("ALL")].iloc[0]
    expected = f"mean width reduction {float(total.Width_reduction_pct):.1f}%"
    legend = [t.get_text() for t in lower.get_legend().get_texts()]
    add("Figure 5", "mean width reduction = interval calculation (ALL)",
        expected, legend, legend == [expected])
    covered = [c for c in upper.collections if c.get_label() == "Observed (covered)"]
    n_covered = len(covered[0].get_offsets()) if covered else 0
    n_expected = covered_count(test_rows.Conformal_covers)
    add("Figure 5", "covered observations = interval calculation",
        f"{n_expected}/{len(test_rows)}", f"{n_covered}/{len(test_rows)}",
        n_covered == n_expected)
    legend = [t.get_text() for t in upper.get_legend().get_texts()]
    add("Figure 5", "legend", "Native 80%, Residual-calibrated 80%, Observed ...", legend,
        legend[:2] == [f"Native {am.NOMINAL:.0%}", f"Residual-calibrated {am.NOMINAL:.0%}"])

    # Figure 6: line = stacked mixture median = centre of the calibrated band
    context, parts = run.context(), run.figure_parts()
    response = context["response"]
    curve, mixture, _ = am.curve_components(context, parts)
    median_z = np.asarray(mixture.ppf(0.5), float)
    at_median = np.asarray(mixture.cdf(median_z), float)
    add("Figure 6", "mixture CDF at the plotted median", "0.5 +/- 1e-6",
        f"max |F - 0.5| = {np.max(np.abs(at_median - 0.5)):.1e}",
        bool(np.all(np.abs(at_median - 0.5) < 1e-6)))
    radius = float(am.conformal_bands(context, parts, am.NOMINAL))
    table_radius = 0.5 * (response.to_z(test_rows.Conformal_high_g.to_numpy(float))
                          - response.to_z(test_rows.Conformal_low_g.to_numpy(float)))
    add("Figure 6", "band radius (z) = calibrated interval radius", f"{radius:.12f}",
        f"max gap {np.max(np.abs(table_radius - radius)):.1e}",
        bool(np.max(np.abs(table_radius - radius)) <= 1e-9))
    figure = captured[po.FIGURES[6][0]]
    cohort_of = {am.label_of(c): c for c in context["cohorts"]}
    cohort_column = curve[am.COHORT_COL].astype(str).to_numpy()
    ages_all = curve[am.AGE_COL].to_numpy(float)
    for axis in figure.axes:
        name = axis.get_title()
        line = [l for l in axis.get_lines() if l.get_label() == f"Stacking ({am.POINT_SUMMARY})"]
        band = [c for c in axis.collections
                if c.get_label() == f"{int(am.NOMINAL * 100)}% residual-calibrated interval"]
        if name not in cohort_of or len(line) != 1 or len(band) != 1:
            add("Figure 6", f"{name}: line and band present", "1 line, 1 band",
                f"{len(line)} lines, {len(band)} bands", False)
            continue
        ages, y = np.asarray(line[0].get_xdata(), float), np.asarray(line[0].get_ydata(), float)
        mask = cohort_column == cohort_of[name]
        order = np.argsort(ages_all[mask], kind="mergesort")
        expected_line = response.from_z(median_z[mask][order])
        gap = float(np.max(np.abs(y - expected_line)))
        add("Figure 6", f"{name}: line = stacked predictive mixture median",
            f"|gap| <= {ATOL:g} g", f"max gap {gap:.1e} g", gap <= ATOL)
        n = len(ages)
        vertices = band[0].get_paths()[0].vertices
        band_ages = vertices[1:n + 1, 0]
        low = vertices[1:n + 1, 1]
        high = vertices[n + 2:2 * n + 2, 1][::-1]
        floor = axis.get_ylim()[0]
        open_rows = low > floor * (1 + 1e-12)
        centre = 0.5 * (response.to_z(low) + response.to_z(high))
        gap = float(np.max(np.abs(centre[open_rows] - response.to_z(y)[open_rows])))
        add("Figure 6", f"{name}: band centre (z) = plotted median",
            "|gap| <= 1e-9", f"max gap {gap:.1e} on {int(open_rows.sum())} of {n} nodes "
            f"({n - int(open_rows.sum())} lower ends clipped at the axis)",
            np.allclose(band_ages, ages, rtol=0, atol=1e-12) and gap <= 1e-9)
    legend = [t.get_text() for t in figure.legends[0].get_texts()] if figure.legends else []
    add("Figure 6", "legend", FIGURE6_LEGEND, legend, legend == FIGURE6_LEGEND)
    return rows


# -----------------------------------------------------------------------------
# Comparison with the published values
# -----------------------------------------------------------------------------
def covered_count(flags) -> int:
    """Number of covered rows; the flags are booleans (or their text)."""
    return int(sum(str(flag).strip().lower() in ("true", "1", "1.0") for flag in flags))


def at_printed_precision(value, published: str) -> str:
    """``value`` written with the digits of ``published``; text stays text."""
    try:
        float(published)
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    decimals = len(published.split(".")[1]) if "." in published else 0
    text = f"{number:.{decimals}f}"
    return text.lstrip("-") if float(text) == 0.0 else text


def full_precision(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def compare_published(run, eda_frame) -> list[dict]:
    """Every published table cell and figure number against the regenerated value."""
    rows = []

    def add(output, item, published, value):
        computed = at_printed_precision(value, published)
        rows.append({"output": output, "item": item, "published": published,
                     "computed": computed, "full_precision": full_precision(value),
                     "passed": computed == published})

    for number, entries in PUBLISHED_TABLES.items():
        table = run.table(number)
        key = table.columns[0]
        lookup = {str(entry): i for i, entry in enumerate(table[key])}
        for entry, cells in entries:
            for column, published in cells.items():
                item = f"{entry} | {column}"
                if entry not in lookup or column not in table.columns:
                    rows.append({"output": f"Table {number}", "item": item,
                                 "published": published, "computed": "missing",
                                 "full_precision": "", "passed": False})
                    continue
                add(f"Table {number}", item, published, table.loc[lookup[entry], column])

    matrix = eda_common.correlation_matrix(eda_frame)
    for (left, right), reference in eda_common.REFERENCE_CORRELATIONS.items():
        add("Figure 2", f"r({left}, {right})", f"{reference:.3f}", matrix.loc[left, right])

    normality, _ = run.normality()
    normality = normality.set_index("kernel")
    for label, (w_text, p_text) in PUBLISHED_FIGURES[4].items():
        add("Figure 4", f"{label} | Shapiro-Wilk W", w_text, normality.loc[label, "shapiro_W"])
        add("Figure 4", f"{label} | Shapiro-Wilk p", p_text, normality.loc[label, "shapiro_p"])

    intervals = run.intervals()
    test_rows = intervals[~intervals.Facility.isin(["mean t7", "mean t8", "ALL"])]
    total = intervals[intervals.Facility.eq("ALL")].iloc[0]
    n = len(test_rows)
    computed = {
        "native 80% mean width (g)": total.Native_width_g,
        "residual-calibrated 80% mean width (g)": total.Conformal_width_g,
        "mean width reduction (%)": total.Width_reduction_pct,
        "native 80% coverage": f"{covered_count(test_rows.Native_covers)}/{n}",
        "residual-calibrated 80% coverage": f"{covered_count(test_rows.Conformal_covers)}/{n}",
    }
    for item, published in PUBLISHED_FIGURES[5].items():
        add("Figure 5", item, published, computed[item])
    return rows


# -----------------------------------------------------------------------------
# Individual scripts, run as separate processes from another directory
# -----------------------------------------------------------------------------
def run_individual(arguments, run, eda_data: Path, scratch: Path):
    output = scratch / "individual"
    output.mkdir(parents=True, exist_ok=True)
    logs, figure_rows, table_summaries, table_details = [], [], [], []
    scripts = [ROOT / "scripts" / "figures" / f"Figure{n}.py" for n in range(1, 7)]
    scripts += [ROOT / "scripts" / "tables" / f"Table{n}.py" for n in range(1, 6)]
    scripts += [ROOT / "scripts" / "text" / f"Section{n}.py" for n in po.TEXTS]
    for script in scripts:
        command = [sys.executable, str(script), "--output-dir", str(output)]
        if script.stem in ("Figure1", "Figure2", "Figure3"):
            command += ["--data", str(eda_data), "--sheet", arguments.sheet]
        else:
            command += ["--data", str(run.data), "--work-dir", str(run.work_dir),
                        "--workers", str(arguments.workers)]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=scratch, capture_output=True, text=True)
        (scratch / f"{script.stem}.log").write_text(result.stdout + result.stderr,
                                                    encoding="utf-8")
        fitted = bool(NEW_FIT.search(result.stdout))
        logs.append({"script": relative(script), "cwd": "temporary folder outside the repository",
                     "return_code": result.returncode,
                     "seconds": round(time.perf_counter() - started, 1),
                     "gp_fit_stage_entered": "Stage 1" in result.stdout,
                     "new_fits": fitted})
        print(f"  {relative(script):32s} exit {result.returncode}  "
              f"{logs[-1]['seconds']:6.1f} s", flush=True)
    for number in range(1, 7):
        stem = eda_common.FIGURE_STEMS.get(number) or po.FIGURES[number][1]
        png = same_png(output / f"{stem}.png", FIGURE_DIR / f"{stem}.png") \
            if (output / f"{stem}.png").is_file() else False
        pdf = same_pdf(output / f"{stem}.pdf", FIGURE_DIR / f"{stem}.pdf") \
            if (output / f"{stem}.pdf").is_file() else False
        figure_rows.append({"figure": f"Figure {number}",
                            "check": f"Figure{number}.py (separate process) = reproduce_all.py",
                            "expected": "identical PNG pixels and PDF bytes (except CreationDate)",
                            "observed": f"PNG {png}, PDF {pdf}", "passed": png and pdf})
    for number, (_, final) in po.TABLES.items():
        target = output / f"{final}.csv"
        label = f"scripts/tables/Table{number}.py -> <temporary folder>/{final}.csv"
        if not target.is_file():
            table_summaries.append({"table": f"Table {number}", "comparison":
                                    f"Table{number}.py (separate process) vs reproduce_all.py",
                                    "left": relative(TABLE_DIR / f"{final}.csv"),
                                    "right": f"{label} (missing)", "passed": False})
            continue
        summary, details = compare_csv(f"Table {number}",
                                       f"Table{number}.py (separate process) vs reproduce_all.py",
                                       TABLE_DIR / f"{final}.csv", target)
        table_summaries.append({**summary, "right": label})
        table_details += details
    for number, (_, final) in po.TEXTS.items():
        target = output / f"{final}.csv"
        label = f"scripts/text/Section{number}.py -> <temporary folder>/{final}.csv"
        comparison = f"Section{number}.py (separate process) vs reproduce_all.py"
        if not target.is_file():
            table_summaries.append({"table": f"Section {number}", "comparison": comparison,
                                    "left": relative(TEXT_DIR / f"{final}.csv"),
                                    "right": f"{label} (missing)", "passed": False})
            continue
        summary, details = compare_csv(f"Section {number}", comparison,
                                       TEXT_DIR / f"{final}.csv", target)
        table_summaries.append({**summary, "right": label})
        table_details += details
    return logs, figure_rows, table_summaries, table_details


def compare_external(label: str, folder: Path):
    """Compare the outputs with another manuscript-numbered result folder.

    Files are found in the folder itself or in its table(s)/ and figures/
    subfolders. The more detailed internal ledgers written by
    ``abalone_median.py`` are deliberately not treated as manuscript tables.
    """
    folders = [folder, folder / "table", folder / "tables", folder / "figures"]

    def find(*names):
        for directory in folders:
            present = {p.name for p in directory.iterdir()} if directory.is_dir() else set()
            for name in names:
                if name in present:            # exact name, also on case-insensitive disks
                    return directory / name
        return None

    figure_rows, summaries, details = [], [], []
    comparison = f"reproduce_all.py vs {label}"
    for number, (internal, final) in po.TABLES.items():
        other = find(f"{final}.csv")
        if other is None:
            summaries.append({"table": f"Table {number}", "comparison": comparison,
                              "left": relative(TABLE_DIR / f"{final}.csv"),
                              "right": f"{label}: not found", "passed": False})
            continue
        summary, cells = compare_csv(f"Table {number}", comparison,
                                     TABLE_DIR / f"{final}.csv", other)
        summaries.append({**summary, "right": f"{label}: {other.name}"})
        details += cells
    for number, (internal, final) in po.FIGURES.items():
        other = find(f"{final}.png", f"abalone_{po.POINT_SUMMARY}_{internal}.png",
                     f"{internal}.png")
        identical = other is not None and same_png(other, FIGURE_DIR / f"{final}.png")
        figure_rows.append({"figure": f"Figure {number}", "check": f"PNG = {label}",
                            "expected": "identical pixels",
                            "observed": f"{other.name if other else 'not found'}: {identical}",
                            "passed": identical})
    return figure_rows, summaries, details


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--example", action="store_true",
                        help="validate synthetic inputs and run all outputs in a separate "
                             "work folder; manuscript comparisons are informational")
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel fits if a checkpoint is missing (default 3)")
    parser.add_argument("--data", help=f"analysis workbook (default: input_data/{po.DATA_NAME})")
    parser.add_argument("--eda-data", help=f"Figures 1-3 workbook "
                                           f"(default: input_data/{eda_common.DATA_NAME})")
    parser.add_argument("--sheet", default=eda_common.SHEET_NAME,
                        help=f"Figures 1-3 sheet (default: {eda_common.SHEET_NAME})")
    parser.add_argument("--work-dir", help="abalone_median.py output root that holds "
                                           "_checkpoints_S2000 (paper default: abalone_paper_results/; "
                                           "example default: its input/environment subfolder)")
    parser.add_argument("--skip-individual", action="store_true",
                        help="do not rerun the numbered scripts as separate processes")
    parser.add_argument("--compare", action="append", default=[], metavar="LABEL=DIR",
                        help="also compare Tables 1-5 and Figures 4-6 with another numbered "
                             "result folder (repeatable)")
    return parser.parse_args()


def main() -> int:
    import pandas as pd

    arguments = parse_arguments()
    started = time.perf_counter()
    source_before = po.sha256(po.ANALYSIS_SOURCE)
    outputs: dict[str, list[Path]] = {}
    released_before = ({relative(p): po.sha256(p) for p in (ROOT / "results").rglob("*")
                        if p.is_file()} if arguments.example else {})

    banner("1-3. Figures 1-3 (no GP fit)")
    eda_data = eda_common.resolve_data(arguments.eda_data)
    df = eda_common.load_data(eda_data, arguments.sheet)
    input_checks = configure_run(arguments, eda_data, df)
    for number in (1, 2, 3):
        outputs[f"Figure {number}"] = eda_common.render_figure(number, df, FIGURE_DIR)
        print(f"  Figure {number}: {', '.join(relative(p) for p in outputs[f'Figure {number}'])}")

    banner("4. Table 1 (no GP fit)")
    run = po.PaperRun(arguments.data, arguments.work_dir, arguments.workers)
    checkpoints = po.checkpoint_files(run.work_dir)
    checkpoint_before = {name: po.sha256(path) if path.is_file() else None
                         for name, path in checkpoints.items()}
    outputs["Table 1"] = [run.write_table(1, TABLE_DIR)]
    stages_after_table1 = sorted(run._stages)

    banner("5. Base GP fits and Figures 4-6")
    run.ensure_fits(("base",))
    for number in (4, 5, 6):
        outputs[f"Figure {number}"] = run.write_figure(number, FIGURE_DIR)

    banner("6. Tables 2-4")
    for number in (2, 3, 4):
        outputs[f"Table {number}"] = [run.write_table(number, TABLE_DIR)]

    banner("7. Ablation fits, Table 5 and the Section 3 values")
    run.ensure_fits(("ablation",))
    outputs["Table 5"] = [run.write_table(5, TABLE_DIR)]
    for number in po.TEXTS:
        outputs[f"Section {number}"] = [run.write_text(number, TEXT_DIR)]

    banner("8. Verification")
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    run_rows = list(input_checks)

    def check(item, expected, observed, passed):
        run_rows.append({"check": item, "expected": str(expected), "observed": str(observed),
                         "passed": bool(passed)})

    source_after = po.sha256(po.ANALYSIS_SOURCE)
    check("abalone_median.py SHA-256 before the run", po.EXPECTED_SOURCE_SHA256, source_before,
          source_before == po.EXPECTED_SOURCE_SHA256)
    check("abalone_median.py SHA-256 after the run", po.EXPECTED_SOURCE_SHA256, source_after,
          source_after == po.EXPECTED_SOURCE_SHA256)
    if arguments.example:
        released_after = {relative(p): po.sha256(p) for p in (ROOT / "results").rglob("*")
                          if p.is_file()}
        check("released manuscript outputs remain unchanged", True,
              released_before == released_after, released_before == released_after)
    check("fit stages entered for Table 1", "none", stages_after_table1 or "none",
          not stages_after_table1)
    expected_files = [p for paths in outputs.values() for p in paths]
    missing = [relative(p) for p in expected_files if not p.is_file()]
    check("numbered outputs present", f"{len(expected_files)} files", missing or "all present",
          not missing)
    for module, version in (("python", platform.python_version()),
                            *((m, __import__(m).__version__) for m in
                              ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn",
                               "torch", "gpytorch", "xgboost"))):
        check(f"version {module}", "", version, True)

    published_rows = compare_published(run, df)
    figure_rows = [{**row} for row in eda_common.check_eda(
        df, compare_published=not arguments.example)]
    table_summaries, table_details = [], []
    with tempfile.TemporaryDirectory(prefix="abalone-verify-") as folder:
        scratch = Path(folder)
        figure_rows += check_figures_456(run, scratch)

        # Every written table CSV against its in-memory table builder.
        integrated = scratch / "integrated"
        integrated.mkdir()
        for number, (internal, final) in po.TABLES.items():
            path = integrated / f"{internal}.csv"
            run.table(number).to_csv(path, index=False, encoding=run.am.ENC)
            summary, details = compare_csv(f"Table {number}",
                                           "written CSV vs table builder",
                                           TABLE_DIR / f"{final}.csv", path)
            table_summaries.append({**summary, "right": f"PaperRun.table({number})"})
            table_details += details
        for number, (internal, final) in po.TEXTS.items():
            path = integrated / f"{internal}.csv"
            run.text(number).to_csv(path, index=False, encoding=run.am.ENC)
            summary, details = compare_csv(f"Section {number}",
                                           "written CSV vs text builder",
                                           TEXT_DIR / f"{final}.csv", path)
            table_summaries.append({**summary, "right": f"PaperRun.text({number})"})
            table_details += details

        if not arguments.skip_individual:
            print("  numbered scripts as separate processes:")
            logs, rows, summaries, details = run_individual(arguments, run, eda_data, scratch)
            figure_rows += rows
            table_summaries += summaries
            table_details += details
            for log in logs:
                check(f"{log['script']} exits 0 (run outside the repository)", 0,
                      log["return_code"], log["return_code"] == 0)
                if log["script"].endswith(("Figure1.py", "Figure2.py", "Figure3.py",
                                           "Table1.py")):
                    check(f"{log['script']} runs no GP fit stage", False,
                          log["gp_fit_stage_entered"], not log["gp_fit_stage_entered"])
                else:
                    check(f"{log['script']} reuses checkpoints (no new fit)", False,
                          log["new_fits"], not log["new_fits"])

    for name, path in checkpoints.items():
        after = po.sha256(path) if path.is_file() else None
        check(f"checkpoint {name}: SHA-256 at start / at end (changes only when fits are "
              f"added)", checkpoint_before[name],
              "unchanged" if after == checkpoint_before[name] else f"changed: {after}", True)

    for item in arguments.compare:
        label, _, folder = item.partition("=")
        if not folder:
            raise SystemExit(f"--compare expects LABEL=DIR, not {item!r}")
        rows, summaries, details = compare_external(label, Path(folder).expanduser())
        figure_rows += rows
        table_summaries += summaries
        table_details += details

    # LF on every OS, so the hashes in manifest.csv and SHA256SUMS.txt survive eol=lf.
    pd.DataFrame(published_rows, columns=["output", "item", "published", "computed",
                                          "full_precision", "passed"]).to_csv(
        VALIDATION_DIR / "manuscript_comparison.csv", index=False, encoding="utf-8",
        lineterminator="\n")
    pd.DataFrame(run_rows).to_csv(VALIDATION_DIR / "run_checks.csv", index=False,
                                  encoding="utf-8", lineterminator="\n")
    pd.DataFrame(figure_rows).to_csv(VALIDATION_DIR / "figure_checks.csv", index=False,
                                     encoding="utf-8", lineterminator="\n")
    pd.DataFrame(table_summaries).to_csv(VALIDATION_DIR / "table_comparison.csv", index=False,
                                         encoding="utf-8", lineterminator="\n")
    pd.DataFrame(table_details, columns=["table", "comparison", "row", "column", "left", "right",
                                         "kind", "abs_difference"]).to_csv(
        VALIDATION_DIR / "table_mismatches.csv", index=False, encoding="utf-8",
        lineterminator="\n")

    banner("9. Manifest")
    manifest = []
    for item, paths in sorted(outputs.items(), key=lambda kv: (kv[0].split()[0],
                                                                int(kv[0].split()[1]))):
        kind, number = item.split()
        number = int(number)
        if kind == "Figure" and number <= 3:
            script = f"scripts/figures/Figure{number}.py"
            source = f"scripts/eda_common.py: {EDA_SOURCES[number]}"
            data = f"{eda_common.DATA_NAME} [{arguments.sheet}]"
        else:
            folder = "text" if kind == "Section" else f"{kind.lower()}s"
            script = f"scripts/{folder}/{kind}{number}.py"
            source = f"abalone_median.py: {po.SOURCES[item]}"
            fits = {"Figure": po.FIGURE_FITS, "Table": po.TABLE_FITS,
                    "Section": po.TEXT_FITS}[kind][number]
            data = po.DATA_NAME + (f" + checkpointed {', '.join(fits)} fits" if fits else "")
        for path in paths:
            manifest.append({"output": item, "file": relative(path),
                             "bytes": path.stat().st_size, "sha256": po.sha256(path),
                             "script": script, "source": source, "input": data})
    pd.DataFrame(manifest).to_csv(MANIFEST, index=False, encoding="utf-8", lineterminator="\n")
    print(f"  {relative(MANIFEST)} ({len(manifest)} files)")

    failed = required_failures(published_rows, run_rows, figure_rows, table_summaries,
                               example=arguments.example)
    print(f"\n  published values {sum(r['passed'] for r in published_rows)}/"
          f"{len(published_rows)} match"
          + (" (informational for synthetic inputs)" if arguments.example else ""))
    print(f"  run checks      {sum(r['passed'] for r in run_rows)}/{len(run_rows)} passed")
    print(f"  figure checks   {sum(r['passed'] for r in figure_rows)}/{len(figure_rows)} passed")
    print(f"  table compares  {sum(r['passed'] for r in table_summaries)}/"
          f"{len(table_summaries)} passed ({len(table_details)} cell mismatches)")
    print(f"  details in {relative(VALIDATION_DIR)}/")
    print(f"\nTotal {time.perf_counter() - started:.0f} s")
    if failed:
        print("\nFAILED:\n  " + "\n  ".join(failed))
        return 1
    print("All required example checks passed." if arguments.example else "All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
