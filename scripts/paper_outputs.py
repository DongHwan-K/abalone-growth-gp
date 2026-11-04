"""Shared runner for manuscript Figures 4-6, Tables 1-5 and the Section 3 values.

Every number comes from ``abalone_median.py``.  This module
sets its input workbook, point summary (median) and output folders, reuses the
fits in ``<work-dir>/_checkpoints_S2000`` (fitting only what is missing), calls
the functions that build each output, and copies the files abalone_median.py
writes under its internal names to their manuscript-numbered names.

The table builders below present the calculations as the five numbered
manuscript tables.  Supporting ablation and interval calculations stay internal
and are not assigned additional table numbers.  The numbers the Results text
(Section 3) quotes are collected in one list, ``results/text/Section3_values.csv``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SOURCE = REPO_ROOT / "abalone_median.py"
EXPECTED_SOURCE_SHA256 = "91ae5a6beebf89815ad638ac26f4ec86ebc214684a67dd1326bb5a24051408fb"
DATA_NAME = "전복데이터.xlsx"
POINT_SUMMARY = "median"
DEFAULT_WORK_DIR = REPO_ROOT / "abalone_paper_results"
FIGURE_OUTPUT = REPO_ROOT / "results" / "figures"
TABLE_OUTPUT = REPO_ROOT / "results" / "tables"
TEXT_OUTPUT = REPO_ROOT / "results" / "text"

# number -> (name abalone_median.py writes, manuscript-numbered name)
FIGURES = {
    4: ("fig_residual_qq", "Figure4_residual_qq"),
    5: ("fig_intervals_p80", "Figure5_intervals_p80"),
    6: ("fig_growth_curves_shared_p80", "Figure6_growth_curves_shared_p80"),
}
TABLES = {
    1: ("table1_facilities", "Table1_facilities"),
    2: ("table2_component_gp", "Table2_component_gp"),
    3: ("table3_comparators", "Table3_comparators"),
    4: ("table4_kernel_ablation", "Table4_kernel_ablation"),
    5: ("table5_framework_variants", "Table5_framework_variants"),
}
# section -> (name written beside abalone_median.py's tables, numbered name)
TEXTS = {
    3: ("section3_values", "Section3_values"),
}

# Fit stages: "base" = the 78 stacking fits, "figure" = the figure states file
# (with its one composite fit), "ablation" = the 275 ablation fits.  Figure and
# ablation fits read their seeds from the base fits.
STAGE_NEEDS = {"base": (), "figure": ("base",), "ablation": ("base",)}
FIGURE_FITS = {4: ("base",), 5: ("figure",), 6: ("figure",)}
TABLE_FITS = {1: (), 2: ("base",), 3: ("base",), 4: ("base",), 5: ("ablation",)}
TEXT_FITS = {3: ("base", "ablation")}

# The abalone_median.py functions behind each output (for the manifest).
SOURCES = {
    "Figure 4": "residual_normality, figure_qq",
    "Figure 5": "assemble_figures, table_intervals_methods, figure_intervals",
    "Figure 6": "assemble_figures, growth_panels",
    "Table 1": "table1_facilities",
    "Table 2": "assemble, table_models",
    "Table 3": "growth_curve_run, table_models",
    "Table 4": "assemble, table_ablation",
    "Table 5": "gather, table_e1, table_e2, table_e3",
    "Section 3": "table_models, growth_curve_run, residual_normality, table_intervals, "
                 "table_ablation, table_e1, table_e2, table_e3",
}

_ANALYSIS = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analysis_module():
    """Import abalone_median.py from the repository root (once)."""
    global _ANALYSIS
    if _ANALYSIS is None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        actual = sha256(ANALYSIS_SOURCE)
        if actual != EXPECTED_SOURCE_SHA256:
            print(f"WARNING: abalone_median.py SHA-256 is {actual}, not the released "
                  f"{EXPECTED_SOURCE_SHA256}", file=sys.stderr)
        import abalone_median

        _ANALYSIS = abalone_median
    return _ANALYSIS


def resolve_data(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        missing_data(f"Input workbook not found: {candidate}")
    for candidate in (REPO_ROOT / "input_data" / DATA_NAME, REPO_ROOT / DATA_NAME):
        if candidate.is_file():
            return candidate.resolve()
    missing_data("Input workbook not found.")


def missing_data(message: str) -> None:
    am = analysis_module()
    columns = [am.TARGET_COL, am.RAW_WEIGHT_COL, am.COHORT_COL, am.TIME_COL, am.AGE_COL,
               *am.ENV_COLS]
    print(message, file=sys.stderr)
    print(f"\nFigures 4-6 and Tables 1-5 need '{DATA_NAME}' in {REPO_ROOT / 'input_data'}\n"
          f"or {REPO_ROOT} (or pass --data).  Required columns: {', '.join(columns)}\n"
          "No substitute data are generated.", file=sys.stderr)
    raise SystemExit(2)


def checkpoint_files(work_dir: Path) -> dict[str, Path]:
    am = analysis_module()
    folder = Path(work_dir) / f"_checkpoints_S{am.MC_DRAWS}"
    return {name: folder / name for name in
            ("fits.jsonl", "ablation_fits.jsonl", f"abalone_{POINT_SUMMARY}_states.jsonl")}


class PaperRun:
    """One configured abalone_median.py session; every step is computed once."""

    def __init__(self, data=None, work_dir=None, workers: int = 3):
        am = self.am = analysis_module()
        self.data = resolve_data(data)
        self.work_dir = Path(work_dir).expanduser().resolve() if work_dir else DEFAULT_WORK_DIR
        self.workers = int(workers)
        am.DATA_PATH = self.data
        am.set_point_summary(POINT_SUMMARY)
        am.set_output_root(self.work_dir)
        for path in (am.OUT, am.CKPT, am.VERSION_DIR, am.TABLE_DIR, am.FIGURE_DIR,
                     am.ABLATION_OUT, am.GROWTH_OUT):
            path.mkdir(parents=True, exist_ok=True)
        self._cache: dict = {}
        self._stages: set[str] = set()
        print(f"Analysis   {ANALYSIS_SOURCE}")
        print(f"Data       {self.data}")
        print(f"Point      {am.POINT_SUMMARY}")
        print(f"Checkpoint {am.CKPT}")
        for name, path in checkpoint_files(self.work_dir).items():
            lines = (sum(1 for line in path.open(encoding="utf-8") if line.strip())
                     if path.is_file() else 0)
            print(f"           {name}: {f'{lines} records (reused)' if lines else 'absent'}")

    def _once(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    # -- fits ----------------------------------------------------------------
    def ensure_fits(self, stages) -> None:
        """Run the fit stages ``stages`` need; checkpointed fits are only read."""
        am = self.am
        for stage in stages:
            if stage in self._stages:
                continue
            self.ensure_fits(STAGE_NEEDS[stage])
            if stage == "base":
                am.stage_fits(self.context(), self.workers)
            elif stage == "ablation":
                am.stage_ablation_fits(self.ablation_context(), self.workers)
            elif stage == "figure":
                am.banner("Figure state — monotone composite (1 fit)")
                am.stage_figure_fits(self.context(), self.ablation_context())
            self._stages.add(stage)

    # -- abalone_median.py steps, in the order main() calls them ---------------
    def context(self):
        return self._once("context", self.am.build_context)

    def ablation_context(self):
        return self._once("ablation_context", self.am.build_ablation_context)

    def parts(self):
        self.ensure_fits(("base",))
        return self._once("parts", lambda: self.am.assemble(self.context()))

    def models(self):
        return self._once("models", lambda: self.am.table_models(self.context(), self.parts()))

    def kernel_ablation(self):
        return self._once("kernel_ablation",
                          lambda: self.am.table_ablation(self.context(), self.parts()))

    def intervals(self):
        return self._once("intervals",
                          lambda: self.am.table_intervals(self.context(), self.parts()))

    def growth(self):
        return self._once("growth", self.am.growth_curve_run)

    def normality(self):
        return self._once("normality",
                          lambda: self.am.residual_normality(self.context(), self.parts()))

    def arms(self):
        def build():
            am = self.am
            self.ensure_fits(("ablation",))
            previous = am.load_previous()
            mine = am.AblationStore(am.ABLATION_CHECK / f"{am.ABLATION_PREFIX}_fits.jsonl",
                                    append=False).records
            arms, status = am.gather(self.ablation_context(), previous, mine)
            incomplete = [(arm, state) for arm, state in status.items() if state != "complete"]
            if incomplete:
                raise RuntimeError(f"ablation arms incomplete: {incomplete}")
            return arms, status, mine
        return self._once("arms", build)

    def e1_stacks(self):
        return self._once("e1", lambda: self.am.table_e1(self.ablation_context(),
                                                         self.arms()[0]))

    def e2(self):
        return self._once("e2", lambda: self.am.table_e2(self.ablation_context(),
                                                         self.arms()[0], self.e1_stacks()))

    def e3(self):
        arms, status, _ = self.arms()
        return self._once("e3", lambda: self.am.table_e3(self.ablation_context(), arms, status))

    def confounders(self):
        arms, status, mine = self.arms()
        return self._once("confounders", lambda: self.am.table_confounders(arms, status, mine))

    def figure_parts(self):
        self.ensure_fits(("figure",))
        return self._once("figure_parts", lambda: self.am.assemble_figures(self.context()))

    def figure_frame(self):
        return self._once("figure_frame", lambda: self.am.table_intervals_methods(
            self.context(), self.figure_parts()))

    # -- manuscript tables -----------------------------------------------------
    def table(self, number: int):
        return self._once(f"table{number}", getattr(self, f"_table{number}"))

    def _table1(self):
        import pandas as pd

        source = self.am.table1_facilities()
        dates = pd.to_datetime(source["stocking_date"])
        stocked = [f"{date.day} {date.strftime('%b %Y')}" for date in dates]
        occasions = source.train_occasions.astype(int) + source.test_occasions.astype(int)
        return pd.DataFrame({
            "Nursery facility": source.facility,
            "Location": source.region,
            "Stocked": stocked,
            "Sampling period (n)": [f"Apr–Nov 2024 ({n})" for n in occasions],
            "Individuals/date": 30,
        })

    def _table2(self):
        am, models = self.am, self.models()
        block = models[models.Table.eq("Table 2")]
        table = block[["Model", "Weight", *am.METRICS]].copy().reset_index(drop=True)
        table["Weight"] = table["Weight"].map(lambda value: "—" if value != value else value)
        return table.rename(columns={
            "g_RMSE": "RMSE (g)", "g_MAE": "MAE (g)",
            "g_MAPE": "MAPE (%)", "g_R2_pred": "R2_pred",
        })

    def _table3(self):
        import pandas as pd

        am, models, growth = self.am, self.models(), self.growth()
        stacking = models[(models.Table == "Table 2") & (models.Model == "Stacking")].iloc[0]
        blank = {metric: "" for metric in am.METRICS}
        rows = [{"Model": "Stacking", "Inputs": "full",
                 **{metric: stacking[metric] for metric in am.METRICS}},
                {"Model": "Parametric growth curves", "Inputs": "", **blank}]
        for record in growth["table"].itertuples():
            model = "von Bertalanffyᵃ" if record.model == "von Bertalanffy" else record.model
            rows.append({"Model": model,
                         "Inputs": "day-age", **{metric: getattr(record, metric)
                                                  for metric in am.METRICS}})
        rows.append({"Model": "Machine learning", "Inputs": "", **blank})
        ml = models[models.Block.eq("baseline")].set_index("Model")
        for name in ("Bayesian ridge", "Random forest", "XGBoost"):
            record = ml.loc[name]
            rows.append({"Model": name, "Inputs": "full",
                         **{metric: record[metric] for metric in am.METRICS}})
        return pd.DataFrame(rows).rename(columns={
            "g_RMSE": "RMSE (g)", "g_MAE": "MAE (g)",
            "g_MAPE": "MAPE (%)", "g_R2_pred": "R2_pred",
        })

    def _table4(self):
        am = self.am
        table = self.kernel_ablation()[["Model", *am.METRICS]].copy()
        table["Model"] = table["Model"].replace({
            "Without Group-additive SE": "Without group-additive SE",
            "Without Neural network": "Without neural network",
        })
        return table.rename(columns={
            "g_RMSE": "RMSE (g)", "g_MAE": "MAE (g)",
            "g_MAPE": "MAPE (%)", "g_R2_pred": "R2_pred",
        }).reset_index(drop=True)

    def environmental_detail(self):
        import numpy as np
        import pandas as pd

        def build():
            am = self.am
            self.e1_stacks()                   # writes the E1 files read below
            e1 = pd.read_csv(am.ablation_out("01", "E1_age_only_models"))
            e1_cp = pd.read_csv(am.ablation_out("01d", "E1_intervals")).set_index("inputs")
            e1 = e1[["inputs", "model", "weight", *am.METRICS, "expH"]].copy()
            stacking = e1.model.eq("Stacking")
            e1["cp80_covered"] = [f"{int(e1_cp.loc[i, 'cp_covered'])}/8" if s else ""
                                  for i, s in zip(e1.inputs, stacking)]
            e1["cp80_width_g"] = [float(e1_cp.loc[i, "cp_width_g"]) if s else np.nan
                                  for i, s in zip(e1.inputs, stacking)]
            return e1
        return self._once("environmental_detail", build)

    def weight_detail(self):
        import pandas as pd

        def build():
            am = self.am
            self.e2()                          # writes the E2 file read below
            e2 = pd.read_csv(am.ablation_out("02", "E2_equal_weights"))
            e2 = e2[e2.weights.isin(["LOO-optimal", "equal"])].reset_index(drop=True)
            return pd.DataFrame({
                "weights": e2.weights, "w": e2.w, **{c: e2[c] for c in am.METRICS},
                "mixture_loo_lpd_g": e2.mixture_loo_lpd_g,
                "cp80_covered": [f"{int(v)}/8" for v in e2.cp_covered],
                "cp80_width_g": e2.cp_width_g})
        return self._once("weight_detail", build)

    def constraint_detail(self):
        am = self.am
        return self._once("constraint_detail", lambda: self.e3()[
            ["support", "arm", "kernels", "weights", "expH", *am.METRICS,
             "down_train", "down_test", "audit_grid"]].reset_index(drop=True))

    def _table5(self):
        import pandas as pd

        am = self.am
        environment = self.environmental_detail()
        weights = self.weight_detail()
        constraints = self.constraint_detail()
        full = environment[(environment.inputs == "full") &
                           (environment.model == "Stacking")].iloc[0]
        age_only = environment[(environment.inputs == "age-only") &
                               (environment.model == "Stacking")].iloc[0]
        equal = weights[weights.weights == "equal"].iloc[0]
        unconstrained = constraints[(constraints.support == "3-kernel") &
                                    constraints.arm.str.contains("unconstrained")].iloc[0]
        rows = [
            {"Variant": "Stacking (proposed)", **{m: full[m] for m in am.METRICS}},
            {"Variant": "Without environmental covariates",
             **{m: age_only[m] for m in am.METRICS}},
            {"Variant": "With equal weights", **{m: equal[m] for m in am.METRICS}},
            {"Variant": "Without the monotone construction",
             **{m: unconstrained[m] for m in am.METRICS}},
        ]
        return pd.DataFrame(rows).rename(columns={
            "g_RMSE": "RMSE (g)", "g_MAE": "MAE (g)",
            "g_MAPE": "MAPE (%)", "g_R2_pred": "R2_pred",
        })

    # -- numbers quoted in the Results text -------------------------------------
    def text(self, number: int):
        return self._once(f"text{number}", getattr(self, f"_section{number}"))

    def _section3(self):
        """Every number the Section 3 text quotes, in the order the text quotes it.

        Columns: ``section`` (subsection), ``item``, ``value`` (full precision; a
        count is written as ``k/n``) and ``printed`` (``value`` at the number of
        decimals the text uses).  Differences between table entries are taken
        at full precision, before rounding.
        """
        import pandas as pd

        am = self.am
        rows = []

        def add(section, item, value, decimals=None, signed=False):
            if isinstance(value, str):
                printed = value
            else:
                value = float(value)
                printed = f"{value:{'+' if signed else ''}.{decimals}f}"
                if float(printed) == 0.0:
                    printed = printed.lstrip("+-")
            rows.append({"section": section, "item": item, "value": value,
                         "printed": printed})

        def label(kernel):
            return am.KERNEL_LABEL[kernel]

        def code_of(facility):
            return next(code for code, name in am.ROMAN.items() if name == facility)

        # 3.1 Predictive accuracy ------------------------------------------------
        normality = self.normality()[0].set_index("kernel")
        for kernel in am.KERNELS:
            add("3.1", f"Shapiro–Wilk p, {label(kernel)}",
                normality.loc[label(kernel), "shapiro_p"], 3)

        t2 = self.table(2).set_index("Model")
        t3 = self.table(3).set_index("Model")
        stacking, additive = t2.loc["Stacking"], t2.loc[label("RBF_add3")]
        add("3.1", "Stacking vs group-additive SE: RMSE reduction (g)",
            float(additive["RMSE (g)"]) - float(stacking["RMSE (g)"]), 3)
        add("3.1", "Stacking vs group-additive SE: MAE reduction (g)",
            float(additive["MAE (g)"]) - float(stacking["MAE (g)"]), 3)
        add("3.1", "Stacking vs group-additive SE: MAPE reduction (percentage points)",
            float(additive["MAPE (%)"]) - float(stacking["MAPE (%)"]), 2)
        candidates = t2.drop(index="Stacking")["MAPE (%)"].astype(float)
        add("3.1", "Individual GP candidates: lowest MAPE (%)", candidates.min(), 2)
        add("3.1", "Individual GP candidates: highest MAPE (%)", candidates.max(), 2)

        von_bertalanffy = t3.loc["von Bertalanffyᵃ"]
        add("3.1", "von Bertalanffy minus stacking: MAPE (percentage points)",
            float(von_bertalanffy["MAPE (%)"]) - float(stacking["MAPE (%)"]), 2)
        add("3.1", "von Bertalanffy minus stacking: RMSE (g)",
            float(von_bertalanffy["RMSE (g)"]) - float(stacking["RMSE (g)"]), 2)

        train = self.context()["train"]
        ratios = []
        for _, block in train.groupby(am.COHORT_COL, observed=True):
            weight = block.sort_values(am.AGE_COL)["_w"].to_numpy(float)
            ratios.append(weight[-1] / weight[-2])
        add("3.1", "Last / preceding training mean weight: lowest facility ratio",
            min(ratios), 2)
        add("3.1", "Last / preceding training mean weight: highest facility ratio",
            max(ratios), 2)

        growth = self.growth()
        params = growth["params"].set_index(["model", "facility"])
        daru = "Jindo-Daru"
        daru_train_max = float(train.loc[train[am.COHORT_COL].astype(str).eq(code_of(daru)),
                                         "_w"].astype(float).max())
        vb_daru = params.loc[("vonBertalanffy", daru)]
        gz_daru = params.loc[("Gompertz", daru)]
        add("3.1", "Jindo-Daru: von Bertalanffy asymptote (g)", vb_daru["W_inf"], 0)
        add("3.1", "Jindo-Daru: von Bertalanffy asymptote standard error (g)",
            vb_daru["W_inf_se"], 0)
        add("3.1", "Jindo-Daru: training maximum mean weight (g)", daru_train_max, 2)
        add("3.1", "Jindo-Daru: Gompertz asymptote (g)", gz_daru["W_inf"], 3)
        add("3.1", "Jindo-Daru: Gompertz asymptote standard error (g)", gz_daru["W_inf_se"], 3)
        add("3.1", "Jindo-Daru: Gompertz asymptote / training maximum",
            float(gz_daru["W_inf"]) / daru_train_max, 2)
        by_facility = growth["by_facility"].set_index(["model", "facility"])
        add("3.1", "Jindo-Daru: Gompertz facility MAPE (%)",
            by_facility.loc[("Gompertz", daru), "g_MAPE"], 2)
        target = growth["target"]
        daru_test = target[target[am.COHORT_COL].astype(str).eq(code_of(daru))] \
            .sort_values(am.AGE_COL)["_w"].astype(float)
        for occasion, weight in zip(("t7", "t8"), daru_test):
            add("3.1", f"Jindo-Daru: observed test mean weight, {occasion} (g)", weight, 2)
        add("3.1", "Bayesian ridge minus von Bertalanffy: MAPE (percentage points)",
            float(t3.loc["Bayesian ridge", "MAPE (%)"]) - float(von_bertalanffy["MAPE (%)"]), 2)
        add("3.1", "Table 3 footnote, Jindo-Daru: von Bertalanffy asymptote relative "
                   "standard error", vb_daru["W_inf_rel_se"], 1)

        # 3.2 Uncertainty quantification and calibration --------------------------
        intervals = self.intervals()
        test_rows = intervals[intervals.Facility.ne("ALL")]
        total = intervals[intervals.Facility.eq("ALL")].iloc[0]
        n_test = len(test_rows)
        native_covered = int(test_rows.Native_covers.astype(bool).sum())
        calibrated_covered = int(test_rows.Conformal_covers.astype(bool).sum())
        add("3.2", "Native 80% interval: test observations covered",
            f"{native_covered}/{n_test}")
        add("3.2", "Native 80% interval: mean width (g)", total.Native_width_g, 2)
        add("3.2", "Residual-calibrated 80% interval: mean width (g)",
            total.Conformal_width_g, 2)
        add("3.2", "Reduction of the mean width (%)", total.Width_reduction_pct, 1)
        add("3.2", "Residual-calibrated 80% interval: test observations covered",
            f"{calibrated_covered}/{n_test}")
        add("3.2", "Residual-calibrated 80% interval: empirical coverage (%)",
            100.0 * calibrated_covered / n_test, 1)
        for occasion in ("t7", "t8"):
            add("3.2", f"Mean per-sample width reduction, {occasion} (%)",
                test_rows[test_rows.Occasion.eq(occasion)].Width_reduction_pct.astype(float)
                .mean(), 1)

        # 3.4 Model component comparisons -----------------------------------------
        for kernel in ("RBF_add3", "NNKernel", "Matern0.5"):
            add("3.4", f"Stacking weight, {label(kernel)}",
                float(t2.loc[label(kernel), "Weight"]), 4)
        ablation = self.kernel_ablation()
        all_three = ablation[ablation.Removed.eq("—")].iloc[0]
        add("3.4", "Effective number of components", all_three.expH, 4)
        for kernel in ("NNKernel", "Matern0.5", "RBF_add3"):
            row = ablation[ablation.Removed.eq(kernel)].iloc[0]
            add("3.4", f"MAPE increase without {label(kernel)} (percentage points)",
                float(row.g_MAPE) - float(all_three.g_MAPE), 2)

        environment = self.environmental_detail()
        full_inputs = environment[environment.inputs.eq("full")]
        age_inputs = environment[environment.inputs.eq("age-only")]
        full_stack = full_inputs[full_inputs.model.eq("Stacking")].iloc[0]
        age_stack = age_inputs[age_inputs.model.eq("Stacking")].iloc[0]
        add("3.4", "Without environmental covariates: stacking MAPE increase "
                   "(percentage points)", float(age_stack.g_MAPE) - float(full_stack.g_MAPE), 2)
        age_candidate = {}
        for kernel in am.KERNELS:
            full_row = full_inputs[full_inputs.model.eq(kernel)].iloc[0]
            age_row = age_inputs[age_inputs.model.str.startswith(f"{kernel} [")].iloc[0]
            age_candidate[kernel] = age_row
            add("3.4", f"Without environmental covariates: {label(kernel)} candidate MAPE "
                       f"increase (percentage points)",
                float(age_row.g_MAPE) - float(full_row.g_MAPE), 2)
        add("3.4", "Without environmental covariates: Neural network weight",
            age_candidate["NNKernel"].weight, 4)
        add("3.4", "Without environmental covariates: effective number of components",
            age_stack.expH, 4)
        add("3.4", "Without environmental covariates: stacking MAPE (%)", age_stack.g_MAPE, 4)
        add("3.4", "Without environmental covariates: group-additive SE candidate MAPE (%)",
            age_candidate["RBF_add3"].g_MAPE, 4)
        add("3.4", "Residual-calibrated 80% mean width, proposed (g)",
            full_stack.cp80_width_g, 4)
        add("3.4", "Residual-calibrated 80% mean width, without environmental covariates (g)",
            age_stack.cp80_width_g, 4)
        add("3.4", "Residual-calibrated 80% test observations covered, proposed",
            str(full_stack.cp80_covered))
        add("3.4", "Residual-calibrated 80% test observations covered, without environmental "
                   "covariates", str(age_stack.cp80_covered))

        weights = self.weight_detail().set_index("weights")
        estimated, equal = weights.loc["LOO-optimal"], weights.loc["equal"]
        add("3.4", "Equal weights: stacking MAPE increase (percentage points)",
            float(equal.g_MAPE) - float(estimated.g_MAPE), 2)
        add("3.4", "LOO mixture log predictive density (grams), estimated weights",
            estimated.mixture_loo_lpd_g, 4, signed=True)
        add("3.4", "LOO mixture log predictive density (grams), equal weights",
            equal.mixture_loo_lpd_g, 4, signed=True)
        add("3.4", "Equal weights: residual-calibrated 80% test observations covered",
            str(equal.cp80_covered))

        constraints = self.constraint_detail()
        three_kernel = constraints[constraints.support.eq("3-kernel")]
        direct = three_kernel[three_kernel.arm.str.startswith("(d)")].iloc[0]
        add("3.4", "Direct-response GP MAPE / proposed MAPE",
            float(direct.g_MAPE) / float(full_stack.g_MAPE), 2)
        add("3.4", "Direct-response GP: decreasing grid intervals in the test window",
            str(direct.down_test).replace(" ", ""))
        mingug = "Wando-Mingug"
        runs = pd.read_csv(am.ablation_out("03c", "E3_decreasing_runs"))   # written by e3()
        runs = runs[runs.arm.eq("(d)") & runs.cohort.astype(str).eq(code_of(mingug))
                    & runs.window.ne("train")]
        if runs.empty:
            raise RuntimeError(f"no decreasing test-window run of the direct-response GP "
                               f"at {mingug}")
        decline = runs.sort_values("drop_g", ascending=False).iloc[0]
        add("3.4", f"Direct-response GP, {mingug}: start of the largest decline, age (d)",
            decline.age_start, 0)
        add("3.4", f"Direct-response GP, {mingug}: start of the largest decline, "
                   f"predicted weight (g)", decline.g_start, 2)
        add("3.4", f"Direct-response GP, {mingug}: end of the largest decline, age (d)",
            decline.age_end, 0)
        add("3.4", f"Direct-response GP, {mingug}: end of the largest decline, "
                   f"predicted weight (g)", decline.g_end, 2)
        add("3.4", f"Direct-response GP, {mingug}: decline (%)", decline.drop_pct, 1)
        return pd.DataFrame(rows, columns=["section", "item", "value", "printed"])

    # -- writing ----------------------------------------------------------------
    def write_table(self, number: int, output_dir=None) -> Path:
        """Write Table ``number`` as abalone_median.py names it, then under its number."""
        am = self.am
        self.ensure_fits(TABLE_FITS[number])
        internal, final = TABLES[number]
        frame = self.table(number)
        source = am.TABLE_DIR / f"{internal}.csv"
        frame.to_csv(source, index=False, encoding=am.ENC, lineterminator="\n")  # LF on every OS
        output_dir = Path(output_dir) if output_dir else TABLE_OUTPUT
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{final}.csv"
        shutil.copyfile(source, target)
        print(f"Table {number}: {target}  ({len(frame)} rows; from {source.name})")
        return target

    def write_text(self, number: int, output_dir=None) -> Path:
        """Write the Section ``number`` values beside the tables, then under its number."""
        am = self.am
        self.ensure_fits(TEXT_FITS[number])
        internal, final = TEXTS[number]
        frame = self.text(number)
        source = am.TABLE_DIR / f"{internal}.csv"
        frame.to_csv(source, index=False, encoding=am.ENC, lineterminator="\n")  # LF on every OS
        output_dir = Path(output_dir) if output_dir else TEXT_OUTPUT
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{final}.csv"
        shutil.copyfile(source, target)
        print(f"Section {number}: {target}  ({len(frame)} values; from {source.name})")
        return target

    def draw_figure(self, number: int) -> None:
        am = self.am
        self.ensure_fits(FIGURE_FITS[number])
        if number == 4:
            normality, residuals = self.normality()
            am.figure_qq(self.context(), normality, residuals)
        elif number == 5:
            am.figure_intervals(self.context(), self.figure_parts(), self.figure_frame(),
                                level=am.NOMINAL)
        elif number == 6:
            am.growth_panels(self.context(), self.figure_parts(), mode="log",
                             level=am.NOMINAL, shared_log=True,
                             name=f"fig_growth_curves_shared{am.suffix_for(am.NOMINAL)}",
                             report=True)
        else:
            raise ValueError(f"Figure {number} is not produced by abalone_median.py")

    def write_figure(self, number: int, output_dir=None) -> list[Path]:
        """Draw Figure ``number`` with abalone_median.py, then copy it under its number."""
        am = self.am
        am.banner(f"Figure {number}")
        self.draw_figure(number)
        internal, final = FIGURES[number]
        output_dir = Path(output_dir) if output_dir else FIGURE_OUTPUT
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for suffix in (".pdf", ".png"):
            source = am.FIGURE_DIR / f"{am.FIGURE_PREFIX}_{internal}{suffix}"
            target = output_dir / f"{final}{suffix}"
            shutil.copyfile(source, target)
            paths.append(target)
            print(f"Figure {number}: {target}  (from {source.name})")
        return paths

    @contextlib.contextmanager
    def captured_figures(self, folder: Path):
        """Redraw into ``folder`` and keep each figure abalone_median.save() receives.

        For verification only: the original save() still writes and closes every
        figure, and nothing the figure functions compute is touched."""
        am = self.am
        original_save, original_dir = am.save, am.FIGURE_DIR
        captured = {}

        def save(figure, name):
            captured[name] = figure
            original_save(figure, name)

        am.save, am.FIGURE_DIR = save, Path(folder)
        try:
            yield captured
        finally:
            am.save, am.FIGURE_DIR = original_save, original_dir


# -----------------------------------------------------------------------------
# Command line of the numbered scripts
# -----------------------------------------------------------------------------
def parse_arguments(kind: str, number: int, argv=None, folder: str | None = None,
                    output: str | None = None):
    parser = argparse.ArgumentParser(
        description=f"Reproduce {output or f'manuscript {kind} {number}'} "
                    f"with abalone_median.py.")
    parser.add_argument("--data", help=f"analysis workbook (default: input_data/{DATA_NAME} "
                                       f"or {DATA_NAME} in the repository root)")
    parser.add_argument("--work-dir", help="abalone_median.py output root that holds "
                                           "_checkpoints_S2000 (default: abalone_paper_results/)")
    parser.add_argument("--output-dir", help=f"folder for the numbered file "
                                             f"(default: results/{folder or kind.lower() + 's'})")
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel fits if a checkpoint is missing (default 3)")
    return parser.parse_args(argv)


def reproduce_figure(number: int, argv=None) -> list[Path]:
    if number not in FIGURES:
        raise SystemExit(f"Figure {number} is produced by scripts/eda_common.py")
    arguments = parse_arguments("Figure", number, argv)
    run = PaperRun(arguments.data, arguments.work_dir, arguments.workers)
    return run.write_figure(number, arguments.output_dir)


def reproduce_table(number: int, argv=None) -> Path:
    if number not in TABLES:
        raise SystemExit(f"The manuscript contains Tables 1-{max(TABLES)}, not Table {number}.")
    arguments = parse_arguments("Table", number, argv)
    run = PaperRun(arguments.data, arguments.work_dir, arguments.workers)
    return run.write_table(number, arguments.output_dir)


def reproduce_text(number: int, argv=None) -> Path:
    if number not in TEXTS:
        raise SystemExit(f"Text values are produced for Section "
                         f"{', '.join(str(n) for n in TEXTS)}, not Section {number}.")
    arguments = parse_arguments("Section", number, argv, folder="text",
                                output=f"the numbers quoted in manuscript Section {number}")
    run = PaperRun(arguments.data, arguments.work_dir, arguments.workers)
    return run.write_text(number, arguments.output_dir)
