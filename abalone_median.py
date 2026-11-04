#!/usr/bin/env python3
"""Nursery Growth Prediction of Pacific Abalone Using Stacked Monotone Gaussian
Processes with Residual-Calibrated Prediction Intervals -- analysis script.

This single file computes every number and the model figures (manuscript
Figures 4-6) of the Results section, checks them against the published values,
and packs them into abalone_paper_package.zip.  The table numbers below are this
script's internal ledger names; scripts/paper_outputs.py assembles them into the
manuscript's Tables 1-5:

    Table 1  the four nursery facilities (section 2.1)
    Table 2  stacking vs its three component GP models           (gram scale)
    Table 3  von Bertalanffy, Gompertz, BayesianRidge, RandomForest, XGBoost
    Table 4  leave-one-kernel-out ablation
    Table 5  E1  environmental inputs removed (age + facility only)
    Table 6  E2  equal stacking weights
    Table 7  E3  how monotonicity is imposed -- (a) integral construction,
                 (b) soft derivative penalty, (c) Riihimaki-Vehtari,
                 (d) unconstrained GP; three-kernel and common two-kernel
    Table 8  E3  what else changes between the four arms
    Table 9  native vs conformal 80% intervals on the eight test rows
             + growth_curve_parameters.csv, by_facility.csv
    §3.1     residual_normality.csv: the 24 training residuals per kernel,
             standardised by the fitted noise sd, with Shapiro-Wilk W and p
    Figures  fig_intervals_p80, fig_growth_curves_shared_p80, fig_residual_qq

THE SETTING
-----------
    monotonicity   positive-growth integral construction (paper eq. 3-4)
    response       ln(total wet weight, g) -> Yeo-Johnson -> standardise
    kernels        Matern(nu=1/2), group-additive SE (RBF_add3),
                   neural network (NNKernel); all three monotone, all three
                   stacking candidates
    prediction     simultaneous -- six training occasions -> t7 and t8 at once
    stacking       Yao et al. (2018) objective (2.2) on pointwise full LOO
    point          from_z(ppf(0.5)): the median of the stacking mixture on the
                   modelling scale, mapped back to grams (POINT_SUMMARY =
                   "median"; --point mean selects from_z(mean_z)).  Every
                   reported RMSE / MAE / MAPE / R2_pred and the line in the
                   growth figure use it, and it is the centre of the conformal
                   interval below, so the point prediction and the interval
                   refer to the same quantity.  The native interval keeps the
                   model's own quantiles, and the stacking objective keeps its
                   moment-matched Gaussian LOO densities, so the weights are
                   unchanged by this choice
    intervals      gamma = 0 conformal.  ONE radius on the z scale: the
                   (n+1)-adjusted quantile of |truth_z - LOO mixture median| over
                   the 24 training rows.  At row i the LOO mixture mixes the
                   three GPs' held-out LOO predictive CDFs with ONE weight vector
                   -- the stacking weights estimated from the full 24-row LOO
                   density matrix -- and its median is the 0.5 quantile of that
                   mixture CDF, found numerically (loo_mixture_summary).  The
                   radius is added symmetrically to the full-data mixture median
                   and mapped back to grams.  This is NOT Jackknife+: no per-fold
                   endpoints, no order statistics of them.
                   (table_intervals_methods also computes a Jackknife+ row and
                   other gamma values, as comparisons; none reaches a table or
                   figure.)

WHAT IT RUNS
------------
354 Gaussian-process fits, three ML baselines and 2 x 4 growth-curve fits:

     72  LOO integral fits      3 kernels x 24 held-out training rows
      3  final integral fits    3 kernels on all 24 training rows
      3  unconstrained GP fits  the same kernels, used by the E3 (d) check
     75  E1 age-only fits       integral, [age, facility] inputs
     75  E3 (b) soft penalty    3 kernels x (24 LOO + final)
     50  E3 (c) Riihimaki-Vehtari  RBF_add3, NNKernel x (24 LOO + final);
                                Matern(1/2) has no derivative process
     75  E3 (d) unconstrained   3 kernels x (24 LOO + final)
      1  monotone composite     the curve 0905_abalone5's figure code loads
                                (not drawn: SHOW_BEST_COMPOSITE is False)

About 25 minutes at three workers on the reference machine.  Everything is
checkpointed to `_checkpoints_S<draws>/`, so an interrupted run resumes and a
second run
only rebuilds tables and figures.

WHERE THE CODE COMES FROM
-------------------------
The analysis was run as a chain of scripts that imported one another.  The
sections of this file are those scripts' code, cut out by exact source
extraction; where two scripts used the same name, one of them is renamed, and
each section's banner names its source file and every rename.  Nothing is
re-implemented.  The derivation was checked mechanically: for every carried-over
definition, the original's syntax tree with the declared renames applied equals
the syntax tree here.

    stacking fits, Tables 2-4, Table 9    0902_abalone2/3, 0905_abalone1/4/5
    E1, E2, E3, Table 8                   0914_abalone1(+_kernels)
      soft penalty                        0828_abalone5, 0905_abalone1
      Riihimaki-Vehtari                   0903_abalone5(+_kernels),
                                          0903_abalone8_kernels,
                                          0905_abalone1_kernels, 0905_abalone1
    von Bertalanffy, Gompertz             0915_abalone1
    figures                               0905_abalone5

The last section, "Package wiring", is new.  It maps checkpoint keys between
the scripts, selects published columns into the paper tables, compares them
with the manuscript values and builds the zip; the only values it computes
itself are the data summary of Table 1 and the per-facility MAPE of
by_facility.csv.

REPRODUCIBILITY
---------------
Every fit is seeded. `fit_integral` consumes its seed twice -- once for the
(deterministic) hyperparameter optimisation and once for the MC_DRAWS = 2,000
Monte Carlo posterior draws that form the predictive -- so the predictive mean
and sd are MC estimates and a different seed moves them by about 1e-1, which is
enough to shift a stacking weight by 0.06-0.12. The seeds below are therefore
not regenerated: they are the exact values the original run assigned, recorded
per fit, and are what makes this script reproduce the published table entries
rather than a fresh realisation of them. Every ablation fit takes the seed of
the stacking fit with the same kernel and the same held-out row (paired seeds,
0914_abalone1); only the integral construction consumes it.

Bitwise reproduction holds on the reference machine (Intel Xeon E5-1660 v4,
AVX2, no AVX-512; torch 2.5.1+cu121, gpytorch 1.15.2, OMP_NUM_THREADS=1, float64
throughout). A different CPU microarchitecture, BLAS build or torch build
changes the rounding order; agreement to ~1e-6 is the portable expectation. One
kernel, RBF_add3, has six free parameters and a flat marginal-likelihood surface
near the optimum, so its Adam argmax can tie and select a step 33 iterations
away; that moves the predictive by 1.3e-8, eight significant digits, and is not
an error.

DATA
----
Place `전복데이터.xlsx` beside this script (or in `input_data/`).  It is not part
of the package.  The analysis uses the 2024 spat-survey rows: four nursery
facilities x eight monthly occasions = 32 rows, the first six occasions per
facility for training (n_tr = 24) and the last two for test (n_test = 8).

USAGE
-----
    python abalone_median.py                      # fits, tables, figures, zip
    python abalone_median.py --workers 3          # parallel fits (default 3)
    python abalone_median.py --fits-only          # run the fits and stop
    python abalone_median.py --tables-only        # tables and the check; no
                                                  # figures, no zip
    python abalone_median.py --point mean         # mean instead of median point
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd
import torch
import gpytorch
import copy
import hashlib
import random
import threading
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import beta, norm
from sklearn.preprocessing import PowerTransformer, StandardScaler

torch.set_default_dtype(torch.float64)


# =============================================================================
# Configuration
# =============================================================================
DATA_PATH = SCRIPT_DIR / "전복데이터.xlsx"
OUT = SCRIPT_DIR / "abalone_paper_results"
# CKPT is defined with MC_DRAWS below: the number of Monte Carlo draws is part of
# what a checkpoint is, so every S gets its own directory.
ENC = "utf-8-sig"

COMBO = "Matern0.5+RBF_add3+NNKernel"
KERNELS = COMBO.split("+")
RESPONSE = "ln_weight_yj"          # ln(weight) -> Yeo-Johnson -> standardise
NOMINAL = 0.80                     # the level reported in the paper
N_TRAIN = 24

# Which summary of the predictive distribution IS the point prediction.  One
# switch, because a version that mixed the two would report an interval centred
# on a quantity other than the one its error metrics score.  It governs
# metrics_of, point_metrics (E1-E3), the conformity score and centre of the
# conformal interval, the line drawn in the growth figure and the curve its
# monotonicity audit reads, and by_facility.  It does NOT touch the native
# interval, which stays the model's own 10th and 90th percentiles, or the
# stacking weights, which stay the LOO predictive densities.  Under "median" the
# conformity score is |truth_z - median of the LOO mixture|, the mixture's own
# 0.5 quantile (loo_mixture_summary), not a blend of component medians.  mean_z
# is still what the LOO densities and the moment calculations use.
POINT_SUMMARY = "median"           # "median" -> from_z(ppf(0.5)); "mean" -> from_z(mean_z)


def point_z(component) -> np.ndarray:
    """The predictive summary POINT_SUMMARY names, on the modelling scale."""
    if POINT_SUMMARY == "mean":
        return np.asarray(component.mean_z, float)
    if POINT_SUMMARY == "median":
        return np.asarray(component.ppf(0.5), float)
    raise ValueError(f"POINT_SUMMARY is {POINT_SUMMARY!r}, not 'mean' or 'median'")

# Paper-facing names for the three kernels.
KERNEL_LABEL = {"Matern0.5": "Matérn(ν = 1/2)",
                "RBF_add3": "Group-additive SE",
                "NNKernel": "Neural network"}
ML_MODEL_NAMES = ["RandomForest", "BayesianRidge", "XGBoost"]
ML_LABEL = {"BayesianRidge": "Bayesian ridge", "RandomForest": "Random forest",
            "XGBoost": "XGBoost"}

# --- data columns (workbook is Korean; these are the source column names) -----
TARGET_COL = "로그무게"            # log weight column shipped in the workbook
COHORT_COL = "품종"                # facility / cohort
TIME_COL = "측정일"                # measurement date
AGE_COL = "일령"                   # day-age
ENV_COLS = ["월평균PH", "월평균염분", "월평균DO", "월평균수온"]  # pH, salinity, DO, temp
RAW_WEIGHT_COL = "전중(g)"         # arithmetic mean total wet weight, grams

# --- model constants ---------------------------------------------------------
SEED = 42
GRID_M = 4                  # integration subdivisions per observed-age interval
S0 = 1.0                    # prior sd of the facility-specific initial level f0
JITTER = 1e-8
JITTER_INT = JITTER
JITTER_RV = JITTER
NOISE_FLOOR = 1e-5
HYPER_LR = 0.05
HYPER_ITER_INTEGRAL = 500   # Adam steps, integral (monotone) construction
HYPER_ITER_PLAIN = 400      # Adam steps, plain GP
MC_DRAWS = 2000             # Monte Carlo posterior draws behind each predictive
# The predictive mean and sd are MC estimates, so S belongs to the run: fits made
# under different S are not interchangeable, and a checkpoint directory holding a
# mixture of them would be silently wrong -- `fit_key` alone cannot tell them
# apart.  Each S therefore keeps its own directory, and every fit record carries
# its own mc_draws.  Measured seed-to-seed spread at S = 2000 (10 repetitions):
# MAPE sd 0.48, stacking weights sd 0.06; at S = 10000, 0.20 and 0.03.
CKPT = OUT / f"_checkpoints_S{MC_DRAWS}"
GN_MAX_ITER = 40
GN_TOL = 1e-9
GLOBAL_NOISE_PRIOR_SHAPE = 2.0
GLOBAL_NOISE_PRIOR_MEAN = 0.05
KERNEL_NAMES = ["RBF", "Matern0.5", "Matern1.5", "Matern2.5", "RQ", "Linear",
                "Polynomial2"]
SCAN_TRAIN_POINTS = 80      # dense per-facility grid used to count decreasing
SCAN_TEST_POINTS = 50       # steps in Table 5
MONOTONE_TOL = 1e-9
BOOTSTRAP_DRAWS = 200
QUANTILE_LEVELS = np.concatenate([[0.0005, 0.001, 0.0025],
                                  np.arange(0.005, 0.99501, 0.005),
                                  [0.9975, 0.999, 0.9995]])

DPI = 180
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.titlesize": 14, "axes.unicode_minus": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
ROMAN = {"완도_금해": "Wando-Geumhae", "완도_민국": "Wando-Mingug",
         "진도_가반": "Jindo-Gaban", "진도_달우": "Jindo-Daru"}
SHORT = {"완도_금해": "W-Geum", "완도_민국": "W-Ming",
         "진도_가반": "J-Gaban", "진도_달우": "J-Daru"}


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


def out(name: str, suffix: str = "csv") -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{name}.{suffix}"

# =============================================================================
# Seeds
#
# The seed each fit was given in the original run.  See REPRODUCIBILITY above:
# these are recorded, not regenerated, because fit_integral spends its seed a
# second time on the Monte Carlo posterior draws.
# =============================================================================
SEEDS = {
    # --- Matern0.5 ---
    "final|Matern0.5": 1700093,
    "plain|Matern0.5": 5000250045,
    "loo|Matern0.5|0": 19300621, "loo|Matern0.5|1": 19400624,
    "loo|Matern0.5|2": 19500627, "loo|Matern0.5|3": 19600630,
    "loo|Matern0.5|4": 19700633, "loo|Matern0.5|5": 19800636,
    "loo|Matern0.5|6": 19900639, "loo|Matern0.5|7": 20000642,
    "loo|Matern0.5|8": 20100645, "loo|Matern0.5|9": 20200648,
    "loo|Matern0.5|10": 20300651, "loo|Matern0.5|11": 20400654,
    "loo|Matern0.5|12": 20500657, "loo|Matern0.5|13": 20600660,
    "loo|Matern0.5|14": 20700663, "loo|Matern0.5|15": 20800666,
    "loo|Matern0.5|16": 20900669, "loo|Matern0.5|17": 21000672,
    "loo|Matern0.5|18": 21100675, "loo|Matern0.5|19": 21200678,
    "loo|Matern0.5|20": 21300681, "loo|Matern0.5|21": 21400684,
    "loo|Matern0.5|22": 21500687, "loo|Matern0.5|23": 21600690,
    # --- RBF_add3 ---
    "final|RBF_add3": 5114653477,
    "plain|RBF_add3": 5000950066,
    "loo|RBF_add3|0": 5102553114, "loo|RBF_add3|1": 5102653117,
    "loo|RBF_add3|2": 5102753120, "loo|RBF_add3|3": 5102853123,
    "loo|RBF_add3|4": 5102953126, "loo|RBF_add3|5": 5103053129,
    "loo|RBF_add3|6": 5103153132, "loo|RBF_add3|7": 5103253135,
    "loo|RBF_add3|8": 5103353138, "loo|RBF_add3|9": 5103453141,
    "loo|RBF_add3|10": 5103553144, "loo|RBF_add3|11": 5103653147,
    "loo|RBF_add3|12": 5103753150, "loo|RBF_add3|13": 5103853153,
    "loo|RBF_add3|14": 5103953156, "loo|RBF_add3|15": 5104053159,
    "loo|RBF_add3|16": 5104153162, "loo|RBF_add3|17": 5104253165,
    "loo|RBF_add3|18": 5104353168, "loo|RBF_add3|19": 5104453171,
    "loo|RBF_add3|20": 5104553174, "loo|RBF_add3|21": 5104653177,
    "loo|RBF_add3|22": 5104753180, "loo|RBF_add3|23": 5104853183,
    # --- NNKernel ---
    "final|NNKernel": 5115053489,
    "plain|NNKernel": 5001350078,
    "loo|NNKernel|0": 5112153402, "loo|NNKernel|1": 5112253405,
    "loo|NNKernel|2": 5112353408, "loo|NNKernel|3": 5112453411,
    "loo|NNKernel|4": 5112553414, "loo|NNKernel|5": 5112653417,
    "loo|NNKernel|6": 5112753420, "loo|NNKernel|7": 5112853423,
    "loo|NNKernel|8": 5112953426, "loo|NNKernel|9": 5113053429,
    "loo|NNKernel|10": 5113153432, "loo|NNKernel|11": 5113253435,
    "loo|NNKernel|12": 5113353438, "loo|NNKernel|13": 5113453441,
    "loo|NNKernel|14": 5113553444, "loo|NNKernel|15": 5113653447,
    "loo|NNKernel|16": 5113753450, "loo|NNKernel|17": 5113853453,
    "loo|NNKernel|18": 5113953456, "loo|NNKernel|19": 5114053459,
    "loo|NNKernel|20": 5114153462, "loo|NNKernel|21": 5114253465,
    "loo|NNKernel|22": 5114353468, "loo|NNKernel|23": 5114453471,
}


# =============================================================================
# Data: workbook loading, facility aggregation, train / test split
# =============================================================================

CONT_COLS = [AGE_COL, *ENV_COLS]


YEAR_FILTER = 2024


N_TRAIN_TIMEPOINTS = 6


N_TEST_TIMEPOINTS = 2


N_REQUIRED_TIMEPOINTS = N_TRAIN_TIMEPOINTS + N_TEST_TIMEPOINTS


LOG_PARAMETER_PRIOR_LOC = 0.0


LOG_PARAMETER_PRIOR_SCALE = 1.0


def resolve_data_path(requested: str | Path) -> Path:
    requested = Path(requested)
    candidates = [
        requested,
        SCRIPT_DIR / requested.name,
        SCRIPT_DIR / "input_data" / requested.name,
        Path.cwd() / requested.name,
        Path.cwd() / "input_data" / requested.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    discovered = []
    for root in [SCRIPT_DIR, SCRIPT_DIR / "input_data", Path.cwd()]:
        if root.exists():
            discovered.extend(sorted(root.glob("*전복*.xlsx")))
            discovered.extend(sorted(root.glob("*abalone*.xlsx")))
    discovered = list(dict.fromkeys(path.resolve() for path in discovered))
    if len(discovered) == 1:
        return discovered[0]
    raise FileNotFoundError(
        "전복 데이터 파일을 찾지 못했습니다. 스크립트와 같은 폴더 또는 "
        "input_data 폴더에 전복데이터.xlsx를 두세요."
    )


def load_data(path: str | Path) -> pd.DataFrame:
    resolved = resolve_data_path(path)
    if resolved.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(resolved, engine="openpyxl")
    elif resolved.suffix.lower() == ".csv":
        frame = pd.read_csv(resolved, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported data format: {resolved.suffix}")

    print("Resolved data path:", resolved)
    print("Raw shape:", frame.shape)

    required = [TARGET_COL, COHORT_COL, TIME_COL, AGE_COL, *ENV_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")

    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="coerce")
    for column in [TARGET_COL, AGE_COL, *ENV_COLS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[COHORT_COL] = frame[COHORT_COL].astype("string").str.strip()

    frame = frame.loc[frame[TIME_COL].dt.year.eq(YEAR_FILTER)].copy()
    frame = frame.dropna(subset=required)
    frame = frame.sort_values(
        [COHORT_COL, TIME_COL, AGE_COL], kind="mergesort"
    ).reset_index(drop=True)

    cohort_levels = sorted(frame[COHORT_COL].astype(str).unique().tolist())
    if len(cohort_levels) < 2:
        raise ValueError("At least two cohorts are required.")

    # -------------------------------------------------------------------------
    # Strict temporal split within each cohort
    #
    # For every cohort, keep only the most recent 8 UNIQUE measurement times:
    #   timepoints 1-6 -> Train
    #   timepoints 7-8 -> Test
    #
    # All rows sharing the same measurement time receive the same split label.
    # Older timepoints outside the most recent 8 are excluded from this run.
    # -------------------------------------------------------------------------
    selected_blocks: list[pd.DataFrame] = []

    for cohort in cohort_levels:
        cohort_frame = (
            frame.loc[frame[COHORT_COL].astype(str).eq(cohort)]
            .copy()
            .sort_values([TIME_COL, AGE_COL], kind="mergesort")
        )

        unique_times = (
            cohort_frame[TIME_COL]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        n_timepoints = len(unique_times)

        if n_timepoints < N_REQUIRED_TIMEPOINTS:
            raise ValueError(
                f"Cohort {cohort!r} has only {n_timepoints} unique timepoints, "
                f"but {N_REQUIRED_TIMEPOINTS} are required "
                f"({N_TRAIN_TIMEPOINTS} train + {N_TEST_TIMEPOINTS} test)."
            )

        selected_times = unique_times.iloc[-N_REQUIRED_TIMEPOINTS:].reset_index(drop=True)
        train_times = selected_times.iloc[:N_TRAIN_TIMEPOINTS]
        test_times = selected_times.iloc[N_TRAIN_TIMEPOINTS:]

        cohort_frame = cohort_frame.loc[
            cohort_frame[TIME_COL].isin(selected_times)
        ].copy()

        cohort_frame["is_test"] = cohort_frame[TIME_COL].isin(test_times)
        cohort_frame["split_role"] = np.where(
            cohort_frame["is_test"], "Test", "Train"
        )

        timepoint_map = {
            timestamp: order
            for order, timestamp in enumerate(selected_times, start=1)
        }
        cohort_frame["selected_timepoint"] = (
            cohort_frame[TIME_COL].map(timepoint_map).astype(int)
        )

        test_horizon_map = {
            timestamp: horizon
            for horizon, timestamp in enumerate(test_times, start=1)
        }
        cohort_frame["test_horizon"] = 0
        test_mask = cohort_frame["is_test"]
        cohort_frame.loc[test_mask, "test_horizon"] = (
            cohort_frame.loc[test_mask, TIME_COL]
            .map(test_horizon_map)
            .astype(int)
        )

        # Retain the original downstream convention: zero-based within-cohort order.
        cohort_frame["cohort_order"] = cohort_frame["selected_timepoint"] - 1
        selected_blocks.append(cohort_frame)

    frame = pd.concat(selected_blocks, ignore_index=True)
    frame[COHORT_COL] = pd.Categorical(
        frame[COHORT_COL].astype(str),
        categories=cohort_levels,
        ordered=True,
    )
    frame = frame.sort_values(
        [COHORT_COL, "selected_timepoint", TIME_COL, AGE_COL],
        kind="mergesort",
    ).reset_index(drop=True)

    print("\nStrict temporal split")
    print(
        f"Per cohort: most recent {N_REQUIRED_TIMEPOINTS} timepoints -> "
        f"Train {N_TRAIN_TIMEPOINTS} + Test {N_TEST_TIMEPOINTS}"
    )
    for cohort, group in frame.groupby(COHORT_COL, observed=True, sort=False):
        train_group = group.loc[~group["is_test"]]
        test_group = group.loc[group["is_test"]]
        test_dates = (
            test_group[TIME_COL]
            .drop_duplicates()
            .sort_values()
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        print(
            f"{cohort}: "
            f"Train={train_group[TIME_COL].nunique()} timepoints, "
            f"Test={test_group[TIME_COL].nunique()} timepoints, "
            f"Test dates={test_dates}"
        )

    # Hard checks so an accidental split change cannot pass silently.
    for cohort, group in frame.groupby(COHORT_COL, observed=True, sort=False):
        n_train_times = int(group.loc[~group["is_test"], TIME_COL].nunique())
        n_test_times = int(group.loc[group["is_test"], TIME_COL].nunique())
        n_total_times = int(group[TIME_COL].nunique())
        if n_train_times != N_TRAIN_TIMEPOINTS:
            raise AssertionError(
                f"Cohort {cohort!r} has {n_train_times} train timepoints; "
                f"expected {N_TRAIN_TIMEPOINTS}."
            )
        if n_test_times != N_TEST_TIMEPOINTS:
            raise AssertionError(
                f"Cohort {cohort!r} has {n_test_times} test timepoints; "
                f"expected {N_TEST_TIMEPOINTS}."
            )
        if n_total_times != N_REQUIRED_TIMEPOINTS:
            raise AssertionError(
                f"Cohort {cohort!r} has {n_total_times} selected timepoints; "
                f"expected {N_REQUIRED_TIMEPOINTS}."
            )

    return frame


class FeatureAdapter:
    """BASE2 feature adapter: scale five continuous columns, keep cohort one-hot unchanged."""

    def __init__(self, cohort_levels: list[str], age_domain: tuple[float, float]):
        self.cohort_levels = list(cohort_levels)
        self.age_domain = tuple(map(float, age_domain))
        self.continuous_scaler = StandardScaler()

    def fit(self, frame: pd.DataFrame) -> "FeatureAdapter":
        self.continuous_scaler.fit(frame[CONT_COLS].to_numpy(dtype=float))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        continuous = self.continuous_scaler.transform(
            frame[CONT_COLS].to_numpy(dtype=float)
        )
        cohort = pd.Categorical(
            frame[COHORT_COL].astype(str), categories=self.cohort_levels
        ).codes
        if np.any(cohort < 0):
            unknown = frame.loc[cohort < 0, COHORT_COL].astype(str).unique().tolist()
            raise ValueError(f"Unknown cohort levels: {unknown}")
        one_hot = np.eye(len(self.cohort_levels), dtype=float)[cohort]
        return np.concatenate([continuous, one_hot], axis=1)


def lognormal_log_prior(
    value: torch.Tensor,
    loc: float = LOG_PARAMETER_PRIOR_LOC,
    scale: float = LOG_PARAMETER_PRIOR_SCALE,
) -> torch.Tensor:
    value = torch.clamp(value, min=1e-12)
    log_value = torch.log(value)
    return (
        -0.5 * ((log_value - loc) / scale) ** 2
        - log_value
        - math.log(scale)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum()


def inverse_gamma_log_prior(
    value: torch.Tensor,
    shape: float,
    scale: float,
) -> torch.Tensor:
    value = torch.clamp(value, min=1e-12)
    shape_tensor = torch.as_tensor(
        shape, dtype=value.dtype, device=value.device
    )
    scale_tensor = torch.as_tensor(
        scale, dtype=value.dtype, device=value.device
    )
    return (
        shape_tensor * torch.log(scale_tensor)
        - torch.lgamma(shape_tensor)
        - (shape_tensor + 1.0) * torch.log(value)
        - scale_tensor / value
    ).sum()


# =============================================================================
# The two non-standard kernels: group-additive SE and the neural network kernel
#
# Input layout (FeatureAdapter.transform):
#     dim 0      standardised day-age
#     dims 1-4   standardised pH / salinity / DO / water temperature
#     dims 5-8   facility one-hot
# so ||x||^2 >= 1 always.
# =============================================================================
AGE_DIMS = [0]


ENV_DIMS = [1, 2, 3, 4]


COHORT_DIMS = [5, 6, 7, 8]


NN_CLAMP = 1e-8


class NeuralNetworkKernel(gpytorch.kernels.Kernel):
    """Williams (1998) infinite-width neural-network covariance.

        k(x,x') = (2/pi) asin( 2 xt' S xt' / sqrt((1+2 xt' S xt)(1+2 xt'' S xt'')) )

    with the augmented input xt = [1, x] and S = diag(s_bias, s_weight * I_9), so
    the kernel carries exactly TWO hyperparameters -- a full diagonal S would be
    ten parameters on 24 rows.

    k(x,x) = (2/pi) asin(2q/(1+2q)) with q = s_bias + s_weight ||x||^2, which is
    increasing in ||x|| but bounded above by 1.  Measured on this data with
    S = I: train mean 0.708, test mean 0.843, ratio 1.191 (Polynomial2: 7.018).
    Bounded-but-not-constant is the point -- the variance may differ across the
    input space, which is what makes the direction different from a stationary
    kernel, but it cannot diverge outside the training hull.

    Sample paths are sums of saturating ridge functions.  Because this project
    puts the GP on the latent growth RATE and integrates, a rate that rises and
    then plateaus gives a decelerating growth curve, which is the shape abalone
    weight actually has.
    """

    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter("raw_bias_variance",
                                torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter("raw_weight_variance",
                                torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_bias_variance", gpytorch.constraints.Positive())
        self.register_constraint("raw_weight_variance", gpytorch.constraints.Positive())
        self.initialize(bias_variance=1.0, weight_variance=1.0)

    @property
    def bias_variance(self):
        return self.raw_bias_variance_constraint.transform(self.raw_bias_variance)

    @bias_variance.setter
    def bias_variance(self, value):
        self._set(self.raw_bias_variance_constraint, "raw_bias_variance", value)

    @property
    def weight_variance(self):
        return self.raw_weight_variance_constraint.transform(self.raw_weight_variance)

    @weight_variance.setter
    def weight_variance(self, value):
        self._set(self.raw_weight_variance_constraint, "raw_weight_variance", value)

    def _set(self, constraint, name, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(getattr(self, name))
        self.initialize(**{name: constraint.inverse_transform(value)})

    def forward(self, x1, x2, diag=False, **params):
        bias = self.bias_variance.reshape(())
        weight = self.weight_variance.reshape(())
        q1 = bias + weight * (x1 ** 2).sum(-1)
        q2 = bias + weight * (x2 ** 2).sum(-1)
        d1 = (1.0 + 2.0 * q1).sqrt()
        d2 = (1.0 + 2.0 * q2).sqrt()
        if diag:
            numerator = 2.0 * (bias + weight * (x1 * x2).sum(-1))
            u = numerator / (d1 * d2)
        else:
            numerator = 2.0 * (bias + weight * (x1 @ x2.transpose(-2, -1)))
            u = numerator / (d1.unsqueeze(-1) * d2.unsqueeze(-2))
        u = u.clamp(-1.0 + NN_CLAMP, 1.0 - NN_CLAMP)
        return (2.0 / math.pi) * torch.asin(u)


def _additive(factory):
    """Sum of one SCALED base kernel per input group.

    The outer ScaleKernel that build_kernel/build_kernel_module adds would make a
    third free scale redundant, so the age component's outputscale is pinned at
    1.0 and excluded from optimisation -- the same device 0902_abalone2 uses to
    keep LinearKernel's variance identified.  Two free ratios plus the outer
    scale is exactly three degrees of freedom, as it should be.
    """
    age = gpytorch.kernels.ScaleKernel(factory(active_dims=AGE_DIMS))
    age.initialize(outputscale=1.0)
    age.raw_outputscale.requires_grad_(False)
    return gpytorch.kernels.AdditiveKernel(
        age,
        gpytorch.kernels.ScaleKernel(factory(active_dims=ENV_DIMS)),
        gpytorch.kernels.ScaleKernel(factory(active_dims=COHORT_DIMS)))


def _rbf(**kwargs):
    return gpytorch.kernels.RBFKernel(**kwargs)


NEW_KERNELS = {
    "RBF_add3": lambda: _additive(_rbf),      # paper eq. (11), group-additive SE
    "NNKernel": lambda: NeuralNetworkKernel(),  # paper eq. (12)
}

# =============================================================================
# Kernel construction and hyperparameter priors
# =============================================================================

def raw_kernel(kernel_name: str):
    # the two kernels of paper eq. (11) and (12), defined above
    if kernel_name in NEW_KERNELS:
        return NEW_KERNELS[kernel_name]()
    if kernel_name == "RBF":
        return gpytorch.kernels.RBFKernel()
    if kernel_name == "Matern0.5":
        return gpytorch.kernels.MaternKernel(nu=0.5)
    if kernel_name == "Matern1.5":
        return gpytorch.kernels.MaternKernel(nu=1.5)
    if kernel_name == "Matern2.5":
        return gpytorch.kernels.MaternKernel(nu=2.5)
    if kernel_name == "RQ":
        return gpytorch.kernels.RQKernel()
    if kernel_name == "Linear":
        # variance is not identified once ScaleKernel supplies an outputscale,
        # so it is pinned at 1.0 and excluded from optimisation.
        kernel = gpytorch.kernels.LinearKernel()
        kernel.initialize(variance=1.0)
        kernel.raw_variance.requires_grad_(False)
        return kernel
    if kernel_name == "Polynomial2":
        return gpytorch.kernels.PolynomialKernel(power=2)
    raise ValueError(f"Unknown kernel: {kernel_name}")


def create_kernel(kernel_name: str):
    return gpytorch.kernels.ScaleKernel(raw_kernel(kernel_name))


# =============================================================================
# Monotone GP: the positive-growth integral construction (paper eq. 3-5)
# =============================================================================

GN_MAX_ITER = 40


GN_TOL = 1e-9


GN_DAMPING0 = 1e-6


class CohortGrid:
    """Per-cohort age grid whose nodes include every observed training age.

    Observed ages are kept as nodes so the integral is exact at the points the
    likelihood uses; each interval is then split into `M` subintervals.  The grid
    is always built from the FULL training age set, so a leave-one-out fit still
    has the held-out age as a node.  `extra_ages` forces additional ages (the test
    ages) to be nodes as well.
    """

    def __init__(self, source: pd.DataFrame, cohorts: list[str], subdivisions: int,
                 extra_ages: dict[str, np.ndarray] | None = None,
                 age_scale: float = 1.0):
        """extra_ages: per-cohort ages that must ALSO be grid nodes (e.g. the test
        ages).  They are merged with the training ages and every consecutive pair
        is subdivided M times, so every required prediction age is exactly a node.
        The previous uniform-step extension did not land on the test ages and
        raised "age ... is not a grid node"."""
        self.cohorts = list(cohorts)
        self.M = int(subdivisions)
        self.extra_ages = extra_ages
        self.age_scale = float(age_scale)
        self.nodes: dict[str, np.ndarray] = {}
        for cohort in cohorts:
            ages = source.loc[source[COHORT_COL].astype(str).eq(cohort), AGE_COL].to_numpy(float)
            if extra_ages is not None and cohort in extra_ages:
                ages = np.concatenate([ages, np.asarray(extra_ages[cohort], float)])
            ages = np.unique(ages)
            pieces = [np.linspace(ages[i], ages[i + 1], self.M + 1)[:-1]
                      for i in range(len(ages) - 1)]
            self.nodes[cohort] = np.concatenate(pieces + [ages[-1:]])
        self.offset, position = {}, 0
        for cohort in cohorts:
            self.offset[cohort] = position
            position += len(self.nodes[cohort])
        self.size = position

    def index_of(self, cohort: str, age: float) -> int:
        grid = self.nodes[str(cohort)]
        i = int(np.argmin(np.abs(grid - float(age))))
        if abs(grid[i] - float(age)) > 1e-6:
            raise AssertionError(f"age {age} for cohort {cohort} is not a grid node")
        return self.offset[str(cohort)] + i

    def trapezoid_matrix(self) -> np.ndarray:
        """A with f = A r, block diagonal, f(node_j) = sum_{i<j} (r_i+r_{i+1})/2 * d_i."""
        A = np.zeros((self.size, self.size))
        for cohort in self.cohorts:
            grid = self.nodes[cohort]
            base = self.offset[cohort]
            # age_scale changes the units the integral is taken in, hence the
            # units of the growth rate.  Nodes stay in days for indexing and for
            # the figures.  A positive rescaling keeps every d_i > 0, so the
            # monotonicity proof (design 4.3) is untouched.
            d = np.diff(grid) * self.age_scale
            for j in range(1, len(grid)):
                A[base + j] = A[base + j - 1]
                A[base + j, base + j - 1] += 0.5 * d[j - 1]
                A[base + j, base + j] += 0.5 * d[j - 1]
        return A

    def frame(self, train: pd.DataFrame) -> pd.DataFrame:
        """Feature frame at every node; environment is interpolated in age
        within cohort from training rows only."""
        blocks = []
        for cohort in self.cohorts:
            grid = self.nodes[cohort]
            rows = train.loc[train[COHORT_COL].astype(str).eq(cohort)].sort_values(AGE_COL)
            block = pd.DataFrame({AGE_COL: grid})
            block[COHORT_COL] = cohort
            block[TIME_COL] = pd.NaT
            for column in ENV_COLS:
                block[column] = np.interp(grid, rows[AGE_COL].to_numpy(float),
                                          rows[column].to_numpy(float))
            blocks.append(block)
        return pd.concat(blocks, ignore_index=True)


def monotonicity_report(grid: CohortGrid, rates: np.ndarray, subdivisions: int = 100) -> dict:
    """Numerically confirm design 4.3, at nodes and between them."""
    A = grid.trapezoid_matrix()
    f = A @ rates
    worst_node, worst_inner = 0.0, 0.0
    for cohort in grid.cohorts:
        base, g = grid.offset[cohort], grid.nodes[cohort]
        block = f[base:base + len(g)]
        worst_node = min(worst_node, float(np.min(np.diff(block))))
        for i in range(len(g) - 1):
            r0, r1 = rates[base + i], rates[base + i + 1]
            u = np.linspace(0.0, 1.0, subdivisions + 1)
            delta = g[i + 1] - g[i]
            inner = block[i] + delta * (r0 * u + 0.5 * (r1 - r0) * u ** 2)
            worst_inner = min(worst_inner, float(np.min(np.diff(inner))))
    return {"min_rate": float(np.min(rates)),
            "min_node_increment": worst_node,
            "min_interpolant_increment": worst_inner,
            "monotone_at_nodes": bool(worst_node >= -1e-12),
            "monotone_between_nodes": bool(worst_inner >= -1e-12)}


def build_kernel(kernel_name: str, device):
    if "+" in kernel_name:
        parts = kernel_name.split("+")
        kernel = gpytorch.kernels.AdditiveKernel(
            *(gpytorch.kernels.ScaleKernel(raw_kernel(q)) for q in parts))
    elif "*" in kernel_name:
        parts = kernel_name.split("*")
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ProductKernel(*(raw_kernel(q) for q in parts)))
    else:
        kernel = create_kernel(kernel_name)
    return kernel.to(device).double()


def kernel_log_prior_integral(kernel) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64,
                        device=next(kernel.parameters()).device)
    for module in kernel.modules():
        if isinstance(module, (gpytorch.kernels.RBFKernel, gpytorch.kernels.MaternKernel)):
            total = total + lognormal_log_prior(module.lengthscale)
        elif isinstance(module, gpytorch.kernels.RQKernel):
            total = total + lognormal_log_prior(module.lengthscale)
            total = total + lognormal_log_prior(module.alpha)
        elif isinstance(module, gpytorch.kernels.PolynomialKernel):
            total = total + lognormal_log_prior(module.offset)
        elif isinstance(module, gpytorch.kernels.ScaleKernel):
            total = total + lognormal_log_prior(module.outputscale)
        # LinearKernel: variance frozen at 1.0, no prior branch (design 11)
    # NeuralNetworkKernel is not built from RBF/Matern/Polynomial/Scale,
    # so its two variances need a branch of their own.
    for sub in kernel.modules():
        if isinstance(sub, NeuralNetworkKernel):
            total = total + lognormal_log_prior(sub.bias_variance)
            total = total + lognormal_log_prior(sub.weight_variance)
    return total


def gauss_newton_mode(u, Lchol, mu_g, A_obs, y, Sigma_inv, max_iter=GN_MAX_ITER):
    """Damped Gauss-Newton for the Laplace mode in the whitened coordinate u.

    g = mu_g + L u, r = softplus(g), m = A_obs r.  The GN Hessian
    I + (J L)^T Sigma^{-1} (J L) is positive definite for every u, which full
    Newton would not guarantee.
    """
    damping = GN_DAMPING0
    identity = torch.eye(u.shape[0], dtype=u.dtype, device=u.device)
    previous = None
    for _ in range(max_iter):
        g = mu_g + Lchol @ u
        rate = torch.nn.functional.softplus(g)
        slope = torch.sigmoid(g)
        residual = y - A_obs @ rate
        objective = 0.5 * residual @ (Sigma_inv @ residual) + 0.5 * u @ u
        JL = (A_obs * slope.unsqueeze(0)) @ Lchol
        gradient = -JL.T @ (Sigma_inv @ residual) + u
        H = identity + JL.T @ Sigma_inv @ JL
        step = torch.linalg.solve(H + damping * identity, gradient)
        candidate = u - step
        g2 = mu_g + Lchol @ candidate
        residual2 = y - A_obs @ torch.nn.functional.softplus(g2)
        objective2 = 0.5 * residual2 @ (Sigma_inv @ residual2) + 0.5 * candidate @ candidate
        if float(objective2) <= float(objective):
            u, damping = candidate, max(damping * 0.5, 1e-10)
        else:
            damping = min(damping * 10.0, 1e4)
        if previous is not None and abs(float(objective) - previous) < GN_TOL:
            break
        previous = float(objective)
    g = mu_g + Lchol @ u
    slope = torch.sigmoid(g)
    JL = (A_obs * slope.unsqueeze(0)) @ Lchol
    H = torch.eye(u.shape[0], dtype=u.dtype, device=u.device) + JL.T @ Sigma_inv @ JL
    return u.detach(), H.detach()


def laplace_log_marginal(u, Lchol, mu_g, A_obs, y, Sigma_inv, Sigma_logdet, H):
    """log p(y) under the Laplace approximation, evaluated at the (detached) mode.

    The implicit dependence of u_hat on the hyperparameters is not propagated;
    at the mode the gradient of the first two terms through u vanishes, so only
    the log|H| term loses a second-order contribution.  Documented approximation.
    """
    n = y.shape[0]
    residual = y - A_obs @ torch.nn.functional.softplus(mu_g + Lchol @ u)
    fit = -0.5 * residual @ (Sigma_inv @ residual) - 0.5 * Sigma_logdet \
        - 0.5 * n * math.log(2.0 * math.pi)
    return fit - 0.5 * u @ u - 0.5 * torch.logdet(H)


def fit_positive_growth_gp(train_fit: pd.DataFrame, grid: CohortGrid,
                           grid_frame: pd.DataFrame, adapter, kernel_name: str,
                           s0: float, seed: int, device, hyper_iter=HYPER_ITER_INTEGRAL) -> dict:
    """Empirical-Bayes fit: alternate Gauss-Newton for the mode with Adam on the
    hyperparameters (kernel, mu_g, sigma_n)."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    x = torch.tensor(adapter.transform(grid_frame), dtype=torch.float64, device=device)
    A_full = torch.tensor(grid.trapezoid_matrix(), dtype=torch.float64, device=device)
    rows = [grid.index_of(str(r[COHORT_COL]), float(r[AGE_COL]))
            for _, r in train_fit.iterrows()]
    A_obs = A_full[rows]
    y = torch.tensor(train_fit["_z"].to_numpy(float), dtype=torch.float64, device=device)
    Z = torch.tensor(
        pd.get_dummies(train_fit[COHORT_COL].astype(str)).reindex(
            columns=grid.cohorts, fill_value=0).to_numpy(float),
        dtype=torch.float64, device=device)
    n = y.shape[0]

    kernel = build_kernel(kernel_name, device)
    # Initialise the prior mean of g at the empirical rate scale.  softplus(0)=0.693
    # while the data need about 0.017 z-units/day, so a zero start puts the prior
    # ~40x too high and the optimiser has to travel a long way.
    span_z = float(train_fit["_z"].max() - train_fit["_z"].min())
    span_age = float(train_fit[AGE_COL].max() - train_fit[AGE_COL].min()) * grid.age_scale
    empirical_rate = max(span_z / max(span_age, 1.0), 1e-4)
    raw_mu = torch.tensor(math.log(math.expm1(empirical_rate)), dtype=torch.float64,
                          device=device, requires_grad=True)
    raw_noise = torch.tensor(math.log(math.expm1(0.05)), dtype=torch.float64,
                             device=device, requires_grad=True)
    parameters = [p for p in kernel.parameters() if p.requires_grad] + [raw_mu, raw_noise]
    optimizer = torch.optim.Adam(parameters, lr=HYPER_LR)

    u = torch.zeros(grid.size, dtype=torch.float64, device=device)
    best = {"value": -math.inf}
    identity_n = torch.eye(n, dtype=torch.float64, device=device)

    for step in range(hyper_iter):
        optimizer.zero_grad(set_to_none=True)
        K = kernel(x).to_dense() + JITTER * torch.eye(grid.size, dtype=torch.float64,
                                                      device=device)
        Lchol = torch.linalg.cholesky(K)
        sigma_n = torch.nn.functional.softplus(raw_noise) + NOISE_FLOOR
        Sigma = sigma_n ** 2 * identity_n + (s0 ** 2) * (Z @ Z.T)
        Sigma_inv = torch.linalg.inv(Sigma)
        Sigma_logdet = torch.logdet(Sigma)
        with torch.no_grad():
            u, H = gauss_newton_mode(u, Lchol.detach(), raw_mu.detach(),
                                     A_obs, y, Sigma_inv.detach())
        JL = (A_obs * torch.sigmoid(raw_mu + Lchol @ u).unsqueeze(0)) @ Lchol
        H_diff = torch.eye(grid.size, dtype=torch.float64, device=device) \
            + JL.T @ Sigma_inv @ JL
        objective = laplace_log_marginal(u, Lchol, raw_mu, A_obs, y, Sigma_inv,
                                         Sigma_logdet, H_diff) \
            + kernel_log_prior_integral(kernel) \
            + inverse_gamma_log_prior(sigma_n ** 2, GLOBAL_NOISE_PRIOR_SHAPE,
                                         (GLOBAL_NOISE_PRIOR_SHAPE - 1.0)
                                         * GLOBAL_NOISE_PRIOR_MEAN)
        loss = -objective
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite objective for {kernel_name}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 20.0)
        value = float(objective)
        if value > best["value"]:
            best = {"value": value, "step": step,
                    "state": {k: v.detach().clone() for k, v in kernel.state_dict().items()},
                    "mu": raw_mu.detach().clone(), "noise": raw_noise.detach().clone(),
                    "u": u.detach().clone()}
        optimizer.step()

    kernel.load_state_dict(best["state"])
    raw_mu, raw_noise = best["mu"], best["noise"]
    with torch.no_grad():
        K = kernel(x).to_dense() + JITTER * torch.eye(grid.size, dtype=torch.float64,
                                                      device=device)
        Lchol = torch.linalg.cholesky(K)
        sigma_n = torch.nn.functional.softplus(raw_noise) + NOISE_FLOOR
        Sigma = sigma_n ** 2 * identity_n + (s0 ** 2) * (Z @ Z.T)
        Sigma_inv = torch.linalg.inv(Sigma)
        u_hat, H = gauss_newton_mode(best["u"], Lchol, raw_mu, A_obs, y, Sigma_inv)
        rate = torch.nn.functional.softplus(raw_mu + Lchol @ u_hat)

    # A composite has several scales and lengthscales, so all of them are collected;
    # the scalar columns keep the first entry so single-kernel output is unchanged.
    scales, lengths, alphas, offsets, variances = [], [], [], [], []
    for module in kernel.modules():
        if isinstance(module, gpytorch.kernels.ScaleKernel):
            scales.append(float(module.outputscale))
        if isinstance(module, (gpytorch.kernels.RBFKernel, gpytorch.kernels.MaternKernel,
                               gpytorch.kernels.RQKernel)):
            lengths.append(float(module.lengthscale.reshape(-1)[0]))
        if isinstance(module, gpytorch.kernels.RQKernel):
            alphas.append(float(module.alpha.reshape(-1)[0]))
        if isinstance(module, gpytorch.kernels.PolynomialKernel):
            offsets.append(float(module.offset.reshape(-1)[0]))
        if isinstance(module, gpytorch.kernels.LinearKernel):
            variances.append(float(module.variance.reshape(-1)[0]))
    first = lambda v: (v[0] if v else math.nan)
    hyper = {"outputscale": first(scales), "sigma_n": float(sigma_n),
             "mu_g": float(raw_mu), "lengthscale": first(lengths),
             "rq_alpha": first(alphas), "poly_offset": first(offsets),
             "linear_variance": first(variances),
             "all_outputscales": scales, "all_lengthscales": lengths,
             "n_kernel_params": int(sum(1 for q in kernel.parameters()
                                        if q.requires_grad))}
    return {"kernel": kernel, "Lchol": Lchol.detach(), "mu_g": raw_mu.detach(),
            "u_hat": u_hat, "H": H, "sigma_n": float(sigma_n), "s0": float(s0),
            "rate": rate.detach().cpu().numpy(), "A_full": A_full.detach(),
            "A_obs": A_obs.detach().cpu().numpy(),
            "y_obs": y.detach().cpu().numpy(),
            "cohort_of_obs": [str(c) for c in train_fit[COHORT_COL].astype(str)],
            "grid": grid, "log_marginal": best["value"], "best_step": best["step"],
            "converged": bool(best["step"] < hyper_iter - 1),
            "hyper": hyper, "device": device}


def f0_posterior_draws(fitted: dict, rate_draws: np.ndarray, cohorts: list[str],
                       rng) -> tuple[np.ndarray, float]:
    """Posterior of the per-cohort initial level, conditional on each g draw.

    f0 was marginalised OUT of the marginal likelihood, which is correct for
    fitting.  For PREDICTION it must be conditioned on the observed rows of that
    cohort: omitting this made the predicted mean at every cohort's first node
    exactly 0 (the prior mean) and inflated the predictive sd to ~s0, which is
    what produced RMSE_z of 3-8 and Jensen factors of 25.

        d = y - A_obs r,   d_c = f0_c + eps
        f0_c | d, g  ~  N( s0^2 * sum(d_c) / (s0^2 n_c + sn^2),
                            s0^2 sn^2 / (s0^2 n_c + sn^2) )
    """
    A_obs = np.asarray(fitted["A_obs"], float)
    y = np.asarray(fitted["y_obs"], float)
    s0, sn = fitted["s0"], fitted["sigma_n"]
    which = np.asarray(fitted["cohort_of_obs"])
    residual = y[None, :] - rate_draws @ A_obs.T                  # S x n
    S = rate_draws.shape[0]
    f0 = np.zeros((S, len(cohorts)))
    for index, cohort in enumerate(cohorts):
        mask = which == cohort
        n_c = int(mask.sum())
        if n_c == 0:
            f0[:, index] = rng.normal(0.0, s0, size=S)
            continue
        denominator = s0 ** 2 * n_c + sn ** 2
        mean = s0 ** 2 * residual[:, mask].sum(axis=1) / denominator
        variance = s0 ** 2 * sn ** 2 / denominator
        f0[:, index] = mean + rng.normal(0.0, math.sqrt(variance), size=S)
    return f0, sn ** 2


def posterior_draws(fitted: dict, seed: int, draws: int = MC_DRAWS) -> np.ndarray:
    """Latent f at every grid node, S x N.  Rao-Blackwellisation of the
    observation noise and of f0 happens later (design 9)."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    H = fitted["H"].cpu()
    Lh = torch.linalg.cholesky(torch.linalg.inv(H))
    eps = torch.randn(draws, H.shape[0], generator=generator, dtype=torch.float64)
    u = fitted["u_hat"].cpu().unsqueeze(0) + eps @ Lh.T
    g = fitted["mu_g"].cpu() + u @ fitted["Lchol"].cpu().T
    rate = torch.nn.functional.softplus(g)
    f = rate @ fitted["A_full"].cpu().T
    return f.numpy(), rate.numpy()


# =============================================================================
# Response transform, plain GP, stacking, conformal calibration
# =============================================================================

RESPONSES = ["ln_weight", "ln_weight_yj", "raw_weight", "raw_weight_yj"]


BOOTSTRAP_DRAWS = 400


SCAN_TRAIN_POINTS = 80


SCAN_TEST_POINTS = 50


class Response:
    """Maps 전중(g) to the modelling scale z and back.

    base(w) = ln(w) or w, optionally Yeo-Johnson'd, then standardised.  Every
    constant -- the YJ lambda, the mean and the scale -- is estimated from the
    rows handed to fit(), which for a LOO fit are its n-1 rows only.

    Every step is strictly increasing in w, so the inverse of a z quantile IS the
    corresponding gram quantile.  The reported point prediction is
    from_z(point_z(.)) -- the predictive median under POINT_SUMMARY = "median" --
    and it is the centre of the conformal interval.
    """

    def __init__(self, mode: str):
        if mode not in RESPONSES:
            raise ValueError(f"unknown response {mode}")
        self.mode = mode
        self.log_base = mode.startswith("ln_")
        self.yeo_johnson = mode.endswith("_yj")
        self.transformer = None
        self.mu = math.nan
        self.sigma = math.nan
        self.lam = math.nan
        self.fit_failed = False
        self.fit_failure_reason = ""
        self.n_outside_range = 0

    def yj_z_bounds(self) -> tuple[float, float]:
        r"""z values outside which the Yeo-Johnson inverse does not exist.

        On the y >= 0 branch YJ(y) = ((y+1)^lam - 1)/lam, so for lam < 0 it is
        bounded ABOVE by -1/lam: no weight maps to a larger transformed value and
        the inverse is undefined there.  Symmetrically lam > 2 is bounded below on
        the y < 0 branch.  For 0 <= lam <= 2 the map is onto the whole line.

        Measured on this data: ln_weight_yj fits lam = +1.60, unbounded;
        raw_weight_yj fits lam = -0.25, so the transformed scale stops at 3.93,
        which is z = +5.57.  The 0.9995 quantile sits at z = 3.29, so a fit with a
        predictive sd above about 1.7 pushes its upper quantiles out of the domain.
        """
        if not self.yeo_johnson or not np.isfinite(self.lam):
            return (-math.inf, math.inf)
        low, high = -math.inf, math.inf
        if self.lam < 0.0:
            high = (-1.0 / self.lam - self.mu) / self.sigma
        if self.lam > 2.0:
            low = (1.0 / (2.0 - self.lam) - self.mu) / self.sigma
        return (low, high)

    def _base(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        return np.log(weights) if self.log_base else weights

    def fit(self, weights: np.ndarray) -> "Response":
        base = self._base(weights).reshape(-1, 1)
        if self.yeo_johnson:
            try:
                self.transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                self.transformer.fit(base)
                self.lam = float(self.transformer.lambdas_[0])
                self.mu = float(self.transformer._scaler.mean_[0])
                self.sigma = float(self.transformer._scaler.scale_[0])
                if not (np.isfinite(self.lam) and np.isfinite(self.sigma) and self.sigma > 0):
                    raise FloatingPointError(f"degenerate Yeo-Johnson fit lambda={self.lam}")
            except Exception as exc:                       # recorded, never silent
                self.fit_failed = True
                self.fit_failure_reason = f"Yeo-Johnson fit failed: {exc}"
        else:
            self.transformer = StandardScaler().fit(base)
            self.mu = float(self.transformer.mean_[0])
            self.sigma = float(self.transformer.scale_[0])
            if not (np.isfinite(self.sigma) and self.sigma > 0):
                self.fit_failed = True
                self.fit_failure_reason = f"degenerate scale {self.sigma}"
        return self

    def to_z(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        base = self._base(weights).reshape(-1)
        finite = np.isfinite(base)
        base = base.copy()
        base[~finite] = 0.0
        z = np.asarray(self.transformer.transform(base.reshape(-1, 1)), dtype=float).ravel()
        z[~finite] = np.nan
        return z

    def from_z(self, z: np.ndarray) -> np.ndarray:
        """z -> 전중(g).  May be NON-POSITIVE for the raw_weight variants: the
        standardised raw scale puts 0 g at z = -0.945, so any predictive spread
        wider than that pushes an interval endpoint below zero.  Nothing is
        clipped here; the callers count and report the occurrences."""
        z = np.asarray(z, dtype=float)
        shape = z.shape
        flat = z.reshape(-1).copy()
        finite = np.isfinite(flat)
        # a numerical guard, not a modelling choice: 40 standardised units is far
        # beyond any meaningful predictive quantile, and sklearn refuses non-finite
        # input outright.  Non-finite entries come back as NaN so the caller sees them.
        low, high = self.yj_z_bounds()
        # Values beyond the Yeo-Johnson image have no inverse at all; they are
        # counted and returned as NaN rather than left to sklearn's warning.
        outside = finite & ((flat < low) | (flat > high))
        self.n_outside_range += int(outside.sum())
        finite = finite & ~outside
        flat[finite] = np.clip(flat[finite], -40.0, 40.0)
        flat[~finite] = 0.0
        base = np.asarray(
            self.transformer.inverse_transform(flat.reshape(-1, 1)), dtype=float).reshape(-1)
        if self.log_base:
            base = np.exp(np.clip(base, -700.0, 700.0))
        base[~finite] = np.nan
        return base.reshape(shape)

    def log_dz_dw(self, weights: np.ndarray) -> np.ndarray:
        """log|dz/dw| — the term that puts every response variable's LPD on the
        common GRAM scale so the B axis can be compared at all.

        Note this term is a per-row constant shared by every candidate kernel, so
        it does NOT change the stacking argmax; see the module docstring.
        """
        weights = np.asarray(weights, dtype=float)
        base = self._base(weights)
        total = np.full(base.shape, -math.log(self.sigma))
        if self.yeo_johnson:
            lam = self.lam
            yj_prime = np.where(base >= 0.0,
                                (lam - 1.0) * np.log1p(np.abs(base)),
                                (1.0 - lam) * np.log1p(np.abs(base)))
            total = total + yj_prime
        if self.log_base:
            total = total - np.log(weights)
        return total

    @classmethod
    def from_params(cls, mode: str, mu: float, sigma: float, lam: float) -> "Response":
        """Rebuild a fitted transform from the numbers stored in a checkpoint,
        without refitting.  Used when assembling results from a resumed run."""
        response = cls(mode)
        scaler = StandardScaler()
        scaler.mean_ = np.array([float(mu)])
        scaler.scale_ = np.array([float(sigma)])
        scaler.var_ = np.array([float(sigma) ** 2])
        scaler.n_features_in_ = 1
        scaler.n_samples_seen_ = 1
        if response.yeo_johnson:
            transformer = PowerTransformer(method="yeo-johnson", standardize=True)
            transformer.lambdas_ = np.array([float(lam)])
            transformer._scaler = scaler
            transformer.n_features_in_ = 1
            response.transformer = transformer
        else:
            response.transformer = scaler
        response.mu, response.sigma, response.lam = float(mu), float(sigma), float(lam)
        return response

    def describe(self) -> dict:
        low, high = self.yj_z_bounds()
        return {"response": self.mode, "response_mu": self.mu, "response_sigma": self.sigma,
                "response_yj_lambda": self.lam,
                "response_z_lower_bound": low, "response_z_upper_bound": high,
                "response_bounded_inverse": bool(np.isfinite(low) or np.isfinite(high)),
                "response_fit_failed": self.fit_failed,
                "response_fit_failure_reason": self.fit_failure_reason}


INPUT_COLS = [AGE_COL, *ENV_COLS, COHORT_COL]


FORBIDDEN = ["_response", "_z", "ln_전중", "로그무게", RAW_WEIGHT_COL]


def input_source(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Rows whose INPUT columns drive every interpolation and grid.

    The projection to INPUT_COLS is what makes leakage impossible; the assertion
    states the invariant rather than trusting the projection.
    """
    frame = pd.concat([train, test], ignore_index=True, sort=False).loc[:, INPUT_COLS].copy()
    leaked = [c for c in FORBIDDEN if c in frame.columns]
    assert not leaked, f"response column(s) reached an input-only frame: {leaked}"
    assert frame[[AGE_COL, *ENV_COLS]].notna().all().all(), "input source has missing values"
    return frame.sort_values([COHORT_COL, AGE_COL], kind="mergesort").reset_index(drop=True)


def age_frame(source: pd.DataFrame, cohorts: list[str], ages_of: dict[str, np.ndarray],
              extra: dict | None = None) -> pd.DataFrame:
    """Feature frame at the requested ages, environment interpolated from inputs."""
    blocks = []
    for cohort in cohorts:
        rows = source.loc[source[COHORT_COL].astype(str).eq(cohort)].sort_values(AGE_COL)
        ages = np.asarray(ages_of[cohort], dtype=float)
        block = pd.DataFrame({AGE_COL: ages})
        block[COHORT_COL] = cohort
        block[TIME_COL] = pd.NaT
        for column in ENV_COLS:
            block[column] = np.interp(ages, rows[AGE_COL].to_numpy(float),
                                      rows[column].to_numpy(float))
        if extra:
            for key, value in extra.items():
                block[key] = value[cohort] if isinstance(value, dict) else value
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def horizon_ages(train, test, cohorts) -> dict[str, tuple[float, float, float]]:
    spans = {}
    for cohort in cohorts:
        tr = train.loc[train[COHORT_COL].astype(str).eq(cohort), AGE_COL].to_numpy(float)
        te = test.loc[test[COHORT_COL].astype(str).eq(cohort), AGE_COL].to_numpy(float)
        spans[cohort] = (float(tr.min()), float(tr.max()), float(max(te.max(), tr.max())))
    return spans


def scan_frame(source, train, test, cohorts) -> pd.DataFrame:
    """One dense grid used both to refine constraints and to audit monotonicity,
    so a refinement can never declare success on a coarser grid than the audit."""
    spans = horizon_ages(train, test, cohorts)
    ages, windows = {}, {}
    for cohort, (low, mid, high) in spans.items():
        values = np.concatenate([np.linspace(low, mid, SCAN_TRAIN_POINTS),
                                 np.linspace(mid, high, SCAN_TEST_POINTS)[1:]])
        ages[cohort] = values
        windows[cohort] = np.where(values <= mid, "train", "test")
    frame = age_frame(source, cohorts, ages)
    frame["window"] = np.concatenate([windows[c] for c in cohorts])
    return frame


def build_kernel_module(spec: str):
    """Sums keep one ScaleKernel per summand; a product gets exactly ONE
    ScaleKernel around the whole ProductKernel so the outputscale stays
    identified.  Linear's variance is frozen at 1.0 inside raw_kernel."""
    if "+" in spec:
        return gpytorch.kernels.AdditiveKernel(
            *(gpytorch.kernels.ScaleKernel(raw_kernel(p)) for p in spec.split("+"))).double()
    if "*" in spec:
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ProductKernel(*(raw_kernel(p) for p in spec.split("*")))).double()
    return gpytorch.kernels.ScaleKernel(raw_kernel(spec)).double()


def kernel_log_prior(module) -> torch.Tensor:
    """LogNormal(0,1) on every positive hyperparameter, as everywhere in this
    project.  RQ gets a branch for alpha as well as lengthscale; LinearKernel gets
    none on purpose because its only parameter is frozen."""
    total = torch.zeros((), dtype=torch.float64)
    for sub in module.modules():
        if isinstance(sub, gpytorch.kernels.RQKernel):
            total = total + lognormal_log_prior(sub.lengthscale)
            total = total + lognormal_log_prior(sub.alpha)
        elif isinstance(sub, (gpytorch.kernels.RBFKernel, gpytorch.kernels.MaternKernel)):
            total = total + lognormal_log_prior(sub.lengthscale)
        elif isinstance(sub, gpytorch.kernels.PolynomialKernel):
            total = total + lognormal_log_prior(sub.offset)
        elif isinstance(sub, gpytorch.kernels.ScaleKernel):
            total = total + lognormal_log_prior(sub.outputscale)
    # NeuralNetworkKernel is not built from RBF/Matern/Polynomial/Scale,
    # so its two variances need a branch of their own.
    for sub in module.modules():
        if isinstance(sub, NeuralNetworkKernel):
            total = total + lognormal_log_prior(sub.bias_variance)
            total = total + lognormal_log_prior(sub.weight_variance)
    return total


def hyperparameter_record(module, noise: float) -> dict:
    scales, lengths, alphas, offsets, variances = [], [], [], [], []
    for sub in module.modules():
        if isinstance(sub, gpytorch.kernels.ScaleKernel):
            scales.append(float(sub.outputscale))
        if isinstance(sub, (gpytorch.kernels.RBFKernel, gpytorch.kernels.MaternKernel,
                            gpytorch.kernels.RQKernel)):
            lengths.append(float(sub.lengthscale.reshape(-1)[0]))
        if isinstance(sub, gpytorch.kernels.RQKernel):
            alphas.append(float(sub.alpha.reshape(-1)[0]))
        if isinstance(sub, gpytorch.kernels.PolynomialKernel):
            offsets.append(float(sub.offset.reshape(-1)[0]))
        if isinstance(sub, gpytorch.kernels.LinearKernel):
            variances.append(float(sub.variance.reshape(-1)[0]))
    first = lambda v: (v[0] if v else math.nan)
    return {"outputscale": first(scales), "lengthscale": first(lengths),
            # A large fitted RQ alpha means the kernel has degenerated to an RBF.
            "rq_alpha": first(alphas), "poly_offset": first(offsets),
            "linear_variance": first(variances),
            "all_outputscales": scales, "all_lengthscales": lengths}


def optimise_marginal(module, X, y, seed, iterations=HYPER_ITER_PLAIN, jitter=JITTER_RV):
    """Plain exact-GP marginal likelihood plus this project's priors."""
    torch.manual_seed(int(seed))
    raw_noise = torch.tensor(math.log(math.expm1(0.1)), dtype=torch.float64,
                             requires_grad=True)
    parameters = [q for q in module.parameters() if q.requires_grad] + [raw_noise]
    optimiser = torch.optim.Adam(parameters, lr=HYPER_LR)
    n = X.shape[0]
    identity = torch.eye(n, dtype=torch.float64)
    best = {"value": -math.inf, "step": 0}
    for step in range(iterations):
        optimiser.zero_grad(set_to_none=True)
        noise = torch.nn.functional.softplus(raw_noise) + NOISE_FLOOR
        K = module(X).to_dense() + (noise ** 2 + jitter) * identity
        L = torch.linalg.cholesky(K)                    # raises; never re-jittered
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
        log_marginal = (-0.5 * y @ alpha - torch.log(L.diagonal()).sum()
                        - 0.5 * n * math.log(2 * math.pi))
        objective = (log_marginal + kernel_log_prior(module)
                     + inverse_gamma_log_prior(noise ** 2, GLOBAL_NOISE_PRIOR_SHAPE,
                                                  (GLOBAL_NOISE_PRIOR_SHAPE - 1.0)
                                                  * GLOBAL_NOISE_PRIOR_MEAN))
        loss = -objective
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite marginal likelihood")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 20.0)
        value = float(objective)
        if value > best["value"]:
            best = {"value": value, "step": step,
                    "state": {k: v.detach().clone() for k, v in module.state_dict().items()},
                    "raw_noise": raw_noise.detach().clone()}
        optimiser.step()
    module.load_state_dict(best["state"])
    noise = float(torch.nn.functional.softplus(best["raw_noise"]) + NOISE_FLOOR)
    return {"log_marginal": best["value"], "best_step": best["step"], "noise": noise,
            "converged": best["step"] < iterations - 1}


class Predictive:
    """Predictive on the z scale, either Gaussian (soft / rv / plain) or a draw
    set (integral, whose push-forward through softplus and exp is not Gaussian).

    latent_mean_z is the posterior mean of the latent f alone, without the
    observation noise; only the integral construction sets it, and only the
    residual check of section 3.1 reads it."""

    def __init__(self, mean_z, sd_z, draws_z=None, latent_mean_z=None):
        self.mean_z = np.asarray(mean_z, dtype=float)
        self.sd_z = np.maximum(np.asarray(sd_z, dtype=float), 1e-12)
        self.draws_z = None if draws_z is None else np.asarray(draws_z, dtype=float)
        self.latent_mean_z = (None if latent_mean_z is None
                              else np.asarray(latent_mean_z, dtype=float))

    @property
    def kind(self) -> str:
        return "draws" if self.draws_z is not None else "gaussian"

    def quantile_z(self, q: float) -> np.ndarray:
        if self.draws_z is not None:
            return np.quantile(self.draws_z, q, axis=0)
        return self.mean_z + norm.ppf(q) * self.sd_z

    def median_z(self) -> np.ndarray:
        return self.quantile_z(0.5)

    def cdf_z(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.draws_z is not None:
            S = self.draws_z.shape[0]
            below = (self.draws_z <= values[None, :]).mean(axis=0)
            return np.clip(below, 1.0 / (2 * S), 1.0 - 1.0 / (2 * S))
        return norm.cdf((values - self.mean_z) / self.sd_z)

    def log_density_z(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.draws_z is not None:
            # Gaussian moment approximation of the MC predictive, applied
            # identically to every model so the comparison stays consistent.
            return norm.logpdf(values, loc=self.mean_z, scale=self.sd_z)
        return norm.logpdf(values, loc=self.mean_z, scale=self.sd_z)


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = module

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def _gaussian_predict(module, X, y, noise, Xq, jitter):
    with torch.no_grad():
        n = X.shape[0]
        K = module(X).to_dense() + (noise ** 2 + jitter) * torch.eye(n, dtype=torch.float64)
        L = torch.linalg.cholesky(K)
        ks = module(Xq, X).to_dense()
        mean = ks @ torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
        v = torch.linalg.solve_triangular(L, ks.transpose(0, 1), upper=False)
        prior = module(Xq).to_dense().diagonal()
        var = (prior - (v ** 2).sum(dim=0)).clamp_min(1e-12) + noise ** 2
        cross = float(ks.abs().max())
    return mean.numpy(), var.numpy() ** 0.5, cross


def fit_plain(spec, X, y, adapter, seed, jitter=JITTER_RV):
    module = build_kernel_module(spec)
    hyper = optimise_marginal(module, X, y, seed, HYPER_ITER_PLAIN, jitter)
    record = {"log_marginal": hyper["log_marginal"], "hyper_best_step": hyper["best_step"],
              "hyper_converged": hyper["converged"], "noise": hyper["noise"]}
    record.update(hyperparameter_record(module, hyper["noise"]))

    def predict(frame):
        Xq = torch.tensor(adapter.transform(frame), dtype=torch.float64)
        mean, sd, cross = _gaussian_predict(module, X, y, hyper["noise"], Xq, jitter)
        return Predictive(mean, sd), cross

    return record, {"kind": "plain", "module": module, "X": X, "y": y,
                    "noise": hyper["noise"], "adapter": adapter, "predict": predict,
                    "jitter": jitter}


def integral_context(source, train, test, cohorts, adapter):
    """Grid whose nodes include every training age AND every test age."""
    extra = {c: test.loc[test[COHORT_COL].astype(str).eq(c), AGE_COL].to_numpy(float)
             for c in cohorts}
    grid = CohortGrid(train, cohorts, GRID_M, extra, age_scale=1.0)
    grid_frame = grid.frame(source)
    return grid, grid_frame


def fit_integral(spec, fit_rows, grid, grid_frame, adapter, cohorts, seed, draws=MC_DRAWS):
    fitted = fit_positive_growth_gp(fit_rows, grid, grid_frame, adapter, spec,
                                       S0, int(seed), torch.device("cpu"), HYPER_ITER_INTEGRAL)
    rate = np.asarray(fitted["rate"], dtype=float)
    record = {
        "log_marginal": fitted["log_marginal"], "hyper_best_step": fitted["best_step"],
        "hyper_converged": bool(fitted["converged"]), "noise": float(fitted["sigma_n"]),
        "grid_size": int(grid.size), "grid_M": GRID_M, "s0": S0,
        "mc_draws": int(draws),
        "rate_min": float(rate.min()), "rate_max": float(rate.max()),
        "rate_ratio": float(rate.max() / max(rate.min(), 1e-300)),
    }
    record.update(fitted["hyper"])
    record.update(monotonicity_report(grid, fitted["rate"], subdivisions=100))
    if not record["monotone_between_nodes"]:
        record["fit_failed"] = True
        record["fit_failure_reason"] = ("structural monotonicity violated — this is an "
                                        "implementation error, not a modelling outcome")

    f_draws, rate_draws = posterior_draws(fitted, int(seed) + 7, draws)
    rng = np.random.default_rng(int(seed) + 8)
    f0, noise_var = f0_posterior_draws(fitted, rate_draws, cohorts, rng)

    def predict(frame):
        columns, cohort_index = [], []
        for _, row in frame.iterrows():
            cohort = str(row[COHORT_COL])
            columns.append(grid.index_of(cohort, float(row[AGE_COL])))
            cohort_index.append(cohorts.index(cohort))
        f = f_draws[:, np.asarray(columns, int)] + f0[:, np.asarray(cohort_index, int)]
        z = f + rng.normal(0.0, math.sqrt(noise_var), size=f.shape)
        # mean and sd are those of the predictive draws z = f + eps, so the mean
        # carries the MC noise of eps; latent_mean_z = mean of f, free of it
        return Predictive(z.mean(axis=0), z.std(axis=0, ddof=0), draws_z=z,
                          latent_mean_z=f.mean(axis=0)), math.nan

    return record, {"kind": "integral", "fitted": fitted, "grid": grid,
                    "adapter": adapter, "predict": predict, "rate": rate,
                    "jitter": JITTER_INT}


def quantile_table(predictive: Predictive) -> np.ndarray:
    return np.stack([predictive.quantile_z(q) for q in QUANTILE_LEVELS], axis=0)


class Component:
    """One model's predictive on the z scale at the 8 test rows.

    Gaussian components (soft / rv / plain) keep their moments and use analytic
    formulas.  The integral construction pushes a Laplace Gaussian through
    softplus and exp, so its predictive is not Gaussian and is carried as a
    quantile table; the CDF is the table read backwards.

    LPD uses the Gaussian moment approximation for EVERY component, applied
    identically so the comparison between methods is not confounded by a
    different density estimator.
    """

    def __init__(self, mean_z, sd_z, table=None):
        self.mean_z = np.asarray(mean_z, float)
        sd = np.asarray(sd_z, float)
        # RandomForest and XGBoost supply a point prediction and no spread; their sd
        # is NaN on purpose.  ppf must NOT propagate that into the point estimate --
        # ndtri(0.5) is exactly 0.0 and 0.0 * nan is nan, which silently erased every
        # RF/XGB metric in the first run and was mislabelled as a negative prediction.
        self.point_only = not np.isfinite(sd).any()
        self.sd_z = np.where(np.isfinite(sd), np.maximum(sd, 1e-12), np.nan)
        self.table = None if table is None else np.asarray(table, float)

    def ppf(self, q) -> np.ndarray:
        """q may be a scalar or a vector of levels; a vector returns (len(q), n).

        scipy.special.ndtri/ndtr are the raw ufuncs behind scipy.stats.norm and
        skip the frozen-distribution argument checking, which dominated the
        profile once this was called a third of a million times.
        """
        q = np.asarray(q, float)
        if self.point_only:
            return (np.repeat(self.mean_z[None, :], q.size, axis=0) if q.ndim
                    else self.mean_z.copy())
        if self.table is None:
            return self.mean_z + ndtri(q)[..., None] * self.sd_z if q.ndim else \
                self.mean_z + float(ndtri(q)) * self.sd_z
        if q.ndim == 0:
            return np.array([np.interp(float(q), QUANTILE_LEVELS, self.table[:, j])
                             for j in range(self.table.shape[1])])
        return np.column_stack([np.interp(q, QUANTILE_LEVELS, self.table[:, j])
                                for j in range(self.table.shape[1])])

    def cdf(self, values) -> np.ndarray:
        """values of shape (n,) or (..., n); returns the same shape."""
        values = np.asarray(values, float)
        if self.point_only:
            return np.full(values.shape, np.nan)
        if self.table is None:
            return ndtr((values - self.mean_z) / self.sd_z)
        if values.ndim == 1:
            return np.array([np.interp(values[j], self.table[:, j], QUANTILE_LEVELS)
                             for j in range(self.table.shape[1])])
        return np.column_stack([np.interp(values[..., j], self.table[:, j],
                                          QUANTILE_LEVELS)
                                for j in range(self.table.shape[1])])

    def logpdf(self, values) -> np.ndarray:
        values = np.asarray(values, float)
        if self.point_only:
            return np.full(values.shape, np.nan)
        return norm.logpdf(values, loc=self.mean_z, scale=self.sd_z)


class MixtureComponent:
    """Weighted mixture of Components; a single-component mixture is exact."""

    def __init__(self, parts: list[Component], weights):
        self.parts = parts
        self.weights = np.asarray(weights, float)
        self.weights = self.weights / self.weights.sum()

    @property
    def mean_z(self):
        return sum(w * p.mean_z for w, p in zip(self.weights, self.parts))

    @property
    def sd_z(self):
        mean = self.mean_z
        second = sum(w * (p.sd_z ** 2 + p.mean_z ** 2)
                     for w, p in zip(self.weights, self.parts))
        return np.sqrt(np.maximum(second - mean ** 2, 1e-18))

    def cdf(self, values):
        return sum(w * p.cdf(values) for w, p in zip(self.weights, self.parts))

    def ppf(self, q: float) -> np.ndarray:
        return self.ppf_grid(np.asarray([float(q)]))[0]

    def ppf_grid(self, levels) -> np.ndarray:
        """Quantiles at every level in one bisection: (len(levels), n).

        The previous version bisected each level separately, so building the
        207-level grid cost 207 x 80 mixture CDF evaluations, each of which
        evaluated every component.  That was 320,640 component CDF calls per
        two settings and 56% of assembly time.  Bisecting the whole grid at once
        is the identical computation with 207x fewer Python-level dispatches.
        """
        levels = np.asarray(levels, float)
        active = self.weights > 0
        if int(active.sum()) == 1:
            part = self.parts[int(np.flatnonzero(active)[0])]
            return np.atleast_2d(part.ppf(levels))
        lo = np.min([p.ppf(0.0005) for p in self.parts], axis=0)
        hi = np.max([p.ppf(0.9995) for p in self.parts], axis=0)
        lo = np.repeat(lo[None, :], len(levels), axis=0)
        hi = np.repeat(hi[None, :], len(levels), axis=0)
        target = levels[:, None]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def logpdf(self, values):
        parts = np.stack([p.logpdf(values) for p in self.parts], axis=1)
        active = self.weights > 0
        return logsumexp(parts[:, active] + np.log(self.weights[active])[None, :], axis=1)


# The backends whose predict() returns a Gaussian Predictive, without draws: the
# plain GP ("shared"), the soft penalty and Riihimaki-Vehtari.  A LOO record
# written before loo_kind existed takes its kind from its method ONLY for these;
# an integral record without its quantile table is an error, never replaced by a
# Gaussian built from its mean and sd.
GAUSSIAN_LOO_METHODS = ("soft", "rv", "shared")


def has_loo_distribution(record) -> bool:
    """True when a LOO record carries the held-out predictive loo_components needs."""
    kind = record.get("loo_kind")
    return kind == "gaussian" or (kind == "draws" and record.get("loo_quantiles") is not None)


def loo_kind_of(record) -> str:
    kind = record.get("loo_kind")
    if kind is not None:
        return str(kind)
    if record.get("method") in GAUSSIAN_LOO_METHODS:
        return "gaussian"
    raise KeyError(f"{record.get('fit_key')}: no loo_kind / loo_quantiles -- this LOO "
                   "record predates them; run the fit stages so they are backfilled")


def loo_components(records) -> Component:
    """One model's held-out LOO predictives over the training rows as ONE Component:
    column i is row i's LOO predictive, on fold i's own z scale.

    An integral (draws) predictive is its stored quantile table on QUANTILE_LEVELS;
    a Gaussian one (plain / soft / rv) is its mean and sd, which is exactly what
    those backends predict."""
    kinds = {loo_kind_of(r) for r in records}
    mean = [float(r["loo_mean_z"]) for r in records]
    sd = [float(r["loo_sd_z"]) for r in records]
    if kinds == {"gaussian"}:
        return Component(mean, sd)
    if kinds != {"draws"}:
        raise ValueError(f"LOO predictive kinds {sorted(kinds)}: expected one kind per model")
    missing = [r.get("fit_key") for r in records if r.get("loo_quantiles") is None]
    if missing:
        raise KeyError(f"{len(missing)} LOO records carry no loo_quantiles "
                       f"(e.g. {missing[0]}); run the fit stages so they are backfilled")
    tables = [np.asarray(r["loo_quantiles"], float) for r in records]
    bad = [r.get("fit_key") for r, t in zip(records, tables)
           if t.shape != (len(QUANTILE_LEVELS), 1)]
    if bad:
        raise ValueError(f"loo_quantiles of {bad[0]} is not ({len(QUANTILE_LEVELS)}, 1)")
    return Component(mean, sd, np.concatenate(tables, axis=1))


def loo_mixture_summary(kernels, loo, weights) -> dict:
    """The LOO stacking mixture at every training row -- the centre of the
    conformity scores of Table 9, E1-E3 and both interval figures.

    Row i: every kernel's held-out LOO predictive CDF at row i, mixed with ONE
    weight vector shared by all rows (the stacking weights of the full LOO density
    matrix), and the 0.5 quantile of that mixture CDF found numerically by
    MixtureComponent.ppf.  A mixture's median is not the weighted sum of its
    components' medians, so that sum is not used."""
    loo_mixture = MixtureComponent([loo_components(loo[k]) for k in kernels], weights)
    loo_truth = np.array([float(r["truth_z"]) for r in loo[kernels[0]]])
    loo_median = np.asarray(loo_mixture.ppf(0.5), float)
    at_median = np.asarray(loo_mixture.cdf(loo_median), float)
    if not np.all(np.abs(at_median - 0.5) < 1e-6):
        raise RuntimeError(f"LOO mixture median is not its 0.5 quantile: F = {at_median}")
    return {"mixture": loo_mixture, "truth": loo_truth, "median": loo_median,
            "mean": np.asarray(loo_mixture.mean_z, float),
            "sd": np.asarray(loo_mixture.sd_z, float), "cdf_at_median": at_median}


def conformity_scores(loo) -> np.ndarray:
    """|truth_z - the LOO mixture summary POINT_SUMMARY names| (its median by default)."""
    return np.abs(np.asarray(loo["truth"], float) - np.asarray(loo[POINT_SUMMARY], float))


def ordered_test(test, cohorts):
    order = []
    for cohort in cohorts:
        rows = test.index[test[COHORT_COL].astype(str).eq(cohort)].tolist()
        order.extend(sorted(rows, key=lambda i: float(test.loc[i, AGE_COL])))
    return test.loc[order].reset_index(drop=True)


def conformal_quantile(scores, level) -> float:
    scores = np.sort(np.asarray(scores, float))
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return math.nan
    rank = min(int(math.ceil((len(scores) + 1) * level)), len(scores))
    return float(scores[rank - 1])


def jackknife_ranks(n: int, level: float):
    tail = (1.0 - level) / 2.0
    lower_exact = int(math.floor((n + 1) * tail))
    upper_exact = int(math.ceil((n + 1) * (1.0 - tail)))
    lower, upper = max(1, lower_exact), min(n, upper_exact)
    return lower, upper, bool(lower != lower_exact or upper != upper_exact)


def mixture_lpd(log_density, weights) -> float:
    log_density = np.asarray(log_density, float)
    weights = np.asarray(weights, float)
    active = weights > 0.0
    if not np.any(active):
        return -math.inf
    return float(np.mean(logsumexp(log_density[:, active]
                                   + np.log(weights[active])[None, :], axis=1)))


def optimize_weights(log_density) -> np.ndarray:
    """Yao et al. (2018) objective (2.2) on the simplex, SLSQP, multi-start.

    The gradient is supplied analytically:

        f(w)        = -(1/n) sum_i log sum_j w_j p_ij
        df/dw_k     = -(1/n) sum_i p_ik / sum_j w_j p_ij

    evaluated as exp(log p_ik - logsumexp_j(log p_ij + log w_j)), which is stable
    for the very small densities this data produces.  Without it SLSQP finite-
    differences the objective, costing k+1 extra evaluations per gradient; with it
    the call is 5.6-7.8x faster and the answer is unchanged (weights agree to
    1.3e-6, objective to 1.5e-12 -- both inside the optimiser's own tolerance).
    That matters because the cohort bootstrap calls this 400 times per setting.
    """
    log_density = np.asarray(log_density, float)
    k = log_density.shape[1]
    if k == 1:
        return np.ones(1)

    def objective(w):
        w = np.clip(np.asarray(w, float), 0.0, None)
        if w.sum() <= 0.0:
            return math.inf, np.zeros(k)
        log_w = np.where(w > 0.0, np.log(np.maximum(w, 1e-300)), -np.inf)
        denominator = logsumexp(log_density + log_w[None, :], axis=1)
        value = -float(np.mean(denominator))
        gradient = -np.exp(log_density - denominator[:, None]).mean(axis=0)
        return value, gradient

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones(k)}]
    best = None
    for start in [np.ones(k) / k, *np.eye(k)]:
        result = minimize(objective, start, jac=True, method="SLSQP",
                          bounds=[(0.0, 1.0)] * k, constraints=constraints,
                          options={"maxiter": 500, "ftol": 1e-10})
        if result.success and np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("stacking weight optimisation failed from every start")
    weights = np.clip(best.x, 0.0, 1.0)
    return weights / weights.sum()


def cohort_bootstrap_weights(log_density, cohort_of_row, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Resample COHORTS, not rows: six rows of one cohort are far from independent,
    so a row-level Bayesian bootstrap understates the weight variability.  Yao et
    al. use theirs for Pseudo-BMA+ (section 2.4), not for stacking."""
    rng = np.random.default_rng(seed)
    cohorts = np.unique(cohort_of_row)
    output = np.full((draws, log_density.shape[1]), np.nan)
    for b in range(draws):
        picked = rng.choice(cohorts, size=len(cohorts), replace=True)
        rows = np.concatenate([np.flatnonzero(cohort_of_row == c) for c in picked])
        try:
            output[b] = optimize_weights(log_density[rows])
        except RuntimeError:
            pass
    return output


def exp_entropy(weights) -> float:
    active = np.asarray(weights, float)
    active = active[active > 0.0]
    return float(np.exp(-np.sum(active * np.log(active))))


def shortest_interval(levels, values, mass):
    """Shortest [a, b] on a monotone quantile grid carrying `mass`.

    Works for any predictive, Gaussian or table-based, and is what makes the
    gram-scale HPD variant differ from the equal-tailed one: a CDF-based score is
    invariant under monotone reparameterisation, so a naive "gram-scale CDF"
    score would return the identical interval to the z-scale one.
    """
    levels = np.asarray(levels, float)
    values = np.asarray(values, float)
    upper_index = np.searchsorted(levels, levels + mass, side="left")
    valid = upper_index < len(levels)
    if not valid.any():
        return values[0], values[-1]
    starts = np.flatnonzero(valid)
    widths = values[upper_index[starts]] - values[starts]
    pick = starts[int(np.argmin(widths))]
    return values[pick], values[upper_index[pick]]


def hpd_mass_for(levels, values, truth) -> float:
    """Predictive mass of the highest-density region whose boundary reaches `truth`.

    Computed from the density directly -- p = d(level)/d(value) on the quantile
    grid -- rather than by bisecting on the shortest interval.  Same quantity for
    a unimodal predictive, one numpy pass instead of forty, and it does not
    inherit the bisection's resolution limit.
    """
    levels = np.asarray(levels, float)
    values = np.asarray(values, float)
    truth = float(truth)
    if not np.isfinite(truth) or truth <= values[0] or truth >= values[-1]:
        return 1.0
    delta_value = np.diff(values)
    delta_level = np.diff(levels)
    midpoints = 0.5 * (values[:-1] + values[1:])
    density = np.where(delta_value > 0.0, delta_level / np.maximum(delta_value, 1e-300),
                       np.inf)
    order = np.argsort(midpoints)
    at_truth = float(np.interp(truth, midpoints[order], density[order]))
    inside = density >= at_truth
    return float(np.clip(delta_level[inside].sum(), 0.0, 1.0))


def fit_ml_models(ctx, target, response_mode, step):
    """Comparators on the same inputs, one fit per (response variable, step)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import BayesianRidge
    from xgboost import XGBRegressor

    rows = ctx["fit_rows"](None)
    response = Response(response_mode).fit(rows["_w"].to_numpy(float))
    if response.fit_failed:
        return {}, response
    y = response.to_z(rows["_w"].to_numpy(float))
    x_train = ctx["adapter"].transform(rows)
    x_test = ctx["adapter"].transform(target)
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=SEED,
                                              n_jobs=1, max_features=1.0),
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
                                subsample=0.9, colsample_bytree=0.9,
                                objective="reg:squarederror", reg_lambda=1.0,
                                random_state=SEED, n_jobs=1, verbosity=0),
        "BayesianRidge": BayesianRidge(),
    }
    out = {}
    for name, model in models.items():
        model.fit(x_train, y)
        if name == "BayesianRidge":
            mean, sd = model.predict(x_test, return_std=True)
            out[name] = Component(np.asarray(mean, float),
                                  np.maximum(np.asarray(sd, float), 1e-9))
        else:
            mean = np.asarray(model.predict(x_test), float)
            out[name] = Component(mean, np.full(len(mean), np.nan))
    return out, response


# =============================================================================
# Pipeline
# =============================================================================
def prepare():
    """Load the workbook, attach the gram-scale response, split train / test."""
    data = load_data(DATA_PATH)
    weights = pd.to_numeric(data[RAW_WEIGHT_COL], errors="coerce")
    assert weights.notna().all(), f"{RAW_WEIGHT_COL} has missing values"
    assert (weights > 0).all(), f"{RAW_WEIGHT_COL} must be strictly positive"
    data["_w"] = weights.astype(float)
    train = data.loc[~data["is_test"]].copy().reset_index(drop=True)
    test = data.loc[data["is_test"]].copy().reset_index(drop=True)
    test = test.sort_values([COHORT_COL, TIME_COL]).reset_index(drop=True)
    test["horizon"] = test.groupby(COHORT_COL, observed=True).cumcount() + 1
    cohorts = sorted(train[COHORT_COL].astype(str).unique().tolist())
    return data, train, test, cohorts


def make_context(train, test, cohorts):
    """Everything the fits share: the input frame, the feature adapter, the
    integration grid, and the dense scan grid Table 5 audits monotonicity on."""
    source = input_source(train, test)
    age_domain = (float(train[AGE_COL].min()), float(train[AGE_COL].max()))
    adapter = FeatureAdapter(cohorts, age_domain).fit(train)
    dense = scan_frame(source, train, test, cohorts)
    grid, grid_nodes = integral_context(source, train, test, cohorts, adapter)

    def fit_rows(position):
        rows = train if position is None else train.drop(index=position)
        return rows.reset_index(drop=True)

    return {"train": train, "test": test, "cohorts": cohorts, "source": source,
            "adapter": adapter, "dense": dense, "integral": (grid, grid_nodes),
            "fit_rows": fit_rows, "age_domain": age_domain}


def build_context():
    data, train, test, cohorts = prepare()
    ctx = make_context(train, test, cohorts)
    target = ordered_test(test, cohorts)
    response = Response(RESPONSE).fit(train["_w"].to_numpy(float))
    return {"ctx": ctx, "train": train, "test": test, "cohorts": cohorts,
            "target": target, "response": response, "grid": ctx["integral"][0],
            "truth_g": target["_w"].to_numpy(float),
            "cohort_of_row": train[COHORT_COL].astype(str).to_numpy()}


def grid_frame(context) -> pd.DataFrame:
    """Every integration node as a prediction row -- the growth curve of Figure 2.

    The integral construction predicts only at exact grid nodes, so the curve is
    drawn ON the grid.  M = 4 subdivisions between consecutive observed ages give
    29 nodes per facility, 116 in all.
    """
    grid, source = context["grid"], context["ctx"]["source"]
    blocks = []
    for cohort in grid.cohorts:
        ages = np.asarray(grid.nodes[cohort], float)
        blocks.append(age_frame(source, [cohort], {cohort: ages}))
    frame = pd.concat(blocks, ignore_index=True)
    boundary = context["train"].groupby(COHORT_COL, observed=True)[AGE_COL].max().to_dict()
    frame["window"] = ["train" if row[AGE_COL] <= float(boundary[str(row[COHORT_COL])])
                       else "test" for _, row in frame.iterrows()]
    return frame


def component_of(record) -> Component:
    table = record.get("test_quantiles")
    return Component(record["test_mean_z"], record["test_sd_z"],
                     None if table is None else np.asarray(table))


def metrics_of(component, context, level=NOMINAL) -> dict:
    """Gram-scale test metrics.  The point prediction is the predictive summary
    POINT_SUMMARY names, mapped back to grams: the predictive mean on the modelling
    scale, or its median.  Whichever it is, it is also the centre of the conformal
    interval, so the reported point prediction and the reported interval refer to
    the same quantity."""
    response, truth_g = context["response"], context["truth_g"]
    point_g = response.from_z(point_z(component))
    reference = float(context["train"]["_w"].mean())
    sd_z = np.asarray(component.sd_z, float)
    row = {"g_RMSE": float(np.sqrt(np.mean((point_g - truth_g) ** 2))),
           "g_MAE": float(np.mean(np.abs(point_g - truth_g))),
           "g_MAPE": float(np.mean(np.abs(point_g - truth_g) / truth_g) * 100.0),
           "g_R2_pred": float(1.0 - np.sum((truth_g - point_g) ** 2)
                              / np.sum((truth_g - reference) ** 2))}
    if np.isfinite(sd_z).all():
        low = response.from_z(component.ppf((1 - level) / 2.0))
        high = response.from_z(component.ppf(1 - (1 - level) / 2.0))
        row["native_width_g"] = float(np.mean(high - low))
        row["native_coverage"] = float(np.mean((truth_g >= low) & (truth_g <= high)))
    else:
        row.update({"native_width_g": math.nan, "native_coverage": math.nan})
    return row


def curve_monotonicity(values, window) -> dict:
    """Decreasing steps of an age-ordered curve, split by train / test window."""
    steps = np.diff(values)
    side = window[1:]
    result = {}
    for label in ("train", "test"):
        mask = side == label
        result[f"violations_{label}"] = int(np.sum(steps[mask] < -MONOTONE_TOL))
        result[f"steps_{label}"] = int(mask.sum())
    result["min_step"] = float(steps.min()) if steps.size else math.nan
    return result


# =============================================================================
# Checkpointing
# =============================================================================
class Store:
    """Append-only JSONL keyed by fit_key, so an interrupted run resumes."""

    def __init__(self, path: Path):
        self.path = path
        self.records = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["fit_key"]] = record
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def __contains__(self, key): return key in self.records

    def __getitem__(self, key): return self.records[key]

    def add(self, record):
        self.records[record["fit_key"]] = record
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self): self.handle.close()


def strip(record: dict) -> dict:
    clean = {}
    for key, value in record.items():
        if key in ("state", "response_obj"):
            continue
        if isinstance(value, np.ndarray):
            clean[key] = [float(v) for v in value.ravel()]
        elif isinstance(value, (np.floating, np.integer)):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def loo_distribution_fields(held_predictive) -> dict:
    """The held-out row's LOO predictive, from the predict() call the worker
    already made: its kind, and for the integral construction its quantile table
    on QUANTILE_LEVELS, shape (len(QUANTILE_LEVELS), 1) kept by tolist()."""
    return {"loo_kind": held_predictive.kind,
            "loo_quantiles": quantile_table(held_predictive).tolist()
            if held_predictive.kind == "draws" else None}


# Stored fields a backfill refit is compared on; it must reproduce every one.
BACKFILL_FIELDS = ("job_seed", "truth_z", "loo_mean_z", "loo_sd_z", "loo_median_z",
                   "loo_lpd_z", "loo_lpd_g", "loo_cdf_at_truth", "log_marginal", "noise",
                   "test_mean_z", "test_sd_z", "test_quantiles")


def backfill_loo_record(stored: dict, regenerated: dict) -> dict:
    """The STORED record plus the regenerated loo_kind / loo_quantiles.

    Nothing stored is overwritten.  The largest absolute difference between the
    regenerated and the stored value of every BACKFILL_FIELDS entry is kept with
    the record, and report_loo_backfill prints any that exceeds REPRO_TOL."""
    if "loo_kind" not in regenerated:
        raise RuntimeError(f"{stored['fit_key']}: the backfill refit failed -- "
                           f"{regenerated.get('fit_failure_reason')}")
    gaps = {}
    for field in BACKFILL_FIELDS:
        if stored.get(field) is None or regenerated.get(field) is None:
            continue
        old = np.asarray(stored[field], float)
        new = np.asarray(regenerated[field], float)
        gaps[field] = (float(np.max(np.abs(old - new))) if old.shape == new.shape
                       else math.inf)
    worst = max(gaps, key=gaps.get) if gaps else ""
    merged = dict(stored)
    merged.update({"loo_kind": regenerated["loo_kind"],
                   "loo_quantiles": regenerated["loo_quantiles"],
                   "loo_backfill_gaps": gaps, "loo_backfill_worst": worst,
                   "loo_backfill_max_gap": gaps[worst] if worst else math.nan,
                   "loo_backfill_consistent": bool(worst) and gaps[worst] <= REPRO_TOL})
    return merged


def report_loo_backfill(records: dict, label: str) -> None:
    """Every backfilled LOO record and how far its refit is from the stored values."""
    rows = [{"fit_key": key, "consistent": bool(r["loo_backfill_consistent"]),
             "max_gap": r["loo_backfill_max_gap"], "worst_field": r["loo_backfill_worst"],
             **{f"gap_{f}": v for f, v in r["loo_backfill_gaps"].items()}}
            for key, r in records.items() if "loo_backfill_gaps" in r]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    path = VERSION_DIR / f"loo_backfill_check_{label}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding=ENC)
    bad = frame[~frame.consistent]
    print(f"  LOO distribution backfill ({label}): {len(frame) - len(bad)} of {len(frame)} "
          f"refits reproduce the stored values within {REPRO_TOL:g} "
          f"(largest gap {frame.max_gap.max():.2e}) -> {path.name}")
    if len(bad):
        print(f"  WARNING -- {len(bad)} refits differ from the stored record; the stored "
              f"values were kept, the quantile table is the refit's:")
        print(bad[["fit_key", "max_gap", "worst_field"]].to_string(index=False))


def run_fit(item, ctx):
    """One fit: monotone integral construction, or the same kernel as a plain GP."""
    started = time.perf_counter()
    record = dict(item)
    record.update({"fit_failed": False, "fit_failure_reason": ""})
    fit_rows = ctx["fit_rows"](item.get("position"))
    response = Response(RESPONSE).fit(fit_rows["_w"].to_numpy(float))
    record.update(response.describe())
    if response.fit_failed:
        raise FloatingPointError(response.fit_failure_reason)
    z = response.to_z(fit_rows["_w"].to_numpy(float))
    fit_rows = fit_rows.copy()
    fit_rows["_z"] = z
    record["n_fit_rows"] = int(len(fit_rows))

    X = torch.tensor(ctx["adapter"].transform(fit_rows), dtype=torch.float64)
    y = torch.tensor(z, dtype=torch.float64)
    seed = int(item["job_seed"])
    if item["monotone"]:
        grid, grid_nodes = ctx["integral"]
        detail, state = fit_integral(item["kernel"], fit_rows, grid, grid_nodes,
                                     ctx["adapter"], ctx["cohorts"], seed)
    else:
        detail, state = fit_plain(item["kernel"], X, y, ctx["adapter"], seed)
    record.update(detail)
    record["fit_seconds"] = float(time.perf_counter() - started)
    return record, state, response


# =============================================================================
# Stage 1 -- the 78 fits
# =============================================================================
def stage_fits(context, workers) -> None:
    banner("Stage 1 — fits (72 LOO + 3 final + 3 unconstrained)")
    ctx, target = context["ctx"], context["target"]
    train, dense = context["train"], context["ctx"]["dense"]
    curve = grid_frame(context)
    store = Store(CKPT / "fits.jsonl")

    items = []
    for kernel in KERNELS:
        for position in range(len(train)):
            items.append({"kernel": kernel, "position": position, "monotone": True,
                          "role": "loo", "fit_key": f"loo|{kernel}|{position}",
                          "job_seed": SEEDS[f"loo|{kernel}|{position}"]})
        items.append({"kernel": kernel, "position": None, "monotone": True,
                      "role": "final", "fit_key": f"final|{kernel}",
                      "job_seed": SEEDS[f"final|{kernel}"]})
        items.append({"kernel": kernel, "position": None, "monotone": False,
                      "role": "plain", "fit_key": f"plain|{kernel}",
                      "job_seed": SEEDS[f"plain|{kernel}"]})

    def worker(item):
        record, state, response = run_fit(item, ctx)
        # ORDER MATTERS.  The integral construction's predict() draws the
        # observation noise from one RNG stream created inside the fit, so every
        # call consumes from it and a later call sees different draws.  Both
        # predictive moments are taken over those draws, so reordering these
        # calls moves them (the SD by about 5e-3); only the latent mean f.mean
        # draws nothing.  The sequence below is the one the published numbers were
        # produced with -- held-out row first for a LOO fit, then the test rows.
        if item["role"] == "loo":
            # The held-out row: its LOO predictive density is the stacking
            # objective's input, and |truth - mean| is the conformity score.
            position = int(item["position"])
            held = train.iloc[[position]]
            held_predictive, _ = state["predict"](held)
        predictive, _ = state["predict"](target)
        record.update({"test_mean_z": [float(v) for v in predictive.mean_z],
                       "test_sd_z": [float(v) for v in predictive.sd_z],
                       "test_quantiles": quantile_table(predictive).tolist()
                       if predictive.kind == "draws" else None})
        if item["role"] == "loo":
            weight = float(held["_w"].iloc[0])
            truth_z = float(response.to_z(np.array([weight]))[0])
            mean_z = float(held_predictive.mean_z[0])
            sd_z = float(held_predictive.sd_z[0])
            lpd_z = float(norm.logpdf(truth_z, loc=mean_z, scale=sd_z))
            record.update({
                "truth_w": weight, "truth_z": truth_z, "loo_mean_z": mean_z,
                # the held-out row's own predictive median (kept for reference);
                # the conformity score uses the median of the LOO MIXTURE, built
                # from the quantile table below
                "loo_median_z": float(np.asarray(held_predictive.median_z(), float)[0]),
                "loo_sd_z": sd_z, "loo_lpd_z": lpd_z,
                "loo_lpd_g": lpd_z + float(response.log_dz_dw(np.array([weight]))[0]),
                "loo_cdf_at_truth": float(held_predictive.cdf_z(np.array([truth_z]))[0])})
            record.update(loo_distribution_fields(held_predictive))
        elif item["role"] == "final":
            curve_predictive, _ = state["predict"](curve)
            record.update({
                "curve_mean_z": [float(v) for v in curve_predictive.mean_z],
                "curve_sd_z": [float(v) for v in curve_predictive.sd_z],
                "curve_quantiles": quantile_table(curve_predictive).tolist()
                if curve_predictive.kind == "draws" else None})
            # Section 3.1 checks the 24 training residuals of this fit, which need
            # the latent posterior mean at the training rows, mean of f WITHOUT the
            # observation noise.  Predicted LAST, so the published test and curve
            # predictives stay the ones the calls above produced -- every predict()
            # consumes from the fit's RNG stream.  train_mean_z, the mean of
            # f + eps, is kept for reference only.
            train_predictive, _ = state["predict"](train)
            record["train_mean_z"] = [float(v) for v in train_predictive.mean_z]
            record["train_latent_mean_z"] = [float(v)
                                             for v in train_predictive.latent_mean_z]
        else:
            dense_predictive, _ = state["predict"](dense)
            record["dense_mean_z"] = [float(v) for v in dense_predictive.mean_z]
        return strip(record)

    # A checkpoint written before a field existed cannot serve a run that needs it:
    # loo_median_z, train_mean_z and train_latent_mean_z were added after the
    # first runs.
    needed = {"loo": ("loo_median_z",),
              "final": ("train_mean_z", "train_latent_mean_z"), "plain": ()}

    def complete(item):
        record = store.records.get(item["fit_key"])
        return record is not None and all(field in record
                                          for field in needed[item["role"]])

    todo = [i for i in items if not complete(i)]
    if not todo:
        print(f"  all {len(items)} fits already checkpointed")
    else:
        print(f"  fitting {len(todo)} of {len(items)} ({workers} workers)", flush=True)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, i) for i in todo]
            for done, future in enumerate(as_completed(futures), start=1):
                store.add(future.result())
                if done % 10 == 0 or done == len(todo):
                    print(f"    {done}/{len(todo)}  "
                          f"{time.perf_counter() - started:.0f} s", flush=True)

    # loo_kind / loo_quantiles came later still, but a record lacking only them is
    # not replaced: the fit is redone by the same worker -- same seed, held-out row
    # first, then the test rows -- and backfill_loo_record adds the two fields to
    # the STORED record, keeping every stored value, and records how far the
    # regenerated predictive is from it.
    backfill = [i for i in items if i["role"] == "loo" and complete(i)
                and not has_loo_distribution(store.records[i["fit_key"]])]
    if backfill:
        print(f"  backfilling the LOO predictive distribution of {len(backfill)} fits "
              f"({workers} workers; stored values are kept)", flush=True)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, i): i for i in backfill}
            for done, future in enumerate(as_completed(futures), start=1):
                key = futures[future]["fit_key"]
                store.add(backfill_loo_record(store.records[key], future.result()))
                if done % 10 == 0 or done == len(backfill):
                    print(f"    {done}/{len(backfill)}  "
                          f"{time.perf_counter() - started:.0f} s", flush=True)
    report_loo_backfill(store.records, "fits")
    failed = [k for k, r in store.records.items() if r.get("fit_failed")]
    store.close()
    if failed:
        raise RuntimeError(f"fits failed: {failed}")


# =============================================================================
# Stage 2 -- stacking weights and the conformal LOO predictive
# =============================================================================
def assemble(context):
    store = Store(CKPT / "fits.jsonl")
    store.close()
    n_rows = len(context["train"])
    missing = [i["fit_key"] for k in KERNELS
               for i in ({"fit_key": f"final|{k}"}, {"fit_key": f"plain|{k}"})
               if i["fit_key"] not in store]
    if missing:
        raise RuntimeError(f"missing fits {missing}; run without --tables-only first")

    records = {k: [store[f"loo|{k}|{p}"] for p in range(n_rows)] for k in KERNELS}
    matrix = np.column_stack([[r["loo_lpd_g"] for r in records[k]] for k in KERNELS])
    weights = optimize_weights(matrix)

    # One weight vector for every conformity score: the stacking weights of the
    # full LOO density matrix.  `cross` keeps its (rows x kernels) shape for the
    # code that reads it, every row holding those same weights.
    cross = np.tile(weights, (n_rows, 1))

    singles = {k: component_of(store[f"final|{k}"]) for k in KERNELS}
    mixture = MixtureComponent([singles[k] for k in KERNELS], weights)

    # The LOO mixture at each training row and its median (loo_mixture_summary).
    loo = loo_mixture_summary(KERNELS, records, weights)

    stack = {"weights": weights, "cross": cross, "lpd": matrix,
             "expH": float(exp_entropy(weights)),
             "mixture_lpd": float(mixture_lpd(matrix, weights)),
             "boot": cohort_bootstrap_weights(matrix, context["cohort_of_row"],
                                              BOOTSTRAP_DRAWS, SEED)}
    return {"store": store, "records": records, "singles": singles,
            "mixture": mixture, "stack": stack, "loo": loo}


def conformal_radius(parts, level=NOMINAL) -> float:
    """gamma = 0 conformal radius on the z scale: the (n+1)-adjusted quantile of
    the LOO absolute residuals, measured from the same predictive summary of the
    LOO mixture that the point prediction uses (its median by default)."""
    scores = conformity_scores(parts["loo"])
    return float(conformal_quantile(scores, level))


def conformal_band(component, parts, context, level=NOMINAL):
    radius = conformal_radius(parts, level)
    response = context["response"]
    centre = point_z(component)
    return response.from_z(centre - radius), response.from_z(centre + radius)
# =============================================================================
# Tables
# =============================================================================
def table_models(context, parts) -> pd.DataFrame:
    """Tables 2 and 3 -- stacking against its own components, and against the
    three machine-learning baselines, on one gram-scale ledger."""
    rows = [{"Block": "component", "Model": "Stacking", "Weight": math.nan,
             **metrics_of(parts["mixture"], context)}]
    for kernel, weight in zip(KERNELS, parts["stack"]["weights"]):
        rows.append({"Block": "component", "Model": KERNEL_LABEL[kernel],
                     "Weight": float(weight),
                     **metrics_of(parts["singles"][kernel], context)})
    ml, _ = fit_ml_models(context["ctx"], context["target"], RESPONSE, 0)
    for name in ML_MODEL_NAMES:
        rows.append({"Block": "baseline", "Model": ML_LABEL[name],
                     "Weight": math.nan, **metrics_of(ml[name], context)})
    frame = pd.DataFrame(rows)
    component = frame[frame["Block"].eq("component")].sort_values("g_RMSE")
    baseline = pd.concat([frame[frame["Model"].eq("Stacking")],
                          frame[frame["Block"].eq("baseline")].sort_values("g_RMSE")])
    return pd.concat([component.assign(Table="Table 2"),
                      baseline.assign(Table="Table 3")], ignore_index=True)


def table_ablation(context, parts) -> pd.DataFrame:
    """Table 4 -- drop one kernel, re-maximise the LOO objective on the rest.

    No refitting: the LOO predictive densities are per kernel, so any subset is
    re-weighted from the stored matrix.
    """
    full = metrics_of(parts["mixture"], context)
    rows = [{"Model": "Stacking (all three)", "Removed": "—", "n_kernels": 3,
             "expH": parts["stack"]["expH"],
             "weights": ", ".join(f"{KERNEL_LABEL[k]}:{w:.3f}"
                                  for k, w in zip(KERNELS, parts["stack"]["weights"])),
             **{column: full[column] for column in ("g_RMSE", "g_MAE", "g_MAPE",
                                                    "g_R2_pred")},
             "delta_g_MAPE": 0.0, "delta_g_MAE": 0.0}]
    for dropped in KERNELS:
        kept = [k for k in KERNELS if k != dropped]
        matrix = np.column_stack([[r["loo_lpd_g"] for r in parts["records"][k]]
                                  for k in kept])
        weights = optimize_weights(matrix)
        metric = metrics_of(MixtureComponent([parts["singles"][k] for k in kept],
                                             weights), context)
        rows.append({"Model": f"Without {KERNEL_LABEL[dropped]}", "Removed": dropped,
                     "n_kernels": 2, "expH": float(exp_entropy(weights)),
                     "weights": ", ".join(f"{KERNEL_LABEL[k]}:{w:.3f}"
                                          for k, w in zip(kept, weights)),
                     **{column: metric[column] for column in ("g_RMSE", "g_MAE",
                                                              "g_MAPE", "g_R2_pred")},
                     "delta_g_MAPE": metric["g_MAPE"] - full["g_MAPE"],
                     "delta_g_MAE": metric["g_MAE"] - full["g_MAE"]})
    return pd.DataFrame(rows)


def table_intervals(context, parts) -> pd.DataFrame:
    """The interval numbers quoted in §3.2: native vs conformal width and
    coverage at the nominal level, overall and per test row."""
    response, truth_g = context["response"], context["truth_g"]
    component = parts["mixture"]
    tail = (1.0 - NOMINAL) / 2.0
    native_lo = response.from_z(component.ppf(tail))
    native_hi = response.from_z(component.ppf(1.0 - tail))
    cp_lo, cp_hi = conformal_band(component, parts, context, NOMINAL)
    target = context["target"]
    rows = []
    for i in range(len(truth_g)):
        native_w = float(native_hi[i] - native_lo[i])
        cp_w = float(cp_hi[i] - cp_lo[i])
        rows.append({
            "Facility": ROMAN.get(str(target[COHORT_COL].iloc[i]),
                                  str(target[COHORT_COL].iloc[i])),
            "Occasion": f"t{6 + int(target['horizon'].iloc[i])}",
            "Age_days": float(target[AGE_COL].iloc[i]),
            "Observed_g": float(truth_g[i]),
            "Native_low_g": float(native_lo[i]), "Native_high_g": float(native_hi[i]),
            "Native_width_g": native_w,
            "Conformal_low_g": float(cp_lo[i]), "Conformal_high_g": float(cp_hi[i]),
            "Conformal_width_g": cp_w,
            "Width_reduction_pct": 100.0 * (1.0 - cp_w / max(native_w, 1e-12)),
            "Native_covers": bool(native_lo[i] <= truth_g[i] <= native_hi[i]),
            "Conformal_covers": bool(cp_lo[i] <= truth_g[i] <= cp_hi[i])})
    frame = pd.DataFrame(rows)
    native_mean = float(frame["Native_width_g"].mean())
    cp_mean = float(frame["Conformal_width_g"].mean())
    frame.loc[len(frame)] = {
        "Facility": "ALL", "Occasion": "—", "Age_days": math.nan,
        "Observed_g": math.nan, "Native_low_g": math.nan, "Native_high_g": math.nan,
        "Native_width_g": native_mean, "Conformal_low_g": math.nan,
        "Conformal_high_g": math.nan, "Conformal_width_g": cp_mean,
        "Width_reduction_pct": 100.0 * (1.0 - cp_mean / max(native_mean, 1e-12)),
        "Native_covers": bool(frame["Native_covers"].all()),
        "Conformal_covers": int(frame["Conformal_covers"].sum())}
    return frame


# =============================================================================
# Entry point
# =============================================================================
def show(frame: pd.DataFrame, columns=None, digits=3) -> None:
    view = frame if columns is None else frame[columns]
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(view.to_string(index=False,
                             float_format=lambda v: f"{v:.{digits}f}"))


# =============================================================================
# Derivative kernels for the Riihimaki-Vehtari arm
#
# Carried over from 0903_abalone5_kernels.py, 0903_abalone8_kernels.py and
# 0905_abalone1_kernels.py, which import one another in that order. Each
# defines its own `wrap`; here they are dk5_wrap, dk8_wrap and wrap (the
# outermost, which fit_rv calls and patch_rv extends), each still delegating
# to the next. The class aliases the later two files re-export, and the second
# copies of _sq_dist / _age_delta in 0905_abalone1_kernels.py (identical
# expressions), are not repeated.
# =============================================================================
AGE = 0          # the age column of the FeatureAdapter output


def _sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)


def _age_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a_i[age] - b_j[age]."""
    return a[:, AGE].unsqueeze(1) - b[:, AGE].unsqueeze(0)


class DerivKernel:
    """Interface: return the four blocks between A and B.

    `has_polynomial_factor` marks kernels that contain a Polynomial2 factor.  For
    those, the finite-difference convergence test is not applicable: Polynomial2 is
    a quadratic, so its third and higher derivatives vanish and the central second
    difference carries NO truncation term.  There is therefore no O(h^2) decay to
    observe, and shrinking h only raises the roundoff error 4*eps/h^2.  Only for
    these kernels is the roundoff-floor criterion allowed as a substitute; seeing
    the same pattern in any other kernel would indicate a genuine error.
    """

    has_polynomial_factor = False

    def blocks(self, a: torch.Tensor, b: torch.Tensor) -> dict:
        raise NotImplementedError


class RBFDeriv(DerivKernel):
    r"""k = exp(-r^2 / (2 l^2)).

        dk/da = -(D/l^2) k              with D = a[age] - b[age]
        dk/db = +(D/l^2) k
        d2k/da db = k (1/l^2 - D^2/l^4)

    No sqrt anywhere, so the r -> 0 limit is not special.
    """

    def __init__(self, module: gpytorch.kernels.RBFKernel):
        self.module = module

    def blocks(self, a, b):
        ls = self.module.lengthscale.reshape(())
        K = torch.exp(-0.5 * _sq_dist(a, b) / ls ** 2)
        D = _age_delta(a, b)
        return {"K": K, "D1": -(D / ls ** 2) * K, "D2": (D / ls ** 2) * K,
                "D12": K * (1.0 / ls ** 2 - D ** 2 / ls ** 4)}


class Matern15Deriv(DerivKernel):
    r"""Matern nu = 3/2:  k = (1 + a) exp(-a),  a = sqrt(3) r / l.

    dk/dr = -(3 r / l^2) exp(-a)                            [the (1+a) and 1 cancel]
    dr/d(x_age) = D / r,  dr/d(x'_age) = -D / r

        dk/da_age = dk/dr * D/r = -(3 D / l^2) exp(-a)      [r cancels: no singularity]
        dk/db_age = +(3 D / l^2) exp(-a)
        d2k/da db = d/da_age [ (3 D / l^2) exp(-a) ]
                  = (3/l^2) exp(-a) + (3 D / l^2) * (-sqrt(3) D /(l r)) exp(-a)
                  = (3/l^2) exp(-a) [ 1 - sqrt(3) D^2 / (l r) ]

    r -> 0 limit: |D| <= r so D^2/r -> 0, hence D12 -> 3/l^2 = -k''(0).
    Verified against a central finite difference of k: at h = 1e-5 the numerical
    k''(0) is -2.999964 versus the analytic -3.0 (l = 1).  The diagonal is filled
    with the limit explicitly because D^2/r is 0/0 there.
    """

    def __init__(self, module: gpytorch.kernels.MaternKernel):
        if abs(float(module.nu) - 1.5) > 1e-12:
            raise ValueError(f"Matern15Deriv requires nu=1.5, got {module.nu}")
        self.module = module

    def blocks(self, a, b):
        ls = self.module.lengthscale.reshape(())
        r2 = _sq_dist(a, b).clamp_min(0.0)
        r = torch.sqrt(r2.clamp_min(1e-300))
        near = r2 < 1e-24                       # coincident points
        D = _age_delta(a, b)
        e = torch.exp(-math.sqrt(3.0) * r / ls)
        K = (1.0 + math.sqrt(3.0) * r / ls) * e
        common = (3.0 / ls ** 2) * e
        ratio = torch.where(near, torch.zeros_like(r), D ** 2 / torch.where(near, torch.ones_like(r), r))
        return {"K": K, "D1": -(3.0 * D / ls ** 2) * e, "D2": (3.0 * D / ls ** 2) * e,
                "D12": common * (1.0 - math.sqrt(3.0) * ratio / ls)}


class Polynomial2Deriv(DerivKernel):
    has_polynomial_factor = True

    r"""k = (<a, b> + c)^2.

        dk/da_age = 2 (<a,b> + c) b_age
        dk/db_age = 2 (<a,b> + c) a_age
        d2k/da db = 2 a_age b_age + 2 (<a,b> + c)
    """

    def __init__(self, module: gpytorch.kernels.PolynomialKernel):
        if int(module.power) != 2:
            raise ValueError(f"Polynomial2Deriv requires power=2, got {module.power}")
        self.module = module

    def blocks(self, a, b):
        c = self.module.offset.reshape(())
        s = a @ b.T + c
        aa = a[:, AGE].unsqueeze(1)
        bb = b[:, AGE].unsqueeze(0)
        return {"K": s ** 2, "D1": 2.0 * s * bb, "D2": 2.0 * s * aa,
                "D12": 2.0 * aa * bb + 2.0 * s}


class ScaleDeriv(DerivKernel):
    @property
    def has_polynomial_factor(self):
        return self.base.has_polynomial_factor

    """outputscale * base — every block scales."""

    def __init__(self, module: gpytorch.kernels.ScaleKernel, base: DerivKernel):
        self.module, self.base = module, base

    def blocks(self, a, b):
        s = self.module.outputscale.reshape(())
        inner = self.base.blocks(a, b)
        return {k: s * v for k, v in inner.items()}


class SumDeriv(DerivKernel):
    @property
    def has_polynomial_factor(self):
        return any(p.has_polynomial_factor for p in self.parts)

    def __init__(self, parts: list[DerivKernel]):
        self.parts = parts

    def blocks(self, a, b):
        out = None
        for part in self.parts:
            block = part.blocks(a, b)
            out = block if out is None else {k: out[k] + block[k] for k in out}
        return out


class ProductDeriv(DerivKernel):
    @property
    def has_polynomial_factor(self):
        return any(p.has_polynomial_factor for p in self.parts)

    """Recursive product rule; handles three or more factors by folding."""

    def __init__(self, parts: list[DerivKernel]):
        self.parts = parts

    def blocks(self, a, b):
        out = self.parts[0].blocks(a, b)
        for part in self.parts[1:]:
            q = part.blocks(a, b)
            out = {
                "K": out["K"] * q["K"],
                "D1": out["D1"] * q["K"] + out["K"] * q["D1"],
                "D2": out["D2"] * q["K"] + out["K"] * q["D2"],
                "D12": (out["D12"] * q["K"] + out["D1"] * q["D2"]
                        + out["D2"] * q["D1"] + out["K"] * q["D12"]),
            }
        return out


def dk5_wrap(module) -> DerivKernel:
    """Map a gpytorch kernel tree onto the analytic derivative implementation."""
    if isinstance(module, gpytorch.kernels.ScaleKernel):
        return ScaleDeriv(module, dk5_wrap(module.base_kernel))
    if isinstance(module, gpytorch.kernels.AdditiveKernel):
        return SumDeriv([dk5_wrap(k) for k in module.kernels])
    if isinstance(module, gpytorch.kernels.ProductKernel):
        return ProductDeriv([dk5_wrap(k) for k in module.kernels])
    if isinstance(module, gpytorch.kernels.RBFKernel):
        return RBFDeriv(module)
    if isinstance(module, gpytorch.kernels.MaternKernel):
        return Matern15Deriv(module)
    if isinstance(module, gpytorch.kernels.PolynomialKernel):
        return Polynomial2Deriv(module)
    raise TypeError(f"no analytic derivative implementation for {type(module).__name__}")


def joint_covariance(kernel: DerivKernel, X: torch.Tensor, Xm: torch.Tensor,
                     jitter: float = 0.0) -> torch.Tensor:
    """[[K_ff, K_ff'], [K_f'f, K_f'f']] for latent values at X and derivatives at Xm."""
    ff = kernel.blocks(X, X)["K"]
    fm = kernel.blocks(X, Xm)["D2"]
    mm = kernel.blocks(Xm, Xm)["D12"]
    top = torch.cat([ff, fm], dim=1)
    bottom = torch.cat([fm.transpose(0, 1), mm], dim=1)
    joint = torch.cat([top, bottom], dim=0)
    if jitter:
        joint = joint + jitter * torch.eye(joint.shape[0], dtype=joint.dtype,
                                           device=joint.device)
    return joint


class LinearDeriv(DerivKernel):
    r"""k = v <a, b>.

        dk/da_age  = v b_age
        dk/db_age  = v a_age
        d2k/da db  = v            (a constant matrix)

    STRUCTURAL CONSEQUENCE, stated here because it drives an interpretation and a
    diagnostic in 0903_abalone8: a GP with this kernel is f(x) = w . x with
    w ~ N(0, v I), so df/dage = w_age is a SINGLE scalar random variable.  The
    cohort one-hot columns do not involve age, so it is the same variable for
    every cohort and every input.  Therefore

        K_f'f'  =  v * ones(m, m),   numeric rank 1 for any m.

    m virtual derivative points carry no more information than one, but EP treats
    them as m independent probit factors on that one latent, so the effective
    strength of the monotonicity prior would scale with m if the constraint were
    ever active.  0903_abalone8 records deriv_block_rank per fit and warns when
    rank << m; --collapse-degenerate-virtual reduces such kernels to one virtual
    point per cohort.
    """

    has_polynomial_factor = True     # kept for compatibility with the abalone5 API
    age_degree = 1

    def __init__(self, module: gpytorch.kernels.LinearKernel):
        self.module = module

    def blocks(self, a, b):
        v = self.module.variance.reshape(())
        ones = torch.ones(a.shape[0], b.shape[0], dtype=a.dtype, device=a.device)
        return {
            "K": v * (a @ b.T),
            "D1": v * b[:, AGE].unsqueeze(0) * ones,
            "D2": v * a[:, AGE].unsqueeze(1) * ones,
            "D12": v * ones,
        }


def dk8_wrap(module) -> DerivKernel:
    """Map a gpytorch kernel tree onto the analytic derivative implementation."""
    if isinstance(module, gpytorch.kernels.LinearKernel):
        return LinearDeriv(module)
    if isinstance(module, gpytorch.kernels.ScaleKernel):
        return ScaleDeriv(module, dk8_wrap(module.base_kernel))
    if isinstance(module, gpytorch.kernels.AdditiveKernel):
        return SumDeriv([dk8_wrap(k) for k in module.kernels])
    if isinstance(module, gpytorch.kernels.ProductKernel):
        return ProductDeriv([dk8_wrap(k) for k in module.kernels])
    return dk5_wrap(module)


def derivative_block_rank(kernel: DerivKernel, Xm: torch.Tensor,
                          tolerance: float = 1e-12) -> int:
    """Numeric rank of K_f'f'.  Linear gives 1 for any m; see LinearDeriv."""
    with torch.no_grad():
        block = kernel.blocks(Xm, Xm)["D12"]
        block = 0.5 * (block + block.transpose(0, 1))
        eigenvalues = torch.linalg.eigvalsh(block)
        largest = float(eigenvalues.max())
        if largest <= 0.0:
            return 0
        return int((eigenvalues > largest * tolerance).sum())


class _RadialDeriv(DerivKernel):
    r"""Blocks for any kernel written as k = f(s) with s = r^2.

        dk/da_age = 2 D f'(s)
        dk/db_age = -2 D f'(s)
        d2k/da_age db_age = -2 f'(s) - 4 D^2 f''(s)

    Checked against RBFDeriv: f = exp(-s/2l^2) gives k(1/l^2 - D^2/l^4), which is
    exactly the expression 0903_abalone5_kernels uses.
    """

    def _f_derivatives(self, s: torch.Tensor):
        raise NotImplementedError

    def blocks(self, a, b):
        s = _sq_dist(a, b)
        delta = _age_delta(a, b)
        value, first, second = self._f_derivatives(s)
        return {"K": value, "D1": 2.0 * delta * first, "D2": -2.0 * delta * first,
                "D12": -2.0 * first - 4.0 * delta ** 2 * second}


class Matern25Deriv(_RadialDeriv):
    r"""k = (1 + t + t^2/3) e^{-t},  t = sqrt(5) r / l.

        f'(s)  = -(5 / 6l^2) (1 + t) e^{-t}
        f''(s) =  (25 / 12l^4) e^{-t}

    so d2k/da db at r = 0 is 5/(3 l^2), finite: nu = 2.5 > 1 means the process is
    mean-square differentiable and the derivative process exists.
    """

    def __init__(self, module: gpytorch.kernels.MaternKernel):
        if abs(float(module.nu) - 2.5) > 1e-12:
            raise ValueError(f"Matern25Deriv requires nu=2.5, got {module.nu}")
        self.module = module

    def _f_derivatives(self, s):
        lengthscale = self.module.lengthscale.reshape(())
        t = math.sqrt(5.0) * s.clamp_min(0.0).sqrt() / lengthscale
        decay = torch.exp(-t)
        value = (1.0 + t + t ** 2 / 3.0) * decay
        first = -(5.0 / (6.0 * lengthscale ** 2)) * (1.0 + t) * decay
        second = (25.0 / (12.0 * lengthscale ** 4)) * decay
        return value, first, second


class RQDeriv(_RadialDeriv):
    r"""k = u^{-alpha},  u = 1 + s / (2 alpha l^2).

        f'(s)  = -u^{-alpha-1} / (2 l^2)
        f''(s) =  (alpha+1) u^{-alpha-2} / (4 alpha l^4)

    A large fitted alpha means the RQ has degenerated to an RBF; the fitted value
    is stored in the diagnostics for exactly that reason.
    """

    def __init__(self, module: gpytorch.kernels.RQKernel):
        self.module = module

    def _f_derivatives(self, s):
        lengthscale = self.module.lengthscale.reshape(())
        alpha = self.module.alpha.reshape(())
        u = 1.0 + s / (2.0 * alpha * lengthscale ** 2)
        value = u ** (-alpha)
        first = -(u ** (-alpha - 1.0)) / (2.0 * lengthscale ** 2)
        second = (alpha + 1.0) * (u ** (-alpha - 2.0)) / (4.0 * alpha * lengthscale ** 4)
        return value, first, second


class NoDerivativeProcess(Exception):
    """Raised when a kernel has no derivative process (Matern0.5)."""


def wrap(module) -> DerivKernel:
    if isinstance(module, gpytorch.kernels.MaternKernel):
        nu = float(module.nu)
        if abs(nu - 0.5) < 1e-12:
            raise NoDerivativeProcess(
                "Matern0.5 (Ornstein-Uhlenbeck) has nowhere-differentiable sample "
                "paths: d2k/da db is -exp(-|D|/l)/l^2 off the diagonal and diverges "
                "on it, so no derivative process exists to condition on.")
        if abs(nu - 2.5) < 1e-12:
            return Matern25Deriv(module)
        return Matern15Deriv(module)
    if isinstance(module, gpytorch.kernels.RQKernel):
        return RQDeriv(module)
    if isinstance(module, gpytorch.kernels.LinearKernel):
        return LinearDeriv(module)
    if isinstance(module, gpytorch.kernels.ScaleKernel):
        return ScaleDeriv(module, wrap(module.base_kernel))
    if isinstance(module, gpytorch.kernels.AdditiveKernel):
        return SumDeriv([wrap(k) for k in module.kernels])
    if isinstance(module, gpytorch.kernels.ProductKernel):
        return ProductDeriv([wrap(k) for k in module.kernels])
    return dk8_wrap(module)


# =============================================================================
# Derivative blocks for RBF_add3 and NNKernel; age-only inputs
#
# Carried over from 0914_abalone1_kernels.py without change. Its functions
# take the kernel registry and the derivative module as arguments; the package
# wiring passes this module for both, where 0914_abalone1 passed its modules.
# =============================================================================
def _dims_of(module) -> list[int] | None:
    dims = getattr(module, "active_dims", None)
    if dims is None:
        return None
    return [int(v) for v in torch.as_tensor(dims).reshape(-1).tolist()]


class GroupRBFDeriv:
    r"""exp(-|a_g - b_g|^2 / (2 l^2)) on the component's active dimensions g.

    If age is in g:  D1 = -(D/l^2) K,  D2 = +(D/l^2) K,  D12 = K (1/l^2 - D^2/l^4)
    otherwise the component does not depend on age and all derivative blocks are 0.
    """

    has_polynomial_factor = False

    def __init__(self, module: gpytorch.kernels.RBFKernel):
        self.module = module
        self.dims = _dims_of(module)

    def blocks(self, a, b):
        ls = self.module.lengthscale.reshape(())
        ag = a if self.dims is None else a[:, self.dims]
        bg = b if self.dims is None else b[:, self.dims]
        K = torch.exp(-0.5 * ((ag.unsqueeze(1) - bg.unsqueeze(0)) ** 2).sum(-1) / ls ** 2)
        if self.dims is not None and AGE not in self.dims:
            zero = torch.zeros_like(K)
            return {"K": K, "D1": zero, "D2": zero, "D12": zero}
        D = a[:, AGE].unsqueeze(1) - b[:, AGE].unsqueeze(0)
        return {"K": K, "D1": -(D / ls ** 2) * K, "D2": (D / ls ** 2) * K,
                "D12": K * (1.0 / ls ** 2 - D ** 2 / ls ** 4)}


class NNKernelDeriv:
    r"""k = (2/pi) asin(u),  u = N / sqrt(P_a P_b)

        N   = 2 (s_b + s_w <a, b>)
        P_a = 1 + 2 (s_b + s_w |a|^2),   P_b likewise

    With x_a = a[age], x_b = b[age] and S = sqrt(P_a P_b):

        du/da   = (2 s_w x_b - N 2 s_w x_a / P_a) / S
        du/db   = (2 s_w x_a - N 2 s_w x_b / P_b) / S
        d2u/dadb = ( 2 s_w (1 - 2 s_w x_b^2 / P_b)
                     - (2 s_w x_a / P_a)(2 s_w x_a - 2 s_w N x_b / P_b) ) / S

    and k' = (2/pi)(1-u^2)^(-1/2),  k'' = (2/pi) u (1-u^2)^(-3/2), so
        D1 = k' u_a,   D2 = k' u_b,   D12 = k'' u_a u_b + k' u_ab.

    u is clamped exactly as NeuralNetworkKernel.forward clamps it, so K agrees
    with the kernel to machine precision.  On a diagonal u = 2q/(1+2q) < 1
    strictly, so the clamp never binds where the derivatives are evaluated.
    """

    has_polynomial_factor = False

    def __init__(self, module, clamp: float):
        self.module = module
        self.clamp = clamp

    def blocks(self, a, b):
        sb = self.module.bias_variance.reshape(())
        sw = self.module.weight_variance.reshape(())
        Pa = (1.0 + 2.0 * (sb + sw * (a ** 2).sum(-1))).unsqueeze(1)
        Pb = (1.0 + 2.0 * (sb + sw * (b ** 2).sum(-1))).unsqueeze(0)
        N = 2.0 * (sb + sw * (a @ b.transpose(-2, -1)))
        S = torch.sqrt(Pa * Pb)
        u = (N / S).clamp(-1.0 + self.clamp, 1.0 - self.clamp)
        xa = a[:, AGE].unsqueeze(1)
        xb = b[:, AGE].unsqueeze(0)
        ua = (2.0 * sw * xb - N * 2.0 * sw * xa / Pa) / S
        ub = (2.0 * sw * xa - N * 2.0 * sw * xb / Pb) / S
        uab = (2.0 * sw * (1.0 - 2.0 * sw * xb ** 2 / Pb)
               - (2.0 * sw * xa / Pa) * (2.0 * sw * xa - 2.0 * sw * N * xb / Pb)) / S
        one_minus = 1.0 - u ** 2
        k1 = (2.0 / math.pi) / torch.sqrt(one_minus)
        k2 = (2.0 / math.pi) * u / one_minus ** 1.5
        return {"K": (2.0 / math.pi) * torch.asin(u), "D1": k1 * ua, "D2": k1 * ub,
                "D12": k2 * ua * ub + k1 * uab}


def patch_rv(DK, KX) -> None:
    """Route the two kernels to the blocks above; everything else unchanged.

    0905_abalone1.fit_rv calls ``DK.wrap`` through the module attribute, and
    DK.wrap's Scale / Additive / Product branches recurse through the SAME
    module-global name, so replacing the attribute reaches components nested at
    any depth -- including the RBF summands inside RBF_add3.
    """
    original = DK.wrap

    def wrap(module):
        if isinstance(module, gpytorch.kernels.RBFKernel) and _dims_of(module) is not None:
            return GroupRBFDeriv(module)
        if isinstance(module, KX.NeuralNetworkKernel):
            return NNKernelDeriv(module, KX.NN_CLAMP)
        return original(module)

    DK.wrap = wrap


def verify_blocks(DK, build, names, n_dims=9, seed=0, tolerance=1e-9) -> pd.DataFrame:
    """Analytic blocks against autograd, on random inputs with a real one-hot.

    Hyperparameters are moved off their initial values first, so a block that is
    right only at lengthscale 1 or unit variances cannot pass.
    """
    torch.manual_seed(seed)
    n_cont = n_dims - 4

    def sample(n):
        x = torch.randn(n, n_dims, dtype=torch.float64)
        x[:, n_cont:] = 0.0
        x[torch.arange(n), n_cont + torch.randint(0, 4, (n,))] = 1.0
        return x

    a, b = sample(6), sample(5)
    rows = []
    for name in names:
        module = build(name).double()
        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.37 * torch.randn_like(parameter))
        deriv = DK.wrap(module)
        with torch.no_grad():
            got = deriv.blocks(a, b)
        aa = a.clone().requires_grad_(True)
        bb = b.clone().requires_grad_(True)
        K = module(aa, bb).to_dense()
        dA = torch.zeros_like(K)
        dB = torch.zeros_like(K)
        dAB = torch.zeros_like(K)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                ga, gb = torch.autograd.grad(K[i, j], (aa, bb), create_graph=True)
                dA[i, j] = ga[i, AGE]
                dB[i, j] = gb[j, AGE]
                dAB[i, j] = torch.autograd.grad(ga[i, AGE], bb, retain_graph=True)[0][j, AGE]
        errors = {"K": float((got["K"] - K.detach()).abs().max()),
                  "D1": float((got["D1"] - dA.detach()).abs().max()),
                  "D2": float((got["D2"] - dB.detach()).abs().max()),
                  "D12": float((got["D12"] - dAB.detach()).abs().max())}
        with torch.no_grad():
            joint = DK.joint_covariance(deriv, a, b, 0.0) if hasattr(DK, "joint_covariance") \
                else None
        rows.append({"kernel": name, "wrapper": type(deriv).__name__,
                     **{f"err_{k}": v for k, v in errors.items()},
                     "scale_D12": float(dAB.abs().max()),
                     "ok": bool(max(errors.values()) < tolerance)})
    return pd.DataFrame(rows)


AGE_ONLY_KERNEL = "RBF_addAC"


class AgeOnlyAdapter:
    """[standardised age, cohort one-hot]: the nine-column adapter minus the four
    environmental columns.  Mirrors 0828_abalone5.FeatureAdapter otherwise --
    the scaler is fitted on the training rows, unknown cohorts raise."""

    def __init__(self, cohort_levels, age_col, cohort_col):
        from sklearn.preprocessing import StandardScaler
        self.cohort_levels = list(cohort_levels)
        self.age_col, self.cohort_col = age_col, cohort_col
        self.scaler = StandardScaler()

    def fit(self, frame: pd.DataFrame) -> "AgeOnlyAdapter":
        self.scaler.fit(frame[[self.age_col]].to_numpy(dtype=float))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        age = self.scaler.transform(frame[[self.age_col]].to_numpy(dtype=float))
        codes = pd.Categorical(frame[self.cohort_col].astype(str),
                               categories=self.cohort_levels).codes
        if np.any(codes < 0):
            raise ValueError("unknown cohort level in age-only transform")
        return np.concatenate([age, np.eye(len(self.cohort_levels))[codes]], axis=1)


def register_age_only(KX) -> None:
    """Add RBF_addAC = k_age + k_cohort to the kernel registry KX.patch reads.

    Same device as KX._additive: the age summand's outputscale is pinned at 1 so
    that, with the outer ScaleKernel, the two groups carry exactly two free
    scales.  Must run BEFORE KX.patch, whose path check builds every registered
    kernel through both backends.
    """
    def factory():
        age = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=[0]))
        age.initialize(outputscale=1.0)
        age.raw_outputscale.requires_grad_(False)
        return gpytorch.kernels.AdditiveKernel(
            age, gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=[1, 2, 3, 4])))

    KX.NEW_KERNELS[AGE_ONLY_KERNEL] = factory
    KX.KERNEL_GROUP[AGE_ONLY_KERNEL] = "axis-separable (age-only)"


# =============================================================================
# Riihimaki-Vehtari expectation propagation
#
# Carried over from 0903_abalone5.py.
# =============================================================================
EP_MAX_ITER = 100


EP_DAMPING = 0.5


EP_TOL = 1e-6


def _log_ndtr(z: torch.Tensor) -> torch.Tensor:
    return torch.special.log_ndtr(z)


def ep_posterior(K: torch.Tensor, n: int, y: torch.Tensor, noise: float,
                 nu_probit: float, damping: float = EP_DAMPING,
                 max_iter: int = EP_MAX_ITER, tol: float = EP_TOL) -> dict:
    """EP over [f (n exact Gaussian sites); f' (m probit sites)].

    The n observation sites are conjugate, so their site parameters are fixed at
    tau = 1/noise^2, nu = y/noise^2 and never iterated.  Only the m probit sites
    Phi(f'_i / nu_probit) are updated, with the moment matching of
    Rasmussen & Williams eq. (3.58) generalised from noise 1 to nu_probit.
    """
    total = K.shape[0]
    m = total - n
    dtype, device = K.dtype, K.device
    tau = torch.zeros(total, dtype=dtype, device=device)
    nu_site = torch.zeros(total, dtype=dtype, device=device)
    tau[:n] = 1.0 / noise ** 2
    nu_site[:n] = y / noise ** 2
    identity = torch.eye(total, dtype=dtype, device=device)

    def posterior(tau_vec, nu_vec):
        root = torch.sqrt(tau_vec.clamp_min(0.0))
        B = identity + root.unsqueeze(1) * K * root.unsqueeze(0)
        L = torch.linalg.cholesky(B)
        V = torch.linalg.solve_triangular(L, root.unsqueeze(1) * K, upper=False)
        Sigma = K - V.transpose(0, 1) @ V
        return Sigma, Sigma @ nu_vec, L

    Sigma, mu, L = posterior(tau, nu_site)
    converged, iterations, delta = False, 0, math.inf
    for iterations in range(1, max_iter + 1):
        previous = tau.clone()
        for i in range(n, total):
            var = Sigma[i, i]
            tau_cav = 1.0 / var - tau[i]
            nu_cav = mu[i] / var - nu_site[i]
            if tau_cav <= 1e-10:
                continue
            var_cav = 1.0 / tau_cav
            mu_cav = nu_cav / tau_cav
            denom = math.sqrt(nu_probit ** 2 + float(var_cav))
            z = mu_cav / denom
            log_phi = -0.5 * z ** 2 - 0.5 * math.log(2.0 * math.pi)
            ratio = torch.exp(log_phi - _log_ndtr(z))
            mu_hat = mu_cav + var_cav * ratio / denom
            var_hat = var_cav - var_cav ** 2 * ratio * (z + ratio) / denom ** 2
            var_hat = var_hat.clamp_min(1e-12)
            tau_new = (1.0 / var_hat - tau_cav).clamp_min(1e-10)
            nu_new = mu_hat / var_hat - nu_cav
            tau[i] = (1 - damping) * tau[i] + damping * tau_new
            nu_site[i] = (1 - damping) * nu_site[i] + damping * nu_new
        Sigma, mu, L = posterior(tau, nu_site)
        delta = float((tau - previous).abs().max())
        if delta < tol:
            converged = True
            break

    root = torch.sqrt(tau.clamp_min(0.0))
    # log Z_EP, Rasmussen & Williams (3.65) rearranged for numerical stability
    T = torch.diag(tau)
    A = K + torch.diag(1.0 / tau.clamp_min(1e-30))
    sign, logdet = torch.linalg.slogdet(A)
    quad = nu_site @ torch.linalg.solve(A, nu_site)
    log_z = -0.5 * logdet - 0.5 * quad - 0.5 * torch.log(tau.clamp_min(1e-30)).sum()
    return {"Sigma": Sigma, "mu": mu, "tau": tau, "nu_site": nu_site,
            "converged": converged, "iterations": iterations, "delta": delta,
            "log_z": float(log_z), "m": m}


def derivative_negativity(kernel, X, Xm, Xq, posterior, jitter=JITTER):
    """P(df/dage < 0) at query points Xq given the EP/Laplace posterior."""
    n = X.shape[0]
    joint = joint_covariance(kernel, X, Xm, jitter)
    cross = torch.cat([kernel.blocks(Xq, X)["D1"],
                       kernel.blocks(Xq, Xm)["D12"]], dim=1)
    prior_var = kernel.blocks(Xq, Xq)["D12"].diagonal()
    with torch.no_grad():
        solved = torch.linalg.solve(joint, cross.transpose(0, 1))
        mean = cross @ torch.linalg.solve(joint, posterior["mu"])
        var = (prior_var - (cross * solved.transpose(0, 1)).sum(dim=1)).clamp_min(1e-12)
        probability = torch.distributions.Normal(0.0, 1.0).cdf(-mean / var.sqrt())
    return mean.detach(), var.detach(), probability.detach()


def predict_monotone(kernel, X, Xm, post, noise, Xq):
    """EP/Laplace predictive: condition on the observations and the virtual sites."""
    with torch.no_grad():
        joint = joint_covariance(kernel, X, Xm, JITTER)
        cross = torch.cat([kernel.blocks(Xq, X)["K"], kernel.blocks(Xq, Xm)["D2"]], dim=1)
        prior = kernel.blocks(Xq, Xq)["K"].diagonal()
        if "tau" in post:
            site_var = 1.0 / post["tau"].clamp_min(1e-30)
            A = joint + torch.diag(site_var)
            site_mean = post["nu_site"] * site_var
            solved = torch.linalg.solve(A, site_mean)
            mean = cross @ solved
            var = (prior - (cross * torch.linalg.solve(A, cross.transpose(0, 1)
                                                       ).transpose(0, 1)).sum(dim=1))
        else:
            Sigma, mu = post["Sigma"], post["mu"]
            base = torch.linalg.solve(joint, mu)
            mean = cross @ base
            reduction = torch.linalg.solve(joint, cross.transpose(0, 1)).transpose(0, 1)
            var = prior - (cross * reduction).sum(dim=1) + (
                (reduction @ Sigma) * reduction).sum(dim=1)
        var = var.clamp_min(1e-12) + noise ** 2
    return mean, var


# =============================================================================
# Soft derivative-sign penalty
#
# Carried over from 0828_abalone5.py. Renamed: dense -> soft_dense, and its
# MONOTONE_TOL (1e-6, used only to count violations) -> SOFT_MONOTONE_TOL,
# because this file's MONOTONE_TOL is 1e-9.
# =============================================================================
LEARNING_RATE = 0.08


NUMERICAL_JITTER = 1e-6


DERIVATIVE_CONSTRAINT_WEIGHT = 0.75


DERIVATIVE_SIGN_TEMPERATURE = 0.15


DERIVATIVE_MARGIN = 0.0


SOFT_MONOTONE_TOL = 1e-6


def soft_dense(matrix: object) -> torch.Tensor:
    if hasattr(matrix, "to_dense"):
        return matrix.to_dense()
    if hasattr(matrix, "evaluate"):
        return matrix.evaluate()
    return matrix


def differentiable_posterior_mean(
    model: ExactDerivativeConstrainedGP,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
) -> torch.Tensor:
    """Exact GP posterior mean with gradients retained w.r.t. query inputs."""
    k_train = soft_dense(model.covar_module(train_x, train_x))
    noise = torch.clamp(likelihood.noise, min=NOISE_FLOOR)
    eye = torch.eye(train_x.shape[0], dtype=train_x.dtype, device=train_x.device)
    k_train = 0.5 * (k_train + k_train.transpose(-1, -2))
    k_train = k_train + (noise + NUMERICAL_JITTER) * eye
    centered_y = train_y - model.mean_module(train_x)
    alpha = torch.linalg.solve(k_train, centered_y.unsqueeze(-1)).squeeze(-1)
    k_query_train = soft_dense(model.covar_module(query_x, train_x))
    return model.mean_module(query_x) + k_query_train @ alpha


def posterior_age_derivative(
    model: ExactDerivativeConstrainedGP,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """Derivative of the exact GP posterior mean w.r.t. standardized age.

    Age is the first continuous feature. StandardScaler has a positive scale,
    therefore derivative sign is identical on standardized and original age axes.
    """
    if not query_x.requires_grad:
        query_x = query_x.detach().clone().requires_grad_(True)
    mean = differentiable_posterior_mean(
        model, likelihood, train_x, train_y, query_x
    )
    gradient = torch.autograd.grad(
        mean.sum(),
        query_x,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return gradient[:, 0]


def derivative_sign_constraint_loss(
    derivatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Virtual positive-derivative likelihood using a logistic sign model.

    This acts directly on df/d(age), unlike finite-difference curve penalties.
    softplus(-(d-margin)/temperature) is the negative log-likelihood for a
    virtual derivative-sign observation saying the derivative should be positive.
    """
    scaled = (derivatives - DERIVATIVE_MARGIN) / DERIVATIVE_SIGN_TEMPERATURE
    nll = torch.nn.functional.softplus(-scaled).mean()
    minimum = derivatives.min() if derivatives.numel() else derivatives.sum() * 0.0
    violations = int((derivatives.detach() < -SOFT_MONOTONE_TOL).sum().item())
    return nll, minimum, violations


# =============================================================================
# Soft-penalty and Riihimaki-Vehtari backends; the dispatching fit
#
# Carried over from 0905_abalone1.py. Its run_fit and make_context are
# run_fit_backend and make_context_backend here: this file's own run_fit and
# make_context (above) serve the 78 stacking fits and take narrower arguments.
# Its HYPER_ITER (400) is this file's HYPER_ITER_PLAIN.
# =============================================================================
JITTER_SOFT = 1e-6            # 0828_abalone5 convention


SOFT_ITER = 400


SOFT_GRID_SIZE = 12           # derivative-penalty grid points per cohort


DERIV_WEIGHT = DERIVATIVE_CONSTRAINT_WEIGHT


VIRTUAL_INITIAL = 12


VIRTUAL_MAX = 32


VIRTUAL_ROUNDS = 3


VIRTUAL_PROB_THRESHOLD = 0.05


# Wall-clock ceiling for the RV virtual-point refinement loop.  It is the only
# unbounded part of a fit: when the constraint stays active the loop keeps adding
# points up to VIRTUAL_MAX per cohort and re-runs EP on a matrix that grows with
# them.  Hitting the ceiling is recorded as refine_time_capped, never silent.
REFINE_TIME_BUDGET = 240.0


def soft_constraint_frame(source, train, test, cohorts):
    spans = horizon_ages(train, test, cohorts)
    ages = {c: np.linspace(low, high, SOFT_GRID_SIZE) for c, (low, _, high) in spans.items()}
    return age_frame(source, cohorts, ages)


def fit_soft(spec, X, y, adapter, constraint_frame, seed):
    module = build_kernel_module(spec)
    torch.manual_seed(int(seed))
    constraint_x = torch.tensor(adapter.transform(constraint_frame), dtype=torch.float64,
                                requires_grad=True)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(NOISE_FLOOR)).double()
    likelihood.initialize(noise=GLOBAL_NOISE_PRIOR_MEAN)
    model = ExactGP(X, y, likelihood, module).double()
    model.train()
    likelihood.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    best = {"value": math.inf, "step": 0}
    for step in range(SOFT_ITER):
        optimiser.zero_grad(set_to_none=True)
        constraint_x.grad = None
        with gpytorch.settings.cholesky_jitter(JITTER_SOFT):
            output = model(X)
            data_loss = -mll(output, y) - kernel_log_prior(module) / y.numel()
            derivatives = posterior_age_derivative(
                model, likelihood, X, y, constraint_x, create_graph=True)
            derivative_loss, minimum, violations = derivative_sign_constraint_loss(
                derivatives)
            loss = data_loss + DERIV_WEIGHT * derivative_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite soft-penalty loss for {spec}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        value = float(loss.detach())
        if value < best["value"]:
            best = {"value": value, "step": step,
                    "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "likelihood": {k: v.detach().clone()
                                   for k, v in likelihood.state_dict().items()}}
        optimiser.step()
    model.load_state_dict(best["model"])
    likelihood.load_state_dict(best["likelihood"])
    model.eval()
    likelihood.eval()
    noise = float(likelihood.noise.detach().reshape(-1)[0])
    with torch.enable_grad(), gpytorch.settings.cholesky_jitter(JITTER_SOFT):
        diagnostic_x = constraint_x.detach().clone().requires_grad_(True)
        derivatives = posterior_age_derivative(model, likelihood, X, y,
                                                    diagnostic_x, create_graph=False)
        _, minimum, violations = derivative_sign_constraint_loss(derivatives)
    record = {"log_marginal": -best["value"], "hyper_best_step": best["step"],
              "hyper_converged": best["step"] < SOFT_ITER - 1, "noise": noise,
              "constraint_grid_min_derivative": float(minimum.detach()),
              "constraint_grid_violations": int(violations),
              "n_constraint_points": int(len(constraint_frame))}
    record.update(hyperparameter_record(module, noise))

    def predict(frame):
        Xq = torch.tensor(adapter.transform(frame), dtype=torch.float64)
        mean, sd, cross = _gaussian_predict(module, X, y, noise, Xq, JITTER_SOFT)
        return Predictive(mean, sd), cross

    return record, {"kind": "soft", "module": module, "model": model,
                    "likelihood": likelihood, "X": X, "y": y, "noise": noise,
                    "adapter": adapter, "predict": predict, "jitter": JITTER_SOFT}


def virtual_frame(source, train, test, cohorts, per_cohort=VIRTUAL_INITIAL):
    spans = horizon_ages(train, test, cohorts)
    ages = {}
    for cohort, (low, _, high) in spans.items():
        base = np.linspace(low, high, per_cohort)
        forced = test.loc[test[COHORT_COL].astype(str).eq(cohort), AGE_COL].to_numpy(float)
        ages[cohort] = np.unique(np.concatenate([base, forced]))
    return age_frame(source, cohorts, ages)


def fit_rv(spec, X, y, adapter, source, train, test, cohorts, seed, nu_probit, dense):
    module = build_kernel_module(spec)
    hyper = optimise_marginal(module, X, y, seed, HYPER_ITER_PLAIN, JITTER_RV)
    kernel = wrap(module)                       # raises NoDerivativeProcess for Matern0.5
    virtual = virtual_frame(source, train, test, cohorts)
    Xq_dense = torch.tensor(adapter.transform(dense), dtype=torch.float64)
    history, capped, time_capped = [], False, False
    post = None
    refine_started = time.perf_counter()
    for round_index in range(VIRTUAL_ROUNDS + 1):
        Xm = torch.tensor(adapter.transform(virtual), dtype=torch.float64)
        with torch.no_grad():
            joint = joint_covariance(kernel, X, Xm, JITTER_RV)
            post = ep_posterior(joint, X.shape[0], y, hyper["noise"], nu_probit)
        _, _, probability = derivative_negativity(kernel, X, Xm, Xq_dense, post)
        probs = probability.detach().numpy()
        window = dense["window"].to_numpy()
        history.append({"round": round_index, "n_virtual": int(len(virtual)),
                        "max_prob_negative": float(probs.max()),
                        "max_prob_negative_train": float(probs[window == "train"].max()),
                        "max_prob_negative_test": float(probs[window == "test"].max()),
                        "ep_converged": bool(post["converged"]),
                        "ep_iterations": int(post["iterations"])})
        if round_index == VIRTUAL_ROUNDS or probs.max() <= VIRTUAL_PROB_THRESHOLD:
            break
        if time.perf_counter() - refine_started > REFINE_TIME_BUDGET:
            time_capped = True
            break
        additions = []
        for cohort in cohorts:
            mask = dense[COHORT_COL].astype(str).eq(cohort).to_numpy()
            existing = int((virtual[COHORT_COL].astype(str) == cohort).sum())
            room = VIRTUAL_MAX - existing
            if room <= 0:
                capped = True
                continue
            local = probs[mask]
            candidates = np.flatnonzero(local > VIRTUAL_PROB_THRESHOLD)
            if len(candidates) == 0:
                continue
            if len(candidates) > room:
                capped = True
            chosen = candidates[np.argsort(-local[candidates])[:room]]
            additions.append(dense.loc[mask].iloc[chosen])
        if not additions:
            break
        virtual = pd.concat([virtual, *additions], ignore_index=True)
    Xm = torch.tensor(adapter.transform(virtual), dtype=torch.float64)
    with torch.no_grad():
        joint = joint_covariance(kernel, X, Xm, JITTER_RV)
        eig = torch.linalg.eigvalsh(0.5 * (joint + joint.transpose(0, 1)))
        condition = float(eig.max() / max(float(eig.min()), 1e-300))
    rank = derivative_block_rank(kernel, Xm)
    last = history[-1]
    record = {
        "log_marginal": hyper["log_marginal"], "hyper_best_step": hyper["best_step"],
        "hyper_converged": hyper["converged"], "noise": hyper["noise"],
        "n_virtual": int(len(virtual)), "refine_rounds": len(history) - 1,
        "cap_reached": bool(capped),
        "cap_reached_with_violation": bool((capped or time_capped)
                                           and last["max_prob_negative"]
                                           > VIRTUAL_PROB_THRESHOLD),
        "refine_time_capped": bool(time_capped),
        "refine_seconds": float(time.perf_counter() - refine_started),
        "ep_converged": bool(post["converged"]), "ep_iterations": int(post["iterations"]),
        "ep_delta": float(post["delta"]), "nu_probit": float(nu_probit),
        "max_prob_negative_train": last["max_prob_negative_train"],
        "max_prob_negative_test": last["max_prob_negative_test"],
        "deriv_block_rank": int(rank),
        "deriv_rank_ratio": float(rank / max(len(virtual), 1)),
        "joint_condition_number": condition,
    }
    record.update(hyperparameter_record(module, hyper["noise"]))
    if not post["converged"]:
        record["fit_failed"] = True
        record["fit_failure_reason"] = (
            f"EP did not converge in {EP_MAX_ITER} iterations "
            f"(delta={float(post['delta']):.3e})")

    def predict(frame):
        Xq = torch.tensor(adapter.transform(frame), dtype=torch.float64)
        mean, var = predict_monotone(kernel, X, Xm, post, hyper["noise"], Xq)
        with torch.no_grad():
            cross = float(kernel.blocks(Xq, X)["K"].abs().max())
        return Predictive(mean.numpy(), var.clamp_min(1e-12).numpy() ** 0.5), cross

    return record, {"kind": "rv", "module": module, "kernel": kernel, "X": X, "y": y,
                    "Xm": Xm, "post": post, "noise": hyper["noise"], "adapter": adapter,
                    "predict": predict, "virtual": virtual, "jitter": JITTER_RV}


def run_fit_backend(item, ctx) -> dict:
    """One fit.  Any failure is recorded and returned; it never stops the sweep."""
    started = time.perf_counter()
    method = item["method"]
    spec = item["kernel"]
    monotone = bool(item["monotone"])
    response_mode = item["response"]
    position = item.get("position")
    extra_rows = item.get("extra_rows", 0)

    record = {k: v for k, v in item.items() if k != "state_key"}
    record.update({"stage_label": str(item["fit_key"]).split("|", 1)[0],
                   "monotone": monotone, "fit_failed": False, "fit_failure_reason": ""})
    try:
        train, test, cohorts = ctx["train"], ctx["test"], ctx["cohorts"]
        source, adapter = ctx["source"], ctx["adapter"]
        fit_rows = ctx["fit_rows"](position, extra_rows)
        response = Response(response_mode).fit(fit_rows["_w"].to_numpy(float))
        record.update(response.describe())
        if response.fit_failed:
            raise FloatingPointError(response.fit_failure_reason)
        z = response.to_z(fit_rows["_w"].to_numpy(float))
        fit_rows = fit_rows.copy()
        fit_rows["_z"] = z
        record["n_fit_rows"] = int(len(fit_rows))

        X = torch.tensor(adapter.transform(fit_rows), dtype=torch.float64)
        y = torch.tensor(z, dtype=torch.float64)
        seed = item["job_seed"]

        if not monotone:
            detail, state = fit_plain(spec, X, y, adapter, seed)
        elif method == "soft":
            detail, state = fit_soft(spec, X, y, adapter, ctx["soft_constraint"], seed)
        elif method == "rv":
            span = float(np.ptp(z)) / max(
                float(train[AGE_COL].max() - train[AGE_COL].min()), 1.0)
            detail, state = fit_rv(spec, X, y, adapter, source, train, test, cohorts,
                                   seed, 0.01 * max(span, 1e-6), ctx["dense"])
        elif method == "integral":
            grid, grid_frame = ctx["integral"]
            detail, state = fit_integral(spec, fit_rows, grid, grid_frame, adapter,
                                         cohorts, seed)
        else:
            raise ValueError(f"unknown method {method}")
        record.update(detail)
        record["response_obj"] = response
        record["state"] = state
    except NoDerivativeProcess as exc:
        record.update({"fit_failed": True, "fit_failure_reason": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:                                  # never stops the sweep
        record.update({"fit_failed": True,
                       "fit_failure_reason": f"{type(exc).__name__}: {exc}"})
    record["fit_seconds"] = float(time.perf_counter() - started)
    return record


def make_context_backend(train, test, cohorts, response_mode=None):
    source = input_source(train, test)
    age_domain = (float(train[AGE_COL].min()), float(train[AGE_COL].max()))
    adapter = FeatureAdapter(cohorts, age_domain).fit(train)
    dense = scan_frame(source, train, test, cohorts)
    soft_constraint = soft_constraint_frame(source, train, test, cohorts)
    grid, grid_frame = integral_context(source, train, test, cohorts, adapter)
    horizon1 = test.loc[test["horizon"].eq(1)].copy()

    def fit_rows(position, extra_rows):
        rows = train if extra_rows == 0 else pd.concat([train, horizon1],
                                                       ignore_index=True, sort=False)
        if position is not None:
            rows = rows.drop(index=position)
        return rows.reset_index(drop=True)

    return {"train": train, "test": test, "cohorts": cohorts, "source": source,
            "adapter": adapter, "dense": dense, "soft_constraint": soft_constraint,
            "integral": (grid, grid_frame), "fit_rows": fit_rows,
            "age_domain": age_domain}


# =============================================================================
# von Bertalanffy and Gompertz comparators
#
# Carried over from 0915_abalone1.py. Renamed: its MONOTONE_TOL (1e-12) ->
# GROWTH_MONOTONE_TOL; show / banner / out / OUT / PREFIX -> growth_*.
# =============================================================================
# ---- instability flags, fixed before looking at any fit ----------------------
BOUND_EPS = 1e-10              # positivity bounds sit this far above zero


AT_BOUND_REL = 1e-6            # an estimate this close to a bound is "on" it


MAX_REL_SE = 1.0               # SE larger than the estimate itself


MAX_COND = 1e10                # scaled J'J this ill-conditioned: SEs are not usable.
                               # Scaled = Jacobian columns multiplied by |parameter|,
                               # i.e. w.r.t. log|parameter|.  The raw J'J condition
                               # number mostly measures the spread in parameter
                               # units (t0 ~ 50 days, k ~ 0.01/day) -- it is ~1e9 for
                               # every facility -- so it is reported but not used.
                               # Switched to the scaled form after the first
                               # validation run; both are written to 01_parameters.


MAX_WINF_RATIO = 10.0          # asymptote beyond 10x the largest training weight


NEAR_OPTIMAL = 0.01            # multistart solutions within 1% of the best SSE ...


MAX_WINF_SPREAD = 2.0          # ... whose W_inf differ by more than 2x


DENSE_POINTS = 500


GROWTH_MONOTONE_TOL = 1e-12


def vb_curve(p, t):
    W, k, t0 = p
    return W * (1.0 - np.exp(-k * (t - t0))) ** 3


def vb_jac(p, t):
    W, k, t0 = p
    e = np.exp(-k * (t - t0))
    u = 1.0 - e
    return np.column_stack([u ** 3, 3.0 * W * u ** 2 * (t - t0) * e,
                            -3.0 * W * u ** 2 * k * e])


def gz_curve(p, t):
    W, b, k = p
    return W * np.exp(-b * np.exp(-k * t))


def gz_jac(p, t):
    W, b, k = p
    E = np.exp(-k * t)
    base = np.exp(-b * E)
    return np.column_stack([base, -W * E * base, W * base * b * t * E])


# Monotonicity and positivity are DIFFERENT conditions and are checked separately.
#   vB:  dw/dt = 3 W k u^2 exp(-k(t - t0)),  u = 1 - exp(-k(t - t0))
#        -> non-decreasing for every t iff W k > 0, whatever t0 is.
#        t0 < first age is what keeps u > 0, i.e. the predicted WEIGHT positive at
#        the observed ages; below t0 the cube of a negative u is a negative weight.
#   Gompertz:  dw/dt = W b k exp(-kt) exp(-b exp(-kt))
#        -> increasing iff W b k > 0;  positive iff W > 0.
def vb_monotone_signs(p, t_min) -> bool:
    W, k, t0 = p
    return bool(W * k > 0)


def vb_positive(p, t_min) -> bool:
    W, k, t0 = p
    return bool(W > 0 and k > 0 and t0 < t_min)


def gz_monotone_signs(p, t_min) -> bool:
    W, b, k = p
    return bool(W * b * k > 0)


def gz_positive(p, t_min) -> bool:
    return bool(p[0] > 0)


MODELS = {
    "vonBertalanffy": {"names": ["W_inf", "k", "t0"], "curve": vb_curve, "jac": vb_jac,
                       "signs": vb_monotone_signs, "positive": vb_positive,
                       "rule": "increasing iff W_inf k > 0",
                       "positivity_rule": "w > 0 at observed ages iff W_inf, k > 0 and "
                                          "t0 < first training age"},
    "Gompertz": {"names": ["W_inf", "b", "k"], "curve": gz_curve, "jac": gz_jac,
                 "signs": gz_monotone_signs, "positive": gz_positive,
                 "rule": "increasing iff W_inf b k > 0",
                 "positivity_rule": "w > 0 iff W_inf > 0"},
}


def bounds_for(model, age_min):
    if model == "vonBertalanffy":
        return (np.array([BOUND_EPS, BOUND_EPS, -np.inf]),
                np.array([np.inf, np.inf, age_min - 1e-6]))
    return np.array([BOUND_EPS, BOUND_EPS, BOUND_EPS]), np.array([np.inf] * 3)


def starts_for(model, age_min, w_max):
    """0902_abalone4's start first (mapped to grams: A = max(log w) + 1 means
    W_inf = e * max w), then a grid wide enough to expose multiple optima."""
    scales = [math.e, 1.2, 2.0, 5.0, 20.0]
    if model == "vonBertalanffy":
        first = [[math.e * w_max, 0.01, age_min - 60.0]]
        grid = itertools.product([s * w_max for s in scales], [0.002, 0.005, 0.01, 0.02, 0.05],
                                 [age_min - 150.0, age_min - 60.0, age_min - 20.0, age_min - 5.0])
    else:
        first = [[math.e * w_max, 20.0, 0.01]]
        grid = itertools.product([s * w_max for s in scales], [5.0, 20.0, 100.0, 500.0],
                                 [0.005, 0.01, 0.02, 0.05])
    seen, growth_out = set(), []
    for start in first + [list(g) for g in grid]:
        key = tuple(np.round(start, 10))
        if key not in seen:
            seen.add(key)
            growth_out.append(start)
    return growth_out


def fit_facility(model, age, w, age_test, age_dense):
    spec = MODELS[model]
    curve, jac = spec["curve"], spec["jac"]
    lower, upper = bounds_for(model, float(age.min()))

    def residual(p):
        return curve(p, age) - w

    def residual_jac(p):
        return jac(p, age)

    solutions = []
    for start in starts_for(model, float(age.min()), float(w.max())):
        x0 = np.clip(np.asarray(start, float), lower + 1e-9, upper - 1e-9)
        with np.errstate(all="ignore"):
            try:
                sol = least_squares(residual, x0, jac=residual_jac, bounds=(lower, upper),
                                    x_scale="jac", method="trf", max_nfev=20000)
            except Exception as exc:                        # recorded, never silent
                solutions.append({"start": list(map(float, x0)), "ok": False,
                                  "message": f"{type(exc).__name__}: {exc}"})
                continue
        sse = float(np.sum(sol.fun ** 2))
        solutions.append({"start": list(map(float, x0)), "ok": bool(sol.success) and np.isfinite(sse),
                          "status": int(sol.status), "message": str(sol.message),
                          "x": sol.x, "sse": sse, "nfev": int(sol.nfev), "jac": sol.jac})
    good = [s for s in solutions if s["ok"]]
    record = {"n_starts": len(solutions), "n_converged": len(good)}
    if not good:
        record.update({"converged": False, "message": solutions[0].get("message", "")})
        return record, None

    best = min(good, key=lambda s: s["sse"])
    p = best["x"]
    n, k_par = len(w), len(p)
    dof = n - k_par
    s2 = best["sse"] / dof
    J = residual_jac(p)
    JTJ = J.T @ J
    cond = float(np.linalg.cond(JTJ))
    scaled = J * np.maximum(np.abs(p), 1e-12)[None, :]
    cond_scaled = float(np.linalg.cond(scaled.T @ scaled))
    try:
        cov = s2 * np.linalg.inv(JTJ) if np.isfinite(cond) and cond < 1e16 else None
    except np.linalg.LinAlgError:
        cov = None
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None)) if cov is not None else np.full(k_par, np.nan)
    at_bound = [bool(np.isfinite(lo) and abs(v - lo) <= AT_BOUND_REL * max(1.0, abs(lo)))
                or bool(np.isfinite(hi) and abs(v - hi) <= AT_BOUND_REL * max(1.0, abs(hi)))
                for v, lo, hi in zip(p, lower, upper)]
    near = [s for s in good if s["sse"] <= best["sse"] * (1.0 + NEAR_OPTIMAL) + 1e-15]
    w_near = np.array([s["x"][0] for s in near])
    corr_wk = (float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]))
               if cov is not None and cov[0, 0] > 0 and cov[1, 1] > 0 else np.nan)

    record.update({
        "converged": True, "status": best["status"], "message": best["message"],
        "nfev": best["nfev"], "sse": best["sse"], "dof": dof, "sigma_g": math.sqrt(s2),
        "cond_JTJ": cond, "cond_JTJ_scaled": cond_scaled, "corr_Winf_2nd": corr_wk,
        "first_train_age": float(age.min()),
        "n_near_optimal": len(near),
        "near_optimal_Winf_min": float(w_near.min()), "near_optimal_Winf_max": float(w_near.max()),
        "Winf_over_max_train": float(p[0] / w.max()),
    })
    for name, value, error, bound in zip(spec["names"], p, se, at_bound):
        record[name] = float(value)
        record[f"{name}_se"] = float(error)
        record[f"{name}_rel_se"] = float(error / abs(value)) if value != 0 else np.nan
        record[f"{name}_at_bound"] = bound

    # Unconstrained diagnostic from the same start as the bounded optimum
    with np.errstate(all="ignore"):
        try:
            free = least_squares(residual, np.asarray(best["start"], float), jac=residual_jac,
                                 x_scale="jac", method="trf", max_nfev=20000)
            free_p, free_sse, free_ok = free.x, float(np.sum(free.fun ** 2)), bool(free.success)
        except Exception as exc:
            free_p, free_sse, free_ok = np.full(k_par, np.nan), np.nan, False
    finite = bool(np.all(np.isfinite(free_p)))
    record.update({"free_converged": free_ok, "free_sse": free_sse,
                   "free_monotone_signs_ok": bool(spec["signs"](free_p, float(age.min())))
                   if finite else False,
                   "free_positive_ok": bool(spec["positive"](free_p, float(age.min())))
                   if finite else False})
    for name, value in zip(spec["names"], free_p):
        record[f"free_{name}"] = float(value)

    predictions = {"test": curve(p, age_test), "dense": curve(p, age_dense),
                   "free_test": curve(free_p, age_test) if np.all(np.isfinite(free_p)) else
                   np.full(len(age_test), np.nan),
                   "free_dense": curve(free_p, age_dense) if np.all(np.isfinite(free_p)) else
                   np.full(len(age_dense), np.nan)}
    record["monotone_signs_ok"] = bool(spec["signs"](p, float(age.min())))
    record["positive_ok"] = bool(spec["positive"](p, float(age.min())))
    return record, predictions


def instability(record) -> list[str]:
    """Human-readable reasons, with numbers, for a table footnote."""
    if not record.get("converged"):
        return ["수렴 실패"]
    reasons = []
    names = MODELS[record["model"]]["names"]
    for name in names:
        if record[f"{name}_at_bound"]:
            text = f"{name} 가 경계에 걸림 ({record[name]:.4g})"
            if record["model"] == "vonBertalanffy" and name == "t0" \
                    and np.isfinite(record.get("free_t0", np.nan)):
                text += (f" — 경계 없이 적합하면 t0 = {record['free_t0']:.1f}일로 첫 train 일령 "
                         f"{record['first_train_age']:.0f}일보다 뒤이고, 그 사이 일령에서 무게를 "
                         f"최소 {record['free_dense_min_g']:.3g} g 로 예측 (SSE "
                         f"{record['sse']:.4g} → {record['free_sse']:.4g})")
            reasons.append(text)
    if not np.isfinite(record["W_inf_se"]) or record["W_inf_rel_se"] > MAX_REL_SE:
        reasons.append(f"W_inf 상대 SE {record['W_inf_rel_se']:.2f} "
                       f"(W_inf {record['W_inf']:.3g} ± {record['W_inf_se']:.3g} g)")
    if record["cond_JTJ_scaled"] > MAX_COND:
        reasons.append(f"척도 정규화 J'J 조건수 {record['cond_JTJ_scaled']:.2e}")
    if record["Winf_over_max_train"] > MAX_WINF_RATIO:
        reasons.append(f"W_inf 가 train 최대 무게의 {record['Winf_over_max_train']:.1f}배")
    spread = record["near_optimal_Winf_max"] / record["near_optimal_Winf_min"]
    if spread > MAX_WINF_SPREAD:
        reasons.append(f"SSE 1% 이내 해 {record['n_near_optimal']}개의 W_inf 가 "
                       f"{record['near_optimal_Winf_min']:.3g}~{record['near_optimal_Winf_max']:.3g} g "
                       f"({spread:.1f}배)")
    return reasons


def count_decreasing(values) -> int:
    values = np.asarray(values, float)
    if not np.all(np.isfinite(values)):
        return -1
    steps = np.diff(values)
    return int(np.sum(steps < -GROWTH_MONOTONE_TOL * max(1.0, float(np.abs(values).max()))))


def metrics(truth, prediction, reference) -> dict:
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if not np.all(np.isfinite(prediction)):
        return {"g_RMSE": np.nan, "g_MAE": np.nan, "g_MAPE": np.nan, "g_R2_pred": np.nan}
    sse = float(np.sum((prediction - truth) ** 2))
    return {"g_RMSE": float(np.sqrt(np.mean((prediction - truth) ** 2))),
            "g_MAE": float(np.mean(np.abs(prediction - truth))),
            "g_MAPE": float(np.mean(np.abs(prediction - truth) / truth) * 100.0),
            "g_R2_pred": float(1.0 - sse / np.sum((truth - reference) ** 2))}


def growth_show(frame, columns=None, digits=4):
    view = frame if columns is None else frame[columns]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.{digits}g}"))


def growth_banner(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100, flush=True)


def growth_out(number, name):
    return GROWTH_OUT / f"{GROWTH_PREFIX}_{number}_{name}.csv"


def growth_curve_run() -> dict:
    """The von Bertalanffy / Gompertz computation of 0915_abalone1.main.

    The two statement runs below are that function's lines 375-470 and
    496-509, carried over unchanged (renames only).  The lines between them
    read the stacking and ML rows from other scripts' CSVs; here those rows come
    from this file, so they are not repeated.  Only `rows = []` and the return
    are new.
    """
    data, train, test, cohorts = prepare()
    target = ordered_test(test, cohorts)
    reference = float(train["_w"].mean())
    print(f"\ntrain {len(train)}행 / test {len(target)}행, R2_pred 기준 train 평균 {reference:.4f} g")

    records, prediction_rows, dense_rows = [], [], []
    predictions = {m: np.full(len(target), np.nan) for m in MODELS}
    free_predictions = {m: np.full(len(target), np.nan) for m in MODELS}
    for cohort in cohorts:
        tr = train[train[COHORT_COL].astype(str).eq(cohort)].sort_values(AGE_COL)
        rows = np.flatnonzero(target[COHORT_COL].astype(str).to_numpy() == cohort)
        age, w = tr[AGE_COL].to_numpy(float), tr["_w"].to_numpy(float)
        age_test = target[AGE_COL].to_numpy(float)[rows]
        age_dense = np.linspace(age.min(), max(age.max(), age_test.max()), DENSE_POINTS)
        boundary = float(age.max())
        for model in MODELS:
            record, pred = fit_facility(model, age, w, age_test, age_dense)
            record.update({"model": model, "facility": ROMAN.get(cohort, cohort),
                           "n_train": len(w), "monotone_rule": MODELS[model]["rule"],
                           "positivity_rule": MODELS[model]["positivity_rule"]})
            if pred is not None:
                predictions[model][rows] = pred["test"]
                free_predictions[model][rows] = pred["free_test"]
                train_mask, test_mask = age_dense <= boundary, age_dense >= boundary
                record.update({
                    "dense_down_train": count_decreasing(pred["dense"][train_mask]),
                    "dense_down_test": count_decreasing(pred["dense"][test_mask]),
                    "free_dense_down_train": count_decreasing(pred["free_dense"][train_mask]),
                    "free_dense_down_test": count_decreasing(pred["free_dense"][test_mask]),
                    "dense_min_g": float(np.min(pred["dense"])),
                    "free_dense_min_g": float(np.min(pred["free_dense"]))
                    if np.all(np.isfinite(pred["free_dense"])) else np.nan,
                    "test_increasing": bool(np.all(np.diff(pred["test"]) > 0)),
                    "free_test_increasing": bool(np.all(np.diff(pred["free_test"]) > 0))
                    if np.all(np.isfinite(pred["free_test"])) else False})
                for i, r in enumerate(rows):
                    prediction_rows.append({
                        "model": model, "facility": ROMAN.get(cohort, cohort),
                        "occasion": f"t{6 + int(target['test_horizon'].iloc[r])}",
                        "age": float(age_test[i]), "observed_g": float(target["_w"].iloc[r]),
                        "predicted_g": float(pred["test"][i]),
                        "predicted_g_unconstrained": float(pred["free_test"][i])})
            record["instability"] = " / ".join(instability(record)) or ""
            records.append(record)

    params = pd.DataFrame(records)
    params.to_csv(growth_out("01", "parameters"), index=False, encoding=ENC)
    pd.DataFrame(prediction_rows).to_csv(growth_out("04", "test_predictions"), index=False, encoding=ENC)

    growth_banner("(a) 수렴")
    growth_show(params, ["model", "facility", "converged", "n_converged", "n_starts", "status",
                  "nfev", "sse", "sigma_g"])

    growth_banner("(b) 파라미터 추정치와 표준오차 (자유도 = 6 − 3 = 3)")
    for model, spec in MODELS.items():
        block = params[params.model == model]
        columns = ["facility"]
        for name in spec["names"]:
            columns += [name, f"{name}_se", f"{name}_rel_se", f"{name}_at_bound"]
        print(f"\n--- {model} ---")
        growth_show(block, columns)
        growth_show(block, ["facility", "cond_JTJ", "cond_JTJ_scaled", "corr_Winf_2nd",
                     "Winf_over_max_train",
                     "n_near_optimal", "near_optimal_Winf_min", "near_optimal_Winf_max"])

    growth_banner("(c) 단조성과 양수 조건 — 경계 적합(본 결과)과 경계 없는 진단 적합")
    for model, spec in MODELS.items():
        print(f"  {model}: 단조 {spec['rule']};  양수 {spec['positivity_rule']}")
    print("  경계 적합은 두 조건을 경계로 강제한다. 경계 없는 적합은 같은 시작점에서 조건 없이 "
          "풀어, 자료가 그 조건을 스스로 지지하는지 본다. 감소 스텝은 첫 train 일령부터 마지막 "
          f"test 일령까지 {DENSE_POINTS}점 격자에서 센다.")
    growth_show(params, ["model", "facility", "monotone_signs_ok", "positive_ok", "test_increasing",
                  "dense_down_train", "dense_down_test", "dense_min_g"])
    print()
    growth_show(params, ["model", "facility", "free_converged", "free_monotone_signs_ok",
                  "free_positive_ok", "free_test_increasing", "free_dense_down_train",
                  "free_dense_down_test", "free_dense_min_g", "sse", "free_sse"])
    free_cols = [c for c in params.columns if c.startswith("free_") and c.split("free_")[1]
                 in sum((spec["names"] for spec in MODELS.values()), [])]
    params[["model", "facility", *free_cols]].to_csv(growth_out("02", "unconstrained_diagnostic"),
                                                     index=False, encoding=ENC)
    params[["model", "facility", "monotone_signs_ok", "positive_ok", "test_increasing",
            "dense_down_train", "dense_down_test", "dense_min_g", "free_monotone_signs_ok",
            "free_positive_ok", "free_test_increasing", "free_dense_down_train",
            "free_dense_down_test", "free_dense_min_g"]].to_csv(
        growth_out("03", "monotonicity"), index=False, encoding=ENC)

    truth = target["_w"].to_numpy(float)
    by_facility = []
    for model in MODELS:
        for cohort in cohorts:
            rows = np.flatnonzero(target[COHORT_COL].astype(str).to_numpy() == cohort)
            m = metrics(truth[rows], predictions[model][rows], reference)
            by_facility.append({"model": model, "facility": ROMAN.get(cohort, cohort),
                                "g_MAE": m["g_MAE"], "g_MAPE": m["g_MAPE"]})
    pd.DataFrame(by_facility).to_csv(growth_out("05", "metrics_by_facility"), index=False, encoding=ENC)
    rows = []
    marks = iter("abcdefghijklmnopqrstuvwxyz")
    footnotes = []
    for model, label in (("vonBertalanffy", "von Bertalanffy"), ("Gompertz", "Gompertz")):
        block = params[params.model == model]
        unstable = block[block.instability != ""]
        mark = ""
        if len(unstable):
            mark = next(marks)
            for _, r in unstable.iterrows():
                footnotes.append({"mark": mark, "model": label, "facility": r.facility,
                                  "reason": r.instability})
        rows.append({"model": label, "inputs": "age", "fitted": "per facility (6 points, 3 params)",
                     "source": GROWTH_PREFIX, **metrics(truth, predictions[model], reference),
                     "footnote": mark})
    return {"params": params, "predictions": predictions, "truth": truth,
            "reference": reference, "target": target, "table": pd.DataFrame(rows),
            "footnotes": pd.DataFrame(footnotes,
                                      columns=["mark", "model", "facility", "reason"]),
            "by_facility": pd.DataFrame(by_facility)}


# =============================================================================
# Ablations E1 (age-only inputs), E2 (equal weights), E3 (how monotonicity is imposed)
#
# Carried over from 0914_abalone1.py. Renamed to keep clear of this file's own
# names: build_context, grid_frame, stage_fits, Store, out, banner, show, OUT,
# CHECK, PREFIX -> build_ablation_context, ablation_grid_frame,
# stage_ablation_fits, AblationStore, ablation_out, ablation_banner,
# ablation_show, ABLATION_OUT, ABLATION_CHECK, ABLATION_PREFIX. Module
# references become this module's names: A1.run_fit / A1.make_context ->
# run_fit_backend / make_context_backend, A3.HYPER_ITER ->
# HYPER_ITER_INTEGRAL, A1.HYPER_ITER -> HYPER_ITER_PLAIN, any other A1.x -> x.
# In table_confounders the local aliases SOFT = A1.SOFT and RV = A1.RV point
# at this module (sys.modules[__name__]), which now holds the 0828_abalone5
# and 0903_abalone5 constants they read. load_previous (package wiring) serves
# the checkpoint keys these functions look up. Declared changes:
# point_metrics' point prediction is from_z(point_z(.)) -- the predictive median
# under the default POINT_SUMMARY, the same rule as metrics_of above -- and
# REFERENCE holds the published values per summary. arm_curve and table_e3 audit
# the point-prediction curve for the same reason, so the monotonicity scan and
# the reported metrics describe one curve. build_stack scores every LOO row
# against the LOO mixture median with the arm's own full-LOO stacking weights
# (loo_mixture_summary), and check_reference records a gap from the published
# values instead of stopping.
# =============================================================================
AGE_SPEC = {"Matern0.5": "Matern0.5", "RBF_add3": AGE_ONLY_KERNEL,
            "NNKernel": "NNKernel"}


RV_KERNELS = ["RBF_add3", "NNKernel"]


COMMON = ["RBF_add3", "NNKernel"]


RV_EXCLUDED = ("Matern0.5 is excluded from the Riihimaki-Vehtari arm: its sample paths "
               "are nowhere differentiable, so no derivative process exists and "
               "0905_abalone1_kernels.wrap raises NoDerivativeProcess (log E20).")


ARMS = {"a": "integral (structural)", "b": "soft penalty",
        "c": "Riihimaki-Vehtari (EP)", "d": "unconstrained GP"}


REPRO_TOL = 1e-6              # log E26: argmax ties leave ~1e-8 across processes


MC_NOISE_NOTE = ("짝지은 시드: age-only 적합은 같은 커널·같은 LOO 행의 전체 입력 적합과 "
                 "같은 시드를 쓴다. 두 판이 같은 표준정규 난수열을 공유하므로 서로 다른 "
                 "시드에서 weight 를 0.06~0.12 움직였던 MC 잡음(log E26)의 공통 부분이 "
                 "차이에서 상쇄된다. 난수가 서로 다른 사후분포를 거쳐 변환되므로 완전히 "
                 "없어지지는 않는다.")


# Measured 2026-09-15 before pairing was adopted: the same fit repeated under two
# different seeds.  Recorded here because it decides what pairing can achieve.
SEED_DEPENDENCE_NOTE = (
    "시드 의존성 (같은 적합을 서로 다른 두 시드로 반복한 사전 검증):\n"
    "  (a) integral, E1 age-only : 시드 의존. 하이퍼파라미터 적합은 비트 동일(log_marginal "
    "차이 0)이지만 예측분포를 만드는 MC draw 가 시드를 쓴다 (NNKernel: test mean 1.3e-2, "
    "LOO sd 3.9e-3 차이).\n"
    "  (b) soft, (c) RV, (d) 비제약 : 시드 무관. 난수를 소비하지 않아 다른 시드에서도 예측이 "
    "비트 동일하다 (soft·RV 는 NNKernel·RBF_add3, 비제약은 RBF_add3·Matern0.5 로 확인).\n"
    "  따라서 짝지은 시드가 MC 잡음을 상쇄하는 곳은 E1(적분 대 적분)뿐이다. E3 의 (b)(c)(d) "
    "는 시드를 받지만 쓰지 않으므로, (a) 와 나머지의 차이에는 (a) 쪽 MC 잡음이 남는다.")


# The published numbers the reference arm must reproduce before anything is
# compared against it: the stacking weights (Matern0.5, RBF_add3, NNKernel), the
# test MAPE, and the mean widths and coverage of the native and residual-
# calibrated 80% intervals.  Per point summary; only the median version is
# published.  The weights and the native width do not depend on the point
# summary.  nan / None leave an item unpinned.
REFERENCE_BY_SUMMARY = {
    "mean": {"weights": (0.1564713883590087, 0.4361206580807558, 0.4074079535602354),
             "g_MAPE": float("nan"),
             "cp_width_g": float("nan"),
             "native_width_g": 6.572218479278382,
             "cp_covered": None},
    "median": {"weights": (0.1564713883590087, 0.4361206580807558, 0.4074079535602354),
               "g_MAPE": 14.287492184557948,
               "cp_width_g": 4.824358844526678,
               "native_width_g": 6.572218479278382,
               "cp_covered": 7},
}
REFERENCE = REFERENCE_BY_SUMMARY[POINT_SUMMARY]


def ablation_out(number: str, name: str) -> Path:
    return ABLATION_OUT / f"{ABLATION_PREFIX}_{number}_{name}.csv"


def ablation_banner(text: str) -> None:
    print("\n" + "=" * 104 + f"\n{text}\n" + "=" * 104, flush=True)


def ablation_show(frame, columns=None, digits=4) -> None:
    view = frame if columns is None else frame[columns]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.{digits}f}"))


class AblationStore:
    """Append-and-flush JSONL.  A failed fit is written but does not count as done."""

    def __init__(self, path: Path, append: bool = True):
        self.path, self.records = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["fit_key"]] = record
        self.handle = path.open("a", encoding="utf-8") if append else None

    def __contains__(self, key):
        return key in self.records and not self.records[key].get("fit_failed")

    def add(self, record):
        self.records[record["fit_key"]] = record
        self.handle.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def anchor_seed(previous: dict, kernel: str, role: str, position) -> int:
    """Paired seeds: every arm uses the seed of the matching (a) fit.

    The anchor is the published integral fit with the same kernel and the same
    held-out row (or the final fit), read from its checkpoint.  E1's age-only fits
    and E3's (b)(c)(d) fits all take it, so a difference between arms is never a
    difference between Monte Carlo realisations.
    """
    key = (f"loo|integral|{RESPONSE}|{kernel}|{position}" if role == "loo"
           else f"final|integral|{RESPONSE}|{kernel}|step0")
    return source_seed(previous, key)


def source_seed(previous: dict, key: str) -> int:
    record = previous.get(key)
    if record is None or "job_seed" not in record:
        raise KeyError(f"{key} 에 job_seed 가 없다 — 원본 시드를 복원할 수 없다.")
    return int(record["job_seed"])


def build_ablation_context():
    data, train, test, cohorts = prepare()
    ctx = make_context_backend(train, test, cohorts)
    # Only the adapter changes for the age-only arm: the integration grid, the
    # dense scan grid, the soft-penalty grid and fit_rows do not depend on it.
    age_ctx = dict(ctx)
    age_ctx["adapter"] = AgeOnlyAdapter(cohorts, AGE_COL, COHORT_COL).fit(train)
    target = ordered_test(test, cohorts)
    return {"ctx": ctx, "age_ctx": age_ctx, "train": train, "test": test,
            "cohorts": cohorts, "target": target,
            "response": Response(RESPONSE).fit(train["_w"].to_numpy(float)),
            "truth_g": target["_w"].to_numpy(float),
            "reference_g": float(train["_w"].mean()),
            "cohort_of_row": train[COHORT_COL].astype(str).to_numpy()}


def ablation_grid_frame(context) -> pd.DataFrame:
    """Integration nodes as prediction rows (the integral predict() accepts only these)."""
    ctx = context["ctx"]
    grid, source = ctx["integral"][0], ctx["source"]
    frame = pd.concat([age_frame(source, [c], {c: np.asarray(grid.nodes[c], float)})
                       for c in grid.cohorts], ignore_index=True)
    boundary = context["train"].groupby(COHORT_COL, observed=True)[AGE_COL].max().to_dict()
    frame["window"] = ["train" if row[AGE_COL] <= float(boundary[str(row[COHORT_COL])])
                       else "test" for _, row in frame.iterrows()]
    return frame


def job_list(context, previous) -> list[dict]:
    n_rows = len(context["train"])
    jobs = []

    def arm(exp, kernel, spec, method, monotone):
        for position in range(n_rows):
            jobs.append({"exp": exp, "kernel": kernel, "spec": spec, "method": method,
                         "monotone": monotone, "position": position, "role": "loo"})
        jobs.append({"exp": exp, "kernel": kernel, "spec": spec, "method": method,
                     "monotone": monotone, "position": None, "role": "final"})

    for kernel in KERNELS:
        arm("age", kernel, AGE_SPEC[kernel], "integral", True)
    for kernel in KERNELS:
        arm("soft", kernel, kernel, "soft", True)
    for kernel in RV_KERNELS:
        arm("rv", kernel, kernel, "rv", True)
    for kernel in KERNELS:
        for position in range(n_rows):
            jobs.append({"exp": "plain", "kernel": kernel, "spec": kernel, "method": "shared",
                         "monotone": False, "position": position, "role": "loo"})
    for kernel in KERNELS:
        # Reproduces 0905_abalone3's screen fit, so it takes that fit's seed.
        jobs.append({"exp": "plain", "kernel": kernel, "spec": kernel, "method": "shared",
                     "monotone": False, "position": None, "role": "final",
                     "reproduce": f"screen|{RESPONSE}|{kernel}|step0"})

    for index, job in enumerate(jobs):
        job["job_index"] = index
        job["fit_key"] = f"{job['exp']}|{job['kernel']}|{job['role']}|{job['position']}"
        job["job_seed"] = anchor_seed(previous, job["kernel"], job["role"], job["position"])
    return jobs


def stage_ablation_fits(context, workers) -> None:
    previous = load_previous()
    jobs = job_list(context, previous)
    store = AblationStore(ABLATION_CHECK / f"{ABLATION_PREFIX}_fits.jsonl")
    # A checkpoint written under a different seed is not the fit this job asks
    # for; refit it rather than mix seeding schemes in one table.
    todo = [job for job in jobs if job["fit_key"] not in store
            or int(store.records[job["fit_key"]].get("job_seed", -1)) != job["job_seed"]
            or (job["role"] == "loo"
                and "loo_median_z" not in store.records[job["fit_key"]])]
    # An integral LOO record that lacks only loo_kind / loo_quantiles is not
    # replaced: it is refitted by the same worker (same seed, held-out row first)
    # and backfill_loo_record adds the two fields to the stored record.  Gaussian
    # arms need no table; loo_kind_of reads their kind from their method.
    refit = {job["fit_key"] for job in todo}
    backfill = [job for job in jobs if job["role"] == "loo" and job["fit_key"] not in refit
                and not has_loo_distribution(store.records[job["fit_key"]])
                and store.records[job["fit_key"]].get("method") not in GAUSSIAN_LOO_METHODS]
    counts = pd.Series([j["exp"] for j in todo]).value_counts().to_dict() if todo else {}
    ablation_banner(f"적합 — 전체 {len(jobs)}개 중 {len(todo)}개 남음 {counts}"
                    + (f", LOO 분포 보충 {len(backfill)}개" if backfill else ""))
    if not todo and not backfill:
        report_loo_backfill(store.records, "ablation")
        store.close()
        return
    train, target = context["train"], context["target"]
    dense = context["ctx"]["dense"]

    def worker(job):
        ctx = context["age_ctx"] if job["exp"] == "age" else context["ctx"]
        item = {"method": job["method"], "response": RESPONSE, "kernel": job["spec"],
                "position": job["position"], "monotone": job["monotone"], "extra_rows": 0,
                "step": 0, "fit_key": job["fit_key"], "job_seed": job["job_seed"],
                "job_index": job["job_index"]}
        record = run_fit_backend(item, ctx)
        state = record.pop("state", None)
        response = record.pop("response_obj", None)
        record.update({"exp": job["exp"], "label": job["kernel"], "role": job["role"]})
        if state is None or record["fit_failed"]:
            return strip(record)
        if job["role"] == "loo":
            # FIRST predict call is the held-out row, as in every original LOO
            # worker: the integral predictive draws its noise from a stream the
            # fit owns, so call order changes the MC sd (log E26).
            held = train.iloc[[int(job["position"])]]
            predictive, _ = state["predict"](held)
            weight = float(held["_w"].iloc[0])
            truth_z = float(response.to_z(np.array([weight]))[0])
            mean_z, sd_z = float(predictive.mean_z[0]), float(predictive.sd_z[0])
            lpd_z = float(norm.logpdf(truth_z, loc=mean_z, scale=sd_z))
            record.update({
                "truth_w": weight, "truth_z": truth_z, "loo_mean_z": mean_z,
                "loo_median_z": float(np.asarray(predictive.median_z(), float)[0]),
                "loo_sd_z": sd_z, "loo_lpd_z": lpd_z,
                "loo_lpd_g": lpd_z + float(response.log_dz_dw(np.array([weight]))[0])})
            record.update(loo_distribution_fields(predictive))
        else:
            predictive, _ = state["predict"](target)          # FIRST: the test rows
            record.update({"test_mean_z": [float(v) for v in predictive.mean_z],
                           "test_sd_z": [float(v) for v in predictive.sd_z],
                           "test_quantiles": quantile_table(predictive).tolist()
                           if predictive.kind == "draws" else None})
            if job["method"] != "integral":
                dense_predictive, _ = state["predict"](dense)
                record.update({"dense_mean_z": [float(v) for v in dense_predictive.mean_z],
                               "dense_sd_z": [float(v) for v in dense_predictive.sd_z]})
            if job.get("reproduce"):
                origin = previous[job["reproduce"]]
                gaps = {name: float(np.max(np.abs(np.asarray(origin[name], float)
                                                  - np.asarray(record[name], float))))
                        for name in ("test_mean_z", "test_sd_z")}
                states_key = f"plain|{RESPONSE}|{job['kernel']}|step0"
                if states_key in previous and "dense_mean_z" in previous[states_key]:
                    gaps["dense_mean_z"] = float(np.max(np.abs(
                        np.asarray(previous[states_key]["dense_mean_z"], float)
                        - np.asarray(record["dense_mean_z"], float))))
                if max(gaps.values()) > REPRO_TOL:
                    raise AssertionError(f"{job['fit_key']} 가 {job['reproduce']} 를 재현하지 "
                                         f"못했다: {gaps}")
                record["repro_gaps"] = gaps
                record["repro_source_seed"] = int(origin["job_seed"])
        return strip(record)

    started, done = time.perf_counter(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job): job for job in todo}
        for future in as_completed(futures):
            record = future.result()
            store.add(record)
            done += 1
            if done % 10 == 0 or done == len(todo):
                elapsed = time.perf_counter() - started
                print(f"    {done}/{len(todo)}  {elapsed:.0f} s  "
                      f"(eta {elapsed / done * (len(todo) - done):.0f} s)", flush=True)
    if backfill:
        print(f"  LOO 분포 보충 {len(backfill)}개 — 같은 시드·같은 호출 순서로 다시 적합, "
              f"저장값은 유지", flush=True)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, job): job for job in backfill}
            for done, future in enumerate(as_completed(futures), start=1):
                key = futures[future]["fit_key"]
                store.add(backfill_loo_record(store.records[key], future.result()))
                if done % 10 == 0 or done == len(backfill):
                    print(f"    {done}/{len(backfill)}  "
                          f"{time.perf_counter() - started:.0f} s", flush=True)
    report_loo_backfill(store.records, "ablation")
    failures = [r for r in store.records.values() if r.get("fit_failed")]
    pd.DataFrame(failures if failures else [],
                 columns=None if failures else ["fit_key", "fit_failure_reason"]
                 ).to_csv(ablation_out("21", "failures"), index=False, encoding=ENC)
    print(f"  적합 실패 {len(failures)}건 → {ablation_out('21', 'failures').name}")
    store.close()


def component(record, prefix="test") -> Component:
    table = record.get(f"{prefix}_quantiles")
    return Component(record[f"{prefix}_mean_z"], record[f"{prefix}_sd_z"],
                        None if table is None else np.asarray(table))


def point_metrics(comp, context) -> dict:
    response, truth_g = context["response"], context["truth_g"]
    # The point prediction is the summary POINT_SUMMARY names, the centre of the
    # conformal interval (was comp.ppf(0.5), the predictive median).
    point_g = response.from_z(point_z(comp))
    sse = float(np.sum((point_g - truth_g) ** 2))
    denominator = float(np.sum((truth_g - context["reference_g"]) ** 2))
    return {"g_RMSE": float(np.sqrt(np.mean((point_g - truth_g) ** 2))),
            "g_MAE": float(np.mean(np.abs(point_g - truth_g))),
            "g_MAPE": float(np.mean(np.abs(point_g - truth_g) / truth_g) * 100.0),
            "g_R2_pred": float(1.0 - sse / denominator)}


def build_stack(kernels, loo, finals, fixed=None) -> dict:
    """Yao stacking on a kernel set.  ``loo[k]`` is a list of 24 LOO records."""
    matrix = np.column_stack([[r["loo_lpd_g"] for r in loo[k]] for k in kernels])
    n_rows = matrix.shape[0]
    if fixed is None:
        # this arm's own stacking weights from its full LOO density matrix, shared
        # by every row's conformity score
        weights = optimize_weights(matrix)
    else:
        weights = np.asarray(fixed, float)         # no data enters fixed weights
    cross = np.tile(weights, (n_rows, 1))
    mixture = MixtureComponent([finals[k] for k in kernels], weights)
    loo_summary = loo_mixture_summary(kernels, loo, weights)
    return {"kernels": kernels, "matrix": matrix, "weights": weights, "cross": cross,
            "mixture": mixture, "loo": loo_summary, "loo_mean": loo_summary["mean"],
            "loo_median": loo_summary["median"], "truth": loo_summary["truth"],
            "expH": float(exp_entropy(weights)),
            "mixture_loo_lpd_g": float(mixture_lpd(matrix, weights))}


def conformal(stack, context, level=NOMINAL) -> dict:
    """gamma = 0 conformal: one radius from the LOO absolute residuals about the
    LOO mixture's median (common weights), added symmetrically to the full-data
    mixture median -- the summary POINT_SUMMARY names."""
    response, truth_g = context["response"], context["truth_g"]
    q = float(conformal_quantile(conformity_scores(stack["loo"]), level))
    centre = point_z(stack["mixture"])
    low, high = response.from_z(centre - q), response.from_z(centre + q)
    tail = (1.0 - level) / 2.0
    native_low = response.from_z(stack["mixture"].ppf(tail))
    native_high = response.from_z(stack["mixture"].ppf(1.0 - tail))
    covered = int(np.sum((truth_g >= low) & (truth_g <= high)))
    cp_width = float(np.mean(high - low))
    native_width = float(np.mean(native_high - native_low))
    return {"cp_covered": covered, "cp_coverage": covered / len(truth_g),
            "cp_width_g": cp_width, "native_width_g": native_width,
            "native_covered": int(np.sum((truth_g >= native_low) & (truth_g <= native_high))),
            "cp_reduction_pct": 100.0 * (1.0 - cp_width / native_width),
            "cp_radius_z": q}


def step_audit(values_z, frame) -> dict:
    """Decreasing steps of a curve, per cohort ordered by age; a step belongs to
    the window of its right endpoint (the rule 0905_abalone4/5 used)."""
    cohort = frame[COHORT_COL].astype(str).to_numpy()
    age = frame[AGE_COL].to_numpy(float)
    window = frame["window"].to_numpy()
    totals = {"down_train": 0, "down_test": 0, "steps_train": 0, "steps_test": 0}
    for name in pd.unique(cohort):
        mask = cohort == name
        order = np.argsort(age[mask], kind="mergesort")
        steps = np.diff(np.asarray(values_z, float)[mask][order])
        side = window[mask][order][1:]
        for label in ("train", "test"):
            totals[f"down_{label}"] += int(np.sum((steps < -MONOTONE_TOL) & (side == label)))
            totals[f"steps_{label}"] += int(np.sum(side == label))
    return totals


def decreasing_runs(values_g, frame, arm, kernels_label) -> list[dict]:
    """Maximal runs of consecutive decreasing steps, reported in grams."""
    cohort = frame[COHORT_COL].astype(str).to_numpy()
    age = frame[AGE_COL].to_numpy(float)
    window = frame["window"].to_numpy()
    rows = []
    for name in pd.unique(cohort):
        mask = cohort == name
        order = np.argsort(age[mask], kind="mergesort")
        g = np.asarray(values_g, float)[mask][order]
        a = age[mask][order]
        w = window[mask][order]
        down = np.diff(g) < -MONOTONE_TOL * max(1.0, float(np.abs(g).max()))
        index = 0
        while index < len(down):
            if not down[index]:
                index += 1
                continue
            start = index
            while index < len(down) and down[index]:
                index += 1
            left, right = start, index            # g[left] -> g[right]
            span = set(w[left:right + 1])
            rows.append({"arm": arm, "kernels": kernels_label,
                         "cohort": str(name),
                         "age_start": float(a[left]), "age_end": float(a[right]),
                         "g_start": float(g[left]), "g_end": float(g[right]),
                         "drop_g": float(g[left] - g[right]),
                         "drop_pct": 100.0 * float(g[left] - g[right]) / float(g[left]),
                         "n_steps": int(right - left),
                         "window": "test" if span == {"test"} else
                                   "train" if span == {"train"} else "train→test"})
    return rows


def gather(context, previous, mine):
    """Every LOO / final record the tables need, grouped by arm, with a hard stop
    on anything missing -- a comparison built on a partial arm is not reported."""
    n_rows = len(context["train"])

    def need(keys, label):
        missing = [k for k in keys if k not in mine or mine[k].get("fit_failed")]
        return missing

    arms, status = {}, {}
    # (a) and the reference: the published fits
    arms["a"] = {
        "loo": {k: [previous[f"cploo|integral|{RESPONSE}|{k}|{p}"] for p in range(n_rows)]
                for k in KERNELS},
        "final": {k: component(previous[f"final|integral|{RESPONSE}|{k}|step0"])
                  for k in KERNELS},
        "curve": {k: previous[f"curve|integral|{RESPONSE}|{k}|step0"] for k in KERNELS}}
    status["a"] = "complete"

    def own(exp, kernels):
        keys = [f"{exp}|{k}|loo|{p}" for k in kernels for p in range(n_rows)]
        keys += [f"{exp}|{k}|final|None" for k in kernels]
        missing = need(keys, exp)
        if missing:
            return None, missing
        unpaired = [key for key in keys
                    if int(mine[key]["job_seed"]) != anchor_seed(
                        previous, key.split("|")[1], key.split("|")[2],
                        None if key.endswith("|None") else int(key.split("|")[3]))]
        if unpaired:
            raise AssertionError(f"{exp} 의 {len(unpaired)}개 적합이 짝지은 시드가 아니다 "
                                 f"(예: {unpaired[0]}). --stage fits 로 다시 적합할 것.")
        return {"loo": {k: [mine[f"{exp}|{k}|loo|{p}"] for p in range(n_rows)] for k in kernels},
                "final": {k: component(mine[f"{exp}|{k}|final|None"]) for k in kernels},
                "dense": {k: mine[f"{exp}|{k}|final|None"] for k in kernels}}, missing

    for exp, kernels in (("age", KERNELS), ("soft", KERNELS), ("rv", RV_KERNELS),
                         ("plain", KERNELS)):
        data, missing = own(exp, kernels)
        arms[exp] = data
        status[exp] = "complete" if data is not None else f"missing {len(missing)}"
    return arms, status


def check_reference(stack, cp, context) -> None:
    """Compare the reference arm with the published numbers and RECORD the result
    (ablation_00_reference_check.csv).  A gap is reported, not fatal: the
    conformal width and coverage move with the LOO-mixture-median score, and the
    reference values are kept as they are, never replaced by the computed ones."""
    computed = {"weights": np.asarray(stack["weights"], float),
                "g_MAPE": point_metrics(stack["mixture"], context)["g_MAPE"],
                "cp_width_g": cp["cp_width_g"], "native_width_g": cp["native_width_g"]}
    rows = []
    for name, value in computed.items():
        reference = np.asarray(REFERENCE[name], float)
        pinned = bool(np.all(np.isfinite(reference)))
        gap = float(np.max(np.abs(np.asarray(value, float) - reference))) if pinned else math.nan
        rows.append({"item": name, "reference": REFERENCE[name],
                     "computed": np.round(value, 6).tolist() if np.ndim(value) else value,
                     "gap": gap, "match": (gap <= REPRO_TOL) if pinned else None})
    pinned = REFERENCE["cp_covered"] is not None
    rows.append({"item": "cp_covered", "reference": REFERENCE["cp_covered"],
                 "computed": int(cp["cp_covered"]), "gap": math.nan,
                 "match": (int(cp["cp_covered"]) == REFERENCE["cp_covered"]) if pinned else None})
    frame = pd.DataFrame(rows)
    frame.to_csv(ablation_out("00", "reference_check"), index=False, encoding=ENC)
    differ = frame[frame.match.eq(False)]
    if len(differ):
        print(f"  기준 arm 과 논문 기준값의 차이 (기록만 하고 계속 진행 → "
              f"{ablation_out('00', 'reference_check').name}):")
        ablation_show(differ, ["item", "reference", "computed", "gap"], digits=6)
    else:
        print(f"  재현 확인: 기준 stacking 이 고정된 논문 기준값과 일치 "
              f"(coverage {cp['cp_covered']}/8)")


def table_e1(context, arms):
    ablation_banner("E1 — age-only 입력 (환경변수 4종 제거)")
    print(f"  RBF_add3 는 환경 그룹이 사라져 age-only 에서 {AGE_ONLY_KERNEL} "
          "(k_age + k_cohort) 가 된다. Matern0.5, NNKernel 은 형태 그대로.")
    rows, weight_rows, interval_rows = [], [], []
    stacks = {}
    for label, data in (("full", arms["a"]), ("age-only", arms["age"])):
        stack = build_stack(KERNELS, data["loo"], data["final"])
        cp = conformal(stack, context)
        stacks[label] = (stack, cp)
        rows.append({"inputs": label, "model": "Stacking", "weight": np.nan,
                     **point_metrics(stack["mixture"], context),
                     "expH": stack["expH"], "mixture_loo_lpd_g": stack["mixture_loo_lpd_g"]})
        for j, k in enumerate(KERNELS):
            name = k if label == "full" else f"{k} [{AGE_SPEC[k]}]"
            rows.append({"inputs": label, "model": name, "weight": float(stack["weights"][j]),
                         **point_metrics(data["final"][k], context),
                         "expH": np.nan,
                         "mixture_loo_lpd_g": float(stack["matrix"][:, j].mean())})
            weight_rows.append({"inputs": label, "kernel": k, "weight": float(stack["weights"][j]),
                                "cross_fitted_mean": float(stack["cross"][:, j].mean()),
                                "cross_fitted_min": float(stack["cross"][:, j].min()),
                                "cross_fitted_max": float(stack["cross"][:, j].max()),
                                "rows_dominant": int(np.sum(
                                    (np.exp(stack["matrix"]) * stack["weights"]).argmax(1) == j))})
        interval_rows.append({"inputs": label, **cp})
    check_reference(*stacks["full"], context)
    frame = pd.DataFrame(rows)
    full = frame[frame.inputs == "full"].reset_index(drop=True)
    age = frame[frame.inputs == "age-only"].reset_index(drop=True)
    delta = pd.DataFrame({"model": full.model,
                          **{f"d_{c}": age[c] - full[c]
                             for c in ("g_RMSE", "g_MAE", "g_MAPE", "g_R2_pred")},
                          "d_weight": age.weight - full.weight})
    weights = pd.DataFrame(weight_rows)
    intervals = pd.DataFrame(interval_rows)
    frame.to_csv(ablation_out("01", "E1_age_only_models"), index=False, encoding=ENC)
    delta.to_csv(ablation_out("01b", "E1_age_only_minus_full"), index=False, encoding=ENC)
    weights.to_csv(ablation_out("01c", "E1_weights"), index=False, encoding=ENC)
    intervals.to_csv(ablation_out("01d", "E1_intervals"), index=False, encoding=ENC)
    ablation_show(frame, ["inputs", "model", "weight", "g_RMSE", "g_MAE", "g_MAPE", "g_R2_pred",
                 "expH", "mixture_loo_lpd_g"])
    print("\n  차이 (age-only − full)")
    ablation_show(delta)
    print("\n  가중치 (보정점수의 24개 행 모두 전체 LOO 밀도로 추정한 이 가중치 하나를 쓴다; "
          "cross_fitted_* 열은 그 행별 가중치라 weight 와 같다)")
    ablation_show(weights)
    print("  " + MC_NOISE_NOTE)
    print(f"\n  명목 {NOMINAL:.0%} conformal (gamma 0)")
    ablation_show(intervals, ["inputs", "cp_covered", "cp_width_g", "native_covered",
                     "native_width_g", "cp_reduction_pct"])
    return stacks


def table_e2(context, arms, stacks):
    ablation_banner("E2 — 균등가중 w = (1/3, 1/3, 1/3)")
    optimal, optimal_cp = stacks["full"]
    equal = build_stack(KERNELS, arms["a"]["loo"], arms["a"]["final"],
                        fixed=[1.0 / 3, 1.0 / 3, 1.0 / 3])
    equal_cp = conformal(equal, context)
    rows = []
    for label, stack, cp in (("LOO-optimal", optimal, optimal_cp),
                             ("equal", equal, equal_cp)):
        rows.append({"weights": label,
                     "w": ", ".join(f"{k}:{w:.4f}" for k, w in zip(KERNELS, stack["weights"])),
                     **point_metrics(stack["mixture"], context),
                     "mixture_loo_lpd_g": stack["mixture_loo_lpd_g"], "expH": stack["expH"],
                     **{k: cp[k] for k in ("cp_covered", "cp_width_g", "native_width_g",
                                           "cp_reduction_pct")}})
    frame = pd.DataFrame(rows)
    diff = {c: float(frame[c].iloc[1] - frame[c].iloc[0])
            for c in ("g_RMSE", "g_MAE", "g_MAPE", "g_R2_pred", "mixture_loo_lpd_g",
                      "cp_width_g", "native_width_g")}
    frame = pd.concat([frame, pd.DataFrame([{"weights": "equal − optimal", "w": "", **diff}])],
                      ignore_index=True)
    frame.to_csv(ablation_out("02", "E2_equal_weights"), index=False, encoding=ENC)
    ablation_show(frame, ["weights", "w", "g_RMSE", "g_MAE", "g_MAPE", "g_R2_pred",
                 "mixture_loo_lpd_g", "cp_covered", "cp_width_g", "native_width_g"])
    z_optimal = mixture_lpd(np.column_stack([[r["loo_lpd_z"] for r in arms["a"]["loo"][k]]
                                                for k in KERNELS]), optimal["weights"])
    print(f"\n  LOO mixture LPD 는 stacking 목적함수와 같은 척도(g 밀도)로 최적 "
          f"{optimal['mixture_loo_lpd_g']:+.4f}, 균등 {equal['mixture_loo_lpd_g']:+.4f} 이다. "
          f"z 척도로는 {z_optimal:+.4f}. 요청서의 −1.6118 은 어느 척도와도 일치하지 않는다 "
          "— 출처 확인 필요.")
    return equal


def arm_stack(context, arms, arm, kernels):
    source = arms[{"a": "a", "b": "soft", "c": "rv", "d": "plain"}[arm]]
    return build_stack(kernels, source["loo"], source["final"])


def arm_curve(context, arms, arm, stack):
    """Stacking point prediction along the audit grid, z and g, and the grid it
    lives on -- the summary POINT_SUMMARY names, as every reported metric uses."""
    response = context["response"]
    if arm == "a":
        frame = ablation_grid_frame(context)
        parts = [component(arms["a"]["curve"][k], prefix="curve") for k in stack["kernels"]]
        label = f"integration grid ({len(frame)} nodes)"
    else:
        frame = context["ctx"]["dense"]
        source = arms[{"b": "soft", "c": "rv", "d": "plain"}[arm]]["dense"]
        parts = [Component(source[k]["dense_mean_z"], source[k]["dense_sd_z"], None)
                 for k in stack["kernels"]]
        label = f"dense scan grid ({len(frame)} points)"
    mixture = MixtureComponent(parts, stack["weights"])
    # The audited curve is the point prediction itself, so the monotonicity scan
    # and the reported metrics describe one curve under either POINT_SUMMARY.
    summary_z = point_z(mixture)
    return frame, summary_z, response.from_z(summary_z), label, parts


def table_e3(context, arms, status):
    ablation_banner("E3 — 단조성 부과 방식")
    print("  " + RV_EXCLUDED)
    print(SEED_DEPENDENCE_NOTE)
    if status["plain"] == "complete":
        # Built-in evidence for (d): its final fits run under the anchor seed yet
        # reproduce 0905_abalone3's screen fits, which used a different seed.
        print("  (d) 시드 무관성의 내장 증거 — final 적합이 다른 시드의 0905_abalone3 "
              "screen 적합을 재현:")
        for kernel, record in arms["plain"]["dense"].items():
            gaps = record.get("repro_gaps", {})
            print(f"     {kernel:10s} 시드 {record['job_seed']} vs screen "
                  f"{record.get('repro_source_seed')}  최대 차이 "
                  f"{max(gaps.values()) if gaps else float('nan'):.2e}")
    results, runs, per_kernel = [], [], []
    for support, kernels_for in (("3-kernel", lambda arm: RV_KERNELS if arm == "c" else KERNELS),
                                 ("common 2-kernel", lambda arm: COMMON)):
        for arm in ("a", "b", "c", "d"):
            key = {"a": "a", "b": "soft", "c": "rv", "d": "plain"}[arm]
            kernels = kernels_for(arm)
            row = {"support": support, "arm": f"({arm}) {ARMS[arm]}",
                   "kernels": "+".join(kernels)}
            if status[key] != "complete":
                row["status"] = f"미완료: {status[key]}"
                results.append(row)
                continue
            stack = arm_stack(context, arms, arm, kernels)
            frame, mean_z, mean_g, grid_label, parts = arm_curve(context, arms, arm, stack)
            audit = step_audit(mean_z, frame)
            row.update({"status": "complete" + (" (Matern0.5 제외)" if arm == "c"
                                                and support == "3-kernel" else ""),
                        "weights": ", ".join(f"{k}:{w:.3f}" for k, w in
                                             zip(kernels, stack["weights"])),
                        "expH": stack["expH"], **point_metrics(stack["mixture"], context),
                        "mixture_loo_lpd_g": stack["mixture_loo_lpd_g"],
                        "down_train": f"{audit['down_train']} / {audit['steps_train']}",
                        "down_test": f"{audit['down_test']} / {audit['steps_test']}",
                        "audit_grid": grid_label})
            results.append(row)
            if support == "3-kernel":
                runs += decreasing_runs(mean_g, frame, f"({arm})", "+".join(kernels))
                for k, part in zip(kernels, parts):
                    single = step_audit(np.asarray(part.mean_z, float), frame)
                    per_kernel.append({"arm": f"({arm})", "kernel": k, **single,
                                       "audit_grid": grid_label})
    frame = pd.DataFrame(results)
    frame.to_csv(ablation_out("03", "E3_stacking_by_arm"), index=False, encoding=ENC)
    print("\n  stacking 성능과 감소 스텝 (train / test 구간)")
    ablation_show(frame, ["support", "arm", "kernels", "status", "expH", "g_MAPE", "g_MAE", "g_RMSE",
                 "g_R2_pred", "down_train", "down_test", "audit_grid"])
    print("\n  가중치")
    ablation_show(frame, ["support", "arm", "weights", "mixture_loo_lpd_g"])
    kernel_frame = pd.DataFrame(per_kernel)
    kernel_frame.to_csv(ablation_out("03b", "E3_steps_by_kernel"), index=False, encoding=ENC)
    print("\n  단독 커널의 감소 스텝 (예측평균 곡선)")
    ablation_show(kernel_frame)

    run_frame = pd.DataFrame(runs)
    run_frame.to_csv(ablation_out("03c", "E3_decreasing_runs"), index=False, encoding=ENC)
    print("\n  stacking 곡선이 실제로 감소하는 구간 (g 단위, 연속 감소 스텝을 하나로 묶음)")
    if len(run_frame):
        run_frame["cohort"] = run_frame["cohort"].map(lambda c: A1_label(c))
        summary = (run_frame.sort_values("drop_g", ascending=False)
                   .groupby(["arm", "cohort"], as_index=False).head(1)
                   .sort_values(["arm", "cohort"]))
        print("  시설별 가장 큰 감소 구간")
        ablation_show(summary, ["arm", "cohort", "age_start", "age_end", "g_start", "g_end",
                       "drop_g", "drop_pct", "n_steps", "window"], digits=2)
        test_runs = run_frame[run_frame.window != "train"]
        print(f"\n  test 구간에 걸친 감소 구간 수: "
              + ", ".join(f"{a} {int((test_runs.arm == a).sum())}"
                          for a in ("(a)", "(b)", "(c)", "(d)")))
    else:
        print("  감소 구간 없음")
    return frame


def A1_label(cohort: str) -> str:
    return ROMAN.get(str(cohort), str(cohort))


def table_confounders(arms, status, mine):
    ablation_banner("E3 교란 요인 — 제약 말고 무엇이 함께 바뀌는가")
    SOFT = sys.modules[__name__]
    RV = sys.modules[__name__]

    def measured(exp, field):
        values = [r.get(field) for k, r in mine.items()
                  if k.startswith(f"{exp}|") and "|final|" in k and r.get(field) is not None]
        return (f"{np.mean(values):.1f} (final fit 평균)" if values else "—")

    rows = [
        ("GP 를 얹는 대상", "잠재 성장률 g", "반응 z", "반응 z", "반응 z"),
        ("평균함수",
         "성장률에 상수 μ_g (학습) + 시설별 초기값 f0 ~ N(0, 1) "
         "(적합에서 주변화, 예측에서 조건부)", "ZeroMean", "ZeroMean", "ZeroMean"),
        ("적분 구조", f"사다리꼴 적분, 관측 일령 사이 M = {GRID_M} 분할", "없음", "없음", "없음"),
        ("단조성 부과",
         "구조적: f = f0 + ∫ softplus(g), 모든 곳에서 증분 ≥ 0",
         f"벌점: 사후평균 일령 도함수의 logistic sign loss, 시설당 {SOFT_GRID_SIZE}점 "
         f"(train 시작 ~ test 끝), 가중치 {DERIV_WEIGHT}, 온도 "
         f"{SOFT.DERIVATIVE_SIGN_TEMPERATURE}",
         f"가상 도함수 관측 Φ(f′/ν), 시설당 {VIRTUAL_INITIAL}점 시작 → 위반확률 "
         f">{VIRTUAL_PROB_THRESHOLD} 인 곳에 최대 {VIRTUAL_MAX}점까지 {VIRTUAL_ROUNDS}회 "
         f"추가, test 일령 포함, ν = 0.01 × 경험 성장률",
         "없음"),
        ("잠재 추론",
         "Laplace (Gauss-Newton mode)",
         "정확 GP (해석적)",
         f"EP (감쇠 {RV.EP_DAMPING}, 최대 {RV.EP_MAX_ITER}회, tol {RV.EP_TOL})",
         "정확 GP (해석적)"),
        ("예측분포",
         f"MC {MC_DRAWS} draw (softplus·적분·f0 전파), LOO 밀도는 moment-matched 정규",
         "가우시안", "EP 가우시안 근사", "가우시안"),
        ("하이퍼파라미터 목적함수",
         f"Laplace 주변우도 + lognormal prior + inverse-gamma noise prior "
         f"(MAP), Adam {HYPER_ITER_INTEGRAL}회 lr {HYPER_LR}",
         f"gpytorch MLL/n − prior/n + {DERIV_WEIGHT}·벌점, noise prior 없음, "
         f"Adam {SOFT_ITER}회 lr {SOFT.LEARNING_RATE}, jitter {JITTER_SOFT}",
         f"정확 MLL + lognormal + inverse-gamma (MAP), 제약 없이 적합 후 고정, "
         f"Adam {HYPER_ITER_PLAIN}회 lr {HYPER_LR}",
         f"정확 MLL + lognormal + inverse-gamma (MAP), Adam {HYPER_ITER_PLAIN}회 lr {HYPER_LR}"),
        ("하이퍼파라미터가 제약을 보는가", "예 (Laplace 주변우도 안에 구조 포함)",
         "예 (벌점이 목적함수에 포함)", "아니오 (제약 없이 적합한 값을 EP 에 고정)", "해당 없음"),
        ("stacking 후보", " + ".join(KERNELS), " + ".join(KERNELS),
         " + ".join(RV_KERNELS) + " (Matern0.5 불가)", " + ".join(KERNELS)),
        ("측정: 가상점 수 / 벌점 격자 위반", "—",
         measured("soft", "constraint_grid_violations"), measured("rv", "n_virtual"), "—"),
    ]
    frame = pd.DataFrame(rows, columns=["항목", "(a) integral", "(b) soft penalty",
                                         "(c) Riihimaki-Vehtari", "(d) unconstrained"])
    frame.to_csv(ablation_out("03d", "E3_confounders"), index=False, encoding=ENC)
    for _, row in frame.iterrows():
        print(f"\n  [{row['항목']}]")
        for column in frame.columns[1:]:
            print(f"     {column:24s} {row[column]}")
    return frame


# =============================================================================
# Figures
#
# Carried over from 0905_abalone5.py. Its NOMINAL (0.90, the level it first
# drew at) is FIG_NOMINAL and its BOOTSTRAP_DRAWS (200) FIG_BOOTSTRAP_DRAWS;
# the published _p80 figures are drawn by passing level=0.80, as they were.
# Renamed: assemble, table_intervals, grid_frame, component_of, Store, FIGS,
# PREFIX, CHECK -> assemble_figures, table_intervals_methods,
# figure_grid_frame, figure_component_of, FigureStore, FIGURE_DIR,
# FIGURE_PREFIX, FIGURE_CHECK. INK replaces this file's earlier six-colour INK
# with the nine-colour palette the figures were drawn with.
# table_intervals_methods computes, besides the native and gamma = 0 intervals
# that are drawn, a Jackknife+ row and the other gamma values; those are
# comparison rows and reach no table or figure. Declared changes:
# growth_panels draws the point_z summary (the mixture median by default) and
# centres the conformal band on that same array; the conformity scores are the
# residuals about the LOO mixture median with the full-LOO stacking weights
# (loo_mixture_summary), not cross-fitted per row. Two legend strings are reworded to
# "residual-calibrated" at the authors' request; the colour keys, variable
# names and every computed quantity are untouched.
# =============================================================================
BAND_METHOD = "gamma_0"


GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# 0.80 added on request.  The Jackknife+ here is TWO-SIDED, so the order
# statistics it needs at level L are floor(25*(1-L)/2) and ceil(25*(1+L)/2):
# at 0.80 that is the 2nd and the 23rd of 24, both inside the sample, so the
# level is attainable unclipped (ceil(25*0.8) = 20 would be the one-sided rank).
# 0.90 sits exactly on the boundary at ranks 1 and 24; 0.95 is the level that
# clips, needing rank 25.
NOMINAL_ALL = [0.50, 0.75, 0.80, 0.90]


BEST_COMPOSITE = "Matern0.5+RBF_add3+NNKernel"   # rank 1 of 23 in abalone4 Table B


SHOW_BEST_COMPOSITE = False    # set False to draw Stacking alone in fig1/fig2


FIG_BOOTSTRAP_DRAWS = 200


# Okabe-Ito, colour-vision-deficiency safe
INK = {"stack": "#0072B2", "band": "#56B4E9", "native": "#999999",
       "conformal": "#D55E00", "composite": "#009E73", "train": "#000000",
       "test": "#CC79A7", "warn": "#D55E00", "grid": "#DDDDDD"}


# Linear-axis variants of the growth panels.  A keeps the whole conformal band;
# B cuts the axis at LINEAR_B_MULTIPLE x the largest observation and marks the
# cut.  Both share one range across the four cohorts, since on a linear axis a
# shared range is what makes the panels comparable.
LINEAR_CEILING_STEP = 5.0      # A's ceiling rounds up to a multiple of this


LINEAR_B_STEP = 2.5            # B's rounds to a finer one: at step 5.0 the 1.5x
                               # ceiling lands on 20 g and clips a single node,
                               # which makes B indistinguishable from A


LINEAR_B_MULTIPLE = 1.5        # B ceiling = 1.5 x the largest observed weight


HIDDEN_FRACTION = 0.015        # an observation this close to zero, as a fraction
                               # of the axis span, is lost in the axis line


FIG_NOMINAL = 0.90


def label_of(cohort: str) -> str:
    return ROMAN.get(str(cohort), str(cohort))


def short_of(cohort: str) -> str:
    return SHORT.get(str(cohort), str(cohort))


def panel_limits(low, high, pad_low=0.35, pad_high=1.6):
    """Log-scale limits with room above for annotation."""
    return float(low) * pad_low, float(high) * pad_high


def suffix_for(level: float) -> str:
    """No suffix at the headline level, so existing file names do not move."""
    return "" if abs(level - FIG_NOMINAL) < 1e-12 else f"_p{int(round(level * 100))}"


def save(figure, name: str) -> None:
    """PDF is the LaTeX master; the PNG is for previewing."""
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300})):
        figure.savefig(FIGURE_DIR / f"{FIGURE_PREFIX}_{name}{suffix}", **kwargs)
    plt.close(figure)
    print(f"  saved {name}.pdf / .png", flush=True)


class FigureStore:
    def __init__(self, path: Path, append: bool = True):
        self.path, self.records = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["fit_key"]] = record
        self.handle = path.open("a", encoding="utf-8") if append else None

    @classmethod
    def read_only(cls, path: Path) -> dict:
        return cls(path, append=False).records

    def __contains__(self, key):
        return key in self.records and not self.records[key].get("fit_failed")

    def __getitem__(self, key):
        return self.records[key]

    def add(self, record):
        self.records[record["fit_key"]] = record
        self.handle.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close(); self.handle = None


def figure_grid_frame(context) -> pd.DataFrame:
    """Every integration node as a prediction row.

    The integral construction's predict() only accepts ages that are exact grid
    nodes, so the growth curve is drawn ON the grid rather than on an independent
    linspace.  M = 4 subdivisions between consecutive observed ages gives 29
    nodes per cohort, which is dense enough to read monotonicity off the line.
    """
    grid, source = context["grid"], context["ctx"]["source"]
    blocks = []
    for cohort in grid.cohorts:
        ages = np.asarray(grid.nodes[cohort], float)
        block = age_frame(source, [cohort], {cohort: ages})
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    boundary = context["train"].groupby(COHORT_COL, observed=True)[AGE_COL].max().to_dict()
    frame["window"] = [
        "train" if row[AGE_COL] <= float(boundary[str(row[COHORT_COL])]) else "test"
        for _, row in frame.iterrows()]
    return frame


def figure_component_of(record) -> Component:
    table = record.get("test_quantiles")
    return Component(record["test_mean_z"], record["test_sd_z"],
                        None if table is None else np.asarray(table))


def assemble_figures(context):
    previous = load_previous()
    states = FigureStore.read_only(FIGURE_CHECK / f"{FIGURE_PREFIX}_states.jsonl")
    missing = [k for k in ALL_KERNELS if f"plain|{RESPONSE}|{k}|step0" not in states]
    missing += [k for k in KERNELS + [BEST_COMPOSITE]
                if f"curve|{METHOD}|{RESPONSE}|{k}|step0" not in states]
    if missing:
        raise RuntimeError(f"상태 적합이 없다: {missing}. --stage fits 를 먼저 돌릴 것.")

    n_rows = len(context["train"])
    lpd = {k: np.array([previous[f"loo|{METHOD}|{RESPONSE}|{k}|{p}"]["loo_lpd_g"]
                        for p in range(n_rows)]) for k in KERNELS}
    cp = {k: [previous[f"cploo|{METHOD}|{RESPONSE}|{k}|{p}"] for p in range(n_rows)]
          for k in KERNELS}
    singles = {k: figure_component_of(previous[f"final|{METHOD}|{RESPONSE}|{k}|step0"])
               for k in KERNELS}
    matrix = np.column_stack([lpd[k] for k in KERNELS])
    weights = optimize_weights(matrix)
    # the same weights for every row's conformity score, as in assemble
    cross = np.tile(weights, (n_rows, 1))
    boot = cohort_bootstrap_weights(matrix, context["cohort_of_row"],
                                       FIG_BOOTSTRAP_DRAWS, SEED)
    stack = {"kernels": KERNELS, "lpd": matrix, "weights": weights, "cross": cross,
             "boot": boot, "expH": float(exp_entropy(weights)),
             "mixture": mixture_lpd(matrix, weights),
             "single": {k: float(matrix[:, j].mean()) for j, k in enumerate(KERNELS)}}
    mixture = MixtureComponent([singles[k] for k in KERNELS], weights)

    # The LOO mixture at each training row -- its median is the conformity score's
    # centre -- from the same function and the same records as assemble's.
    loo = dict(loo_mixture_summary(KERNELS, cp, weights))
    loo.update({
           "target_mean": sum(cross[:, j][:, None]
                              * np.array([r["target_mean_z"] for r in cp[k]])
                              for j, k in enumerate(KERNELS)),
           "cdf": sum(cross[:, j] * np.array([r["loo_cdf_at_truth"] for r in cp[k]])
                      for j, k in enumerate(KERNELS))})
    loo["target_sd"] = np.sqrt(np.maximum(
        sum(cross[:, j][:, None] * (np.array([r["target_sd_z"] for r in cp[k]]) ** 2
                                    + np.array([r["target_mean_z"] for r in cp[k]]) ** 2)
            for j, k in enumerate(KERNELS)) - loo["target_mean"] ** 2, 1e-18))
    return {"previous": previous, "states": states, "lpd": lpd, "cp": cp,
            "singles": singles, "stack": stack, "mixture": mixture, "loo": loo}


def curve_components(context, parts):
    """Mixture and best-composite predictive on the integration grid."""
    states, curve = parts["states"], figure_grid_frame(context)
    parts_list = []
    for kernel in KERNELS:
        record = states[f"curve|{METHOD}|{RESPONSE}|{kernel}|step0"]
        table = record.get("curve_quantiles")
        parts_list.append(Component(record["curve_mean_z"], record["curve_sd_z"],
                                       None if table is None else np.asarray(table)))
    mixture = MixtureComponent(parts_list, parts["stack"]["weights"])
    record = states[f"curve|{METHOD}|{RESPONSE}|{BEST_COMPOSITE}|step0"]
    table = record.get("curve_quantiles")
    composite = Component(record["curve_mean_z"], record["curve_sd_z"],
                             None if table is None else np.asarray(table))
    return curve, mixture, composite


def conformal_bands(context, parts, level=FIG_NOMINAL, gamma=0.0):
    """gamma-conformal radius on the z scale: q * sd^gamma, sd^0 = 1.  The scores
    are conformity_scores, the residuals about the LOO mixture median."""
    loo = parts["loo"]
    scores = conformity_scores(loo) / np.maximum(loo["sd"], 1e-12) ** gamma
    return float(conformal_quantile(scores, level))


def table_intervals_methods(context, parts) -> pd.DataFrame:
    """Native and conformal intervals for the mixture at the eight test rows."""
    response, truth_g = context["response"], context["truth_g"]
    component, loo = parts["mixture"], parts["loo"]
    n_rows, n_test = len(context["train"]), len(truth_g)
    mean_z = point_z(component)          # the centre the gamma bands are built on
    sd_z = np.asarray(component.sd_z, float)
    standardized = conformity_scores(loo) / np.maximum(loo["sd"], 1e-12)
    rows = []
    for level in NOMINAL_ALL:
        tail = (1.0 - level) / 2.0
        entries = [("native", response.from_z(component.ppf(tail)),
                    response.from_z(component.ppf(1.0 - tail)))]
        lower_rank, upper_rank, clipped = jackknife_ranks(n_rows, level)
        radius = standardized[:, None] * np.maximum(loo["target_sd"], 1e-12)
        lo_c = response.from_z(loo["target_mean"] - radius)
        hi_c = response.from_z(loo["target_mean"] + radius)
        entries.append(("jackknife+",
                        np.partition(lo_c, lower_rank - 1, axis=0)[lower_rank - 1],
                        np.partition(hi_c, upper_rank - 1, axis=0)[upper_rank - 1]))
        for gamma in GAMMAS:
            q = conformal_bands(context, parts, level, gamma)
            radius_gamma = q * np.maximum(sd_z, 1e-12) ** gamma
            entries.append((f"gamma_{gamma:g}",
                            response.from_z(mean_z - radius_gamma),
                            response.from_z(mean_z + radius_gamma)))
        native_z = float(np.mean(response.to_z(entries[0][2])
                                 - response.to_z(entries[0][1])))
        native_g = float(np.mean(entries[0][2] - entries[0][1]))
        for method, low, high in entries:
            low_z, high_z = response.to_z(low), response.to_z(high)
            width_z = float(np.mean(high_z - low_z))
            coverage = float(np.mean((truth_g >= low) & (truth_g <= high)))
            rows.append({
                "Method": method, "Nominal": level, "Coverage": coverage,
                "Coverage_n": f"{int(round(coverage * n_test))}/{n_test}",
                "Width_z": width_z, "Width_g": float(np.mean(high - low)),
                "Native_Width_g": native_g,
                "Ratio_vs_Native": width_z / max(native_z, 1e-12),
                "Width_Reduction_pct": 100.0 * (1.0 - width_z / max(native_z, 1e-12)),
                "Width_Reduction_pct_g": 100.0 * (1.0 - float(np.mean(high - low))
                                                  / max(native_g, 1e-12)),
                "shorter_than_native": bool(width_z < native_z - 1e-12),
                "coverage_ok": bool(coverage >= level - 1.0 / n_test),
                "lower_rank": lower_rank if method == "jackknife+" else math.nan,
                "upper_rank": upper_rank if method == "jackknife+" else math.nan,
                "rank_clipped": bool(clipped) if method == "jackknife+" else False,
                "lower_g": low.tolist(), "upper_g": high.tolist()})
    return pd.DataFrame(rows)


def figure_intervals(context, parts, intervals, level=FIG_NOMINAL, name=None):
    """Native against conformal at the eight test rows.

    The reduction drawn as the reference line is the AGGREGATE one the tables
    report -- (mean native width - mean conformal width) / mean native width on
    the gram scale -- not the mean of the per-row reductions.  The two differ
    because the widths vary by row; the per-row mean is printed for the caption.
    """
    name = name or f"fig_intervals{suffix_for(level)}"
    response, truth_g, target = context["response"], context["truth_g"], context["target"]
    block = intervals[intervals.Nominal == level].set_index("Method")
    native_lo = np.asarray(block.loc["native", "lower_g"], float)
    native_hi = np.asarray(block.loc["native", "upper_g"], float)
    cp_lo = np.asarray(block.loc[BAND_METHOD, "lower_g"], float)
    cp_hi = np.asarray(block.loc[BAND_METHOD, "upper_g"], float)
    labels = [f"{short_of(c)}\nt{6 + int(h)}"
              for c, h in zip(target[COHORT_COL].astype(str), target["test_horizon"])]
    x = np.arange(len(truth_g))
    per_row = 100.0 * (1.0 - (cp_hi - cp_lo) / (native_hi - native_lo))
    aggregate = 100.0 * (1.0 - float(np.mean(cp_hi - cp_lo))
                         / float(np.mean(native_hi - native_lo)))

    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(7.4, 5.6), height_ratios=[3, 1.15], sharex=True,
        layout="constrained")
    for offset, lo, hi, colour, text in (
            (-0.15, native_lo, native_hi, INK["native"], f"Native {level:.0%}"),
            (+0.15, cp_lo, cp_hi, INK["conformal"],
             f"Residual-calibrated {level:.0%}")):
        upper.vlines(x + offset, lo, hi, color=colour, linewidth=8, alpha=0.85,
                     label=text)
    inside = (truth_g >= cp_lo) & (truth_g <= cp_hi)
    upper.scatter(x[inside], truth_g[inside], s=44, color="black", zorder=5,
                  label="Observed (covered)")
    if (~inside).any():
        upper.scatter(x[~inside], truth_g[~inside], s=62, color=INK["warn"],
                      marker="X", zorder=6, label="Observed (not covered)")
    upper.set_ylabel("Whole weight (g)")
    upper.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)

    lower.bar(x, per_row, color=INK["conformal"], alpha=0.85, width=0.6)
    lower.axhline(aggregate, color="black", linestyle="--", linewidth=1.3,
                  label=f"mean width reduction {aggregate:.1f}%")
    lower.axhline(0.0, color="#666666", linewidth=0.8)
    lower.set_ylabel("Width\nreduction (%)")
    lower.set_xticks(x)
    lower.set_xticklabels(labels, fontsize=9)
    lower.set_ylim(min(per_row.min() * 1.35, -3.0), max(per_row.max(), aggregate) * 1.42)
    lower.legend(frameon=False, loc="upper left", fontsize=9)
    save(figure, name)

    horizon = target["test_horizon"].to_numpy(int)
    cohort = target[COHORT_COL].astype(str).to_numpy()
    print(f"    {name}: 기준선 = 집계 감소율 {aggregate:.1f}% (표와 동일 정의). "
          f"행별 감소율의 평균은 {per_row.mean():.1f}% — 캡션용")
    print(f"    (b) 명목 {level:.0%} 관측별 폭 감소율:")
    for index in range(len(truth_g)):
        print(f"          {label_of(cohort[index]):16s} t{6 + horizon[index]}  "
              f"native {native_hi[index] - native_lo[index]:6.3f} g -> "
              f"CP {cp_hi[index] - cp_lo[index]:6.3f} g   {per_row[index]:+6.2f}%")
    for value in (1, 2):
        mask = horizon == value
        print(f"    (c) t{6 + value} 4개 평균 감소율 {per_row[mask].mean():+6.2f}% "
              f"(native 평균 폭 {np.mean(native_hi[mask] - native_lo[mask]):.3f} g -> "
              f"CP {np.mean(cp_hi[mask] - cp_lo[mask]):.3f} g)")
    outside = np.flatnonzero(~inside)
    if outside.size:
        for index in outside:
            print(f"    (d) 명목 {level:.0%} conformal 밴드를 벗어난 관측: "
                  f"{label_of(cohort[index])} t{6 + horizon[index]}, "
                  f"실측 {truth_g[index]:.2f} g, 밴드 "
                  f"[{cp_lo[index]:.2f}, {cp_hi[index]:.2f}] g")
    else:
        print(f"    (d) 명목 {level:.0%} 에서 밴드를 벗어난 test 관측 없음 (8/8)")


def growth_panels(context, parts, mode="log", name="fig_growth_curves",
                  ceiling=None, clip=False, report=False, level=FIG_NOMINAL,
                  shared_log=False):
    """Four cohorts of growth curve, drawn on a log or a linear weight axis.

    mode="log"     per-panel limits.  The cohorts span very different weights --
                   Jindo-Daru starts at 0.0020 g and Wando-Geumhae at 0.2110 g, a
                   factor of 100 -- so a shared log axis flattens half the panels.
    mode="linear"  one shared range [0, ceiling] for all four, because on a
                   linear axis a shared range is what makes the panels
                   comparable.  ``clip`` cuts the band at the ceiling and marks
                   the cut so the reader can see the band continues.
    shared_log     log axis with ONE range for all four, snapped to whole decades
                   so every tick is a power of ten.  Costs headroom -- the band
                   tops out near 18 g and the decade above it is 100 g -- but it
                   is the only version in which the cohorts can be compared
                   directly, which per-panel limits make impossible.
    """
    curve, mixture, composite = curve_components(context, parts)
    response, train, test = context["response"], context["train"], context["test"]
    # The drawn line is the point prediction (the mixture median under
    # POINT_SUMMARY = "median"), and the conformal band is that SAME centre array
    # +- one radius.
    centre_z = point_z(mixture)
    point_curve = response.from_z(centre_z)
    q = conformal_bands(context, parts, level)
    low = response.from_z(centre_z - q)
    high = response.from_z(centre_z + q)
    composite_median = response.from_z(composite.ppf(0.5))
    cohorts = context["cohorts"]
    cohort_column = curve[COHORT_COL].astype(str).to_numpy()

    if report:
        best = int(np.argmax(high))
        print(f"    (a) {level:.0%} 밴드 상한 최댓값 {high[best]:.3f} g — "
              f"{label_of(cohort_column[best])}, 일령 "
              f"{float(curve[AGE_COL].to_numpy(float)[best]):.0f}d")
        for cohort in cohorts:
            mask = cohort_column == str(cohort)
            top = int(np.argmax(high[mask]))
            print(f"          {label_of(cohort):16s} 상한 {high[mask].max():8.3f} g "
                  f"@ {float(curve[AGE_COL].to_numpy(float)[mask][top]):.0f}d")

    decade_low = decade_high = None
    if shared_log:
        observed_all = np.concatenate([train["_w"].to_numpy(float),
                                       test["_w"].to_numpy(float)])
        decade_low = 10.0 ** math.floor(math.log10(float(observed_all.min())))
        decade_high = 10.0 ** math.ceil(math.log10(max(float(high.max()),
                                                       float(observed_all.max()))))
        if report:
            print(f"    (a) 공통 로그 범위 [{decade_low:g}, {decade_high:g}] g "
                  f"— 자료 실제 범위는 [{observed_all.min():.4g}, "
                  f"{max(float(high.max()), float(observed_all.max())):.4g}] g "
                  f"(밴드 상한 포함), {int(round(math.log10(decade_high/decade_low)))}개 decade")

    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), layout="constrained",
                                sharey=(mode == "linear" or shared_log))
    clipped_total, hidden_rows = 0, []
    for index, cohort in enumerate(cohorts):
        axis = axes.ravel()[index]
        mask = cohort_column == str(cohort)
        order = np.argsort(curve[AGE_COL].to_numpy(float)[mask], kind="mergesort")
        ages = curve[AGE_COL].to_numpy(float)[mask][order]
        boundary = float(train.loc[train[COHORT_COL].astype(str).eq(str(cohort)),
                                   AGE_COL].max())
        block_train = train[train[COHORT_COL].astype(str).eq(str(cohort))]
        block_test = test[test[COHORT_COL].astype(str).eq(str(cohort))]
        observed = np.concatenate([block_train["_w"].to_numpy(float),
                                   block_test["_w"].to_numpy(float)])
        band_low = low[mask][order]
        band_high = high[mask][order]

        if mode == "log" and shared_log:
            floor, top = decade_low, decade_high
            clipped_total += int(np.sum(band_low < floor))
            band_low = np.maximum(band_low, floor)
        elif mode == "log":
            floor, top = panel_limits(observed.min(),
                                      max(observed.max(), band_high.max()))
            clipped_total += int(np.sum(band_low < floor))
            band_low = np.maximum(band_low, floor)
        else:
            floor, top = 0.0, float(ceiling)
            if clip:
                over = band_high > top
                clipped_total += int(over.sum())
                band_high = np.minimum(band_high, top)

        axis.axvspan(boundary, ages.max(), color=INK["test"], alpha=0.07, zorder=0)
        axis.fill_between(ages, band_low, band_high, color=INK["band"], alpha=0.32,
                          linewidth=0,
                          label=f"{int(level*100)}% residual-calibrated interval",
                          zorder=1)
        if mode == "linear" and clip:
            over = high[mask][order] > top
            if over.any():
                axis.plot(ages[over], np.full(int(over.sum()), top), marker="^",
                          linestyle="none", markersize=4.5, color=INK["band"],
                          markeredgecolor=INK["stack"], markeredgewidth=0.6,
                          clip_on=False, zorder=7,
                          label="band continues above axis")
        if SHOW_BEST_COMPOSITE:
            axis.plot(ages, composite_median[mask][order], color=INK["composite"],
                      linewidth=1.6, linestyle="--", label="Best monotone composite",
                      zorder=2)
        axis.plot(ages, point_curve[mask][order], color=INK["stack"], linewidth=2.2,
                  label=f"Stacking ({POINT_SUMMARY})", zorder=3)
        axis.axvline(boundary, color="#666666", linestyle=":", linewidth=1.2,
                     label="Train / test boundary", zorder=2)
        axis.scatter(block_train[AGE_COL], block_train["_w"], s=26, color=INK["train"],
                     marker="o", zorder=5, label="Train (observed)")
        axis.scatter(block_test[AGE_COL], block_test["_w"], s=52, color=INK["test"],
                     marker="^", edgecolor="black", linewidth=0.6, zorder=6,
                     label="Test (observed)")
        if mode == "log":
            axis.set_yscale("log")
            if shared_log:
                axis.yaxis.set_major_locator(LogLocator(base=10.0))
                axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_ylim(floor, top)
        axis.set_title(label_of(cohort), pad=4)
        if index >= 2:
            axis.set_xlabel("Age (days)")
        if index % 2 == 0 or mode == "log":
            axis.set_ylabel("Whole weight (g)")

        if mode == "linear":
            # An observation whose marker sits within HIDDEN_FRACTION of the axis
            # span from zero cannot be told from the axis line at this figure size.
            limit = HIDDEN_FRACTION * (top - floor)
            ordered = block_train.sort_values(AGE_COL)["_w"].to_numpy(float)
            hidden = int(np.sum(ordered < limit))
            hidden_rows.append({
                "cohort": label_of(cohort), "hidden": hidden,
                "through": f"{hidden}번째" if hidden else "-",
                "limit_g": limit,
                "largest_hidden_g": float(ordered[:hidden].max()) if hidden else np.nan,
                "observed_max_g": float(observed.max()),
                "observed_span_pct": 100.0 * float(observed.max()) / (top - floor)})

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    seen, keep_handles, keep_labels = set(), [], []
    for handle, text in zip(handles, labels):
        if text not in seen:
            seen.add(text); keep_handles.append(handle); keep_labels.append(text)
    figure.legend(keep_handles, keep_labels, loc="outside lower center", ncol=3,
                  frameon=False)
    save(figure, name)

    if mode == "log":
        print(f"    {name}: 밴드 하한이 축 아래로 잘린 격자점 {clipped_total}개 "
              f"(전체 {len(curve)}개) — 캡션에 명시할 것")
    else:
        frame = pd.DataFrame(hidden_rows)
        print(f"    {name}: y 범위 [0, {ceiling:g}] g"
              + (f", 밴드가 상한을 넘어 잘린 격자점 {clipped_total}개" if clip
                 else ", 밴드 전체 포함"))
        print(f"    (b) 축선에 묻히는 관측 (기준: 축 범위의 "
              f"{HIDDEN_FRACTION*100:.1f}% = {frame.limit_g.iloc[0]:.3f} g 미만)")
        for _, row in frame.iterrows():
            print(f"          {row.cohort:16s} train 앞 {row.through:>6s} 관측이 묻힘 "
                  f"(가장 큰 값 {row.largest_hidden_g:.4f} g)"
                  if row.hidden else
                  f"          {row.cohort:16s} 묻히는 관측 없음")
        print(f"    (c) 관측이 차지하는 세로 영역: "
              + ", ".join(f"{r.cohort} {r.observed_span_pct:.1f}%"
                          for _, r in frame.iterrows())
              + f" (전체 최대 {frame.observed_span_pct.max():.1f}%)")


# =============================================================================
# Package wiring
#
# NEW code, not carried over.  It connects the sections above to one set of
# checkpoints, selects the published columns into the paper tables, checks them
# against the manuscript and builds the zip.  It fits nothing except the one
# composite curve the figure code reads, and computes no model quantity; the only
# values it derives itself are the data summary in Table 1 and the per-facility
# MAPE in by_facility.csv.
# =============================================================================
METHOD = "integral"
# 0905_abalone5.assemble stops unless the unconstrained fit of every kernel in
# ALL_KERNELS is in its states file.  That list was the thirteen-kernel census the
# figures were first explored against; neither published figure draws an
# unconstrained fit, so the three stacking kernels are what is provided.
ALL_KERNELS = KERNELS
KERNEL_GROUP: dict = {}
# The job_seed 0905_abalone4 gave the monotone composite
# (mono|integral|ln_weight_yj|Matern0.5+RBF_add3+NNKernel|step0); 0905_abalone5
# refitted its curve with the same seed.
COMPOSITE_SEED = 7000710057

# Outputs are kept apart per point summary; the fits are not, because no fit
# depends on which summary is reported.
VERSION_DIR = OUT / f"point_{POINT_SUMMARY}"
TABLE_DIR = VERSION_DIR / "table"
FIGURE_DIR = OUT / f"point_{POINT_SUMMARY}" / "figures"
FIGURE_PREFIX = f"abalone_{POINT_SUMMARY}"
FIGURE_CHECK = CKPT
ABLATION_OUT = OUT / f"point_{POINT_SUMMARY}" / "ablation_detail"
ABLATION_CHECK = CKPT
ABLATION_PREFIX = "ablation"
GROWTH_OUT = OUT / f"point_{POINT_SUMMARY}" / "growth_detail"
GROWTH_PREFIX = "growth"
DOCS_DIR = SCRIPT_DIR / "abalone_paper_results"
PACKAGE_ZIP = SCRIPT_DIR / f"abalone_paper_package_{POINT_SUMMARY}.zip"
PACKAGE_DOCS = ["README.md", "requirements.txt", "environment.yml"]
PACKAGE_TABLES = ["table1_facilities", "table2_component_gp", "table3_comparators",
                  "table4_kernel_ablation", "table5_environmental_ablation",
                  "table6_weight_ablation", "table7_constraint_arms",
                  "table8_constraint_confounders", "table9_intervals",
                  "growth_curve_parameters", "by_facility", "residual_normality"]
PACKAGE_FIGURES = ["fig_intervals_p80", "fig_growth_curves_shared_p80", "fig_residual_qq"]

# What 0914_abalone1 did at import: register the age-only kernel, then route the
# Riihimaki-Vehtari derivative blocks of RBF_add3 and NNKernel.  This module is
# both the kernel registry and the derivative module.
register_age_only(sys.modules[__name__])
patch_rv(sys.modules[__name__], sys.modules[__name__])


def set_point_summary(value: str) -> None:
    """Switch the point prediction, and with it every output path."""
    global POINT_SUMMARY, REFERENCE, VERSION_DIR, TABLE_DIR, FIGURE_DIR, FIGURE_PREFIX
    global ABLATION_OUT, GROWTH_OUT, PACKAGE_ZIP
    if value not in ("mean", "median"):
        raise ValueError(f"point summary must be 'mean' or 'median', not {value!r}")
    POINT_SUMMARY = value
    REFERENCE = REFERENCE_BY_SUMMARY[value]
    VERSION_DIR = OUT / f"point_{value}"
    TABLE_DIR, FIGURE_DIR = VERSION_DIR / "table", VERSION_DIR / "figures"
    ABLATION_OUT = VERSION_DIR / "ablation_detail"
    GROWTH_OUT = VERSION_DIR / "growth_detail"
    FIGURE_PREFIX = f"abalone_{value}"
    PACKAGE_ZIP = PACKAGE_ZIP.parent / f"abalone_paper_package_{value}.zip"


def set_output_root(root) -> None:
    """Point every output and checkpoint path at ROOT."""
    global OUT, CKPT, FIGURE_CHECK, ABLATION_CHECK, PACKAGE_ZIP
    OUT = Path(root)
    CKPT = OUT / f"_checkpoints_S{MC_DRAWS}"
    FIGURE_CHECK = ABLATION_CHECK = CKPT
    PACKAGE_ZIP = OUT / f"abalone_paper_package_{POINT_SUMMARY}.zip"
    set_point_summary(POINT_SUMMARY)


def load_previous() -> dict:
    """The fit records the ablation and figure sections look up, under the keys the
    original scripts used.  There they were read from the 0905_abalone1/3/4/5
    checkpoints; here each one is the fit `stage_fits` made with the same seed:

        loo|integral|RESPONSE|k|p        <- loo|k|p   stacking objective, anchor seeds
        cploo|integral|RESPONSE|k|p      <- loo|k|p   conformal LOO; its test-row
                                                      predictive is named target_*
        final|integral|RESPONSE|k|step0  <- final|k
        curve|integral|RESPONSE|k|step0  <- final|k   curve on the integration grid
        screen|RESPONSE|k|step0,
        plain|RESPONSE|k|step0           <- plain|k   unconstrained GP

    Every record is passed on whole, so the loo / cploo entries carry loo_kind
    and loo_quantiles (and a backfilled record's loo_backfill_* fields) to
    loo_mixture_summary; the last line of fits.jsonl for a key is the one read.
    """
    records = FigureStore.read_only(CKPT / "fits.jsonl")
    previous = {}
    for kernel in KERNELS:
        for position in range(N_TRAIN):
            record = records[f"loo|{kernel}|{position}"]
            previous[f"loo|{METHOD}|{RESPONSE}|{kernel}|{position}"] = record
            previous[f"cploo|{METHOD}|{RESPONSE}|{kernel}|{position}"] = {
                **record, "target_mean_z": record["test_mean_z"],
                "target_sd_z": record["test_sd_z"]}
        final, plain = records[f"final|{kernel}"], records[f"plain|{kernel}"]
        previous[f"final|{METHOD}|{RESPONSE}|{kernel}|step0"] = final
        previous[f"curve|{METHOD}|{RESPONSE}|{kernel}|step0"] = final
        previous[f"screen|{RESPONSE}|{kernel}|step0"] = plain
        previous[f"plain|{RESPONSE}|{kernel}|step0"] = plain
    return previous


def stage_figure_fits(context, ablation_context) -> None:
    """Write the states file assemble_figures reads.

    The three curves and three unconstrained fits are the stacking fits already
    made; the monotone composite is fitted here, with the call order of
    0905_abalone5's state worker -- the test rows, then the integration grid.
    """
    path = FIGURE_CHECK / f"{FIGURE_PREFIX}_states.jsonl"
    key = f"curve|{METHOD}|{RESPONSE}|{BEST_COMPOSITE}|step0"
    composite = FigureStore.read_only(path).get(key)
    if composite is None or composite.get("fit_failed") \
            or int(composite.get("job_seed", -1)) != COMPOSITE_SEED:
        print(f"  fitting {BEST_COMPOSITE}", flush=True)
        item = {"method": METHOD, "response": RESPONSE, "kernel": BEST_COMPOSITE,
                "position": None, "monotone": True, "extra_rows": 0, "step": 0,
                "role": "integral", "job_index": 0, "fit_key": key,
                "job_seed": COMPOSITE_SEED}
        record = run_fit_backend(item, ablation_context["ctx"])
        state = record.pop("state", None)
        record.pop("response_obj", None)
        if state is None or record["fit_failed"]:
            raise RuntimeError(f"{key} failed: {record.get('fit_failure_reason')}")
        predictive, _ = state["predict"](context["target"])
        record.update({"test_mean_z": [float(v) for v in predictive.mean_z],
                       "test_sd_z": [float(v) for v in predictive.sd_z],
                       "test_quantiles": quantile_table(predictive).tolist()
                       if predictive.kind == "draws" else None})
        curve_predictive, _ = state["predict"](figure_grid_frame(context))
        record.update({
            "curve_mean_z": [float(v) for v in curve_predictive.mean_z],
            "curve_sd_z": [float(v) for v in curve_predictive.sd_z],
            "curve_quantiles": quantile_table(curve_predictive).tolist()
            if curve_predictive.kind == "draws" else None,
            "rate": [float(v) for v in np.asarray(state["rate"], float)]})
        composite = strip(record)
    else:
        print("  already checkpointed")
    records = FigureStore.read_only(CKPT / "fits.jsonl")
    lines = []
    for kernel in KERNELS:
        lines.append({**records[f"plain|{kernel}"],
                      "fit_key": f"plain|{RESPONSE}|{kernel}|step0"})
        lines.append({**records[f"final|{kernel}"],
                      "fit_key": f"curve|{METHOD}|{RESPONSE}|{kernel}|step0"})
    lines.append(composite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in lines:
            handle.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")


# -----------------------------------------------------------------------------
# Paper tables
# -----------------------------------------------------------------------------
METRICS = ["g_RMSE", "g_MAE", "g_MAPE", "g_R2_pred"]
REGION = {"완도": "Wando", "진도": "Jindo"}


def table1_facilities() -> pd.DataFrame:
    """Section 2.1: the four nursery facilities and what was sampled from each."""
    data, train, test, cohorts = prepare()
    rows = []
    for cohort in cohorts:
        block = data[data[COHORT_COL].astype(str).eq(cohort)]
        tr = train[train[COHORT_COL].astype(str).eq(cohort)].sort_values(AGE_COL)
        te = test[test[COHORT_COL].astype(str).eq(cohort)].sort_values(AGE_COL)
        region = str(block["지역"].iloc[0]).strip() if "지역" in block else ""
        stocked = (", ".join(sorted(pd.to_datetime(block["입식일"]).dt.strftime("%Y-%m-%d")
                                    .unique())) if "입식일" in block else "")
        rows.append({
            "facility": ROMAN.get(cohort, cohort), "region": REGION.get(region, region),
            "stocking_date": stocked,
            "train_occasions": int(tr[TIME_COL].nunique()),
            "test_occasions": int(te[TIME_COL].nunique()),
            "train_dates": ", ".join(pd.to_datetime(tr[TIME_COL]).dt.strftime("%Y-%m-%d")),
            "test_dates": ", ".join(pd.to_datetime(te[TIME_COL]).dt.strftime("%Y-%m-%d")),
            "train_age_days": f"{int(tr[AGE_COL].min())}-{int(tr[AGE_COL].max())}",
            "test_age_days": ", ".join(str(int(a)) for a in te[AGE_COL]),
            "train_weight_g": f"{tr['_w'].iloc[0]:.4f}-{tr['_w'].iloc[-1]:.3f}",
            "test_weight_g": ", ".join(f"{v:.3f}" for v in te["_w"])})
    return pd.DataFrame(rows)


def by_facility_table(context, parts, growth) -> pd.DataFrame:
    """Test MAPE per facility, one row per model (the g_MAPE formula of metrics_of,
    on the same from_z(point_z(.)) point prediction)."""
    response, truth, target = context["response"], context["truth_g"], context["target"]

    def point_of(component):
        return response.from_z(point_z(component))

    points = {"Stacking": point_of(parts["mixture"])}
    for kernel in KERNELS:
        points[KERNEL_LABEL[kernel]] = point_of(parts["singles"][kernel])
    for model, label in (("vonBertalanffy", "von Bertalanffy"), ("Gompertz", "Gompertz")):
        points[label] = growth["predictions"][model]
    ml, _ = fit_ml_models(context["ctx"], target, RESPONSE, 0)
    for name in ML_MODEL_NAMES:
        points[ML_LABEL[name]] = point_of(ml[name])
    cohort = target[COHORT_COL].astype(str).to_numpy()
    rows = []
    for model, point in points.items():
        row = {"model": model}
        for name in context["cohorts"]:
            mask = cohort == name
            row[ROMAN.get(name, name)] = float(
                np.mean(np.abs(point[mask] - truth[mask]) / truth[mask]) * 100.0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    # the growth curves' rows must equal what 0915_abalone1's own loop wrote
    source = growth["by_facility"].pivot(index="model", columns="facility", values="g_MAPE")
    for model, label in (("vonBertalanffy", "von Bertalanffy"), ("Gompertz", "Gompertz")):
        mine = frame.set_index("model").loc[label, source.columns].to_numpy(float)
        assert np.allclose(mine, source.loc[model].to_numpy(float), rtol=0, atol=1e-10)
    return frame


def residual_normality(context, parts):
    """Section 3.1: the 24 training residuals of the full-data fit, per kernel.

    residual_i = (z_i - latent posterior mean at x_i) / sigma_n, on the final
    transformed scale, standardised by the fitted observation-noise sd.  The latent
    posterior mean is train_latent_mean_z, the mean of the S draws of f without the
    observation noise -- NOT train_mean_z, the predictive mean, which averages
    f + eps and so carries the MC noise of eps.  The training ages are
    integration-grid nodes, so the construction predicts at them directly.  The
    residual is still MC-estimated over f: its Shapiro-Wilk p moves in the third
    decimal with the number of draws (0.534 -> 0.517 for Matern(1/2) between
    S = 2,000 and S = 10,000).
    """
    from scipy.stats import shapiro
    store, response = parts["store"], context["response"]
    z = response.to_z(context["train"]["_w"].to_numpy(float))
    rows, residuals = [], {}
    for kernel in KERNELS:
        record = store[f"final|{kernel}"]
        if "train_latent_mean_z" not in record:
            raise RuntimeError(
                f"final|{kernel} carries no train_latent_mean_z: this checkpoint "
                "predates the latent-mean residual check, and stage_fits did not refit it.")
        noise = float(record["noise"])
        residual = (z - np.asarray(record["train_latent_mean_z"], float)) / noise
        statistic, p_value = shapiro(residual)
        residuals[kernel] = residual
        rows.append({"kernel": KERNEL_LABEL[kernel], "n": int(len(residual)),
                     "noise_sd_z": noise, "residual_mean": float(residual.mean()),
                     "residual_sd": float(residual.std(ddof=1)),
                     "shapiro_W": float(statistic), "shapiro_p": float(p_value)})
    return pd.DataFrame(rows), residuals


# Manuscript Figure 4: one colour and one marker per facility (Okabe-Ito), the
# grey reference envelope and the dashed y = x line.
FACILITY_STYLE = {"완도_금해": ("#0072B2", "o"), "완도_민국": ("#D55E00", "s"),
                  "진도_가반": ("#009E73", "^"), "진도_달우": ("#CC79A7", "D")}
QQ_ENVELOPE = "#E1E0D9"
QQ_REFERENCE = "#898781"
QQ_MARKER_AREA = 34.0


def figure_qq(context, normality, residuals) -> None:
    """Figure 4: one normal Q-Q panel per kernel for the 24 standardised training
    residuals of residual_normality, coloured and shaped by facility.

    The residuals and the Shapiro-Wilk W and p are residual_normality's, unchanged.
    x is the Blom plotting position Phi^-1((k - 3/8)/(n + 1/4)).  The grey band is
    the POINTWISE 95% reference envelope of standard-normal order statistics,
    Phi^-1 of the 2.5% and 97.5% quantiles of Beta(k, n + 1 - k): a reference for
    each order statistic on its own, not a simultaneous band and not a predictive
    interval of the model."""
    table = normality.set_index("kernel")
    # residual i belongs to training row i (residual_normality reads train in order)
    facility = context["train"][COHORT_COL].astype(str).to_numpy()
    n = len(facility)
    k = np.arange(1, n + 1)
    theoretical = ndtri((k - 0.375) / (n + 0.25))
    lower = norm.ppf(beta.ppf(0.025, k, n + 1 - k))
    upper = norm.ppf(beta.ppf(0.975, k, n + 1 - k))
    # one range for the three panels that holds every point and the whole envelope
    x_limit = 4.0 / 3.0 * float(np.max(np.abs(theoretical)))
    y_limit = 1.25 * max([float(np.max(np.abs(np.concatenate([lower, upper]))))]
                         + [float(np.max(np.abs(residuals[kernel]))) for kernel in KERNELS])

    figure, axes = plt.subplots(1, len(KERNELS), figsize=(8.4, 3.7), sharex=True,
                                sharey=True, layout="constrained")
    for index, (axis, kernel) in enumerate(zip(np.atleast_1d(axes), KERNELS)):
        # sort the residuals and carry each one's facility with the same index
        order = np.argsort(np.asarray(residuals[kernel], float), kind="mergesort")
        values = np.asarray(residuals[kernel], float)[order]
        values_facility = facility[order]
        axis.fill_between(theoretical, lower, upper, color=QQ_ENVELOPE, alpha=0.8,
                          linewidth=0, zorder=1)
        axis.axline((0.0, 0.0), slope=1.0, color=QQ_REFERENCE, linewidth=1.1,
                    linestyle="--", zorder=2)
        for cohort in context["cohorts"]:
            colour, marker = FACILITY_STYLE[cohort]
            mask = values_facility == cohort
            axis.scatter(theoretical[mask], values[mask], s=QQ_MARKER_AREA, marker=marker,
                         color=colour, edgecolor="white", linewidth=0.6, zorder=3)
        row = table.loc[KERNEL_LABEL[kernel]]
        axis.set_title(f"({'abc'[index]}) {KERNEL_LABEL[kernel]}", loc="left", fontsize=12)
        axis.annotate(f"Shapiro–Wilk (n = {int(row.n)})\n"
                      f"W = {row.shapiro_W:.3f}, p = {row.shapiro_p:.3f}",
                      (0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
                      fontsize=10, zorder=4,
                      bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                            "edgecolor": QQ_ENVELOPE, "linewidth": 1.0})
        axis.set_xlim(-x_limit, x_limit)
        axis.set_ylim(-y_limit, y_limit)
    np.atleast_1d(axes)[0].set_ylabel("Standardised residual", fontsize=12)
    figure.supxlabel("Theoretical standard normal quantile", fontsize=12)
    handles = [Line2D([], [], linestyle="none", marker=FACILITY_STYLE[c][1],
                      markersize=math.sqrt(QQ_MARKER_AREA),
                      markerfacecolor=FACILITY_STYLE[c][0], markeredgecolor="white",
                      markeredgewidth=0.6, label=label_of(c))
               for c in context["cohorts"]]
    handles += [Patch(facecolor=QQ_ENVELOPE, alpha=0.8, edgecolor="none",
                      label="95% pointwise reference envelope"),
                Line2D([], [], color=QQ_REFERENCE, linewidth=1.1, linestyle="--",
                       label="Reference line (y = x)")]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
                  fontsize=10)
    save(figure, "fig_residual_qq")
    print(f"    {len(KERNELS)} panels, {n} training residuals each, standardised by the "
          f"fitted noise sd; facility colours and markers, 95% pointwise reference "
          f"envelope of the standard-normal order statistics")


def paper_tables(context, parts, models, ablation, intervals, growth, e3, confounders):
    tables = {"table1_facilities": table1_facilities()}

    block = models[models.Table.eq("Table 2")]
    tables["table2_component_gp"] = block[["Model", "Weight", *METRICS]].rename(
        columns={"Model": "model", "Weight": "weight"}).reset_index(drop=True)

    growth_rows = growth["table"][["model", "inputs", "fitted", *METRICS, "footnote"]]
    ml = models[models.Block.eq("baseline")]
    ml_rows = pd.DataFrame({"model": ml.Model, "inputs": "age + environment + facility",
                            "fitted": "pooled", **{c: ml[c] for c in METRICS},
                            "footnote": ""})
    table3 = pd.concat([growth_rows, ml_rows], ignore_index=True)
    notes = growth["footnotes"]
    table3["footnote_text"] = [
        "; ".join(f"{n.facility}: {n.reason}" for n in notes[notes["mark"] == mark].itertuples())
        if mark else "" for mark in table3.footnote]
    tables["table3_comparators"] = table3

    tables["table4_kernel_ablation"] = ablation[
        ["Model", "Removed", "weights", "expH", *METRICS, "delta_g_MAPE",
         "delta_g_MAE"]].rename(columns={"Model": "model", "Removed": "removed"})

    e1 = pd.read_csv(ablation_out("01", "E1_age_only_models"))
    e1_cp = pd.read_csv(ablation_out("01d", "E1_intervals")).set_index("inputs")
    e1 = e1[["inputs", "model", "weight", *METRICS, "expH"]].copy()
    stacking = e1.model.eq("Stacking")
    e1["cp80_covered"] = [f"{int(e1_cp.loc[i, 'cp_covered'])}/8" if s else ""
                          for i, s in zip(e1.inputs, stacking)]
    e1["cp80_width_g"] = [float(e1_cp.loc[i, "cp_width_g"]) if s else np.nan
                          for i, s in zip(e1.inputs, stacking)]
    tables["table5_environmental_ablation"] = e1

    e2 = pd.read_csv(ablation_out("02", "E2_equal_weights"))
    e2 = e2[e2.weights.isin(["LOO-optimal", "equal"])].reset_index(drop=True)
    tables["table6_weight_ablation"] = pd.DataFrame({
        "weights": e2.weights, "w": e2.w, **{c: e2[c] for c in METRICS},
        "mixture_loo_lpd_g": e2.mixture_loo_lpd_g,
        "cp80_covered": [f"{int(v)}/8" for v in e2.cp_covered],
        "cp80_width_g": e2.cp_width_g})

    tables["table7_constraint_arms"] = e3[
        ["support", "arm", "kernels", "weights", "expH", *METRICS, "down_train",
         "down_test", "audit_grid"]].reset_index(drop=True)
    tables["table8_constraint_confounders"] = confounders

    rows = intervals[intervals.Facility.ne("ALL")].copy()
    rows["Native_covers"] = rows.Native_covers.astype(bool)
    rows["Conformal_covers"] = rows.Conformal_covers.astype(bool)
    total = intervals[intervals.Facility.eq("ALL")].iloc[0]
    summary = []
    for occasion in ("t7", "t8"):
        part = rows[rows.Occasion.eq(occasion)]
        summary.append({"Facility": f"mean {occasion}", "Occasion": occasion,
                        "Native_width_g": float(part.Native_width_g.mean()),
                        "Conformal_width_g": float(part.Conformal_width_g.mean()),
                        "Width_reduction_pct": float(part.Width_reduction_pct.mean())})
    summary.append({"Facility": "ALL", "Occasion": "",
                    "Native_width_g": float(total.Native_width_g),
                    "Conformal_width_g": float(total.Conformal_width_g),
                    "Width_reduction_pct": float(total.Width_reduction_pct),
                    "Native_covers": f"{int(rows.Native_covers.sum())}/8",
                    "Conformal_covers": f"{int(rows.Conformal_covers.sum())}/8"})
    tables["table9_intervals"] = pd.concat([rows, pd.DataFrame(summary)], ignore_index=True)

    params = growth["params"]
    keep = ["model", "facility", "converged", "sse", "sigma_g"]
    for spec in MODELS.values():
        for name in spec["names"]:
            keep += [name, f"{name}_se", f"{name}_rel_se", f"{name}_at_bound"]
    keep += ["cond_JTJ_scaled", "Winf_over_max_train", "monotone_signs_ok", "positive_ok",
             "test_increasing", "dense_down_train", "dense_down_test", "instability"]
    tables["growth_curve_parameters"] = params[[c for c in dict.fromkeys(keep)
                                                if c in params.columns]]

    tables["by_facility"] = by_facility_table(context, parts, growth)
    return tables


# -----------------------------------------------------------------------------
# Reproduction check against the manuscript
# -----------------------------------------------------------------------------
# Published entries that differ from the computation by a known rounding or
# typesetting slip, checked against the stated value instead:
# {(group, item): (expected, why)}.  There are none.
MANUSCRIPT_ERRATA: dict = {}


# The values the manuscript prints, per point summary, at S = 2000, as printed.
# A computed value matches when it rounds to the printed digits.  A key that is
# absent or None is not pinned: it is computed and reported, but not compared.
# Only the median version is published, so nothing is pinned for "mean"; a mean
# run reports the median value of every item beside its own.
MANUSCRIPT_VALUES = {
    "median": {
        "Table 2|Stacking g_RMSE": "1.6363",
        "Table 2|Stacking g_MAE": "1.2852",
        "Table 2|Stacking g_MAPE": "14.2875",
        "Table 2|Stacking g_R2_pred": "0.9552",
        "Table 2|RBF_add3 g_RMSE": "1.6891",
        "Table 2|RBF_add3 g_MAE": "1.3807",
        "Table 2|RBF_add3 g_MAPE": "15.0548",
        "Table 2|RBF_add3 g_R2_pred": "0.9522",
        "Table 2|NNKernel g_RMSE": "1.8292",
        "Table 2|NNKernel g_MAE": "1.4622",
        "Table 2|NNKernel g_MAPE": "16.8381",
        "Table 2|NNKernel g_R2_pred": "0.9440",
        "Table 2|Matern(1/2) g_RMSE": "3.7423",
        "Table 2|Matern(1/2) g_MAE": "2.8679",
        "Table 2|Matern(1/2) g_MAPE": "27.3997",
        "Table 2|Matern(1/2) g_R2_pred": "0.7655",
        "Table 2|RBF_add3 weight": "0.4361",
        "Table 2|NNKernel weight": "0.4074",
        "Table 2|Matern(1/2) weight": "0.1565",
        "Table 3|von Bertalanffy g_RMSE": "2.5365",
        "Table 3|von Bertalanffy g_MAE": "2.3094",
        "Table 3|von Bertalanffy g_MAPE": "23.2569",
        "Table 3|von Bertalanffy g_R2_pred": "0.8923",
        "Table 3|Gompertz g_RMSE": "3.7218",
        "Table 3|Gompertz g_MAE": "3.4059",
        "Table 3|Gompertz g_MAPE": "36.5780",
        "Table 3|Gompertz g_R2_pred": "0.7680",
        "Table 3|Bayesian ridge g_RMSE": "4.8324",
        "Table 3|Bayesian ridge g_MAE": "3.8225",
        "Table 3|Bayesian ridge g_MAPE": "36.1085",
        "Table 3|Bayesian ridge g_R2_pred": "0.6089",
        "Table 3|Random forest g_RMSE": "5.7878",
        "Table 3|Random forest g_MAE": "5.3725",
        "Table 3|Random forest g_MAPE": "53.3253",
        "Table 3|Random forest g_R2_pred": "0.4390",
        "Table 3|XGBoost g_RMSE": "5.7462",
        "Table 3|XGBoost g_MAE": "5.1809",
        "Table 3|XGBoost g_MAPE": "50.6192",
        "Table 3|XGBoost g_R2_pred": "0.4470",
        "Table 4|Stacking (all three) exp(H)": "2.7676",
        "Table 4|Stacking (all three) g_RMSE": "1.6363",
        "Table 4|Stacking (all three) g_MAE": "1.2852",
        "Table 4|Stacking (all three) g_MAPE": "14.2875",
        "Table 4|Stacking (all three) g_R2_pred": "0.9552",
        "Table 4|Without Matérn(ν = 1/2) exp(H)": "1.8756",
        "Table 4|Without Matérn(ν = 1/2) g_RMSE": "1.6812",
        "Table 4|Without Matérn(ν = 1/2) g_MAE": "1.3297",
        "Table 4|Without Matérn(ν = 1/2) g_MAPE": "15.3060",
        "Table 4|Without Matérn(ν = 1/2) g_R2_pred": "0.9527",
        "Table 4|Without Group-additive SE exp(H)": "1.5337",
        "Table 4|Without Group-additive SE g_RMSE": "1.6886",
        "Table 4|Without Group-additive SE g_MAE": "1.3266",
        "Table 4|Without Group-additive SE g_MAPE": "15.2493",
        "Table 4|Without Group-additive SE g_R2_pred": "0.9523",
        "Table 4|Without Neural network exp(H)": "1.6760",
        "Table 4|Without Neural network g_RMSE": "1.9189",
        "Table 4|Without Neural network g_MAE": "1.6180",
        "Table 4|Without Neural network g_MAPE": "16.8461",
        "Table 4|Without Neural network g_R2_pred": "0.9383",
        "E1|age-only Stacking g_RMSE": "2.2362",
        "E1|age-only Stacking g_MAE": "1.8544",
        "E1|age-only Stacking g_MAPE": "18.9343",
        "E1|age-only Stacking g_R2_pred": "0.9163",
        "E1|age-only Stacking expH": "1.7826",
        "E1|weight Matern0.5 full": "0.1565",
        "E1|weight Matern0.5 age-only": "0.2649",
        "E1|weight RBF_add3 full": "0.4361",
        "E1|weight RBF_add3 age-only": "0.7351",
        "E1|weight NNKernel full": "0.4074",
        "E1|weight NNKernel age-only": "0.0000",
        "E1|CP coverage full": "7/8",
        "E1|CP coverage age-only": "6/8",
        "E1|CP width full": "4.8244",
        "E1|CP width age-only": "5.3385",
        "E2|equal weights g_RMSE": "1.8181",
        "E2|equal weights g_MAE": "1.4762",
        "E2|equal weights g_MAPE": "15.5634",
        "E2|equal weights g_R2_pred": "0.9446",
        "E2|equal weights mixture_loo_lpd_g": "-0.0070",
        "E2|equal weights CP coverage": "6/8",
        "E3|(a) g_MAPE": "14.2875",
        "E3|(a) test decreasing steps": "0/32",
        "E3|(b) g_MAPE": "32.6366",
        "E3|(b) test decreasing steps": "41/196",
        "E3|(c) g_MAPE": "30.1010",
        "E3|(c) test decreasing steps": "0/196",
        "E3|(d) g_MAPE": "48.5878",
        "E3|(d) test decreasing steps": "146/196",
        "Intervals|native mean width g": "6.5722",
        "Intervals|native coverage": "8/8",
        "Intervals|conformal mean width g": "4.8244",
        "Intervals|conformal coverage": "7/8",
        "Intervals|width reduction %": "26.6",
        "Intervals|t7 mean reduction %": "11.5",
        "Intervals|t8 mean reduction %": "35.4",
        "Section 3.1|Matern0.5 Shapiro-Wilk W": "0.964",
        "Section 3.1|Matern0.5 Shapiro-Wilk p": "0.534",
        "Section 3.1|RBF_add3 Shapiro-Wilk W": "0.956",
        "Section 3.1|RBF_add3 Shapiro-Wilk p": "0.365",
        "Section 3.1|NNKernel Shapiro-Wilk W": "0.950",
        "Section 3.1|NNKernel Shapiro-Wilk p": "0.270",
    },
    "mean": {},
}


def reproduction_targets(tables, e3, intervals):
    """(group, item, printed, computed, median) for every number the manuscript quotes.

    `printed` is what this point summary must print, or None when the value is not
    pinned -- those are reported as pending, not as mismatches.  `median` is the
    published (median-version) value, so a run under another point summary shows
    exactly what changes.
    """
    pinned = MANUSCRIPT_VALUES[POINT_SUMMARY]
    median_values = MANUSCRIPT_VALUES["median"]
    t2 = tables["table2_component_gp"].set_index("model")
    t3 = tables["table3_comparators"].set_index("model")
    t4 = tables["table4_kernel_ablation"].reset_index(drop=True)
    e1 = tables["table5_environmental_ablation"]
    e1_weights = pd.read_csv(ablation_out("01c", "E1_weights"))
    e2 = tables["table6_weight_ablation"].set_index("weights")
    normality = tables["residual_normality"].set_index("kernel")
    arms = e3[e3.support.eq("3-kernel")]
    rows = intervals[intervals.Facility.ne("ALL")]
    total = intervals[intervals.Facility.eq("ALL")].iloc[0]
    items = []

    def add(group, item, value):
        key = f"{group}|{item}"
        items.append((group, item, pinned.get(key), value, median_values.get(key)))

    for c in METRICS:
        add("Table 2", f"Stacking {c}", t2.loc["Stacking", c])
    for kernel, short in (("RBF_add3", "RBF_add3"), ("NNKernel", "NNKernel"),
                          ("Matern0.5", "Matern(1/2)")):
        label = KERNEL_LABEL[kernel]
        add("Table 2", f"{short} weight", t2.loc[label, "weight"])
        for c in METRICS:
            add("Table 2", f"{short} {c}", t2.loc[label, c])
    for label in ("von Bertalanffy", "Gompertz", ML_LABEL["BayesianRidge"],
                  ML_LABEL["RandomForest"], ML_LABEL["XGBoost"]):
        for c in METRICS:
            add("Table 3", f"{label} {c}", t3.loc[label, c])
    for i in range(len(t4)):
        add("Table 4", f"{t4.model[i]} exp(H)", t4.expH[i])
        for c in METRICS:
            add("Table 4", f"{t4.model[i]} {c}", t4[c][i])
    full = e1[e1.inputs.eq("full") & e1.model.eq("Stacking")].iloc[0]
    age = e1[e1.inputs.eq("age-only") & e1.model.eq("Stacking")].iloc[0]
    for c in [*METRICS, "expH"]:
        add("E1", f"age-only Stacking {c}", age[c])
    for kernel in KERNELS:
        pick = e1_weights[e1_weights.kernel.eq(kernel)].set_index("inputs").weight
        add("E1", f"weight {kernel} full", pick["full"])
        add("E1", f"weight {kernel} age-only", pick["age-only"])
    add("E1", "CP coverage full", full.cp80_covered)
    add("E1", "CP coverage age-only", age.cp80_covered)
    add("E1", "CP width full", full.cp80_width_g)
    add("E1", "CP width age-only", age.cp80_width_g)
    for c in [*METRICS, "mixture_loo_lpd_g"]:
        add("E2", f"equal weights {c}", e2.loc["equal", c])
    add("E2", "equal weights CP coverage", e2.loc["equal", "cp80_covered"])
    for arm in ("(a)", "(b)", "(c)", "(d)"):
        row = arms[arms.arm.str.startswith(arm)].iloc[0]
        add("E3", f"{arm} g_MAPE", row.g_MAPE)
        add("E3", f"{arm} test decreasing steps", row.down_test)
    add("Intervals", "native coverage", f"{int(rows.Native_covers.astype(bool).sum())}/8")
    add("Intervals", "native mean width g", total.Native_width_g)
    add("Intervals", "conformal coverage",
        f"{int(rows.Conformal_covers.astype(bool).sum())}/8")
    add("Intervals", "conformal mean width g", total.Conformal_width_g)
    add("Intervals", "width reduction %", total.Width_reduction_pct)
    add("Intervals", "t7 mean reduction %", rows[rows.Occasion.eq("t7")].Width_reduction_pct.mean())
    add("Intervals", "t8 mean reduction %", rows[rows.Occasion.eq("t8")].Width_reduction_pct.mean())
    for kernel in KERNELS:
        label = KERNEL_LABEL[kernel]
        add("Section 3.1", f"{kernel} Shapiro-Wilk W", normality.loc[label, "shapiro_W"])
        add("Section 3.1", f"{kernel} Shapiro-Wilk p", normality.loc[label, "shapiro_p"])
    return items


def at_printed_precision(value, printed: str) -> str:
    # a count ("7/8", "41/196") is text under either setting; everything else is
    # rounded to as many decimals as the manuscript prints
    if isinstance(value, str) or "/" in str(printed):
        return str(value).replace(" ", "")
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    text = f"{float(value):.{decimals}f}"
    return text.lstrip("-") if float(text) == 0.0 else text


def reproduction_check(tables, e3, intervals, figure_frame) -> pd.DataFrame:
    rows = []
    for group, item, printed, value, median in reproduction_targets(tables, e3, intervals):
        pending = printed is None
        reference = (median or "0.0001") if pending else printed   # digits to print at
        computed = at_printed_precision(value, reference)
        expected, note = MANUSCRIPT_ERRATA.get((group, item), (printed, ""))
        rows.append({"group": group, "item": item,
                     "manuscript": "pending" if pending else printed,
                     "median": "" if median == printed else (median or ""),
                     "computed": computed,
                     "full_precision": value if isinstance(value, str) else float(value),
                     "match": True if pending else computed == expected,
                     "note": note or ("not pinned for this point summary" if pending else "")})
    if figure_frame is not None:
        # the bands the interval figure draws are the intervals Table 9 reports
        block = figure_frame[np.isclose(figure_frame.Nominal, NOMINAL)].set_index("Method")
        total = intervals[intervals.Facility.eq("ALL")].iloc[0]
        for method, column in (("native", "Native_width_g"),
                               (BAND_METHOD, "Conformal_width_g")):
            figure_value = float(block.loc[method, "Width_g"])
            gap = abs(figure_value - float(total[column]))
            rows.append({"group": "Figure = Table 9", "item": f"{method} mean width g",
                         "manuscript": f"{float(total[column]):.6f}", "median": "",
                         "computed": f"{figure_value:.6f}", "full_precision": gap,
                         "match": gap < 1e-9, "note": "full_precision is |figure - table|"})
    return pd.DataFrame(rows)


def write_run_documents() -> None:
    """The package docs are generated here, so the single .py stays sufficient."""
    requirements = ("numpy==2.5.4\npandas==3.1.0\nscipy==1.18.1\nscikit-learn==1.9.1\n"
                    "matplotlib>=3.8\nopenpyxl\ntorch==2.5.1\ngpytorch==1.15.2\nxgboost==3.4.1\n")
    point = "from_z(mean_z)" if POINT_SUMMARY == "mean" else "from_z(ppf(0.5))"
    readme = f"""# Abalone GP growth prediction — reproduction package

Monotonicity-constrained stacked Gaussian processes with conformal calibration.
This package reports the predictive {POINT_SUMMARY} as the point prediction.

Contents:
- code/abalone_paper_code.py: the single reproduction script.
- code/requirements.txt, code/environment.yml: dependencies.
- table/: Tables 1-9 of the Results section, growth_curve_parameters.csv,
  by_facility.csv and residual_normality.csv (section 3.1).
- figures/: fig_intervals_p80, fig_growth_curves_shared_p80 and
  fig_residual_qq, as PDF and 300-dpi PNG.

Install the dependencies in requirements.txt (or
`conda env create -f environment.yml`), place `전복데이터.xlsx` beside the
script or in `input_data/`, and run `python abalone_paper_code.py`.
Input data are not distributed in this package.
- `--workers N`: parallel fits (default 3).
- `--fits-only`: run the fits and stop.
- `--tables-only`: tables and the reproduction check; no figures, no zip.
- `--point mean|median`: point prediction (default median).

The run makes 354 GP fits, three ML baselines and 2 x 4 growth-curve fits,
about 25 minutes at three workers. Fits are checkpointed in
`abalone_paper_results/_checkpoints_S{MC_DRAWS}/` beside the script, so an
interrupted run resumes and a second run only rebuilds tables and figures.
The fits do not depend on the point summary and are shared by both.

The point prediction is the median of the stacking mixture on the modelling
scale, mapped back to grams ({point}). Every RMSE, MAE, MAPE and R2_pred, the
growth-figure line and the centre of the conformal interval use it. The
stacking weights come from the Yao et al. (2018) objective on the pointwise
full LOO densities and do not depend on the point summary.

The {NOMINAL:.0%} interval is gamma = 0 conformal: one radius on the z scale,
the (n+1)-adjusted quantile of |truth_z - LOO mixture median| over the 24
training rows. Each LOO mixture uses the full-LOO stacking weights, and its
median is found by inverting the mixture CDF. The radius is added to and
subtracted from the full-data mixture median, then mapped back to grams. This
is not Jackknife+. Jackknife+ and other gamma values are computed only as
comparisons; none reaches a table or figure.

Every fit is seeded with the value the original run recorded. Bitwise
reproduction holds on the reference machine (Intel Xeon E5-1660 v4; torch
2.5.1+cu121, gpytorch 1.15.2, OMP_NUM_THREADS=1, float64). On other hardware
or builds, agreement to about 1e-6 is expected.

Outputs, in `abalone_paper_results/point_{POINT_SUMMARY}/` beside the script:
- table/ and figures/: as packaged here.
- ablation_detail/ and growth_detail/: supplementary analyses.
- reproduction_check.csv: computed values versus the manuscript, compared at
  the printed precision. A mismatch is recorded with match = False; it does
  not stop the run or the package.
- run_metadata.csv: settings, stacking weights and software versions.
"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "README.md").write_text(readme, encoding="utf-8")
    (DOCS_DIR / "requirements.txt").write_text(requirements, encoding="utf-8")
    (DOCS_DIR / "environment.yml").write_text(
        "name: abalone-paper\nchannels:\n  - conda-forge\ndependencies:\n"
        "  - python=3.12.16\n  - pip\n  - pip:\n"
        + "".join(f"      - {line}\n" for line in requirements.splitlines()),
        encoding="utf-8")


def build_zip() -> Path:
    write_run_documents()
    PACKAGE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__).resolve(), "code/abalone_paper_code.py")
        for name in PACKAGE_DOCS:
            archive.write(DOCS_DIR / name, f"code/{name}")
        for name in PACKAGE_TABLES:
            archive.write(TABLE_DIR / f"{name}.csv", f"table/{name}.csv")
        for name in PACKAGE_FIGURES:
            for extension in ("pdf", "png"):
                archive.write(FIGURE_DIR / f"{FIGURE_PREFIX}_{name}.{extension}",
                              f"figures/{name}.{extension}")
        names = archive.namelist()
    leaked = [n for n in names if n.lower().endswith((".xlsx", ".xls"))
              or (n.lower().endswith(".csv") and not n.startswith("table/"))]
    if leaked:
        raise RuntimeError(f"data file in the package: {leaked}")
    return PACKAGE_ZIP


# =============================================================================
# Main
# =============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel fits (default 3)")
    parser.add_argument("--fits-only", action="store_true",
                        help="run the fits and stop")
    parser.add_argument("--tables-only", action="store_true",
                        help="tables and the reproduction check; no figures, no zip")
    parser.add_argument("--point", choices=("mean", "median"), default=POINT_SUMMARY,
                        help="which predictive summary is the point prediction "
                             f"(default {POINT_SUMMARY})")
    return parser.parse_args()


def main() -> None:
    global DATA_PATH
    arguments = parse_arguments()
    set_point_summary(arguments.point)
    found = next((path for path in (DATA_PATH, SCRIPT_DIR / "input_data" / DATA_PATH.name)
                  if path.exists()), None)
    if found is None:
        raise SystemExit(
            f"{DATA_PATH.name} not found.  Place the workbook beside this script "
            f"or in {SCRIPT_DIR / 'input_data'}.")
    DATA_PATH = found
    for path in (OUT, CKPT, VERSION_DIR, TABLE_DIR, FIGURE_DIR, ABLATION_OUT, GROWTH_OUT):
        path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    banner("Setting")
    print(f"  monotonicity  positive-growth integral construction")
    print(f"  response      {RESPONSE}  (ln weight -> Yeo-Johnson -> standardise)")
    print(f"  kernels       {', '.join(KERNEL_LABEL[k] for k in KERNELS)}")
    print(f"  intervals     nominal {NOMINAL:.0%}, gamma = 0 conformal radius from "
          f"|truth_z - LOO mixture {POINT_SUMMARY}|,\n                the LOO mixture "
          f"weighted by the full-LOO stacking weights")
    print(f"  point         {POINT_SUMMARY} of the predictive, from_z("
          f"{'mean_z' if POINT_SUMMARY == 'mean' else 'ppf(0.5)'}) -- metrics, the "
          f"conformal centre,\n                the drawn curve and its audit all use it; "
          f"the native interval keeps\n                the model's own quantiles")
    print(f"  draws         S = {MC_DRAWS}")
    print(SEED_DEPENDENCE_NOTE)

    context = build_context()
    ablation_context = build_ablation_context()
    print(f"  data          {len(context['train'])} train rows, "
          f"{len(context['test'])} test rows, {len(context['cohorts'])} facilities")

    stage_fits(context, arguments.workers)
    stage_ablation_fits(ablation_context, arguments.workers)
    banner("Figure state — monotone composite (1 fit)")
    stage_figure_fits(context, ablation_context)
    if arguments.fits_only:
        print(f"\nfits done in {time.perf_counter() - started:.0f} s")
        return

    parts = assemble(context)
    banner("Stacking weights")
    for kernel, weight in zip(KERNELS, parts["stack"]["weights"]):
        print(f"  {KERNEL_LABEL[kernel]:22s} {weight:.4f}")
    print(f"  effective number of components exp(H) = {parts['stack']['expH']:.3f}")
    models = table_models(context, parts)
    ablation = table_ablation(context, parts)
    intervals = table_intervals(context, parts)

    growth = growth_curve_run()

    banner("Section 3.1 — normality of the training residuals")
    normality, residuals = residual_normality(context, parts)
    show(normality, ["kernel", "n", "noise_sd_z", "residual_mean", "residual_sd",
                     "shapiro_W", "shapiro_p"], digits=4)

    previous = load_previous()
    mine = AblationStore(ABLATION_CHECK / f"{ABLATION_PREFIX}_fits.jsonl", append=False).records
    arms, status = gather(ablation_context, previous, mine)
    print("\nablation arms:", status)
    incomplete = [(arm, state) for arm, state in status.items() if state != "complete"]
    if incomplete:
        raise RuntimeError(f"ablation arms incomplete: {incomplete}")
    stacks = table_e1(ablation_context, arms)
    table_e2(ablation_context, arms, stacks)
    e3 = table_e3(ablation_context, arms, status)
    confounders = table_confounders(arms, status, mine)

    figure_frame = None
    if not arguments.tables_only:
        banner(f"Figures — nominal {NOMINAL:.0%}")
        figure_parts = assemble_figures(context)
        figure_frame = table_intervals_methods(context, figure_parts)
        figure_intervals(context, figure_parts, figure_frame, level=NOMINAL)
        growth_panels(context, figure_parts, mode="log", level=NOMINAL, shared_log=True,
                      name=f"fig_growth_curves_shared{suffix_for(NOMINAL)}", report=True)
        figure_qq(context, normality, residuals)

    banner("Paper tables")
    tables = paper_tables(context, parts, models, ablation, intervals, growth, e3, confounders)
    tables["residual_normality"] = normality
    for name, frame in tables.items():
        frame.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding=ENC)
        print(f"  {name}.csv  ({len(frame)} rows)")

    banner("Reproduction check against the published values")
    check = reproduction_check(tables, e3, intervals, figure_frame)
    check.to_csv(VERSION_DIR / "reproduction_check.csv", index=False, encoding=ENC)
    with pd.option_context("display.width", 200):
        print(check[["group", "item", "manuscript", "computed", "match"]].to_string(index=False))
    print(f"\n  {int(check.match.sum())} / {len(check)} match at the printed precision")
    for (group, item), (_, note) in MANUSCRIPT_ERRATA.items():
        print(f"  published erratum — {group} {item}: {note}")
    pending = check[check.manuscript.eq("pending")]
    if len(pending):
        print(f"\n  {len(pending)} values are not pinned for the {POINT_SUMMARY} point "
              f"summary; they are computed and reported only (median = published value):")
        print(pending[["group", "item", "median", "computed"]].to_string(index=False))
    if not check.match.all():
        # Recorded in reproduction_check.csv with match = False; a gap from the
        # published values does not stop the run or the package.
        print(f"\n  MISMATCH — {int((~check.match).sum())} values differ from the published "
              f"values (recorded in reproduction_check.csv; outputs and the package are "
              f"still built):")
        print(check[~check.match].to_string(index=False))

    pd.DataFrame([{
        "combo": COMBO, "response": RESPONSE, "nominal": NOMINAL,
        "point_summary": POINT_SUMMARY,
        "conformity_score": f"|truth_z - LOO mixture {POINT_SUMMARY}|, full-LOO stacking "
                            f"weights for every row",
        "reproduction_match": f"{int(check.match.sum())}/{len(check)}",
        "weights": ", ".join(f"{k}:{w:.6f}" for k, w in zip(KERNELS, parts["stack"]["weights"])),
        "expH": parts["stack"]["expH"], "mixture_loo_lpd_g": parts["stack"]["mixture_lpd"],
        "seed": SEED, "mc_draws": MC_DRAWS, "torch": torch.__version__,
        "gpytorch": gpytorch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    }]).to_csv(VERSION_DIR / "run_metadata.csv", index=False, encoding=ENC)

    if not arguments.tables_only:
        banner("Package")
        print(f"  {build_zip()}")
    print(f"\nAll outputs written to {VERSION_DIR}")
    print(f"Total {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
