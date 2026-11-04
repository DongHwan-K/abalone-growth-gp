"""Shared data handling and styling for manuscript Figures 1-3.

Three plotting functions (``plot_environment_by_site``,
``plot_correlation_heatmap`` and ``plot_monthly_log_weight_distributions``)
draw the figures; saving is separated from drawing so every figure is written
as a PDF and a 300-dpi PNG under its manuscript number.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_NAME = "전복 4개세트통합데이터.xlsx"
SHEET_NAME = "dataset_d_spat_2024"
REQUIRED_COLUMNS = ["날짜", "품종", "전중(g)", "수온", "DO", "PH", "염분"]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures"

FIGURE_STEMS = {
    1: "Figure1_environmental_variables_by_site",
    2: "Figure2_correlation_heatmap_logweight_environment",
    3: "Figure3_monthly_log_weight_distributions",
}

TARGET_SITES = ["완도_금해", "완도_민국", "진도_가반", "진도_달우"]
SITE_COLORS = {
    "완도_금해": "#0072B2",
    "완도_민국": "#D55E00",
    "진도_가반": "#009E73",
    "진도_달우": "#CC79A7",
}
SITE_LINESTYLES = {
    "완도_금해": "-",
    "완도_민국": (0, (5, 2)),
    "진도_가반": (0, (1, 1)),
    "진도_달우": (0, (5, 1, 1, 1)),
}
ENV_COLUMNS = ["DO", "수온", "PH", "염분"]
ENV_LABELS = {
    "DO": "DO (mg L$^{-1}$)",
    "수온": "Temperature (°C)",
    "PH": "pH",
    "염분": "Salinity (psu)",
}
SITE_LABELS = {
    "완도_금해": "Wando-Geumhae",
    "완도_민국": "Wando-Mingug",
    "진도_가반": "Jindo-Gaban",
    "진도_달우": "Jindo-Daru",
}
CORRELATION_ORDER = ["log_weight", "DO", "Temperature", "Salinity", "pH"]

# Font sizes of manuscript Figs. 4 and 6 (the rcParams of abalone_median.py).
PAPER_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
}
PAPER_DPI = 300

# Rounded Pearson correlations of the manuscript's Figure 2.  Used only by
# check_correlations(); the figure itself is always computed from the data.
REFERENCE_CORRELATIONS = {
    ("log_weight", "DO"): -0.041,
    ("log_weight", "Temperature"): 0.259,
    ("log_weight", "Salinity"): 0.442,
    ("log_weight", "pH"): 0.197,
    ("DO", "Temperature"): -0.622,
    ("DO", "Salinity"): 0.310,
    ("DO", "pH"): 0.465,
    ("Temperature", "Salinity"): -0.281,
    ("Temperature", "pH"): -0.365,
    ("Salinity", "pH"): 0.847,
}


def paper_style():
    """Matplotlib defaults plus PAPER_RC, independent of any global rcParams
    (importing abalone_median.py, for example, turns on axes.grid)."""
    return plt.style.context(["default", PAPER_RC])


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
def missing_input(message: str) -> None:
    print(message, file=sys.stderr)
    print(
        f"\nFigures 1-3 need the workbook '{DATA_NAME}', sheet '{SHEET_NAME}',\n"
        f"in {REPO_ROOT / 'input_data'} or {REPO_ROOT} (or pass --data / --sheet).\n"
        f"Required columns: {', '.join(REQUIRED_COLUMNS)}\n"
        "No substitute data are generated.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def resolve_data(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        missing_input(f"Input workbook not found: {candidate}")
    for candidate in (REPO_ROOT / "input_data" / DATA_NAME, REPO_ROOT / DATA_NAME):
        if candidate.is_file():
            return candidate.resolve()
    missing_input("Input workbook not found.")


def load_data(path: str | Path | None = None, sheet: str = SHEET_NAME) -> pd.DataFrame:
    resolved = resolve_data(path)
    try:
        df = pd.read_excel(resolved, sheet_name=sheet)
    except ValueError as error:
        missing_input(f"Cannot read sheet '{sheet}' from {resolved}: {error}")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_input(f"{resolved} [{sheet}] lacks the columns: {', '.join(missing)}")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df[df["품종"].isin(TARGET_SITES)].copy()
    print(f"Input: {resolved} [{sheet}], {len(df)} rows from {df['품종'].nunique()} facilities")
    return df


# -----------------------------------------------------------------------------
# Data summaries drawn by the figures
# -----------------------------------------------------------------------------
def environment_means(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1: date x facility means of the four environmental variables."""
    return (
        df.groupby(["날짜", "품종"], as_index=False)[ENV_COLUMNS]
        .mean()
        .sort_values(["품종", "날짜"])
    )


def correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2: individual ln(weight), then date x facility means."""
    corr_df = df.dropna(subset=["전중(g)", "DO", "수온", "PH", "염분"]).copy()
    corr_df["log_weight"] = np.log(corr_df["전중(g)"])
    return (
        corr_df.groupby(["날짜", "품종"], as_index=False)[["log_weight", "DO", "수온", "PH", "염분"]]
        .mean()
        .rename(columns={"수온": "Temperature", "염분": "Salinity", "PH": "pH"})
    )


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return correlation_summary(df)[CORRELATION_ORDER].corr()


def monthly_log_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 3: positive individual weights on the log scale, by month."""
    data = df.loc[df["전중(g)"] > 0, ["날짜", "품종", "전중(g)"]].dropna().copy()
    data["log_weight"] = np.log(data["전중(g)"])
    data["month"] = data["날짜"].dt.to_period("M")
    return data


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_environment_by_site(df):
    env_df = environment_means(df)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True, layout="constrained")
    axes = axes.flatten()

    for ax, column in zip(axes, ENV_COLUMNS):
        for site in TARGET_SITES:
            site_df = env_df[env_df["품종"] == site]
            ax.plot(
                site_df["날짜"],
                site_df[column],
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                color=SITE_COLORS[site],
                linestyle=SITE_LINESTYLES[site],
                label=SITE_LABELS[site],
            )

        ax.set_ylabel(ENV_LABELS[column])
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(alpha=0.3)

    for ax in axes[2:]:
        ax.set_xlabel("Measurement date (2024)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    return fig


def plot_correlation_heatmap(df):
    import seaborn as sns

    corr_matrix = correlation_matrix(df)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"fontsize": 10},
        cbar_kws={"shrink": 0.85},
        ax=ax,
    )
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def plot_monthly_log_weight_distributions(df):
    """Individual log-weights by month and site, each with a half-violin KDE.

    The jitter seed is fixed (2024); each half-violin is a ``gaussian_kde``.
    """
    data = monthly_log_weights(df)
    months = sorted(data["month"].unique())
    site_offsets = np.linspace(-0.3, -0.08, len(TARGET_SITES))
    rng = np.random.default_rng(2024)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")

    for month_index, month in enumerate(months):
        for site_index, site in enumerate(TARGET_SITES):
            site_mask = (data["month"] == month) & (data["품종"] == site)
            values = data.loc[site_mask, "log_weight"].to_numpy()
            if len(values) == 0:
                continue
            color = SITE_COLORS[site]

            point_x = (
                month_index
                + site_offsets[site_index]
                + rng.uniform(-0.025, 0.025, len(values))
            )
            ax.scatter(
                point_x,
                values,
                s=10,
                color=color,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.2,
                zorder=3,
            )

            if len(np.unique(values)) <= 1:
                continue
            kde = gaussian_kde(values)
            padding = max(np.std(values) * 0.35, 0.08)
            value_grid = np.linspace(values.min() - padding, values.max() + padding, 240)
            density = kde(value_grid)
            density_width = 0.34 * density / density.max()
            density_base = month_index + 0.03

            ax.fill_betweenx(
                value_grid,
                density_base,
                density_base + density_width,
                color=color,
                alpha=0.1,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                density_base + density_width,
                value_grid,
                color=color,
                linestyle=SITE_LINESTYLES[site],
                linewidth=1.4,
                alpha=0.95,
                zorder=2 + site_index * 0.05,
            )

    month_labels = [month.strftime("%b") for month in months]
    ax.set_xticks(np.arange(len(month_labels)), month_labels)
    ax.set_xlim(-0.45, len(month_labels) - 0.5)
    ax.set_xlabel("Month (2024)")
    ax.set_ylabel("log_weight")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SITE_COLORS[site],
            linestyle=SITE_LINESTYLES[site],
            linewidth=1.4,
            marker="o",
            markersize=4.5,
            markerfacecolor=SITE_COLORS[site],
            markeredgecolor="white",
            label=SITE_LABELS[site],
        )
        for site in TARGET_SITES
    ]
    fig.legend(handles=legend_handles, loc="outside upper center", ncol=4, frameon=False)
    return fig


PLOTTERS = {
    1: plot_environment_by_site,
    2: plot_correlation_heatmap,
    3: plot_monthly_log_weight_distributions,
}


# -----------------------------------------------------------------------------
# Rendering, checks and the command line
# -----------------------------------------------------------------------------
def render_figure(number: int, df: pd.DataFrame, output_dir: str | Path | None = None) -> list[Path]:
    """Draw Figure ``number`` and write it as PDF and 300-dpi PNG."""
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with paper_style():
        fig = PLOTTERS[number](df)
        for suffix in (".pdf", ".png"):
            path = output_dir / f"{FIGURE_STEMS[number]}{suffix}"
            fig.savefig(path, dpi=PAPER_DPI, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths


def figure_text(number: int, df: pd.DataFrame) -> dict:
    """Titles, legend entries and axis labels of Figure ``number`` (drawn, not saved)."""
    with paper_style():
        fig = PLOTTERS[number](df)
        fig.canvas.draw()
        axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        titles = [fig._suptitle.get_text() if fig._suptitle else ""]
        titles += [ax.get_title(loc=loc) for ax in fig.axes for loc in ("left", "center", "right")]
        legends = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
        legends += [text.get_text() for ax in axes if ax.get_legend()
                    for text in ax.get_legend().get_texts()]
        labels = [ax.get_xlabel() for ax in axes] + [ax.get_ylabel() for ax in axes]
        ticks = [tick.get_text() for ax in axes
                 for tick in ax.get_xticklabels() + ax.get_yticklabels()]
        plt.close(fig)
    return {"titles": [t for t in titles if t], "legend": legends, "labels": labels,
            "ticks": ticks}


def check_eda(df: pd.DataFrame, *, compare_published: bool = True) -> list[dict]:
    """Check Figures 1-3 against the manuscript or the current example inputs."""
    rows = []

    def add(figure, check, expected, observed, passed):
        rows.append({"figure": f"Figure {figure}", "check": check, "expected": str(expected),
                     "observed": str(observed), "passed": bool(passed)})

    env = environment_means(df)
    per_site = env.groupby("품종")["날짜"].nunique()
    add(1, "facilities x occasions", "4 x 8",
        f"{env['품종'].nunique()} x {sorted(set(per_site))}",
        env["품종"].nunique() == 4 and set(per_site) == {8} and len(env) == 32)
    add(1, "facility-date rows without missing environment", 32,
        int(env[ENV_COLUMNS].notna().all(axis=1).sum()),
        int(env[ENV_COLUMNS].notna().all(axis=1).sum()) == 32)

    summary = correlation_summary(df)
    add(2, "facility-date means entering the correlation", 32, len(summary), len(summary) == 32)
    matrix = correlation_matrix(df)
    add(2, "variable order", CORRELATION_ORDER, list(matrix.columns),
        list(matrix.columns) == CORRELATION_ORDER)
    annotations = {}
    if not compare_published:
        with paper_style():
            figure = plot_correlation_heatmap(df)
            annotations = {(int(text.get_position()[1] - 0.5),
                            int(text.get_position()[0] - 0.5)): text.get_text()
                           for text in figure.axes[0].texts}
            plt.close(figure)
    for (left, right), reference in REFERENCE_CORRELATIONS.items():
        value = float(matrix.loc[left, right])
        if compare_published:
            add(2, f"r({left}, {right}) at 3 decimals", f"{reference:.3f}", f"{value:.3f}",
                f"{value:.3f}" == f"{reference:.3f}")
        else:
            cell = (matrix.index.get_loc(left), matrix.columns.get_loc(right))
            observed = annotations.get(cell, "missing")
            add(2, f"r({left}, {right}): annotation matches input", f"{value:.3f}", observed,
                np.isfinite(value) and abs(value) <= 1 + 1e-12
                and observed == f"{value:.3f}")

    monthly = monthly_log_weights(df)
    months = sorted(monthly["month"].unique())
    month_text = [m.strftime("%Y-%m") for m in months]
    add(3, "months", "2024-04 ... 2024-11 (8)", f"{month_text[0]} ... {month_text[-1]} ({len(months)})",
        month_text == [f"2024-{m:02d}" for m in range(4, 12)])
    counts = monthly.groupby(["품종", "month"]).size()
    add(3, "individual observations (positive weights)", len(df.loc[df["전중(g)"] > 0]),
        len(monthly), len(monthly) == len(df.loc[df["전중(g)"] > 0]))
    add(3, "observations per facility", "",
        "; ".join(f"{SITE_LABELS[s]} {int(monthly['품종'].eq(s).sum())}" for s in TARGET_SITES),
        True)
    add(3, "facility x month cells (all with >= 2 values for the KDE)", 32,
        f"{len(counts)} (min {int(counts.min())}, max {int(counts.max())})",
        len(counts) == 32 and int(counts.min()) >= 2)

    hyphen_names = [SITE_LABELS[s] for s in TARGET_SITES]
    for number in (1, 2, 3):
        text = figure_text(number, df)
        add(number, "no figure or panel title", "", " | ".join(text["titles"]), not text["titles"])
        if number in (1, 3):
            add(number, "legend order", hyphen_names, text["legend"], text["legend"] == hyphen_names)
        everything = text["legend"] + text["labels"] + text["ticks"]
        bad = [t for t in everything
               if any(s in t for s in TARGET_SITES) or re.search(r"(Wando|Jindo)_", t)]
        add(number, "facility names hyphenated", "no underscore names", bad or "none", not bad)
        if number == 3:
            add(3, "y-axis label", "log_weight", text["labels"][-1],
                text["labels"][-1] == "log_weight")
    return rows


def reproduce_figure(number: int, argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=f"Reproduce manuscript Figure {number}.")
    parser.add_argument("--data", help=f"workbook (default: input_data/{DATA_NAME})")
    parser.add_argument("--sheet", default=SHEET_NAME, help=f"sheet (default: {SHEET_NAME})")
    parser.add_argument("--output-dir", help="output folder (default: results/figures)")
    arguments = parser.parse_args(argv)
    df = load_data(arguments.data, arguments.sheet)
    paths = render_figure(number, df, arguments.output_dir)
    for path in paths:
        print(f"Figure {number}: {path}")
    return paths
