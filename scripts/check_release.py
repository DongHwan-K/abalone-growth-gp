#!/usr/bin/env python3
"""Release checks that need only the Python standard library.

    python scripts/check_release.py                    # check the release tree
    python scripts/check_release.py --write-checksums  # rewrite SHA256SUMS.txt
    python scripts/check_release.py --build-zip OUT.zip  # build, then check, the ZIP
    python scripts/check_release.py --check-zip OUT.zip

The release tree is the repository minus private data, checkpoints, generated
work folders, caches and temporary files (the same files .gitignore keeps out).
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import re
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = "91ae5a6beebf89815ad638ac26f4ec86ebc214684a67dd1326bb5a24051408fb"
SUMS = "SHA256SUMS.txt"
ZIP_PREFIX = "abalone-growth-gp"

# Reviewed synthetic inputs only; a same-name private replacement is rejected.
EXAMPLE_WORKBOOKS = {
    "input_data/전복데이터.xlsx":
        "0a3224d84c470f79711533506a17d0c78bbbd30a7a69a5cb90a9f415cb550e2f",
    "input_data/전복 4개세트통합데이터.xlsx":
        "d6d360148f66090fadb5a1e57831a556d15d6b041c8bd990a04d35e03b9717a1",
}

FIGURES = [
    "Figure1_environmental_variables_by_site",
    "Figure2_correlation_heatmap_logweight_environment",
    "Figure3_monthly_log_weight_distributions",
    "Figure4_residual_qq",
    "Figure5_intervals_p80",
    "Figure6_growth_curves_shared_p80",
]
TABLES = [
    "Table1_facilities",
    "Table2_component_gp",
    "Table3_comparators",
    "Table4_kernel_ablation",
    "Table5_framework_variants",
]
TEXTS = [
    "Section3_values",
]
REQUIRED = [
    *EXAMPLE_WORKBOOKS,
    "README.md", "CITATION.cff", "LICENSE", "requirements.txt", "environment.yml",
    "abalone_median.py", "reproduce_all.py", "input_data/README.md",
    "scripts/__init__.py", "scripts/paper_outputs.py", "scripts/eda_common.py",
    "scripts/check_release.py", ".github/workflows/validate.yml",
    "docs/REPRODUCIBILITY.md", "docs/FIGURE_PROVENANCE.md",
    "results/README.md", "results/manifest.csv",
    "results/validation/manuscript_comparison.csv",
    "results/validation/run_checks.csv", "results/validation/figure_checks.csv",
    "results/validation/table_comparison.csv", "results/validation/table_mismatches.csv",
    *[f"scripts/figures/Figure{n}.py" for n in range(1, 7)],
    *[f"scripts/tables/Table{n}.py" for n in range(1, 6)],
    "scripts/text/Section3.py",
    *[f"results/figures/{name}.{ext}" for name in FIGURES for ext in ("pdf", "png")],
    *[f"results/tables/{name}.csv" for name in TABLES],
    *[f"results/text/{name}.csv" for name in TEXTS],
]

# Never part of a release (directories end with "/").
EXCLUDED_DIRS = [".git/", "abalone_paper_results/", "__pycache__/", ".matplotlib/",
                 ".pytest_cache/", ".mypy_cache/", ".ipynb_checkpoints/", ".venv/", "venv/",
                 ".idea/", ".vscode/"]
EXCLUDED_FILES = ["*.xlsx", "*.xls", "*.pyc", "*.pyo", ".DS_Store", "*.zip", "~$*", "*.tmp",
                  "*.bak", "*.swp", "*.orig", ".env", ".Rhistory"]
# Present in the release tree means a leak or a leftover.
FORBIDDEN = ["*.xlsx", "*.xls", "*.jsonl", "_checkpoints_S*", "*.pyc", "__pycache__",
             "Final EDA*.py", "abalone_transform_comparison*.py", "*(1)*", "*(2)*",
             "fig_*.pdf", "fig_*.png", "table[0-9]*_*.csv", "abalone_median_fig_*"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded(relative: str) -> bool:
    relative = unicodedata.normalize("NFC", relative)
    if relative in EXAMPLE_WORKBOOKS:
        return False
    parts = PurePosixPath(relative).parts
    for directory in EXCLUDED_DIRS:
        if directory.rstrip("/") in parts[:-1] or relative.startswith(directory):
            return True
    if parts[0] == "input_data" and parts[-1] not in ("README.md", ".gitkeep"):
        return True
    return any(fnmatch.fnmatchcase(parts[-1], pattern) for pattern in EXCLUDED_FILES)


def release_files() -> list[str]:
    files = []
    for path in ROOT.rglob("*"):
        if path.is_file() or path.is_symlink():
            relative = unicodedata.normalize("NFC", path.relative_to(ROOT).as_posix())
            if not excluded(relative):
                files.append(relative)
    return sorted(files)


def forbidden_names(names) -> list[str]:
    bad = []
    for name in names:
        name = unicodedata.normalize("NFC", name)
        if name in EXAMPLE_WORKBOOKS:
            continue
        parts = PurePosixPath(name).parts
        if any(fnmatch.fnmatchcase(part, pattern) for part in parts for pattern in FORBIDDEN):
            bad.append(name)
        elif name.startswith("results/supplementary/"):
            bad.append(name)
    return bad


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def broken_links(files: list[str]) -> list[str]:
    broken = []
    for name in files:
        if not name.endswith(".md"):
            continue
        text = (ROOT / name).read_text(encoding="utf-8")
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in MARKDOWN_LINK.findall(text):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            resolved = (ROOT / name).parent / path
            if not resolved.exists():
                broken.append(f"{name} -> {target}")
            elif resolved.resolve().is_relative_to(ROOT.resolve()) and \
                    resolved.is_file() and excluded(resolved.resolve().relative_to(
                        ROOT.resolve()).as_posix()):
                broken.append(f"{name} -> {target} (not part of the release)")
    return broken


def read_sums() -> dict[str, str]:
    entries = {}
    for line in (ROOT / SUMS).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip().removeprefix("./")] = digest
    return entries


def write_sums() -> None:
    files = [name for name in release_files() if name != SUMS]
    lines = [f"{sha256(ROOT / name)}  ./{name}" for name in files]
    (ROOT / SUMS).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {SUMS} ({len(lines)} files).")


def check_tree() -> list[str]:
    problems = []
    for name, expected in EXAMPLE_WORKBOOKS.items():
        path = ROOT / name
        if path.is_file() and sha256(path) != expected:
            problems.append(f"example workbook SHA-256 differs: {name}; "
                            "only reviewed synthetic data may be released")
    source = sha256(ROOT / "abalone_median.py")
    if source != EXPECTED_SOURCE_SHA256:
        problems.append(f"abalone_median.py SHA-256 {source} != {EXPECTED_SOURCE_SHA256}")

    files = release_files()
    problems += [f"missing: {name}" for name in REQUIRED if name not in files]
    problems += [f"must not be released: {name}" for name in forbidden_names(files)]
    unexpected = [name for name in files if name.startswith("results/")
                  and not (name.startswith(("results/figures/Figure", "results/tables/Table",
                                            "results/text/Section", "results/validation/"))
                           or name in ("results/README.md", "results/manifest.csv"))]
    problems += [f"unexpected result file: {name}" for name in unexpected]

    # Workbooks belong in input_data/ only, and nothing private or generated may be
    # tracked by git, even where the ignore rules would keep it out of a fresh add.
    stray = sorted(path.relative_to(ROOT).as_posix()
                   for pattern in ("*.xlsx", "*.xls") for path in ROOT.rglob(pattern))
    problems += [f"workbook outside input_data/: {name}" for name in stray
                 if not name.startswith(("input_data/", "abalone_paper_results/", ".git/",
                                         ".venv/", "venv/"))]
    if (ROOT / ".git").exists():
        try:
            output = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True,
                                    capture_output=True, text=True).stdout
            tracked = [name for name in output.split("\0") if name]
        except (OSError, subprocess.CalledProcessError):
            tracked = []
        problems += [f"tracked by git but private or generated: {name}" for name in tracked
                     if excluded(name) or forbidden_names([name])]

    for name in files:
        if name.startswith("results/") and name.endswith(".csv"):
            rows = read_csv(ROOT / name)
            header_only = name == "results/validation/table_mismatches.csv"
            if not rows or not rows[0] or (len(rows) < 2 and not header_only):
                problems.append(f"empty or malformed CSV: {name}")
    for name in ("manuscript_comparison", "run_checks", "figure_checks", "table_comparison"):
        path = ROOT / "results" / "validation" / f"{name}.csv"
        if path.is_file():
            rows = read_csv(path)
            column = rows[0].index("passed") if rows and "passed" in rows[0] else None
            failed = [row for row in rows[1:] if column is None or row[column] != "True"]
            if failed:
                problems.append(f"results/validation/{name}.csv has {len(failed)} failed rows")
    mismatches = ROOT / "results" / "validation" / "table_mismatches.csv"
    if mismatches.is_file() and len(read_csv(mismatches)) > 1:
        problems.append("results/validation/table_mismatches.csv lists cell mismatches")

    manifest = ROOT / "results" / "manifest.csv"
    if manifest.is_file():
        rows = read_csv(manifest)
        header = rows[0]
        listed = set()
        for row in rows[1:]:
            entry = dict(zip(header, row))
            listed.add(entry["file"])
            path = ROOT / entry["file"]
            if not path.is_file():
                problems.append(f"manifest lists a missing file: {entry['file']}")
            elif sha256(path) != entry["sha256"]:
                problems.append(f"manifest SHA-256 differs: {entry['file']}")
        expected = {f"results/figures/{n}.{e}" for n in FIGURES for e in ("pdf", "png")}
        expected |= {f"results/tables/{n}.csv" for n in TABLES}
        expected |= {f"results/text/{n}.csv" for n in TEXTS}
        problems += [f"manifest omits {name}" for name in sorted(expected - listed)]

    problems += [f"broken Markdown link: {link}" for link in broken_links(files)]

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in ("*.xlsx", "abalone_paper_results/", "_checkpoints_S*/", "__pycache__/"):
        if rule not in gitignore:
            problems.append(f".gitignore lacks {rule}")

    if not (ROOT / SUMS).is_file():
        problems.append(f"missing {SUMS}")
    else:
        sums = read_sums()
        listed = set(sums)
        actual = set(files) - {SUMS}
        problems += [f"{SUMS} omits {name}" for name in sorted(actual - listed)]
        problems += [f"{SUMS} lists a file outside the release: {name}"
                     for name in sorted(listed - actual)]
        problems += [f"{SUMS} digest differs: {name}" for name in sorted(actual & listed)
                     if sha256(ROOT / name) != sums[name]]
    return problems


def build_zip(target: Path) -> None:
    target = target.resolve()
    if target.is_relative_to(ROOT.resolve()):
        raise SystemExit("Write the ZIP outside the repository.")
    files = release_files()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            info = zipfile.ZipInfo.from_file(ROOT / name, f"{ZIP_PREFIX}/{name}")
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (ROOT / name).read_bytes())
    print(f"Wrote {target} ({len(files)} files).")


def check_zip(target: Path) -> list[str]:
    problems = []
    expected = set(release_files())
    sums = read_sums() if (ROOT / SUMS).is_file() else {}
    with zipfile.ZipFile(target) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        outside = [n for n in names if not n.startswith(f"{ZIP_PREFIX}/")]
        problems += [f"ZIP entry outside {ZIP_PREFIX}/: {n}" for n in outside]
        inner = {n[len(ZIP_PREFIX) + 1:]: n for n in names if n.startswith(f"{ZIP_PREFIX}/")}
        problems += [f"ZIP must not contain: {n}" for n in forbidden_names(inner)]
        problems += [f"ZIP contains an excluded file: {n}" for n in inner if excluded(n)]
        problems += [f"ZIP lacks: {n}" for n in sorted(expected - set(inner))]
        problems += [f"ZIP has a file outside the release: {n}"
                     for n in sorted(set(inner) - expected)]
        for name, entry in inner.items():
            data = archive.read(entry)
            if name in sums and sha256_bytes(data) != sums[name]:
                problems.append(f"ZIP copy differs from {SUMS}: {name}")
            if name in expected and data != (ROOT / name).read_bytes():
                problems.append(f"ZIP copy differs from the tree: {name}")
        if archive.testzip() is not None:
            problems.append("ZIP CRC check failed")
    if not problems:
        print(f"ZIP checks passed: {target} ({len(inner)} files, "
              f"SHA-256 {sha256(target)}).")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-checksums", action="store_true",
                        help=f"rewrite {SUMS} from the release tree, then check")
    parser.add_argument("--build-zip", type=Path, help="build the release ZIP, then check it")
    parser.add_argument("--check-zip", type=Path, help="check an existing release ZIP")
    arguments = parser.parse_args()
    if arguments.write_checksums:
        write_sums()
    problems = check_tree()
    if not problems:
        print(f"Release checks passed ({len(release_files())} files).")
    if arguments.build_zip and not problems:
        build_zip(arguments.build_zip)
        problems += check_zip(arguments.build_zip)
    if arguments.check_zip:
        problems += check_zip(arguments.check_zip)
    if problems:
        raise SystemExit("Release checks failed:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
