"""Manuscript Figure 5: native and residual-calibrated 80% test intervals."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper_outputs import reproduce_figure  # noqa: E402

if __name__ == "__main__":
    reproduce_figure(5)
