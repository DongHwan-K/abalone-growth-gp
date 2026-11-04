"""Manuscript Figure 4: normal Q-Q plots of the standardised training residuals."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper_outputs import reproduce_figure  # noqa: E402

if __name__ == "__main__":
    reproduce_figure(4)
