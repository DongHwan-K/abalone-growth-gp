"""Manuscript Figure 3: monthly individual log-weight distributions by facility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eda_common import reproduce_figure  # noqa: E402

if __name__ == "__main__":
    reproduce_figure(3)
