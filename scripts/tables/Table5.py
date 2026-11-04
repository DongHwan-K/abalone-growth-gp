"""Manuscript Table 5: framework variants and ablation comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper_outputs import reproduce_table  # noqa: E402

if __name__ == "__main__":
    reproduce_table(5)
