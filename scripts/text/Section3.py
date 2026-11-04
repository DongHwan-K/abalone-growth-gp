"""Manuscript Section 3: the numbers quoted in the Results text."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper_outputs import reproduce_text  # noqa: E402

if __name__ == "__main__":
    reproduce_text(3)
