"""CLI: run the retrieval eval and write results to SQLite.

Usage:
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_eval import run_eval

if __name__ == "__main__":
    run_eval()
