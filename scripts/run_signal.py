# scripts/run_signal.py
"""CLI: generate a signal for one ticker, judge its faithfulness, print both.

Usage:
    python scripts/run_signal.py AAPL
    python scripts/run_signal.py AAPL --quarter 2026Q2
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.faithfulness import judge
from src.generation.generate import generate_signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and judge a trading signal for one ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--quarter", default=None, help="e.g. 2026Q2 -- restricts retrieval to that filing")
    args = parser.parse_args()

    signal, chunks = generate_signal(args.ticker, fiscal_quarter=args.quarter)
    print(json.dumps(signal.model_dump(), indent=2))

    score, claims = judge(signal.reasoning, chunks)
    print(f"\nfaithfulness: {score:.2%}")
    for c in claims:
        print(f"  [{c['label']}] {c['claim']}")
