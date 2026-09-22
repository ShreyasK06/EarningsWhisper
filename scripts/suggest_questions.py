# scripts/suggest_questions.py
"""Draft candidate golden-set questions for human labeling.

Samples chunks stratified across ticker and section, asks the LLM to draft
2 candidate questions each chunk would answer, and writes candidates to
data/golden/candidates.jsonl for a human to review and label.

IMPORTANT: these are candidates only. A human must read each one, decide
the real relevant_chunk_id(s) by hand, and add verified entries to
data/golden/eval_set.jsonl in that file's existing format. An AI-drafted
question paired with an AI-assumed answer chunk is circular and would
invalidate the eval set -- this script never writes to eval_set.jsonl.

Usage:
    python scripts/suggest_questions.py --n-chunks 30
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.generation.llm_client import get_client

CANDIDATES_PATH = config.GOLDEN_DIR / "candidates.jsonl"

DRAFT_SYSTEM = """You are drafting evaluation questions for a retrieval system over
earnings-release excerpts. Given one excerpt, write exactly 2 specific,
answerable questions that this excerpt (and ideally only this excerpt)
would answer. Questions should reference concrete facts, numbers, or
statements from the excerpt -- not generic questions.

Return ONLY JSON: {"questions": ["...", "..."]}"""


def load_chunks():
    with (config.PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stratified_sample(chunks: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample roughly evenly across (ticker, section) groups."""
    groups = defaultdict(list)
    for c in chunks:
        groups[(c["ticker"], c["section"])].append(c)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    sampled = []
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    i = 0
    while len(sampled) < n and any(groups[k] for k in group_keys):
        key = group_keys[i % len(group_keys)]
        if groups[key]:
            sampled.append(groups[key].pop())
        i += 1
    return sampled[:n]


def draft_questions_for_chunk(client, chunk: dict) -> list[str]:
    text = client.text_call(DRAFT_SYSTEM, f"EXCERPT:\n{chunk['text']}").strip()
    text = text.removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)["questions"]


def suggest(n_chunks: int = 30):
    chunks = load_chunks()
    sampled = stratified_sample(chunks, n_chunks)
    print(f"Sampled {len(sampled)} chunks (stratified by ticker/section) from {len(chunks)} total")

    client = get_client()
    candidates = []
    for chunk in sampled:
        try:
            questions = draft_questions_for_chunk(client, chunk)
        except Exception as e:
            print(f"  SKIP {chunk['chunk_id']}: {e}")
            continue
        for q in questions:
            candidates.append({
                "question": q,
                "chunk_id": chunk["chunk_id"],
                "ticker": chunk["ticker"],
                "section": chunk["section"],
            })
        print(f"  {chunk['chunk_id']}: {len(questions)} candidate(s)")

    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_PATH.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"\nWrote {len(candidates)} candidates -> {CANDIDATES_PATH}")
    print("Next: review each candidate by hand, verify/correct the relevant chunk_id(s),")
    print("and add confirmed entries to data/golden/eval_set.jsonl in its existing format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft candidate golden-set questions for human labeling.")
    parser.add_argument("--n-chunks", type=int, default=30, help="number of chunks to sample")
    args = parser.parse_args()
    suggest(n_chunks=args.n_chunks)
