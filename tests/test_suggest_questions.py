# tests/test_suggest_questions.py
import scripts.suggest_questions as sq_mod


def _chunk(ticker, section, i):
    return {"chunk_id": f"{ticker}_{section}_{i:03d}", "ticker": ticker, "section": section, "text": "x"}


def test_stratified_sample_covers_every_group_before_repeating():
    chunks = []
    for ticker in ("AAA", "BBB"):
        for section in ("Remarks", "Tables"):
            for i in range(3):
                chunks.append(_chunk(ticker, section, i))

    # 4 groups x 3 chunks each = 12 total; sample 4 (one per group)
    sampled = sq_mod.stratified_sample(chunks, n=4, seed=1)

    groups_seen = {(c["ticker"], c["section"]) for c in sampled}
    assert len(sampled) == 4
    assert len(groups_seen) == 4  # every group represented before any group repeats


def test_stratified_sample_is_deterministic_for_a_given_seed():
    chunks = [_chunk("AAA", "Remarks", i) for i in range(5)] + [_chunk("BBB", "Tables", i) for i in range(5)]

    first = sq_mod.stratified_sample(chunks, n=6, seed=7)
    second = sq_mod.stratified_sample(chunks, n=6, seed=7)

    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]


def test_stratified_sample_never_returns_more_than_requested_or_available():
    chunks = [_chunk("AAA", "Remarks", i) for i in range(2)]

    assert len(sq_mod.stratified_sample(chunks, n=10, seed=0)) == 2
    assert len(sq_mod.stratified_sample(chunks, n=1, seed=0)) == 1
