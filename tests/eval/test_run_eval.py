# tests/eval/test_run_eval.py
import src.eval.run_eval as run_eval_mod


def test_evaluate_computes_mean_metrics_and_records_run(monkeypatch):
    cases = [
        {"question": "q1", "relevant_chunk_ids": ["A"]},
        {"question": "q2", "relevant_chunk_ids": ["Z"]},
    ]

    def fake_search(question, k=5):
        return [{"chunk_id": cid} for cid in ["A", "B", "C", "D", "E"]]

    recorded = {}

    def fake_record_run(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(run_eval_mod, "record_run", fake_record_run)

    metrics = run_eval_mod.evaluate(search_fn=fake_search, k=5, config_hash="test-cfg", cases=cases)

    # q1: "A" is rank 1 -> recall=1, rr=1; q2: "Z" not retrieved -> recall=0, rr=0
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr"] == 0.5
    assert recorded["config_hash"] == "test-cfg"
    assert recorded["recall_at_5"] == 0.5


def test_sweep_runs_every_config_k_pair_and_records_each(monkeypatch):
    cases = [{"question": "q1", "relevant_chunk_ids": ["A"]}]

    def fake_search(question, k=5):
        return [{"chunk_id": cid} for cid in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]]

    recorded = []

    def fake_record_run(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(run_eval_mod, "record_run", fake_record_run)

    results = run_eval_mod.sweep({"fake": fake_search}, k_values=[1, 5], cases=cases)

    assert set(results.keys()) == {("fake", 1), ("fake", 5)}
    assert len(recorded) == 2
    # k=1: "A" still rank-1 hit -> recall=1.0 at k=1 too
    assert results[("fake", 1)]["recall_at_5"] == 1.0
    # legacy columns only populated for the k=5 call
    k1_call = next(r for r in recorded if r["k"] == 1)
    k5_call = next(r for r in recorded if r["k"] == 5)
    assert k1_call["recall_at_5"] is None
    assert k5_call["recall_at_5"] == 1.0
    assert k1_call["recall_at_k"] == 1.0
    assert k1_call["k"] == 1
