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
