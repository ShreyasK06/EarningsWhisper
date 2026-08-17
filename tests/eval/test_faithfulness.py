from src.eval.faithfulness import score_claims


def test_score_claims_computes_supported_fraction():
    claims = [
        {"claim": "a", "label": "supported"},
        {"claim": "b", "label": "unsupported"},
        {"claim": "c", "label": "supported"},
        {"claim": "d", "label": "contradicted"},
    ]
    assert score_claims(claims) == 0.5


def test_score_claims_empty_list_is_zero():
    assert score_claims([]) == 0.0


def test_score_claims_all_supported_is_one():
    claims = [{"claim": "a", "label": "supported"}, {"claim": "b", "label": "supported"}]
    assert score_claims(claims) == 1.0
