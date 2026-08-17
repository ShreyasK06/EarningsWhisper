from src.generation.prompts import build_user_prompt


def test_build_user_prompt_includes_chunk_ids_and_sections():
    chunks = [{"chunk_id": "AAPL_1", "section": "PreparedRemarks", "text": "revenue grew"}]
    prompt = build_user_prompt("AAPL", chunks)
    assert "AAPL_1" in prompt
    assert "PreparedRemarks" in prompt
    assert "revenue grew" in prompt
    assert "AAPL" in prompt
