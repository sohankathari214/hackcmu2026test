from src.retrieval.similar_events import retrieve_similar_events


def test_retrieval_uses_numeric_non_identifying_features():
    episode = {"patient_id": "patient_001", "pre_action_state": {"glucose": {"current_mg_dl": 145}}, "planned_action": {"action_type": "exercise", "exercise": {"duration_minutes": 30, "intensity": "moderate"}}, "observed": {"g60": 120}}
    result = retrieve_similar_events([episode], {}, k=10)
    assert len(result) == 1
    assert result[0]["starting_state_summary"]["glucose"] == 145
