from src.episodes.store import EpisodeStore
def test_store_round_trip(tmp_path):
    store=EpisodeStore(tmp_path/"episodes.jsonl"); store.append({"episode_id":"x","patient_id":"p"}); assert store.patient("p")[0]["episode_id"]=="x"
