PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) scripts/prepare_data.py

train:
	$(PYTHON) scripts/train_population.py

full-poc:
	$(PYTHON) scripts/train_full_poc.py

train-risk:
	$(PYTHON) scripts/train_risk_models.py

train-personal:
	$(PYTHON) scripts/train_personalization.py

evaluate:
	$(PYTHON) scripts/evaluate.py

personalization:
	$(PYTHON) scripts/run_personalization_experiment.py

api:
	$(PYTHON) -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

demo:
	$(PYTHON) scripts/demo_prediction.py

test:
	$(PYTHON) -m pytest -q
