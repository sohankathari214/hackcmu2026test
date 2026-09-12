# GlucoPilot

GlucoPilot is a hackathon-runnable Python prototype for personalized short-term glucose forecasting in Type 1 diabetes. It is designed as a research and engineering prototype only, not a clinical system, not an emergency monitoring system, and not a source of insulin-dosing advice.

## What this repository does

- Loads structured medical profile, current state, and optional doctor policy JSON inputs.
- Normalizes data into event-style tables.
- Builds timestamp/window-based features using only information available at the snapshot time.
- Trains separate population LightGBM models for 30, 60, 90, and 120 minute trajectories.
- Trains quantile models, hypoglycemia classifiers, and optional hyperglycemia classifiers.
- Supports cold-start prediction, baseline comparisons, and patient-specific residual models.
- Runs scenario comparisons (baseline vs. proposed action), similar-event retrieval, and policy evaluation.
- Exposes an API for forecast and compare-scenarios endpoints.
- Produces evaluation metrics and plots.

## What this repository does not do

- It does not provide insulin dosing recommendations.
- It does not claim clinical validation.
- It does not provide emergency alerts or medical guidance.
- It does not extrapolate unsupported actions such as alcohol or caffeine without explicit training data support.

## Data Provenance

**Real HUPA-UCM data:** CGM, insulin raw values, carbohydrate raw values, heart rate, steps, calories, and time. `scripts/prepare_hupa.py` discovers every `data/raw/Preprocessed/HUPA*P.csv`, retains raw fields, and applies only configured conversions.

**Synthetic POC data:** exercise metadata/effects, caffeine, alcohol, stress, illness, hydration, and hidden simulator-only patient sensitivities. These transparent response curves are demonstration assumptions, not medical ground truth.

**Hybrid data:** a real HUPA physiological starting state plus a synthetic proposed action and synthetic response perturbation. Hybrid rows retain explicit provenance. This is not clinically validated and never provides insulin dosing advice.

Run the end-to-end proof of concept with `python3 scripts/train_full_poc.py` (or `make full-poc`).

## Grey-box simulator and decision support

`src/physiology/` provides a transparent five-minute, Monte-Carlo grey-box trajectory simulator. Its POC parameters and source links are in `configs/params.yaml`. `src/decision/` evaluates baseline and proposed scenarios, while `src/safety/` can block scenarios with missing glucose, configured exercise thresholds, ketone red flags, or clinician-authored hard rules. These components are decision-support demonstrations only: they are not clinically validated, do not prescribe insulin, and do not replace emergency or clinician care.

After the full pipeline, run `python3 scripts/train_personalization.py` and `python3 scripts/run_personalization_experiment.py` to persist general per-patient residual models and produce the chronological reveal curve. These models intentionally exclude simulator-only hidden traits.

## Architecture

MedicalProfile / CurrentState -> feature builder -> population models -> scenario forecast -> doctor policy -> structured outputs

After an actual action occurs, the system can create an Episode, compute residuals, and store patient-specific residual models for later personalization.

## Inputs

The repository expects validated JSON-compatible Python objects that represent:

- medical profile
- current state
- optional doctor policy
- optional episodes

See the example JSON files under examples/.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

Or with the included Makefile:

```bash
make install
```

## Repository layout

- configs/: YAML configuration for models, features, and runtime.
- data/: raw/interim/processed datasets.
- examples/: example input JSON payloads.
- src/: main Python package.
- scripts/: training, evaluation, and demo entrypoints.
- tests/: pytest suite.
- artifacts/: save models, metrics, and plots.

## Preparing data

The system supports a generic normalized event schema and adapters for HUPA and Ohio-style data.

### HUPA

Place HUPA files under data/raw/hupa/ and point the adapter at the files you want to use through the config. The adapter uses configurable column mappings. If downloaded filenames differ, update the adapter settings in configs/

Expected normalized internal schema:

- glucose_events: patient_id, timestamp, glucose_mg_dl
- insulin_events: patient_id, timestamp, bolus_units, basal_rate, insulin_type
- meal_events: patient_id, timestamp, carbs_g, protein_g, fat_g
- activity_events: patient_id, timestamp, steps, heart_rate, calories, activity_label
- sleep_events: patient_id, start, end, duration_hours, quality

## Training

```bash
python3 scripts/train_population.py
python3 scripts/train_risk_models.py
python3 scripts/train_personalization.py
```

## Evaluation

```bash
python3 scripts/evaluate.py
```

This writes metrics, plots, and per-horizon baseline comparisons.

## Personalization experiment

```bash
python3 scripts/run_personalization_experiment.py
```

This evaluates population-only versus incrementally personalized performance for held-out patients.

## Launching the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Or:

```bash
make api
```

## Example prediction

```bash
python3 scripts/demo_prediction.py
```

If trained models are unavailable, the script explains how to train them first.

## Main scripts

- scripts/prepare_data.py
- scripts/train_population.py
- scripts/train_risk_models.py
- scripts/train_personalization.py
- scripts/evaluate.py
- scripts/run_personalization_experiment.py
- scripts/demo_prediction.py

## Notes

- Random seeds are fixed for reproducibility.
- Feature names are logged and stored with the models.
- Model registry metadata is persisted under artifacts/.
- This repository is intentionally a prototype and does not claim medical validity.
