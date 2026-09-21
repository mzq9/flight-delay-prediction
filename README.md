# Flight Delay Prediction

Predict whether a U.S. domestic flight will depart at least 15 minutes late using only information available before departure.

## Why this project

Flight delays affect travelers and airline operations. This project uses Bureau of Transportation Statistics on-time performance data to identify pre-departure signals of a delay and evaluate a reproducible machine-learning baseline.

## Key design decision: prevent data leakage

The original course analysis included columns such as `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, and `LATE_AIRCRAFT_DELAY`. Those values describe the cause of a delay after it occurs, so they are deliberately excluded from the predictive model.

The model only uses features that can be known before the flight departs:

- operating carrier
- origin and destination airports
- month and day of week
- scheduled departure time
- scheduled flight distance

The target is `DEP_DEL15`: whether departure delay was at least 15 minutes.

## Dataset

Download monthly U.S. on-time performance CSV files from the [Bureau of Transportation Statistics](https://www.transtats.bts.gov/ONTIME/). Put CSV files in `data/raw/`; raw data is not stored in this repository.

Required columns:

```text
FL_DATE (or YEAR, MONTH, DAY_OF_MONTH)
DEP_DEL15
OP_UNIQUE_CARRIER
ORIGIN
DEST
MONTH
DAY_OF_WEEK
CRS_DEP_TIME
DISTANCE
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data-dir data/raw --model-output models/flight_delay_model.joblib
```

The script uses a chronological 80/20 train-test split, reports ROC-AUC, precision, recall, F1, and confusion matrix, then saves the trained pipeline.

## Repository structure

```text
data/README.md         Data download instructions; raw files remain local
src/train.py           Leakage-aware training and evaluation pipeline
requirements.txt       Reproducible Python dependencies
```

## Original work and attribution

This project began as a team course project for DS2500. This repository is a portfolio-focused refactor of the original analysis, with a revised pre-departure prediction design and reproducible training pipeline. Before publishing, replace this note with the accurate course and collaborator attribution you want to share.

