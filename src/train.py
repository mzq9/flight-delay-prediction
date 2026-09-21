"""Train a leakage-aware flight-delay classifier from BTS on-time data."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "DEP_DEL15"
CATEGORICAL_FEATURES = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]
NUMERIC_FEATURES = ["MONTH", "DAY_OF_WEEK", "CRS_DEP_TIME", "DISTANCE"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
LEAKAGE_COLUMNS = {
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DELAY_GROUP",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "ARR_DELAY",
    "ARR_DEL15",
}


def load_csv_files(data_dir: Path) -> pd.DataFrame:
    """Read and concatenate the monthly CSV files in ``data_dir``."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}")

    frames = [pd.read_csv(file, low_memory=False) for file in csv_files]
    print(f"Loaded {len(csv_files)} file(s) and {sum(len(frame) for frame in frames):,} rows.")
    return pd.concat(frames, ignore_index=True)


def create_flight_date(df: pd.DataFrame) -> pd.Series:
    """Return a flight date for a chronological evaluation split."""
    if "FL_DATE" in df.columns:
        return pd.to_datetime(df["FL_DATE"], errors="coerce")

    date_columns = {"YEAR", "MONTH", "DAY_OF_MONTH"}
    if date_columns.issubset(df.columns):
        return pd.to_datetime(
            {"year": df["YEAR"], "month": df["MONTH"], "day": df["DAY_OF_MONTH"]},
            errors="coerce",
        )

    raise ValueError("Data must contain FL_DATE or YEAR, MONTH, and DAY_OF_MONTH.")


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid target rows and only the features available before departure."""
    required = set(MODEL_FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    model_df = df[MODEL_FEATURES + [TARGET]].copy()
    model_df["flight_date"] = create_flight_date(df)
    model_df = model_df.dropna(subset=[TARGET, "flight_date"])
    model_df[TARGET] = model_df[TARGET].astype(int)

    unexpected_leakage = LEAKAGE_COLUMNS.intersection(model_df.columns)
    if unexpected_leakage:
        raise AssertionError(f"Leakage columns found in model data: {sorted(unexpected_leakage)}")

    return model_df.sort_values("flight_date").reset_index(drop=True)


def build_pipeline() -> Pipeline:
    """Create preprocessing plus a class-balanced random-forest baseline."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", classifier)])


def chronological_split(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use later dates as the test set, matching a realistic deployment setting."""
    split_index = int(len(model_df) * 0.8)
    if split_index == 0 or split_index == len(model_df):
        raise ValueError("At least two valid rows are required for a train-test split.")
    return model_df.iloc[:split_index].copy(), model_df.iloc[split_index:].copy()


def evaluate(model: Pipeline, test_df: pd.DataFrame) -> None:
    """Print classification metrics for the held-out future period."""
    x_test = test_df[MODEL_FEATURES]
    y_test = test_df[TARGET]
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nEvaluation on chronologically held-out data")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test, predictions, zero_division=0):.3f}")
    print("\nClassification report")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix")
    print(ConfusionMatrixDisplay.from_predictions(y_test, predictions).confusion_matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing BTS CSV files.")
    parser.add_argument("--model-output", type=Path, default=Path("models/flight_delay_model.joblib"))
    args = parser.parse_args()

    raw_df = load_csv_files(args.data_dir)
    model_df = prepare_model_data(raw_df)
    train_df, test_df = chronological_split(model_df)

    print(f"Training period: {train_df.flight_date.min().date()} to {train_df.flight_date.max().date()}")
    print(f"Test period:     {test_df.flight_date.min().date()} to {test_df.flight_date.max().date()}")
    print(f"Delayed-flight rate, train: {train_df[TARGET].mean():.1%}")

    pipeline = build_pipeline()
    pipeline.fit(train_df[MODEL_FEATURES], train_df[TARGET])
    evaluate(pipeline, test_df)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_output)
    print(f"\nSaved model to {args.model_output}")


if __name__ == "__main__":
    main()
