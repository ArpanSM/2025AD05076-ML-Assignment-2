"""Train and evaluate the classification models used by the Streamlit app."""

from __future__ import annotations

import json
import ssl
import subprocess
import urllib.request
from pathlib import Path

import certifi
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from metrics import calculate_metrics


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
SOURCE_XLS = DATA_DIR / "default_of_credit_card_clients.xls"
SOURCE_CSV = DATA_DIR / "default_of_credit_card_clients.csv"
TEST_CSV = BASE_DIR / "test_data.csv"

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
TARGET_COLUMN = "default_payment_next_month"
ID_COLUMN = "ID"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_BUILDERS = {
    "Logistic Regression": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "kNN": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
            ),
        ]
    ),
    "Naive Bayes": lambda: Pipeline(
        [("scaler", StandardScaler()), ("classifier", GaussianNB())]
    ),
    "Random Forest (Ensemble)": lambda: RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=RANDOM_STATE,
    ),
}


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply stable column names to the original UCI table."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.rename(columns={"default payment next month": TARGET_COLUMN})


def load_dataset() -> pd.DataFrame:
    """Load the cached CSV or download and convert the UCI Excel file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_CSV.exists():
        return _normalise_columns(pd.read_csv(SOURCE_CSV))

    if not SOURCE_XLS.exists():
        print("Downloading the UCI dataset...")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urllib.request.urlopen(DATASET_URL, context=ssl_context) as response:
                SOURCE_XLS.write_bytes(response.read())
        except urllib.error.URLError:
            subprocess.run(
                ["curl", "--fail", "--location", "--output", str(SOURCE_XLS), DATASET_URL],
                check=True,
            )

    frame = pd.read_excel(SOURCE_XLS, header=1)
    frame = _normalise_columns(frame)
    frame.to_csv(SOURCE_CSV, index=False)
    return frame


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Validate the dataset and separate predictors from the binary target."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    feature_columns = [
        column for column in frame.columns if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    if len(feature_columns) < 12 or len(frame) < 500:
        raise ValueError("The dataset does not satisfy the assignment size requirements.")

    features = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(int)
    if set(target.unique()) != {0, 1}:
        raise ValueError("This implementation expects a binary target encoded as 0 and 1.")
    return features, target, feature_columns


def main() -> None:
    frame = load_dataset()
    features, target, feature_columns = prepare_features(frame)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | str]] = []
    artifact_names: dict[str, str] = {}

    for model_name, builder in MODEL_BUILDERS.items():
        model = builder()
        model.fit(train_features, train_target)
        metrics = calculate_metrics(model, test_features, test_target)
        artifact_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        artifact_path = MODEL_DIR / f"{artifact_name}.joblib"
        joblib.dump(model, artifact_path, compress=3)
        artifact_names[model_name] = artifact_path.name
        metrics_rows.append({"Model": model_name, **metrics})

    test_frame = test_features.copy()
    test_frame[TARGET_COLUMN] = test_target.to_numpy()
    test_frame.to_csv(TEST_CSV, index=False)

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(MODEL_DIR / "metrics.csv", index=False)
    metadata = {
        "dataset": "UCI Default of Credit Card Clients",
        "dataset_url": DATASET_URL,
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "id_column_dropped": ID_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "metrics_average": "weighted for Precision, Recall, and F1",
        "models": artifact_names,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved test data to {TEST_CSV}")
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()