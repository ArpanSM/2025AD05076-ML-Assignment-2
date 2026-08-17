"""Streamlit interface for comparing the trained classification models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from metrics import calculate_metrics


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_TEST_DATA = BASE_DIR / "test_data.csv"

st.set_page_config(page_title="Credit Default Lab", page_icon="ML", layout="wide")


@st.cache_data
def load_metadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_model(artifact_name: str):
    return joblib.load(MODEL_DIR / artifact_name)


def validate_data(frame: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.Series | None]:
    feature_columns = metadata["feature_columns"]
    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(missing_features)}")

    features = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    target_column = metadata["target_column"]
    target = None
    if target_column in frame.columns:
        target = pd.to_numeric(frame[target_column], errors="raise").astype(int)
        if not set(target.unique()).issubset({0, 1}):
            raise ValueError(f"{target_column} must contain only 0 and 1 values.")
    return features, target


def show_confusion_matrix(model, features: pd.DataFrame, target: pd.Series) -> None:
    predictions = model.predict(features)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(4.8, 3.8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No default", "Default"],
        yticklabels=["No default", "Default"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Confusion matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#16324f",
                fontweight="bold",
            )
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


def main() -> None:
    metadata = load_metadata()
    model_names = list(metadata["models"])

    st.title("Credit Default Lab")
    st.caption(
        "Compare six classification models on the UCI Default of Credit Card Clients dataset."
    )

    with st.sidebar:
        st.subheader("Experiment controls")
        uploaded_file = st.file_uploader("Upload labeled test data", type="csv")
        selected_model = st.selectbox("Choose a model", model_names)
        show_comparison = st.checkbox("Show all-model comparison", value=True)
        st.divider()
        st.write("**Expected rows:** at least 500")
        st.write(f"**Predictors:** {len(metadata['feature_columns'])}")
        st.write("**Target:** default payment next month")

    if uploaded_file is None:
        frame = pd.read_csv(DEFAULT_TEST_DATA)
        source_label = "bundled test_data.csv"
    else:
        frame = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name

    try:
        features, target = validate_data(frame, metadata)
    except (TypeError, ValueError) as error:
        st.error(f"The CSV could not be used: {error}")
        st.stop()

    st.info(f"Using {source_label}: {len(frame):,} rows loaded.")
    preview_tab, results_tab = st.tabs(["Data preview", "Model results"])

    with preview_tab:
        st.dataframe(frame.head(10), width="stretch", hide_index=True)
        st.download_button(
            "Download predictions template",
            data=features.head(10).to_csv(index=False),
            file_name="prediction_template.csv",
            mime="text/csv",
        )

    with results_tab:
        model = load_model(metadata["models"][selected_model])
        st.subheader(selected_model)
        if target is None:
            predictions = model.predict(features)
            result_frame = features.copy()
            result_frame["prediction"] = predictions
            st.warning(
                f"The uploaded CSV has no {metadata['target_column']} column, so evaluation metrics "
                "are unavailable. Predictions are shown below."
            )
            st.dataframe(result_frame, use_container_width=True, hide_index=True)
        else:
            metrics = calculate_metrics(model, features, target)
            metric_columns = st.columns(6)
            for column, (metric_name, value) in zip(metric_columns, metrics.items()):
                column.metric(metric_name, f"{value:.4f}")

            matrix_column, report_column = st.columns(2)
            with matrix_column:
                show_confusion_matrix(model, features, target)
            with report_column:
                predictions = model.predict(features)
                report = classification_report(
                    target,
                    predictions,
                    labels=[0, 1],
                    target_names=["No default", "Default"],
                    output_dict=True,
                    zero_division=0,
                )
                st.markdown("#### Classification report")
                st.dataframe(
                    pd.DataFrame(report).transpose().round(4),
                    width="stretch",
                )

            prediction_frame = frame.copy()
            prediction_frame["prediction"] = predictions
            st.markdown("#### Row-level predictions")
            st.dataframe(prediction_frame, width="stretch", hide_index=True)

    if show_comparison and target is not None:
        st.divider()
        st.subheader("Model comparison")
        comparison_rows = []
        for model_name in model_names:
            comparison_model = load_model(metadata["models"][model_name])
            comparison_rows.append(
                {"Model": model_name, **calculate_metrics(comparison_model, features, target)}
            )
        st.dataframe(
            pd.DataFrame(comparison_rows).set_index("Model").style.format("{:.4f}"),
            width="stretch",
        )


if __name__ == "__main__":
    main()