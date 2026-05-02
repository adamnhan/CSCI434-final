from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from traffic_classifier.features import build_session_dataset, build_window_dataset
from traffic_classifier.modeling import (
    build_models,
    cap_windows_per_capture,
    hold_out_session_id_per_label,
    session_majority_vote,
    split_features_and_target as split_window_features_and_target,
)


MAX_WINDOWS_PER_CAPTURE = 120
WINDOW_CONFIDENCE_FALLBACK_THRESHOLD = 0.45
SESSION_METADATA_COLUMNS = ["label", "capture_name", "session_id"]


def split_session_features_and_target(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    return df.loc[:, feature_columns], df["label"]


def build_session_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, random_state=random_state)),
        ]
    )


def main() -> None:
    capture_dir = Path("Wireshark Captures")
    window_df = build_window_dataset(capture_dir, window_size=75, min_packets=40)
    balanced_window_df = cap_windows_per_capture(window_df, MAX_WINDOWS_PER_CAPTURE)
    session_df = build_session_dataset(capture_dir)
    session_feature_columns = [column for column in session_df.columns if column not in SESSION_METADATA_COLUMNS]

    rows: list[dict[str, object]] = []
    for session_id in sorted(session_df["session_id"].unique()):
        window_split = hold_out_session_id_per_label(balanced_window_df, int(session_id))
        X_window_train, y_window_train = split_window_features_and_target(window_split.train_df)
        window_model = build_models(random_state=42)["random_forest"]
        window_model.fit(X_window_train, y_window_train)
        window_votes = session_majority_vote(window_model, window_split.test_df).rename(
            columns={"predicted_label": "window_prediction", "vote_share": "window_vote_share"}
        )

        session_train_df = session_df.loc[session_df["session_id"] != session_id]
        session_test_df = session_df.loc[session_df["session_id"] == session_id]
        X_session_train, y_session_train = split_session_features_and_target(session_train_df, session_feature_columns)
        X_session_test, _ = split_session_features_and_target(session_test_df, session_feature_columns)
        session_model = build_session_model(random_state=42)
        session_model.fit(X_session_train, y_session_train)
        session_predictions = pd.DataFrame(
            {
                "capture_name": session_test_df["capture_name"].to_numpy(),
                "session_prediction": session_model.predict(X_session_test),
            }
        )

        fold_predictions = window_votes.merge(session_predictions, on="capture_name", how="left")
        for row in fold_predictions.to_dict("records"):
            use_session_fallback = row["window_vote_share"] < WINDOW_CONFIDENCE_FALLBACK_THRESHOLD
            hybrid_prediction = row["session_prediction"] if use_session_fallback else row["window_prediction"]
            rows.append(
                {
                    "held_out_session_id": int(session_id),
                    "capture_name": row["capture_name"],
                    "true_label": row["true_label"],
                    "window_prediction": row["window_prediction"],
                    "window_vote_share": float(row["window_vote_share"]),
                    "session_prediction": row["session_prediction"],
                    "hybrid_prediction": hybrid_prediction,
                    "used_session_fallback": bool(use_session_fallback),
                    "correct": hybrid_prediction == row["true_label"],
                }
            )

    result_df = pd.DataFrame(rows)
    accuracy = accuracy_score(result_df["true_label"], result_df["hybrid_prediction"])
    macro_f1 = f1_score(result_df["true_label"], result_df["hybrid_prediction"], average="macro")

    print("Hybrid session-aware evaluation")
    print(f"Window model: balanced random forest, max {MAX_WINDOWS_PER_CAPTURE} windows per capture")
    print(f"Session fallback: logistic regression when window vote share < {WINDOW_CONFIDENCE_FALLBACK_THRESHOLD:.2f}")
    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"Overall macro F1: {macro_f1:.3f}")
    print()

    fold_summary = (
        result_df.groupby("held_out_session_id")["correct"]
        .mean()
        .rename("accuracy")
        .reset_index()
    )
    print("Fold accuracy")
    print(fold_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()

    print("Predictions")
    print(
        result_df.sort_values(["held_out_session_id", "capture_name"]).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print()

    errors = result_df.loc[~result_df["correct"]]
    print("Errors")
    if errors.empty:
        print("None")
    else:
        print(
            errors.sort_values(["held_out_session_id", "capture_name"]).to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )


if __name__ == "__main__":
    main()
