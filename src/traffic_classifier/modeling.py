from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = ["label", "capture_name", "session_id", "window_start_packet", "window_packet_count", "window_size"]


@dataclass
class ModelResult:
    name: str
    accuracy: float
    macro_f1: float
    report: str
    confusion: pd.DataFrame


@dataclass
class SessionSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    held_out_sessions: list[str]


def split_features_and_target(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = feature_df.drop(columns=METADATA_COLUMNS)
    y = feature_df["label"]
    return X, y


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=4000, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=400, random_state=random_state)),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[ModelResult]:
    results: list[ModelResult] = []
    ordered_labels = sorted(y_test.unique())

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append(
            ModelResult(
                name=name,
                accuracy=accuracy_score(y_test, preds),
                macro_f1=f1_score(y_test, preds, average="macro"),
                report=classification_report(y_test, preds, digits=3, zero_division=0),
                confusion=pd.DataFrame(
                    confusion_matrix(y_test, preds, labels=ordered_labels),
                    index=ordered_labels,
                    columns=ordered_labels,
                ),
            )
        )

    return results


def summarize_results(results: list[ModelResult]) -> pd.DataFrame:
    rows = [{"model": result.name, "accuracy": result.accuracy, "macro_f1": result.macro_f1} for result in results]
    return pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)


def random_forest_feature_importance(model: Pipeline, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": estimator.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importances.head(top_n).reset_index(drop=True)


def cap_windows_per_capture(feature_df: pd.DataFrame, max_windows: int, random_state: int = 42) -> pd.DataFrame:
    if max_windows <= 0:
        return feature_df.copy()

    sampled = []
    for _, capture_df in feature_df.groupby("capture_name", sort=False):
        take_n = min(len(capture_df), max_windows)
        sampled.append(capture_df.sample(n=take_n, random_state=random_state))

    return (
        pd.concat(sampled, ignore_index=True)
        .sort_values(["label", "session_id", "window_start_packet"])
        .reset_index(drop=True)
    )


def hold_out_latest_session_per_label(feature_df: pd.DataFrame) -> SessionSplit:
    required_columns = {"label", "capture_name", "session_id"}
    missing = required_columns - set(feature_df.columns)
    if missing:
        raise ValueError(f"Feature frame missing required columns for session split: {sorted(missing)}")

    held_out_sessions: list[str] = []
    test_frames: list[pd.DataFrame] = []
    train_frames: list[pd.DataFrame] = []

    for label, label_df in feature_df.groupby("label", sort=True):
        sessions = (
            label_df[["capture_name", "session_id"]]
            .drop_duplicates()
            .sort_values(["session_id", "capture_name"])
            .reset_index(drop=True)
        )
        if len(sessions) < 2:
            raise ValueError(f"Need at least two capture sessions for label '{label}' to do a session holdout split.")

        held_out = str(sessions.iloc[-1]["capture_name"])
        held_out_sessions.append(held_out)

        test_mask = label_df["capture_name"] == held_out
        test_frames.append(label_df.loc[test_mask])
        train_frames.append(label_df.loc[~test_mask])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    return SessionSplit(train_df=train_df, test_df=test_df, held_out_sessions=held_out_sessions)


def hold_out_session_id_per_label(feature_df: pd.DataFrame, session_id: int) -> SessionSplit:
    required_columns = {"label", "capture_name", "session_id"}
    missing = required_columns - set(feature_df.columns)
    if missing:
        raise ValueError(f"Feature frame missing required columns for session split: {sorted(missing)}")

    held_out_sessions: list[str] = []
    test_frames: list[pd.DataFrame] = []
    train_frames: list[pd.DataFrame] = []

    for label, label_df in feature_df.groupby("label", sort=True):
        test_mask = label_df["session_id"] == session_id
        if not bool(test_mask.any()):
            raise ValueError(f"Missing blind-test session {session_id} for label '{label}'.")
        if bool((~test_mask).sum() == 0):
            raise ValueError(f"Need at least one non-test session for label '{label}'.")

        held_out_sessions.extend(label_df.loc[test_mask, "capture_name"].drop_duplicates().tolist())
        test_frames.append(label_df.loc[test_mask])
        train_frames.append(label_df.loc[~test_mask])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    return SessionSplit(train_df=train_df, test_df=test_df, held_out_sessions=sorted(held_out_sessions))


def session_majority_vote(model: Pipeline, feature_df: pd.DataFrame) -> pd.DataFrame:
    metadata = feature_df.loc[:, ["capture_name", "label"]].copy()
    X, _ = split_features_and_target(feature_df)
    window_preds = pd.Series(model.predict(X), index=feature_df.index, name="predicted_label")
    voted = (
        pd.concat([metadata, window_preds], axis=1)
        .groupby(["capture_name", "label", "predicted_label"])
        .size()
        .rename("window_votes")
        .reset_index()
        .sort_values(["capture_name", "window_votes", "predicted_label"], ascending=[True, False, True])
    )
    winners = (
        voted.groupby("capture_name", as_index=False)
        .first()
        .rename(columns={"label": "true_label"})
    )
    total_votes = (
        voted.groupby("capture_name")["window_votes"].sum().rename("total_windows").reset_index()
    )
    winners = winners.merge(total_votes, on="capture_name", how="left")
    winners["vote_share"] = winners["window_votes"] / winners["total_windows"]
    winners["correct"] = winners["predicted_label"] == winners["true_label"]
    return winners.loc[:, ["capture_name", "true_label", "predicted_label", "window_votes", "total_windows", "vote_share", "correct"]]
