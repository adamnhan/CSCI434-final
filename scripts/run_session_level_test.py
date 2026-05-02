from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from traffic_classifier.features import build_session_dataset
from traffic_classifier.modeling import random_forest_feature_importance


BLIND_TEST_SESSION_ID = 5
METADATA_COLUMNS = ["label", "capture_name", "session_id"]
VOLUME_HEAVY_COLUMNS = [
    "packet_count",
    "duration_sec",
    "packet_rate",
    "total_bytes",
    "byte_rate",
    "first_5s_packet_count",
    "first_10s_packet_count",
]


def split_features_and_target(df: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if feature_columns is None:
        return df.drop(columns=METADATA_COLUMNS), df["label"]
    return df.loc[:, feature_columns], df["label"]


def build_session_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
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
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=400, random_state=random_state)),
            ]
        ),
        "linear_svm_k25": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(score_func=f_classif, k=25)),
                ("model", SVC(kernel="linear", C=1.0, random_state=random_state)),
            ]
        ),
    }


def evaluate_suite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    title: str,
    feature_columns: list[str],
) -> dict[str, Pipeline]:
    X_train, y_train = split_features_and_target(train_df, feature_columns)
    X_test, y_test = split_features_and_target(test_df, feature_columns)
    models = build_session_models()
    rows = []

    print(title)
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "macro_f1": f1_score(y_test, preds, average="macro"),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()

    best_name = str(summary.iloc[0]["model"])
    best_model = models[best_name]
    preds = best_model.predict(X_test)
    labels = sorted(y_test.unique())
    print(f"Best model detail: {best_name}")
    print(classification_report(y_test, preds, digits=3, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y_test, preds, labels=labels), index=labels, columns=labels).to_string())
    print()
    print("Predictions")
    print(
        pd.DataFrame(
            {
                "capture_name": test_df["capture_name"].to_numpy(),
                "true_label": y_test.to_numpy(),
                "predicted_label": preds,
                "correct": preds == y_test.to_numpy(),
            }
        ).sort_values("capture_name").to_string(index=False)
    )
    print()

    if best_name in {"random_forest", "extra_trees"}:
        importances = random_forest_feature_importance(best_model, X_train.columns.tolist(), top_n=20)
        print("Top feature importances")
        print(importances.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print()

    return models


def evaluate_leave_one_session_id_out(session_df: pd.DataFrame, title: str, feature_columns: list[str]) -> None:
    rows = []
    models = build_session_models()

    for held_out_session_id in sorted(session_df["session_id"].unique()):
        train_df = session_df.loc[session_df["session_id"] != held_out_session_id]
        test_df = session_df.loc[session_df["session_id"] == held_out_session_id]
        X_train, y_train = split_features_and_target(train_df, feature_columns)
        X_test, y_test = split_features_and_target(test_df, feature_columns)

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rows.append(
                {
                    "held_out_session_id": held_out_session_id,
                    "model": name,
                    "accuracy": accuracy_score(y_test, preds),
                    "macro_f1": f1_score(y_test, preds, average="macro"),
                }
            )

    fold_df = pd.DataFrame(rows)
    print(title)
    print(
        fold_df.groupby("model")[["accuracy", "macro_f1"]]
        .mean()
        .sort_values(["macro_f1", "accuracy"], ascending=False)
        .to_string(float_format=lambda value: f"{value:.3f}")
    )
    print()
    print("Fold accuracy by held-out session id")
    print(
        fold_df.pivot(index="held_out_session_id", columns="model", values="accuracy")
        .to_string(float_format=lambda value: f"{value:.3f}")
    )
    print()


def main() -> None:
    session_df = build_session_dataset(Path("Wireshark Captures"))
    print("Session dataset shape:", session_df.shape)
    print(
        session_df.loc[:, ["capture_name", "label", "session_id", "packet_count", "duration_sec"]]
        .sort_values(["label", "session_id"])
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print()

    train_df = session_df.loc[session_df["session_id"] != BLIND_TEST_SESSION_ID].reset_index(drop=True)
    test_df = session_df.loc[session_df["session_id"] == BLIND_TEST_SESSION_ID].reset_index(drop=True)
    full_columns = [column for column in session_df.columns if column not in METADATA_COLUMNS]
    reduced_columns = [column for column in full_columns if column not in VOLUME_HEAVY_COLUMNS]

    evaluate_suite(train_df, test_df, "Session-level blind test: all five classes, full features", full_columns)
    evaluate_suite(
        train_df,
        test_df,
        "Session-level blind test: all five classes, no raw volume/duration/rate columns",
        reduced_columns,
    )
    evaluate_leave_one_session_id_out(
        session_df,
        "Leave-one-session-id-out: all five classes, full features",
        full_columns,
    )
    evaluate_leave_one_session_id_out(
        session_df,
        "Leave-one-session-id-out: all five classes, no raw volume/duration/rate columns",
        reduced_columns,
    )

    binary_df = session_df.loc[session_df["label"].isin(["nbcnews", "walmart"])].reset_index(drop=True)
    binary_train = binary_df.loc[binary_df["session_id"] != BLIND_TEST_SESSION_ID].reset_index(drop=True)
    binary_test = binary_df.loc[binary_df["session_id"] == BLIND_TEST_SESSION_ID].reset_index(drop=True)
    binary_full_columns = [column for column in binary_df.columns if column not in METADATA_COLUMNS]
    binary_reduced_columns = [column for column in binary_full_columns if column not in VOLUME_HEAVY_COLUMNS]
    evaluate_suite(binary_train, binary_test, "Session-level blind test: NBC News vs Walmart only, full features", binary_full_columns)
    evaluate_suite(
        binary_train,
        binary_test,
        "Session-level blind test: NBC News vs Walmart only, no raw volume/duration/rate columns",
        binary_reduced_columns,
    )
    evaluate_leave_one_session_id_out(
        binary_df,
        "Leave-one-session-id-out: NBC News vs Walmart only, full features",
        binary_full_columns,
    )
    evaluate_leave_one_session_id_out(
        binary_df,
        "Leave-one-session-id-out: NBC News vs Walmart only, no raw volume/duration/rate columns",
        binary_reduced_columns,
    )


if __name__ == "__main__":
    main()
