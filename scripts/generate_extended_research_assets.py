from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from traffic_classifier.features import build_session_dataset, load_capture_csv


ASSET_DIR = Path("report_assets")
CAPTURE_DIR = Path("Wireshark Captures")
METADATA_COLUMNS = ["label", "capture_name", "session_id"]


def build_linear_svm_k25(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=25)),
            ("model", SVC(kernel="linear", C=1.0, random_state=random_state)),
        ]
    )


def save_model_comparison() -> None:
    comparison = pd.DataFrame(
        [
            {
                "configuration": "Window RF\nblind session 05",
                "accuracy": 0.778,
                "macro_f1": 0.780,
            },
            {
                "configuration": "Window RF +\nmajority vote LOSO",
                "accuracy": 0.867,
                "macro_f1": pd.NA,
            },
            {
                "configuration": "Session LR\nblind session 05",
                "accuracy": 1.000,
                "macro_f1": 1.000,
            },
            {
                "configuration": "Session RF\nLOSO",
                "accuracy": 0.800,
                "macro_f1": 0.747,
            },
            {
                "configuration": "Session linear SVM\nk=25 LOSO",
                "accuracy": 0.900,
                "macro_f1": 0.880,
            },
        ]
    )
    comparison.to_csv(ASSET_DIR / "figure_7_extended_model_comparison.csv", index=False)

    plot_df = comparison.melt(
        id_vars="configuration",
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    ).dropna()

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=plot_df, x="configuration", y="score", hue="metric", palette=["#2f6f9f", "#c46a3a"])
    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1.2, label="90% target")
    ax.set_title("Extended Evaluation Results")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "figure_7_extended_model_comparison.png", dpi=200)
    plt.close()


def session_loso_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    session_df = build_session_dataset(CAPTURE_DIR)
    feature_columns = [column for column in session_df.columns if column not in METADATA_COLUMNS]
    rows: list[dict[str, object]] = []

    for held_out_session_id in sorted(session_df["session_id"].unique()):
        train_df = session_df.loc[session_df["session_id"] != held_out_session_id]
        test_df = session_df.loc[session_df["session_id"] == held_out_session_id]
        model = build_linear_svm_k25()
        model.fit(train_df.loc[:, feature_columns], train_df["label"])
        predictions = model.predict(test_df.loc[:, feature_columns])

        for capture_name, true_label, predicted_label in zip(
            test_df["capture_name"],
            test_df["label"],
            predictions,
        ):
            rows.append(
                {
                    "held_out_session_id": int(held_out_session_id),
                    "capture_name": capture_name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": true_label == predicted_label,
                }
            )

    prediction_df = pd.DataFrame(rows)
    fold_rows = []
    for held_out_session_id, fold_predictions in prediction_df.groupby("held_out_session_id"):
        fold_rows.append(
            {
                "held_out_session_id": int(held_out_session_id),
                "accuracy": accuracy_score(fold_predictions["true_label"], fold_predictions["predicted_label"]),
                "macro_f1": f1_score(
                    fold_predictions["true_label"],
                    fold_predictions["predicted_label"],
                    average="macro",
                ),
                "correct": int(fold_predictions["correct"].sum()),
                "total": int(len(fold_predictions)),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    return prediction_df, fold_df


def save_loso_figures(prediction_df: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    prediction_df.to_csv(ASSET_DIR / "figure_8_linear_svm_loso_predictions.csv", index=False)
    fold_df.to_csv(ASSET_DIR / "figure_8_linear_svm_loso_fold_accuracy.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=fold_df, x="held_out_session_id", y="accuracy", color="#2f6f9f")
    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_title("Linear SVM Session-Level Accuracy by Held-Out Session ID")
    ax.set_xlabel("Held-out session ID")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "figure_8_linear_svm_loso_fold_accuracy.png", dpi=200)
    plt.close()

    labels = sorted(prediction_df["true_label"].unique())
    matrix = confusion_matrix(prediction_df["true_label"], prediction_df["predicted_label"], labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.to_csv(ASSET_DIR / "figure_9_linear_svm_loso_confusion_matrix.csv")

    plt.figure(figsize=(6.5, 5.5))
    ax = sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5)
    ax.set_title("Linear SVM LOSO Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "figure_9_linear_svm_loso_confusion_matrix.png", dpi=200)
    plt.close()


def save_nbc_walmart_comparison() -> None:
    rows = []
    for file_name in ["nbcnews5.csv", "walmart5.csv"]:
        df = load_capture_csv(CAPTURE_DIR / file_name)
        rows.append(
            {
                "label": str(df["label"].iloc[0]),
                "capture_name": str(df["capture_name"].iloc[0]),
                "duration_sec": float(df["time"].max() - df["time"].min()),
                "mean_length": float(df["length"].mean()),
                "jumbo_ratio": float(df["length"].gt(1200).mean()),
                "tcp_ratio": float(df["protocol"].eq("tcp").mean()),
                "udp_ratio": float(df["protocol"].eq("udp").mean()),
                "tls13_ratio": float(df["protocol"].eq("tlsv1.3").mean()),
                "inbound_ratio": float(df["is_inbound"].mean()),
                "outbound_ratio": float(df["is_outbound"].mean()),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(ASSET_DIR / "figure_10_nbc_walmart_blind_profile.csv", index=False)
    plot_df = comparison.melt(
        id_vars=["label", "capture_name"],
        value_vars=["jumbo_ratio", "tcp_ratio", "udp_ratio", "tls13_ratio", "inbound_ratio", "outbound_ratio"],
        var_name="feature",
        value_name="value",
    )

    plt.figure(figsize=(9, 4.8))
    ax = sns.barplot(data=plot_df, x="feature", y="value", hue="label", palette=["#2f6f9f", "#c46a3a"])
    ax.set_title("NBC News vs Walmart Blind Session 05 Metadata Profile")
    ax.set_xlabel("")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Label")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "figure_10_nbc_walmart_blind_profile.png", dpi=200)
    plt.close()


def save_selected_features() -> None:
    session_df = build_session_dataset(CAPTURE_DIR)
    feature_columns = [column for column in session_df.columns if column not in METADATA_COLUMNS]
    model = build_linear_svm_k25()
    model.fit(session_df.loc[:, feature_columns], session_df["label"])

    variance_step = model.named_steps["variance"]
    select_step = model.named_steps["select"]
    variance_columns = pd.Index(feature_columns)[variance_step.get_support()]
    selected_features = variance_columns[select_step.get_support()]
    scores = pd.Series(select_step.scores_, index=variance_columns).loc[selected_features]
    selected_df = (
        pd.DataFrame({"feature": selected_features, "f_score": scores.to_numpy()})
        .sort_values("f_score", ascending=False)
        .reset_index(drop=True)
    )
    selected_df.to_csv(ASSET_DIR / "table_5_linear_svm_selected_features.csv", index=False)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    save_model_comparison()
    predictions, folds = session_loso_predictions()
    save_loso_figures(predictions, folds)
    save_nbc_walmart_comparison()
    save_selected_features()
    overall_accuracy = accuracy_score(predictions["true_label"], predictions["predicted_label"])
    pooled_macro_f1 = f1_score(predictions["true_label"], predictions["predicted_label"], average="macro")
    mean_fold_macro_f1 = folds["macro_f1"].mean()
    print(f"Linear SVM k=25 LOSO accuracy: {overall_accuracy:.3f}")
    print(f"Linear SVM k=25 LOSO mean fold macro F1: {mean_fold_macro_f1:.3f}")
    print(f"Linear SVM k=25 LOSO pooled macro F1: {pooled_macro_f1:.3f}")
    print(f"Wrote extended research assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
