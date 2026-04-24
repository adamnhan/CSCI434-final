from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
import seaborn as sns

from traffic_classifier.features import build_capture_summary, build_window_dataset
from traffic_classifier.modeling import (
    build_models,
    cap_windows_per_capture,
    evaluate_models,
    hold_out_session_id_per_label,
    random_forest_feature_importance,
    session_majority_vote,
    split_features_and_target,
    summarize_results,
)


BLIND_TEST_SESSION_ID = 5
MAX_WINDOWS_PER_CAPTURE = 120
TRANSPORT_HEAVY_COLUMNS = [
    "udp_ratio",
    "tcp_ratio",
    "tls12_ratio",
    "tls13_ratio",
    "ipv6_ratio",
    "dst_443_ratio",
    "src_443_ratio",
    "outbound_ratio",
    "inbound_ratio",
    "direction_balance",
]


def save_dataframe_table(df: pd.DataFrame, out_path: Path, title: str, font_size: int = 10) -> None:
    display_df = df.copy()
    fig_width = max(8, len(display_df.columns) * 1.6)
    fig_height = max(2.5, len(display_df) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=16)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#264653")
        else:
            cell.set_facecolor("#f8f9fa" if row % 2 == 1 else "#e9ecef")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_pipeline_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)

    boxes = [
        (0.5, "Wireshark\ncapture sessions"),
        (3.0, "CSV export\nand parsing"),
        (5.5, "Packet-window\nconstruction"),
        (8.0, "Feature\nextraction"),
        (10.5, "Train on\nsessions 00-04"),
        (12.7, "Blind test\non session 05"),
    ]

    for x, text in boxes:
        rect = patches.FancyBboxPatch(
            (x, 1.3),
            1.8,
            1.2,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.8,
            edgecolor="#264653",
            facecolor="#e9f5f2",
        )
        ax.add_patch(rect)
        ax.text(x + 0.9, 1.9, text, ha="center", va="center", fontsize=11)

    for start_x in [2.3, 4.8, 7.3, 9.8, 12.3]:
        ax.annotate(
            "",
            xy=(start_x + 0.6, 1.9),
            xytext=(start_x, 1.9),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#264653"},
        )

    ax.text(7, 3.3, "Encrypted Traffic Website Classification Pipeline", ha="center", fontsize=16, weight="bold")
    ax.text(7, 0.5, "Balanced window sampling and session-level evaluation prevent large captures and leakage from dominating results.", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_suite(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    keep_cols = ["label", "capture_name", "session_id", "window_start_packet", "window_packet_count", "window_size", *feature_columns]
    train_subset = train_df.loc[:, keep_cols]
    test_subset = test_df.loc[:, keep_cols]
    X_train, y_train = split_features_and_target(train_subset)
    X_test, y_test = split_features_and_target(test_subset)
    models = build_models(random_state=42)
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    return {
        "models": models,
        "results": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_subset": test_subset,
    }


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    repo_root = Path.cwd()
    out_dir = repo_root / "report_assets"
    out_dir.mkdir(exist_ok=True)

    capture_dir = repo_root / "Wireshark Captures"
    feature_df = build_window_dataset(capture_dir, window_size=75, min_packets=40)
    balanced_df = cap_windows_per_capture(feature_df, MAX_WINDOWS_PER_CAPTURE)
    capture_summary = build_capture_summary(capture_dir)
    balanced_split = hold_out_session_id_per_label(balanced_df, BLIND_TEST_SESSION_ID)

    train_X, _ = split_features_and_target(balanced_df)
    full_columns = train_X.columns.tolist()
    reduced_columns = [column for column in full_columns if column not in TRANSPORT_HEAVY_COLUMNS]

    full_suite = fit_suite(balanced_split.train_df, balanced_split.test_df, full_columns)
    reduced_suite = fit_suite(balanced_split.train_df, balanced_split.test_df, reduced_columns)

    model_table = pd.concat(
        [
            summarize_results(full_suite["results"]).assign(feature_set="balanced_full"),
            summarize_results(reduced_suite["results"]).assign(feature_set="balanced_reduced"),
        ],
        ignore_index=True,
    )
    model_table = model_table.loc[:, ["feature_set", "model", "accuracy", "macro_f1"]]
    model_table["accuracy"] = model_table["accuracy"].map(lambda value: f"{value:.3f}")
    model_table["macro_f1"] = model_table["macro_f1"].map(lambda value: f"{value:.3f}")

    dataset_table = (
        capture_summary.groupby("label")
        .agg(
            sessions=("capture_name", "nunique"),
            packet_min=("packet_count", "min"),
            packet_max=("packet_count", "max"),
            duration_min=("duration_sec", "min"),
            duration_max=("duration_sec", "max"),
        )
        .reset_index()
    )
    dataset_table["packet_range"] = dataset_table.apply(lambda row: f"{int(row['packet_min'])}-{int(row['packet_max'])}", axis=1)
    dataset_table["duration_range_sec"] = dataset_table.apply(lambda row: f"{row['duration_min']:.1f}-{row['duration_max']:.1f}", axis=1)
    dataset_table = dataset_table.loc[:, ["label", "sessions", "packet_range", "duration_range_sec"]]

    ablation_sets = {
        "all_features": full_columns,
        "no_capture_duration": [column for column in full_columns if column != "capture_duration"],
        "no_duration_or_rate": [column for column in full_columns if column not in ["capture_duration", "packet_rate"]],
        "length_and_timing_only": [
            column
            for column in full_columns
            if column.startswith("length_") or column.startswith("delta_") or column in ["capture_duration", "packet_rate"]
        ],
    }
    ablation_rows: list[dict[str, object]] = []
    for feature_set_name, columns in ablation_sets.items():
        ablation_suite = fit_suite(balanced_split.train_df, balanced_split.test_df, columns)
        rf_row = summarize_results(ablation_suite["results"]).query("model == 'random_forest'").iloc[0]
        ablation_rows.append(
            {
                "feature_set": feature_set_name,
                "n_features": len(columns),
                "accuracy": f"{rf_row['accuracy']:.3f}",
                "macro_f1": f"{rf_row['macro_f1']:.3f}",
            }
        )
    ablation_table = pd.DataFrame(ablation_rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)

    best_rf = full_suite["models"]["random_forest"]
    best_rf_result = [result for result in full_suite["results"] if result.name == "random_forest"][0]
    feature_importance = random_forest_feature_importance(best_rf, full_suite["X_train"].columns.tolist(), top_n=15)
    feature_importance["importance"] = feature_importance["importance"].map(lambda value: float(f"{value:.4f}"))

    session_votes = session_majority_vote(best_rf, full_suite["test_subset"]).copy()
    session_votes["vote_share_numeric"] = session_votes["vote_share"]
    session_votes["vote_share"] = session_votes["vote_share"].map(lambda value: f"{value:.3f}")

    make_pipeline_figure(out_dir / "figure_1_pipeline.png")

    save_dataframe_table(dataset_table, out_dir / "table_1_dataset_summary.png", "Table 1. Dataset Summary")
    dataset_table.to_csv(out_dir / "table_1_dataset_summary.csv", index=False)

    save_dataframe_table(model_table, out_dir / "table_2_model_comparison.png", "Table 2. Blind-Test Model Comparison")
    model_table.to_csv(out_dir / "table_2_model_comparison.csv", index=False)

    plt.figure(figsize=(7, 6))
    sns.heatmap(best_rf_result.confusion, annot=True, fmt="d", cmap="Blues")
    plt.title("Figure 2. Blind-Test Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_2_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close()
    best_rf_result.confusion.to_csv(out_dir / "figure_2_confusion_matrix.csv")

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_importance, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title("Figure 3. Top Random-Forest Features")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_3_feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close()
    feature_importance.to_csv(out_dir / "figure_3_feature_importance.csv", index=False)

    save_dataframe_table(ablation_table, out_dir / "table_3_feature_ablation.png", "Table 3. Feature Ablation Results")
    ablation_table.to_csv(out_dir / "table_3_feature_ablation.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=session_votes,
        x="capture_name",
        y="vote_share_numeric",
        hue="correct",
        dodge=False,
        palette={True: "#2a9d8f", False: "#e76f51"},
    )
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylim(0, 1.0)
    plt.title("Figure 4. Session-Level Majority-Vote Confidence")
    plt.ylabel("Vote Share For Predicted Label")
    plt.xlabel("Capture Session")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "figure_4_session_vote_confidence.png", dpi=220, bbox_inches="tight")
    plt.close()
    session_votes.to_csv(out_dir / "figure_4_session_vote_confidence.csv", index=False)

    save_dataframe_table(
        session_votes.drop(columns=["vote_share_numeric"]),
        out_dir / "table_4_session_vote_summary.png",
        "Table 4. Session-Level Vote Summary",
        font_size=9,
    )
    session_votes.drop(columns=["vote_share_numeric"]).to_csv(out_dir / "table_4_session_vote_summary.csv", index=False)

    print(f"Generated report assets in {out_dir}")


if __name__ == "__main__":
    main()
