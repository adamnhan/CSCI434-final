from __future__ import annotations

from pathlib import Path

import pandas as pd

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
def fit_and_report(train_df: pd.DataFrame, test_df: pd.DataFrame, title: str, feature_columns: list[str]) -> dict[str, object]:
    keep_cols = ["label", "capture_name", "session_id", "window_start_packet", "window_packet_count", "window_size", *feature_columns]
    train_subset = train_df.loc[:, keep_cols]
    test_subset = test_df.loc[:, keep_cols]
    X_train, y_train = split_features_and_target(train_subset)
    X_test, y_test = split_features_and_target(test_subset)

    models = build_models(random_state=42)
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    print(title)
    print(summarize_results(results).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    return {"models": models, "results": results, "X_columns": feature_columns, "test_subset": test_subset}


def main() -> None:
    capture_dir = Path("Wireshark Captures")
    feature_df = build_window_dataset(capture_dir, window_size=75, min_packets=40)
    balanced_df = cap_windows_per_capture(feature_df, MAX_WINDOWS_PER_CAPTURE)
    capture_summary = build_capture_summary(capture_dir)

    blind_sessions = capture_summary.loc[capture_summary["session_id"] == BLIND_TEST_SESSION_ID].copy()
    print(f"Blind test session id: {BLIND_TEST_SESSION_ID}")
    print(
        blind_sessions.loc[:, ["capture_name", "label", "packet_count", "duration_sec", "top_protocols"]]
        .sort_values(["label", "capture_name"])
        .to_string(index=False)
    )
    print()

    split = hold_out_session_id_per_label(feature_df, BLIND_TEST_SESSION_ID)
    balanced_split = hold_out_session_id_per_label(balanced_df, BLIND_TEST_SESSION_ID)

    X_full, _ = split_features_and_target(feature_df)
    full_columns = X_full.columns.tolist()
    reduced_columns = [column for column in full_columns if column not in TRANSPORT_HEAVY_COLUMNS]

    X_balanced, _ = split_features_and_target(balanced_df)
    balanced_full_columns = X_balanced.columns.tolist()
    balanced_reduced_columns = [column for column in balanced_full_columns if column not in TRANSPORT_HEAVY_COLUMNS]

    suites = [
        fit_and_report(split.train_df, split.test_df, "Blind test: full feature set", full_columns),
        fit_and_report(split.train_df, split.test_df, "Blind test: reduced feature set", reduced_columns),
        fit_and_report(
            balanced_split.train_df,
            balanced_split.test_df,
            f"Blind test: balanced full feature set (max {MAX_WINDOWS_PER_CAPTURE} windows per capture)",
            balanced_full_columns,
        ),
        fit_and_report(
            balanced_split.train_df,
            balanced_split.test_df,
            f"Blind test: balanced reduced feature set (max {MAX_WINDOWS_PER_CAPTURE} windows per capture)",
            balanced_reduced_columns,
        ),
    ]

    best_suite = suites[2]
    best_model = best_suite["models"]["random_forest"]
    print("Window-level report for balanced random forest")
    best_result = [result for result in best_suite["results"] if result.name == "random_forest"][0]
    print(best_result.report)
    print(best_result.confusion.to_string())
    print()

    session_votes = session_majority_vote(best_model, best_suite["test_subset"])
    print("Session-level majority vote for balanced random forest")
    print(session_votes.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    print("Session-level accuracy:", f"{session_votes['correct'].mean():.3f}")
    print()

    importances = random_forest_feature_importance(best_model, best_suite["X_columns"])
    print("Top random forest features on blind test training split")
    print(importances.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
