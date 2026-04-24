from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic_classifier.features import build_capture_summary, build_window_dataset
from traffic_classifier.modeling import (
    build_models,
    cap_windows_per_capture,
    evaluate_models,
    hold_out_latest_session_per_label,
    random_forest_feature_importance,
    split_features_and_target,
    summarize_results,
)


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
MAX_WINDOWS_PER_CAPTURE = 120


def run_suite(train_df, test_df, title: str, X_columns) -> dict[str, object]:
    X_train, y_train = split_features_and_target(train_df.loc[:, ["label", "capture_name", "session_id", "window_start_packet", "window_packet_count", "window_size", *X_columns]])
    X_test, y_test = split_features_and_target(test_df.loc[:, ["label", "capture_name", "session_id", "window_start_packet", "window_packet_count", "window_size", *X_columns]])
    models = build_models(random_state=42)
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    return {"title": title, "models": models, "results": results, "X_columns": X_columns}


def main() -> None:
    capture_dir = Path("Wireshark Captures")
    feature_df = build_window_dataset(capture_dir, window_size=75, min_packets=40)
    capture_summary = build_capture_summary(capture_dir)
    session_split = hold_out_latest_session_per_label(feature_df)
    balanced_feature_df = cap_windows_per_capture(feature_df, MAX_WINDOWS_PER_CAPTURE)
    balanced_session_split = hold_out_latest_session_per_label(balanced_feature_df)

    print("Window dataset shape:", feature_df.shape)
    print(feature_df.groupby("label").size().sort_index().to_string())
    print()
    print("Capture sessions")
    print(
        capture_summary.loc[:, ["capture_name", "label", "session_id", "packet_count", "duration_sec", "top_protocols"]]
        .sort_values(["label", "session_id"])
        .to_string(index=False)
    )
    print()
    print("Held-out test sessions:", ", ".join(session_split.held_out_sessions))
    print()
    print(f"Capped windows per capture for balanced analysis: {MAX_WINDOWS_PER_CAPTURE}")
    print(
        balanced_feature_df.groupby(["label", "capture_name"]).size().rename("windows").reset_index()
        .sort_values(["label", "capture_name"])
        .to_string(index=False)
    )
    print()

    X, y = split_features_and_target(feature_df)
    full_columns = X.columns.tolist()
    reduced_columns = [column for column in full_columns if column not in TRANSPORT_HEAVY_COLUMNS]
    balanced_full_columns = split_features_and_target(balanced_feature_df)[0].columns.tolist()
    balanced_reduced_columns = [column for column in balanced_full_columns if column not in TRANSPORT_HEAVY_COLUMNS]

    suites = [
        run_suite(session_split.train_df, session_split.test_df, "Full feature set", full_columns),
        run_suite(
            session_split.train_df,
            session_split.test_df,
            "Reduced feature set without transport-heavy columns",
            reduced_columns,
        ),
        run_suite(
            balanced_session_split.train_df,
            balanced_session_split.test_df,
            f"Balanced full feature set (max {MAX_WINDOWS_PER_CAPTURE} windows per capture)",
            balanced_full_columns,
        ),
        run_suite(
            balanced_session_split.train_df,
            balanced_session_split.test_df,
            f"Balanced reduced feature set (max {MAX_WINDOWS_PER_CAPTURE} windows per capture)",
            balanced_reduced_columns,
        ),
    ]

    for suite in suites:
        print(suite["title"])
        print(summarize_results(suite["results"]).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print()

    full_suite = suites[0]
    for result in full_suite["results"]:
        print(f"=== {result.name} ({full_suite['title']}) ===")
        print(result.report)
        print(result.confusion.to_string())
        print()

    rf_importances = random_forest_feature_importance(full_suite["models"]["random_forest"], full_suite["X_columns"])
    print("Top random forest features (full feature set)")
    print(rf_importances.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
