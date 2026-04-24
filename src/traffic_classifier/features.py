from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


PORT_RE = re.compile(r"(?P<src>\d+)\s+>\s+(?P<dst>\d+)")
CAPTURE_RE = re.compile(r"^(?P<label>[a-zA-Z_]+?)(?:[_-]?(?P<session>\d+))?$")


def parse_capture_name(path: str | Path) -> tuple[str, int, str]:
    path = Path(path)
    stem = path.stem.lower()
    match = CAPTURE_RE.match(stem)
    if not match:
        return stem, 0, stem

    label = match.group("label").rstrip("_-").lower()
    session_str = match.group("session")
    session_id = int(session_str) if session_str is not None else 0
    capture_name = f"{label}_{session_id:02d}"
    return label, session_id, capture_name


def load_capture_csv(path: str | Path) -> pd.DataFrame:
    """Load one Wireshark CSV export and attach normalized helper columns."""
    path = Path(path)
    label, session_id, capture_name = parse_capture_name(path)
    df = pd.read_csv(path)
    df["label"] = label
    df["session_id"] = session_id
    df["capture_name"] = capture_name
    df["time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0)
    df["length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(0).astype(int)
    df["protocol"] = df["Protocol"].fillna("UNKNOWN").astype(str).str.lower()
    df["source_addr"] = df["Source"].fillna("").astype(str)
    df["destination_addr"] = df["Destination"].fillna("").astype(str)

    parsed_ports = df["Info"].fillna("").astype(str).apply(_extract_ports)
    df["src_port"] = parsed_ports.str[0]
    df["dst_port"] = parsed_ports.str[1]

    # Approximate traffic direction using conventional HTTPS port behavior.
    df["is_outbound"] = ((df["dst_port"] == 443) & (df["src_port"] != 443)).astype(int)
    df["is_inbound"] = ((df["src_port"] == 443) & (df["dst_port"] != 443)).astype(int)
    df["direction"] = np.where(df["is_outbound"] == 1, 1, np.where(df["is_inbound"] == 1, -1, 0))
    df["is_ipv6"] = (df["source_addr"].str.contains(":", regex=False) | df["destination_addr"].str.contains(":", regex=False)).astype(int)

    deltas = df["time"].diff().fillna(0.0)
    df["delta_time"] = deltas.clip(lower=0.0)
    return df


def build_window_dataset(
    capture_dir: str | Path,
    window_size: int = 100,
    min_packets: int = 30,
) -> pd.DataFrame:
    """Convert each capture into fixed-size packet windows with aggregated features."""
    capture_dir = Path(capture_dir)
    rows: list[dict[str, float | int | str]] = []

    for csv_path in sorted(capture_dir.glob("*.csv")):
        df = load_capture_csv(csv_path)
        for start in range(0, len(df), window_size):
            window = df.iloc[start : start + window_size].copy()
            if len(window) < min_packets:
                continue
            rows.append(_window_features(window, start, window_size))

    return pd.DataFrame(rows)


def _extract_ports(info: str) -> tuple[int | None, int | None]:
    match = PORT_RE.search(info)
    if not match:
        return (None, None)
    return (int(match.group("src")), int(match.group("dst")))


def _ratio(series: pd.Series, predicate) -> float:
    if len(series) == 0:
        return 0.0
    return float(predicate(series).mean())


def _entropy(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0))
    return 0.0 if math.isnan(value) else value


def _run_lengths(values: pd.Series) -> list[int]:
    filtered = [int(v) for v in values if int(v) != 0]
    if not filtered:
        return []

    runs: list[int] = []
    current = filtered[0]
    current_len = 1
    for value in filtered[1:]:
        if value == current:
            current_len += 1
        else:
            runs.append(current_len)
            current = value
            current_len = 1
    runs.append(current_len)
    return runs


def _run_stat(values: list[int], op: str) -> float:
    if not values:
        return 0.0
    if op == "max":
        return float(max(values))
    if op == "mean":
        return float(sum(values) / len(values))
    raise ValueError(f"Unsupported op: {op}")


def build_capture_summary(capture_dir: str | Path) -> pd.DataFrame:
    capture_dir = Path(capture_dir)
    rows: list[dict[str, float | int | str]] = []

    for csv_path in sorted(capture_dir.glob("*.csv")):
        df = load_capture_csv(csv_path)
        protocol_counts = df["protocol"].value_counts()
        rows.append(
            {
                "label": str(df["label"].iloc[0]),
                "session_id": int(df["session_id"].iloc[0]),
                "capture_name": str(df["capture_name"].iloc[0]),
                "file_name": csv_path.name,
                "packet_count": int(len(df)),
                "duration_sec": float(df["time"].max() - df["time"].min()),
                "udp_ratio": _ratio(df["protocol"], lambda s: s.eq("udp")),
                "tcp_ratio": _ratio(df["protocol"], lambda s: s.eq("tcp")),
                "quic_ratio": _ratio(df["protocol"], lambda s: s.eq("quic")),
                "tls12_ratio": _ratio(df["protocol"], lambda s: s.eq("tlsv1.2")),
                "tls13_ratio": _ratio(df["protocol"], lambda s: s.eq("tlsv1.3")),
                "top_protocols": ", ".join(f"{name}:{count}" for name, count in protocol_counts.head(5).items()),
            }
        )

    return pd.DataFrame(rows)


def _window_features(window: pd.DataFrame, start: int, window_size: int) -> dict[str, float | int | str]:
    lengths = window["length"]
    deltas = window["delta_time"]
    protocols = window["protocol"]
    directions = window["direction"]
    direction_runs = _run_lengths(directions)
    nonzero_directions = directions[directions != 0]
    direction_changes = float((nonzero_directions != nonzero_directions.shift()).sum() - 1) if len(nonzero_directions) > 1 else 0.0

    features: dict[str, float | int | str] = {
        "label": str(window["label"].iloc[0]),
        "session_id": int(window["session_id"].iloc[0]),
        "capture_name": str(window["capture_name"].iloc[0]),
        "window_start_packet": start,
        "window_packet_count": int(len(window)),
        "capture_duration": float(window["time"].iloc[-1] - window["time"].iloc[0]),
        "packet_rate": float(len(window) / max(window["time"].iloc[-1] - window["time"].iloc[0], 1e-6)),
        "length_mean": float(lengths.mean()),
        "length_std": _safe_std(lengths),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "length_median": float(lengths.median()),
        "length_p90": float(lengths.quantile(0.9)),
        "length_entropy": _entropy(lengths),
        "length_small_ratio": _ratio(lengths, lambda s: s.le(128)),
        "length_medium_ratio": _ratio(lengths, lambda s: s.gt(128) & s.le(512)),
        "length_large_ratio": _ratio(lengths, lambda s: s.gt(512) & s.le(1200)),
        "length_jumbo_ratio": _ratio(lengths, lambda s: s.gt(1200)),
        "delta_mean": float(deltas.mean()),
        "delta_std": _safe_std(deltas),
        "delta_max": float(deltas.max()),
        "delta_p90": float(deltas.quantile(0.9)),
        "delta_zero_ratio": _ratio(deltas, lambda s: s.eq(0.0)),
        "delta_fast_ratio": _ratio(deltas, lambda s: s.gt(0.0) & s.le(0.01)),
        "delta_medium_ratio": _ratio(deltas, lambda s: s.gt(0.01) & s.le(0.1)),
        "delta_slow_ratio": _ratio(deltas, lambda s: s.gt(0.1)),
        "udp_ratio": _ratio(protocols, lambda s: s.eq("udp")),
        "tcp_ratio": _ratio(protocols, lambda s: s.eq("tcp")),
        "tls12_ratio": _ratio(protocols, lambda s: s.eq("tlsv1.2")),
        "tls13_ratio": _ratio(protocols, lambda s: s.eq("tlsv1.3")),
        "outbound_ratio": float(window["is_outbound"].mean()),
        "inbound_ratio": float(window["is_inbound"].mean()),
        "direction_balance": float(window["is_outbound"].mean() - window["is_inbound"].mean()),
        "direction_change_ratio": float(direction_changes / max(len(nonzero_directions) - 1, 1)),
        "direction_run_mean": _run_stat(direction_runs, "mean"),
        "direction_run_max": _run_stat(direction_runs, "max"),
        "dst_443_ratio": _ratio(window["dst_port"], lambda s: s.eq(443)),
        "src_443_ratio": _ratio(window["src_port"], lambda s: s.eq(443)),
        "unique_src_ports": int(window["src_port"].dropna().nunique()),
        "unique_dst_ports": int(window["dst_port"].dropna().nunique()),
        "unique_sources": int(window["source_addr"].nunique()),
        "unique_destinations": int(window["destination_addr"].nunique()),
        "ipv6_ratio": float(window["is_ipv6"].mean()),
        "window_size": int(window_size),
    }

    return features
