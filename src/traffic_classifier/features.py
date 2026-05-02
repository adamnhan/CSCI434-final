from __future__ import annotations

from functools import lru_cache
import ipaddress
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
    df["source_is_private"] = df["source_addr"].apply(_is_private_address).astype(int)
    df["destination_is_private"] = df["destination_addr"].apply(_is_private_address).astype(int)

    parsed_ports = df["Info"].fillna("").astype(str).apply(_extract_ports)
    df["src_port"] = parsed_ports.str[0]
    df["dst_port"] = parsed_ports.str[1]

    # Approximate traffic direction using HTTPS ports, falling back to local/private endpoint behavior.
    port_outbound = (df["dst_port"] == 443) & (df["src_port"] != 443)
    port_inbound = (df["src_port"] == 443) & (df["dst_port"] != 443)
    address_outbound = (df["source_is_private"] == 1) & (df["destination_is_private"] == 0)
    address_inbound = (df["source_is_private"] == 0) & (df["destination_is_private"] == 1)
    df["is_outbound"] = (port_outbound | (~port_inbound & address_outbound)).astype(int)
    df["is_inbound"] = (port_inbound | (~port_outbound & address_inbound)).astype(int)
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


def build_session_dataset(capture_dir: str | Path) -> pd.DataFrame:
    """Convert each capture into one session-level feature row."""
    capture_dir = Path(capture_dir)
    rows: list[dict[str, float | int | str]] = []

    for csv_path in sorted(capture_dir.glob("*.csv")):
        df = load_capture_csv(csv_path)
        rows.append(_session_features(df))

    return pd.DataFrame(rows)


def _extract_ports(info: str) -> tuple[int | None, int | None]:
    match = PORT_RE.search(info)
    if not match:
        return (None, None)
    return (int(match.group("src")), int(match.group("dst")))


@lru_cache(maxsize=4096)
def _is_private_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_private


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


def _flow_features(window: pd.DataFrame) -> dict[str, float | int]:
    flows: dict[tuple[str, str, int, int, str], list[float]] = {}
    address_pairs: set[tuple[str, str]] = set()
    remote_endpoints: set[str] = set()

    for row in window.itertuples(index=False):
        src_port = -1 if pd.isna(row.src_port) else int(row.src_port)
        dst_port = -1 if pd.isna(row.dst_port) else int(row.dst_port)
        key = (row.source_addr, row.destination_addr, src_port, dst_port, row.protocol)
        stats = flows.setdefault(key, [0.0, 0.0, float(row.time), float(row.time)])
        stats[0] += 1.0
        stats[1] += float(row.length)
        stats[2] = min(stats[2], float(row.time))
        stats[3] = max(stats[3], float(row.time))

        address_pairs.add((row.source_addr, row.destination_addr))
        if int(row.source_is_private) == 0:
            remote_endpoints.add(row.source_addr)
        if int(row.destination_is_private) == 0:
            remote_endpoints.add(row.destination_addr)

    if not flows:
        return {
            "flow_count": 0,
            "flow_packets_mean": 0.0,
            "flow_packets_max": 0.0,
            "flow_bytes_mean": 0.0,
            "flow_bytes_max": 0.0,
            "flow_duration_mean": 0.0,
            "flow_duration_max": 0.0,
            "flow_short_ratio": 0.0,
            "top_flow_packet_share": 0.0,
            "top_flow_byte_share": 0.0,
            "address_pair_count": 0,
            "remote_endpoint_count": 0,
        }

    packets = np.array([stats[0] for stats in flows.values()])
    byte_counts = np.array([stats[1] for stats in flows.values()])
    durations = np.array([stats[3] - stats[2] for stats in flows.values()])
    total_packets = max(float(packets.sum()), 1.0)
    total_bytes = max(float(byte_counts.sum()), 1.0)
    return {
        "flow_count": int(len(flows)),
        "flow_packets_mean": float(packets.mean()),
        "flow_packets_max": float(packets.max()),
        "flow_bytes_mean": float(byte_counts.mean()),
        "flow_bytes_max": float(byte_counts.max()),
        "flow_duration_mean": float(durations.mean()),
        "flow_duration_max": float(durations.max()),
        "flow_short_ratio": float((packets <= 2).mean()),
        "top_flow_packet_share": float(packets.max() / total_packets),
        "top_flow_byte_share": float(byte_counts.max() / total_bytes),
        "address_pair_count": int(len(address_pairs)),
        "remote_endpoint_count": int(len(remote_endpoints)),
    }


def _burst_features(window: pd.DataFrame, threshold: float = 0.05) -> dict[str, float | int]:
    burst_packets: list[int] = []
    burst_bytes: list[float] = []
    burst_durations: list[float] = []
    current_packets = 0
    current_bytes = 0.0
    start_time = 0.0
    end_time = 0.0
    idle_gaps = 0

    for index, row in enumerate(window.itertuples(index=False)):
        delta = float(row.delta_time)
        if index > 0 and delta > threshold:
            burst_packets.append(current_packets)
            burst_bytes.append(current_bytes)
            burst_durations.append(end_time - start_time)
            current_packets = 0
            current_bytes = 0.0
            start_time = float(row.time)
            idle_gaps += 1
        elif index == 0:
            start_time = float(row.time)

        current_packets += 1
        current_bytes += float(row.length)
        end_time = float(row.time)

    if current_packets > 0:
        burst_packets.append(current_packets)
        burst_bytes.append(current_bytes)
        burst_durations.append(end_time - start_time)

    if not burst_packets:
        return {
            "burst_count_50ms": 0,
            "burst_packets_mean_50ms": 0.0,
            "burst_packets_max_50ms": 0.0,
            "burst_bytes_mean_50ms": 0.0,
            "burst_bytes_max_50ms": 0.0,
            "burst_duration_mean_50ms": 0.0,
            "burst_duration_max_50ms": 0.0,
            "idle_gap_ratio_50ms": 0.0,
        }

    burst_packet_values = np.array(burst_packets)
    burst_byte_values = np.array(burst_bytes)
    burst_duration_values = np.array(burst_durations)
    return {
        "burst_count_50ms": int(len(burst_packet_values)),
        "burst_packets_mean_50ms": float(burst_packet_values.mean()),
        "burst_packets_max_50ms": float(burst_packet_values.max()),
        "burst_bytes_mean_50ms": float(burst_byte_values.mean()),
        "burst_bytes_max_50ms": float(burst_byte_values.max()),
        "burst_duration_mean_50ms": float(burst_duration_values.mean()),
        "burst_duration_max_50ms": float(burst_duration_values.max()),
        "idle_gap_ratio_50ms": float(idle_gaps / max(len(window), 1)),
    }


def _session_burst_features(df: pd.DataFrame, threshold: float, suffix: str) -> dict[str, float | int]:
    burst_packets: list[int] = []
    burst_bytes: list[float] = []
    burst_durations: list[float] = []
    current_packets = 0
    current_bytes = 0.0
    start_time = 0.0
    end_time = 0.0
    idle_gaps = 0

    for index, row in enumerate(df.itertuples(index=False)):
        delta = float(row.delta_time)
        if index > 0 and delta > threshold:
            burst_packets.append(current_packets)
            burst_bytes.append(current_bytes)
            burst_durations.append(end_time - start_time)
            current_packets = 0
            current_bytes = 0.0
            start_time = float(row.time)
            idle_gaps += 1
        elif index == 0:
            start_time = float(row.time)

        current_packets += 1
        current_bytes += float(row.length)
        end_time = float(row.time)

    if current_packets > 0:
        burst_packets.append(current_packets)
        burst_bytes.append(current_bytes)
        burst_durations.append(end_time - start_time)

    if not burst_packets:
        return {
            f"burst_count_{suffix}": 0,
            f"burst_packets_mean_{suffix}": 0.0,
            f"burst_packets_p90_{suffix}": 0.0,
            f"burst_packets_max_{suffix}": 0.0,
            f"burst_bytes_mean_{suffix}": 0.0,
            f"burst_bytes_p90_{suffix}": 0.0,
            f"burst_bytes_max_{suffix}": 0.0,
            f"burst_duration_mean_{suffix}": 0.0,
            f"burst_duration_p90_{suffix}": 0.0,
            f"burst_duration_max_{suffix}": 0.0,
            f"idle_gap_ratio_{suffix}": 0.0,
        }

    packet_values = np.array(burst_packets)
    byte_values = np.array(burst_bytes)
    duration_values = np.array(burst_durations)
    return {
        f"burst_count_{suffix}": int(len(packet_values)),
        f"burst_packets_mean_{suffix}": float(packet_values.mean()),
        f"burst_packets_p90_{suffix}": float(np.quantile(packet_values, 0.9)),
        f"burst_packets_max_{suffix}": float(packet_values.max()),
        f"burst_bytes_mean_{suffix}": float(byte_values.mean()),
        f"burst_bytes_p90_{suffix}": float(np.quantile(byte_values, 0.9)),
        f"burst_bytes_max_{suffix}": float(byte_values.max()),
        f"burst_duration_mean_{suffix}": float(duration_values.mean()),
        f"burst_duration_p90_{suffix}": float(np.quantile(duration_values, 0.9)),
        f"burst_duration_max_{suffix}": float(duration_values.max()),
        f"idle_gap_ratio_{suffix}": float(idle_gaps / max(len(df), 1)),
    }


def _subwindow_features(window: pd.DataFrame) -> dict[str, float]:
    rows: dict[str, float] = {}
    split_indices = np.array_split(np.arange(len(window)), 3)
    segments = [window.iloc[indices] for indices in split_indices]
    for index, segment in enumerate(segments, start=1):
        prefix = f"third{index}"
        if len(segment) == 0:
            rows[f"{prefix}_length_mean"] = 0.0
            rows[f"{prefix}_length_p90"] = 0.0
            rows[f"{prefix}_delta_mean"] = 0.0
            rows[f"{prefix}_inbound_ratio"] = 0.0
            rows[f"{prefix}_outbound_ratio"] = 0.0
            rows[f"{prefix}_large_ratio"] = 0.0
            continue

        rows[f"{prefix}_length_mean"] = float(segment["length"].mean())
        rows[f"{prefix}_length_p90"] = float(segment["length"].quantile(0.9))
        rows[f"{prefix}_delta_mean"] = float(segment["delta_time"].mean())
        rows[f"{prefix}_inbound_ratio"] = float(segment["is_inbound"].mean())
        rows[f"{prefix}_outbound_ratio"] = float(segment["is_outbound"].mean())
        rows[f"{prefix}_large_ratio"] = _ratio(segment["length"], lambda s: s.gt(512))

    rows["length_mean_early_late_delta"] = rows["third1_length_mean"] - rows["third3_length_mean"]
    rows["inbound_ratio_early_late_delta"] = rows["third1_inbound_ratio"] - rows["third3_inbound_ratio"]
    rows["delta_mean_early_late_delta"] = rows["third1_delta_mean"] - rows["third3_delta_mean"]
    return rows


def _top_share(values: pd.Series, top_n: int) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    return float(values.sort_values(ascending=False).head(top_n).sum() / total)


def _session_flow_features(df: pd.DataFrame) -> dict[str, float | int]:
    flow_df = df.loc[
        :,
        ["source_addr", "destination_addr", "src_port", "dst_port", "protocol", "length", "time"],
    ].copy()
    flow_df["src_port"] = flow_df["src_port"].fillna(-1).astype(int)
    flow_df["dst_port"] = flow_df["dst_port"].fillna(-1).astype(int)
    flow_stats = flow_df.groupby(
        ["source_addr", "destination_addr", "src_port", "dst_port", "protocol"],
        dropna=False,
        sort=False,
    ).agg(
        packets=("length", "size"),
        bytes=("length", "sum"),
        start_time=("time", "min"),
        end_time=("time", "max"),
    )

    remote_endpoints = pd.concat(
        [
            df.loc[df["source_is_private"] == 0, "source_addr"],
            df.loc[df["destination_is_private"] == 0, "destination_addr"],
        ],
        ignore_index=True,
    )

    if flow_stats.empty:
        return {
            "flow_count": 0,
            "address_pair_count": 0,
            "remote_endpoint_count": int(remote_endpoints.nunique()),
            "flow_packets_mean": 0.0,
            "flow_packets_median": 0.0,
            "flow_packets_p90": 0.0,
            "flow_packets_max": 0.0,
            "flow_bytes_mean": 0.0,
            "flow_bytes_median": 0.0,
            "flow_bytes_p90": 0.0,
            "flow_bytes_max": 0.0,
            "flow_duration_mean": 0.0,
            "flow_duration_p90": 0.0,
            "flow_duration_max": 0.0,
            "flow_short_ratio": 0.0,
            "top1_flow_byte_share": 0.0,
            "top3_flow_byte_share": 0.0,
            "top5_flow_byte_share": 0.0,
        }

    flow_stats["duration"] = flow_stats["end_time"] - flow_stats["start_time"]
    address_pairs = df.loc[:, ["source_addr", "destination_addr"]].drop_duplicates()
    return {
        "flow_count": int(len(flow_stats)),
        "address_pair_count": int(len(address_pairs)),
        "remote_endpoint_count": int(remote_endpoints.nunique()),
        "flow_packets_mean": float(flow_stats["packets"].mean()),
        "flow_packets_median": float(flow_stats["packets"].median()),
        "flow_packets_p90": float(flow_stats["packets"].quantile(0.9)),
        "flow_packets_max": float(flow_stats["packets"].max()),
        "flow_bytes_mean": float(flow_stats["bytes"].mean()),
        "flow_bytes_median": float(flow_stats["bytes"].median()),
        "flow_bytes_p90": float(flow_stats["bytes"].quantile(0.9)),
        "flow_bytes_max": float(flow_stats["bytes"].max()),
        "flow_duration_mean": float(flow_stats["duration"].mean()),
        "flow_duration_p90": float(flow_stats["duration"].quantile(0.9)),
        "flow_duration_max": float(flow_stats["duration"].max()),
        "flow_short_ratio": float(flow_stats["packets"].le(2).mean()),
        "top1_flow_byte_share": _top_share(flow_stats["bytes"], 1),
        "top3_flow_byte_share": _top_share(flow_stats["bytes"], 3),
        "top5_flow_byte_share": _top_share(flow_stats["bytes"], 5),
    }


def _session_segment_features(df: pd.DataFrame, seconds: float, suffix: str) -> dict[str, float | int]:
    start_time = float(df["time"].min())
    segment = df.loc[df["time"].le(start_time + seconds)]
    if segment.empty:
        return {
            f"first_{suffix}_packet_count": 0,
            f"first_{suffix}_byte_share": 0.0,
            f"first_{suffix}_length_mean": 0.0,
            f"first_{suffix}_length_p90": 0.0,
            f"first_{suffix}_inbound_ratio": 0.0,
            f"first_{suffix}_outbound_ratio": 0.0,
            f"first_{suffix}_tcp_ratio": 0.0,
            f"first_{suffix}_udp_ratio": 0.0,
        }

    total_bytes = max(float(df["length"].sum()), 1.0)
    return {
        f"first_{suffix}_packet_count": int(len(segment)),
        f"first_{suffix}_byte_share": float(segment["length"].sum() / total_bytes),
        f"first_{suffix}_length_mean": float(segment["length"].mean()),
        f"first_{suffix}_length_p90": float(segment["length"].quantile(0.9)),
        f"first_{suffix}_inbound_ratio": float(segment["is_inbound"].mean()),
        f"first_{suffix}_outbound_ratio": float(segment["is_outbound"].mean()),
        f"first_{suffix}_tcp_ratio": _ratio(segment["protocol"], lambda s: s.eq("tcp")),
        f"first_{suffix}_udp_ratio": _ratio(segment["protocol"], lambda s: s.eq("udp")),
    }


def _session_features(df: pd.DataFrame) -> dict[str, float | int | str]:
    lengths = df["length"]
    deltas = df["delta_time"]
    protocols = df["protocol"]
    directions = df["direction"]
    direction_runs = _run_lengths(directions)
    nonzero_directions = directions[directions != 0]
    direction_changes = float((nonzero_directions != nonzero_directions.shift()).sum() - 1) if len(nonzero_directions) > 1 else 0.0
    inbound_bytes = float(df.loc[df["is_inbound"] == 1, "length"].sum())
    outbound_bytes = float(df.loc[df["is_outbound"] == 1, "length"].sum())
    total_bytes = max(float(lengths.sum()), 1.0)
    duration = float(df["time"].max() - df["time"].min())

    features: dict[str, float | int | str] = {
        "label": str(df["label"].iloc[0]),
        "session_id": int(df["session_id"].iloc[0]),
        "capture_name": str(df["capture_name"].iloc[0]),
        "packet_count": int(len(df)),
        "duration_sec": duration,
        "packet_rate": float(len(df) / max(duration, 1e-6)),
        "total_bytes": float(lengths.sum()),
        "byte_rate": float(lengths.sum() / max(duration, 1e-6)),
        "length_mean": float(lengths.mean()),
        "length_std": _safe_std(lengths),
        "length_median": float(lengths.median()),
        "length_p75": float(lengths.quantile(0.75)),
        "length_p90": float(lengths.quantile(0.9)),
        "length_p95": float(lengths.quantile(0.95)),
        "length_entropy": _entropy(lengths),
        "length_small_ratio": _ratio(lengths, lambda s: s.le(128)),
        "length_medium_ratio": _ratio(lengths, lambda s: s.gt(128) & s.le(512)),
        "length_large_ratio": _ratio(lengths, lambda s: s.gt(512) & s.le(1200)),
        "length_jumbo_ratio": _ratio(lengths, lambda s: s.gt(1200)),
        "delta_mean": float(deltas.mean()),
        "delta_std": _safe_std(deltas),
        "delta_max": float(deltas.max()),
        "delta_p90": float(deltas.quantile(0.9)),
        "delta_p95": float(deltas.quantile(0.95)),
        "delta_fast_ratio": _ratio(deltas, lambda s: s.gt(0.0) & s.le(0.01)),
        "delta_medium_ratio": _ratio(deltas, lambda s: s.gt(0.01) & s.le(0.1)),
        "delta_slow_ratio": _ratio(deltas, lambda s: s.gt(0.1)),
        "udp_ratio": _ratio(protocols, lambda s: s.eq("udp")),
        "tcp_ratio": _ratio(protocols, lambda s: s.eq("tcp")),
        "quic_ratio": _ratio(protocols, lambda s: s.eq("quic")),
        "tls12_ratio": _ratio(protocols, lambda s: s.eq("tlsv1.2")),
        "tls13_ratio": _ratio(protocols, lambda s: s.eq("tlsv1.3")),
        "outbound_ratio": float(df["is_outbound"].mean()),
        "inbound_ratio": float(df["is_inbound"].mean()),
        "direction_balance": float(df["is_outbound"].mean() - df["is_inbound"].mean()),
        "inbound_byte_share": float(inbound_bytes / total_bytes),
        "outbound_byte_share": float(outbound_bytes / total_bytes),
        "direction_change_ratio": float(direction_changes / max(len(nonzero_directions) - 1, 1)),
        "direction_run_mean": _run_stat(direction_runs, "mean"),
        "direction_run_max": _run_stat(direction_runs, "max"),
        "dst_443_ratio": _ratio(df["dst_port"], lambda s: s.eq(443)),
        "src_443_ratio": _ratio(df["src_port"], lambda s: s.eq(443)),
        "unique_src_ports": int(df["src_port"].dropna().nunique()),
        "unique_dst_ports": int(df["dst_port"].dropna().nunique()),
        "unique_sources": int(df["source_addr"].nunique()),
        "unique_destinations": int(df["destination_addr"].nunique()),
        "private_source_ratio": float(df["source_is_private"].mean()),
        "private_destination_ratio": float(df["destination_is_private"].mean()),
        "ipv6_ratio": float(df["is_ipv6"].mean()),
    }

    features.update(_session_flow_features(df))
    features.update(_session_burst_features(df, 0.05, "50ms"))
    features.update(_session_burst_features(df, 0.25, "250ms"))
    features.update(_session_segment_features(df, 5.0, "5s"))
    features.update(_session_segment_features(df, 10.0, "10s"))
    return features


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
        "private_source_ratio": float(window["source_is_private"].mean()),
        "private_destination_ratio": float(window["destination_is_private"].mean()),
        "ipv6_ratio": float(window["is_ipv6"].mean()),
        "window_size": int(window_size),
    }

    features.update(_flow_features(window))
    features.update(_burst_features(window))
    features.update(_subwindow_features(window))

    return features
