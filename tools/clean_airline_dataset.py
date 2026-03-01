#!/usr/bin/env python3
"""Clean and normalize the 2018-2019 airline aggregate CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Optional


KEEP_COLUMNS = [
    "Year",
    "Month",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "Cancelled",
    "Diverted",
    "ArrDel15",
]

FILTER_ONLY_COLUMNS = [
    "ArrDelay",
]


def _normalize_headers(headers: Iterable[str]) -> Dict[str, str]:
    return {h.strip().lower(): h for h in headers}


def _get_required_field(
    row: Dict[str, str], header_map: Dict[str, str], logical_name: str
) -> str:
    source = header_map.get(logical_name.lower())
    if source is None:
        raise KeyError(logical_name)
    return row.get(source, "")


def _parse_int(value: str) -> Optional[int]:
    raw = value.strip()
    if raw == "":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _parse_float(value: str) -> Optional[float]:
    raw = value.strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_hhmm_to_minutes(value: str) -> Optional[int]:
    parsed = _parse_int(value)
    if parsed is None:
        return None
    if parsed < 0:
        return None
    hhmm = str(parsed).zfill(4)
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def clean_airline_csv(input_path: Path, output_path: Path, max_rows: Optional[int] = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "read_rows": 0,
        "written_rows": 0,
        "dropped_missing_arrdelay": 0,
        "dropped_cancelled": 0,
        "dropped_missing_required": 0,
        "dropped_invalid_numeric": 0,
        "dropped_invalid_time": 0,
    }

    with input_path.open("r", encoding="utf-8", newline="", errors="replace") as rf, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as wf:
        reader = csv.DictReader(rf)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        header_map = _normalize_headers(reader.fieldnames)
        for col in KEEP_COLUMNS + FILTER_ONLY_COLUMNS:
            if col.lower() not in header_map:
                raise ValueError(f"Missing required column in input: {col}")

        writer = csv.DictWriter(wf, fieldnames=KEEP_COLUMNS)
        writer.writeheader()

        for row in reader:
            if max_rows is not None and stats["read_rows"] >= max_rows:
                break
            stats["read_rows"] += 1

            cancelled = _parse_int(_get_required_field(row, header_map, "Cancelled"))
            if cancelled is None:
                stats["dropped_invalid_numeric"] += 1
                continue
            if cancelled == 1:
                stats["dropped_cancelled"] += 1
                continue

            arr_delay_raw = _get_required_field(row, header_map, "ArrDelay").strip()
            if arr_delay_raw == "":
                stats["dropped_missing_arrdelay"] += 1
                continue

            year = _parse_int(_get_required_field(row, header_map, "Year"))
            month = _parse_int(_get_required_field(row, header_map, "Month"))
            day_of_week = _parse_int(_get_required_field(row, header_map, "DayOfWeek"))
            elapsed = _parse_float(_get_required_field(row, header_map, "CRSElapsedTime"))
            distance = _parse_float(_get_required_field(row, header_map, "Distance"))
            diverted = _parse_int(_get_required_field(row, header_map, "Diverted"))
            arr_del_15 = _parse_int(_get_required_field(row, header_map, "ArrDel15"))

            if None in (year, month, day_of_week, elapsed, distance, diverted, arr_del_15):
                stats["dropped_invalid_numeric"] += 1
                continue

            if elapsed <= 0 or distance <= 0:
                stats["dropped_invalid_numeric"] += 1
                continue

            dep_minutes = _parse_hhmm_to_minutes(_get_required_field(row, header_map, "CRSDepTime"))
            arr_minutes = _parse_hhmm_to_minutes(_get_required_field(row, header_map, "CRSArrTime"))
            if dep_minutes is None or arr_minutes is None:
                stats["dropped_invalid_time"] += 1
                continue

            reporting_airline = _get_required_field(row, header_map, "Reporting_Airline").strip().upper()
            origin = _get_required_field(row, header_map, "Origin").strip().upper()
            dest = _get_required_field(row, header_map, "Dest").strip().upper()
            if not reporting_airline or not origin or not dest:
                stats["dropped_missing_required"] += 1
                continue

            cleaned = {
                "Year": str(year),
                "Month": str(month),
                "DayOfWeek": str(day_of_week),
                "Reporting_Airline": reporting_airline,
                "Origin": origin,
                "Dest": dest,
                "CRSDepTime": str(dep_minutes),
                "CRSArrTime": str(arr_minutes),
                "CRSElapsedTime": f"{elapsed:.2f}",
                "Distance": f"{distance:.2f}",
                "Cancelled": str(cancelled),
                "Diverted": str(diverted),
                "ArrDel15": "1" if arr_del_15 >= 1 else "0",
            }
            writer.writerow(cleaned)
            stats["written_rows"] += 1

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    for key, value in stats.items():
        print(f"{key}={value}")


def _read_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="", errors="replace") as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            yield row


def _top_values(path: Path, field: str, n: int) -> List[str]:
    counts: Counter[str] = Counter()
    for row in _read_rows(path):
        value = (row.get(field) or "").strip().upper()
        if value:
            counts[value] += 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:n]]


def encode_model_ready(cleaned_path: Path, model_output_path: Path, top_n_airports: int = 50) -> None:
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    airline_values = _top_values(cleaned_path, "Reporting_Airline", 1000)
    top_origins = _top_values(cleaned_path, "Origin", top_n_airports)
    top_dests = _top_values(cleaned_path, "Dest", top_n_airports)

    static_numeric = [
        "Year",
        "Month",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "Distance",
        "Cancelled",
        "Diverted",
        "ArrDel15",
    ]
    airline_cols = [f"airline__{v}" for v in airline_values]
    origin_cols = [f"origin__{v}" for v in top_origins] + ["origin__OTHER"]
    dest_cols = [f"dest__{v}" for v in top_dests] + ["dest__OTHER"]
    output_columns = static_numeric + airline_cols + origin_cols + dest_cols

    rows_written = 0
    with cleaned_path.open("r", encoding="utf-8", newline="", errors="replace") as rf, model_output_path.open(
        "w", encoding="utf-8", newline=""
    ) as wf:
        reader = csv.DictReader(rf)
        writer = csv.DictWriter(wf, fieldnames=output_columns)
        writer.writeheader()

        top_origin_set = set(top_origins)
        top_dest_set = set(top_dests)
        airline_set = set(airline_values)

        for row in reader:
            out = {}
            for col in static_numeric:
                out[col] = row.get(col, "")

            airline = (row.get("Reporting_Airline") or "").strip().upper()
            for v in airline_values:
                out[f"airline__{v}"] = "1" if airline == v else "0"
            if airline and airline not in airline_set:
                # This should not happen because airline_values contains all observed categories.
                pass

            origin = (row.get("Origin") or "").strip().upper()
            for v in top_origins:
                out[f"origin__{v}"] = "1" if origin == v else "0"
            out["origin__OTHER"] = "1" if origin not in top_origin_set else "0"

            dest = (row.get("Dest") or "").strip().upper()
            for v in top_dests:
                out[f"dest__{v}"] = "1" if dest == v else "0"
            out["dest__OTHER"] = "1" if dest not in top_dest_set else "0"

            writer.writerow(out)
            rows_written += 1

    print(f"Model-ready output: {model_output_path}")
    print(f"model_rows_written={rows_written}")
    print(f"airline_feature_count={len(airline_values)}")
    print(f"origin_top_n={len(top_origins)}")
    print(f"dest_top_n={len(top_dests)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and normalize airline 2018-2019 dataset.")
    parser.add_argument(
        "--input",
        default="data/airline/airline_2018_2019.csv",
        help="Path to aggregate source CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/airline/airline_2018_2019_clean.csv",
        help="Output path for cleaned CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for fast smoke runs.",
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help="Optional path for model-ready encoded CSV.",
    )
    parser.add_argument(
        "--top-n-airports",
        type=int,
        default=50,
        help="Top-N airports to one-hot for Origin and Dest; all others mapped to OTHER.",
    )
    parser.add_argument(
        "--encode-only-cleaned",
        action="store_true",
        help="Skip cleaning and encode an already-cleaned file from --input to --model-output.",
    )
    args = parser.parse_args()
    if args.encode_only_cleaned:
        if not args.model_output:
            raise ValueError("--model-output is required when --encode-only-cleaned is set.")
        encode_model_ready(Path(args.input), Path(args.model_output), args.top_n_airports)
        return

    clean_airline_csv(Path(args.input), Path(args.output), args.max_rows)
    if args.model_output:
        encode_model_ready(Path(args.output), Path(args.model_output), args.top_n_airports)


if __name__ == "__main__":
    main()
