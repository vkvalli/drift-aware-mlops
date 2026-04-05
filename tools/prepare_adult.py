import argparse
import csv
import random
from pathlib import Path


def _write_rows(out_csv: Path, rows: list[dict], headers: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _from_existing_csv(input_csv: Path, out_csv: Path, target_column: str) -> None:
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        headers = reader.fieldnames or []
    if target_column not in headers:
        raise ValueError(f"Target column '{target_column}' not found in input CSV")
    _write_rows(out_csv, rows, headers)


def _from_openml(out_csv: Path, rows: int | None, target_column: str) -> None:
    try:
        from sklearn.datasets import fetch_openml
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for OpenML Adult dataset preparation"
        ) from exc

    ds = fetch_openml(name="adult", version=2, as_frame=True)
    frame = ds.frame.copy()

    target_name = ds.target.name if ds.target is not None else "class"
    if target_name not in frame.columns:
        raise ValueError(f"Expected OpenML target column '{target_name}' in frame")

    frame = frame.dropna().copy()
    frame[target_name] = (
        frame[target_name]
        .astype(str)
        .str.strip()
        .map({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1})
    )
    frame = frame.dropna(subset=[target_name]).copy()
    frame[target_name] = frame[target_name].astype(int)

    if target_name != target_column:
        frame = frame.rename(columns={target_name: target_column})

    if rows is not None and rows > 0:
        frame = frame.head(rows)

    _write_rows(
        out_csv,
        rows=frame.to_dict(orient="records"),
        headers=frame.columns.tolist(),
    )


def _synthetic(out_csv: Path, rows: int, seed: int, target_column: str) -> None:
    rng = random.Random(seed)
    headers = ["age", "education_num", "hours_per_week", "sex", target_column]
    out_rows: list[dict] = []
    for _ in range(rows):
        sex = "Male" if rng.random() > 0.45 else "Female"
        high_income = (
            1
            if (
                rng.randint(18, 65) > 35
                and rng.random() > (0.58 if sex == "Female" else 0.42)
            )
            else 0
        )
        out_rows.append(
            {
                "age": str(rng.randint(18, 65)),
                "education_num": str(rng.randint(8, 16)),
                "hours_per_week": str(rng.randint(20, 60)),
                "sex": sex,
                target_column: str(high_income),
            }
        )
    _write_rows(out_csv, out_rows, headers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Adult dataset CSV for fairness evaluation"
    )
    parser.add_argument("--input", default="", help="Optional existing Adult CSV")
    parser.add_argument(
        "--source",
        choices=["openml", "synthetic"],
        default="openml",
        help="Dataset source when --input is not provided",
    )
    parser.add_argument("--out", default="data/adult.csv")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-column", default="class")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_csv = Path(args.out)
    if args.input:
        _from_existing_csv(Path(args.input), out_csv, target_column=args.target_column)
    elif args.source == "openml":
        _from_openml(out_csv, rows=args.rows, target_column=args.target_column)
    else:
        _synthetic(
            out_csv, rows=args.rows, seed=args.seed, target_column=args.target_column
        )


if __name__ == "__main__":
    main()
