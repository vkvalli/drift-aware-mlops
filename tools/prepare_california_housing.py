import argparse
import csv
from pathlib import Path


def prepare(out_csv: Path, target_column: str = "MedHouseVal") -> None:
    try:
        from sklearn.datasets import fetch_california_housing
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for California Housing preparation"
        ) from exc

    ds = fetch_california_housing(as_frame=True)
    frame = ds.frame.copy()
    if "MedHouseVal" != target_column:
        frame = frame.rename(columns={"MedHouseVal": target_column})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(frame.columns.tolist())
        for row in frame.itertuples(index=False):
            writer.writerow(list(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare California Housing CSV for regression pipeline"
    )
    parser.add_argument("--out", default="data/california_housing.csv")
    parser.add_argument("--target-column", default="MedHouseVal")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare(Path(args.out), target_column=args.target_column)


if __name__ == "__main__":
    main()
