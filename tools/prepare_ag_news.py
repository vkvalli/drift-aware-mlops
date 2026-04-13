import argparse
import csv
import random
from pathlib import Path


def _write_rows(out_csv: Path, rows: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "description", "class"])
        writer.writeheader()
        writer.writerows(rows)


def _from_existing_csv(input_csv: Path, out_csv: Path) -> None:
    rows: list[dict] = []
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = str(row.get("title", "")).strip()
            desc = str(row.get("description", row.get("text", ""))).strip()
            klass = str(row.get("class", row.get("label", ""))).strip()
            if not klass:
                continue
            rows.append({"title": title, "description": desc, "class": klass})
    _write_rows(out_csv, rows)


def _from_huggingface(out_csv: Path, rows: int, seed: int) -> None:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face datasets is required for --source hf. "
            "Install with: pip install datasets"
        ) from exc

    ds = load_dataset("ag_news", split="train+test")
    ds = ds.shuffle(seed=seed)
    if rows > 0:
        ds = ds.select(range(min(rows, len(ds))))

    out_rows: list[dict] = []
    for rec in ds:
        # Keep class IDs as 1..4 so the text label parser remains consistent.
        cls = int(rec["label"]) + 1
        raw_text = str(rec.get("text", "")).strip()
        title = str(rec.get("title", "")).strip()
        description = str(rec.get("description", "")).strip()
        if not title and raw_text:
            # AG News HF examples usually have a single text field, so derive
            # a compact title surrogate from the first sentence.
            title = raw_text.split(". ", 1)[0].strip()
        if not description and raw_text:
            description = raw_text
        if not title and not description:
            continue
        out_rows.append({"title": title, "description": description, "class": str(cls)})
    _write_rows(out_csv, out_rows)


def _synthetic(out_csv: Path, rows: int, seed: int) -> None:
    rng = random.Random(seed)
    labels = [1, 2, 3, 4]
    templates = {
        1: ("World", "Global politics and diplomacy update"),
        2: ("Sports", "Match result and player performance recap"),
        3: ("Business", "Market update and quarterly earnings report"),
        4: ("Sci/Tech", "AI research and product launch coverage"),
    }
    out_rows: list[dict] = []
    for i in range(rows):
        cls = labels[i % len(labels)]
        title_base, desc_base = templates[cls]
        out_rows.append(
            {
                "title": f"{title_base} headline {i}",
                "description": f"{desc_base}. synthetic token {rng.randint(0, 9999)}",
                "class": str(cls),
            }
        )
    _write_rows(out_csv, out_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare AG News CSV for text classification pipeline"
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "csv", "hf"],
        default="synthetic",
        help="Data source: synthetic generator, existing CSV, or Hugging Face AG News",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional input AG News CSV (title/description/class)",
    )
    parser.add_argument("--out", default="data/ag_news.csv")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_csv = Path(args.out)
    if args.source == "csv":
        if not args.input:
            raise ValueError("--input is required when --source csv")
        _from_existing_csv(Path(args.input), out_csv)
    elif args.source == "hf":
        _from_huggingface(out_csv, rows=args.rows, seed=args.seed)
    elif args.input:
        _from_existing_csv(Path(args.input), out_csv)
    else:
        _synthetic(out_csv, rows=args.rows, seed=args.seed)


if __name__ == "__main__":
    main()
