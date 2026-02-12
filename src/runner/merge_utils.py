import json
from pathlib import Path


def append_file_to_canonical(src_path: Path, canonical_path: Path) -> None:
    """
    Append all JSONL lines from src_path to canonical_path unchanged.
    Assumes src_path already has full event dicts including event_type and logged_at.
    """
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    if not src_path.exists():
        return
    with src_path.open("r", encoding="utf-8") as sf, canonical_path.open(
        "a", encoding="utf-8"
    ) as cf:
        for line in sf:
            json.loads(line)
            cf.write(line.rstrip("\n") + "\n")
