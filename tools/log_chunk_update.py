import argparse
from datetime import datetime, timezone
from pathlib import Path


def _split_csv_field(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_bullets(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def append_chunk_log(
    log_path: Path,
    chunk_id: str,
    files_changed: list[str],
    why_changed: list[str],
    verify_commands: list[str],
    expected_output: list[str],
    new_metrics: list[str],
    new_plots: list[str],
    new_log_fields: list[str],
    limitations: list[str],
    next_chunk: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    section = f"""
## Chunk {chunk_id} ({timestamp})

### What Changed
{_format_bullets(files_changed)}

### Why It Changed
{_format_bullets(why_changed)}

### What To Verify
{_format_bullets(verify_commands)}

### Expected Output
{_format_bullets(expected_output)}

### New Metrics
{_format_bullets(new_metrics)}

### New Plots
{_format_bullets(new_plots)}

### New Log Fields
{_format_bullets(new_log_fields)}

### Known Limitations
{_format_bullets(limitations)}

### Next Chunk
- {next_chunk or "none"}
""".strip()

    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    log_path.write_text(existing + "\n\n" + section + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a thesis-ready implementation chunk entry to docs/research_log.md"
    )
    parser.add_argument("--log-path", default="docs/research_log.md")
    parser.add_argument("--chunk-id", required=True)
    parser.add_argument(
        "--files-changed", required=True, help="Comma-separated file list"
    )
    parser.add_argument(
        "--why-changed", required=True, help="Comma-separated reasons/requirements"
    )
    parser.add_argument(
        "--verify-commands", required=True, help="Comma-separated commands"
    )
    parser.add_argument(
        "--expected-output",
        required=True,
        help="Comma-separated expected output summaries",
    )
    parser.add_argument("--new-metrics", default="")
    parser.add_argument("--new-plots", default="")
    parser.add_argument("--new-log-fields", default="")
    parser.add_argument("--limitations", default="")
    parser.add_argument("--next-chunk", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    append_chunk_log(
        log_path=Path(args.log_path),
        chunk_id=args.chunk_id,
        files_changed=_split_csv_field(args.files_changed),
        why_changed=_split_csv_field(args.why_changed),
        verify_commands=_split_csv_field(args.verify_commands),
        expected_output=_split_csv_field(args.expected_output),
        new_metrics=_split_csv_field(args.new_metrics),
        new_plots=_split_csv_field(args.new_plots),
        new_log_fields=_split_csv_field(args.new_log_fields),
        limitations=_split_csv_field(args.limitations),
        next_chunk=args.next_chunk.strip(),
    )


if __name__ == "__main__":
    main()
