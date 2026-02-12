import json
from pathlib import Path
from typing import Dict, Generator, Tuple, Optional


class FileBus:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self, start_offset: int = 0) -> Generator[Tuple[Dict, int], None, None]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(start_offset)
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                record = json.loads(line)
                yield record, offset

    def publish(self, event: Dict) -> int:
        line = json.dumps(event) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            offset = f.tell()
            f.write(line)
        return offset
