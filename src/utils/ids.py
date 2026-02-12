from dataclasses import dataclass


def next_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:06d}"


@dataclass
class IdCounter:
    prefix: str
    start: int = 1

    def __post_init__(self) -> None:
        self._current = self.start

    def next(self) -> str:
        value = next_id(self.prefix, self._current)
        self._current += 1
        return value
