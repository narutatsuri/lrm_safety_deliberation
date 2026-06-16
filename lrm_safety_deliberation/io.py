"""Small JSON / JSONL I/O helpers used across the library.

The research codebase reimplemented these readers in nearly every script; they
are consolidated here so there is exactly one (UTF-8, blank-line-tolerant,
atomic-write) implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "read_jsonl",
    "iter_jsonl",
    "read_concat_json",
    "write_jsonl",
    "append_jsonl",
    "read_json",
    "write_json",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts (blank lines skipped)."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file row by row (use for files too large to hold in memory)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_concat_json(path: str | Path) -> list[dict[str, Any]]:
    """Read a file of concatenated JSON objects, tolerant of formatting.

    Handles both newline-delimited JSONL and pretty-printed (multi-line) objects
    written back-to-back.  Some pipeline ``results.jsonl`` files are written in
    the latter form; this reader parses either.
    """
    txt = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    idx, n = 0, len(txt)
    while idx < n:
        while idx < n and txt[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(txt, idx)
        out.append(obj)
        idx = end
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write rows to a JSONL file (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows to a JSONL file (creates parent dirs); used by resumable jobs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    """Atomically write an object as pretty JSON (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)
