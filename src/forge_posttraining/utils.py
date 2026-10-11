"""Exact decoded-prefix alignment and JSONL utilities."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional


class TokenAligner:
    def __init__(self, tokenizer, ids: list[int], skip_special_tokens: bool):
        self.tokenizer, self.ids, self.skip = (tokenizer, list(ids), skip_special_tokens)
        self._cache: dict[int, str] = {}
        self.text = self.decode(len(self.ids))

    def decode(self, t: int) -> str:
        if t not in self._cache:
            self._cache[t] = self.tokenizer.decode(self.ids[:t], skip_special_tokens=self.skip)
        return self._cache[t]

    def tokens_for_prefix(self, prefix: str) -> Optional[int]:
        if not self.ids:
            return None
        lo, hi = (1, len(self.ids))
        while lo < hi:
            mid = (lo + hi) // 2
            if len(self.decode(mid)) >= len(prefix):
                hi = mid
            else:
                lo = mid + 1
        text = self.decode(lo)
        if text.startswith(prefix) and (not text[len(prefix) :].strip()):
            return lo
        return None


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSON objects with line-number diagnostics."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at line {number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write a caller-selected output without overwriting existing evidence."""
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
