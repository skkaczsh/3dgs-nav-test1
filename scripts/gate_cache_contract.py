"""Shared helpers for validating cached promotion gate JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


DEFAULT_STALE_KEYS: tuple[str, ...] = ("status", "candidate", "metrics", "reasons")


def resolve_relative_path(value: Any, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def stale_gate_reasons(
    cached: dict[str, Any],
    recomputed: dict[str, Any] | None,
    *,
    prefix: str,
    keys: Iterable[str] = DEFAULT_STALE_KEYS,
) -> list[str]:
    if recomputed is None:
        return []
    return [f"{prefix}_stale_{key}" for key in keys if cached.get(key) != recomputed.get(key)]
