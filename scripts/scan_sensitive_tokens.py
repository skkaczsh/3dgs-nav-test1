#!/usr/bin/env python3
"""Scan repository text files for high-risk credential patterns."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
}


PATTERNS = {
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "github_classic_token": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "github_fine_grained_token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
}


def should_skip(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part in DEFAULT_EXCLUDE_DIRS for part in rel.parts)


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not should_skip(path, root):
            yield path


def scan_file(path: Path, root: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
    except OSError:
        return []
    findings = []
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            findings.append({"path": rel, "line": line_no, "kind": name})
    return findings


def scan(root: Path, max_file_bytes: int = 2_000_000) -> dict:
    root = root.resolve()
    findings = []
    checked = 0
    skipped_large = 0
    for path in iter_files(root):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if max_file_bytes > 0 and size > max_file_bytes:
            skipped_large += 1
            continue
        checked += 1
        findings.extend(scan_file(path, root))
    return {
        "root": str(root),
        "checked_files": checked,
        "skipped_large_files": skipped_large,
        "max_file_bytes": int(max_file_bytes),
        "finding_count": len(findings),
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = scan(args.root, max_file_bytes=args.max_file_bytes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["finding_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
