#!/usr/bin/env python3
"""Validate a lightweight dataset delivery package."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import tarfile
from pathlib import Path


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


def validate_relative_links(base_dir: Path, html_path: Path) -> list[str]:
    parser = LinkParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for href in parser.links:
        if href.startswith(("http://", "https://", "mailto:", "#", "/")):
            continue
        path = (base_dir / href).resolve()
        if not path.exists():
            missing.append(href)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--package-dir", type=Path, default=root / "dataset_delivery_0000_0999")
    parser.add_argument("--tgz", type=Path, default=root / "dataset_delivery_0000_0999.tgz")
    parser.add_argument("--output", type=Path, default=root / "dataset_delivery_0000_0999_validation.json")
    args = parser.parse_args()

    errors = []
    package_manifest = args.package_dir / "package_manifest.json"
    large_files = args.package_dir / "large_files.json"
    readme = args.package_dir / "README.md"
    qa_index_md = args.package_dir / "qa_index.md"
    qa_index_html = args.package_dir / "qa_index.html"
    for path in [package_manifest, large_files, readme, qa_index_md, qa_index_html]:
        if not path.exists() or path.stat().st_size <= 0:
            errors.append(f"missing or empty package file: {path}")
    if qa_index_html.exists():
        for href in validate_relative_links(args.package_dir, qa_index_html):
            errors.append(f"qa_index.html has missing relative link: {href}")

    manifest = {}
    if package_manifest.exists():
        manifest = json.loads(package_manifest.read_text(encoding="utf-8"))
        if not manifest.get("passed"):
            errors.append("package_manifest passed is false")
        for row in manifest.get("files", []):
            if row.get("packaged"):
                packaged_path = args.package_dir / row.get("package_path", "")
                if not packaged_path.exists() or packaged_path.stat().st_size <= 0:
                    errors.append(f"missing packaged artifact: {packaged_path}")
            elif row.get("required"):
                source = Path(row.get("path", ""))
                if not source.exists():
                    errors.append(f"missing referenced required artifact: {source}")

    if not args.tgz.exists() or args.tgz.stat().st_size <= 0:
        errors.append(f"missing tgz: {args.tgz}")
    else:
        try:
            with tarfile.open(args.tgz, "r:gz") as tf:
                names = set(tf.getnames())
            if f"{args.package_dir.name}/package_manifest.json" not in names:
                errors.append("tgz missing package_manifest.json")
            if f"{args.package_dir.name}/README.md" not in names:
                errors.append("tgz missing README.md")
            if f"{args.package_dir.name}/qa_index.html" not in names:
                errors.append("tgz missing qa_index.html")
            if f"{args.package_dir.name}/qa_index.md" not in names:
                errors.append("tgz missing qa_index.md")
        except tarfile.TarError as exc:
            errors.append(f"invalid tgz: {exc}")

    result = {
        "package_dir": str(args.package_dir),
        "tgz": str(args.tgz),
        "passed": not errors,
        "errors": errors,
        "packaged_file_count": sum(1 for row in manifest.get("files", []) if row.get("packaged")),
        "large_file_count": len(manifest.get("large_files", [])),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
