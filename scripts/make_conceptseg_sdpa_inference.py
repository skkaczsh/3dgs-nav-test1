#!/usr/bin/env python3
"""Create a ConceptSeg-R1 inference script that avoids optional flash-attn.

The upstream single-example script hardcodes flash_attention_2 and can import
an editable-installed namespace `sam3` package before the repo-local SAM3 tree.
This generator produces a local smoke-test script that:

- prepends the repo-local `sam3-main` directory to `sys.path`;
- removes an already-imported namespace `sam3` module when it has no `__file__`;
- switches model loading to `attn_implementation="sdpa"`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


BOOTSTRAP_TEMPLATE = '''\
import sys
from pathlib import Path

_CONCEPTSEG_REPO = Path({repo!r})
_SAM3_MAIN = _CONCEPTSEG_REPO / "sam3-main"
if _SAM3_MAIN.exists():
    sys.path.insert(0, str(_SAM3_MAIN))
_sam3 = sys.modules.get("sam3")
if _sam3 is not None and getattr(_sam3, "__file__", None) is None:
    del sys.modules["sam3"]
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    repo = args.source.resolve().parents[2]
    bootstrap = BOOTSTRAP_TEMPLATE.format(repo=str(repo))
    text = text.replace(
        'attn_implementation="flash_attention_2"',
        'attn_implementation="sdpa"',
    )

    marker = "import os\n"
    if marker not in text:
        raise SystemExit(f"cannot find insertion marker in {args.source}")
    text = text.replace(marker, bootstrap + "\n" + marker, 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    args.output.chmod(0o755)
    print(args.output)


if __name__ == "__main__":
    main()
