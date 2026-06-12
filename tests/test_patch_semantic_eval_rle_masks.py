import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch_semantic_eval_rle_masks.py"


def test_patch_adds_rle_decoder(tmp_path):
    run_eval = tmp_path / "run_eval.py"
    run_eval.write_text(
        """from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

@dataclass
class Mask:
    segmentation: np.ndarray
    area: int
    score: float
    bbox: list[int]
    source: str

def decode_sam2_masks(path: Path) -> list[Mask]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    items = data.get("masks", data if isinstance(data, list) else [])
    masks: list[Mask] = []
    for item in items:
        seg = np.array(item["segmentation"], dtype=bool)
        masks.append(Mask(segmentation=seg, area=int(seg.sum()), score=0.5, bbox=[0,0,0,0], source="sam2"))
    return masks

class SAM3Runner:
    pass
""",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(PATCHER), "--run-eval", str(run_eval)],
        check=True,
        text=True,
        capture_output=True,
    )
    text = run_eval.read_text(encoding="utf-8")
    assert "decode_sam_segmentation" in text

    spec = importlib.util.spec_from_file_location("patched_run_eval", run_eval)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["patched_run_eval"] = module
    spec.loader.exec_module(module)
    mask = module.decode_sam_segmentation({"size": [2, 3], "counts": [1, 2, 3]})
    assert mask.tolist() == [[False, True, False], [True, False, False]]
