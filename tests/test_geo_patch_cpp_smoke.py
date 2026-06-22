from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def test_geo_patch_cpp_smoke_builds_and_runs() -> None:
    if shutil.which("g++") is None:
        return

    repo = Path(__file__).resolve().parents[1]
    build = subprocess.run(
        ["bash", str(repo / "scripts" / "build_geo_patch_cpp_smoke.sh")],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    binary = Path(build.stdout.strip().splitlines()[-1])
    run = subprocess.run([str(binary)], cwd=repo, check=True, text=True, capture_output=True)
    assert "geo_patch_region_model_smoke ok" in run.stdout
