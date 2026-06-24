from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_rtx5070_geo_patch_route.sh"


def test_legacy_geo_patch_route_dry_run_warns_about_dense_replacement() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={"PATH": "/bin:/usr/bin", "RUN": "0"},
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "deprecated_viewer_input_route=1" in result.stdout
    assert "dense_replacement=scripts/run_rtx5070_geo_patch_energy.sh" in result.stdout


def test_legacy_geo_patch_route_refuses_run_without_explicit_override() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={"PATH": "/bin:/usr/bin", "RUN": "1"},
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "refusing to run deprecated viewer-input route" in result.stderr


def test_legacy_geo_patch_route_documents_explicit_override() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'ALLOW_VIEWER_INPUT_ROUTE="${ALLOW_VIEWER_INPUT_ROUTE:-0}"' in text
    assert "RUN=1 ALLOW_VIEWER_INPUT_ROUTE=1" in text
