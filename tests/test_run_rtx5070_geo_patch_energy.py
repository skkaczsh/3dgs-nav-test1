from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_rtx5070_geo_patch_energy.sh"


def test_rtx5070_geo_patch_energy_defaults_to_dense_voxel_input() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "dense_las_voxel003_20260624/dense_las_voxel003_binary.ply" in text
    assert "frame_object_points_stride10.ply" not in text.split('INPUT_PLY="${INPUT_PLY:-', 1)[1].split('}"', 1)[0]


def test_rtx5070_geo_patch_energy_rejects_stride_viewer_input() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={
            "PATH": "/bin:/usr/bin",
            "INPUT_PLY": "/tmp/frame_object_points_stride10.ply",
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "forbidden production input" in result.stderr


def test_rtx5070_geo_patch_energy_dry_run_reports_dense_input() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={
            "PATH": "/bin:/usr/bin",
            "RUN": "0",
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "dry_run=1" in result.stdout
    assert "dense_las_voxel003_20260624/dense_las_voxel003_binary.ply" in result.stdout
