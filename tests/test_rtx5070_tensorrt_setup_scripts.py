from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_setup_rtx5070_tensorrt_defaults_to_cuda132_and_dry_run():
    text = (SCRIPTS / "setup_rtx5070_tensorrt_env.sh").read_text(encoding="utf-8")

    assert "TRT_VERSION=\"${TRT_VERSION:-11.0.0.114-1+cuda13.2}\"" in text
    assert "PY_TRT_VERSION=\"${PY_TRT_VERSION:-11.0.0.114}\"" in text
    assert "APPLY=\"${APPLY:-0}\"" in text
    assert "pip install is intentionally not invoked in dry-run mode" in text
    assert "pip index versions tensorrt-cu13" in text
    assert "--dry-run" not in text
    assert '"${APT_GET}" install -s' in text
    assert "APPLY=1" in text
    assert "cuda13.3" not in text


def test_setup_rtx5070_tensorrt_installs_cpp_runtime_dev_and_python_helpers():
    text = (SCRIPTS / "setup_rtx5070_tensorrt_env.sh").read_text(encoding="utf-8")

    for package in [
        "libnvinfer-dev",
        "libnvinfer-safe-headers-dev",
        "libnvinfer-bin",
        "libnvonnxparsers-dev",
        "tensorrt-cu13",
        "onnx",
        "polygraphy",
        "cuda-python",
    ]:
        assert package in text


def test_verify_rtx5070_tensorrt_builds_cpp_and_tiny_engine():
    text = (SCRIPTS / "verify_rtx5070_tensorrt_env.sh").read_text(encoding="utf-8")

    assert "#include <NvInfer.h>" in text
    assert "builder_ok=1" in text
    assert "tiny_conv.onnx" in text
    assert "PASSED TensorRT.trtexec" in text
    assert "CUDA_VISIBLE_DEVICES" in text
