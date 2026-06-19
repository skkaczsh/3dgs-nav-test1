from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_setup_rtx5070_tensorrt_defaults_to_cuda132_and_dry_run():
    text = (SCRIPTS / "setup_rtx5070_tensorrt_env.sh").read_text(encoding="utf-8")

    assert "TRT_VERSION=\"${TRT_VERSION:-11.0.0.114-1+cuda13.2}\"" in text
    assert "APPLY=\"${APPLY:-0}\"" in text
    assert "PYTHON_ONLY=\"${PYTHON_ONLY:-0}\"" in text
    assert "pip install is intentionally not invoked in dry-run mode." in text
    assert "--dry-run" not in text
    assert '"${APT_GET}" install -s' in text
    assert "APPLY=1" in text
    assert "cuda13.3" not in text


def test_setup_rtx5070_tensorrt_has_python_only_mode():
    text = (SCRIPTS / "setup_rtx5070_tensorrt_env.sh").read_text(encoding="utf-8")

    assert 'if [[ "${PYTHON_ONLY}" == "1" ]]' in text
    assert "skip apt TensorRT C++ runtime/dev install" in text
    assert "python_only_tensorrt_readiness.json" in text
    assert "check_rtx5070_tensorrt_readiness.py" in text


def test_setup_rtx5070_tensorrt_installs_cpp_runtime_dev_and_python_helpers():
    text = (SCRIPTS / "setup_rtx5070_tensorrt_env.sh").read_text(encoding="utf-8")

    for package in [
        "libnvinfer-dev",
        "libnvinfer-safe-headers-dev",
        "libnvinfer-bin",
        "libnvonnxparsers-dev",
        "python3-libnvinfer",
        "python3-libnvinfer-lean",
        "python3-libnvinfer-dispatch",
        "onnx",
        "polygraphy",
    ]:
        assert package in text

    assert '"tensorrt-cu13' not in text


def test_verify_rtx5070_tensorrt_builds_cpp_and_tiny_engine():
    text = (SCRIPTS / "verify_rtx5070_tensorrt_env.sh").read_text(encoding="utf-8")

    assert 'TRTEXEC="${TRTEXEC:-$(command -v trtexec || true)}"' in text
    assert "#include <NvInfer.h>" in text
    assert "builder_ok=1" in text
    assert "tiny_conv.onnx" in text
    assert "PASSED TensorRT.trtexec" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "tensorrt_python=missing_optional_for_cpp_smoke" in text
