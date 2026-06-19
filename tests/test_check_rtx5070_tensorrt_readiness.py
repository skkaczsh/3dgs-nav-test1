from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "check_rtx5070_tensorrt_readiness_for_test",
    SCRIPTS / "check_rtx5070_tensorrt_readiness.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_parse_sections_extracts_repeated_libs_and_gpu():
    sections = module.parse_sections(
        "\n".join(
            [
                "section=gpu",
                "0, NVIDIA GeForce RTX 5070 Ti, 575.57.08, 16303 MiB",
                "section=cpp",
                "header=/usr/include/NvInfer.h",
                "lib=libnvinfer.so.10",
                "lib=libnvonnxparser.so.10",
            ]
        )
    )

    assert sections["gpu"][0]["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert sections["cpp"]["header"] == "/usr/include/NvInfer.h"
    assert sections["cpp"]["lib"] == ["libnvinfer.so.10", "libnvonnxparser.so.10"]


def test_evaluate_reports_ready_when_all_tensorrt_parts_exist():
    sections = module.parse_sections(
        "\n".join(
            [
                "section=cuda",
                "cuda_home_exists=1",
                "section=trtexec",
                "trtexec_path=/usr/src/tensorrt/bin/trtexec",
                "section=cpp",
                "header=/usr/include/NvInfer.h",
                "lib=libnvinfer.so.10",
                "section=python",
                "module_tensorrt=1",
                "module_torch_tensorrt=0",
                "module_onnx=1",
                "module_onnxsim=1",
                "module_polygraphy=1",
                "module_onnxruntime=1",
                "onnxruntime_providers=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
                "torch_cuda_available=1",
            ]
        )
    )

    readiness = module.evaluate(sections)

    assert readiness["onnx_export_ready"] is True
    assert readiness["python_tensorrt_ready"] is True
    assert readiness["cpp_tensorrt_ready"] is True
    assert readiness["onnxruntime_tensorrt_ready"] is True
    assert readiness["missing"]["trtexec"] is False


def test_evaluate_reports_current_missing_tensorrt_shape():
    sections = module.parse_sections(
        "\n".join(
            [
                "section=cuda",
                "cuda_home_exists=0",
                "section=python",
                "module_tensorrt=0",
                "module_torch_tensorrt=0",
                "module_onnx=0",
                "module_onnxsim=0",
                "module_polygraphy=0",
                "module_onnxruntime=0",
                "torch_cuda_available=1",
            ]
        )
    )

    readiness = module.evaluate(sections)

    assert readiness["torch_cuda_ready"] is True
    assert readiness["onnx_export_ready"] is False
    assert readiness["python_tensorrt_ready"] is False
    assert readiness["cpp_tensorrt_ready"] is False
    assert readiness["missing"]["python_onnx"] is True


def test_remote_script_contains_no_host_specific_paths():
    class Args:
        venv = "/tmp/venv"

    script = module.remote_script(Args())

    assert "section=trtexec" in script
    assert "VENV=/tmp/venv" in script
    assert '"$VENV/bin/python"' in script
    assert '"$candidate" --version 2>&1 || true' in script
