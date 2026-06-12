# SAM2 TensorRT Environment

This documents the scan-train SAM2 TensorRT base environment used before
building a dedicated SAM2 C++ inference path.

## Server

- host: `scan-train`
- SSH: `root@10.0.8.114 -p 31909`
- required bind address from this workstation: `192.168.100.119`
- work dir: `/root/epfs/sam2_tensorrt`

## Installed Components

- Python env: `/root/epfs/conda_envs/vlm_seg`
- Python packages:
  - `tensorrt-cu12==11.0.0.114`
  - `onnx`
  - `onnxsim`
  - `polygraphy`
  - `cuda-python`
- C++ TensorRT packages:
  - `libnvinfer-dev==10.9.0.34-1+cuda11.8`
  - `libnvinfer-bin==10.9.0.34-1+cuda11.8`
  - matching `libnvinfer*`, `libnvonnxparsers*` runtime/dev packages
- C++ include path:
  - `/usr/include/x86_64-linux-gnu`
  - `/usr/local/cuda/include`
- C++ libraries:
  - `/usr/lib/x86_64-linux-gnu`
  - `/usr/local/cuda/lib64`
- `trtexec`:
  - `/usr/src/tensorrt/bin/trtexec`

## Verify

Run on scan-train:

```bash
cd /root/epfs/new_route_scripts
./verify_sam2_tensorrt_env.sh
```

Expected evidence:

- Python imports `torch`, `tensorrt`, `onnx`, `onnxsim`, `polygraphy`.
- C++ smoke binary compiles and prints `builder_ok=1`.
- Tiny ONNX model builds a TensorRT FP16 engine at
  `/root/epfs/sam2_tensorrt/engines/tiny_conv_fp16.plan`.
- `trtexec_tiny_conv.log` contains `PASSED TensorRT.trtexec`.

## Notes

- C++ TensorRT is pinned to CUDA 11.8 because `/usr/local/cuda` points to
  `/usr/local/cuda-11.8` on scan-train.
- Python TensorRT 11 is installed for export/prototyping experiments, but the
  C++ ABI baseline is TensorRT 10.9 CUDA 11.8.
- `pycuda` is intentionally not required. It failed to build against the
  current container CUDA header layout and is not needed for the C++ path.

