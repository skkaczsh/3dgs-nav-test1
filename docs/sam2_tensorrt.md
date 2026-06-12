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

## Accuracy Comparison Gate

There is not yet a SAM2 model engine or C++ SAM2 runner in this repository.
The current verified C++/TensorRT artifact is only the base toolchain plus a
tiny ONNX smoke model. Therefore, a real SAM2 C++ vs Python mask-quality
comparison cannot be claimed yet.

Once a C++/TensorRT SAM2 runner exists, compare it against the current Python
baseline with:

```bash
python3 compare_sam_mask_dirs.py \
  --baseline-dir /root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_combined \
  --candidate-dir /root/epfs/sam2_tensorrt/sam_masks_candidate \
  --manifest /root/epfs/new_route_stage1_skymask/semantic_manifest_2000_2999.json \
  --limit 50 \
  --json-output /root/epfs/sam2_tensorrt/reports/python_vs_trt_masks.json \
  --csv-output /root/epfs/sam2_tensorrt/reports/python_vs_trt_masks.csv
```

Promotion criteria for replacing the Python SAM2 generator:

- mean matched-mask IoU should be high on the validation sample.
- coverage delta should be small, especially on ground/wall/railing frames.
- unmatched baseline masks should not concentrate on thin objects such as
  railings, pipes, edges, or equipment handles.
- downstream semantic label records and target/object fusion should not regress.
