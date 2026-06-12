# SAM2 TensorRT Environment

This documents the scan-train SAM2 TensorRT environment and the current
subgraph-level C++ runner foundation.

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

## Build SAM2 Engines

Run on scan-train after syncing scripts to `/root/epfs/new_route_scripts`:

```bash
cd /root/epfs/new_route_scripts
GPU_ID=0 BATCH_SIZE=64 ./build_sam2_tensorrt_engines.sh
```

Current verified artifacts:

- encoder ONNX:
  `/root/epfs/sam2_tensorrt/onnx/sam2_hiera_l_image_encoder.onnx`
- decoder ONNX:
  `/root/epfs/sam2_tensorrt/onnx/sam2_hiera_l_point_decoder_b64.onnx`
- encoder engine:
  `/root/epfs/sam2_tensorrt/engines/sam2_hiera_l_image_encoder_fp16.plan`
- decoder engine:
  `/root/epfs/sam2_tensorrt/engines/sam2_hiera_l_point_decoder_b64_fp16.plan`

The decoder ONNX export is intentionally done on CPU. CUDA export currently
hits a PyTorch ONNX tracer device-mismatch path in SAM2's `repeat_image`
decoder branch. The CPU-exported decoder graph builds and runs under TensorRT.

## C++ Runner Smoke

Build and run the C++ smoke runner on scan-train:

```bash
cd /root/epfs/new_route_scripts
./build_sam2_tensorrt_runner.sh
CUDA_VISIBLE_DEVICES=0 /root/epfs/sam2_tensorrt/bin/sam2_trt_runner \
  --engine /root/epfs/sam2_tensorrt/engines/sam2_hiera_l_image_encoder_fp16.plan \
  --engine /root/epfs/sam2_tensorrt/engines/sam2_hiera_l_point_decoder_b64_fp16.plan \
  --run
```

Verified output includes:

- encoder input/output tensors:
  `image`, `high_res_0`, `high_res_1`, `image_embed`
- decoder input/output tensors:
  `image_embed`, `high_res_0`, `high_res_1`, `point_coords`, `point_labels`,
  `low_res_masks`, `iou_predictions`
- both engines report `run ok` from C++ TensorRT runtime.

Current trtexec smoke performance on scan-train GPU0:

- encoder: about `53 qps`, mean GPU compute about `18.8 ms`
- decoder batch 64: about `84 qps`, mean GPU compute about `11.8 ms`

## C++ AMG Runner

The current production-shaped runner is:

- source: `tools/sam2_trt_amg_runner.cpp`
- remote binary:
  `/root/epfs/sam2_tensorrt/bin/sam2_trt_amg_runner`

Build it on scan-train with:

```bash
cd /root/epfs/new_route_scripts
SRC=/root/epfs/new_route_tools/sam2_trt_amg_runner.cpp \
OUT=/root/epfs/sam2_tensorrt/bin/sam2_trt_amg_runner \
./build_sam2_tensorrt_runner.sh
```

Run an isolated candidate sample:

```bash
CUDA_VISIBLE_DEVICES=0 /root/epfs/sam2_tensorrt/bin/sam2_trt_amg_runner \
  --images "/root/epfs/new_route_stage1_skymask/sam2_input_2000_2999/cam0_00200[0-4].png" \
  --output-dir /root/epfs/sam2_tensorrt/sam_masks_candidate_smoke5 \
  --crop-n-layers 1 \
  --overwrite
```

The runner writes Python-compatible artifact names and schema:

- `{image_id}_sam_masks.json`
- `{image_id}_sam_masks.png`
- `{image_id}_numbered.png`
- `{image_id}_sam_done.flag`

For candidate benchmarks, use `--output-mode uncompressed_rle`. The comparison
tool can read both the original bool-list `binary_mask` schema and the
uncompressed RLE schema, while RLE avoids hundreds of MB of JSON per image.
Use `--output-mode binary_mask` only when testing exact compatibility with
legacy consumers that directly expect a 2D bool list in `segmentation`.

Current implementation coverage:

- SAM2 image preprocessing: resize to `1024x1024`, RGB, ImageNet mean/std.
- full-image and `crop_n_layers=1` crop boxes.
- `points_per_side=32`, `points_per_batch=64`.
- TensorRT encoder and point decoder execution.
- bilinear mask upsample to original image size.
- predicted IoU, stability score, area filtering.
- crop-edge filtering, box NMS, crop NMS, overlap resolution.
- JSON/overlay/numbered PNG/flag output compatible with downstream consumers.
- `binary_mask` and `uncompressed_rle` JSON output modes.

Current gaps:

- It has not yet implemented small-region hole/island cleanup equivalent to
  SAM2's CUDA connected-components postprocess.
- Some downstream scripts still assume bool-list `segmentation`; those should
  be updated before promoting RLE artifacts into the main semantic pipeline.
- It is not promoted to the main mask directory until a 20-50 image benchmark
  passes the promotion gates below.

Verified smoke comparison against Python SAM2 on `cam0_002000` to
`cam0_002004`:

- images: `5`
- mean baseline masks: `30.4`
- mean candidate masks: `32.8`
- mean baseline coverage: `0.8062`
- mean candidate coverage: `0.8120`
- mean coverage delta: `+0.0058`
- mean matched IoU: `0.9663`
- mean unmatched baseline masks: `3.6`
- elapsed wall time: `95.9 s` for 5 images with compatible bool-list JSON.

Verified RLE benchmark on `cam0_002000` to `cam0_002009`:

- images: `10`
- mean baseline masks: `22.9`
- mean candidate masks: `25.5`
- mean baseline coverage: `0.7005`
- mean candidate coverage: `0.7290`
- mean coverage delta: `+0.0285`
- mean matched IoU: `0.9534`
- mean unmatched baseline masks: `3.2`
- mean unmatched candidate masks: `5.8`
- candidate output directory size including PNG previews: `34 MB`

The high matched IoU means the TensorRT encoder/decoder path is numerically
close enough for a larger side benchmark. With bool-list output, runtime is
dominated by CPU mask upsample/postprocess and very large JSON writes. RLE
removes the JSON-size bottleneck for candidate benchmarks.

## Notes

- C++ TensorRT is pinned to CUDA 11.8 because `/usr/local/cuda` points to
  `/usr/local/cuda-11.8` on scan-train.
- Python TensorRT 11 is installed for export/prototyping experiments, but the
  C++ ABI baseline is TensorRT 10.9 CUDA 11.8.
- `pycuda` is intentionally not required. It failed to build against the
  current container CUDA header layout and is not needed for the C++ path.

## Accuracy Comparison Gate

There are now SAM2 encoder/point-decoder TensorRT engines, a C++ smoke runner,
and a production-shaped C++ AMG runner that writes compatible
`*_sam_masks.json` artifacts. This is still a side-track candidate until the
larger benchmark proves that thin-object recall and downstream target/object
quality do not regress.

Once the C++/TensorRT AMG runner writes candidate masks, compare it against the
current Python baseline with:

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
