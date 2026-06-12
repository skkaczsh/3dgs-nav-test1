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
  --crop-nms-thresh 0.65 \
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

Verified RLE benchmark on `cam0_002000` to `cam0_002019`:

- images: `20`
- mean baseline masks: `27.6`
- mean candidate masks: `31.9`
- mean baseline coverage: `0.6659`
- mean candidate coverage: `0.7158`
- mean coverage delta: `+0.0499`
- mean matched IoU: `0.9469`
- mean unmatched baseline masks: `3.2`
- mean unmatched candidate masks: `7.5`
- candidate output directory size including PNG previews: `65 MB`

The 20-image benchmark confirms model-level alignment, but it also shows that
the TensorRT AMG candidate is more permissive than the Python baseline. The
worst observed frame was `cam0_002005`, where candidate coverage was `+0.2083`
above baseline. Next tuning should focus on crop/candidate NMS and overlap
resolution rather than encoder/decoder export.

NMS sweep result on `cam0_002000` to `cam0_002009`:

- default `box=0.7,crop=0.7,pred=0.7`: coverage delta `+0.0285`,
  matched IoU `0.9534`, unmatched candidate `5.8`.
- `box=0.7,crop=0.65,pred=0.7`: coverage delta `+0.0283`,
  matched IoU `0.9539`, unmatched candidate `5.2`.
- raising `pred_iou_thresh` to `0.75` or `0.8` further reduces coverage delta,
  but increases unmatched baseline masks, so it is not the default candidate.

Benchmark launcher default is therefore `box=0.7,crop=0.65,pred=0.7`. The C++
runner itself still defaults to SAM2-like `crop_nms_thresh=0.7`; pass explicit
parameters for benchmarks and promotion tests.

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

## Production Candidate Runner

Use `run_server_sam2_trt_production.sh` for production-shaped candidate output.
It writes `uncompressed_rle` JSON by default and patches
`semantic_eval/run_eval.py` with `patch_semantic_eval_rle_masks.py` so the
existing VLM, merge, completion, and artifact-writing path can decode compact
RLE masks. This avoids duplicating the current hundreds-of-GB bool-list SAM2
mask cache.

Example 20-image validation run:

```bash
START=2000 END=2999 \
IMAGE_GLOB='/root/epfs/new_route_stage1_skymask/sam2_input_2000_2999/cam0_0020[0-1][0-9].png' \
OUTPUT_DIR=/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_20 \
REPORT_DIR=/root/epfs/sam2_tensorrt/reports/production_2000_2999_20 \
BASELINE_DIR=/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_combined \
bash run_server_sam2_trt_production.sh
```

The script runs:

- Optional semantic-eval RLE loader patch when `OUTPUT_MODE=uncompressed_rle`.
- C++ TensorRT AMG candidate generation.
- Candidate manifest generation.
- Optional baseline comparison when `BASELINE_DIR` is set.
- `gate_sam2_trt_promotion.py`, which fails nonzero if the candidate exceeds
  configured drift thresholds.

Default promotion gate thresholds:

- `mean_matched_iou >= 0.93`
- `abs(mean_coverage_delta) <= 0.06`
- `abs(row coverage_delta) <= 0.25`
- `mean_unmatched_baseline_masks <= 4.0`
- `mean_unmatched_candidate_masks <= 8.0`

If the gate passes, the candidate directory is eligible for downstream
semantic-eval smoke testing. It still should not replace
`sam_masks_${START}_${END}_combined` until target/object QA on the same frame
range is at least neutral against the Python baseline.

Set `OUTPUT_MODE=binary_mask` only for strict legacy compatibility checks. That
mode is disk-expensive: observed Python baseline bool-list masks are already
hundreds of MB per image, and the full `2000-2999` combined cache is hundreds of
GB.

Verified production-shaped RLE smoke on `cam0_002000` to `cam0_002009`:

- candidate dir:
  `/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_rle10`
- output size: `34 MB` for 10 images.
- gate status: `pass`
- mean matched IoU: `0.9539`
- mean coverage delta: `+0.0283`
- mean unmatched baseline masks: `3.1`
- mean unmatched candidate masks: `5.2`
- downstream RLE smoke:
  `/root/epfs/sam2_tensorrt/semantic_eval_rle_smoke10`, two images reached
  `sam2_qwen` artifact generation through `semantic_eval/run_eval.py`.

Verified wider production-shaped RLE test on `cam0_002000` to `cam0_002049`:

- candidate dir:
  `/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_rle50`
- output size: `168 MB` for 50 images.
- gate status: `fail`
- mean matched IoU: `0.9474`
- mean coverage delta: `+0.0841`
- mean unmatched baseline masks: `5.54`
- mean unmatched candidate masks: `9.08`
- worst coverage delta: `+0.3646` on `cam0_002039`.

Interpretation: TensorRT encoder/decoder alignment is acceptable, but the C++
AMG postprocessing is still too permissive on a wider sample. Keep Python SAM2
as the production mask source until stricter C++ AMG parameters or postprocess
parity reduce over-coverage. A stricter 50-image run with
`PRED_IOU_THRESH=0.75` is the next candidate to evaluate.
