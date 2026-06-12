# SAM2 TensorRT Side-Track Notes

Purpose:

- Evaluate whether TensorRT can reduce SAM2 mask generation time without
  changing the current semantic route.
- Keep the main production route on the validated PyTorch SAM2 generator until
  a side-by-side benchmark proves equivalent mask quality.
- Do not spend main-route GPU time on TensorRT conversion while 1000-1999
  production is still incomplete.

Current decision:

- TensorRT acceleration is technically feasible for SAM2 encoder/decoder
  inference, but it is not a drop-in replacement for the current dense
  automatic mask generation pipeline.
- Encoder and point-decoder ONNX export now work, and both subgraphs build FP16
  TensorRT engines that execute from a C++ runtime smoke runner.
- A first C++ AMG runner now writes Python-compatible mask artifacts for
  full-image plus `crop_n_layers=1` testing. It is not yet promoted to main
  production.
- Treat it as an optimization side track, not a model-quality fix.
- The expected bottleneck must be measured end to end. Python mask generation,
  crop/point sampling, JSON I/O, Qwen review, and target/object fusion can
  dominate even if the SAM2 model kernels become faster.

Candidate implementations:

- Current path: ONNX to TensorRT engines plus a small C++ runtime runner under
  EPFS.
- Remaining work: optimize CPU post-processing/JSON output, add small-region
  cleanup parity, and run 20-50 image side-by-side quality benchmark.
- Fallback path: Torch-TensorRT with `torch.export` and Dynamo compile if ONNX
  runner completion becomes uneconomical.

Benchmark scope:

- Use 20-50 images from the same 1000-1999 production range.
- Include easy broad-surface frames and hard thin-object frames.
- Use the same input images, same sky mask policy, and same dense mask
  post-processing thresholds as the current `pure_sam_mask_generator.py` route.

Required metrics:

- Wall time per image.
- GPU memory and utilization.
- Mask count per image.
- Mask coverage ratio.
- Area distribution and small-mask residual ratio.
- Approximate IoU or overlap between PyTorch SAM2 masks and TensorRT SAM2
  masks after matching.
- Downstream `sam2_prompt_v3_sky_label_merge_completion` parse success and
  target/object QA on the same sample.

Promotion gates:

- Runtime improvement is material on end-to-end mask generation, not just model
  forward time.
- Mask coverage and thin-object recall do not regress.
- Downstream target/object label distribution does not degrade.
- Output filenames and JSON schema remain compatible with the existing
  `sam_masks_1000_1999_combined` consumers.
- A reproducible runner and rollback path exist before any production use.

No-go conditions:

- Do not replace the current SAM2 generator during an active production range.
- Do not write TensorRT outputs into the main mask directory until equivalence
  is proven.
- Do not use TensorRT acceleration to compensate for poor semantic labels; that
  is a VLM/prompt/object-fusion problem, not a SAM2 kernel-speed problem.
- Do not download model caches to server root filesystems.

Operational note:

- If tested on `scan-train`, use an isolated EPFS directory, a separate tmux
  session, and GPU1 only when it is not needed by the current SAM2 tail shard.
- Keep Qwen concurrency at 4 on `scan-vlm`; SAM2 TensorRT experiments should
  not reduce VLM catch-up throughput.
