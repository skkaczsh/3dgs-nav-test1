# Semantic Dataset Runbook

This document records the current validated route and server operations for the
0-999 frame semantic dataset run. It is intentionally operational: use current
server state as authority when numbers differ.

## Validated Route

- Projection route: `scripts/project_color.py` and `scripts/project_semantic.py`.
- Geometry chain: `img_pos.txt + cam_in_ex.txt + Tcl + Til`.
- Default 2D semantic combo: `sam2_prompt_v3_sky_label_merge_completion`.
- Deprecated comparison route: `MT20260511-165822/semantic_pointcloud_pipeline`.
  Do not use it for the main dataset run.

## Server Layout

- Prefer direct endpoints while local SSH aliases contain stale bind addresses:
  `scan-train = root@10.0.8.114 -p 31909`, `scan-vlm = root@10.0.8.114 -p
  31079`.
- On the 2026-06-12 local network, bind traffic to Wi-Fi with
  `BIND_ADDRESS=192.168.100.119` / `ssh -o BindAddress=192.168.100.119`.
- `scan-vlm`
  - Qwen VL: `localhost:8001`, `-np 4`.
  - Split A semantic runner.
  - Semantic-to-fusion watcher.
- `scan-train`
  - Qwen VL: `localhost:8002`, `-np 4`.
  - Qwen VL: `localhost:8003`, `-np 4`.
  - Split B and C semantic runners.
  - ConceptSeg smoke watcher.

## Key Paths

- Stage directory: `/root/epfs/new_route_stage1_skymask`
- Processed root: `/root/epfs/manifold_3dgs_project/processed`
- Split outputs:
  - `/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_a`
  - `/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_b`
  - `/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_c`
  - `/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_d`
- Final merged semantic output:
  - `/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999`
- Target/object fusion output:
  - `/root/epfs/new_route_stage1_skymask/target_object_fusion_0000_0999`
- Correct RGB PLY input for fusion:
  - `/root/epfs/new_route_stage1_skymask/output/frame_XXXX.ply`
- Target/object fusion quality gate:
  - `MIN_MERGE_CONFIDENCE=0.5` by default.
  - Low-confidence targets are still preserved, but they do not actively merge
    into existing objects by geometry/color alone.
  - `vlm_mixed=true` targets are blocked from merging unless
    `vlm_can_merge_to_surface=true`.

## Watchers

- Do not restart active semantic runners or Qwen servers while artifacts are
  still increasing. First verify a real stall by checking process trees and
  recent `labels.json` mtime counts.
- Semantic-to-fusion watcher:
  - Script: `scripts/watch_server_semantic_to_fusion.sh`
  - Remote process writes: `/root/epfs/new_route_stage1_skymask/target_object_fusion_0000_0999.pid`
  - Trigger: final combo reaches `3000/3000`.
  - Actions: merge split outputs, then run `scripts/run_server_target_object_fusion.sh`.
- ConceptSeg watcher:
  - Script: `scripts/watch_server_conceptseg_after_c.sh`
  - Trigger: split C final combo reaches `913/913`.
  - Actions: stop Qwen on port `8003`, then run `scripts/run_server_conceptseg_smoke.sh`.
  - Default inference entry: `conceptseg_inference_single_example_sdpa.py`.
  - Hugging Face auth must be configured locally on each server; never commit
    or document the token value. Store it in `~/.cache/huggingface/token` and
    `~/.huggingface/token` with `0600` permissions.
  - HF cache must live on EPFS. `scripts/run_server_conceptseg_smoke.sh`
    defaults to `HF_HOME=/root/epfs/hf_home` and
    `HUGGINGFACE_HUB_CACHE=/root/epfs/hf_home/hub`.
  - Do not use the root filesystem for SAM3 downloads. A failed download under
    `/root/.cache/huggingface/hub/models--facebook--sam3` previously filled the
    50GB root overlay.

## Verified Preflight

- `scripts/config.py` defaults point to:
  - `/root/epfs/new_route_data`
  - `/root/epfs/new_route_data/calib`
  - `/root/epfs/new_route_data/ply`
  - `/root/epfs/new_route_stage1_skymask`
- Target/object smoke on split D frames `363,365,367` completed:
  - Frames: `3/3`
  - Targets: `11`
  - Objects: `6`
  - Ambiguous ratio: `0.0`
- Scene-aware VLM quality fields are now preserved from `labels.json` into
  Target records and used by object fusion. Re-run target/object fusion after
  regenerating scene-aware semantic artifacts.
- ConceptSeg-R1 smoke dependency fixes completed:
  - Installed `scikit-image` and `scikit-learn` in the remote
    `conceptseg-r1` environment.
  - Added an SDPA smoke entry to avoid hard-required `flash_attn`.
  - Added repo-local SAM3 path bootstrap to avoid namespace-package import
    issues.
  - After configuring authorized Hugging Face credentials and moving HF cache
    to EPFS, `LIMIT=1` smoke succeeded on scan-vlm:
    `/root/epfs/new_route_stage1_skymask/conceptseg_smoke_hfhome_20260609_101800/cam0_000000_floor.png`.
- Local tests:
  - New route: `python3 -m pytest -q tests/test_new_route_scripts.py new_route/tests/test_target_object_fusion.py`
  - Old route baseline: `PYTHONPATH=/Users/skkac/Work/SCAN/MT20260511-165822 python3 -m pytest -q MT20260511-165822/tests/test_projection.py MT20260511-165822/tests/test_backproject.py MT20260511-165822/tests/test_merge.py`

## Delivery Acceptance

Run this local gate before handing off the 0-999 package or extending the run
beyond the current frame range:

```bash
cd /Users/skkac/Work/SCAN/new_route

python3 scripts/build_dataset_delivery_manifest.py
python3 scripts/validate_dataset_delivery_manifest.py \
  --manifest /Users/skkac/Work/SCAN/route_status_20260610/dataset_delivery_manifest_0000_0999.json \
  --output /Users/skkac/Work/SCAN/route_status_20260610/dataset_delivery_manifest_0000_0999_validation.json
python3 scripts/package_dataset_delivery.py --clean
python3 scripts/validate_dataset_package.py \
  --output /Users/skkac/Work/SCAN/dataset_delivery_0000_0999_validation.json
python3 scripts/run_delivery_acceptance.py
```

Expected:

- `passed: true`
- package: `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999.tgz`
- QA index:
  `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999/qa_index.html`
- acceptance report:
  `/Users/skkac/Work/SCAN/route_status_20260610/delivery_acceptance_20260611.json`
- next manual gate: visual acceptance in the PLY viewer or CloudCompare.

## Monitor Commands

Refresh semantic split progress:

```bash
ssh -F /dev/null -o BindAddress=192.168.100.119 -p 31079 root@10.0.8.114 'python3 /root/epfs/new_route_scripts/qa_semantic_splits.py \
  --split a /root/epfs/new_route_stage1_skymask/semantic_manifest_ready_a_current.json /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_a \
  --split b /root/epfs/new_route_stage1_skymask/semantic_manifest_ready_b_current.json /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_b \
  --split c /root/epfs/new_route_stage1_skymask/semantic_manifest_ready_c_current.json /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_c \
  --split d /root/epfs/new_route_stage1_skymask/semantic_manifest_final_d_missing.json /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_d \
  --output /root/epfs/new_route_stage1_skymask/semantic_splits_progress_launched.json'
```

Check scan-train runners:

```bash
ssh -F /dev/null -o BindAddress=192.168.100.119 -p 31909 root@10.0.8.114 'tmux ls || true; nvidia-smi'
```

Check scan-vlm watcher:

```bash
ssh -F /dev/null -o BindAddress=192.168.100.119 -p 31079 root@10.0.8.114 'pgrep -af "run_server_semantic_completion_sharded|llama-server" || true; nvidia-smi'
```
