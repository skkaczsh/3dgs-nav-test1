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

- SSH option for both servers: `ssh -o BindAddress=192.168.100.113`.
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
- Local tests:
  - New route: `python3 -m pytest -q tests/test_new_route_scripts.py new_route/tests/test_target_object_fusion.py`
  - Old route baseline: `PYTHONPATH=/Users/skkac/Work/SCAN/MT20260511-165822 python3 -m pytest -q MT20260511-165822/tests/test_projection.py MT20260511-165822/tests/test_backproject.py MT20260511-165822/tests/test_merge.py`

## Monitor Commands

Refresh semantic split progress:

```bash
ssh -o BindAddress=192.168.100.113 scan-vlm 'python3 /root/epfs/new_route_scripts/qa_semantic_splits.py \
  --split a /root/epfs/new_route_stage1_skymask/semantic_manifest_ready_a_current.json /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_a \
  --split b /root/epfs/new_route_stage1_skymask/semantic_manifest_ready_b_current.json /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_b \
  --split c /root/epfs/new_route_stage1_skymask/semantic_manifest_ready_c_current.json /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_c \
  --split d /root/epfs/new_route_stage1_skymask/semantic_manifest_final_d_missing.json /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_d \
  --output /root/epfs/new_route_stage1_skymask/semantic_splits_progress_launched.json'
```

Check scan-train runners:

```bash
ssh -o BindAddress=192.168.100.113 scan-train 'ps -p 38162,38163,37840,38004,36685 -o pid,etime,pcpu,stat,cmd || true; pstree -ap 38162; pstree -ap 38163; nvidia-smi'
```

Check scan-vlm watcher:

```bash
ssh -o BindAddress=192.168.100.113 scan-vlm 'ps -p 14227,14007,13816 -o pid,etime,pcpu,stat,cmd || true; tail -n 20 /root/epfs/new_route_stage1_skymask/logs/watch_semantic_to_fusion.log'
```
