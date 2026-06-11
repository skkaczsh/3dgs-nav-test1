# Server Resume Runbook

This runbook records the next concrete steps for server resume and local
delivery verification.

## Current Connectivity

- Latest verified route: `ssh -p 31909 root@10.0.8.114` for `scan-train`.
- Latest verified route: `ssh -p 31079 root@10.0.8.114` for `scan-vlm`.
- If an explicit `BindAddress` fails after a network change, omit it first and
  verify the direct SSH route before editing keys or scripts.
- For repeated SSH work, use `tmux` on the remote host and avoid restarting
  active runners until process trees and artifact mtimes confirm a real stall.

Do not rotate keys or change scripts just because a previous `BindAddress`
stopped working.

The local remote wrappers now support direct endpoints, so prefer this form
while SSH aliases have stale `BindAddress` values:

```bash
SSH_HOST=10.0.8.114 SSH_PORT=31909 SSH_USER=root SERVER=scan-train \
  bash scripts/run_remote_server_target_object_fusion.sh
```

Regenerate the current server/task queue before starting new work:

```bash
cd /Users/skkac/Work/SCAN/new_route

python3 scripts/check_infra_readiness.py
python3 scripts/prepare_visual_acceptance_review.py
python3 scripts/validate_visual_acceptance_review.py
python3 scripts/check_next_increment_readiness.py
python3 scripts/prepare_parallel_execution_queue.py
```

This writes:

- `/Users/skkac/Work/SCAN/route_status_20260610/infra_readiness_20260611.json`
- `/Users/skkac/Work/SCAN/route_status_20260610/visual_acceptance_review_20260611.json`
- `/Users/skkac/Work/SCAN/route_status_20260610/next_increment_readiness_1000_1999.json`
- `/Users/skkac/Work/SCAN/route_status_20260610/parallel_execution_queue_20260611.json`

The next main-route increment remains blocked until all required checks in
`visual_acceptance_review_20260611.json` are set to `accepted`, and
`validate_visual_acceptance_review.py --require-accepted` passes.
Current 1000-1999 preflight has scanner section PLY and calibration sources, but
still needs camera frame extraction before color/SAM/Qwen/target-object phases.

## Offline Mode

While outside the server LAN, prioritize local work that improves the next
server run without requiring remote access:

- Keep recovery scripts import-safe and self-contained.
- Maintain review packages, manual CSV workflows, and delivery manifests.
- Improve prompt/schema handling and Target/Object fusion logic.
- Add local tests for server-runner shell scripts and JSON contracts.
- Update runbooks with exact resume commands.

Defer these until LAN connectivity returns:

- Qwen review execution.
- 0-999 semantic regeneration.
- ConceptSeg-R1 GPU smoke or model downloads.
- Old-route server-side reruns.

## Main Route: Qwen Review Resume

Before executing remote work, generate the local command plan from the latest
readiness report:

```bash
cd /Users/skkac/Work/SCAN/new_route

python3 scripts/prepare_server_resume_commands.py
python3 scripts/validate_server_resume_commands.py
```

This writes:

- `/Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands.json`
- `/Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands.sh`
- `/Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands_validation.json`

The generated shell plan runs the required main-route phases in order, runs
strict output validation, and only then prints new-model / old-route side-track
commands as optional follow-ups.
The semantic refresh and target/object fusion phases use remote wrappers that
sync local `scripts/` to `/root/epfs/new_route_scripts` and execute the
`/root/epfs/...` workload on `scan-train`; do not run the lower-level
`run_server_semantic_completion_sharded.sh` or
`run_server_target_object_fusion.sh` directly on the laptop. The plan also runs
`scripts/run_server_dataset_readiness.sh` after the scene-aware semantic refresh
so strict output validation has the required dataset readiness input.

The strict validation step is already included in the generated shell plan. It
can also be rerun manually after inspecting or replacing artifacts:

```bash
python3 scripts/validate_server_resume_outputs.py --strict
```

This writes:

- `/Users/skkac/Work/SCAN/route_status_20260610/server_resume_output_validation.json`

`--strict` must pass before treating the 0-999 dataset as ready for the new
model side track or renewed old-route comparison.

Run after server connectivity returns:

```bash
cd /Users/skkac/Work/SCAN/new_route

# If the old bind address is active again, omit BIND_ADDRESS.
# If using the current wlan address, set BIND_ADDRESS=192.168.0.3.
BIND_ADDRESS=192.168.0.3 \
SERVER=scan-train \
CONCURRENCY=4 \
bash scripts/resume_server_qwen_review.sh
```

Expected local outputs:

- `/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2/vlm_review_qwen_compact/vlm_merge_review_results.jsonl`
- `/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2/vlm_review_qwen_compact_applied/review_merged_long_objects.jsonl`
- `/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2/vlm_review_qwen_compact_applied/qa_reviewed_merge_report.json`

Acceptance gate:

- Qwen review `error_count == 0`
- reviewed merge QA `passed == true`
- No automatic merge should be trusted without `decision=merge` and confidence above threshold.

## Main Route: Scene-Aware 2D Prompt Resume

The sharded semantic completion runner now patches the server-side
`semantic_eval/review_merged_labels_prompt_v2.py` and
`semantic_eval/complete_unknown_regions.py` prompts before running. This keeps
the existing `{"items":[...]}` parser contract but adds rooftop scene
constraints and thin-object/floor disambiguation.

Dry-run the patch first if the server source has changed:

```bash
cd /Users/skkac/Work/SCAN/new_route

python3 scripts/patch_semantic_eval_scene_prompts.py \
  --semantic-root /root/epfs/manifold_3dgs_project/semantic_eval \
  --dry-run
```

The full sharded runner applies the patch by default:

```bash
PATCH_SCENE_PROMPTS=1 \
SHARDS=4 \
bash scripts/run_server_semantic_completion_sharded.sh
```

Set `PATCH_SCENE_PROMPTS=0` only when intentionally reproducing the older
prompt baseline.

For incremental `scan-vlm` catch-up runs, use:

```bash
bash scripts/run_remote_scan_vlm_semantic_extra.sh
```

This runner filters candidate SAM2 masks with `--min-sam-age-seconds` before
Qwen work starts, then validates JSON only for the small set of VLM-extra
candidates. Do not switch this back to an existence-only check: SAM mask JSON
can be visible while it is still being written, which causes `run_eval.py` to
fail with `JSONDecodeError`.

After a target/object run, summarize coarse-label plus identity coverage:

```bash
python3 scripts/summarize_identity_enrichment.py \
  --semantic-eval-dir /root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_1000_1999 \
  --objects-jsonl /root/epfs/new_route_stage1_skymask/<target_object_run>/objects.jsonl \
  --output-json /root/epfs/new_route_stage1_skymask/<target_object_run>/identity_enrichment_report.json \
  --description-csv /root/epfs/new_route_stage1_skymask/<target_object_run>/identity_descriptions.csv
```

The expected semantic model is two-level: keep `label` constrained for
statistics and rendering, and use `description`, `identity_hint`, and
`attributes` for object identity and merge QA.

For unattended catch-up while `scan-train` continues producing SAM masks, start
the scan-vlm loop instead of manually re-running the one-shot command:

```bash
bash scripts/start_remote_scan_vlm_extra_loop.sh
```

It syncs `scripts/` to scan-vlm and starts a tmux session named
`vlm_extra_loop_1000_1999`; if tmux is unavailable on the server, it falls back
to `nohup` with a pid file under the logs directory. Each cycle selects only
stable SAM mask JSON files, validates the small VLM-extra candidate set, caps
the batch size with `MAX_ITEMS_PER_CYCLE`, and then runs the same sharded
semantic completion route. Use `MAX_CYCLES=1` for a dry operational check.

When SAM2 mask generation is the bottleneck, prefer using `scan-train` GPU1 for
a second SAM2 shard and let `scan-vlm` handle Qwen. The train-side Qwen server
is optional while the scan-vlm loop is healthy. A safe operational pattern is:

```bash
# On scan-train. Stop only the train-side VLM loop/server; keep the main GPU0
# SAM2 session running.
tmux kill-session -t semantic_ready_loop_1000_1999 2>/dev/null || true
pkill -f 'llama-server.*--port 8001' 2>/dev/null || true

# Build a non-overlapping tail input directory from currently missing masks.
python3 - <<'PY'
from pathlib import Path
inp = Path("/root/epfs/new_route_stage1_skymask/sam2_input_1000_1999")
out = Path("/root/epfs/new_route_stage1_skymask/sam_masks_1000_1999_combined")
stage = Path("/root/epfs/new_route_stage1_skymask/sam2_input_1000_1999_gpu1_tail")
missing = sorted(p for p in inp.iterdir() if not (out / f"{p.stem}_sam_masks.json").exists())
selected = missing[-650:]
stage.mkdir(parents=True, exist_ok=True)
for old in stage.glob("*.png"):
    old.unlink()
for p in selected:
    (stage / p.name).symlink_to(p.resolve() if p.is_symlink() else p)
print({"selected": len(selected), "first": selected[0].name if selected else None, "last": selected[-1].name if selected else None})
PY

tmux new-session -d -s sam2_1000_1999_gpu1_tail \
  "cd /root/epfs/vlm_seg_project/two_phase_pipeline && \
   CUDA_VISIBLE_DEVICES=1 ./run_with_env.sh pure_sam_mask_generator.py \
   --images '/root/epfs/new_route_stage1_skymask/sam2_input_1000_1999_gpu1_tail/*.png' \
   --output-dir '/root/epfs/new_route_stage1_skymask/sam_masks_1000_1999_combined' \
   --workers 2 > /root/epfs/new_route_stage1_skymask/logs/sam2_1000_1999_gpu1_tail.log 2>&1"
```

Use a tail shard rather than the earliest missing files because the existing
GPU0 SAM2 process advances in sorted order. This reduces the chance of two
workers writing the same mask at the same time. The generator also checks
`*_sam_done.flag` inside each worker, so accidental overlap is recoverable.

## Manual Review Fallback

If Qwen remains unavailable, use the packaged human review bundle:

- `/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery/review_html/index.html`
- `/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery/review_html/manual_merge_decisions.csv`

After filling the CSV:

```bash
cd /Users/skkac/Work/SCAN/new_route

python3 scripts/run_manual_merge_review_workflow.py \
  --manual-csv /Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery/review_html/manual_merge_decisions.csv \
  --review-jsonl /Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery/cross_candidate_review_items.jsonl \
  --objects /Users/skkac/Work/SCAN/server_frame_fine_long_assoc_v008/frame_fine_tracklet_long_assoc_0000_0999_v008_gap60_v2_samecand_loose/long_objects.jsonl \
  --output-dir /Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/manual_workflow_reviewed \
  --min-confidence 0.75
```

The workflow now runs QA automatically and exits non-zero if invariants fail.

## Verify Review Delivery Package

```bash
cd /Users/skkac/Work/SCAN/new_route

python3 scripts/verify_review_delivery_manifest.py \
  --zip-path /Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery.zip
```

Expected:

- `passed: true`
- `expected_file_count: 21`
- `errors: []`

## Verify Dataset Delivery Package

Use this local acceptance gate for the current 0-999 dense semantic dataset:

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
- manifest validation:
  `/Users/skkac/Work/SCAN/route_status_20260610/dataset_delivery_manifest_0000_0999_validation.json`
- package validation:
  `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999_validation.json`
- acceptance report:
  `/Users/skkac/Work/SCAN/route_status_20260610/delivery_acceptance_20260611.json`
- package:
  `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999.tgz`

## New Model Side Track

Current ConceptSeg-R1 status:

- Source/weights were prepared on `scan-train`.
- The 90-item constrained run completed, but should remain review-only.
- 3D refinement components are useful as conservative split/refine proposals,
  not as dense semantic source.
- Continue only when it does not occupy main-route resources.

Next side-track action after server returns:

```bash
# Inspect GPU and existing model files first.
ssh -F /dev/null -p 31909 root@10.0.8.114 \
  'nvidia-smi; ls -lah /root/epfs/model_side_tracks/ConceptSeg-R1'
```

Do not promote ConceptSeg-R1 to main path unless it beats `sam2_prompt_v3_sky_label_merge_completion` on the same review artifacts.

## Old Route Side Track

Current old-route status:

- Old route remains visual/colorization reference only
- Do not use it as semantic source unless it passes the same reviewed object QA gates

Next action after server returns:

```bash
# Pull or regenerate old route smoke only if main route resources are idle.
ls /Users/skkac/Work/SCAN/server_old_route_smoke
```

## Canonical Local Review State

- Stage summary:
  `/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2/frame_fine_cross_candidate_review_pack_0000_0999_v008_strict_high_v2/stage_summary/cross_candidate_review_stage_summary.md`
- Delivery zip:
  `/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery.zip`
- Canonical pending workflow:
  `/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2/frame_fine_cross_candidate_review_pack_0000_0999_v008_strict_high_v2/manual_workflow_pending`
