# Offline Work Queue

This file tracks useful work while the machine is outside the server LAN. Do
not treat server timeouts as regressions during this state.

## Priority 1: Main Route Readiness

- Keep `scripts/resume_server_qwen_review.sh` runnable after script imports
  change.
- Keep `scripts/run_server_semantic_completion_sharded.sh` reproducible with
  scene-aware prompt patching enabled by default.
- Keep `scripts/run_server_target_object_fusion.sh` exposing quality-gate
  parameters, especially `MIN_MERGE_CONFIDENCE`.
- Preserve VLM quality fields from 2D labels through Target records and Object
  fusion QA.

## Priority 2: Local QA

- Add tests for shell runner contracts whenever a script depends on another
  copied file.
- Keep sensitive token scans passing. The default scan checks repository text
  files up to 2MB and does not print matched secret values.
- Keep review delivery verification passing:

```bash
python3 scripts/verify_review_delivery_manifest.py \
  --zip-path /Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery.zip
```

- Keep core local tests passing:

```bash
bash scripts/run_offline_quality_checks.sh
```

On success, the runner writes:

- `/Users/skkac/Work/SCAN/route_status_20260610/offline_quality_latest.json`

Prepare a local resume-readiness report before probing the servers again:

```bash
python3 scripts/prepare_server_resume_report.py
```

## Priority 3: Deferred Server Resume

When back on the server LAN:

1. Run `scripts/diagnose_server_connectivity.py`.
2. If reachable, run `scripts/resume_server_qwen_review.sh`.
3. If semantic artifacts need regeneration, run
   `scripts/run_server_semantic_completion_sharded.sh` with
   `PATCH_SCENE_PROMPTS=1` and `SHARDS=4`.
4. Re-run `scripts/run_server_target_object_fusion.sh` after scene-aware
   semantic artifacts are ready.
5. Keep ConceptSeg-R1 and old route side tracks idle unless main-route GPU
   demand is low.
