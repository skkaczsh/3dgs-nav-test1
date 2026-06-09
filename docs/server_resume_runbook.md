# Server Resume Runbook

This runbook records the next concrete steps once `scan-train` / `scan-vlm` connectivity returns.

## Current Blocker

- Current operator context: outside the server LAN. Do not repeatedly run SSH
  or TCP probes until the machine is back on the reachable LAN/VPN.
- Local active IPv4 observed: `192.168.0.3`
- SSH config currently resolves both servers with `BindAddress 192.168.100.115`
- Last checks timed out:
  - `10.0.8.114:31909` (`scan-train`)
  - `10.0.8.114:31079` (`scan-vlm`)

Do not rotate keys or change scripts until the TCP ports are reachable again.

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

## Verify Delivery Package

```bash
cd /Users/skkac/Work/SCAN/new_route

python3 scripts/verify_review_delivery_manifest.py \
  --zip-path /Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery.zip
```

Expected:

- `passed: true`
- `expected_file_count: 21`
- `errors: []`

## New Model Side Track

Current ConceptSeg-R1 status:

- Source/weights were prepared on `scan-train`
- Smoke result was not strong enough to replace main SAM2+Qwen route
- Continue only when it does not occupy main-route resources

Next side-track action after server returns:

```bash
# Inspect GPU and existing model files first.
ssh scan-train 'nvidia-smi; ls -lah /root/epfs/model_side_tracks/ConceptSeg-R1'
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
