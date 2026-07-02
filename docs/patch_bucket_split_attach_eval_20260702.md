# Patch Bucket Split + Attachment Evaluation 2026-07-02

## Purpose

Evaluate whether bucket-connectivity splitting can reduce mixed GeoPatch objects
without creating excessive residual fragments.

## Input

- Dense source chain: `geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623`
- Region input: `_cpp_region_grower_input.bin`
- Initial labels: `energy_attach_v4_contact_evidence/geo_patches_energy_attach_v4_contact_evidence_labels.bin`
- Voxel density: 0.03 m dense Opt-LAS derived region model

## Runs

| Run | Output patches | High entropy patches | Merge accepts | Accepted attachment |
| --- | ---: | ---: | ---: | ---: |
| bucket split v1 | 195,743 | 9,581 | 953 | 0 |
| bucket split + attachment v2 | 188,536 | 8,410 | 8,118 | 7,120 |
| stricter mid-anchor v3 | 190,778 | 8,763 | 5,883 | 4,899 |
| split-provenance attachment v4 | 192,518 | 9,018 | 4,144 | 2,967 + 209 split-provenance |
| fragment-evidence attachment v5 | 189,898 | 8,615 | 6,759 | 5,565 fragment + 211 split-provenance |

## Interpretation

- Bucket split alone is useful diagnostically, but it increases high-entropy
  fragments and should not be promoted directly.
- Attachment absorption is necessary after bucket splitting.  The v2 setting is
  currently the best of these three tests because it reduces both patch count
  and high-entropy count relative to v1.
- v3 is more conservative, but the stricter attachment gates leave too many
  fragments unabsorbed.
- v4 proves that relaxing only newly created `bucket_connectivity_split`
  children is too narrow.  It accepted only 209 split-provenance attachments,
  because many bad fragments are created by ordinary split branches or later
  boundary movement, not only by the new child label.
- v5 adds contact/fragment evidence, accepts 5,565 fragment-evidence
  attachments, and recovers most of v2's benefit while keeping global
  attachment gates strict.  It is architecturally cleaner than v2 but still
  slightly worse on aggregate high-entropy count.
- The v2/v5 comparison QA shows that large-patch risk is nearly unchanged:
  v2 has 20 large high-entropy patches and v5 has 22.  The remaining difference
  is mostly small-fragment behavior, not major over-merge behavior.
- The result is still not a promoted baseline because high-entropy patches remain
  above the input count.  The next improvement should make attachment aware of
  local contact and fragmentation evidence with better bucket compatibility
  handling, rather than relying only on literal child-label provenance or
  globally relaxing merge rules.
- Patch/object exports from this branch are geometry ownership inputs only.
  `semantic_label` must stay `unknown`; structural buckets are stored as
  `geometry_type` / `geometry_label` and may only become semantic labels after a
  later evidence-fusion stage.

## Comparison QA

```bash
BASE=/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623
OUT=${BASE}/geo_patch_run_comparison_v2_v5_20260702
PYTHONPATH=. python scripts/compare_geo_patch_runs.py \
  --run v2=${BASE}/energy_bucket_split_attach_v2_20260702/geo_patches_bucket_split_attach_v2.jsonl,${BASE}/energy_bucket_split_attach_v2_20260702/geo_patches_bucket_split_attach_v2_report.json,${BASE}/energy_bucket_split_attach_v2_20260702/merge_log.jsonl \
  --run v5=${BASE}/energy_bucket_split_frag_attach_v5_20260702/geo_patches_bucket_split_frag_attach_v5.jsonl,${BASE}/energy_bucket_split_frag_attach_v5_20260702/geo_patches_bucket_split_frag_attach_v5_report.json,${BASE}/energy_bucket_split_frag_attach_v5_20260702/merge_log.jsonl \
  --output-json "${OUT}/comparison.json" \
  --output-md "${OUT}/comparison.md"
```

Summary:

| run | patches | high entropy | large high entropy | large low purity | merge accepts |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 | 188,536 | 8,410 | 20 | 20 | 8,118 |
| v5 | 189,898 | 8,615 | 22 | 20 | 6,759 |

## Reproduce

```bash
cd /Users/skkac/Work/SCAN/new_route
RUN=1 OUT_NAME=energy_bucket_split_attach_v2_20260702 \
  scripts/run_scan_train_patch_bucket_split_attach_eval.sh
```

## Review URL

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/energy_bucket_split_attach_v2_20260702/geo_patches_bucket_split_attach_v2_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/energy_bucket_split_attach_v2_20260702/geo_patches_bucket_split_attach_v2.jsonl&mode=object&stride=1&pointSize=1.2
```

v5 mechanism-clean review:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/energy_bucket_split_frag_attach_v5_20260702/geo_patches_bucket_split_frag_attach_v5_stride10.ply&objects=/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/energy_bucket_split_frag_attach_v5_20260702/geo_patches_bucket_split_frag_attach_v5.jsonl&mode=object&stride=1&pointSize=1.2
```
