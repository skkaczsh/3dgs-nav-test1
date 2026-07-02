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

## Interpretation

- Bucket split alone is useful diagnostically, but it increases high-entropy
  fragments and should not be promoted directly.
- Attachment absorption is necessary after bucket splitting.  The v2 setting is
  currently the best of these three tests because it reduces both patch count
  and high-entropy count relative to v1.
- v3 is more conservative, but the stricter attachment gates leave too many
  fragments unabsorbed.
- The result is still not a promoted baseline because high-entropy patches remain
  above the input count.  The next improvement should make attachment aware of
  split provenance and local contact features, rather than globally relaxing
  merge rules.

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
