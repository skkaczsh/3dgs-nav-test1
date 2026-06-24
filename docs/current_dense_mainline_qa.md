# Current Dense Mainline QA

Base: `/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623`

## Object Refinement

| metric | v7 | v8 | delta |
|---|---:|---:|---:|
| candidate_count | 239 | 6656 | 6417 |
| accepted_candidate_rows | 96 | 1235 | 1139 |
| output_object_count | 197534 | 196395 | -1139 |
| mixed_object_voxel_ratio_020 | 0.18651164133709347 | 0.186092957448242 | -0.0004186838888514677 |
| object_count_in_overlap_preview | 51269 | 50831 | -438 |

## Surface Guard

| label | v9 points | v17 points | delta |
|---|---:|---:|---:|
| car | 8486 | 8486 | 0 |
| floor | 496641 | 496641 | 0 |
| grass | 205461 | 205461 | 0 |
| railing | 13599 | 13599 | 0 |
| unknown | 73085 | 73085 | 0 |
| wall | 650971 | 650971 | 0 |

## Gates

- Object ownership must remain exclusive.
- Mixed-object coarse voxel ratio must not regress while fragmentation decreases.
- Surface guard must not demote floor/wall to unknown solely due to missing teacher evidence.
- Semantic variants remain QA references until visual inspection passes.
