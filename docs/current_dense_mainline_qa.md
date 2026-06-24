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

Unknown point delta v17-v9: `0`

## Rejected Guard Diagnostics

| variant | unknown points | delta vs v9 | top rejection reasons |
|---|---:|---:|---|
| objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor | 1079157 | 1006072 | kept_unchecked_label=98413, wall_fragment_too_small_without_teacher=37037, floor_fragment_too_small_without_teacher=27477 |
| objects_v16_teacher_v20_grid6_geometry_guard_surface_recall | 1030136 | 957051 | kept_unchecked_label=98413, kept_wall_geometry_guard=38507, kept_floor_geometry_guard=27674 |

## Gates

- Object ownership must remain exclusive.
- Mixed-object coarse voxel ratio must not regress while fragmentation decreases.
- Surface guard must not demote floor/wall to unknown solely due to missing teacher evidence.
- Semantic variants remain QA references until visual inspection passes.
