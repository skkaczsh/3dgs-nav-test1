# GeoPatch Run Comparison

| run | patches | high entropy | large high entropy | large low purity | merge accepts | accepted profiles |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v4_spg | 197208 | 6410 | 1 | 12 | 422 | contact_bridge:13, near_bbox_bridge:198, score:211 |
| v7_uncertain | 197014 | 6361 | 1 | 12 | 616 | contact_bridge:16, near_bbox_bridge:195, score:105, uncertain_fragment_bridge:300 |

## v4_spg

- voxel p50/p90/p99/max: `1` / `11` / `255` / `2850203`
- entropy p50/p90/p99: `-0.000` / `0.918` / `1.485`
- top patches:
  - `4` voxels=2850203 geom=horizontal entropy=-0.000 purity=1.000 extent_ratio=30.0
  - `2` voxels=2666723 geom=vertical entropy=-0.000 purity=1.000 extent_ratio=2.7
  - `31` voxels=1156414 geom=vertical entropy=-0.000 purity=1.000 extent_ratio=2.7
  - `70448` voxels=1037993 geom=rough_mixed entropy=0.799 purity=0.799 extent_ratio=4.1
  - `7` voxels=724650 geom=horizontal entropy=-0.000 purity=1.000 extent_ratio=4.7

## v7_uncertain

- voxel p50/p90/p99/max: `1` / `11` / `244` / `2867111`
- entropy p50/p90/p99: `-0.000` / `0.918` / `1.485`
- top patches:
  - `4` voxels=2867111 geom=horizontal entropy=0.059 purity=0.994 extent_ratio=27.7
  - `2` voxels=2695662 geom=vertical entropy=0.098 purity=0.989 extent_ratio=2.7
  - `31` voxels=1174917 geom=vertical entropy=0.134 purity=0.984 extent_ratio=2.7
  - `70448` voxels=1037475 geom=rough_mixed entropy=0.799 purity=0.799 extent_ratio=4.1
  - `7` voxels=744733 geom=horizontal entropy=0.209 purity=0.973 extent_ratio=4.6
