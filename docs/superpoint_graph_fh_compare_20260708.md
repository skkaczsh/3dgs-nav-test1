# GeoPatch Run Comparison

| run | patches | high entropy | large high entropy | large low purity | merge accepts | accepted profiles |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v4_spg | 197208 | 6410 | 1 | 12 | 422 | contact_bridge:13, near_bbox_bridge:198, score:211 |
| fh120k | 197349 | 6415 | 1 | 14 | 281 | contact_bridge:11, near_bbox_bridge:171, score:99 |
| fh240k | 197326 | 6415 | 1 | 12 | 304 | contact_bridge:16, near_bbox_bridge:184, score:104 |

## v4_spg

- voxel p50/p90/p99/max: `1` / `11` / `255` / `2850203`
- entropy p50/p90/p99: `-0.000` / `0.918` / `1.485`
- top patches:
  - `4` voxels=2850203 geom=horizontal entropy=-0.000 purity=1.000 extent_ratio=30.0
  - `2` voxels=2666723 geom=vertical entropy=-0.000 purity=1.000 extent_ratio=2.7
  - `31` voxels=1156414 geom=vertical entropy=-0.000 purity=1.000 extent_ratio=2.7
  - `70448` voxels=1037993 geom=rough_mixed entropy=0.799 purity=0.799 extent_ratio=4.1
  - `7` voxels=724650 geom=horizontal entropy=-0.000 purity=1.000 extent_ratio=4.7

## fh120k

- voxel p50/p90/p99/max: `1` / `11` / `261` / `2786013`
- entropy p50/p90/p99: `-0.000` / `0.918` / `1.485`
- top patches:
  - `4` voxels=2786013 geom=horizontal entropy=-0.000 purity=1.000 extent_ratio=59.0
  - `2` voxels=2442344 geom=vertical entropy=-0.000 purity=1.000 extent_ratio=2.7
  - `31` voxels=1118007 geom=vertical entropy=-0.000 purity=1.000 extent_ratio=2.7
  - `70415` voxels=660617 geom=rough_mixed entropy=1.067 purity=0.696 extent_ratio=1.8
  - `7` voxels=470534 geom=horizontal entropy=-0.000 purity=1.000 extent_ratio=30.4

## fh240k

- voxel p50/p90/p99/max: `1` / `11` / `259` / `2786013`
- entropy p50/p90/p99: `-0.000` / `0.918` / `1.485`
- top patches:
  - `4` voxels=2786013 geom=horizontal entropy=-0.000 purity=1.000 extent_ratio=59.0
  - `2` voxels=2442448 geom=vertical entropy=-0.000 purity=1.000 extent_ratio=2.7
  - `31` voxels=1156405 geom=vertical entropy=-0.000 purity=1.000 extent_ratio=2.7
  - `70448` voxels=1037475 geom=rough_mixed entropy=0.799 purity=0.799 extent_ratio=4.1
  - `70415` voxels=660617 geom=rough_mixed entropy=1.067 purity=0.696 extent_ratio=1.8
