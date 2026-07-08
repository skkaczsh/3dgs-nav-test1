# Superpoint Graph Edge Sparsity 2026-07-08

Input: dense Opt-LAS `0.03m` voxel region input.

| run | patches | edge pairs | isolated patches | isolated voxels | isolated size 1 | isolated 2-9 | isolated 10-99 | isolated 100-999 | isolated 1000-9999 | isolated 10000+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 SPG | 197,208 | 7,569 | 189,029 | 2,716,607 | 105,374 | 63,356 | 16,732 | 3,280 | 274 | 13 |
| FH240 | 197,326 | 7,672 | 189,036 | 3,244,373 | 105,374 | 63,356 | 16,732 | 3,279 | 279 | 16 |

Largest isolated examples:

- v4: patch `58` horizontal 43,885 voxels; patch `42` vertical 38,899 voxels.
- FH240: patch `7` horizontal 470,534 voxels; patch `58` horizontal 43,885 voxels.

Conclusion:

- The candidate graph is the bottleneck. More than 95% of patches have no graph
  edge after clustering.
- Most isolated patches are tiny 1-9 voxel fragments, but there are also large
  isolated surfaces. This is not solved by tuning `FH_K`.
- `precluster_small_patches.py` can only merge patches that already have contact
  edges, so it cannot fix the dominant no-edge case.

Next useful work:

1. Add a candidate-edge diagnostic for isolated patches: nearest compatible
   occupied-cell neighbors by fine voxel/grid bucket, not just bbox gap.
2. Only after the diagnostic shows real missing near neighbors, add those edges
   into `cluster_superpoint_graph.py`.
3. Keep v4 as metric baseline until a candidate-edge run reduces isolated
   large patches or high-entropy count without increasing fine overlap.
