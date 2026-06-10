# Old Route Side-Track Notes

Purpose:

- Keep the old route as a visual/colorization reference only.
- Do not use the old `transforms.json` / 3DGS coordinate chain as a semantic
  source.
- Any future 3DGS restart must be rebuilt from the validated scanner-native
  route: `img_pos.txt + cam_in_ex.txt + Tcl + Til`.

Current usable local artifacts:

- Smoke summary:
  `/Users/skkac/Work/SCAN/server_old_route_smoke/world_colorize_summary.json`
- Smoke preview:
  `/Users/skkac/Work/SCAN/server_old_route_smoke/old_route_world_color_smoke_s8_v010_best_chroma_xy.png`
- Local debug PLYs:
  `/Users/skkac/Work/SCAN/MT20260511-165822/outputs/debug_colorize_20_mid360_direct*.ply`
- Larger local reference:
  `/Users/skkac/Work/SCAN/MT20260511-165822/outputs/world_fused_visual_colorize_s200_v006_f120_remote.ply`

Current smoke result:

- sections: `8`
- source points: `64,437`
- fused points: `31,323`
- color frames: `12`
- colored points: `27,613`
- colored ratio: `0.8816`
- sample mode: `chroma_patch`
- sample radius: `6`
- fusion mode: `best_chroma`
- skymask source: `/root/epfs/manifold_3dgs_project/processed/final_masks`

Interpretation:

- The old route is useful as a colorization sanity check because the Mid360
  geometry is correct in the fixed direct route.
- It is not a semantic production path. The earlier semantic failures were
  caused by the deprecated `transforms.json + project_world_points()` chain and
  by low-quality 3DGS training, not by the scanner-native color projection
  itself.
- Next valid old-route work is a controlled comparison against the new route:
  same frame range, same skymask, same best-observation color rule, and visual
  QA only. It should not consume Qwen/SAM2 resources while main-route data
  preparation is running.
