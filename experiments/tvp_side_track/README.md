# TVP Side-Track

This folder holds the minimal server-side validation assets for
`Thinking-with-Visual-Primitives-pytorch`.

## Current scope

- `tvp_candidate_manifest_10.json`
  - 10-sample manifest built from accepted fine-object candidates.
- `run_tvp_manifest_inference.py`
  - raw manifest runner that parses `<|ref|>...<|box|>` and `<|point|>...`
    outputs.

## 2026-06-16 smoke result

Server: `scan-train`

Validated models:

- `yunfengwang/TVP-OPD-Qwen2VL-2B`
- `yunfengwang/TVP-SFTBox-Qwen2VL-2B`

Single-sample prompt:

- `Locate the railing in the image.`

Observed behavior on the current railing sample:

- both models loaded successfully on the server
- both models returned plain text
- both models returned `0` boxes and `0` points
- both responses said: `There is no railing visible in the image.`

## Interpretation

At the moment TVP is not a drop-in replacement for the dense semantic route.

- It is a visual-primitive model for box/point reasoning, not dense mask
  generation.
- On our current rooftop railing sample, it does not yet produce usable
  primitives.
- The most reasonable role for TVP remains a proposal / spatial-anchor
  side-track, not the main 2D dense segmentation path.

## Integration verdict for current project

The user's suggested idea was: merge the full PLY first, then use
`img_pos/cam_in_ex` plus TVP semantics to project back into space.

For the current repository, that is not the direct fit:

- TVP does not emit dense semantic maps; it emits sparse visual primitives such
  as boxes and points.
- Therefore TVP can help with `where to look`, but not with dense surface or
  thin-structure coverage by itself.
- If we use it at all, the correct integration point is:
  `TVP primitive proposal -> SAM2 local mask -> validated 3D projection`,
  not `TVP -> direct whole-scene semantic projection`.

## Engineering note

The server runner now supports a persistent snapshot directory via
`--download-dir` / `TVP_DOWNLOAD_DIR` so model weights no longer spill into
`/tmp`.
