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

## Engineering note

The server runner now supports a persistent snapshot directory via
`--download-dir` / `TVP_DOWNLOAD_DIR` so model weights no longer spill into
`/tmp`.
