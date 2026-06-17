# Workspace Cleanup - 2026-06-17

Purpose:

- prepare the workspace for a new dataset
- preserve reusable technical conclusions
- remove failed or reproducible local artifacts that should not drive the next
  route

## Keep

- source code, scripts, tests, manifests, and README files
- route decision documents under `new_route/docs`
- compact JSON/Markdown reports copied from local server result directories
- representative reports already committed in experiment folders
- raw scan data and calibration assets unless explicitly superseded

## Delete

- local `server_*` synchronized result directories when they only contain
  regenerated experiment outputs
- old SAM2/Mimo projection products from failed routes
- large PLY/PNG/NPY visual artifacts under experiment `outputs*`,
  `analysis*`, `samples`, and `staged_samples`
- Python `__pycache__` folders
- ad-hoc debug PLY/PNG files from colorization and route experiments

## Preserved conclusions

- full-image SAM2/SAM3 auto-mask plus VLM label is not a reliable dense
  semantic main route
- generic Mask2Former/OneFormer ADE20K/Cityscapes/Mapillary models are smoother
  than noisy SAM2 masks but lack rooftop-domain priors
- GroundingDINO/Florence-style detector plus SAM2 is the most useful fine-object
  branch so far, especially for `railing` and `pipe`
- ConceptSeg-R1 and broad VLM-driven box proposals are useful as side evidence,
  not as a replacement for geometry-aware projection
- future work should prioritize geometry-aware surface extraction first, then
  run fine-object detection on non-stable residuals

## Local report archive

Compact reports from deleted local result folders are archived under:

- `/Users/skkac/Work/SCAN/new_route/docs/preserved_run_reports_20260617`

This archive is intended for text-level traceability only. Heavy point clouds,
rendered panels, intermediate masks, and rerunnable predictions are deliberately
not kept.
