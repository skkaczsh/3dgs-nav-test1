# Paired 3DGS Route Archive - 2026-06-17

This archive preserves the reusable documentation from the old `paired` route
before removing its local Git history from the workspace.

## Archived Source

- original path: `/Users/skkac/Work/SCAN/paired`
- archived because: the route is no longer on the main dense point-level
  semantic path
- cleanup action: preserve text-level knowledge, remove local Git history pack
  that occupied about 1.1 GB

## Preserved Files

- `DEBUG_RETROSPECTIVE.md`
- `ASSETS.md`
- `SKILL.md`
- `session-log-template.md`
- `sam_vlm_pipeline/TECH_ROUTE.md`
- `paired_docs/assets-debt.yaml`
- `paired_docs/knowledge.yaml`
- `paired_docs/roadmap.html`
- `paired_docs/superpowers/TIMELINE.md`
- `paired_docs/superpowers/session-logs/2026-05-19-1000--session.md`
- `paired_docs/superpowers/tech-assets/ASSETS.md`

## Route Conclusion

The old paired route spent most of its effort on 3DGS / gsplat coordinate,
depth, and VLM fusion diagnostics. It produced useful low-level lessons about
gsplat depth normalization, opacity handling, splat coordinate conventions, and
the cost of debugging untrusted geometry. It did not become the project main
route because the dense point-level semantic goal needs reliable scanner-camera
projection and geometry-aware mask fusion, while this route remained tied to
unstable 3DGS reconstruction and many ad-hoc diagnostic scripts.

## Lessons To Keep

- Do not treat normalized gsplat depth as metric depth without validating the
  renderer convention.
- Keep scanner-camera projection as the trusted geometric backbone.
- Preserve small, named diagnostic conclusions; delete rerunnable visual and PLY
  products.
- Do not let exploratory route repositories keep large Git object history inside
  the active dataset workspace.

## Not Preserved Here

- large historical Git objects
- old splat / PCD / PLY data products
- generated debug images
- ad-hoc diagnostic script history

The working files under `/Users/skkac/Work/SCAN/paired` were not deleted by this
archive step. Only its local `.git` history is intended to be removed.
