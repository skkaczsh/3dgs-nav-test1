#!/usr/bin/env bash
# Run after the first-pass VLM report is complete; it never mutates that pass.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ROOT SOURCE_FRAME_SUPPORT CONTACT_EDGES FULL_GEOMETRY_OBJECTS OUTPUT_DIR" >&2
  exit 2
fi

root=$1
source_support=$2
contact_edges=$3
geometry_objects=$4
out=$5
objects_jsonl=${OBJECTS_JSONL:-"$root/objects.jsonl"}
evidence_dir=${EVIDENCE_DIR:-"$root/evidence"}
first=${REVIEW_DIR:-"$root/qwen_review"}
test ! -e "$out"
test -f "$geometry_objects"
test -f "$objects_jsonl"
test -f "$evidence_dir/object_image_evidence.jsonl"
test -f "$first/mimo_object_review.jsonl"
mkdir -p "$out"

expected=$(python3 - "$evidence_dir/object_image_evidence.jsonl" <<'PY'
import json, sys
print(len({int(row["object_id"]) for row in map(json.loads, open(sys.argv[1])) if row}))
PY
)
python3 - "$first/mimo_object_review_report.json" "$expected" <<'PY'
import json, sys
report=json.load(open(sys.argv[1]))
expected=int(sys.argv[2])
assert report["reviewed_objects"] == expected, report
assert report["parse_ok"] == report["reviewed_objects"], report
PY

cp "$first/mimo_object_review.jsonl" "$out/first_review.jsonl"
python3 scripts/select_structural_refinement_candidates.py \
  --objects-jsonl "$objects_jsonl" --review-jsonl "$out/first_review.jsonl" \
  --output-objects "$out/structural_candidates.jsonl" --output-ids "$out/structural_candidate_ids.json" \
  --report "$out/structural_candidates_report.json" --min-confidence 0.8

mkdir -p "$out/structure_review"
if [[ $(wc -l < "$out/structural_candidates.jsonl") -gt 0 ]]; then
  # ponytail: the caller supplies the VLM endpoint credentials; no second config layer.
  python3 scripts/run_mimo_object_review.py --objects-jsonl "$out/structural_candidates.jsonl" \
    --evidence-jsonl "$evidence_dir/object_image_evidence.jsonl" --output-dir "$out/structure_review" \
    --task structure --top-k 2 --concurrency 4 --timeout 180 --retries 1 --max-tokens 1024 --image-mode both
else
  : > "$out/structure_review/mimo_object_review.jsonl"
fi

python3 scripts/merge_superpoint_review_rounds.py \
  --first-review-jsonl "$out/first_review.jsonl" \
  --structural-review-jsonl "$out/structure_review/mimo_object_review.jsonl" \
  --output-jsonl "$out/merged_reviews.jsonl" --report "$out/merged_reviews_report.json" --min-confidence 0.8
python3 scripts/materialize_superpoint_observation_ledger.py \
  --evidence-jsonl "$evidence_dir/object_image_evidence.jsonl" --objects-jsonl "$objects_jsonl" \
  --review-jsonl "$out/merged_reviews.jsonl" --source-frame-support "$source_support" \
  --output-jsonl "$out/observations.jsonl" --report "$out/observations_report.json"
python3 scripts/build_superpoint_anchor_posteriors.py --objects-jsonl "$objects_jsonl" \
  --review-jsonl "$out/merged_reviews.jsonl" --output-jsonl "$out/anchor_posteriors.jsonl" --min-confidence 0.8
python3 scripts/propagate_superpoint_structural_anchors.py --contact-edges "$contact_edges" \
  --anchor-posteriors "$out/anchor_posteriors.jsonl" --output-jsonl "$out/structural_posteriors.jsonl" \
  --geometry-objects-jsonl "$geometry_objects" \
  --report "$out/structural_posterior_report.json" --min-faces 10 --contact-faces-norm 100 --color-sigma 40 --max-hops 2 --min-confidence 0.35 --min-margin 0.15
python3 scripts/build_superpoint_structure_regions.py --structural-posteriors "$out/structural_posteriors.jsonl" \
  --contact-edges "$contact_edges" --objects-jsonl "$geometry_objects" \
  --output-regions "$out/regions.jsonl" --output-assignments "$out/region_assignments.jsonl" \
  --report "$out/regions_report.json" --min-faces 10 --contact-faces-norm 100 --color-sigma 40
