#!/usr/bin/env bash
set -euo pipefail

IMAGE_GLOB="${IMAGE_GLOB:-/root/epfs/new_route_stage1_skymask/sam2_input_2000_2999/cam0_00200[0-9].png}"
BASE_DIR="${BASE_DIR:-/root/epfs/sam2_tensorrt/sweeps/nms_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
BOX_VALUES="${BOX_VALUES:-0.55 0.65 0.7}"
CROP_VALUES="${CROP_VALUES:-0.55 0.65 0.7}"
PRED_VALUES="${PRED_VALUES:-0.7}"
STABILITY_VALUES="${STABILITY_VALUES:-0.92}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${BASE_DIR}"
SUMMARY_CSV="${BASE_DIR}/summary.csv"
echo "run_id,box_nms,crop_nms,pred_iou,stability,images,baseline_masks,candidate_masks,coverage_delta,matched_iou,unmatched_baseline,unmatched_candidate" > "${SUMMARY_CSV}"

for box in ${BOX_VALUES}; do
  for crop in ${CROP_VALUES}; do
    for pred in ${PRED_VALUES}; do
      for stability in ${STABILITY_VALUES}; do
        run_id="box${box}_crop${crop}_pred${pred}_stab${stability}"
        candidate_dir="${BASE_DIR}/${run_id}/candidate"
        report_dir="${BASE_DIR}/${run_id}/report"
        echo "[sweep] ${run_id}"
        IMAGE_GLOB="${IMAGE_GLOB}" \
          CANDIDATE_DIR="${candidate_dir}" \
          REPORT_DIR="${report_dir}" \
          GPU_ID="${GPU_ID}" \
          OUTPUT_MODE=uncompressed_rle \
          BOX_NMS_THRESH="${box}" \
          CROP_NMS_THRESH="${crop}" \
          PRED_IOU_THRESH="${pred}" \
          STABILITY_SCORE_THRESH="${stability}" \
          bash "${SCRIPT_DIR}/run_server_sam2_trt_benchmark.sh" > "${BASE_DIR}/${run_id}.log" 2>&1

        python3 - "${report_dir}/sam2_trt_benchmark_compare.json" "${SUMMARY_CSV}" "${run_id}" "${box}" "${crop}" "${pred}" "${stability}" <<'PY'
import csv
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
run_id, box, crop, pred, stability = sys.argv[3:8]
summary = json.loads(report_path.read_text(encoding="utf-8"))["summary"]
with csv_path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        run_id,
        box,
        crop,
        pred,
        stability,
        summary["images"],
        summary["mean_baseline_masks"],
        summary["mean_candidate_masks"],
        summary["mean_coverage_delta"],
        summary["mean_matched_iou"],
        summary["mean_unmatched_baseline_masks"],
        summary["mean_unmatched_candidate_masks"],
    ])
PY
      done
    done
  done
done

python3 - "${SUMMARY_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
def score(row):
    coverage = abs(float(row["coverage_delta"]))
    unmatched_b = float(row["unmatched_baseline"])
    unmatched_c = float(row["unmatched_candidate"])
    iou_penalty = max(0.0, 0.95 - float(row["matched_iou"]))
    return coverage * 10.0 + unmatched_b * 0.1 + unmatched_c * 0.05 + iou_penalty * 10.0

for row in sorted(rows, key=score)[:10]:
    row["score"] = f"{score(row):.4f}"
    print(row)
PY

echo "summary_csv=${SUMMARY_CSV}"
