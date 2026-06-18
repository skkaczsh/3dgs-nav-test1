#!/usr/bin/env bash
set -euo pipefail

# Pull the current RTX 5070Ti parking candidate route artifacts for local
# semantic_ply_viewer review.  This mirrors only review-sized outputs and reports,
# not the entire remote work directory.

SERVER="${SERVER:-scan-rtx5070}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
LOCAL_OUT="${LOCAL_OUT:-${LOCAL_REPO}/server_parking_priority_s10}"
DRY_RUN="${DRY_RUN:-0}"
PULL_QA_CROPS="${PULL_QA_CROPS:-0}"

rsync_opts=(-av --progress)
ssh_opts=(-o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}")
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=(-o "BindAddress=${BIND_ADDRESS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  rsync_opts+=(-n)
fi
printf -v rsync_ssh '%q ' ssh "${ssh_opts[@]}"
rsync_ssh="${rsync_ssh% }"

pull_dir() {
  local remote_rel="$1"
  local local_rel="$2"
  mkdir -p "${LOCAL_OUT}/${local_rel}"
  rsync "${rsync_opts[@]}" -e "${rsync_ssh}" \
    "${SERVER}:${REMOTE_WORK}/${remote_rel}/" \
    "${LOCAL_OUT}/${local_rel}/"
}

pull_file() {
  local remote_rel="$1"
  local local_rel="$2"
  mkdir -p "$(dirname "${LOCAL_OUT}/${local_rel}")"
  rsync "${rsync_opts[@]}" -e "${rsync_ssh}" \
    "${SERVER}:${REMOTE_WORK}/${remote_rel}" \
    "${LOCAL_OUT}/${local_rel}"
}

pull_qa_pack() {
  local remote_rel="$1"
  local local_rel="$2"
  if [[ "${PULL_QA_CROPS}" == "1" ]]; then
    pull_dir "${remote_rel}" "${local_rel}"
    return 0
  fi
  pull_file "${remote_rel}/frame_local_object_qa_report.json" "${local_rel}/frame_local_object_qa_report.json"
  pull_file "${remote_rel}/frame_local_object_qa_contact.jpg" "${local_rel}/frame_local_object_qa_contact.jpg"
  pull_file "${remote_rel}/frame_local_object_qa_candidates.jsonl" "${local_rel}/frame_local_object_qa_candidates.jsonl"
  pull_file "${remote_rel}/frame_local_object_qa_evidence.jsonl" "${local_rel}/frame_local_object_qa_evidence.jsonl"
}

echo "[1/5] checking remote candidate artifacts: ${SERVER}:${REMOTE_WORK}"
ssh "${ssh_opts[@]}" "${SERVER}" REMOTE_WORK="${REMOTE_WORK}" 'bash -s' <<'REMOTE'
set -euo pipefail
required=(
  frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_object_points_stride10.ply
  frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_objects_viewer.jsonl
  frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_local_object_qa_report.json
  guarded_v2_surface_refinement_all_risk_compare/qa_compare.md
  guarded_v2_surface_refinement_all_risk_compare/qa_compare.json
)
for rel in "${required[@]}"; do
  path="${REMOTE_WORK}/${rel}"
  if [[ ! -s "${path}" ]]; then
    echo "missing=${path}" >&2
    exit 1
  fi
  stat -c 'ok bytes=%s path=%n' "${path}"
done
REMOTE

echo "[2/5] pulling candidate viewer"
pull_dir \
  "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" \
  "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070"

echo "[3/5] pulling QA packs"
pull_qa_pack \
  "frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" \
  "frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070"
pull_qa_pack \
  "frame_local_object_qa_guarded_v2_full_s10_strict_surface_rtx5070" \
  "frame_local_object_qa_guarded_v2_full_s10_strict_surface_rtx5070"
pull_qa_pack \
  "frame_local_object_qa_guarded_v2_full_s10_ground_artifact_guard_strict_rtx5070" \
  "frame_local_object_qa_guarded_v2_full_s10_ground_artifact_guard_strict_rtx5070"
pull_qa_pack \
  "frame_local_object_qa_guarded_v2_full_s10_strict_surface_object_relabel_safe_span_rtx5070" \
  "frame_local_object_qa_guarded_v2_full_s10_strict_surface_object_relabel_safe_span_rtx5070"

echo "[4/5] pulling comparison and compact reports"
pull_dir \
  "guarded_v2_surface_refinement_all_risk_compare" \
  "guarded_v2_surface_refinement_all_risk_compare"
pull_file \
  "frame_objects_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/object_relabel_report.json" \
  "guarded_v2_ground_guard_object_relabel_reports/object_relabel_report.json"
pull_file \
  "frame_targets_guarded_v2_full_s10_ground_artifact_guard_rtx5070/geometry_refine_summary.json" \
  "guarded_v2_ground_artifact_guard_reports/geometry_refine_summary.json"

echo "[5/5] local review URL"
cat <<'URL'
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
URL

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
fi
if [[ "${PULL_QA_CROPS}" != "1" ]]; then
  echo "qa_crops=skipped set PULL_QA_CROPS=1 to pull crop images"
fi
