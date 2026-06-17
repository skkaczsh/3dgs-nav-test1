#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


ROOT = Path("/Users/skkac/Work/SCAN")
REMOTE = "root@10.0.8.114"
REMOTE_PORT = "31909"
REMOTE_TMP = "/root/epfs/vlm_seg_project/tmp_surface_fine_combo_small_eval"
REMOTE_SCRIPT = "/root/epfs/vlm_seg_project/tmp_surface_baseline_small_eval/run_surface_baseline_small_eval.py"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--remote-name", default="outputs_mask2former_ade20k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=["mask2former_ade20k"])
    parser.add_argument("--sync-local-dir", type=Path, default=None)
    args = parser.parse_args()

    samples_dir = args.samples_dir.resolve()
    remote_samples = f"{REMOTE_TMP}/samples"
    remote_output_dir = f"{REMOTE_TMP}/{args.remote_name}"

    run(["ssh", "-F", "/dev/null", "-p", REMOTE_PORT, REMOTE, f"mkdir -p {shlex.quote(REMOTE_TMP)}"])
    run(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -F /dev/null -p {REMOTE_PORT}",
            f"{samples_dir}/",
            f"{REMOTE}:{remote_samples}/",
        ]
    )

    model_args = " ".join(shlex.quote(model) for model in args.models)
    remote_cmd = f"""
set -e
export HF_HOME=/root/epfs/hf_home
export HUGGINGFACE_HUB_CACHE=/root/epfs/hf_cache
export TRANSFORMERS_CACHE=/root/epfs/hf_cache/transformers
/root/epfs/conda_envs/conceptseg-r1/bin/python {shlex.quote(REMOTE_SCRIPT)} \
  --samples-dir {shlex.quote(remote_samples)} \
  --output-dir {shlex.quote(remote_output_dir)} \
  --device {shlex.quote(args.device)} \
  --models {model_args}
"""
    run(["ssh", "-F", "/dev/null", "-p", REMOTE_PORT, REMOTE, remote_cmd])

    if args.sync_local_dir:
        args.sync_local_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh -F /dev/null -p {REMOTE_PORT}",
                f"{REMOTE}:{remote_output_dir}/",
                f"{args.sync_local_dir.resolve()}/",
            ]
        )

    print(remote_output_dir)


if __name__ == "__main__":
    main()
