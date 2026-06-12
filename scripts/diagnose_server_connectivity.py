#!/usr/bin/env python3
"""Diagnose local connectivity to scan servers."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
from pathlib import Path
from typing import Sequence


def run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def local_ipv4() -> list[dict]:
    code, out, _ = run(["ifconfig"], timeout=5)
    if code != 0:
        return []
    rows = []
    iface = ""
    for line in out.splitlines():
        if line and not line.startswith("\t") and ":" in line:
            iface = line.split(":", 1)[0]
        stripped = line.strip()
        if stripped.startswith("inet "):
            parts = stripped.split()
            if len(parts) >= 2:
                rows.append({"interface": iface, "address": parts[1]})
    return rows


def ssh_config(host: str) -> dict:
    code, out, err = run(["ssh", "-G", host], timeout=5)
    cfg = {"host": host, "returncode": code, "stderr": err.strip()}
    for line in out.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(" ")
        if key in {"hostname", "port", "user", "bindaddress", "identityfile"}:
            cfg[key] = value.strip()
    return cfg


def tcp_check(hostname: str, port: int, timeout: float) -> dict:
    result = {"hostname": hostname, "port": port, "reachable": False, "error": ""}
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((hostname, port))
        result["reachable"] = True
    except OSError as exc:
        result["error"] = str(exc)
    finally:
        sock.close()
    return result


def parse_endpoint(value: str) -> dict:
    name, _, endpoint = value.partition("=")
    if not endpoint:
        name = value
        endpoint = value
    hostname, _, port_text = endpoint.rpartition(":")
    if not hostname or not port_text:
        raise argparse.ArgumentTypeError(f"endpoint must be NAME=HOST:PORT or HOST:PORT, got {value!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid endpoint port in {value!r}") from exc
    return {"name": name, "hostname": hostname, "port": port}


def diagnose(hosts: list[str], direct_endpoints: Sequence[dict], timeout: float) -> dict:
    ipv4 = local_ipv4()
    local_addresses = {row["address"] for row in ipv4}
    host_reports = []
    for host in hosts:
        cfg = ssh_config(host)
        hostname = cfg.get("hostname", "")
        port = int(cfg.get("port", "22"))
        bind = cfg.get("bindaddress", "")
        tcp = tcp_check(hostname, port, timeout) if hostname else {"reachable": False, "error": "missing hostname"}
        host_reports.append(
            {
                "host": host,
                "ssh_config": cfg,
                "tcp": tcp,
                "bind_address_present_locally": (not bind) or bind in local_addresses,
            }
        )
    direct_reports = [
        {
            "name": row["name"],
            "tcp": tcp_check(row["hostname"], row["port"], timeout),
        }
        for row in direct_endpoints
    ]
    return {
        "local_ipv4": ipv4,
        "hosts": host_reports,
        "direct_endpoints": direct_reports,
        "all_hosts_reachable": all(row["tcp"]["reachable"] for row in host_reports),
        "all_direct_endpoints_reachable": all(row["tcp"]["reachable"] for row in direct_reports),
        "all_reachable": all(row["tcp"]["reachable"] for row in host_reports)
        and all(row["tcp"]["reachable"] for row in direct_reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", nargs="+", default=["scan-train", "scan-vlm"])
    parser.add_argument(
        "--direct-endpoints",
        nargs="+",
        type=parse_endpoint,
        default=[
            {"name": "scan-train-direct", "hostname": "10.0.8.114", "port": 31909},
            {"name": "scan-vlm-direct", "hostname": "10.0.8.114", "port": 31079},
        ],
        help="Direct TCP endpoints as NAME=HOST:PORT. Defaults to the two scan server forwarded SSH ports.",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = diagnose(args.hosts, args.direct_endpoints, args.timeout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_reachable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
