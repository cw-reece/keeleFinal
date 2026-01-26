# run_metadata.py
"""Lightweight experiment run metadata + folder management.

What it does:
- Creates a timestamped run folder under experiments/runs/<run_id>/
- Snapshots your config file into the run folder
- Captures git info (commit, branch, dirty state)
- Captures environment info (python/platform; optional torch/cuda; optional pip freeze)
- Writes run.json (metadata) + metrics.json (when you call write_metrics)

Designed to be imported from train/eval scripts OR used directly.

No hard deps beyond the Python standard library.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def local_now_run_id(tag: str) -> str:
    """Example: 20260112_1430_baseline_vilt_seed42"""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag.strip())
    return f"{ts}_{safe_tag}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr). Never raises."""
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}

    rc, out, err = run_cmd(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_root)
    if rc != 0 or out.lower() != "true":
        info["error"] = err or "Not a git repository"
        return info

    info["available"] = True

    rc, commit, _ = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    rc2, branch, _ = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    rc3, status, _ = run_cmd(["git", "status", "--porcelain"], cwd=repo_root)

    info["commit"] = commit if rc == 0 else None
    info["branch"] = branch if rc2 == 0 else None
    info["dirty"] = bool(status) if rc3 == 0 else None
    info["status_porcelain"] = status if rc3 == 0 else None
    return info


def get_torch_info() -> Dict[str, Any]:
    """Optional: collects torch/cuda info if torch is installed."""
    try:
        import torch  # type: ignore

        cuda_available = torch.cuda.is_available()
        out = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        if cuda_available:
            out["gpu_count"] = torch.cuda.device_count()
            out["gpu_name_0"] = torch.cuda.get_device_name(0)
        return out
    except Exception as e:
        return {"torch_available": False, "error": f"{type(e).__name__}: {e}"}


def get_pip_freeze() -> Dict[str, Any]:
    """Optional: captures 'pip freeze'. Can be slow but is very useful."""
    rc, out, err = run_cmd([sys.executable, "-m", "pip", "freeze"])
    if rc == 0:
        return {"pip_freeze": out.splitlines()}
    return {"pip_freeze_error": err or "pip freeze failed"}


def get_system_info(include_pip_freeze: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "user": os.getenv("USER") or os.getenv("USERNAME"),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": get_torch_info(),
    }
    if include_pip_freeze:
        info.update(get_pip_freeze())
    return info


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def snapshot_config(config_path: Path, run_dir: Path) -> Dict[str, Any]:
    """Copies config into run_dir and records a sha256 hash."""
    if not config_path.exists():
        return {"config_found": False, "config_path": str(config_path)}

    ensure_dir(run_dir)
    dest_name = f"config{config_path.suffix or '.yaml'}"
    dest = run_dir / dest_name
    shutil.copy2(config_path, dest)

    return {
        "config_found": True,
        "config_original_path": str(config_path),
        "config_snapshot_path": str(dest),
        "config_sha256": sha256_file(dest),
    }


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    run_json_path: Path


def init_run(
    *,
    tag: str,
    config_path: str | Path,
    out_root: str | Path = "experiments/runs",
    repo_root: str | Path = ".",
    include_pip_freeze: bool = False,
    extra: Optional[Dict[str, Any]] = None,
    argv: Optional[list[str]] = None,
) -> RunContext:
    """Create run folder + write run.json metadata. Returns RunContext."""
    out_root_p = Path(out_root)
    repo_root_p = Path(repo_root)

    run_id = local_now_run_id(tag)
    run_dir = out_root_p / run_id
    ensure_dir(run_dir)

    cfg_info = snapshot_config(Path(config_path), run_dir)
    git_info = get_git_info(repo_root_p.resolve())
    sys_info = get_system_info(include_pip_freeze=include_pip_freeze)

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "tag": tag,
        "argv": argv if argv is not None else sys.argv,
        "paths": {"run_dir": str(run_dir)},
        "git": git_info,
        "system": sys_info,
        "config": cfg_info,
    }
    if extra:
        payload["extra"] = extra

    run_json = run_dir / "run.json"
    write_json(run_json, payload)

    return RunContext(run_id=run_id, run_dir=run_dir, run_json_path=run_json)


def write_metrics(run_dir: str | Path, metrics: Dict[str, Any]) -> Path:
    """Write experiments/runs/<run_id>/metrics.json"""
    run_dir_p = Path(run_dir)
    metrics_path = run_dir_p / "metrics.json"
    write_json(metrics_path, metrics)
    return metrics_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", default="experiments/runs")
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--pip_freeze", action="store_true")
    args = ap.parse_args()

    ctx = init_run(
        tag=args.tag,
        config_path=args.config,
        out_root=args.out_root,
        repo_root=args.repo_root,
        include_pip_freeze=args.pip_freeze,
        argv=sys.argv,
    )

    print(f"Created run: {ctx.run_id}")
    print(f"Run dir: {ctx.run_dir}")
    print(f"Metadata: {ctx.run_json_path}")
