"""Experiment recording for the contest.

Local JSONL is always written (works offline, in the judge image and on a laptop).
A SwanLab cloud run is attempted only when SwanLab is installed *and* cloud mode is
explicitly requested, because the official evaluation environment is offline.

Environment (never put secrets in the image; platform secret fields / shell env only):

    SWANLAB_API_KEY    SwanLab API key
    SWANLAB_PROJECT    project name, e.g. "industrial-contest"
    SWANLAB_WORKSPACE  workspace / team name
    SWANLAB_MODE       "cloud" | "local" | "disabled" | "offline"  (default: local)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

LOCAL_LOG = Path(__file__).resolve().parent.parent / "reports" / "experiments_cloud.jsonl"


def _swanlab():
    try:
        import swanlab  # noqa: F401
        return swanlab
    except Exception:
        return None


def cloud_enabled():
    mode = os.environ.get("SWANLAB_MODE", "local").strip().lower()
    return mode in ("cloud", "online") and _swanlab() is not None


def _record(payload):
    LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def log_round(dataset, report, *, parent_sha256=None, artifact_sha256=None,
              config=None, project=None, workspace=None, run_name=None, extra=None):
    """Record one training round; returns a dict describing where it was written."""
    payload = {
        "dataset": dataset,
        "round_index": report.get("round_index"),
        "task": report.get("task"),
        "track": report.get("track", "boost"),
        "validation": report.get("validation"),
        "members": report.get("members"),
        "feature": report.get("feature"),
        "pca": report.get("pca"),
        "training_seconds": report.get("training_seconds"),
        "parent_sha256": parent_sha256 or report.get("parent_sha256"),
        "artifact_sha256": artifact_sha256,
        "config": config or {},
        "extra": extra or {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "local",
        "url": "",
    }
    swanlab = _swanlab() if cloud_enabled() else None
    if swanlab is None:
        payload["note"] = "swanlab 未安装或 SWANLAB_MODE 未设为 cloud；已写入本地 JSONL"
        _record(payload)
        return payload
    try:
        init_kwargs = {
            "project": project or os.environ.get("SWANLAB_PROJECT", "industrial-contest"),
            "workspace": workspace or os.environ.get("SWANLAB_WORKSPACE") or None,
            "experiment_name": run_name or f"{dataset}-round-{report.get('round_index', 0)}",
            "config": {"dataset": dataset, **(config or {}),
                       "feature": report.get("feature"), "pca": report.get("pca"),
                       "parent_sha256": payload["parent_sha256"]},
            "mode": os.environ.get("SWANLAB_MODE", "cloud"),
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        run = swanlab.init(**init_kwargs)
        metrics = {f"val/{k}": float(v) for k, v in (report.get("validation") or {}).items()
                   if isinstance(v, (int, float))}
        metrics["round"] = int(report.get("round_index", 0))
        metrics["train/seconds"] = float(report.get("training_seconds") or 0.0)
        swanlab.log(metrics, step=int(report.get("round_index", 0)))
        payload["backend"] = "swanlab"
        payload["url"] = getattr(run, "url", "") or ""
        if not payload["url"]:
            getter = getattr(swanlab, "get_url", None)
            payload["url"] = getter() if callable(getter) else ""
        swanlab.finish()
    except Exception as exc:  # network/credentials errors must never break training
        payload["backend"] = "local"
        payload["note"] = f"swanlab 记录失败，已保留本地记录：{type(exc).__name__}: {exc}"
    _record(payload)
    return payload
