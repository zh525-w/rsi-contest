"""Replay the local experiment records into SwanLab (run once, with credentials).

    export SWANLAB_API_KEY=...      # platform secret / shell env only
    export SWANLAB_PROJECT=industrial-contest
    export SWANLAB_WORKSPACE=<team>
    python solution/upload_experiments.py

Creates one SwanLab run per dataset, logs every recorded round as a step, and
prints the run URLs to paste into the submission form / technical report.
If SwanLab is not installed the script exits with an instruction instead of
silently pretending the upload happened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT.parent / "reports"


def load_rows():
    pack = REPORTS / "training-records" / "records.jsonl"
    if pack.is_file():
        return [json.loads(line) for line in pack.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for name in ("experiments.jsonl", "experiments_cloud.jsonl"):
        path = REPORTS / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    try:
        import swanlab
    except Exception:
        print("swanlab 未安装：pip install swanlab 后重试（密钥只放环境变量，不要写进镜像）")
        return 2
    key = os.environ.get("SWANLAB_API_KEY")
    if key:
        # non-interactive login; the key must come from the platform secret field / shell env
        swanlab.login(api_key=key)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["A", "B", "C"], default=None)
    args = parser.parse_args()
    rows = load_rows()
    if args.dataset:
        rows = [row for row in rows if row.get("dataset") == args.dataset]
    if not rows:
        print("没有可上传的记录：先运行 solution/train_final.py 或 train_strict.py")
        return 1
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row.get("dataset", "?"), []).append(row)
    for dataset, items in sorted(by_dataset.items()):
        run = swanlab.init(project=os.environ.get("CONTEST_SWAN_PROJECT", "industrial-contest"),
                           workspace=os.environ.get("CONTEST_SWAN_WORKSPACE") or None,
                           experiment_name=f"{dataset}-experiments",
                           config={"dataset": dataset, "records": len(items)})
        for step, item in enumerate(items, 1):
            metrics = {}
            for key in ("macro_f1", "accuracy", "rmse", "mae"):
                value = (item.get("validation") or {}).get(key)
                if isinstance(value, (int, float)):
                    metrics[f"val/{key}"] = float(value)
            metrics["round"] = int(item.get("round", item.get("round_index", step)) or step)
            swanlab.log(metrics, step=step)
        url = getattr(run, "url", "") or ""
        swanlab.finish()
        print(f"{dataset}: {len(items)} records -> {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
