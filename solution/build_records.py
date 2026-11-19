"""Build the consolidated training-record pack from every recorded source.

Outputs (reports/training-records/):
  records.jsonl   one normalized row per recorded experiment / round
  summary.csv     flat table for spreadsheets
  summary.md      human-readable table for the technical report
  manifest.json   file list with sha256 + generation time
No network, no secrets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "training-records"
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iso_mtime(path):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))


def metric_value(record, task=None):
    validation = record.get("validation") or {}
    for key in ("macro_f1", "rmse", "mae", "accuracy"):
        if key in validation:
            return key, validation[key]
    return None, None


records = []

# 1) curated decision log (hypothesis -> change -> evidence -> decision)
for index, row in enumerate(load_jsonl(REPORTS / "experiments.jsonl"), 1):
    key, value = metric_value(row)
    records.append({
        "id": f"decision-{row.get('dataset')}-r{row.get('round')}-{index}",
        "kind": "decision",
        "dataset": row.get("dataset"),
        "round": row.get("round"),
        "track": row.get("track", "boost"),
        "hypothesis": row.get("hypothesis"),
        "change": row.get("change"),
        "config": row.get("config") or {},
        "metric_name": key,
        "metric_value": value,
        "validation": row.get("validation"),
        "baseline": row.get("baseline"),
        "decision": row.get("decision"),
        "reason": row.get("reason"),
        "source": "reports/experiments.jsonl",
    })

# 2) per-run experiment records written by train_final / harness
for row in load_jsonl(REPORTS / "experiments_cloud.jsonl"):
    key, value = metric_value(row)
    records.append({
        "id": f"run-{row.get('dataset')}-r{row.get('round_index')}-{row.get('track', 'boost')}",
        "kind": "training_run",
        "dataset": row.get("dataset"),
        "round": row.get("round_index"),
        "track": row.get("track", "boost"),
        "config": {"members": row.get("members"), "feature": row.get("feature"),
                   "pca": row.get("pca"), **(row.get("config") or {})},
        "metric_name": key,
        "metric_value": value,
        "validation": row.get("validation"),
        "parent_sha256": row.get("parent_sha256"),
        "artifact_sha256": row.get("artifact_sha256"),
        "training_seconds": row.get("training_seconds"),
        "created_at": row.get("created_at"),
        "backend": row.get("backend"),
        "source": "reports/experiments_cloud.jsonl",
    })

# 3) final artifacts (boost track)
for key in ("A", "B", "C"):
    path = ROOT / "solution" / "models" / key / "training.json"
    if not path.is_file():
        continue
    report = json.loads(path.read_text(encoding="utf-8"))
    metric, value = metric_value(report)
    records.append({
        "id": f"artifact-{key}-boost",
        "kind": "artifact",
        "dataset": key,
        "round": report.get("round_index"),
        "track": "boost",
        "config": {"members": report.get("members"), "feature": report.get("feature"),
                   "pca": report.get("pca"), "strategy": report.get("strategy")},
        "metric_name": metric,
        "metric_value": value,
        "validation": report.get("validation"),
        "parent_sha256": report.get("parent_sha256"),
        "artifact_sha256": report.get("artifact_sha256"),
        "source": f"solution/models/{key}/training.json",
        "source_mtime": iso_mtime(path),
    })

# 4) strict-continuation artifacts
for key in ("A", "B", "C"):
    path = ROOT / "solution" / "models_strict" / key / "training.json"
    if not path.is_file():
        continue
    report = json.loads(path.read_text(encoding="utf-8"))
    metric, value = metric_value({"validation": report.get("validation_after")})
    records.append({
        "id": f"artifact-{key}-strict",
        "kind": "artifact",
        "dataset": key,
        "round": None,
        "track": "strict",
        "config": {"epochs": report.get("epochs_requested"),
                   "learning_rate": report.get("learning_rate"),
                   "best_epoch": report.get("best_epoch")},
        "metric_name": metric,
        "metric_value": value,
        "validation": report.get("validation_after"),
        "baseline": report.get("validation_before"),
        "parent_sha256": report.get("parent_checkpoint_sha256"),
        "artifact_sha256": report.get("checkpoint_sha256"),
        "source": f"solution/models_strict/{key}/training.json",
        "source_mtime": iso_mtime(path),
    })

records.sort(key=lambda item: (str(item.get("dataset")), item.get("kind"), str(item.get("id"))))

with (OUT / "records.jsonl").open("w", encoding="utf-8") as stream:
    for record in records:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

fields = ["id", "kind", "dataset", "round", "track", "metric_name", "metric_value",
          "decision", "config", "parent_sha256", "artifact_sha256", "source", "created_at"]
with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        row = dict(record)
        row["config"] = json.dumps(row.get("config") or {}, ensure_ascii=False, sort_keys=True)
        row["metric_value"] = "" if row.get("metric_value") is None else row["metric_value"]
        writer.writerow(row)

lines = ["# 训练记录汇总", "",
         f"生成时间：{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
         f"记录数：{len(records)}（decision {sum(r['kind']=='decision' for r in records)} / "
         f"training_run {sum(r['kind']=='training_run' for r in records)} / "
         f"artifact {sum(r['kind']=='artifact' for r in records)}）", "",
         "| 数据集 | 类型 | 轮次 | 轨道 | 指标 | 值 | 决策 | 来源 |",
         "|---|---|---|---|---|---|---|---|"]
for record in records:
    value = record.get("metric_value")
    value = "" if value is None else (f"{value:.4f}" if isinstance(value, float) else value)
    lines.append(f"| {record.get('dataset')} | {record.get('kind')} | {record.get('round') or ''} | "
                 f"{record.get('track')} | {record.get('metric_name') or ''} | {value} | "
                 f"{record.get('decision') or ''} | {record.get('source')} |")
(OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# human-readable appendix: every record with hypothesis / change / evidence
appendix = ["# 附录：训练与实验记录明细", "",
            f"生成时间：{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}；"
            f"共 {len(records)} 条记录。数据只来自平台公开投影的训练/验证文件。", ""]
for dataset in ("A", "B", "C"):
    subset = [r for r in records if r.get("dataset") == dataset]
    if not subset:
        continue
    appendix.append(f"## 数据集 {dataset}（{len(subset)} 条）")
    appendix.append("")
    for record in subset:
        appendix.append(f"### {record['id']}")
        appendix.append(f"- 类型/轨道：{record.get('kind')} / {record.get('track')}")
        if record.get("round"):
            appendix.append(f"- 轮次：{record['round']}")
        if record.get("hypothesis"):
            appendix.append(f"- 假设：{record['hypothesis']}")
        if record.get("change"):
            appendix.append(f"- 改动：{record['change']}")
        if record.get("config"):
            appendix.append(f"- 配置：`{json.dumps(record['config'], ensure_ascii=False, sort_keys=True)}`")
        if record.get("validation"):
            appendix.append(f"- 验证指标：`{json.dumps(record['validation'], ensure_ascii=False)}`")
        if record.get("baseline"):
            appendix.append(f"- 对照：`{json.dumps(record['baseline'], ensure_ascii=False)}`")
        if record.get("decision"):
            appendix.append(f"- 决策：{record['decision']}（{record.get('reason') or ''}）")
        if record.get("parent_sha256"):
            appendix.append(f"- 父检查点：`{record['parent_sha256']}`")
        if record.get("artifact_sha256"):
            appendix.append(f"- 产物摘要：`{record['artifact_sha256']}`")
        appendix.append(f"- 来源：`{record['source']}`")
        appendix.append("")
(OUT / "appendix.md").write_text("\n".join(appendix) + "\n", encoding="utf-8")

manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "records": len(records), "files": []}
for path in sorted(OUT.iterdir()):
    if path.name == "manifest.json":
        continue
    manifest["files"].append({
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    })
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
print(f"wrote {len(records)} records to {OUT}")
