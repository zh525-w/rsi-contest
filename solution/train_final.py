"""Offline training entry: fit final artifacts for A/B/C from the provided baseline."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contest_core as core
import experiment_log

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "models"
key = sys.argv[1]
parent = ROOT / f"task-{key.lower()}" / "checkpoint.npz"
train = ROOT / f"task-{key.lower()}" / "public" / "train.jsonl"
val = ROOT / f"task-{key.lower()}" / "public" / "validation.jsonl"
t0 = time.time()
bundle, report = core.train_bundle(key, train, val, parent, round_index=1,
                                   log=lambda m: print(m, flush=True))
OUT.mkdir(parents=True, exist_ok=True)
(target := OUT / key).mkdir(exist_ok=True)
path = bundle.save(target / "checkpoint.npz")
report["parent_sha256"] = core._sha256(parent)
report["artifact_sha256"] = core._sha256(path)
report["experiment_record"] = experiment_log.log_round(
    key, report, parent_sha256=report["parent_sha256"],
    artifact_sha256=report["artifact_sha256"],
    config={"members": report["members"], "feature": report["feature"], "pca": report["pca"]},
    extra={"labels": "public/validation only", "device": "cpu", "numpy": "2.3.5"},
)
(target / "training.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "experiment_record"}, indent=2,
                 ensure_ascii=False))
print("experiment record:", json.dumps(report["experiment_record"], ensure_ascii=False))
print("saved", path, f"{time.time()-t0:.1f}s")
