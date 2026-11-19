"""Strict baseline continuation: keep the provided MLP and keep training it.

Uses the platform's own ``baselines/model.py`` and ``baselines/data.py`` so the
resulting ``checkpoint.npz`` is byte-compatible with ``contest.cpu-mlp.v1`` and
is a verifiable continuation of the shipped checkpoint (same weights, Adam
state, feature schema and target scaling; only epochs/LR change).
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "task-c" / "baselines"))
from model import Model, checkpoint_digest  # noqa: E402
import experiment_log  # noqa: E402
from data import read_dataset  # noqa: E402

CLASSES_A = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
CONFIG = {
    "A": {"profile": "A", "task": "classification", "classes": CLASSES_A, "epochs": 200, "lr_factor": 0.35},
    "B": {"profile": "B", "task": "regression", "classes": [], "epochs": 100, "lr_factor": 0.30},
    "C": {"profile": "C", "task": "regression", "classes": [], "epochs": 60, "lr_factor": 0.35},
}


def metrics(model, data):
    pred = model.predict(data["x"])
    if model.meta["task"] == "classification":
        labels = pred.argmax(1)
        f1 = []
        for c in range(len(model.meta["classes"])):
            tp = float(np.sum((labels == c) & (data["y"] == c)))
            fp = float(np.sum((labels == c) & (data["y"] != c)))
            fn = float(np.sum((labels != c) & (data["y"] == c)))
            f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return {"accuracy": float(np.mean(labels == data["y"])), "macro_f1": float(np.mean(f1))}
    err = pred - data["y"]
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}


def run(key):
    cfg = CONFIG[key]
    base = ROOT / f"task-{key.lower()}"
    model = Model.load(base / "checkpoint.npz")
    train = read_dataset(base / "public" / "train.jsonl", cfg["task"], cfg["classes"])
    val = read_dataset(base / "public" / "validation.jsonl", cfg["task"], cfg["classes"], train["paths"])
    before = metrics(model, val)
    model.meta["learning_rate"] = float(model.meta["learning_rate"]) * cfg["lr_factor"]
    best_score, best_epoch = None, 0
    best_state = None
    history = []
    started = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.fit(train["x"], train["y"], 1)
        met = metrics(model, val)
        score = met["macro_f1"] if cfg["task"] == "classification" else -met["rmse"]
        history.append({"epoch": epoch, **met})
        if best_score is None or score > best_score + 1e-9:
            best_score, best_epoch = score, epoch
            best_state = (copy.deepcopy(model.params), copy.deepcopy(model.first),
                          copy.deepcopy(model.second), copy.deepcopy(model.meta))
    model.params, model.first, model.second, model.meta = best_state
    out = ROOT / "solution" / "models_strict" / key
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "checkpoint.npz")
    after = metrics(model, val)
    report = {"format": "contest.cpu-baseline-report.v1", "profile": key, "task": cfg["task"],
              "purpose": "strict_continuation_of_provided_checkpoint",
              "parent_checkpoint_sha256": checkpoint_digest(base / "checkpoint.npz"),
              "checkpoint_sha256": checkpoint_digest(out / "checkpoint.npz"),
              "initial_epochs": 0, "final_epochs": model.meta["epochs"],
              "final_steps": model.meta["steps"], "best_epoch": best_epoch,
              "epochs_requested": cfg["epochs"], "learning_rate": model.meta["learning_rate"],
              "validation_before": before, "validation_after": after,
              "training_seconds": round(time.time() - started, 1),
              "history": history}
    report["experiment_record"] = experiment_log.log_round(
        key, {**report, "track": "strict", "members": 1},
        parent_sha256=report["parent_checkpoint_sha256"],
        artifact_sha256=report["checkpoint_sha256"],
        config={"epochs": cfg["epochs"], "lr_factor": cfg["lr_factor"], "hidden": model.meta["hidden"]},
        extra={"labels": "public/validation only", "device": "cpu"})
    (out / "training.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    with (out / "history.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row) + "\n")
    print(key, "before", before, "after", after, "best_epoch", best_epoch,
          f"{time.time() - started:.1f}s")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["A", "B", "C"]
    for key in keys:
        run(key)
