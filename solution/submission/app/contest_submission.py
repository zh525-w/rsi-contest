"""Contest submission: one round of industrial-model self-training plus prediction.

Only the documented SDK surface is used:

  train(rnd):   rnd.prompt / dataset_key / round_index / task / model / history /
                parent / state_workspace / save_state / submit_candidate / finish
  predict(sample, model_dir):  sample.task, sample.inputs, model_dir ("/model")

The round always:
  1. reads the public train/validation projection and the parent checkpoint,
  2. asks the hosted model for a keep-or-revert judgement on the evidence,
  3. continues training (see ``contest_core.train_bundle``),
  4. submits the candidate with the measured validation change,
  5. stores the artifact as the next round's native state,
  6. finishes with the selected artifact and the real model call ids.
"""
import json
import os
import shutil
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import contest_core as core

_FORMAT = "contest.industrial.submission.v1"
_BUNDLE = None


def _workspace():
    return Path(os.environ.get("CONTEST_WORKSPACE", "/workspace"))


def _projection(rnd):
    task = getattr(rnd, "task", None)
    if isinstance(task, dict):
        projection = task.get("public_projection") or {}
        return (projection.get("train_path") or projection.get("train") or "/data/train.jsonl",
                projection.get("validation_path") or projection.get("validation") or "/data/validation.jsonl")
    return "/data/train.jsonl", "/data/validation.jsonl"


def _parent_checkpoint(rnd):
    parent = getattr(rnd, "parent", None)
    if parent is None:
        return None
    try:
        path = Path(parent.path("checkpoint.npz"))
    except Exception:
        path = Path("/previous/checkpoint.npz")
    return str(path) if path.is_file() else None


def _candidate_objects(text):
    """Yield JSON objects found anywhere in a reply (prose / code fences allowed)."""
    import re

    for match in re.finditer(r"\{[^{}]*\}", text):
        raw = match.group(0)
        cleaned = (raw.replace("\u201c", '"').replace("\u201d", '"')
                      .replace("\uff1a", ":").replace("\uff0c", ",")
                      .replace("\u3002", ".").replace("\uff1b", ";"))
        try:
            value = json.loads(cleaned)
        except Exception:
            continue
        if isinstance(value, dict):
            yield value


def _pick(payloads, keys):
    for payload in payloads:
        for key in keys:
            if key in payload:
                return payload
    return None


def _regex_select(text):
    import re

    match = re.search(r'"?(?:select|decision)"?\s*[:\uff1a]\s*"?\s*(candidate|parent)', text or "", re.I)
    return match.group(1).lower() if match else None


def _regex_strategy(text, count):
    import re

    match = re.search(r'"?(?:strategy|option|choice)"?\s*[:\uff1a]\s*"?\s*(\d+)', text or "")
    if match:
        value = int(match.group(1))
        if 0 <= value < count:
            return value
    return None


def _parse_decision(text, fallback):
    """Extract {"select": ...} from a possibly chatty or truncated model reply."""
    payload = _pick(list(_candidate_objects(text or "")), ("select", "decision"))
    if payload:
        select = str(payload.get("select", payload.get("decision", ""))).strip().lower()
        if select in ("candidate", "parent"):
            return select, str(payload.get("reason", ""))[:800]
    # A chatty or truncated reply is not trusted: the validation guard decides.
    return fallback, ""


def _change_summary(key, round_index, report, parent_metrics):
    if report.get("track") == "strict":
        return (f"第{round_index}轮（严格续训）：保持父模型结构与标准化，继续训练 {report['epochs']} epoch，"
                f"父模型指标={parent_metrics}，本轮={report['validation']}。")
    parent = "父检查点" if parent_metrics else "无"
    if key == "A":
        return (f"第{round_index}轮：以对数特征+类权重训练 {report['members']} 个 64 单元 tanh MLP 集成"
                f"（父模型 {parent} 验证 Macro-F1={parent_metrics.get('macro_f1') if parent_metrics else 'NA'}，"
                f"本轮={report['validation']['macro_f1']:.4f}），训练仅用 public/train，模型选择用 public/validation。")
    if key == "B":
        return (f"第{round_index}轮：对数+平方特征上训练 {report['members']} 个 depth2 提升树成员，"
                f"目标为分钟 RUL；父 MLP 对照 RMSE={parent_metrics.get('rmse') if parent_metrics else 'NA'}，"
                f"本轮 RMSE={report['validation']['rmse']:.2f}。")
    return (f"第{round_index}轮：原始设置+传感器特征（含8个PCA分量）训练 {report['members']} 个 depth3 提升树成员，"
            f"目标为循环 RUL；父 MLP 对照 RMSE={parent_metrics.get('rmse') if parent_metrics else 'NA'}，"
            f"本轮 RMSE={report['validation']['rmse']:.2f}。")


STRATEGIES = {
    "A": [
        {"name": "log64-5seed", "desc": "有符号 log1p 特征 + 64 单元 MLP × 5 种子（默认，跨批次更稳）",
         "override": {}},
        {"name": "raw64-3seed", "desc": "原始特征 + 64 单元 MLP × 3 种子（保留父模型量纲，方差更大）",
         "override": {"feature": {"mode": "raw"}, "seeds": [0, 1, 2]}},
    ],
    "B": [
        {"name": "log2-depth2-5seed", "desc": "log + 平方特征、depth2 提升树 × 5 种子（默认）",
         "override": {}},
        {"name": "log-depth2-7seed", "desc": "仅 log 特征、depth2 × 7 种子、120 棵（更保守的树数）",
         "override": {"feature": {"mode": "log"}, "seeds": [0, 1, 2, 3, 4, 5, 6],
                      "gbm": {"n_estimators": 120}}},
    ],
    "C": [
        {"name": "pca8-depth3-2seed", "desc": "原始特征 + 8 个 PCA 分量、depth3 × 2 种子（默认）",
         "override": {}},
        {"name": "raw-depth3-2seed350", "desc": "不使用 PCA、depth3 × 2 种子、350 棵",
         "override": {"pca": 0, "gbm": {"n_estimators": 350}}},
    ],
}


def _pick_strategy(rnd, key, parent_metrics, history):
    """Ask the hosted model to choose one code-level strategy; fall back to 0."""
    options = STRATEGIES.get(key, [])
    if not options:
        return 0, "", None
    listing = "\n".join(f"[{i}] {item['name']}：{item['desc']}" for i, item in enumerate(options))
    prompt = (
        "你是工业模型自进化系统的策略代理。只输出一个 JSON 对象，不要解释。\n"
        f"数据集={key} 轮次={rnd.round_index}\n"
        f"父模型验证指标={json.dumps(parent_metrics, ensure_ascii=False) if parent_metrics else '无'}\n"
        f"历史记录={json.dumps(history[-3:], ensure_ascii=False) if history else '空'}\n"
        f"可选训练策略：\n{listing}\n"
        "只输出一个 JSON 对象：键为 strategy，值为 0 或 1。不要复述本段，不要分析。"
    )
    receipt = rnd.model([
        {"role": "system", "content": "/no_think 你只输出一个 JSON 对象，不要分析、不要 Markdown、不要多余文字。"},
        {"role": "user", "content": prompt},
    ], max_tokens=min(384, int(getattr(rnd, "max_output_tokens", 384))), temperature=0,
        enable_thinking=False)
    payload = _pick(list(_candidate_objects(receipt.text or "")), ("strategy", "option", "choice"))
    if payload:
        try:
            index = int(payload.get("strategy", payload.get("option", payload.get("choice", 0))))
            if 0 <= index < len(options):
                return index, str(payload.get("reason", ""))[:500], receipt
        except Exception:
            pass
    # No complete JSON object -> keep the validated default strategy.
    return 0, "", receipt


def train(rnd):
    key = rnd.dataset_key
    train_path, val_path = _projection(rnd)
    parent_path = _parent_checkpoint(rnd)
    round_index = int(rnd.round_index)
    history = rnd.history()
    work = _workspace() / f"round-{round_index}"
    if work.exists():
        for stale in work.glob("*"):
            stale.unlink() if stale.is_file() else None
    work.mkdir(parents=True, exist_ok=True)

    track = os.environ.get("CONTEST_TRACK", "boost").strip().lower()
    parent_metrics = {}
    if parent_path:
        try:
            parent_metrics = core.Bundle.load(parent_path).meta.get("validation") or {}
        except Exception:
            parent_metrics = {}
    strategy_index, strategy_reason, _ = (0, "", None)
    if track != "strict":
        strategy_index, strategy_reason, _ = _pick_strategy(rnd, key, parent_metrics, history)
    strategy = STRATEGIES.get(key, [{"name": "default", "override": {}}])[strategy_index]
    if track == "strict" and parent_path:
        bundle, report = core.train_strict_bundle(key, train_path, val_path, parent_path,
                                                 round_index=round_index)
    else:
        bundle, report = core.train_bundle(key, train_path, val_path, parent_path,
                                          round_index=round_index,
                                          recipe_override=strategy.get("override"),
                                          log=lambda message: print(message, file=sys.stderr, flush=True))
    report["strategy"] = strategy["name"]
    report["strategy_reason"] = strategy_reason
    checkpoint = work / "checkpoint.npz"
    bundle.save(checkpoint)
    (work / "training.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    if os.environ.get("CONTEST_CLOUD_LOG") == "1" and key in ("A", "B", "C"):
        try:
            import experiment_log
            report["experiment_record"] = experiment_log.log_round(
                key, report, artifact_sha256=core._sha256(checkpoint),
                config={"members": report["members"], "feature": report["feature"],
                        "pca": report["pca"], "round_index": round_index})
        except Exception as exc:  # offline judge image: recording must stay optional
            print(f"cloud log skipped: {exc}", file=sys.stderr, flush=True)
    (work / "history.jsonl").write_text(
        json.dumps({"round": round_index, "validation": report["validation"],
                    "members": report["members"], "seconds": report["training_seconds"]}) + "\n",
        encoding="utf-8")

    submitted = rnd.submit_candidate(
        work,
        hypothesis=("继续训练可提升该数据集主指标；本轮在父检查点基础上扩大集成/追加残差树，"
                    "并以 public/validation 与父模型对照决定保留哪个产物。"),
        change_summary=_change_summary(key, round_index, report,
                                       bundle.meta.get("parent_validation")),
        configuration={"members": report["members"], "feature": report["feature"],
                       "pca": report["pca"], "round_index": round_index,
                       "strategy": report.get("strategy")},
        seed=int(report.get("seed", 0)),
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=["python3", "/app/harness.py", "predict"],
    )

    # Hosted-model judgement: keep the new candidate or revert to the parent.
    parent_metrics = bundle.meta.get("parent_validation") or parent_metrics
    prompt = (
        "你是工业模型自进化系统的决策代理。只输出一个 JSON 对象，不要解释。\n"
        f"数据集={key} 任务={report['task']} 轮次={round_index}\n"
        f"父模型验证指标={json.dumps(parent_metrics, ensure_ascii=False) if parent_metrics else '无'}\n"
        f"新候选验证指标={json.dumps(report['validation'], ensure_ascii=False)}\n"
        f"候选说明={_change_summary(key, round_index, report, parent_metrics)}\n"
        f"历史={json.dumps(history[-3:], ensure_ascii=False) if history else '空'}\n"
        "主指标：A 为 macro-F1 越高越好；B/C 为 RMSE 越低越好。"
        "只输出一个 JSON 对象：键为 select，值为 candidate 或 parent。不要复述本段，不要分析。"
    )
    receipt = rnd.model([
        {"role": "system", "content": "/no_think 你只输出一个 JSON 对象，不要分析、不要 Markdown、不要多余文字。"},
        {"role": "user", "content": prompt},
    ], max_tokens=min(256, int(getattr(rnd, "max_output_tokens", 256))), temperature=0,
        enable_thinking=False)
    select, reason = _parse_decision(receipt.text, "candidate")
    # metric guard: never replace the parent with a candidate that measured worse
    parent_score = None
    if parent_path:
        try:
            previous = core.Bundle.load(parent_path)
            previous_metrics = previous.meta.get("validation") or {}
            parent_score = (previous_metrics.get("macro_f1") if report["task"] == "classification"
                            else previous_metrics.get("rmse"))
        except Exception:
            parent_score = None
    candidate_score = (report["validation"].get("macro_f1") if report["task"] == "classification"
                       else report["validation"].get("rmse"))
    if parent_score is None and parent_metrics:
        parent_score = parent_metrics.get("macro_f1" if report["task"] == "classification" else "rmse")
    if parent_score is not None:
        improves = (candidate_score > parent_score + 1e-6 if report["task"] == "classification"
                    else candidate_score < parent_score - 1e-6)
        if not improves:
            select = "parent"
            reason = reason or f"候选 {candidate_score:.4f} 未超过父模型 {parent_score:.4f}，保留父模型。"

    selected = submitted["artifact_id"] if select == "candidate" else rnd.parent.artifact_id
    state = _workspace() / f"state-{round_index}"
    if state.exists():
        shutil.rmtree(state)
    state.mkdir(parents=True, exist_ok=True)
    source = checkpoint if select == submitted["artifact_id"] else Path(parent_path or "")
    if source and source.is_file():
        shutil.copy(source, state / "checkpoint.npz")
    rnd.save_state(state)

    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=("在父检查点基础上继续训练并扩大集成容量，可提升该数据集的主指标且不破坏可复现性。"),
        change_summary=_change_summary(key, round_index, report, parent_metrics),
        reason=(reason or f"候选验证指标 {report['validation']} 相对父模型 "
                f"{parent_metrics or '基线'} 由平台测量给出，按主指标方向选择。"),
        configuration={"members": report["members"], "feature": report["feature"],
                       "pca": report["pca"], "round_index": round_index,
                       "strategy": report.get("strategy")},
        seed=int(report.get("seed", 0)),
        history_ids=[item.get("id") for item in history if isinstance(item, dict)],
        model_call_ids=list(rnd.call_ids),
    )


def predict(sample, model_dir):
    global _BUNDLE
    if _BUNDLE is None:
        path = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
        _BUNDLE = core.Bundle.load(path)
    if getattr(sample, "task", None) not in (None, _BUNDLE.meta["task"]):
        raise ValueError("sample task does not match the artifact task")
    return _BUNDLE.predict_inputs(sample.inputs)
