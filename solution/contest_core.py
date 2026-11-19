"""contest_core: self-contained NumPy model stack for the industrial-model contest.

Contents
  * JSONL dataset loading with the frozen feature-schema rules,
  * feature pipelines (signed log1p, squares, PCA),
  * a tanh/relu MLP with Adam + EMA (task A, and the parent models),
  * a small squared-loss gradient-boosted tree ensemble (tasks B and C),
  * one artifact format, ``checkpoint.npz`` ("contest.industrial.v2"), holding
    preprocessing state, every ensemble member and the metadata needed for
    reproducible prediction, plus parent-model continuation.

Only NumPy is required.  The module is shared by ``contest_submission`` inside
the contest image and by the offline training scripts in this repository.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

FORMAT = "contest.industrial.v2"
CLASSES_A = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]

# ------------------------------------------------------------------ data IO


def _flatten(value, out):
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(value[key], out)
    elif isinstance(value, list):
        for child in value:
            _flatten(child, out)
    else:
        out.append(float(value))


def flatten_inputs(inputs):
    out = []
    _flatten(inputs, out)
    return out


def read_jsonl(path, task, classes=None):
    ids, feats, targets, metas = [], [], [], []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.append(obj["id"])
            feats.append(flatten_inputs(obj["inputs"]))
            metas.append(obj.get("meta", {}))
            answer = obj["answer"]
            targets.append(classes.index(answer) if task == "classification" else float(answer))
    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64 if task == "classification" else np.float64)
    return {"ids": ids, "x": x, "y": y, "meta": metas}


# ------------------------------------------------------------------ metrics


def classification_metrics(prob, y):
    pred = np.asarray(prob).argmax(1)
    y = np.asarray(y)
    f1 = []
    for c in range(np.asarray(prob).shape[1]):
        tp = float(np.sum((pred == c) & (y == c)))
        fp = float(np.sum((pred == c) & (y != c)))
        fn = float(np.sum((pred != c) & (y == c)))
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {"accuracy": float(np.mean(pred == y)), "macro_f1": float(np.mean(f1)),
            "per_class_f1": [float(v) for v in f1]}


def regression_metrics(pred, y):
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def nasa_score(pred, y):
    d = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


# ----------------------------------------------------------- preprocessing


class Scaler:
    def __init__(self, mode="std"):
        self.mode = mode
        self.mean = None
        self.scale = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "robust":
            self.mean = np.median(x, axis=0)
            q75, q25 = np.percentile(x, [75, 25], axis=0)
            self.scale = (q75 - q25) / 1.349
        else:
            self.mean = x.mean(axis=0)
            self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        return self

    def transform(self, x):
        return np.clip((np.asarray(x, dtype=np.float64) - self.mean) / self.scale, -10, 10)


def signed_log(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def build_features(x, cfg):
    z = np.asarray(x, dtype=np.float64)
    if cfg.get("mode") == "log":
        z = signed_log(z)
    out = [z]
    if cfg.get("squares"):
        out.append(z ** 2)
    return np.hstack(out)


# ---------------------------------------------------------------------- MLP


def _act(z, kind):
    if kind == "relu":
        return np.maximum(z, 0)
    if kind == "gelu":
        return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
    return np.tanh(z)


def _act_grad(z, a, kind):
    if kind == "relu":
        return (z > 0).astype(np.float64)
    if kind == "gelu":
        t = np.tanh(0.7978845608 * (z + 0.044715 * z ** 3))
        return 0.5 * (1 + t) + 0.5 * z * (1 - t ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * z ** 2)
    return 1 - a ** 2


class MLP:
    def __init__(self, n_in, hidden, n_out, task, *, activation="tanh", seed=0, weights=None):
        self.task = task
        self.hidden = list(hidden)
        self.activation = activation
        rng = np.random.default_rng(seed)
        sizes = [n_in] + self.hidden + [n_out]
        self.params = {}
        for i in range(len(sizes) - 1):
            scale = math.sqrt(2.0 / sizes[i]) if activation == "relu" else math.sqrt(1.0 / sizes[i])
            self.params[f"W{i}"] = rng.normal(0, scale, (sizes[i], sizes[i + 1]))
            self.params[f"b{i}"] = np.zeros(sizes[i + 1])
        if weights:
            for k, v in weights.items():
                v = np.asarray(v, dtype=np.float64)
                if k in self.params and self.params[k].shape == v.shape:
                    self.params[k] = v.copy()
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.ema = {k: v.copy() for k, v in self.params.items()}
        self.step = 0

    def forward(self, x, params=None):
        p = self.params if params is None else params
        cache = {}
        h = np.asarray(x, dtype=np.float64)
        n_layers = len(self.hidden) + 1
        for i in range(n_layers):
            z = h @ p[f"W{i}"] + p[f"b{i}"]
            cache[f"in{i}"] = h
            if i < n_layers - 1:
                h = _act(z, self.activation)
                cache[f"a{i}"] = h
            else:
                out = z
            cache[f"z{i}"] = z
        return out, cache

    def predict_proba(self, x, params=None):
        out, _ = self.forward(x, params)
        if self.task == "regression":
            return out[:, 0]
        out = out - out.max(axis=1, keepdims=True)
        e = np.exp(out)
        return e / e.sum(axis=1, keepdims=True)


def train_mlp(x_tr, y_tr, x_va, y_va, task, cfg, *, seed=0, log=None):
    """Train one MLP with Adam + optional EMA; returns (model, best_params, history)."""
    n_out = cfg.get("n_classes", 1) if task == "classification" else 1
    model = MLP(x_tr.shape[1], cfg["hidden"], n_out, task,
                activation=cfg.get("activation", "tanh"), seed=seed, weights=cfg.get("weights"))
    if cfg.get("weights"):
        model.ema = {k: v.copy() for k, v in model.params.items()}
    batch = cfg.get("batch_size", 128)
    epochs = cfg["epochs"]
    lr0 = cfg["lr"]
    wd = cfg.get("weight_decay", 0.0)
    noise = cfg.get("input_noise", 0.0)
    ema_decay = cfg.get("ema_decay", 0.0)
    patience = cfg.get("patience", 0)
    rng = np.random.default_rng(seed + 9973)
    n = len(x_tr)
    steps_per_epoch = max(1, math.ceil(n / batch))
    total_steps = max(1, epochs * steps_per_epoch)
    warm = int(cfg.get("warmup_frac", 0.0) * total_steps)
    class_weight = None
    if task == "classification" and cfg.get("class_weight", 0.0) > 0:
        alpha = cfg["class_weight"]
        counts = np.bincount(y_tr, minlength=n_out).astype(np.float64)
        class_weight = (n / (n_out * np.maximum(counts, 1))) ** alpha
    best = {"score": -np.inf, "epoch": 0,
            "params": {k: v.copy() for k, v in (model.ema if ema_decay else model.params).items()}}
    history, bad = [], 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb, yb = x_tr[idx], y_tr[idx]
            if noise > 0:
                xb = xb + rng.normal(0, noise, xb.shape)
            model.step += 1
            t = min(1.0, model.step / total_steps)
            lr = lr0 * (0.5 * (1 + math.cos(math.pi * t)) if cfg.get("cosine", True) else 1.0)
            if warm and model.step < warm:
                lr = lr0 * (0.1 + 0.9 * model.step / warm)
            out, cache = model.forward(xb)
            n_layers = len(model.hidden) + 1
            grads = {}
            if task == "classification":
                logits = out - out.max(axis=1, keepdims=True)
                e = np.exp(logits)
                p = e / e.sum(axis=1, keepdims=True)
                target = np.zeros_like(p)
                target[np.arange(len(yb)), yb] = 1.0
                w = class_weight[yb] if class_weight is not None else np.ones(len(yb))
                delta = (w[:, None] * (p - target)) / w.sum()
            else:
                error = out[:, 0] - yb
                delta = (2 * error / len(yb))[:, None]
            grads[f"W{n_layers-1}"] = cache[f"a{n_layers-2}"].T @ delta
            grads[f"b{n_layers-1}"] = delta.sum(axis=0)
            d = delta
            for i in range(n_layers - 1, 0, -1):
                z_prev = cache[f"z{i-1}"]
                g = (d @ model.params[f"W{i}"].T) * _act_grad(z_prev, _act(z_prev, model.activation),
                                                             model.activation)
                grads[f"W{i-1}"] = cache[f"in{i-1}"].T @ g
                grads[f"b{i-1}"] = g.sum(axis=0)
                d = g
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            scale = min(1.0, cfg.get("clip", 5.0) / max(norm, 1e-12))
            for name, g in grads.items():
                g = g * scale
                model.m[name] = 0.9 * model.m[name] + 0.1 * g
                model.v[name] = 0.999 * model.v[name] + 0.001 * g * g
                mhat = model.m[name] / (1 - 0.9 ** model.step)
                vhat = model.v[name] / (1 - 0.999 ** model.step)
                model.params[name] -= lr * (mhat / (np.sqrt(vhat) + 1e-8) + wd * model.params[name])
            if ema_decay:
                for name in model.params:
                    model.ema[name] = ema_decay * model.ema[name] + (1 - ema_decay) * model.params[name]
        if epoch % cfg.get("eval_every", 1) == 0 or epoch == epochs:
            params = model.ema if ema_decay else model.params
            if task == "classification":
                met = classification_metrics(model.predict_proba(x_va, params), y_va)
                score = met["macro_f1"]
            else:
                met = regression_metrics(model.predict_proba(x_va, params), y_va)
                score = -met["rmse"]
            history.append({"epoch": epoch, "score": float(score), **met})
            if log:
                log(f"epoch {epoch:3d} score={score:.5f} {met}")
            if score > best["score"] + 1e-6:
                best = {"score": score, "epoch": epoch,
                        "params": {k: v.copy() for k, v in params.items()}}
                bad = 0
            else:
                bad += 1
                if patience and bad >= patience:
                    break
    return model, best, history


# ------------------------------------------------------------ trees / GBM


def _best_split(x, g, idx, min_leaf):
    n = len(idx)
    total = g[idx].sum()
    base = total * total / n
    best = (-1, 0.0, 0.0)
    gi = g[idx]
    for f in range(x.shape[1]):
        col = x[idx, f]
        order = np.argsort(col, kind="stable")
        vals = col[order]
        cs = np.cumsum(gi[order])
        p = np.arange(min_leaf - 1, n - min_leaf)
        if len(p) == 0:
            continue
        p = p[vals[p] < vals[p + 1]]
        if len(p) == 0:
            continue
        nl = p + 1
        gl = cs[p]
        gr = total - gl
        gain = gl * gl / nl + gr * gr / (n - nl) - base
        j = int(np.argmax(gain))
        if gain[j] > best[2]:
            best = (f, float(vals[p[j]]), float(gain[j]))
    return best


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value=0.0):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = value


def _build(x, g, idx, depth, max_depth, min_leaf, min_gain=1e-6):
    node = _Node(float(g[idx].mean()) if len(idx) else 0.0)
    if depth >= max_depth or len(idx) < 2 * min_leaf:
        return node
    f, t, gain = _best_split(x, g, idx, min_leaf)
    if f < 0 or gain <= min_gain:
        return node
    mask = x[idx, f] <= t
    node.feature, node.threshold = f, t
    node.left = _build(x, g, idx[mask], depth + 1, max_depth, min_leaf, min_gain)
    node.right = _build(x, g, idx[~mask], depth + 1, max_depth, min_leaf, min_gain)
    return node


def _flatten_tree(node):
    """Preorder traversal to parallel node arrays (feature=-1 marks a leaf)."""
    feature, threshold, left, right, value = [], [], [], [], []

    def visit(n):
        me = len(feature)
        feature.append(n.feature)
        threshold.append(n.threshold)
        left.append(-1)
        right.append(-1)
        value.append(n.value)
        if n.feature >= 0:
            left[me] = visit(n.left)
            right[me] = visit(n.right)
        return me

    visit(node)
    return (np.asarray(feature, dtype=np.int32), np.asarray(threshold, dtype=np.float64),
            np.asarray(left, dtype=np.int32), np.asarray(right, dtype=np.int32),
            np.asarray(value, dtype=np.float64))


def pack_trees(trees):
    """Concatenate local trees into member arrays; child indices become global."""
    fields = {"feature": [], "threshold": [], "left": [], "right": [], "value": []}
    offsets = [0]
    base = 0
    for feature, threshold, left, right, value in trees:
        fields["feature"].append(feature)
        fields["threshold"].append(threshold)
        fields["left"].append(np.where(left >= 0, left + base, -1))
        fields["right"].append(np.where(right >= 0, right + base, -1))
        fields["value"].append(value)
        base += len(feature)
        offsets.append(base)
    packed = {k: np.concatenate(v) for k, v in fields.items()}
    return packed, np.asarray(offsets, dtype=np.int64)


def predict_packed_tree(packed, root, x):
    feature, threshold = packed["feature"], packed["threshold"]
    left, right, value = packed["left"], packed["right"], packed["value"]
    out = np.empty(len(x))
    stack = [(root, np.arange(len(x)))]
    while stack:
        node, idx = stack.pop()
        if len(idx) == 0:
            continue
        if feature[node] < 0:
            out[idx] = value[node]
            continue
        mask = x[idx, feature[node]] <= threshold[node]
        stack.append((left[node], idx[mask]))
        stack.append((right[node], idx[~mask]))
    return out


class GBM:
    """Squared-loss gradient boosting with depth-limited trees (packed output)."""

    def __init__(self, n_estimators=150, lr=0.03, max_depth=2, min_leaf=40,
                 subsample=0.8, seed=0):
        self.n_estimators = n_estimators
        self.lr = lr
        self.depth = max_depth
        self.min_leaf = min_leaf
        self.subsample = subsample
        self.seed = seed
        self.base = 0.0
        self.trees = []

    def fit(self, x, y, base=0.0, init_pred=None):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        self.base = float(base)
        pred = np.full(len(y), self.base) if init_pred is None else np.array(init_pred, dtype=np.float64)
        for _ in range(self.n_estimators):
            g = pred - y
            idx = rng.choice(len(y), max(int(self.subsample * len(y)), 2 * self.min_leaf), replace=False)
            tree = _flatten_tree(_build(x, g, idx, 0, self.depth, self.min_leaf))
            self.trees.append(tree)
            pred -= self.lr * predict_packed_tree(
                {"feature": tree[0], "threshold": tree[1], "left": tree[2], "right": tree[3],
                 "value": tree[4]}, 0, x)
        return self


# ------------------------------------------------------------------- bundle


def _sha256(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_baseline(path):
    """Load a provided ``contest.cpu-mlp.v1`` checkpoint as the parent model."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        params = {k: data[k].copy() for k in ("w1", "b1", "w2", "b2")}
        mean, scale = data["mean"].copy(), data["scale"].copy()
    return {"kind": "baseline", "meta": meta, "params": params, "mean": mean, "scale": scale,
            "sha256": _sha256(path)}


class Bundle:
    def __init__(self, meta, arrays):
        self.meta = meta
        self.arrays = arrays

    # ---- io
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        meta = json.loads(str(arrays.pop("meta")))
        if meta.get("format") != FORMAT:
            raise ValueError("unsupported artifact format")
        return cls(meta, arrays)

    def save(self, path):
        arrays = dict(self.arrays)
        arrays["meta"] = np.array(json.dumps(self.meta, ensure_ascii=False, allow_nan=False))
        np.savez_compressed(path, **arrays)
        return path

    # ---- feature pipeline
    def features(self, x_raw):
        z = build_features(x_raw, self.meta.get("feature", {}))
        z = np.clip((z - np.asarray(self.meta["feat_mean"])) / np.asarray(self.meta["feat_scale"]),
                    -10, 10)
        if self.meta.get("pca"):
            z = np.hstack([z, z @ self.arrays["pca_basis"]])
        return z

    # ---- members
    def _mlp_member(self, member, prefix, x):
        n_layers = len(member["hidden"]) + 1
        weights = {f"W{j}": self.arrays[f"{prefix}_W{j}"] for j in range(n_layers)}
        weights.update({f"b{j}": self.arrays[f"{prefix}_b{j}"] for j in range(n_layers)})
        model = MLP(x.shape[1], member["hidden"],
                    len(self.meta["classes"]) if self.meta["task"] == "classification" else 1,
                    self.meta["task"], activation=member.get("activation", "tanh"), weights=weights)
        return model.predict_proba(x)

    def _gbm_member(self, member, prefix, x_raw, z):
        pred = np.full(len(z), float(member["base"]))
        if member.get("init"):
            xr = np.clip((x_raw - self.arrays[f"{prefix}_init_mean"]) /
                         self.arrays[f"{prefix}_init_scale"], -10, 10)
            init_out = self._mlp_member({"hidden": member["init"]["hidden"],
                                         "activation": member["init"].get("activation", "tanh")},
                                        f"{prefix}_init", xr)
            pred = pred + init_out * float(member["init"].get("target_scale", 1.0)) \
                + float(member["init"].get("target_mean", 0.0))
        packed = {k: self.arrays[f"{prefix}_{k}"] for k in
                  ("feature", "threshold", "left", "right", "value")}
        offsets = self.arrays[f"{prefix}_offset"]
        for t in range(len(offsets) - 1):
            pred = pred - float(member["lr"]) * predict_packed_tree(packed, int(offsets[t]), z)
        return pred

    def _member_pred(self, member, index, x_raw, z):
        if member["kind"] == "mlp":
            return self._mlp_member(member, f"m{index}", z)
        if member["kind"] == "gbm":
            return self._gbm_member(member, f"g{index}", x_raw, z)
        if member["kind"] == "parent_mlp":
            xr = np.clip((x_raw - self.arrays["pm_mean"]) / self.arrays["pm_scale"], -10, 10)
            return self._mlp_member({"hidden": [self.meta["parent"]["hidden"]],
                                     "activation": self.meta["parent"].get("activation", "tanh")},
                                    "pm", xr)
        raise ValueError(f"unknown member kind {member['kind']}")

    # ---- prediction
    def predict(self, x_raw):
        x_raw = np.atleast_2d(np.asarray(x_raw, dtype=np.float64))
        z = self.features(x_raw)
        preds = [self._member_pred(m, i, x_raw, z) for i, m in enumerate(self.meta["members"])]
        out = np.mean(preds, axis=0)
        if self.meta["task"] == "regression":
            out = out * float(self.meta.get("target_scale", 1.0)) + float(self.meta.get("target_mean", 0.0))
            return np.maximum(out, 0.0)
        return out

    def predict_inputs(self, inputs):
        x = np.asarray([flatten_inputs(inputs)], dtype=np.float64)
        if self.meta["task"] == "classification":
            prob = self.predict(x)[0]
            labels = self.meta["classes"]
            order = np.argsort(-prob, kind="stable")
            return [{"label": labels[k], "score": float(prob[k])} for k in order]
        return {"value": float(self.predict(x)[0])}


# -------------------------------------------------------------- train recipes
#
# Each recipe is the *same* code path used offline and inside the harness:
#   feature pipeline -> preprocessing state -> ensemble -> validation report.
# Rounds extend the parent: MLP members continue their weights, GBM members
# append trees on the current residual.

RECIPES = {
    "A": {
        "task": "classification",
        "classes": CLASSES_A,
        "feature": {"mode": "log"},
        "pca": 0,
        "mlp": {"hidden": [64], "activation": "tanh", "batch_size": 64, "lr": 1e-3,
                "weight_decay": 1e-4, "ema_decay": 0.995, "class_weight": 1.0,
                "warmup_frac": 0.05, "eval_every": 1, "epochs": 300},
        "contract_epochs": 300,
        "seeds": [0, 1, 2, 3, 4],
    },
    "B": {
        "task": "regression",
        "classes": [],
        "feature": {"mode": "log", "squares": True},
        "pca": 0,
        "gbm": {"n_estimators": 150, "lr": 0.03, "max_depth": 2, "min_leaf": 40,
                "subsample": 0.8},
        "contract_trees": 60,
        "seeds": [0, 1, 2, 3, 4],
    },
    "C": {
        "task": "regression",
        "classes": [],
        "feature": {"mode": "raw"},
        "pca": 8,
        "gbm": {"n_estimators": 300, "lr": 0.05, "max_depth": 3, "min_leaf": 40,
                "subsample": 0.4},
        "contract_trees": 100,
        "seeds": [0, 1],
    },
}

_TREE_FIELDS = ("feature", "threshold", "left", "right", "value")


def _package_mlp(arrays, prefix, params, hidden, activation, seed):
    for j in range(len(hidden) + 1):
        arrays[f"{prefix}_W{j}"] = np.asarray(params[f"W{j}"], dtype=np.float64)
        arrays[f"{prefix}_b{j}"] = np.asarray(params[f"b{j}"], dtype=np.float64)


def _migrate_baseline(parent, hidden):
    """Widen the parent MLP to ``hidden`` units while preserving its function."""
    w1, b1, w2, b2 = (parent["params"][k] for k in ("w1", "b1", "w2", "b2"))
    n_in, old = w1.shape
    width = hidden[0]
    if width < old:
        raise ValueError("cannot shrink parent width")
    rng = np.random.default_rng(7)
    W0 = rng.normal(0, math.sqrt(1.0 / n_in), (n_in, width))
    W0[:, :old] = w1
    b0 = np.zeros(width)
    b0[:old] = b1
    W1 = np.zeros((width, w2.shape[1]))
    W1[:old] = w2
    params = {"W0": W0, "b0": b0, "W1": W1, "b1": b2}
    return params


def _append_trees(arrays, prefix, member, gbm, round_index):
    """Append newly fitted trees (trained on the current residual) to a member."""
    old = {k: arrays[f"{prefix}_{k}"] for k in _TREE_FIELDS}
    old_offset = arrays[f"{prefix}_offset"]
    new_packed, new_offset = pack_trees(gbm.trees)
    shift = len(old["feature"])
    merged = {}
    for k in _TREE_FIELDS:
        extra = new_packed[k]
        if k in ("left", "right"):
            extra = np.where(extra >= 0, extra + shift, -1)
        merged[k] = np.concatenate([old[k], extra])
    arrays.update({f"{prefix}_{k}": merged[k] for k in _TREE_FIELDS})
    arrays[f"{prefix}_offset"] = np.concatenate([old_offset, new_offset[1:] + shift])
    member["n_trees"] = int(len(arrays[f"{prefix}_offset"]) - 1)


def _fit_pca(ztr, n_components):
    cov = np.cov(ztr, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(-values)[:n_components]
    return vectors[:, order], float(np.sum(values[order]) / np.sum(values))


def _merge_recipe(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_recipe(out[key], value)
        else:
            out[key] = value
    return out


def train_bundle(dataset_key, train_path, val_path, parent_path=None, *, round_index=1,
                 log=print, extra_epochs=None, recipe_override=None):
    """Train (or continue) one artifact for ``dataset_key``.

    Returns ``(bundle, report)``.  The report is recorded in the round metadata
    and written to ``training.json`` by the offline scripts.
    """
    if dataset_key not in RECIPES:
        raise ValueError(f"unknown dataset key {dataset_key}")
    recipe = _merge_recipe(RECIPES[dataset_key], recipe_override)
    task = recipe["task"]
    classes = recipe["classes"]
    train = read_jsonl(train_path, task, classes)
    val = read_jsonl(val_path, task, classes)
    feat_cfg = dict(recipe["feature"])
    raw_tr, raw_va = train["x"], val["x"]

    # ---- feature pipeline + preprocessing state (fitted on train only)
    feat_tr = build_features(raw_tr, feat_cfg)
    feat_va = build_features(raw_va, feat_cfg)
    scaler = Scaler("std").fit(feat_tr)
    ztr, zva = scaler.transform(feat_tr), scaler.transform(feat_va)
    arrays = {}
    meta = {"format": FORMAT, "dataset_key": dataset_key, "task": task, "classes": list(classes),
            "feature": feat_cfg, "pca": int(recipe.get("pca", 0)),
            "feat_mean": [float(v) for v in scaler.mean],
            "feat_scale": [float(v) for v in scaler.scale],
            "round_index": int(round_index), "members": []}
    if recipe.get("pca"):
        basis, evr = _fit_pca(ztr, int(recipe["pca"]))
        arrays["pca_basis"] = basis
        ztr = np.hstack([ztr, ztr @ basis])
        zva = np.hstack([zva, zva @ basis])
        meta["pca_evr"] = evr

    # ---- parent checkpoint (baseline NPZ or our own bundle)
    parent, parent_kind = None, None
    if parent_path:
        try:
            with np.load(parent_path, allow_pickle=False) as data:
                is_bundle = "meta" in data.files and str(data["meta"]).find(FORMAT) >= 0
        except Exception:
            is_bundle = False
        if is_bundle:
            parent = Bundle.load(parent_path)
            parent_kind = "bundle"
            if parent.meta.get("dataset_key") != dataset_key:
                raise ValueError("parent artifact belongs to a different dataset")
        else:
            parent = load_baseline(parent_path)
            parent_kind = "baseline"
            if parent["meta"].get("task") != task:
                raise ValueError("parent task does not match this dataset")
            meta["parent"] = {"hidden": int(parent["meta"]["hidden"]), "activation": "tanh",
                              "sha256": parent["sha256"]}

    started = time.time()
    members = meta["members"]
    if task == "classification":
        api = recipe["mlp"]
        warm = []
        if parent_kind == "bundle":
            for i, member in enumerate(parent.meta.get("members", [])):
                if member["kind"] != "mlp":
                    continue
                nl = len(member["hidden"]) + 1
                warm.append({**{f"W{j}": parent.arrays[f"m{i}_W{j}"].copy() for j in range(nl)},
                             **{f"b{j}": parent.arrays[f"m{i}_b{j}"].copy() for j in range(nl)}})
                break
        elif parent_kind == "baseline":
            warm.append(_migrate_baseline(parent, api["hidden"]))
        epochs = api["epochs"] if round_index <= 1 else recipe["contract_epochs"]
        if extra_epochs:
            epochs = int(extra_epochs)
        for n, seed in enumerate(recipe["seeds"]):
            cfg = dict(api)
            cfg["epochs"] = epochs
            cfg["n_classes"] = len(classes)
            cfg["weights"] = warm[n] if n < len(warm) else None
            _, best, _ = train_mlp(ztr, train["y"], zva, val["y"], task, cfg,
                                   seed=seed + 100 * (round_index - 1), log=None)
            _package_mlp(arrays, f"m{len(members)}", best["params"], api["hidden"],
                         api["activation"], seed)
            members.append({"kind": "mlp", "hidden": list(api["hidden"]),
                            "activation": api["activation"],
                            "seed": int(seed + 100 * (round_index - 1)),
                            "epochs": int(epochs), "best_epoch": int(best["epoch"])})
        # parent MLP can join the ensemble on its own (raw-feature) representation;
        # keep it only when the public validation set says it helps.
        parent_member = False
        if parent_kind == "baseline":
            arrays["pm_mean"] = parent["mean"].copy()
            arrays["pm_scale"] = parent["scale"].copy()
            arrays["pm_W0"] = parent["params"]["w1"].copy()
            arrays["pm_b0"] = parent["params"]["b1"].copy()
            arrays["pm_W1"] = parent["params"]["w2"].copy()
            arrays["pm_b1"] = parent["params"]["b2"].copy()
            probe = Bundle({**meta, "members": members + [{"kind": "parent_mlp"}]}, dict(arrays))
            meta["parent_validation"] = classification_metrics(
                Bundle({**meta, "members": [{"kind": "parent_mlp"}]}, dict(arrays)).predict(raw_va),
                val["y"])
            with_parent = classification_metrics(probe.predict(raw_va), val["y"])
            without = classification_metrics(Bundle(meta, dict(arrays)).predict(raw_va), val["y"])
            parent_member = with_parent["macro_f1"] > without["macro_f1"] + 1e-6
            if parent_member:
                members.append({"kind": "parent_mlp"})
            else:
                for k in [k for k in arrays if k.startswith("pm_")]:
                    arrays.pop(k)
            log(f"A parent member included={parent_member} "
                f"with={with_parent['macro_f1']:.4f} without={without['macro_f1']:.4f}")
    else:
        base_cfg = recipe["gbm"]
        # continue the parent bundle: append trees fitted on each member's residual
        if parent_kind == "bundle":
            for i, member in enumerate(parent.meta.get("members", [])):
                if member["kind"] != "gbm":
                    continue
                prefix = f"g{len(members)}"
                for k in _TREE_FIELDS:
                    arrays[f"{prefix}_{k}"] = parent.arrays[f"g{i}_{k}"].copy()
                arrays[f"{prefix}_offset"] = parent.arrays[f"g{i}_offset"].copy()
                pred = parent._member_pred(member, i, raw_tr, parent.features(raw_tr))
                gbm = GBM(n_estimators=recipe["contract_trees"], lr=base_cfg["lr"],
                          max_depth=base_cfg["max_depth"], min_leaf=base_cfg["min_leaf"],
                          subsample=base_cfg["subsample"],
                          seed=int(member.get("seed", 0)) + round_index)
                gbm.fit(ztr, train["y"] - pred, base=0.0)
                new_member = {**member, "round_index": int(round_index)}
                _append_trees(arrays, prefix, new_member, gbm, round_index)
                members.append(new_member)
        if not members:
            for n, seed in enumerate(recipe["seeds"]):
                cfg = dict(base_cfg)
                cfg["seed"] = seed + 100 * (round_index - 1)
                gbm = GBM(**cfg)
                gbm.fit(ztr, train["y"], base=float(train["y"].mean()))
                packed, offset = pack_trees(gbm.trees)
                prefix = f"g{n}"
                arrays.update({f"{prefix}_{k}": packed[k] for k in _TREE_FIELDS})
                arrays[f"{prefix}_offset"] = offset
                members.append({"kind": "gbm", "lr": base_cfg["lr"], "base": float(gbm.base),
                                "n_trees": int(len(offset) - 1), "seed": int(cfg["seed"]),
                                "round_index": int(round_index)})
        if parent_kind == "baseline":
            # measure the parent as the reference point of this round
            xr = np.clip((raw_va - parent["mean"]) / parent["scale"], -10, 10)
            w1, b1, w2, b2 = (parent["params"][k] for k in ("w1", "b1", "w2", "b2"))
            base_pred = np.maximum((np.tanh(xr @ w1 + b1) @ w2[:, 0] + b2[0])
                                   * float(parent["meta"].get("target_scale", 1.0))
                                   + float(parent["meta"].get("target_mean", 0.0)), 0)
            meta["parent_validation"] = regression_metrics(base_pred, val["y"])
            log(f"parent baseline validation={meta['parent_validation']}")

    bundle = Bundle(meta, arrays)
    val_pred = bundle.predict(raw_va)
    if task == "classification":
        metrics = classification_metrics(val_pred, val["y"])
        score = metrics["macro_f1"]
    else:
        metrics = regression_metrics(val_pred, val["y"])
        metrics["nasa_score"] = nasa_score(val_pred, val["y"])
        score = -metrics["rmse"]
    report = {"dataset_key": dataset_key, "task": task, "round_index": int(round_index),
              "members": len(meta["members"]), "seed": int(recipe["seeds"][0]),
              "validation": metrics, "score": float(score),
              "train_rows": len(train["ids"]), "validation_rows": len(val["ids"]),
              "feature": feat_cfg, "pca": int(recipe.get("pca", 0)),
              "training_seconds": round(time.time() - started, 2), "parent_kind": parent_kind,
              "parent_sha256": _sha256(parent_path) if parent_path else None}
    meta["validation"] = metrics
    meta["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return bundle, report


def train_strict_bundle(dataset_key, train_path, val_path, parent_path, *, round_index=1, log=print):
    """Strict continuation: keep the parent MLP architecture/normalisation and keep training it.

    Mirrors the platform recipe (``baseline/model.py``): same feature scalar,
    same target scaling, same hidden width; only epochs and learning rate change.
    The artifact uses the bundle container but carries exactly one MLP member.
    """
    if dataset_key not in RECIPES:
        raise ValueError(f"unknown dataset key {dataset_key}")
    recipe = _merge_recipe(RECIPES[dataset_key], recipe_override)
    task = recipe["task"]
    classes = recipe["classes"]
    train = read_jsonl(train_path, task, classes)
    val = read_jsonl(val_path, task, classes)
    raw_tr, raw_va = train["x"], val["x"]

    with np.load(parent_path, allow_pickle=False) as data:
        is_bundle = "meta" in data.files and str(data["meta"]).find(FORMAT) >= 0
    if is_bundle:
        parent = Bundle.load(parent_path)
        idx = next((i for i, m in enumerate(parent.meta["members"]) if m["kind"] == "mlp"), None)
        if idx is None:
            raise ValueError("strict parent bundle has no MLP member")
        member = parent.meta["members"][idx]
        hidden = int(member["hidden"][0])
        nl = len(member["hidden"]) + 1
        weights = {f"W{j}": parent.arrays[f"m{idx}_W{j}"].copy() for j in range(nl)}
        weights.update({f"b{j}": parent.arrays[f"m{idx}_b{j}"].copy() for j in range(nl)})
        mean = np.asarray(parent.meta["feat_mean"], dtype=np.float64)
        scale = np.asarray(parent.meta["feat_scale"], dtype=np.float64)
        t_mean = float(parent.meta.get("target_mean", 0.0))
        t_scale = float(parent.meta.get("target_scale", 1.0))
        base_lr = float(member.get("lr", 1e-3))
        seed = int(member.get("seed", 0))
    else:
        parent = load_baseline(parent_path)
        hidden = int(parent["meta"]["hidden"])
        weights = {"W0": parent["params"]["w1"].copy(), "b0": parent["params"]["b1"].copy(),
                   "W1": parent["params"]["w2"].copy(), "b1": parent["params"]["b2"].copy()}
        mean, scale = parent["mean"].copy(), parent["scale"].copy()
        t_mean = float(parent["meta"].get("target_mean", 0.0))
        t_scale = float(parent["meta"].get("target_scale", 1.0))
        base_lr = float(parent["meta"].get("learning_rate", 1e-3))
        seed = int(parent["meta"].get("seed", 42))

    xr_tr = np.clip((raw_tr - mean) / scale, -10, 10)
    xr_va = np.clip((raw_va - mean) / scale, -10, 10)
    epochs = 200 if task == "classification" else (100 if dataset_key == "B" else 60)
    cfg = {"hidden": [hidden], "activation": "tanh", "batch_size": 128, "lr": base_lr * 0.35,
           "weight_decay": 0.0, "ema_decay": 0.0, "cosine": False, "patience": 0,
           "eval_every": 1, "epochs": epochs, "n_classes": len(classes) if task == "classification" else 1,
           "weights": weights}
    y_tr = train["y"] if task == "classification" else (train["y"] - t_mean) / t_scale
    y_va = val["y"] if task == "classification" else (val["y"] - t_mean) / t_scale
    model, best, history = train_mlp(xr_tr, y_tr, xr_va, y_va, task, cfg, seed=seed, log=None)

    arrays = {}
    _package_mlp(arrays, "m0", best["params"], [hidden], "tanh", seed)
    meta = {"format": FORMAT, "dataset_key": dataset_key, "task": task, "classes": list(classes),
            "feature": {"mode": "raw"}, "pca": 0,
            "feat_mean": [float(v) for v in mean], "feat_scale": [float(v) for v in scale],
            "target_mean": t_mean, "target_scale": t_scale, "round_index": int(round_index),
            "strict_track": True,
            "members": [{"kind": "mlp", "hidden": [hidden], "activation": "tanh",
                         "seed": int(seed), "epochs": int(epochs), "best_epoch": int(best["epoch"]),
                         "lr": float(cfg["lr"])}],
            "parent_sha256": _sha256(parent_path)}
    bundle = Bundle(meta, arrays)
    pred = bundle.predict(raw_va)
    if task == "classification":
        metrics = classification_metrics(pred, val["y"])
        score = metrics["macro_f1"]
    else:
        metrics = regression_metrics(pred, val["y"])
        score = -metrics["rmse"]
    meta["validation"] = metrics
    meta["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = {"dataset_key": dataset_key, "task": task, "round_index": int(round_index),
              "members": 1, "seed": int(seed), "validation": metrics, "score": float(score),
              "train_rows": len(train["ids"]), "validation_rows": len(val["ids"]),
              "feature": {"mode": "raw"}, "pca": 0, "track": "strict",
              "training_seconds": 0.0, "parent_kind": "bundle" if is_bundle else "baseline",
              "parent_sha256": _sha256(parent_path), "epochs": epochs}
    print(f"strict {dataset_key}: before={meta.get('parent_validation')} after={metrics} "
          f"best_epoch={best['epoch']}", file=__import__("sys").stderr, flush=True)
    return bundle, report
