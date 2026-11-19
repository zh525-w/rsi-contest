# 工业模型自进化系统 — 训练报告（A/B/C）

> RSI Challenge · 智训引擎 ｜ 队伍：张昊 ｜ 2026-11-19 ｜ 数据包 cpu-a/b/c-20261110-v1


> 数据与代码均来自平台公开包（`cpu-a/b/c-20261110-v1`）。训练只用 `public/train.jsonl` 做梯度更新，
> 模型选择只用 `public/validation.jsonl`；未使用隐藏集、未获取隐藏标签、未引入外部数据。

## 1. 结论摘要

以平台给定 baseline 检查点为起点，三个数据集在公开验证集上的结果：

| 数据集 | 任务 | 主指标 | 平台 baseline | 本轮模型 | 变化 |
|---|---|---|---|---|---|
| A 气体识别 | 6 类分类 | Macro-F1 ↑ | 0.7140 | **0.7393** | +0.0253（+3.5%） |
| B 轴承 RUL | 回归（分钟） | RMSE ↓ | 681.16 | **326.02** | −355.14（−52.1%） |
| C 发动机 RUL | 回归（循环） | RMSE ↓ | 52.87 | **49.07** | −3.80（−7.2%） |

同口径的**严格同族续训**对照（只继续训练平台 MLP，不改结构/特征）：

| 数据集 | baseline | 严格续训 | 变化 |
|---|---|---|---|
| A | 0.7140 | 0.7268 | +1.8% |
| B | 681.16 | 653.10 | −4.1% |
| C | 52.87 | 51.31 | −2.9% |

结论：**A 保持 baseline 的 MLP 家族并直接续训父权重**；**B/C 的父 MLP 在新轴承/新发动机上出现明显负迁移**
（把父模型预测当初始值或把父隐层当特征，验证误差反而变差，见 §4.2），因此 B/C 在保留父检查点血缘与对照的前提下
改用“同族续训 + 提升树校正”的可验证方案。两套产物都已生成，可随时切换（见 §6）。

## 2. 系统与复现

```
solution/
  contest_core.py        # 数据IO/特征/MLP/提升树/产物打包（纯 NumPy）
  train_final.py         # 离线训练 A/B/C 主产物 -> solution/models/<key>/
  train_strict.py        # 严格同族续训 -> solution/models_strict/<key>/
  models/{A,B,C}/        # checkpoint.npz + training.json
  models_strict/{A,B,C}/ # contest.cpu-mlp.v1 检查点（平台 Model.load 可校验）
  submission/
    Dockerfile
    build.ps1 / build.sh / build-wsl.ps1
    app/
      harness.py          # 官方入口：contest_harness.cli
      contest_submission.py
      contest_core.py
      experiment_log.py   # 可选 SwanLab 记录（默认离线）
      contest_harness/    # 官方 SDK 1.0.2（vendored）
```

复现命令：

```sh
python solution/train_final.py A      # ~6 s
python solution/train_final.py B      # ~40 s
python solution/train_final.py C      # ~3 min
python solution/train_strict.py A B C # ~9 s
```

环境：CPU、NumPy 2.3.5、Python 3.12；单轮训练在 1 核上 < 5 分钟，满足平台 CPU 资源限制。

## 3. 方法与每数据集设计

### A：气体传感器漂移下的 6 类识别

- 输入：128 维（16 传感器 × 8 特征）；训练=批次 1（445 条），验证=批次 2（1244 条），隐藏集为后续批次，存在明显漂移。
- 预处理：有符号 `log1p` 压缩重尾量纲 → 训练集均值/方差标准化 → 裁剪到 ±10。
- 模型：64 单元 tanh MLP，Adam（lr 1e-3，余弦退火 + warmup）、权重衰减 1e-4、EMA(0.995)、
  类别加权 `(N/(K·n_c))^1`（Macro-F1 需要小类召回）、5 个随机种子概率平均。
- 迁移：1 号成员由父检查点权重迁移（128×32 → 128×64，复制原 32 个单元、其余零输出初始化，函数等价）后继续训练。
- 未采用：测试时批归一化/CORAL（验证 F1 0.60–0.65，负迁移）、多分类提升树（0.53）。

### B：轴承剩余寿命（分钟）

- 输入：34 维双通道特征；训练 9 根轴承（123–2538 分钟），验证 3 根（42–1515 分钟），按整根轴承隔离。
- 关键观察：同一特征值在不同轴承对应不同 RUL，绝对寿命尺度无法仅由单帧特征确定，父 MLP 的预测偏差在跨轴承时不可校正。
- 模型：`log1p + 平方` 特征（68 维）→ 标准化 → 5 个 depth-2 提升树成员（lr 0.03、min_leaf 40、行采样 0.8、150 棵）平均。
- 对照实验：ridge 480、MLP 533、depth-3 树 459（过拟合）、depth-1 树 340、父模型初始化 637、父隐层特征 415、严格续训 653。

### C：涡扇发动机剩余寿命（循环）

- 输入：3 运行设置 + 21 传感器；训练 496 台发动机 / 113,089 条，验证 106 台 / 23,049 条，按整台隔离。
- 模型：原始特征 → 标准化 → 追加 8 个 PCA 分量（解释 99.7% 方差）→ 2 个 depth-3 提升树成员（lr 0.05、min_leaf 40、行采样 0.4、300 棵）平均。
- 对照：depth-3 GBM 49.95、+PCA 48.80、双种子集成 49.07、父模型 52.87、严格续训 51.31。
- 说明：单帧特征对 >150 循环的样本几乎不可分，RMSE 下限受此限制；NASA 非对称分数一并报告（§5）。

## 4. 自进化闭环与关键实验证据

每轮执行：读历史 → 诊断（父模型/候选误差）→ 实验（训练候选）→ 平台测量 → 决策（保留/回滚）→ 保存原生状态。
`contest_submission.train()` 中，模型调用（`rnd.model`）读取父/候选指标与历史后给出 keep/revert 判断，
代码同时用“候选必须优于父模型”的指标护栏兜底，避免把变差的候选写进下一轮状态。

### 4.1 三轮以上完整记录（摘录）

| 轮 | 数据集 | 假设 | 改动 | 验证结果 | 决策 |
|---|---|---|---|---|---|
| R1 | A | 复现 baseline 以确认评测口径 | 原样 20 epoch | F1 0.7141（baseline 0.7140） | 口径一致，继续 |
| R2 | A | 重尾特征应做对数压缩，宽网络+EMA 更稳 | log1p、h64、300 epoch、EMA | 0.7309 | 采纳 |
| R3 | A | Macro-F1 需要小类加权 | 类别权重 α=1 | 0.7499（3 种子） | 采纳 |
| R4 | A | 漂移可用测试时矩匹配消除 | 在线/整批标准化 | 0.60–0.65 | 拒绝（负迁移） |
| R5 | A | 树模型更适合表格特征 | 多分类 GBM | 0.5323 | 拒绝 |
| R1 | B | 父模型误差来自容量不足 | 继续训练 MLP | 653（−4.1%） | 保留为对照 |
| R2 | B | 线性/浅层模型可刻画退化 | ridge log 特征 | 480 | 采纳为基线 |
| R3 | B | 单帧绝对 RUL 需非线性分段 | depth-2 GBM | 329 | 采纳 |
| R4 | B | 更多深度能提升 | depth-3 GBM | 459 | 拒绝（过拟合） |
| R5 | B | 父模型知识可迁移 | 父预测初始化 / 父隐层特征 | 637 / 415 | 拒绝（负迁移） |
| R1 | C | 单帧 RUL 可由传感器组合回归 | depth-3 GBM | 49.95 | 采纳 |
| R2 | C | 多传感器共线性可用 PCA 压缩 | +8 PCA 分量 | 48.80 | 采纳 |
| R3 | C | 种子集成降低方差 | 2 种子平均 | 49.07 | 采纳（稳定性优先） |

### 4.2 负迁移证据（B）

- 父 MLP 验证 RMSE 681；把其预测作为提升树初值：637（更差）。
- 把父隐层 24 维 tanh 激活追加为特征：415（明显差于 326）。
- 严格续训父 MLP 100 epoch（lr×0.3，早停）：653。

即：父 MLP 在训练轴承上拟合了“轴承绝对寿命尺度”，该尺度不能跨轴承迁移；继续同族训练只能小幅降低误差，
必须换用对单调/分段退化更稳健的树模型才能获得量级改善。这属于策略层自进化的取舍，已在 `change_summary` 中如实记录。

## 5. 评测明细

| 数据集 | 指标 | baseline | 主模型 | 严格续训 |
|---|---|---|---|---|
| A | Accuracy | 0.8786 | 0.8947 | 0.8867 |
| A | Macro-F1 | 0.7140 | 0.7393 | 0.7268 |
| B | RMSE（分钟） | 681.16 | 326.02 | 653.10 |
| B | MAE（分钟） | 563.11 | 274.52 | 547.73 |
| C | RMSE（循环） | 52.87 | 49.07 | 51.31 |
| C | MAE（循环） | 38.90 | 34.63 | 37.49 |

A 的分类别 F1（验证）：Ethanol 0.712、Ethylene 0.972、Ammonia 0.990、Acetaldehyde 0.053、Acetone 0.995、Toluene 0.714。
Acetaldehyde 在批次 2 几乎全部被判为 Ethanol，属于传感器漂移导致的类别整体位移，是当前方法的主要风险点。

C 的 NASA 非对称分数（越低估惩罚越小、越高估越大）在主模型上约为 8.6e9，量级较大，原因是少量高估样本被指数放大；
该分数对样本集敏感，仅作参考。

## 6. 提交与产物

- `solution/models/<A|B|C>/checkpoint.npz`：主产物（`contest.industrial.v2`，含预处理、成员、父检查点摘要与验证指标）。
- `solution/models_strict/<A|B|C>/checkpoint.npz`：严格续训产物，保持 `contest.cpu-mlp.v1` 格式，平台 `Model.load` 可直接校验。
- 镜像内 `python3 /app/harness.py describe|train|predict` 已按 SDK 文档实现：
  `contest_submission.py` 只使用 `rnd.prompt/dataset_key/round_index/task/model/history/parent/state/submit_candidate/finish`。
- 每轮候选写入 `/workspace/round-<n>/`，仅提交 `checkpoint.npz` 与元数据，远小于 64 MiB / 64 文件限制；
  本轮状态写入新目录 `/workspace/state-<n>/`（`state_workspace` 拒绝已存在路径），不触碰 `/app`、`/previous`。
- **镜像**：`solution/submission/` 为完整构建上下文——官方 `harness.py`、vendored SDK 1.0.2、
  `contest_submission.py`、纯 NumPy 模型代码、`Dockerfile`、`build.ps1/build.sh/build-wsl.ps1`。
  本机 WSL2（Ubuntu 26.04 + docker.io 29.1.3）构建出 `contest-harness-1.0.tar`（62.6 MiB，linux/amd64，
  sha256 `88b83cd07a9f29a265e060e0f0f8082878565cc124393e4b035f62e6ffb21b5f`），
  `docker load` 后在容器内复测 `describe` 与 `predict` 通过。
- **训练记录**：`reports/training-records/`（`solution/build_records.py` 可重复生成）与 SwanLab 云端链接（§9）。
- 如需改用严格同族续训作为提交产物，设置 `CONTEST_TRACK=strict` 或用 `train_strict.py` 生成的检查点替换
  `models/`；B/C 主指标会回落到 646 / 51.3。

## 7. 风险与合规

- **漂移**：A 的隐藏集为后续批次，验证集只覆盖批次 2，Macro-F1 可能低于本地值；已用类别加权与对数特征降低对小类先验的依赖。
- **分布外轴承/发动机**：B/C 的隐藏集按整根/整台隔离，验证集只有 3 根 / 106 台，绝对 RUL 尺度不确定性大；树模型的分段性优于线性外推，但仍可能在极长寿命样本上低估。
- **合规**：未读取 `/hidden`、未使用隐藏标签、未把 `id/meta` 或目标值拼进特征；`data.py` 的禁用字段检查在 `contest_core` 中保持等价语义（仅数值特征）。
- **可复现**：固定种子、固定 epoch/树数、单一 NumPy 实现；产物含 `parent_sha256`、`validation`、`created_at`。

## 8. 模型身份与云端实验记录

- **模型身份（已验证）**：PyTRIO 账号可见 `Qwen/Qwen3.5-4B` 与 `Qwen/Qwen3.8-27B`（上下文 65536），
  用后者做策略/保留决策的实测调用可正常返回并可解析 JSON，因此按白名单模型 `Qwen/Qwen3.8-27B` 提交；
  若后续用 PyTRiO 训练出自己的持久 Sampler，再改为完整 `trio://...` 路径。参数 ≤ 27B。
  PyTRiO API Key 仅填平台密钥栏，镜像内 `rnd.model()` 走 host gateway，不含端点与密钥。
- **真实调用验证**：用 PyTRIO 直连 Qwen/Qwen3.8-27B 跑通完整一轮 A（策略调用 7 token、保留决策 12 token，
  两次回执号写入 `model_call_ids`），策略选择 `log64-5seed`、候选 0.7393 > 父模型 0.7140，提交候选。
  Qwen3 系列默认带思考过程；提示词因此不含字面 JSON 示例，只解析完整 JSON 对象，失败即回落默认配方/指标护栏。
- **每轮记录字段**：`dataset / round_index / validation / members / feature / pca /
  parent_sha256 / artifact_sha256 / training_seconds`，写入 `training.json`、`history.jsonl`
  与 `reports/experiments_cloud.jsonl`。
- **训练记录包**：`reports/training-records/`（`solution/build_records.py` 可重复生成）：
  `records.jsonl`（24 条：15 条决策 + 3 条运行 + 6 条产物）、`summary.csv`、`summary.md`、
  `appendix.md`（逐条假设/改动/指标/决策/父与产物摘要）、`manifest.json`（sha256）。
- **镜像**：`solution/submission/` 为完整构建上下文——官方 `harness.py`、vendored SDK 1.0.2、
  `contest_submission.py`、纯 NumPy 模型代码、`Dockerfile`、`build.ps1/build.sh/build-wsl.ps1`。
  本机 WSL2（Ubuntu 26.04 + docker.io 29.1.3）实际构建出 `contest-harness-1.0.tar`（62.6 MiB，linux/amd64），
  `docker load` 后在容器内复测 `describe` 与 `predict` 通过。

## 9. 云端实验记录

SwanLab 项目：<https://swanlab.cn/@slience917/industrial-contest>（共 24 条记录）

| 数据集 | run | 记录数 |
|---|---|---|
| A | <https://swanlab.cn/@slience917/industrial-contest/runs/lv4la8tw> | 9 |
| B | <https://swanlab.cn/@slience917/industrial-contest/runs/7ace77dc> | 8 |
| C | <https://swanlab.cn/@slience917/industrial-contest/runs/4ivaa95n> | 7 |

每个 run 按 step 记录 `val/macro_f1`、`val/accuracy`、`val/rmse`、`val/mae` 与轮次；
记录来源为 `reports/training-records/records.jsonl`，与 `appendix.md` 逐条对应。
