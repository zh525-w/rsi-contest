# 提交包使用说明

## 目录

```
submission/
  Dockerfile              # python:3.12-slim-bookworm + numpy + 官方 SDK（vendored）
  requirements.txt        # numpy==2.3.5
  build.ps1 / build.sh    # 构建、架构检查、describe 自检、docker save
  app/
    harness.py            # 官方 4 行入口：from contest_harness.cli import main
    contest_submission.py # train(rnd) / predict(sample, model_dir)
    contest_core.py       # 模型与数据代码（纯 NumPy）
    experiment_log.py     # 可选 SwanLab 记录（默认离线，只写本地 JSONL）
    contest_harness/      # 官方 SDK 1.0.2（vendored，随镜像走）
```

镜像内：`/app` 只读，`/workspace` 可写，`/data/{train,validation}.jsonl` 与 `/previous` 只读，
`/model` 在预测时挂载选中产物。

## 构建与上传（平台要求 Linux/amd64 未压缩 tar）

```sh
# Windows + Docker Desktop
powershell -ExecutionPolicy Bypass -File solution/submission/build.ps1 -Save

# 本机 WSL2（已安装 docker.io，无需 Docker Desktop）
powershell -ExecutionPolicy Bypass -File solution/submission/build-wsl.ps1

# Linux / macOS
sh solution/submission/build.sh --save
```

本机已产出并验证的产物：`solution/submission/contest-harness-1.0.tar`
（62.6 MiB，linux/amd64，sha256 `88b83cd07a9f29a265e060e0f0f8082878565cc124393e4b035f62e6ffb21b5f`），
已在本机 Docker 上 `docker load` 后跑通 `describe` 与 `predict`。

脚本依次执行：

1. `docker build --platform linux/amd64 -t contest-harness:1.0 -f Dockerfile .`
2. `docker image inspect --format '{{.Os}}/{{.Architecture}}'` → 必须输出 `linux/amd64`
3. `docker run --rm contest-harness:1.0 describe` → 输出
   `{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk_version":"1.0.2","tasks":["classification","regression"]}`
4. `docker save -o contest-harness-1.0.tar contest-harness:1.0`

把 `contest-harness-1.0.tar` 上传到「作品提交 → Harness 镜像」。**不要**上传 zip、源码 tar 或
`docker export` 产物；`docker build` 会自动产生多架构 manifest，`docker save` 要求单镜像，
脚本已固定 `--platform linux/amd64`。

预期体积：约 160–190 MiB（python:3.12-slim + numpy）。若平台对体积敏感，可把基础镜像换成
`python:3.12-alpine`（约减半），模型代码与 SDK 均为纯 Python，不依赖 glibc 特性。

## 本地自测（无需平台）

```sh
# 1) describe（官方 SDK 入口）
PYTHONPATH=solution/submission/app python solution/submission/app/harness.py describe

# 2) 预测
CONTEST_ARTIFACT_PATH=<候选路径> python solution/submission/app/harness.py predict < rows.jsonl
```

完整一轮训练需要平台 host 帧；仓库里的 `work/sdk_round_test.py` 用官方 SDK 自带
`tests/support.py` 的 Host 替身跑通第 1 轮与第 2 轮（含回滚），可作为回归测试。

## 每轮两次真实模型调用

1. **策略选择**（训练前）：把父模型指标、最近三轮历史、以及该数据集的可选训练策略
   （特征模式 / 集成规模 / 树数 / PCA 等代码级选项）交给托管模型，按返回的
   `{"strategy": i, "reason": "..."}` 选择配置并在提交信息里记录 `strategy`。
2. **保留/回滚**（平台测量后）：把候选与父模型指标交给模型给出 `{"select": ...}`；
   代码再用“候选必须优于父模型（含 baseline 的 `parent_validation`）”护栏兜底，回滚时
   `selected_artifact_id` 填 `rnd.parent.artifact_id`，并把父检查点写回本轮 state。

两次调用都带 `enable_thinking=False`（SDK 允许的选项），回执号写入 `finish(model_call_ids=...)`；
策略表见 `contest_submission.STRATEGIES`。提示词不包含字面 JSON 示例，只接受完整 JSON 对象，
解析失败时策略回落到已验证默认配方、保留决策回落到指标护栏。

## 一轮训练做什么

1. 读 `task_contract.public_projection.train_path / validation_path`（缺省 `/data/*.jsonl`）与父检查点。
2. 按 `dataset_key` 选择配方（A：log+类别加权 MLP 集成；B：log+平方特征提升树；C：原始特征+PCA 提升树）。
3. 父检查点处理：A 迁移父权重后续训；B/C 记录父指标作为对照，跨轮追加残差树。
4. 写 `checkpoint.npz` + `training.json`，`rnd.submit_candidate(...)` 提交（`code_paths=[]`，
   代码已在镜像内固定）。
5. 模型决策 + 指标护栏，选中产物写入 `/workspace/state-<round>/` 并 `rnd.save_state`。
6. `rnd.finish(...)` 结束本轮。
