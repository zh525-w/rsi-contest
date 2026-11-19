# RSI Challenge · 工业模型自进化系统（A/B/C）

参赛作品源码与文档。三个数据集（气体识别、轴承 RUL、发动机 RUL）在平台给定 baseline 检查点上续训/改进，
并通过 Harness 镜像完成「训练 → 平台测量 → 保留/回滚」的自进化闭环。

## 结果（公开验证集）

| 数据集 | 任务 | 主指标 | 平台 baseline | 本作品 |
|---|---|---|---|---|
| A 气体识别 | 6 类分类 | Macro-F1 ↑ | 0.7140 | **0.7393** |
| B 轴承 RUL | 回归（分钟） | RMSE ↓ | 681.16 | **326.02** |
| C 发动机 RUL | 回归（循环） | RMSE ↓ | 52.87 | **49.07** |

严格同族续训对照：A 0.7268 / B 646.37 / C 51.31（`CONTEST_TRACK=strict`）。

## 目录

```
solution/
  contest_core.py        # 数据IO、特征、MLP、提升树、产物打包（纯 NumPy）
  train_final.py         # 离线训练主产物
  train_strict.py        # 严格同族续训（baseline 原生格式）
  build_records.py       # 训练记录包生成
  experiment_log.py      # 可选 SwanLab 记录（默认离线）
  upload_experiments.py  # 将本地记录上传 SwanLab
  model-identity.json    # 模型身份（Qwen/Qwen3.8-27B）
  submission/            # Harness 镜像构建上下文（官方 SDK 1.0.2 + 入口 + 选手模块）
reports/
  report.md              # 技术报告源文件
  技术报告.pdf            # 技术报告（6 页）
  提交材料清单.md/.pdf     # 材料清单、模型调用信息、来源声明、视频脚本
  training-records/      # 24 条训练记录与云端链接
```

## 复现

```sh
python solution/train_final.py A      # ~6 s
python solution/train_final.py B      # ~40 s
python solution/train_final.py C      # ~3 min
python solution/train_strict.py A B C # 严格续训对照
python solution/build_records.py      # 生成 reports/training-records/
```

镜像构建（Linux/amd64，需 Docker 或本机 WSL2）：

```sh
powershell -ExecutionPolicy Bypass -File solution/submission/build-wsl.ps1
# 或
sh solution/submission/build.sh --save
```

## 提交信息

- 驱动模型：`Qwen/Qwen3.8-27B`（PyTRiO 白名单，≤27B）
- 实验记录：<https://swanlab.cn/@slience917/industrial-contest>
- Harness 镜像：`contest-harness-1.0.tar`（62.6 MiB，linux/amd64，sha256 `88b83cd0…1b5f`）

## 合规

仅使用平台公开的 `public/train.jsonl` 与 `public/validation.jsonl`；未使用隐藏集、隐藏标签或外部数据。
