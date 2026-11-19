# 训练记录汇总

生成时间：2026-11-19T12:55:54Z
记录数：24（decision 15 / training_run 3 / artifact 6）

| 数据集 | 类型 | 轮次 | 轨道 | 指标 | 值 | 决策 | 来源 |
|---|---|---|---|---|---|---|---|
| A | artifact | 1 | boost | macro_f1 | 0.7393 |  | solution/models/A/training.json |
| A | artifact |  | strict | macro_f1 | 0.7268 |  | solution/models_strict/A/training.json |
| A | decision | 1 | boost | macro_f1 | 0.7141 | keep | reports/experiments.jsonl |
| A | decision | 2 | boost | macro_f1 | 0.7309 | keep | reports/experiments.jsonl |
| A | decision | 3 | boost | macro_f1 | 0.7499 | keep | reports/experiments.jsonl |
| A | decision | 4 | boost | macro_f1 | 0.6190 | revert | reports/experiments.jsonl |
| A | decision | 5 | boost | macro_f1 | 0.5323 | revert | reports/experiments.jsonl |
| A | decision | 6 | boost | macro_f1 | 0.7393 | keep | reports/experiments.jsonl |
| A | training_run | 1 | boost | macro_f1 | 0.7393 |  | reports/experiments_cloud.jsonl |
| B | artifact | 1 | boost | rmse | 326.0215 |  | solution/models/B/training.json |
| B | artifact |  | strict | rmse | 653.1033 |  | solution/models_strict/B/training.json |
| B | decision | 1 | boost | rmse | 653.1000 | keep_as_control | reports/experiments.jsonl |
| B | decision | 2 | boost | rmse | 480.1000 | keep | reports/experiments.jsonl |
| B | decision | 3 | boost | rmse | 326.0200 | keep | reports/experiments.jsonl |
| B | decision | 4 | boost | rmse | 459.1000 | revert | reports/experiments.jsonl |
| B | decision | 5 | boost | rmse | 637.4000 | revert | reports/experiments.jsonl |
| B | training_run | 1 | boost | rmse | 326.0215 |  | reports/experiments_cloud.jsonl |
| C | artifact | 1 | boost | rmse | 49.0704 |  | solution/models/C/training.json |
| C | artifact |  | strict | rmse | 51.3095 |  | solution/models_strict/C/training.json |
| C | decision | 1 | boost | rmse | 49.9500 | keep | reports/experiments.jsonl |
| C | decision | 2 | boost | rmse | 48.8000 | keep | reports/experiments.jsonl |
| C | decision | 3 | boost | rmse | 49.0700 | keep | reports/experiments.jsonl |
| C | decision | 4 | boost | rmse | 51.3100 | keep_as_control | reports/experiments.jsonl |
| C | training_run | 1 | boost | rmse | 49.0704 |  | reports/experiments_cloud.jsonl |
