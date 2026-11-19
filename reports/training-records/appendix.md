# 附录：训练与实验记录明细

生成时间：2026-11-19T12:55:54Z；共 24 条记录。数据只来自平台公开投影的训练/验证文件。

## 数据集 A（9 条）

### artifact-A-boost
- 类型/轨道：artifact / boost
- 轮次：1
- 配置：`{"feature": {"mode": "log"}, "members": 5, "pca": 0, "strategy": null}`
- 验证指标：`{"accuracy": 0.8946945337620579, "macro_f1": 0.7392536961478594, "per_class_f1": [0.7116279069767442, 0.9715994020926756, 0.9900990099009901, 0.05263157894736842, 0.9952785646836638, 0.7142857142857143]}`
- 父检查点：`sha256:fd1e08b2aad08b2e9754f9b2dfdc029481c0da7ecdedca0abb0ec6fcc818c757`
- 产物摘要：`sha256:7ef42e0379e508cfa58bdafbed2c47d44097bf5ed167a75ad2a45d569c3a6be3`
- 来源：`solution/models/A/training.json`

### artifact-A-strict
- 类型/轨道：artifact / strict
- 配置：`{"best_epoch": 14, "epochs": 200, "learning_rate": 0.00105}`
- 验证指标：`{"accuracy": 0.8866559485530546, "macro_f1": 0.7268296482589388}`
- 对照：`{"accuracy": 0.8786173633440515, "macro_f1": 0.7140297220788709}`
- 父检查点：`sha256:fd1e08b2aad08b2e9754f9b2dfdc029481c0da7ecdedca0abb0ec6fcc818c757`
- 产物摘要：`sha256:93ed550a5e6ef9cc1ab069f31274c05fa081a8b38fd5767124949240e866bf68`
- 来源：`solution/models_strict/A/training.json`

### decision-A-r1-1
- 类型/轨道：decision / boost
- 轮次：1
- 假设：复现平台 baseline 以确认评测口径
- 改动：原样 CPU MLP，20 epoch，lr 3e-3
- 验证指标：`{"accuracy": 0.8794, "macro_f1": 0.7141}`
- 对照：`{"accuracy": 0.8786, "macro_f1": 0.714}`
- 决策：keep（与 training.json 记录一致，口径可信）
- 来源：`reports/experiments.jsonl`

### decision-A-r2-2
- 类型/轨道：decision / boost
- 轮次：2
- 假设：重尾特征对数压缩 + 更宽网络 + EMA 能提升跨批次泛化
- 改动：signed log1p、hidden 64、300 epoch、EMA 0.995、余弦退火
- 验证指标：`{"accuracy": 0.8939, "macro_f1": 0.7309}`
- 对照：`{"macro_f1": 0.714}`
- 决策：keep（Macro-F1 +0.017，验证集一致提升）
- 来源：`reports/experiments.jsonl`

### decision-A-r3-3
- 类型/轨道：decision / boost
- 轮次：3
- 假设：Macro-F1 以类别等权计分，小类需要加权
- 改动：类别权重 (N/(K*n_c))^1
- 验证指标：`{"accuracy": 0.8971, "macro_f1": 0.7499}`
- 对照：`{"macro_f1": 0.714}`
- 决策：keep（小类召回提高，宏平均提升）
- 来源：`reports/experiments.jsonl`

### decision-A-r4-4
- 类型/轨道：decision / boost
- 轮次：4
- 假设：传感器漂移可用测试时矩匹配消除
- 改动：整批/在线标准化对齐训练统计量
- 验证指标：`{"accuracy": 0.8489, "macro_f1": 0.619}`
- 对照：`{"macro_f1": 0.714}`
- 决策：revert（所有变体均低于 baseline，属负迁移）
- 来源：`reports/experiments.jsonl`

### decision-A-r5-5
- 类型/轨道：decision / boost
- 轮次：5
- 假设：树模型更适合表格特征
- 改动：多分类梯度提升树（softmax 损失）
- 验证指标：`{"accuracy": 0.7677, "macro_f1": 0.5323}`
- 对照：`{"macro_f1": 0.714}`
- 决策：revert（明显低于 MLP，拒绝）
- 来源：`reports/experiments.jsonl`

### decision-A-r6-6
- 类型/轨道：decision / boost
- 轮次：6
- 假设：种子集成 + 父权重迁移可稳定收益
- 改动：5 种子集成，成员0由父 32→64 单元迁移后续训
- 验证指标：`{"accuracy": 0.8947, "macro_f1": 0.7393}`
- 对照：`{"macro_f1": 0.714}`
- 决策：keep（高于 baseline 且保留父检查点血缘）
- 来源：`reports/experiments.jsonl`

### run-A-r1-boost
- 类型/轨道：training_run / boost
- 轮次：1
- 配置：`{"feature": {"mode": "log"}, "members": 5, "pca": 0}`
- 验证指标：`{"accuracy": 0.8946945337620579, "macro_f1": 0.7392536961478594, "per_class_f1": [0.7116279069767442, 0.9715994020926756, 0.9900990099009901, 0.05263157894736842, 0.9952785646836638, 0.7142857142857143]}`
- 父检查点：`sha256:fd1e08b2aad08b2e9754f9b2dfdc029481c0da7ecdedca0abb0ec6fcc818c757`
- 产物摘要：`sha256:7ef42e0379e508cfa58bdafbed2c47d44097bf5ed167a75ad2a45d569c3a6be3`
- 来源：`reports/experiments_cloud.jsonl`

## 数据集 B（8 条）

### artifact-B-boost
- 类型/轨道：artifact / boost
- 轮次：1
- 配置：`{"feature": {"mode": "log", "squares": true}, "members": 5, "pca": 0, "strategy": null}`
- 验证指标：`{"mae": 274.5151922996821, "rmse": 326.02152863028647, "nasa_score": 1.9059331157704762e+52}`
- 父检查点：`sha256:c7d819fcee9d25459c583b5ffdb35593d4a47b524ed152d0226bf81a283314b4`
- 产物摘要：`sha256:864db2e542a36118bd993ec8ac9bc6e87b7a43863d728a82296e24420cf667e1`
- 来源：`solution/models/B/training.json`

### artifact-B-strict
- 类型/轨道：artifact / strict
- 配置：`{"best_epoch": 8, "epochs": 100, "learning_rate": 0.0009}`
- 验证指标：`{"rmse": 653.1033319733872, "mae": 547.7300562055322}`
- 对照：`{"rmse": 681.1598326043113, "mae": 563.1058414514887}`
- 父检查点：`sha256:c7d819fcee9d25459c583b5ffdb35593d4a47b524ed152d0226bf81a283314b4`
- 产物摘要：`sha256:458b34a183b3a6254685c27d6595c82d34c92f441380459a3242ae24162453af`
- 来源：`solution/models_strict/B/training.json`

### decision-B-r1-7
- 类型/轨道：decision / boost
- 轮次：1
- 假设：父模型误差来自训练不足
- 改动：严格同族续训：lr×0.3，100 epoch，早停
- 验证指标：`{"rmse": 653.1, "mae": 547.7}`
- 对照：`{"rmse": 681.16, "mae": 563.11}`
- 决策：keep_as_control（仅 −4.1%，同族续训上限有限）
- 来源：`reports/experiments.jsonl`

### decision-B-r2-8
- 类型/轨道：decision / boost
- 轮次：2
- 假设：线性模型可刻画退化趋势
- 改动：ridge 回归（log 特征 + 平方特征）
- 验证指标：`{"rmse": 480.1, "mae": 390.2}`
- 对照：`{"rmse": 681.16}`
- 决策：keep（明显优于父模型，作为强基线）
- 来源：`reports/experiments.jsonl`

### decision-B-r3-9
- 类型/轨道：decision / boost
- 轮次：3
- 假设：绝对 RUL 需要非线性分段映射
- 改动：depth-2 提升树 150 棵，5 种子集成
- 验证指标：`{"rmse": 326.02, "mae": 274.52}`
- 对照：`{"rmse": 681.16}`
- 决策：keep（RMSE 降 52%，跨轴承稳健）
- 来源：`reports/experiments.jsonl`

### decision-B-r4-10
- 类型/轨道：decision / boost
- 轮次：4
- 假设：更深树能再提升
- 改动：depth-3/leaf20
- 验证指标：`{"rmse": 459.1, "mae": 421.2}`
- 对照：`{"rmse": 326.02}`
- 决策：revert（过拟合轴承个体，验证变差）
- 来源：`reports/experiments.jsonl`

### decision-B-r5-11
- 类型/轨道：decision / boost
- 轮次：5
- 假设：父模型知识可迁移到新轴承
- 改动：父预测作为提升树初值 / 父隐层作为特征
- 验证指标：`{"rmse": 637.4, "mae": 530.9}`
- 对照：`{"rmse": 326.02}`
- 决策：revert（两种迁移均负迁移，说明父 MLP 学到的是训练轴承寿命尺度）
- 来源：`reports/experiments.jsonl`

### run-B-r1-boost
- 类型/轨道：training_run / boost
- 轮次：1
- 配置：`{"feature": {"mode": "log", "squares": true}, "members": 5, "pca": 0}`
- 验证指标：`{"mae": 274.5151922996821, "rmse": 326.02152863028647, "nasa_score": 1.9059331157704762e+52}`
- 父检查点：`sha256:c7d819fcee9d25459c583b5ffdb35593d4a47b524ed152d0226bf81a283314b4`
- 产物摘要：`sha256:864db2e542a36118bd993ec8ac9bc6e87b7a43863d728a82296e24420cf667e1`
- 来源：`reports/experiments_cloud.jsonl`

## 数据集 C（7 条）

### artifact-C-boost
- 类型/轨道：artifact / boost
- 轮次：1
- 配置：`{"feature": {"mode": "raw"}, "members": 2, "pca": 8, "strategy": null}`
- 验证指标：`{"mae": 34.62967362730431, "rmse": 49.07041011658132, "nasa_score": 8558060186.204752}`
- 父检查点：`sha256:98ec6e30e38aee56ca15b7d624a8f412226d46fe97b170667a567601c700bc4e`
- 产物摘要：`sha256:961e35cd16c47ad92777e0b058ddb22838cd21f537f6511046a12ad47a490f90`
- 来源：`solution/models/C/training.json`

### artifact-C-strict
- 类型/轨道：artifact / strict
- 配置：`{"best_epoch": 51, "epochs": 60, "learning_rate": 0.0007}`
- 验证指标：`{"rmse": 51.30947245580156, "mae": 37.49252640322865}`
- 对照：`{"rmse": 52.86857256611151, "mae": 38.89982982666355}`
- 父检查点：`sha256:98ec6e30e38aee56ca15b7d624a8f412226d46fe97b170667a567601c700bc4e`
- 产物摘要：`sha256:4b602f36e62cab3fb63f2388867cd1876d7ee844f672a149fdf90f3714bff7d9`
- 来源：`solution/models_strict/C/training.json`

### decision-C-r1-12
- 类型/轨道：decision / boost
- 轮次：1
- 假设：单帧传感器组合可回归 RUL
- 改动：depth-3 提升树 350 棵
- 验证指标：`{"rmse": 49.95, "mae": 35.79}`
- 对照：`{"rmse": 52.87, "mae": 38.9}`
- 决策：keep（稳定优于父 MLP）
- 来源：`reports/experiments.jsonl`

### decision-C-r2-13
- 类型/轨道：decision / boost
- 轮次：2
- 假设：传感器共线性可用 PCA 压缩
- 改动：追加 8 个 PCA 分量（99.7% 方差）
- 验证指标：`{"rmse": 48.8, "mae": 34.39}`
- 对照：`{"rmse": 52.87}`
- 决策：keep（再降 1.15）
- 来源：`reports/experiments.jsonl`

### decision-C-r3-14
- 类型/轨道：decision / boost
- 轮次：3
- 假设：种子集成降低方差
- 改动：2 种子 × 300 棵平均
- 验证指标：`{"rmse": 49.07, "mae": 34.63}`
- 对照：`{"rmse": 52.87}`
- 决策：keep（略逊单种子但更稳定，作为提交产物）
- 来源：`reports/experiments.jsonl`

### decision-C-r4-15
- 类型/轨道：decision / boost
- 轮次：4
- 假设：父模型可迁移
- 改动：严格同族续训 lr×0.35，60 epoch
- 验证指标：`{"rmse": 51.31, "mae": 37.49}`
- 对照：`{"rmse": 52.87}`
- 决策：keep_as_control（仅 −2.9%，作为同族对照）
- 来源：`reports/experiments.jsonl`

### run-C-r1-boost
- 类型/轨道：training_run / boost
- 轮次：1
- 配置：`{"feature": {"mode": "raw"}, "members": 2, "pca": 8}`
- 验证指标：`{"mae": 34.62967362730431, "rmse": 49.07041011658132, "nasa_score": 8558060186.204752}`
- 父检查点：`sha256:98ec6e30e38aee56ca15b7d624a8f412226d46fe97b170667a567601c700bc4e`
- 产物摘要：`sha256:961e35cd16c47ad92777e0b058ddb22838cd21f537f6511046a12ad47a490f90`
- 来源：`reports/experiments_cloud.jsonl`

