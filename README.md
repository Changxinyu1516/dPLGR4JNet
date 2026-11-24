# dPLGR4JNet

这是一个独立的 PyTorch 降雨–径流建模项目，核心计算改编自 `torchhydro`。独立训练入口可使用仓库 `data` 目录中的单流域 CSV 样例训练和评估。两河流域的原始 `torchhydro` 实验入口见下文。

## 支持的模型

| 配置名 | GR4J 参数 | 径流校正网络 | 旧配置名 |
|---|---|---|---|
| `dPLGR4JStatic` | 静态 | 无 | `dPLGR4J` |
| `dPLGR4JStaticWithMLP` | 静态 | 因果残差 MLP | `dPLGR4JNet` |
| `dPLGR4JDynamic` | 动态 | 无 | `dPLGR4Jd` |
| `dPLGR4JDynamicWithMLP` | 动态 | 因果残差 MLP | `dPLGR4JNetd` |
| `dPLGR4JDynamicWaterBalance` | 动态，带水量平衡与蒸散发约束 | 无 | `DplLstmGr4jWaterBalance` |
| `dPLGR4JDynamicWaterBalanceWithMLP` | 动态，带水量平衡与蒸散发约束 | 因果残差 MLP | `DplLstmGr4jWaterBalanceWithMLP` |

动态模型可让 X1、X2、X3 中的指定参数逐日变化，X4 保持固定。YAML 和命令行仍可使用表中的旧配置名。Python 类名对应配置名，前缀写作 `DPLGR4J`，例如 `DPLGR4JDynamicWaterBalance`。

## 安装

需要 Python 3.10+：

```powershell
python -m pip install -r requirements.txt
```

## 样例数据

默认配置根据 `station_name` 自动读取：

```text
data/{station_name}_with_forcing.csv
```

CSV 至少需要日期、降水、潜在蒸散发、径流以及配置中列出的气象特征。物理模型始终将前两个特征作为降水和潜在蒸散发，因此不要改变这两个特征的顺序。

默认样例的 `runoff` 单位是 `m3/s`，程序使用流域面积转换为 GR4J 所需的 `mm/day`，也支持 `mm/day` 和 `ft3/s`。当 `area_km2: null` 时，面积按 `station_name` 从 `data/camels_topo.txt` 自动读取，默认使用 `area_gages2` 列；也可通过 `area_column` 改用 `area_geospa_fabric`，或直接填写 `area_km2` 覆盖属性文件。

## 运行

默认采用两个时期：

```python
TRAIN_PERIOD = ["1980-01-01", "2006-12-31"]
VALID_PERIOD = ["2007-01-01", "2014-12-31"]
TEST_PERIOD = None
```

`VALID_PERIOD` 在这里是训练结束后的独立评估期：训练 epoch 不读取它，不使用它反向传播，也不使用它选择最佳模型、调整学习率或提前停止。最佳模型按训练损失保存；恢复最佳模型后，才对完整 `TRAIN_PERIOD` 和 `VALID_PERIOD` 分别评估并导出结果。

每个训练 epoch 会输出训练窗口上的参考指标：

```text
epoch=001 train_loss=... NSE=... RMSE=... KGE=...
```

这些数值同时写入 `history.csv` 的 `train_NSE`、`train_RMSE`、`train_KGE` 列。训练结束后的独立时期指标保存在 `metrics_summary.csv` 和 `valid_metrics.json`。

优化器、梯度裁剪和学习率调度的公共参数集中在 `main.py` 的 `SHARED_TRAINING_PARAMS` 中；四套模型训练配置通过 `**SHARED_TRAINING_PARAMS` 显式继承，不再依赖隐藏的默认值。

### 可编辑主函数（推荐）

项目根目录的 [main.py](main.py) 是与 `torchhydro/examples/dpl_gr4j_cxy.py`、`dpl_gr4j_dyn_cxy.py` 风格相近的集中实验入口。四种模型使用四套独立参数块，基础模型中不会混入 MLP 参数，静态模型中也不会混入动态参数。

训练依赖关系为：

```text
dPLGR4JStatic  ──best_model.pt──> dPLGR4JStaticWithMLP
dPLGR4JDynamic ──best_model.pt──> dPLGR4JDynamicWithMLP
```

两个带 MLP 的模型会读取相应基础模型的 LSTM+GR4J 权重，残差 MLP 采用零初始化，然后以较小的 `dpl_learning_rate` 和较大的 `mlp_learning_rate` 联合微调。

第一步，训练静态或动态基础模型：

```python
MODEL_NAME = "dPLGR4JStatic"  # 动态流程改为 dPLGR4JDynamic
STATION_NAME = "01013500"
DRY_RUN = False
```

运行：

```powershell
python main.py
```

第二步，把模型改为相应 Net 模型后再次运行：

```python
MODEL_NAME = "dPLGR4JStaticWithMLP"  # 动态流程改为 dPLGR4JDynamicWithMLP
```

默认检查点路径为：

```text
outputs/{station_name}/dPLGR4JStatic/best_model.pt
outputs/{station_name}/dPLGR4JDynamic/best_model.pt
```

新名称的目录中缺少检查点时，主函数会查找旧名称目录。也可通过 `STATIC_DPL_CHECKPOINT` 或 `DYNAMIC_DPL_CHECKPOINT` 指定权重文件。如果希望一次命令自动先训练缺失的基础模型，可设置：

```python
AUTO_TRAIN_PREREQUISITE = True
```

主函数会检查检查点模型类型，以及 `hidden_size`、特征顺序、暖机长度、参数聚合方式和动态参数索引等是否一致，避免把不兼容的基础权重用于 Net 模型。如果只想检查数据和前向计算，将 `DRY_RUN` 改为 `True`；但 Net 模型即使 dry run 也必须具有基础检查点。

修改训练时期、窗口长度或 `WINDOW_STRIDE` 后，原基础检查点会被判定为不兼容，需要先重新训练 `dPLGR4JStatic`/`dPLGR4JDynamic`。启用 `AUTO_TRAIN_PREREQUISITE` 时，程序会自动重训缺失或不兼容的基础模型。

### 命令行入口

先检查数据和模型前向计算：

```powershell
python -m dplgr4jnet --config configs/default.yaml --dry-run
```

切换流域时，程序会自动匹配对应的 CSV 和 `camels_topo.txt` 面积：

```powershell
python -m dplgr4jnet --config configs/default.yaml `
  --station-name 01022500 `
  --model dPLGR4JDynamic `
  --dry-run
```

一次性检查四个模型的前向与反向传播：

```powershell
python tests/smoke_models.py
```

训练默认的静态基础模型 `dPLGR4JStatic`：

```powershell
python -m dplgr4jnet --config configs/default.yaml
```

选择动态基础模型：

```powershell
python -m dplgr4jnet --config configs/default.yaml --model dPLGR4JDynamic
```

通过命令行运行 Net 模型时也必须提供基础检查点：

```powershell
python -m dplgr4jnet --config configs/default.yaml `
  --model dPLGR4JStaticWithMLP `
  --pretrained-dpl-path outputs/01013500/dPLGR4JStatic/best_model.pt
```

常用覆盖参数：

```powershell
python -m dplgr4jnet --config configs/default.yaml `
  --model dPLGR4JDynamicWithMLP `
  --pretrained-dpl-path outputs/01013500/dPLGR4JDynamic/best_model.pt `
  --epochs 20 `
  --data-dir data `
  --output-dir outputs/my_run
```

动态模型通过以下配置选择逐日变化的参数，索引分别为 X1=0、X2=1、X3=2；X4 是单位线卷积核，不能逐日变化。

```yaml
model:
  name: dPLGR4JDynamicWithMLP
  pretrained_dpl_path: outputs/01013500/dPLGR4JDynamic/best_model.pt
  param_var_index: [0]
  param_test_way: mean_time
  param_smoothness_weight: 0.01
```

## 输出

每次完整运行会保存：

- `best_model.pt`：最佳训练损失对应的模型权重、配置、epoch 和损失；
- `history.csv`、`metrics_summary.csv`：训练过程和 train/valid/test 指标；
- `{split}_predictions.csv`：观测与模拟流量，同时包含 mm/day 和 m³/s；
- `{split}_parameters.csv`：逐日物理参数、实际采用的归一化参数以及 LSTM 原始生成参数；
- `{split}_predictions.xlsx`：流量、指标和参数的 Excel 汇总；
- `feature_scaler.json`、`station_meta.json`、`resolved_config.yaml`：复现实验所需信息。

静态模型的物理参数会逐日重复，便于与流量按日期关联；动态模型会保存真实的逐日参数轨迹。默认仓库已用样例流域完成一轮验证，结果位于 `outputs/sample_verified`，动态参数验证位于 `outputs/dynamic_verified`。

## 水量平衡与蒸散发约束

[两河流域原实验脚本](examples/dpl_gr4j_waterbal.py)保留了原实验的分期、特征、两阶段训练配置及 Excel 导出入口。该脚本位于 `examples/`，供使用自定义 `torchhydro` 环境和 `SXdata` 的原实验复跑使用。运行前需配置 `hydro_setting.yml`、本地数据和预训练权重路径。下文的独立训练流程使用本仓库的模型代码和 CAMELS 流域 `01411300` 的 CSV 样例。

独立版本提供 `dPLGR4JDynamicWaterBalance` 和 `dPLGR4JDynamicWaterBalanceWithMLP`。样例中仅 X2 逐日变化，MLP 使用当天和前四天的特征。两阶段都使用稳定版 NSE 损失、水量平衡残差平方均值及超出 PET 的蒸散发惩罚，权重分别为 1.0、0.1 和 0.05。水平衡计入实际地下水交换、产流库和汇流库的蓄水变化、两条单位线的在途水量及容量截断损失。

使用 [独立样例配置](configs/water_balance.yaml)检查前向计算并训练第一阶段：

```powershell
python -m dplgr4jnet --config configs/water_balance.yaml --dry-run
python -m dplgr4jnet --config configs/water_balance.yaml
```

以第一阶段的最佳权重初始化第二阶段，并给两个分支配置各自的学习率：

```powershell
python -m dplgr4jnet --config configs/water_balance.yaml `
  --model dPLGR4JDynamicWaterBalanceWithMLP `
  --pretrained-dpl-path outputs/water_balance/base/best_model.pt `
  --output-dir outputs/water_balance/mlp
```

每个评估时期会额外输出 `{split}_water_balance.csv`，并在该时期的 Excel 文件中增加 `water_balance` 工作表。逐日 X1–X4 在对应的参数 CSV 和 Excel 工作表中。使用原实验的检查点时，应保持 LSTM 隐藏维度、输入特征顺序及标准化方式与原试验一致。

运行 `python tests/smoke_water_balance.py` 可检查三项损失、两阶段训练和导出结果。

## 许可与来源

本项目包含从 BSD-3-Clause 许可的 `torchhydro` 改编的实现。详情见 [LICENSE](LICENSE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSES/torchhydro-BSD-3-Clause.txt](LICENSES/torchhydro-BSD-3-Clause.txt)。
