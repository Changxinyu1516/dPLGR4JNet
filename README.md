# dPLGR4JNet

这是一个独立的 PyTorch 降雨–径流建模项目，核心计算来自并同步到 `torchhydro` 中的 `dpl4gr4j.py` 与 `dpl4gr4j_dyna.py`。项目不要求安装 `torchhydro`，可直接使用仓库 `data` 目录中的单流域 CSV 样例训练和评估。

## 支持的模型

| 配置名 | GR4J 参数 | 径流校正网络 |
|---|---|---|
| `dPLGR4J` | 每个样本一组静态参数 | 无 |
| `dPLGR4JNet` | 每个样本一组静态参数 | 因果残差 MLP |
| `dPLGR4Jd` | X1/X2/X3 中的指定参数逐日变化，X4 固定 | 无 |
| `dPLGR4JNetd` | X1/X2/X3 中的指定参数逐日变化，X4 固定 | 因果残差 MLP |

四种模型使用相同的数据、训练与导出接口。最新版实现包含连续暖机、GR4J 的 90%/10% 单位线分流、动态参数平滑正则、预训练 dPL 检查点加载、MLP 因果历史窗口和残差零初始化。

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
dPLGR4J  ──best_model.pt──> dPLGR4JNet
dPLGR4Jd ──best_model.pt──> dPLGR4JNetd
```

两个 Net 模型不是从头训练：它们会读取相应基础模型的 LSTM+GR4J 权重，残差 MLP 采用零初始化，然后以较小的 `dpl_learning_rate` 和较大的 `mlp_learning_rate` 联合微调。

第一步，训练静态或动态基础模型：

```python
MODEL_NAME = "dPLGR4J"  # 动态流程改为 dPLGR4Jd
STATION_NAME = "01013500"
DRY_RUN = False
```

运行：

```powershell
python main.py
```

第二步，把模型改为相应 Net 模型后再次运行：

```python
MODEL_NAME = "dPLGR4JNet"  # 动态流程改为 dPLGR4JNetd
```

默认检查点路径为：

```text
outputs/{station_name}/dPLGR4J/best_model.pt
outputs/{station_name}/dPLGR4Jd/best_model.pt
```

如需使用其他检查点，可设置 `STATIC_DPL_CHECKPOINT` 或 `DYNAMIC_DPL_CHECKPOINT`。如果希望一次命令自动先训练缺失的基础模型，可设置：

```python
AUTO_TRAIN_PREREQUISITE = True
```

主函数会检查检查点模型类型，以及 `hidden_size`、特征顺序、暖机长度、参数聚合方式和动态参数索引等是否一致，避免把不兼容的基础权重用于 Net 模型。如果只想检查数据和前向计算，将 `DRY_RUN` 改为 `True`；但 Net 模型即使 dry run 也必须具有基础检查点。

修改训练时期、窗口长度或 `WINDOW_STRIDE` 后，原基础检查点会被判定为不兼容，需要先重新训练 `dPLGR4J`/`dPLGR4Jd`。启用 `AUTO_TRAIN_PREREQUISITE` 时，程序会自动重训缺失或不兼容的基础模型。

### 命令行入口

先检查数据和模型前向计算：

```powershell
python -m dplgr4jnet --config configs/default.yaml --dry-run
```

切换流域时，程序会自动匹配对应的 CSV 和 `camels_topo.txt` 面积：

```powershell
python -m dplgr4jnet --config configs/default.yaml `
  --station-name 01022500 `
  --model dPLGR4Jd `
  --dry-run
```

一次性检查四个模型的前向与反向传播：

```powershell
python tests/smoke_models.py
```

训练默认的静态基础模型 `dPLGR4J`：

```powershell
python -m dplgr4jnet --config configs/default.yaml
```

选择动态基础模型：

```powershell
python -m dplgr4jnet --config configs/default.yaml --model dPLGR4Jd
```

通过命令行运行 Net 模型时也必须提供基础检查点：

```powershell
python -m dplgr4jnet --config configs/default.yaml `
  --model dPLGR4JNet `
  --pretrained-dpl-path outputs/01013500/dPLGR4J/best_model.pt
```

常用覆盖参数：

```powershell
python -m dplgr4jnet --config configs/default.yaml `
  --model dPLGR4JNetd `
  --pretrained-dpl-path outputs/01013500/dPLGR4Jd/best_model.pt `
  --epochs 20 `
  --data-dir data `
  --output-dir outputs/my_run
```

动态模型通过以下配置选择逐日变化的参数，索引分别为 X1=0、X2=1、X3=2；X4 是单位线卷积核，不能逐日变化。

```yaml
model:
  name: dPLGR4JNetd
  pretrained_dpl_path: outputs/01013500/dPLGR4Jd/best_model.pt
  param_var_index: [0]
  param_test_way: mean_time
  param_smoothness_weight: 0.01
```

## 输出

每次完整运行会保存：

- `best_model.pt`：最佳验证损失对应的模型权重、配置、epoch 和损失；
- `history.csv`、`metrics_summary.csv`：训练过程和 train/valid/test 指标；
- `{split}_predictions.csv`：观测与模拟流量，同时包含 mm/day 和 m³/s；
- `{split}_parameters.csv`：逐日物理参数、实际采用的归一化参数以及 LSTM 原始生成参数；
- `{split}_predictions.xlsx`：流量、指标和参数的 Excel 汇总；
- `feature_scaler.json`、`station_meta.json`、`resolved_config.yaml`：复现实验所需信息。

静态模型的物理参数会逐日重复，便于与流量按日期关联；动态模型会保存真实的逐日参数轨迹。默认仓库已用样例流域完成一轮验证，结果位于 `outputs/sample_verified`，动态参数验证位于 `outputs/dynamic_verified`。

## 许可与来源

本项目包含从 BSD-3-Clause 许可的 `torchhydro` 改编的实现。详情见 [LICENSE](LICENSE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSES/torchhydro-BSD-3-Clause.txt](LICENSES/torchhydro-BSD-3-Clause.txt)。
