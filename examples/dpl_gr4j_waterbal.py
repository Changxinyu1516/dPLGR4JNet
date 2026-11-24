"""两河流域的 dPL-GR4JNet 三项损失实验入口，依赖自定义 torchhydro 环境。

从项目根目录运行：python examples/dpl_gr4j_waterbal.py
企业电脑需配置 hydro_setting.yml，并核对 PRETRAINED_DPL_PATH。
本示例保留原动态 GR4J + 历史 MLP 计算，只更换独立模型名称及损失配置。

当前默认是第二阶段：加载已经训练好的 DPL 最佳权重，再联合微调 DPL 和训练 MLP。
如果你已有第一阶段的 best_model.pth，设置其路径后直接运行，保留 MLP 参数。
若要用本脚本运行第一阶段的纯 DPL，按下面五处注释切换：
1. 将 RESULT_SUBDIR 改为纯 DPL 专用目录，避免两阶段结果互相覆盖。
2. 将 PRETRAINED_DPL_PATH 设为 None，避免检查第二阶段才需要的旧权重文件。
3. 将 model_name 改为 DplLstmGr4jWaterBalance。
4. 注释整个“B. 仅 DPL+MLP 使用”参数组；保留 A 和 C 两组。
5. 将 opt_param 改为纯 DPL 的单学习率配置，去掉 dpl_lr、mlp_lr。
第一阶段结束后，再切回第二阶段，并把最佳权重路径填到 PRETRAINED_DPL_PATH。

运行结束后自动读取本次试验的 best_model.pth，连续推理训练期与测试期，
在同一个 Excel 中导出 X1~X4 参数、蓄水状态、蒸散发、各通量及水量平衡诊断。
如果已经训练完成，只需要补导出，将 EXPORT_ONLY 改为 True，无需重新训练。
"""

import os

from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro.trainers.gr4j_state_export import export_dpl_gr4j_states


EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
# 【切换位置 1】当前是第二阶段输出目录；第一阶段请使用下面的独立目录。
RESULT_SUBDIR = os.path.join("results", "dpl_gr4j_waterbal_sxresult")
# RESULT_SUBDIR = os.path.join("results", "dpl_gr4j_waterbal_dpl_sxresult")
# False：训练、评价后导出；True：只读取上方试验目录已有的 best_model.pth 导出。
# 只导出时仍需保留与已有权重匹配的模型结构、输入及时期配置。
EXPORT_ONLY = False

# 【切换位置 2】第二阶段：填第一阶段纯 DPL 训练所得的 best_model.pth。
# 这是待加载的权重路径；本脚本不会自动先训练一次纯 DPL。
PRETRAINED_DPL_PATH = os.path.join(
    EXAMPLE_DIR, "results", "EMS Paper Result", "dPLGR4JNet_sxresult_汇总结果",
    "dplgr4jd", "两河动态参数", "x2动态", "best_model.pth",
)
# 第一阶段纯 DPL：注释上面的路径赋值，启用下一行。
# PRETRAINED_DPL_PATH = None
# 注意：仅设置 None 而不更换 model_name，仍是从头训练 DPL+MLP，并不会切成纯 DPL。


def main():
    if not EXPORT_ONLY and PRETRAINED_DPL_PATH is not None and not os.path.isfile(PRETRAINED_DPL_PATH):
        raise FileNotFoundError(
            f"请设置企业电脑的预训练权重路径，或设为 None 从头训练：{PRETRAINED_DPL_PATH}"
        )
    args = cmd(
        project_dir=EXAMPLE_DIR,
        sub=RESULT_SUBDIR,
        source_cfgs={
            "source_name": "SXdata",
            "source_path": SETTING["local_data_path"]["datasets-origin"],
            "download": False,
        },
        ctx=[0],
        # 【切换位置 3】当前启用第二阶段 DPL+MLP；两行只能保留一行。
        model_name="DplLstmGr4jWaterBalanceWithMLP",
        # model_name="DplLstmGr4jWaterBalance",  # 第一阶段纯 DPL。
        model_hyperparam={
            # ===== A. DPL 共用参数：纯 DPL 和 DPL+MLP 两阶段都保留 =====
            # 第二阶段须与预训练 DPL 的结构及输入变量顺序保持一致。
            "n_input_features": 5,  # LSTM 输入特征数，与下方 var_t 对应。
            "n_output_features": 4,  # LSTM 生成 X1~X4 四个参数，不是径流输出数。
            "n_hidden_states": 32,  # LSTM 隐藏层维度，加载权重时必须匹配。
            "warmup_length": 0,  # GR4J 预热长度，与下方数据集 warmup_length 一致。
            "param_limit_func": "sigmoid",  # 将 LSTM 输出映射到归一化参数范围。
            "param_test_way": "final",  # 非动态参数取末时刻；也可用 mean_time。
            "param_var_index": [1],  # 仅 X2 动态；X4 必须静态。
            "param_smoothness_weight": 0.0,  # 旧的 X1/X3 时间平滑惩罚，三项损失时设 0。

            # ===== B. 仅 DPL+MLP 使用：【切换位置 4】纯 DPL 时整组注释 =====
            # 从 n_dynamic_features 到 correction_penalty_weight，纯 DPL 构造函数
            # 不接受这些参数。当前第二阶段应全部保留，即使其数值为 0。
            "n_dynamic_features": 5,  # 传入 MLP 的动态气象特征数。
            "calibration_history_length": 5,  # MLP 使用当天及过去 4 天的特征。
            "calibration_hidden_dim": [64, 32],  # MLP 两个隐藏层的神经元数。
            "calibration_dropout": 0.15,  # MLP 训练时的 Dropout 比例。
            "zero_init_mlp": True,  # 输出层零初始化，初始校正为 0，承接 DPL 输出。
            "transform_physical_features": True,  # 对 MLP 输入的物理特征做变换。
            "enforce_nonnegative": True,  # 评估/推理时将最终径流截断为非负。
            "pretrained_dpl_path": PRETRAINED_DPL_PATH,  # 加载第一阶段最佳 DPL 权重。
            # 兼容标记，不冻结 DPL；初始校正为零由 zero_init_mlp 保证。
            "preserve_pretrained_baseline": True,
            "dpl_anchor_weight": 0.0,  # 旧的 DPL 权重锚定惩罚，三项损失时设 0。
            "correction_penalty_weight": 0.0,  # 旧的 MLP 幅度惩罚，设 0 避免重复计罚。

            # ===== C. 新损失共用参数：两阶段均支持，不随 MLP 参数一起注释 =====
            # L = lambda_q*NSELoss + lambda_b*L_balance + lambda_e*L_ET。
            # 0.01 是可调整的起点，未做数据集寻优；ET 项正常情况下通常为 0。
            "lambda_q": 1.0,  # 径流拟合损失系数。
            "lambda_b": 0.1,  # 含实际地下水交换、总蓄水变化的水量平衡损失系数。
            "lambda_e": 0.05,  # 实际蒸散发超过 PET 的物理惩罚系数。
        },
        dataset="DplDataset",
        loss_func="NSELoss",  # 原稳定版 NSELoss；三项加权由新模型接口完成。
        loss_param={},
        scaler="DapengScaler",
        sampler="BasinBatchSampler",
        scaler_params={
            "prcp_norm_cols": [],
            "gamma_norm_cols": [
                StandardVariable.PRECIPITATION_CHM,
                StandardVariable.EVAPOTRANSPIRATION,
            ],
            "pbm_norm": True,
        },
        gage_id=["两河"],
        train_period=["2010-01-01", "2019-12-31"],
        valid_period=["2020-01-01", "2023-12-31"],
        test_period=["2020-01-01", "2023-12-31"],
        batch_size=4,  # batchsize设置：巫溪8,8, 兴山1,8, 两河8,4
        forecast_length=180,
        warmup_length=0,
        var_t=[
            StandardVariable.PRECIPITATION_CHM,
            StandardVariable.EVAPOTRANSPIRATION,
            StandardVariable.TEMPERATURE_MEAN,
            StandardVariable.RELATIVE_HUMIDITY,
            StandardVariable.SOLAR_RADIATION,
        ],
        var_out=[StandardVariable.STREAMFLOW],
        n_output=1,
        train_epoch=15,
        early_stopping=True,
        patience=8,
        valid_batch_mode="test",
        model_loader={"load_way": "best", "train_eval_use_rolling": False},
        opt="Adam",
        # 【切换位置 5】当前第二阶段：DPL 小学习率微调，MLP 较大学习率训练。
        # 两个分支都会更新；这不是“冻结 DPL、只训练 MLP”。
        # 纯 DPL 时注释当前 opt_param 整行，启用下面的单学习率配置。
        opt_param={"lr": 1e-4, "dpl_lr": 5e-7, "mlp_lr": 1e-4},
        # opt_param={"lr": 1e-4},  # 第一阶段纯 DPL，不传 dpl_lr 或 mlp_lr。
        lr_scheduler={"lr_factor": 0.5, "lr_patience": 1},
        save_epoch=1,
        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "R2", "KGE", "FHV", "FLV"],
    )
    cfgs = default_config_file()
    update_cfg(cfgs, args)
    cfgs["evaluation_cfgs"]["eval_unit"] = "m3/s"
    # 本脚本在下方显式调用新模型导出；关闭其他自动导出入口，避免重复或误用旧适配器。
    cfgs["evaluation_cfgs"]["export_gr4j_states"] = False
    cfgs["evaluation_cfgs"]["export_states"] = False
    if not EXPORT_ONLY:
        train_and_evaluate(cfgs)
    # 只加载当前试验的最佳完整权重，不再依赖第一阶段的预训练文件，也不会再次训练。
    # 按需把 periods 改为 ("train", "valid", "test")，需同时配置对应时期。
    export_dpl_gr4j_states(cfgs, periods=("train", "test"))
    return cfgs


if __name__ == "__main__":
    main()
