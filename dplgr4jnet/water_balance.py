"""Daily GR4J diagnostics and physics loss terms from the original experiment."""

import math

import torch
from torch import Tensor

from dplgr4jnet.model import (
    DPLGR4JDynamic,
    DPLGR4JDynamicWithMLP,
    calculate_evap_store,
    calculate_precip_store,
)

def water_balance_diagnostics(
    warmup_length: int, inputs, parameters, s_begin, s_end, r_begin, r_end,
    pr, perc, q9, q1, q_physical,
    s_norm, r_norm, f,
) -> dict[str, Tensor]:
    """核算实际执行的物理方程，并考虑地下水交换的可用水量限制。

    仅补算诊断所需的中间通量，不改动原有产流和汇流计算。
    两条支路均使用名义交换量 F，但实际流出不得超过各自可用水量。
    容量截断损失单独记录，不并入地下水交换；单位线蓄水表示尚未输出的
    在途水量，不包含已经进入汇流水库的水。
    """
    x1, _, x3, _ = parameters.unbind(-1)
    # 产流层水量核算：总实际蒸散发包含降水与 PET 抵消项以及土壤库蒸发。
    p, pet = inputs[..., 0], inputs[..., 1]
    pn = torch.clamp_min(p - pet, 0.0)
    en = torch.clamp_min(pet - p, 0.0)
    s_effective = torch.clamp(s_begin, torch.zeros_like(s_begin), x1)
    ps = calculate_precip_store(s_effective, pn, x1)
    es = calculate_evap_store(s_effective, en, x1)
    s_before_perc = s_effective - es + ps
    s_clamped = torch.clamp(s_before_perc, torch.zeros_like(s_begin), x1)
    interception = torch.minimum(p, pet)
    et_actual = interception + es
    s_clip = (s_begin - s_effective) + (s_before_perc - s_clamped)

    r_effective = torch.clamp(r_begin, torch.zeros_like(r_begin), x3)
    # 沿用实际计算得到的名义交换量 f，补算两条支路的出流及容量截断损失。
    r_before_outflow = torch.clamp_min(r_effective + q9 + f, 0.0)
    qr = r_before_outflow * (
        1.0 - (1.0 + (r_before_outflow / x3) ** 4) ** -0.25
    )
    qd = torch.clamp_min(q1 + f, 0.0)
    r_clip = r_begin - r_effective

    # 诊断累加使用 float64，减小长序列中单位线在途水量的累计舍入误差。
    # 物理模拟仍保持原数据类型，原计算中的微小水平衡残差会如实保留。
    def audit(value):
        return value.to(torch.float64)

    def previous(value):
        return torch.cat([torch.zeros_like(value[:1]), value[:-1]], dim=0)

    uh1_end = torch.cumsum(0.9 * audit(pr) - audit(q9), dim=0)
    uh2_end = torch.cumsum(0.1 * audit(pr) - audit(q1), dim=0)
    uh1_begin, uh2_begin = previous(uh1_end), previous(uh2_end)
    # 名义地下水流出受当前可用水量限制；不把浮点加法误差归入地下水交换。
    # 数值计算产生的偏差留在水平衡残差中，避免被诊断通量掩盖。
    g_routing = torch.maximum(audit(f), -(audit(r_effective) + audit(q9)))
    g_direct = torch.maximum(audit(f), -audit(q1))
    g_actual = g_routing + g_direct
    storage_begin = audit(s_begin) + audit(r_begin) + uh1_begin + uh2_begin
    storage_end = audit(s_end) + audit(r_end) + uh1_end + uh2_end
    storage_change = storage_end - storage_begin
    clip_loss = audit(s_clip) + audit(r_clip)
    production_residual = (
        audit(p) - audit(et_actual) - audit(pr)
        - (audit(s_end) - audit(s_begin)) - audit(s_clip)
    )
    routing_residual = (
        audit(q9) + audit(q1) + g_actual - audit(q_physical)
        - (audit(r_end) - audit(r_begin)) - audit(r_clip)
    )
    wb = (audit(p) - audit(et_actual) + g_actual - audit(q_physical)
          - storage_change - clip_loss)
    zero = torch.zeros_like(q_physical)
    # 依次记录驱动与通量、日初日末状态、在途水量、截断损失及闭合残差。
    # 纯 DPL 模型没有 MLP 校正，相关字段先置零；组合模型在后续前向中更新。
    fields = {
        "P_mm": p, "PET_mm": pet, "Pn_mm": pn, "En_mm": en,
        "interception_mm": interception, "Es_mm": es,
        "ET_actual_mm": et_actual, "Ps_mm": ps, "Pr_mm": pr,
        "Perc_mm": perc, "Q9_mm": q9, "Q1_mm": q1,
        "Qr_mm": qr, "Qd_mm": qd, "F_nominal_mm": f,
        "G_routing_mm": g_routing, "G_direct_mm": g_direct,
        "G_actual_mm": g_actual,
        "S_begin_mm": s_begin, "S_effective_begin_mm": s_effective,
        "S_end_mm": s_end, "S_begin_over_X1": s_effective / x1,
        "S_end_over_X1": s_norm,
        "R_begin_mm": r_begin, "R_effective_begin_mm": r_effective,
        "R_end_mm": r_end, "R_begin_over_X3": r_effective / x3,
        "R_end_over_X3": r_norm,
        "UH1_begin_mm": uh1_begin, "UH1_end_mm": uh1_end,
        "UH2_begin_mm": uh2_begin, "UH2_end_mm": uh2_end,
        "storage_begin_mm": storage_begin, "storage_end_mm": storage_end,
        "storage_change_mm": storage_change,
        "S_clip_loss_mm": s_clip, "R_clip_loss_mm": r_clip,
        "storage_clip_loss_mm": clip_loss,
        "Q_physical_mm": q_physical, "Q_sim_mm": q_physical,
        "mlp_correction_raw_mm": zero,
        "mlp_correction_effective_mm": zero,
        "Q_nonnegative_adjustment_mm": zero,
        "wb_production_residual_mm": production_residual,
        "wb_routing_residual_mm": routing_residual,
        "wb_physical_residual_mm": wb,
        "wb_physical_without_clip_mm": wb + clip_loss,
        "wb_model_residual_mm": wb,
    }
    return {
        name: value[warmup_length:].unsqueeze(-1)
        for name, value in fields.items()
    }


def gr4j_physics_loss_terms(diagnostics: dict[str, Tensor]) -> dict[str, Tensor]:
    """使用同一次前向的正式期物理量计算损失，保留完整梯度。

    在 float64 中核算闭合残差，避免水量相减时的舍入误差。物理损失
    对全部时间步及批次取均值，不因观测径流缺测而取消物理约束。
    G_actual 是两支路受可用水量限制后的净交换，不能以 F 或 2F 替代。
    clip_loss 单独核算现有容量截断，不将其隐藏为地下水交换。
    """
    def audit(name):
        return diagnostics[name].to(torch.float64)

    balance_residual = (
        audit("P_mm") + audit("G_actual_mm") - audit("ET_actual_mm")
        - audit("Q_sim_mm") - audit("storage_change_mm")
        - audit("storage_clip_loss_mm")
    )
    et_excess = torch.relu(audit("ET_actual_mm") - audit("PET_mm"))
    return {
        "water_balance": balance_residual.square().mean(),
        "evapotranspiration": et_excess.square().mean(),
    }


_GR4J_STATE_FIELDS = (
    "Q_physical_mm", "S_end_over_X1", "R_end_over_X3",
    "Pr_mm", "Perc_mm", "F_nominal_mm",
)


class _WaterBalanceLoss:
    def __init__(self, *args, lambda_q=1.0, lambda_b=0.01, lambda_e=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_q, self.lambda_b, self.lambda_e = map(float, (lambda_q, lambda_b, lambda_e))
        if not all(math.isfinite(weight) and weight >= 0 for weight in (
            self.lambda_q, self.lambda_b, self.lambda_e
        )):
            raise ValueError("lambda_q, lambda_b and lambda_e must be finite and non-negative")
        self._last_physics_losses: dict[str, Tensor] | None = None

    def compute_model_loss(self, output: Tensor, target: Tensor, criterion) -> Tensor:
        if self._last_physics_losses is None:
            raise RuntimeError("Call forward before compute_model_loss")
        return (
            self.lambda_q * criterion(output, target)
            + self.lambda_b * self._last_physics_losses["water_balance"]
            + self.lambda_e * self._last_physics_losses["evapotranspiration"]
        )

    def _prediction(self, diagnostics, pb_model, return_extras):
        self._last_physics_losses = gr4j_physics_loss_terms(diagnostics)
        prediction = diagnostics["Q_sim_mm"]
        if return_extras:
            return prediction, {
                "parameters_generated_normalized": pb_model.last_generated_normalized_parameters,
                "parameters_normalized": pb_model.last_normalized_parameters,
                "parameters_physical": pb_model.last_physical_parameters,
                "diagnostics": diagnostics,
            }
        return prediction


class DPLGR4JDynamicWaterBalance(_WaterBalanceLoss, DPLGR4JDynamic):
    """Dynamic dPL-GR4J with runoff, water-balance and ET loss terms."""

    def _diagnostics(self, x: Tensor, z: Tensor) -> dict[str, Tensor]:
        return self.pb_model(x[..., :2], self.generate_parameters(z), return_diagnostics=True)

    def forward(
        self, x: Tensor, z: Tensor, return_state: bool = False,
        return_diagnostics: bool = False, *, return_extras: bool = False,
    ):
        diagnostics = self._diagnostics(x, z)
        prediction = self._prediction(diagnostics, self.pb_model, return_extras)
        if return_diagnostics:
            return diagnostics
        if return_state:
            return tuple(diagnostics[name] for name in _GR4J_STATE_FIELDS)
        return prediction


class DPLGR4JDynamicWaterBalanceWithMLP(_WaterBalanceLoss, DPLGR4JDynamicWithMLP):
    """Water-balance dPL-GR4J with a causal residual MLP."""

    dpl_class = DPLGR4JDynamicWaterBalance

    def forward(
        self, x: Tensor, z: Tensor, return_diagnostics: bool = False,
        *, return_extras: bool = False,
    ):
        diagnostics = self.dplgr4j._diagnostics(x, z)
        base, *physical_features = (diagnostics[name] for name in _GR4J_STATE_FIELDS)
        corrected, correction = self._calibrate(base, tuple(physical_features), z)
        diagnostics.update({
            "Q_sim_mm": corrected,
            "mlp_correction_raw_mm": correction,
            "mlp_correction_effective_mm": corrected - base,
            "Q_nonnegative_adjustment_mm": corrected - (base + correction),
            "wb_model_residual_mm": (
                diagnostics["wb_physical_residual_mm"]
                - (corrected.to(torch.float64) - base.to(torch.float64))
            ),
        })
        prediction = self._prediction(diagnostics, self.dplgr4j.pb_model, return_extras)
        return diagnostics if return_diagnostics else prediction


DplLstmGr4jWaterBalance = DPLGR4JDynamicWaterBalance
DplLstmGr4jWaterBalanceWithMLP = DPLGR4JDynamicWaterBalanceWithMLP

