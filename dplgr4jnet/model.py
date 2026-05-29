from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def uh_conv(x: Tensor, uh_made: Tensor) -> Tensor:
    uh = uh_made.permute(1, 2, 0)
    inputs = x.permute(2, 1, 0)
    batch_size = x.shape[1]
    outputs = F.conv1d(
        inputs,
        torch.flip(uh, [2]),
        groups=batch_size,
        padding=uh.shape[-1] - 1,
    )
    return outputs[:, :, : x.shape[0]].permute(2, 1, 0)


def calculate_precip_store(s: Tensor, precip_net: Tensor, x1: Tensor) -> Tensor:
    numerator = x1 * (1.0 - (s / x1) ** 2) * torch.tanh(precip_net / x1)
    denominator = 1.0 + (s / x1) * torch.tanh(precip_net / x1)
    return numerator / denominator


def calculate_evap_store(s: Tensor, evap_net: Tensor, x1: Tensor) -> Tensor:
    numerator = s * (2.0 - s / x1) * torch.tanh(evap_net / x1)
    denominator = 1.0 + (1.0 - s / x1) * torch.tanh(evap_net / x1)
    return numerator / denominator


def calculate_perc(current_store: Tensor, x1: Tensor) -> Tensor:
    return current_store * (1.0 - (1.0 + (4.0 / 9.0 * current_store / x1) ** 4) ** -0.25)


def production(p_and_e: Tensor, x1: Tensor, s_level: Tensor | None = None) -> tuple[Tensor, Tensor]:
    device = p_and_e.device
    precip_difference = p_and_e[:, 0] - p_and_e[:, 1]
    precip_net = torch.maximum(precip_difference, torch.zeros_like(precip_difference, device=device))
    evap_net = torch.maximum(-precip_difference, torch.zeros_like(precip_difference, device=device))

    if s_level is None:
        s_level = 0.6 * x1.detach()

    s_level = torch.clamp(s_level, torch.zeros_like(s_level), x1)
    precip_store = calculate_precip_store(s_level, precip_net, x1)
    evap_store = calculate_evap_store(s_level, evap_net, x1)

    s_update = s_level - evap_store + precip_store
    s_update = torch.clamp(s_update, torch.zeros_like(s_update), x1)
    perc = calculate_perc(s_update, x1)
    s_update = s_update - perc
    current_runoff = perc + (precip_net - precip_store)
    return current_runoff, s_update


def uh_gr4j(x4: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    device = x4.device
    uh1_ordinates = []
    uh2_ordinates = []
    for i in range(len(x4)):
        uh1_t1 = torch.arange(0.0, torch.ceil(x4[i]).detach().cpu().item(), device=device)
        uh1_t = torch.arange(1.0, torch.ceil(x4[i] + 1.0).detach().cpu().item(), device=device)

        uh2_t1_seq = torch.arange(0.0, torch.floor(x4[i] + 1.0).detach().cpu().item(), device=device)
        uh2_t1_tail = torch.arange(
            torch.floor(x4[i] + 1.0).detach().cpu().item(),
            torch.ceil(2.0 * x4[i]).detach().cpu().item(),
            device=device,
        )
        uh2_t_seq = torch.arange(1.0, torch.floor(x4[i] + 1.0).detach().cpu().item(), device=device)
        uh2_t_tail = torch.arange(
            torch.floor(x4[i] + 1.0).detach().cpu().item(),
            torch.ceil(2.0 * x4[i] + 1.0).detach().cpu().item(),
            device=device,
        )

        s_curve1_t1 = (uh1_t1 / x4[i]) ** 2.5
        s_curve2a_t1 = 0.5 * (uh2_t1_seq / x4[i]) ** 2.5
        s_curve2b_t1 = 1.0 - 0.5 * (2.0 - uh2_t1_tail / x4[i]) ** 2.5
        s_curve2_t1 = torch.cat([s_curve2a_t1, s_curve2b_t1])

        limit_uh1 = 1.0 - F.relu(1.0 - uh1_t / x4[i])
        limit_uh2_small = uh2_t_seq / x4[i]
        limit_uh2_large = F.relu(2.0 - uh2_t_tail / x4[i])

        s_curve1_t = limit_uh1**2.5
        s_curve2a_t = 0.5 * limit_uh2_small**2.5
        s_curve2b_t = 1.0 - 0.5 * limit_uh2_large**2.5
        s_curve2_t = torch.cat([s_curve2a_t, s_curve2b_t])

        uh1_ordinates.append(s_curve1_t - s_curve1_t1)
        uh2_ordinates.append(s_curve2_t - s_curve2_t1)
    return uh1_ordinates, uh2_ordinates


def routing(q9: Tensor, q1: Tensor, x2: Tensor, x3: Tensor, r_level: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if r_level is None:
        r_level = 0.7 * x3.detach()

    r_level = torch.clamp(r_level, torch.zeros_like(r_level), x3)
    groundwater_ex = x2 * (r_level / x3) ** 3.5
    r_updated = torch.maximum(torch.zeros_like(r_level), r_level + q9 + groundwater_ex)
    qr = r_updated * (1.0 - (1.0 + (r_updated / x3) ** 4) ** -0.25)
    r_updated = r_updated - qr
    qd = torch.maximum(torch.zeros_like(groundwater_ex), q1 + groundwater_ex)
    q = qr + qd
    return q, r_updated


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x0 = F.relu(self.linear_in(x))
        out, _ = self.lstm(x0)
        out = self.linear_out(self.dropout(out))
        return out


class Gr4j4Dpl(nn.Module):
    def __init__(self, warmup_length: int) -> None:
        super().__init__()
        self.warmup_length = warmup_length
        self.feature_size = 2
        self.x1_scale = (100.0, 1200.0)
        self.x2_scale = (-5.0, 3.0)
        self.x3_scale = (20.0, 300.0)
        self.x4_scale = (1.1, 2.9)

    def scale_parameters(self, parameters: Tensor) -> Tensor:
        x1 = self.x1_scale[0] + parameters[:, 0] * (self.x1_scale[1] - self.x1_scale[0])
        x2 = self.x2_scale[0] + parameters[:, 1] * (self.x2_scale[1] - self.x2_scale[0])
        x3 = self.x3_scale[0] + parameters[:, 2] * (self.x3_scale[1] - self.x3_scale[0])
        x4 = self.x4_scale[0] + parameters[:, 3] * (self.x4_scale[1] - self.x4_scale[0])
        return torch.stack([x1, x2, x3, x4], dim=1)

    def forward(
        self,
        p_and_e: Tensor,
        parameters: Tensor,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = p_and_e.device
        scaled = self.scale_parameters(parameters)
        x1 = scaled[:, 0]
        x2 = scaled[:, 1]
        x3 = scaled[:, 2]
        x4 = scaled[:, 3]

        if self.warmup_length > 0:
            with torch.no_grad():
                p_and_e_warmup = p_and_e[: self.warmup_length, :, :]
                cal_init = Gr4j4Dpl(0).to(device)
                warmup_result = cal_init(p_and_e_warmup, parameters, return_state=True)
                _, s_norm_init, r_norm_init, _, _, _ = warmup_result
                s0 = s_norm_init[:, 0] * x1
                r0 = r_norm_init[:, 0] * x3
        else:
            s0 = 0.5 * x1.detach()
            r0 = 0.5 * x3.detach()

        inputs = p_and_e[self.warmup_length :, :, :]
        seq_len = inputs.shape[0]
        batch_size = inputs.shape[1]

        streamflow_ = torch.zeros((seq_len, batch_size), device=device)
        prs = torch.zeros((seq_len, batch_size), device=device)
        percs = torch.zeros((seq_len, batch_size), device=device)

        s = s0
        for i in range(seq_len):
            pr, s = production(inputs[i, :, :], x1, s)
            prs[i, :] = pr
            percs[i, :] = calculate_perc(s, x1)

        prs_x = prs.unsqueeze(2)
        conv_q9, conv_q1 = uh_gr4j(x4)

        q9 = torch.zeros((seq_len, batch_size, 1), device=device)
        q1 = torch.zeros((seq_len, batch_size, 1), device=device)

        for j in range(batch_size):
            q9[:, j : j + 1, :] = uh_conv(prs_x[:, j : j + 1, :], conv_q9[j].reshape(-1, 1, 1))
            q1[:, j : j + 1, :] = uh_conv(prs_x[:, j : j + 1, :], conv_q1[j].reshape(-1, 1, 1))

        gw_exchanges = torch.zeros((seq_len, batch_size), device=device)
        r = r0
        for i in range(seq_len):
            r_clamped = torch.clamp(r, torch.zeros_like(r), x3)
            gw_exchanges[i, :] = x2 * (r_clamped / x3) ** 3.5
            q, r = routing(q9[i, :, 0], q1[i, :, 0], x2, x3, r)
            streamflow_[i, :] = q

        streamflow = streamflow_.unsqueeze(2)

        if not return_state:
            return streamflow

        s_norm = (s / x1).unsqueeze(-1)
        r_norm = (r / x3).unsqueeze(-1)
        return (
            streamflow,
            s_norm,
            r_norm,
            prs.unsqueeze(-1),
            percs.unsqueeze(-1),
            gw_exchanges.unsqueeze(-1),
        )


class DplLstmGr4j(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        n_hidden_states: int,
        warmup_length: int,
        param_limit_func: str = "sigmoid",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_input_features = n_input_features
        self.param_limit_func = param_limit_func
        self.dl_model = SimpleLSTM(
            input_size=n_input_features,
            output_size=n_output_features,
            hidden_size=n_hidden_states,
            dropout=dropout,
        )
        self.pb_model = Gr4j4Dpl(warmup_length)

    def generate_parameters(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z_dynamic = z[:, :, : self.n_input_features]
        generated = self.dl_model(z_dynamic)
        if self.param_limit_func == "sigmoid":
            params_sequence = torch.sigmoid(generated)
        elif self.param_limit_func == "clamp":
            params_sequence = torch.clamp(generated, 0.0, 1.0)
        else:
            raise NotImplementedError(f"Unsupported param_limit_func: {self.param_limit_func}")
        params = params_sequence[-1, :, :]
        return params, params_sequence

    def forward(
        self,
        x: Tensor,
        z: Tensor,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        params, _ = self.generate_parameters(z)
        return self.pb_model(
            x[:, :, : self.pb_model.feature_size],
            params,
            return_state=return_state,
        )


class CalibrationMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64, dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.residual_proj = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        residual = x[:, :1]
        return out + residual


class DplLstmGr4jWithMLP(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        n_hidden_states: int,
        warmup_length: int,
        calibration_hidden_dim: int = 64,
        calibration_output_dim: int = 1,
        calibration_dropout: float = 0.4,
        param_limit_func: str = "sigmoid",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.warmup_length = warmup_length
        self.dynamic_feature_count = n_input_features
        self.dplgr4j = DplLstmGr4j(
            n_input_features=n_input_features,
            n_output_features=n_output_features,
            n_hidden_states=n_hidden_states,
            warmup_length=warmup_length,
            param_limit_func=param_limit_func,
            dropout=dropout,
        )
        calibration_input_dim = 6 + n_input_features
        self.calibrator = CalibrationMLP(
            input_size=calibration_input_dim,
            output_size=calibration_output_dim,
            hidden_size=calibration_hidden_dim,
            dropout=calibration_dropout,
        )

    def forward(
        self,
        x: Tensor,
        z: Tensor,
        return_extras: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        z_dynamic = z[:, :, : self.dynamic_feature_count]
        params_normalized, _ = self.dplgr4j.generate_parameters(z_dynamic)
        params_physical = self.dplgr4j.pb_model.scale_parameters(params_normalized)

        gr4j_result = self.dplgr4j.pb_model(
            x[:, :, : self.dplgr4j.pb_model.feature_size],
            params_normalized,
            return_state=True,
        )
        streamflow, s_norm, r_norm, prs, perc, gw_ex = gr4j_result

        seq_len = streamflow.shape[0]
        batch_size = streamflow.shape[1]
        s_expand = s_norm.unsqueeze(0).expand(seq_len, batch_size, 1)
        r_expand = r_norm.unsqueeze(0).expand(seq_len, batch_size, 1)
        z_dynamic_aligned = z_dynamic[self.warmup_length :, :, :]

        mlp_in = torch.cat(
            [streamflow, s_expand, r_expand, prs, perc, gw_ex, z_dynamic_aligned],
            dim=-1,
        )
        feat = mlp_in.shape[-1]
        mlp_out = self.calibrator(mlp_in.reshape(-1, feat)).reshape(seq_len, batch_size, -1)

        if not return_extras:
            return mlp_out

        return mlp_out, {
            "base_streamflow": streamflow,
            "parameters_normalized": params_normalized,
            "parameters_physical": params_physical,
            "S_norm": s_norm,
            "R_norm": r_norm,
            "prs": prs,
            "perc": perc,
            "gw_ex": gw_ex,
        }


class DPLGR4JNet(DplLstmGr4jWithMLP):
    def __init__(
        self,
        n_input_features: int,
        n_hidden_states: int,
        warmup_length: int,
        param_limit_func: str = "clamp",
        dropout: float = 0.0,
        calibration_hidden_size: int = 64,
        calibration_dropout: float = 0.4,
    ) -> None:
        super().__init__(
            n_input_features=n_input_features,
            n_output_features=4,
            n_hidden_states=n_hidden_states,
            warmup_length=warmup_length,
            calibration_hidden_dim=calibration_hidden_size,
            calibration_output_dim=1,
            calibration_dropout=calibration_dropout,
            param_limit_func=param_limit_func,
            dropout=dropout,
        )
