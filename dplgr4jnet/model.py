from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PARAMETER_NAMES = ("x1", "x2", "x3", "x4")
PARAMETER_SCALES = ((20.0, 1500.0), (-10.0, 5.0), (20.0, 500.0), (0.5, 15.0))


def uh_conv(x: Tensor, uh_made: Tensor) -> Tensor:
    """Apply one unit-hydrograph kernel per basin in a batch."""
    uh = uh_made.permute(1, 2, 0)
    inputs = x.permute(2, 1, 0)
    outputs = F.conv1d(
        inputs,
        torch.flip(uh, [2]),
        groups=x.shape[1],
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
    return current_store * (
        1.0 - (1.0 + (4.0 / 9.0 * current_store / x1) ** 4) ** -0.25
    )


def production(
    p_and_e: Tensor,
    x1: Tensor,
    s_level: Optional[Tensor] = None,
    return_perc: bool = False,
):
    precip_difference = p_and_e[:, 0] - p_and_e[:, 1]
    precip_net = torch.clamp_min(precip_difference, 0.0)
    evap_net = torch.clamp_min(-precip_difference, 0.0)
    if s_level is None:
        s_level = 0.6 * x1.detach()
    s_level = torch.clamp(s_level, torch.zeros_like(s_level), x1)
    precip_store = calculate_precip_store(s_level, precip_net, x1)
    evap_store = calculate_evap_store(s_level, evap_net, x1)
    s_update = torch.clamp(
        s_level - evap_store + precip_store, torch.zeros_like(s_level), x1
    )
    perc = calculate_perc(s_update, x1)
    s_update = s_update - perc
    runoff = perc + precip_net - precip_store
    if return_perc:
        return runoff, s_update, perc
    return runoff, s_update


def uh_gr4j(x4: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    device = x4.device
    uh1_ordinates: list[Tensor] = []
    uh2_ordinates: list[Tensor] = []
    for value in x4:
        uh1_t0 = torch.arange(0.0, torch.ceil(value).detach().item(), device=device)
        uh1_t1 = torch.arange(1.0, torch.ceil(value + 1.0).detach().item(), device=device)
        uh2_t0a = torch.arange(0.0, torch.floor(value + 1.0).detach().item(), device=device)
        uh2_t0b = torch.arange(
            torch.floor(value + 1.0).detach().item(),
            torch.ceil(2.0 * value).detach().item(),
            device=device,
        )
        uh2_t1a = torch.arange(1.0, torch.floor(value + 1.0).detach().item(), device=device)
        uh2_t1b = torch.arange(
            torch.floor(value + 1.0).detach().item(),
            torch.ceil(2.0 * value + 1.0).detach().item(),
            device=device,
        )
        curve1_t0 = (uh1_t0 / value) ** 2.5
        curve2_t0 = torch.cat(
            [0.5 * (uh2_t0a / value) ** 2.5, 1.0 - 0.5 * (2.0 - uh2_t0b / value) ** 2.5]
        )
        curve1_t1 = (1.0 - F.relu(1.0 - uh1_t1 / value)) ** 2.5
        curve2_t1 = torch.cat(
            [0.5 * (uh2_t1a / value) ** 2.5, 1.0 - 0.5 * F.relu(2.0 - uh2_t1b / value) ** 2.5]
        )
        uh1_ordinates.append(curve1_t1 - curve1_t0)
        uh2_ordinates.append(curve2_t1 - curve2_t0)
    return uh1_ordinates, uh2_ordinates


def routing(
    q9: Tensor,
    q1: Tensor,
    x2: Tensor,
    x3: Tensor,
    r_level: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    if r_level is None:
        r_level = 0.7 * x3.detach()
    r_level = torch.clamp(r_level, torch.zeros_like(r_level), x3)
    groundwater_exchange = x2 * (r_level / x3) ** 3.5
    r_updated = torch.clamp_min(r_level + q9 + groundwater_exchange, 0.0)
    qr = r_updated * (1.0 - (1.0 + (r_updated / x3) ** 4) ** -0.25)
    r_updated = r_updated - qr
    qd = torch.clamp_min(q1 + groundwater_exchange, 0.0)
    return qr + qd, r_updated


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.linear_in(x))
        x, _ = self.lstm(x)
        return self.linear_out(self.dropout(x))


def _limit_parameters(generated: Tensor, method: str) -> Tensor:
    if method not in {"sigmoid", "clamp"}:
        raise ValueError("param_limit_func must be 'sigmoid' or 'clamp'")
    # Upstream historically maps both accepted settings through sigmoid.
    return torch.sigmoid(generated)


def _reduce_parameters(parameters: Tensor, method: str) -> Tensor:
    if method == "final":
        return parameters[-1]
    if method == "mean_time":
        return parameters.mean(dim=0)
    if method == "mean_basin":
        return parameters[-1].mean(dim=0, keepdim=True).expand(parameters.shape[1], -1)
    raise ValueError("param_test_way must be 'final', 'mean_time', or 'mean_basin'")


def _scale_parameters(parameters: Tensor) -> Tensor:
    values = [low + parameters[..., i] * (high - low) for i, (low, high) in enumerate(PARAMETER_SCALES)]
    return torch.stack(values, dim=-1)


class Gr4j4Dpl(nn.Module):
    """GR4J with one static parameter set per batch member."""

    feature_size = 2

    def __init__(self, warmup_length: int):
        super().__init__()
        self.warmup_length = int(warmup_length)
        self.last_generated_normalized_parameters: Optional[Tensor] = None
        self.last_normalized_parameters: Optional[Tensor] = None
        self.last_physical_parameters: Optional[Tensor] = None

    def scale_parameters(self, parameters: Tensor) -> Tensor:
        return _scale_parameters(parameters)

    def forward(self, p_and_e: Tensor, parameters: Tensor, return_state: bool = False):
        if p_and_e.ndim != 3 or p_and_e.shape[-1] < 2:
            raise ValueError("p_and_e must have shape [time, batch, at least 2]")
        seq_len, batch_size = p_and_e.shape[:2]
        if not 0 <= self.warmup_length < seq_len:
            raise ValueError("warmup_length must satisfy 0 <= warmup_length < sequence length")
        physical = self.scale_parameters(parameters)
        self.last_normalized_parameters = parameters.detach()
        self.last_physical_parameters = physical.detach()
        x1, x2, x3, x4 = physical.unbind(dim=-1)
        q_full = p_and_e.new_zeros((seq_len, batch_size))
        pr_full = p_and_e.new_zeros((seq_len, batch_size))
        perc_full = p_and_e.new_zeros((seq_len, batch_size))
        s_full = p_and_e.new_zeros((seq_len, batch_size))
        r_full = p_and_e.new_zeros((seq_len, batch_size))
        exchange_full = p_and_e.new_zeros((seq_len, batch_size))
        s = 0.5 * x1.detach()
        for i in range(seq_len):
            pr, s, perc = production(p_and_e[i, :, :2], x1, s, return_perc=True)
            pr_full[i], perc_full[i], s_full[i] = pr, perc, s / x1
        uh1, uh2 = uh_gr4j(x4)
        q9 = p_and_e.new_zeros((seq_len, batch_size, 1))
        q1 = p_and_e.new_zeros((seq_len, batch_size, 1))
        for j in range(batch_size):
            source = pr_full[:, j : j + 1].unsqueeze(-1)
            q9[:, j : j + 1] = 0.9 * uh_conv(source, uh1[j].reshape(-1, 1, 1))
            q1[:, j : j + 1] = 0.1 * uh_conv(source, uh2[j].reshape(-1, 1, 1))
        r = 0.5 * x3.detach()
        for i in range(seq_len):
            r_clamped = torch.clamp(r, torch.zeros_like(r), x3)
            exchange_full[i] = x2 * (r_clamped / x3) ** 3.5
            q, r = routing(q9[i, :, 0], q1[i, :, 0], x2, x3, r)
            q_full[i], r_full[i] = q, r / x3
        outputs = tuple(v[self.warmup_length :].unsqueeze(-1) for v in (q_full, s_full, r_full, pr_full, perc_full, exchange_full))
        return outputs if return_state else outputs[0]


class Gr4j4DplWithDynamic(nn.Module):
    """GR4J with selected time-varying X1/X2/X3 parameters; X4 stays static."""

    feature_size = 2

    def __init__(self, warmup_length: int, param_var_index=None, param_test_way: str = "mean_time"):
        super().__init__()
        self.warmup_length = int(warmup_length)
        self.param_var_index = self._validate_indices(param_var_index)
        if param_test_way not in {"final", "mean_time"}:
            raise ValueError("dynamic param_test_way must be 'final' or 'mean_time'")
        self.param_test_way = param_test_way
        self.last_generated_normalized_parameters: Optional[Tensor] = None
        self.last_normalized_parameters: Optional[Tensor] = None
        self.last_physical_parameters: Optional[Tensor] = None
        self._parameters_for_smoothness: Optional[Tensor] = None

    @staticmethod
    def _validate_indices(indices) -> tuple[int, ...]:
        indices = [0] if indices is None else indices
        result = tuple(sorted(set(int(i) for i in indices)))
        if set(result).difference({0, 1, 2, 3}):
            raise ValueError("param_var_index only accepts 0 (X1), 1 (X2), 2 (X3), 3 (X4)")
        if 3 in result:
            raise ValueError("X4 cannot vary through time because it defines the sequence convolution kernel")
        return result

    def _select_effective_parameters(self, parameters: Tensor) -> Tensor:
        if parameters.ndim != 3 or parameters.shape[-1] != 4:
            raise ValueError("dynamic parameters must have shape [time, batch, 4]")
        effective = []
        for index in range(4):
            value = parameters[..., index]
            if index not in self.param_var_index:
                value = value[-1:] if self.param_test_way == "final" else value.mean(dim=0, keepdim=True)
                value = value.expand(parameters.shape[0], -1)
            effective.append(value)
        return torch.stack(effective, dim=-1)

    def parameter_smoothness_loss(self) -> Tensor:
        parameters = self._parameters_for_smoothness
        indices = [i for i in self.param_var_index if i in {0, 2}]
        if parameters is None or parameters.shape[0] < 2 or not indices:
            return parameters.new_zeros(()) if parameters is not None else torch.zeros(())
        selected = parameters[..., indices]
        return torch.mean((selected[1:] - selected[:-1]) ** 2)

    def forward(self, p_and_e: Tensor, parameters: Tensor, return_state: bool = False):
        if p_and_e.shape[:2] != parameters.shape[:2]:
            raise ValueError("p_and_e and dynamic parameters must share time and batch dimensions")
        seq_len, batch_size = p_and_e.shape[:2]
        if not 0 <= self.warmup_length < seq_len:
            raise ValueError("warmup_length must satisfy 0 <= warmup_length < sequence length")
        effective = self._select_effective_parameters(parameters)
        physical = _scale_parameters(effective)
        self.last_generated_normalized_parameters = parameters.detach()
        self.last_normalized_parameters = effective.detach()
        self.last_physical_parameters = physical.detach()
        self._parameters_for_smoothness = parameters
        x1, x2, x3 = (physical[..., i] for i in range(3))
        x4 = physical[0, :, 3]
        q_full = p_and_e.new_zeros((seq_len, batch_size))
        pr_full = p_and_e.new_zeros((seq_len, batch_size))
        perc_full = p_and_e.new_zeros((seq_len, batch_size))
        s_full = p_and_e.new_zeros((seq_len, batch_size))
        r_full = p_and_e.new_zeros((seq_len, batch_size))
        exchange_full = p_and_e.new_zeros((seq_len, batch_size))
        s = 0.5 * x1[0].detach()
        for i in range(seq_len):
            pr, s, perc = production(p_and_e[i, :, :2], x1[i], s, return_perc=True)
            pr_full[i], perc_full[i], s_full[i] = pr, perc, s / x1[i]
        uh1, uh2 = uh_gr4j(x4)
        q9 = p_and_e.new_zeros((seq_len, batch_size, 1))
        q1 = p_and_e.new_zeros((seq_len, batch_size, 1))
        for j in range(batch_size):
            source = pr_full[:, j : j + 1].unsqueeze(-1)
            q9[:, j : j + 1] = 0.9 * uh_conv(source, uh1[j].reshape(-1, 1, 1))
            q1[:, j : j + 1] = 0.1 * uh_conv(source, uh2[j].reshape(-1, 1, 1))
        r = 0.5 * x3[0].detach()
        for i in range(seq_len):
            r_clamped = torch.clamp(r, torch.zeros_like(r), x3[i])
            exchange_full[i] = x2[i] * (r_clamped / x3[i]) ** 3.5
            q, r = routing(q9[i, :, 0], q1[i, :, 0], x2[i], x3[i], r)
            q_full[i], r_full[i] = q, r / x3[i]
        outputs = tuple(v[self.warmup_length :].unsqueeze(-1) for v in (q_full, s_full, r_full, pr_full, perc_full, exchange_full))
        return outputs if return_state else outputs[0]


def _parameter_extras(pb_model: nn.Module, states: tuple[Tensor, ...]) -> dict[str, Any]:
    q, s, r, pr, perc, exchange = states
    return {
        "base_streamflow": q,
        "parameters_generated_normalized": pb_model.last_generated_normalized_parameters,
        "parameters_normalized": pb_model.last_normalized_parameters,
        "parameters_physical": pb_model.last_physical_parameters,
        "S_norm": s,
        "R_norm": r,
        "prs": pr,
        "perc": perc,
        "gw_ex": exchange,
    }


class _DplBase(nn.Module):
    pb_model_class = Gr4j4Dpl

    def __init__(self, n_input_features: int, n_output_features: int, n_hidden_states: int, warmup_length: int, param_limit_func: str = "sigmoid", param_test_way: str = "mean_time", dropout: float = 0.0, **pb_kwargs):
        super().__init__()
        if n_output_features != 4:
            raise ValueError("GR4J requires n_output_features=4")
        self.n_input_features = int(n_input_features)
        self.param_func = param_limit_func
        self.param_test_way = param_test_way
        self.dl_model = SimpleLSTM(n_input_features, 4, n_hidden_states, dropout)
        if self.pb_model_class is Gr4j4DplWithDynamic:
            pb_kwargs["param_test_way"] = param_test_way
        self.pb_model = self.pb_model_class(warmup_length, **pb_kwargs)

    def generate_parameters(self, z: Tensor) -> Tensor:
        if z.ndim != 3 or z.shape[-1] < self.n_input_features:
            raise ValueError("LSTM input must be [time, batch, feature] with enough features")
        generated = _limit_parameters(self.dl_model(z[..., : self.n_input_features]), self.param_func)
        self.pb_model.last_generated_normalized_parameters = generated.detach()
        return generated

    def regularization_loss(self) -> Tensor:
        return next(self.parameters()).new_zeros(())


class DPLGR4J(_DplBase):
    """Static-parameter dPL-GR4J."""

    def forward(self, x: Tensor, z: Tensor, return_extras: bool = False):
        generated = self.generate_parameters(z)
        parameters = _reduce_parameters(generated, self.param_test_way)
        states = self.pb_model(x[..., :2], parameters, return_state=True)
        return (states[0], _parameter_extras(self.pb_model, states)) if return_extras else states[0]


class DPLGR4Jd(_DplBase):
    """Dynamic-parameter dPL-GR4J."""

    pb_model_class = Gr4j4DplWithDynamic

    def __init__(self, *args, param_var_index=None, param_smoothness_weight: float = 0.0, **kwargs):
        self.param_smoothness_weight = float(param_smoothness_weight)
        if self.param_smoothness_weight < 0:
            raise ValueError("param_smoothness_weight must be non-negative")
        param_test_way = kwargs.pop("param_test_way", "mean_time")
        super().__init__(*args, **kwargs, param_var_index=param_var_index, param_test_way=param_test_way)

    def forward(self, x: Tensor, z: Tensor, return_extras: bool = False):
        states = self.pb_model(x[..., :2], self.generate_parameters(z), return_state=True)
        return (states[0], _parameter_extras(self.pb_model, states)) if return_extras else states[0]

    def regularization_loss(self) -> Tensor:
        return self.param_smoothness_weight * self.pb_model.parameter_smoothness_loss().to(next(self.parameters()).device)


class CalibrationMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int | Sequence[int] = (64, 32), dropout: float = 0.05, history_length: int = 1, zero_init_output: bool = True, transform_physical_features: bool = True):
        super().__init__()
        hidden_dims = (hidden_size, hidden_size) if isinstance(hidden_size, int) else tuple(hidden_size)
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("calibration_hidden_dim must contain positive integers")
        self.feature_size = input_size
        self.history_length = history_length
        self.transform_physical_features = transform_physical_features
        layers: list[nn.Module] = []
        previous = input_size * history_length
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        self.output_layer = nn.Linear(previous, output_size)
        layers.append(self.output_layer)
        self.net = nn.Sequential(*layers)
        if zero_init_output:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def _prepare(self, x: Tensor) -> Tensor:
        if not self.transform_physical_features:
            return x
        q, s, r, pr, perc, exchange = torch.split(x[..., :6], 1, dim=-1)
        physical = torch.cat([torch.log1p(torch.clamp_min(q, 0)), s, r, torch.log1p(torch.clamp_min(pr, 0)), torch.log1p(torch.clamp_min(perc, 0)), torch.sign(exchange) * torch.log1p(torch.abs(exchange))], dim=-1)
        return torch.cat([physical, x[..., 6:]], dim=-1)

    def forward(self, x: Tensor, streamflow: Optional[Tensor] = None, return_correction: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        expected = (self.history_length, self.feature_size)
        if x.ndim != 3 or x.shape[1:] != expected:
            raise ValueError(f"calibrator input must have shape [sample, {expected[0]}, {expected[1]}]")
        correction = self.net(self._prepare(x).flatten(start_dim=1))
        base = x[:, -1, :1] if streamflow is None else streamflow
        corrected = base + correction
        return (corrected, correction) if return_correction else corrected


class _DplNetBase(nn.Module):
    dpl_class = DPLGR4J

    def __init__(self, n_input_features: int, n_output_features: int, n_hidden_states: int, warmup_length: int, n_dynamic_features: Optional[int] = None, calibration_hidden_dim: int | Sequence[int] = (64, 32), calibration_output_dim: int = 1, calibration_dropout: float = 0.05, calibration_history_length: int = 1, enforce_nonnegative: bool = True, zero_init_mlp: bool = True, transform_physical_features: bool = True, pretrained_dpl_path=None, preserve_pretrained_baseline: bool = True, dpl_anchor_weight: float = 0.0, correction_penalty_weight: float = 0.0, **dpl_kwargs):
        super().__init__()
        if calibration_output_dim != 1:
            raise ValueError("Only one streamflow output is supported")
        self.warmup_length = int(warmup_length)
        self.n_input_features = int(n_input_features)
        self.n_dynamic_features = self.n_input_features if n_dynamic_features is None else int(n_dynamic_features)
        self.calibration_history_length = int(calibration_history_length)
        self.enforce_nonnegative = bool(enforce_nonnegative)
        self.dpl_anchor_weight = float(dpl_anchor_weight)
        self.correction_penalty_weight = float(correction_penalty_weight)
        self._last_correction: Optional[Tensor] = None
        self.dplgr4j = self.dpl_class(n_input_features=n_input_features, n_output_features=n_output_features, n_hidden_states=n_hidden_states, warmup_length=warmup_length, **dpl_kwargs)
        if pretrained_dpl_path:
            self._load_pretrained_dpl(pretrained_dpl_path)
        self.preserve_pretrained_baseline = bool(preserve_pretrained_baseline and pretrained_dpl_path)
        self._anchor_names: dict[str, str] = {}
        if self.dpl_anchor_weight > 0:
            for index, (name, parameter) in enumerate(self.dplgr4j.named_parameters()):
                buffer_name = f"_dpl_anchor_{index}"
                self.register_buffer(buffer_name, parameter.detach().clone(), persistent=False)
                self._anchor_names[name] = buffer_name
        self.calibrator = CalibrationMLP(6 + self.n_dynamic_features, calibration_output_dim, calibration_hidden_dim, calibration_dropout, self.calibration_history_length, zero_init_mlp, transform_physical_features)

    def _load_pretrained_dpl(self, checkpoint_path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        if any(key.startswith("dplgr4j.") for key in state):
            state = {key.removeprefix("dplgr4j."): value for key, value in state.items() if key.startswith("dplgr4j.")}
        self.dplgr4j.load_state_dict(state, strict=True)

    def _history(self, features: Tensor) -> Tensor:
        sequences = []
        for lag in range(self.calibration_history_length - 1, -1, -1):
            if lag == 0:
                shifted = features
            elif lag >= len(features):
                shifted = features[:1].expand(len(features), -1, -1)
            else:
                shifted = torch.cat([features[:1].expand(lag, -1, -1), features[:-lag]], dim=0)
            sequences.append(shifted)
        return torch.stack(sequences, dim=2)

    def forward(self, x: Tensor, z: Tensor, return_extras: bool = False):
        _, extras = self.dplgr4j(x, z, return_extras=True)
        base = extras["base_streamflow"]
        dynamic = z[self.warmup_length :, :, : self.n_dynamic_features]
        features = torch.cat([base, extras["S_norm"], extras["R_norm"], extras["prs"], extras["perc"], extras["gw_ex"], dynamic], dim=-1)
        history = self._history(features).reshape(-1, self.calibration_history_length, features.shape[-1])
        corrected, correction = self.calibrator(history, base.reshape(-1, 1), True)
        self._last_correction = correction
        prediction = corrected.view_as(base)
        if self.enforce_nonnegative and not self.training:
            prediction = torch.clamp_min(prediction, 0.0)
        extras["correction"] = correction.view_as(base)
        return (prediction, extras) if return_extras else prediction

    def optimizer_parameter_groups(self, default_lr: float, dpl_lr=None, mlp_lr=None):
        return [{"params": self.dplgr4j.parameters(), "lr": float(default_lr if dpl_lr is None else dpl_lr)}, {"params": self.calibrator.parameters(), "lr": float(default_lr if mlp_lr is None else mlp_lr)}]

    def regularization_loss(self) -> Tensor:
        loss = self.dplgr4j.regularization_loss()
        if self.dpl_anchor_weight > 0:
            distances = [(parameter - getattr(self, self._anchor_names[name])).square().mean() for name, parameter in self.dplgr4j.named_parameters()]
            loss = loss + self.dpl_anchor_weight * torch.stack(distances).mean()
        if self.correction_penalty_weight > 0 and self._last_correction is not None:
            loss = loss + self.correction_penalty_weight * self._last_correction.square().mean()
        return loss


class DPLGR4JNet(_DplNetBase):
    """Static dPL-GR4J with a causal residual MLP."""


class DPLGR4JNetd(_DplNetBase):
    """Dynamic dPL-GR4J with a causal residual MLP."""

    dpl_class = DPLGR4Jd


MODEL_REGISTRY = {
    "dplgr4j": DPLGR4J,
    "dplgr4jnet": DPLGR4JNet,
    "dplgr4jd": DPLGR4Jd,
    "dplgr4jnetd": DPLGR4JNetd,
}


def build_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)


# Compatibility with torchhydro class names.
DplLstmGr4j = DPLGR4J
DplLstmGr4jWithMLP = DPLGR4JNet
DplLstmGr4jDynamic = DPLGR4Jd
DplLstmGr4jDynamicWithMLP = DPLGR4JNetd


__all__ = ["DPLGR4J", "DPLGR4JNet", "DPLGR4Jd", "DPLGR4JNetd", "build_model"]
