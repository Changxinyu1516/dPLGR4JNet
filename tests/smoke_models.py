"""Forward/backward and two-stage warm-start checks for all public models."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dplgr4jnet.config import DEFAULT_CONFIG
from dplgr4jnet.model import canonical_model_name
from dplgr4jnet.trainer import create_model


def assert_backward(model, x: torch.Tensor, z: torch.Tensor) -> None:
    model.train()
    prediction = model(x, z)
    loss = prediction.mean() + model.regularization_loss()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert prediction.shape == (35, 2, 1)
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def base_config(name: str) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["model"]["name"] = name
    config["model"]["pretrained_dpl_path"] = None
    return config


def check_pair(
    base_name: str,
    net_name: str,
    checkpoint_path: Path,
    x: torch.Tensor,
    z: torch.Tensor,
) -> None:
    config = base_config(base_name)
    base = create_model(config, feature_count=6, warmup_length=5)
    base.eval()
    with torch.no_grad():
        base_prediction = base(x, z)
    torch.save(
        {
            "model_name": base_name,
            "model_state_dict": base.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )

    net_config = deepcopy(config)
    net_config["model"]["name"] = net_name
    net_config["model"]["pretrained_dpl_path"] = str(checkpoint_path)
    net = create_model(net_config, feature_count=6, warmup_length=5)
    net.eval()
    with torch.no_grad():
        net_initial_prediction = net(x, z)
    # The residual output layer is zero-initialized, so epoch-0 Net output must
    # exactly reproduce the corresponding pretrained dPL baseline.
    assert torch.allclose(base_prediction, net_initial_prediction, atol=1e-6, rtol=1e-6)

    assert_backward(base, x, z)
    assert_backward(net, x, z)
    print(f"{base_name} -> {net_name}: warm-start and backward=ok")


def main() -> None:
    torch.manual_seed(7)
    x = torch.rand(40, 2, 6)
    z = torch.randn(40, 2, 6)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        check_pair("dPLGR4JStatic", "dPLGR4JStaticWithMLP", root / "static.pt", x, z)
        check_pair("dPLGR4JDynamic", "dPLGR4JDynamicWithMLP", root / "dynamic.pt", x, z)
        assert canonical_model_name("dPLGR4J") == "dPLGR4JStatic"
        assert canonical_model_name("dPLGR4JNet") == "dPLGR4JStaticWithMLP"
        assert canonical_model_name("dPLGR4Jd") == "dPLGR4JDynamic"
        assert canonical_model_name("dPLGR4JNetd") == "dPLGR4JDynamicWithMLP"


if __name__ == "__main__":
    main()
