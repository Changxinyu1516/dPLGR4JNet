"""Exercise three-term training, warm-start and daily diagnostic exports."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dplgr4jnet.config import load_config
from dplgr4jnet.losses import StabilizedNSELoss
from dplgr4jnet.model import WATER_BALANCE_BASE, WATER_BALANCE_NET, canonical_model_name
from dplgr4jnet.trainer import create_model


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    assert canonical_model_name("DplLstmGr4jWaterBalance") == WATER_BALANCE_BASE
    assert canonical_model_name("DplLstmGr4jWaterBalanceWithMLP") == WATER_BALANCE_NET
    config_path = ROOT / "configs" / "water_balance.yaml"
    command = [
        sys.executable, "-m", "dplgr4jnet", "--config", str(config_path),
        "--device", "cpu", "--epochs", "1",
    ]
    with TemporaryDirectory() as directory:
        base = Path(directory) / "base"
        combined = Path(directory) / "combined"
        subprocess.run([*command, "--output-dir", str(base)], cwd=ROOT, check=True)

        weights = torch.load(base / "best_model.pt", weights_only=True)["model_state_dict"]
        torchhydro_weights = {
            "module." + key.replace("linear_in", "linearIn").replace("linear_out", "linearOut"): value
            for key, value in weights.items()
        }
        legacy_path = Path(directory) / "torchhydro.pth"
        torch.save(torchhydro_weights, legacy_path)
        config = load_config(config_path)
        base_model = create_model(config, 5, 0).eval()
        base_model.load_state_dict(weights)
        config["model"].update(name=WATER_BALANCE_NET, pretrained_dpl_path=str(legacy_path))
        net = create_model(config, 5, 0).eval()

        raw = 10 * torch.rand(60, 2, 5)
        normalized = torch.randn_like(raw)
        observed = torch.rand(60, 2, 1)
        with torch.no_grad():
            torch.testing.assert_close(net(raw, normalized), base_model(raw, normalized))

        net.train()
        torch.nn.init.constant_(net.calibrator.output_layer.bias, 0.3)
        simulated, extras = net(raw, normalized, return_extras=True)
        diagnostics = extras["diagnostics"]
        assert diagnostics["wb_physical_residual_mm"].abs().max() < 1e-4
        assert diagnostics["ET_actual_mm"].le(diagnostics["PET_mm"] + 1e-6).all()
        criterion = StabilizedNSELoss()
        loss = net.compute_model_loss(simulated, observed, criterion)
        torch.testing.assert_close(
            loss,
            criterion(simulated, observed) + 0.1 * diagnostics["wb_model_residual_mm"].square().mean(),
        )
        loss.backward()
        assert net.calibrator.output_layer.bias.grad.abs().sum() > 0
        assert net.dplgr4j.dl_model.linear_out.weight.grad.abs().sum() > 0

        subprocess.run([
            *command, "--model", WATER_BALANCE_NET, "--pretrained-dpl-path",
            str(base / "best_model.pt"), "--output-dir", str(combined),
        ], cwd=ROOT, check=True)
        for run_dir in (base, combined):
            for split in ("train", "valid"):
                states = pd.read_csv(run_dir / f"{split}_water_balance.csv")
                parameters = pd.read_csv(run_dir / f"{split}_parameters.csv")
                predictions = pd.read_csv(run_dir / f"{split}_predictions.csv")
                assert states["date"].equals(parameters["date"])
                assert states["Q_sim_mm"].equals(predictions["predicted_mm_day"])
                with pd.ExcelFile(run_dir / f"{split}_predictions.xlsx") as workbook:
                    assert "water_balance" in workbook.sheet_names

    print("Water balance, two-stage warm-start and diagnostic exports verified.")


if __name__ == "__main__":
    main()
