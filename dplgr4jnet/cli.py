from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dplgr4jnet.config import load_config, save_config
from dplgr4jnet.trainer import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the standalone dPLGR4JNet model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-dir", help="Override data.data_dir in config or via environment.")
    parser.add_argument("--station-name", help="Override station_name in config.")
    parser.add_argument("--basin-name", help="Override basin_name in config.")
    parser.add_argument("--runoff-name", help="Override runoff_name in config.")
    parser.add_argument("--output-dir", help="Override experiment.output_dir in config.")
    parser.add_argument("--epochs", type=int, help="Override training.epochs in config.")
    parser.add_argument("--device", help="Override training.device in config.")
    parser.add_argument("--dry-run", action="store_true", help="Only test data loading and a single forward pass.")
    return parser


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.station_name:
        cfg["data"]["station_name"] = args.station_name
    if args.basin_name:
        cfg["data"]["basin_name"] = args.basin_name
    if args.runoff_name:
        cfg["data"]["runoff_name"] = args.runoff_name
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.device:
        cfg["training"]["device"] = args.device
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")

    summary = run_experiment(cfg, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
