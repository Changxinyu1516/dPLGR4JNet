from __future__ import annotations

import math

import numpy as np


def nse(obs: np.ndarray, sim: np.ndarray, eps: float = 1e-6) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    denominator = np.sum((obs - obs.mean()) ** 2) + eps
    return float(1.0 - np.sum((sim - obs) ** 2) / denominator)


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(sim) - np.asarray(obs)) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(sim) - np.asarray(obs))))


def bias(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.mean(np.asarray(sim) - np.asarray(obs)))


def r2(obs: np.ndarray, sim: np.ndarray, eps: float = 1e-6) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    denominator = np.sum((obs - obs.mean()) ** 2) + eps
    return float(1.0 - np.sum((obs - sim) ** 2) / denominator)


def kge(obs: np.ndarray, sim: np.ndarray, eps: float = 1e-6) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if len(obs) < 2:
        return float("nan")
    obs_std = np.std(obs) + eps
    sim_std = np.std(sim) + eps
    obs_mean = np.mean(obs) + eps
    sim_mean = np.mean(sim) + eps
    corr = np.corrcoef(obs, sim)[0, 1] if len(obs) > 1 else float("nan")
    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean
    return float(1.0 - math.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def summarize_metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    return {
        "NSE": nse(obs, sim),
        "RMSE": rmse(obs, sim),
        "MAE": mae(obs, sim),
        "Bias": bias(obs, sim),
        "R2": r2(obs, sim),
        "KGE": kge(obs, sim),
    }
