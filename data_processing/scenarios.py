"""Build train / validation / test scenario sets from dataset history.

A scenario is one day: a (24, U) demand matrix, a grid capacity matrix,
a background load matrix and a utilization profile.

Scenario types (fractions set in ScenarioConfig):
  weekday    a real weekday sampled from the pool
  weekend    a real weekend day
  peak       one of the highest-total-demand days
  perturbed  a real day with either grid capacity reduced by a factor
             or demand surged in one random district

Days before split_date feed the train pool; days after are split
between validation and test, so validation-based acceptance and final
evaluation never see training days. This mirrors the paper's
Omega_train / Omega_val / Omega_test separation.
"""

import numpy as np
import pandas as pd

from config import ExperimentConfig
from model.instance import Scenario
from data_processing.common import SyntheticGrid


def _zeta(demand_day: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """Utilization profile: city demand shape rescaled to [zeta_min, zeta_max]."""
    m = cfg.model
    tot = demand_day.sum(axis=1)
    lo, hi = tot.min(), tot.max()
    shape = (tot - lo) / (hi - lo) if hi > lo else np.zeros(24)
    return m.zeta_min + (m.zeta_max - m.zeta_min) * shape


def _make(name, demand_day, grid_of_zone, n_grid, cfg, provider,
          grid_scale=1.0, surge_district=None) -> Scenario:
    d = demand_day.copy()
    if surge_district is not None:
        d[:, grid_of_zone == surge_district] *= cfg.scenarios.demand_surge
    bg, gcap = provider.arrays(d, grid_of_zone, n_grid, cfg, grid_scale)
    return Scenario(name=name, prob=1.0, demand=d, grid_cap=gcap,
                    bg_load=bg, zeta=_zeta(d, cfg))


def _sample_pool(days, rng, sc, n, grid_of_zone, n_grid, cfg, tag, provider):
    """Draw n scenarios from a list of (date, demand) days."""
    weekdays = [(dt, d) for dt, d in days if dt.dayofweek < 5]
    weekends = [(dt, d) for dt, d in days if dt.dayofweek >= 5]
    by_total = sorted(days, key=lambda x: x[1].sum(), reverse=True)
    peaks = by_total[:max(3, len(days) // 10)]

    counts = {
        "weekday": round(sc.frac_weekday * n),
        "weekend": round(sc.frac_weekend * n),
        "peak": round(sc.frac_peak * n),
    }
    counts["perturbed"] = max(0, n - sum(counts.values()))

    out = []
    def pick(pool, k):
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[i] for i in idx]

    for dt, d in pick(weekdays, counts["weekday"]):
        out.append(_make(f"{tag}_wd_{dt.date()}", d, grid_of_zone, n_grid,
                         cfg, provider))
    for dt, d in pick(weekends, counts["weekend"]):
        out.append(_make(f"{tag}_we_{dt.date()}", d, grid_of_zone, n_grid,
                         cfg, provider))
    for dt, d in pick(peaks, counts["peak"]):
        out.append(_make(f"{tag}_pk_{dt.date()}", d, grid_of_zone, n_grid,
                         cfg, provider))
    for dt, d in pick(days, counts["perturbed"]):
        if rng.random() < 0.5:
            out.append(_make(f"{tag}_gridcut_{dt.date()}", d, grid_of_zone,
                             n_grid, cfg, provider,
                             grid_scale=sc.grid_reduction))
        else:
            out.append(_make(f"{tag}_surge_{dt.date()}", d, grid_of_zone,
                             n_grid, cfg, provider,
                             surge_district=int(rng.integers(n_grid))))
    for s in out:
        s.prob = 1.0 / len(out)
    return out


def build_scenarios(raw, grid_of_zone, n_grid, cfg: ExperimentConfig,
                    grid_provider=None):
    provider = grid_provider or SyntheticGrid()
    sc = cfg.scenarios
    rng = np.random.default_rng(sc.seed)
    days = raw.days()
    split = pd.Timestamp(sc.split_date)
    train_days = [(dt, d) for dt, d in days if dt < split]
    held_days = [(dt, d) for dt, d in days if dt >= split]
    rng.shuffle(held_days)
    n_held_val = len(held_days) // 2
    val_days, test_days = held_days[:n_held_val], held_days[n_held_val:]

    scen_train = _sample_pool(train_days, rng, sc, sc.n_train,
                              grid_of_zone, n_grid, cfg, "tr", provider)
    scen_val = _sample_pool(val_days, rng, sc, sc.n_val,
                            grid_of_zone, n_grid, cfg, "val", provider)
    scen_test = _sample_pool(test_days, rng, sc, sc.n_test,
                             grid_of_zone, n_grid, cfg, "te", provider)
    return scen_train, scen_val, scen_test
