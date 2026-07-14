"""Build train / validation / test scenario sets from UrbanEV history.

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
from data_processing.urbanev import DAILY_LOAD_SHAPE


def _grid_arrays(demand_day: np.ndarray, grid_of_zone: np.ndarray,
                 n_grid: int, cfg: ExperimentConfig,
                 grid_scale: float = 1.0):
    """Background load and capacity per grid region for one day.

    Background load per district is the daily shape scaled so that its
    peak equals the district's peak charging demand (a proxy: districts
    with more charging activity are denser and carry more base load).
    Capacity g_{z,t} = margin * (peak background + reference station
    load), where the reference station load assumes one medium build in
    a quarter of the district's zones. This is a stated assumption, to
    be replaced by an IEEE 33-bus mapping via pandapower later.
    """
    m = cfg.model
    # district demand per hour (24, Z), kWh ~ kW over one hour
    dd = np.zeros((24, n_grid))
    for z in range(n_grid):
        dd[:, z] = demand_day[:, grid_of_zone == z].sum(axis=1)
    peak = dd.max(axis=0)                                  # (Z,)
    bg = DAILY_LOAD_SHAPE[:, None] * peak[None, :]         # (24, Z)

    zones_per_district = np.bincount(grid_of_zone, minlength=n_grid)
    med = cfg.levels[len(cfg.levels) // 2]
    ref_station_load = (m.grid_ref_frac * zones_per_district
                        * med.capacity_kw * m.zeta_max)
    gcap = m.grid_margin * (peak + ref_station_load)       # (Z,)
    gcap = np.tile(gcap[None, :], (24, 1))
    # Grid reduction scales the headroom ABOVE background load, not total
    # capacity. Scaling total capacity below background creates overload
    # that exists even with zero stations, a constant loss offset that
    # pollutes the CVaR term without discriminating between solutions.
    if grid_scale != 1.0:
        gcap = bg + grid_scale * np.maximum(0.0, gcap - bg)
    return bg, gcap


def _zeta(demand_day: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """Utilization profile: city demand shape rescaled to [zeta_min, zeta_max]."""
    m = cfg.model
    tot = demand_day.sum(axis=1)
    lo, hi = tot.min(), tot.max()
    shape = (tot - lo) / (hi - lo) if hi > lo else np.zeros(24)
    return m.zeta_min + (m.zeta_max - m.zeta_min) * shape


def _make(name, demand_day, grid_of_zone, n_grid, cfg,
          grid_scale=1.0, surge_district=None) -> Scenario:
    d = demand_day.copy()
    if surge_district is not None:
        d[:, grid_of_zone == surge_district] *= cfg.scenarios.demand_surge
    bg, gcap = _grid_arrays(d, grid_of_zone, n_grid, cfg, grid_scale)
    return Scenario(name=name, prob=1.0, demand=d, grid_cap=gcap,
                    bg_load=bg, zeta=_zeta(d, cfg))


def _sample_pool(days, rng, sc, n, grid_of_zone, n_grid, cfg, tag):
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
        out.append(_make(f"{tag}_wd_{dt.date()}", d, grid_of_zone, n_grid, cfg))
    for dt, d in pick(weekends, counts["weekend"]):
        out.append(_make(f"{tag}_we_{dt.date()}", d, grid_of_zone, n_grid, cfg))
    for dt, d in pick(peaks, counts["peak"]):
        out.append(_make(f"{tag}_pk_{dt.date()}", d, grid_of_zone, n_grid, cfg))
    for dt, d in pick(days, counts["perturbed"]):
        if rng.random() < 0.5:
            out.append(_make(f"{tag}_gridcut_{dt.date()}", d, grid_of_zone,
                             n_grid, cfg, grid_scale=sc.grid_reduction))
        else:
            out.append(_make(f"{tag}_surge_{dt.date()}", d, grid_of_zone,
                             n_grid, cfg,
                             surge_district=int(rng.integers(n_grid))))
    for s in out:
        s.prob = 1.0 / len(out)
    return out


def build_scenarios(raw, grid_of_zone, n_grid, cfg: ExperimentConfig):
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
                              grid_of_zone, n_grid, cfg, "tr")
    scen_val = _sample_pool(val_days, rng, sc, sc.n_val,
                            grid_of_zone, n_grid, cfg, "val")
    scen_test = _sample_pool(test_days, rng, sc, sc.n_test,
                             grid_of_zone, n_grid, cfg, "te")
    return scen_train, scen_val, scen_test
