"""Load the UrbanEV Shenzhen dataset and build a ProblemInstance.

Dataset: https://github.com/IntelligentSystemsLab/UrbanEV
(Li et al., "UrbanEV: An open benchmark dataset for urban electric
vehicle charging demand prediction", Scientific Data 2025)

Files used
----------
inf.csv       station_id, longitude, latitude, charge_count, TAZID, ...
volume.csv    hourly charging volume (kWh) per traffic zone, 2022-09 .. 2023-02
distance.csv  road distance matrix between zones (meters)
adj.csv       zone adjacency (0/1)

Mapping onto the model
----------------------
demand regions U   = 275 traffic zones
candidates   V     = the same zones (green-field planning over zones)
ground set   E     = V x capacity levels from ExperimentConfig.levels
coverage a_{u,e}   = lvl_factor(l) * exp(-dist_km(u, zone_e)/decay), cut at radius
routes / synergy   = adjacent zone pairs with road distance <= D_max
grid regions Z     = district groups, TAZID prefix (1xx..5xx)
demand d_{u,t}     = hourly volume, one scenario = one day (T = 24)

Grid capacity is synthetic (a stated assumption): background load per
district follows a standard daily curve scaled to district demand, and
g_{z,t} = margin * (peak background + reference station load).
"""

import os

import numpy as np
import pandas as pd

from config import ExperimentConfig
from model.instance import ProblemInstance


# Standard normalized daily load curve (fraction of daily peak per hour),
# shape based on typical urban distribution feeders: overnight trough,
# morning ramp, evening peak.
DAILY_LOAD_SHAPE = np.array([
    0.55, 0.50, 0.47, 0.45, 0.45, 0.48, 0.56, 0.68, 0.78, 0.82, 0.84, 0.85,
    0.84, 0.83, 0.82, 0.83, 0.86, 0.92, 1.00, 0.98, 0.92, 0.82, 0.70, 0.60,
])


class UrbanEVData:
    """Raw tables plus derived index structures, before scenario building."""

    def __init__(self, data_dir: str):
        self.inf = pd.read_csv(os.path.join(data_dir, "inf.csv"))
        vol = pd.read_csv(os.path.join(data_dir, "volume.csv"),
                          parse_dates=["time"])
        self.volume = vol.set_index("time")
        # zone order = volume columns; keep it canonical everywhere
        self.zone_ids = [c for c in self.volume.columns]
        dist = pd.read_csv(os.path.join(data_dir, "distance.csv"))
        dist = dist.loc[:, [c for c in dist.columns if not c.startswith("Unnamed")]]
        dist.index = dist.columns
        self.dist_km = dist.loc[self.zone_ids, self.zone_ids].to_numpy() / 1000.0
        adj = pd.read_csv(os.path.join(data_dir, "adj.csv"))
        adj = adj.loc[:, [c for c in adj.columns if not c.startswith("Unnamed")]]
        adj.index = adj.columns
        self.adj = adj.loc[self.zone_ids, self.zone_ids].to_numpy()

    @property
    def n_zones(self):
        return len(self.zone_ids)

    def district_of_zones(self) -> np.ndarray:
        """Grid region index per zone from the TAZID hundreds group.

        TAZIDs are 102..1173, so id // 100 recovers the district
        (1xx..11xx -> 11 districts, matching Shenzhen's admin layout).
        """
        prefixes = [int(z) // 100 for z in self.zone_ids]
        uniq = sorted(set(prefixes))
        lookup = {p: i for i, p in enumerate(uniq)}
        return np.array([lookup[p] for p in prefixes]), len(uniq)

    def days(self) -> list:
        """List of (date, (24, U) demand array) for complete days."""
        out = []
        for date, grp in self.volume.groupby(self.volume.index.date):
            if len(grp) == 24:
                out.append((pd.Timestamp(date), grp.to_numpy()))
        return out


def build_instance(cfg: ExperimentConfig) -> ProblemInstance:
    from data_processing.scenarios import build_scenarios  # avoid cycle

    raw = UrbanEVData(cfg.data_dir)
    n_zones = raw.n_zones
    n_levels = len(cfg.levels)

    # ---- ground set E ---------------------------------------------------
    zone_of = np.repeat(np.arange(n_zones), n_levels)
    level_of = np.tile(np.arange(n_levels), n_zones)
    cost = np.array([cfg.levels[l].cost for l in level_of])
    qcap = np.array([cfg.levels[l].capacity_kw for l in level_of])

    # ---- coverage A[u, e] ----------------------------------------------
    m = cfg.model
    geo = np.exp(-raw.dist_km / m.cov_decay_km)
    geo[raw.dist_km > m.cov_radius_km] = 0.0
    np.fill_diagonal(geo, 1.0)
    lvl_f = np.array(m.lvl_factors)[level_of]            # (E,)
    A = geo[:, zone_of] * lvl_f[None, :]                 # (U, E)
    if m.use_congestion:
        # mean hourly demand per zone over the TRAINING period only
        split = pd.Timestamp(cfg.scenarios.split_date)
        train_vol = raw.volume[raw.volume.index < split]
        dbar = train_vol.mean(axis=0).to_numpy()         # (U,) kWh/h ~ kW
        attracted = geo.T @ dbar                         # (n_zones,)
        zeta_bar = 0.5 * (m.zeta_min + m.zeta_max)
        sat = np.minimum(1.0, zeta_bar * qcap /
                         np.maximum(attracted[zone_of], 1e-9))  # (E,)
        A = A * sat[None, :]
    A = np.clip(A, 0.0, 0.999)  # keep 1 - a > 0 for the incremental state

    # ---- synergy zone-pair weights --------------------------------------
    B = np.where((raw.adj > 0) & (raw.dist_km <= m.d_max_km), m.kappa, 0.0)
    np.fill_diagonal(B, 0.0)  # no self synergy; partition constraint also
    # forbids two elements in the same zone, so the diagonal is never used

    # ---- grid regions ----------------------------------------------------
    grid_of_zone, n_grid = raw.district_of_zones()
    grid_of = grid_of_zone[zone_of]

    # ---- time weights ----------------------------------------------------
    w_t = np.full(24, m.w_offpeak)
    w_t[list(m.peak_hours)] = m.w_peak

    # ---- scenarios --------------------------------------------------------
    scen_train, scen_val, scen_test = build_scenarios(
        raw, grid_of_zone, n_grid, cfg)

    return ProblemInstance(
        n_zones=n_zones, zone_of=zone_of, level_of=level_of,
        cost=cost, qcap=qcap, budget=cfg.algo.budget,
        A=A, B_zone=B, grid_of=grid_of, n_grid=n_grid, w_t=w_t,
        scen_train=scen_train, scen_val=scen_val, scen_test=scen_test,
        zone_ids=raw.zone_ids,
        level_names=[l.name for l in cfg.levels],
        demand_scale=m.demand_scale,
    )
