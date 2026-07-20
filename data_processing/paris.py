"""Load the Smarter Mobility (Paris Belib') dataset, second test instance.

Dataset: https://gitlab.com/smarter-mobility-data-challenge/tutorials
(Amara-Ouali et al., "Forecasting Electric Vehicle Charging Station
Occupancy: Smarter Mobility Data Challenge", JDMLR 2024, arXiv:2306.06142)

File used: train.csv with 15-minute plug occupancy for 91 Belib'
stations, 2020-07-03 .. 2021-02-18. Columns (extras are ignored):

    date, Station, Available, Charging, Passive, Other,
    Latitude, Longitude, Postcode, area, tod, dow, trend

Mapping onto the model
----------------------
demand regions U   = 91 stations (each station is its own zone)
candidates   V     = the same stations (re-planning capacity at sites)
ground set   E     = V x capacity levels from ExperimentConfig.levels
d_{u,t}            = hourly mean of #Charging plugs * plug_power_kw (kWh)
coverage a_{u,e}   = lvl_factor * exp(-road_km / decay), cut at radius
routes / synergy   = station pairs with road distance <= adjacency_km
grid regions Z     = arrondissements (Postcode) or challenge areas
scenarios          = real days, split by date into train / val / test

Stated assumptions (cite in the paper):
- Occupancy-to-energy conversion with a single average plug power
  (plug_power_kw). The Belib' network of that period is AC-dominated.
- Road distance defaults to great-circle * detour_factor unless OSMnx
  distances are enabled (--roads osmnx) or cached.
- Pandemic-period demand (2020-2021) is atypical; Paris is therefore
  the secondary generalization instance, not the primary case study.
"""

import os

import numpy as np
import pandas as pd

from config import ExperimentConfig
from data_processing.common import haversine_km
from model.instance import ProblemInstance

REQUIRED_COLS = ("date", "Station", "Charging", "Latitude", "Longitude")


class ParisData:
    """Raw tables plus derived structures; same interface as UrbanEVData."""

    def __init__(self, data_dir: str, cfg: ExperimentConfig):
        pc = cfg.paris
        path = os.path.join(data_dir, pc.train_csv)
        df = pd.read_csv(path, parse_dates=["date"])
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}; "
                             "expected the challenge train.csv schema")

        # ---- station metadata (canonical order = sorted station ids) ----
        meta_cols = [c for c in ("Latitude", "Longitude", "Postcode", "area")
                     if c in df.columns]
        meta = (df.dropna(subset=["Latitude", "Longitude"])
                  .groupby("Station")[meta_cols].first().sort_index())
        self.zone_ids = meta.index.tolist()
        self.lat = meta["Latitude"].to_numpy(float)
        self.lon = meta["Longitude"].to_numpy(float)
        self._meta = meta
        self._district_by = pc.district_by

        # ---- occupancy -> hourly energy demand (kWh) --------------------
        # 15-min Charging counts, hourly mean, * plug power * 1 h.
        occ = (df.pivot_table(index="date", columns="Station",
                              values="Charging", aggfunc="mean")
                 .reindex(columns=self.zone_ids)
                 .sort_index())
        occ = occ.resample("1h").mean()
        occ = occ.interpolate(limit=pc.max_gap_hours, limit_area="inside")
        # demand_growth: plan for projected adoption above the observed
        # (pandemic-period) utilization; a stated assumption.
        self.volume = (occ * pc.plug_power_kw
                       * pc.demand_growth)           # (time, U) kWh per hour
        self._min_day_coverage = pc.min_day_coverage

        # ---- road distances --------------------------------------------
        self.dist_km = self._distances(data_dir, cfg)
        self.adj = ((self.dist_km <= pc.adjacency_km)
                    & (self.dist_km > 0)).astype(float)

    # ------------------------------------------------------------------
    def _distances(self, data_dir, cfg) -> np.ndarray:
        pc = cfg.paris
        geo = haversine_km(self.lat, self.lon) * pc.detour_factor
        if cfg.roads == "geodesic":
            return geo
        try:
            from data_processing.roads_osmnx import road_distance_km
            d = road_distance_km(self.lat, self.lon,
                                 cache_dir=os.path.join(data_dir, "cache"))
            # patch disconnected pairs with the geodesic approximation
            bad = ~np.isfinite(d)
            d[bad] = geo[bad]
            return d
        except Exception as e:
            if cfg.roads == "osmnx":
                raise
            print(f"  [paris] OSMnx unavailable ({e}); "
                  f"falling back to great-circle x {pc.detour_factor}")
            return geo

    # ------------------------------------------------------------------
    @property
    def n_zones(self):
        return len(self.zone_ids)

    def district_of_zones(self):
        """Grid region per station: arrondissement (Postcode) or area."""
        m = self._meta
        if self._district_by == "postcode" and "Postcode" in m.columns:
            pc = m["Postcode"]
            # stations with a missing postcode fall back to their area
            # (a NaN would otherwise become its own spurious district)
            if pc.isna().any() and "area" in m.columns:
                pc = pc.fillna("area_" + m["area"].astype(str))
            keys = pc.astype(str).str.strip().str.replace(
                r"\.0$", "", regex=True).tolist()
        elif "area" in m.columns:
            keys = m["area"].astype(str).tolist()
        else:
            raise ValueError("no Postcode or area column for districts")
        uniq = sorted(set(keys))
        lookup = {k: i for i, k in enumerate(uniq)}
        return np.array([lookup[k] for k in keys]), len(uniq)

    def days(self) -> list:
        """List of (date, (24, U) demand kWh) for sufficiently full days."""
        out = []
        need = int(round(24 * self._min_day_coverage))
        for date, grp in self.volume.groupby(self.volume.index.date):
            if len(grp) != 24:
                continue
            ok_hours = int(grp.notna().all(axis=1).sum())
            if ok_hours < need:
                continue
            arr = grp.to_numpy(float)
            if np.isnan(arr).any():  # only when min_day_coverage < 1
                col_mean = np.nanmean(arr, axis=0)
                idx = np.where(np.isnan(arr))
                arr[idx] = np.take(col_mean, idx[1])
            out.append((pd.Timestamp(date), arr))
        return out


def build_instance(cfg: ExperimentConfig) -> ProblemInstance:
    from data_processing.scenarios import build_scenarios  # avoid cycle
    from data_processing.common import make_grid_provider

    raw = ParisData(cfg.data_dir, cfg)
    n_zones = raw.n_zones
    n_levels = len(cfg.levels)

    # ---- ground set E ---------------------------------------------------
    zone_of = np.repeat(np.arange(n_zones), n_levels)
    level_of = np.tile(np.arange(n_levels), n_zones)
    cost = np.array([cfg.levels[l].cost for l in level_of])
    qcap = np.array([cfg.levels[l].capacity_kw for l in level_of])

    # ---- coverage A[u, e] (same functional form as UrbanEV) -------------
    m = cfg.model
    geo = np.exp(-raw.dist_km / m.cov_decay_km)
    geo[raw.dist_km > m.cov_radius_km] = 0.0
    np.fill_diagonal(geo, 1.0)
    lvl_f = np.array(m.lvl_factors)[level_of]
    A = geo[:, zone_of] * lvl_f[None, :]
    if m.use_congestion:
        split = pd.Timestamp(cfg.scenarios.split_date)
        train_vol = raw.volume[raw.volume.index < split]
        dbar = np.nan_to_num(train_vol.mean(axis=0).to_numpy())
        attracted = geo.T @ dbar
        zeta_bar = 0.5 * (m.zeta_min + m.zeta_max)
        sat = np.minimum(1.0, zeta_bar * qcap /
                         np.maximum(attracted[zone_of], 1e-9))
        A = A * sat[None, :]
    A = np.clip(A, 0.0, 0.999)

    # ---- synergy zone-pair weights --------------------------------------
    B = np.where((raw.adj > 0) & (raw.dist_km <= m.d_max_km), m.kappa, 0.0)
    np.fill_diagonal(B, 0.0)

    # ---- grid regions ----------------------------------------------------
    grid_of_zone, n_grid = raw.district_of_zones()
    grid_of = grid_of_zone[zone_of]

    # ---- time weights ----------------------------------------------------
    w_t = np.full(24, m.w_offpeak)
    w_t[list(m.peak_hours)] = m.w_peak

    # ---- scenarios --------------------------------------------------------
    provider = make_grid_provider(cfg.grid_model, raw, cfg)
    scen_train, scen_val, scen_test = build_scenarios(
        raw, grid_of_zone, n_grid, cfg, grid_provider=provider)

    return ProblemInstance(
        n_zones=n_zones, zone_of=zone_of, level_of=level_of,
        cost=cost, qcap=qcap, budget=cfg.algo.budget,
        A=A, B_zone=B, grid_of=grid_of, n_grid=n_grid, w_t=w_t,
        scen_train=scen_train, scen_val=scen_val, scen_test=scen_test,
        zone_ids=raw.zone_ids,
        level_names=[l.name for l in cfg.levels],
        demand_scale=m.demand_scale,
    )
