"""Shared building blocks for all dataset loaders.

Every loader (urbanev.py, paris.py, ...) produces a "raw" object with the
same duck-typed interface, so scenarios.py and the grid providers work
unchanged across datasets:

    raw.zone_ids            list of zone identifiers (canonical order)
    raw.n_zones             int
    raw.dist_km             (U, U) road distance matrix in km
    raw.adj                 (U, U) 0/1 zone adjacency
    raw.district_of_zones() -> (grid_of_zone: (U,) int array, n_grid: int)
    raw.days()              -> list of (pd.Timestamp, (24, U) demand kWh)

Grid providers turn one day of demand into (bg_load, grid_cap) arrays;
the synthetic provider reproduces the original stated assumption, and
grid_ieee33.py adds an IEEE 33-bus mapping via pandapower.
"""

import numpy as np

# Standard normalized daily load curve (fraction of daily peak per hour),
# shape based on typical urban distribution feeders: overnight trough,
# morning ramp, evening peak.
DAILY_LOAD_SHAPE = np.array([
    0.55, 0.50, 0.47, 0.45, 0.45, 0.48, 0.56, 0.68, 0.78, 0.82, 0.84, 0.85,
    0.84, 0.83, 0.82, 0.83, 0.86, 0.92, 1.00, 0.98, 0.92, 0.82, 0.70, 0.60,
])


def haversine_km(lat, lon):
    """Pairwise great-circle distance matrix (km) from coordinate vectors.

    lat, lon: (U,) arrays in degrees. Returns (U, U).
    """
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :]
         * np.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class SyntheticGrid:
    """Original stated assumption for grid capacity (dataset-agnostic).

    Background load per district follows DAILY_LOAD_SHAPE scaled so its
    peak equals the district's peak charging demand (proxy: districts
    with more charging activity are denser and carry more base load).
    Capacity g_{z,t} = margin * (peak background + reference station
    load), where the reference load assumes a medium build in
    grid_ref_frac of the district's zones.
    """

    name = "synthetic"

    def arrays(self, demand_day, grid_of_zone, n_grid, cfg, grid_scale=1.0):
        m = cfg.model
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
        return _apply_grid_scale(bg, gcap, grid_scale)


def _apply_grid_scale(bg, gcap, grid_scale):
    """Grid reduction scales the headroom ABOVE background load, not total
    capacity. Scaling total capacity below background creates overload
    that exists even with zero stations, a constant loss offset that
    pollutes the CVaR term without discriminating between solutions."""
    if grid_scale != 1.0:
        gcap = bg + grid_scale * np.maximum(0.0, gcap - bg)
    return bg, gcap


def make_grid_provider(name: str, raw=None, cfg=None):
    """Factory: 'synthetic' (default) or 'ieee33' (needs pandapower)."""
    if name in (None, "synthetic"):
        return SyntheticGrid()
    if name == "ieee33":
        from data_processing.grid_ieee33 import IEEE33Grid  # lazy import
        return IEEE33Grid(raw, cfg)
    raise ValueError(f"unknown grid model '{name}' (use synthetic | ieee33)")
