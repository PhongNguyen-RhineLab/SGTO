"""IEEE 33-bus grid provider (pandapower), replacing the synthetic grid.

Replaces the stated assumption "capacity = margin * (peak background +
reference station load)" with limits derived from a standard test
system, following the approach of arXiv:2408.11269 (real stations
coupled to the IEEE 33-bus network).

Mapping
-------
1. The 32 load buses of case33bw are ordered along the feeder and
   partitioned into n_grid contiguous groups, aligned so that each
   group's share of network base load tracks the district's share of
   charging demand (heavy districts sit on heavy feeder sections).
2. District background load l_{z,t} = calib_day * base_kw[z] * shape(t),
   where calib_day rescales the 3.715 MW test network to city magnitude:
   the network's total base load is mapped to the sum of district peak
   charging demands of that day (same magnitude proxy as the synthetic
   provider, so both grid models are comparable).
3. District capacity g_z = calib_day * hosting_kw[z]. hosting_kw[z] is
   the VOLTAGE-CONSTRAINED hosting capacity: the largest uniform load
   multiplier on the district's buses (others at base) keeping min bus
   voltage >= vmin (default 0.90 pu), found by binary search over AC
   power flows. Line thermal ratings in case33bw are placeholders
   (max_i_ka = 99999), so voltage is the binding physical limit.

Everything expensive (power flows) runs once at construction;
per-scenario work is a couple of array multiplications.

Requires: pip install pandapower
"""

import numpy as np

from data_processing.common import DAILY_LOAD_SHAPE, _apply_grid_scale


class IEEE33Grid:
    name = "ieee33"

    def __init__(self, raw, cfg, n_pf_iters: int = 12):
        try:
            import pandapower as pp
            import pandapower.networks as pn
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "grid model 'ieee33' needs pandapower: "
                "pip install pandapower") from e
        self._pp = pp
        self.vmin = getattr(cfg.model, "ieee33_vmin", 0.90)

        net = pn.case33bw()
        grid_of_zone, n_grid = raw.district_of_zones()
        self.n_grid = n_grid

        # --- demand share per district (alignment target) ----------------
        # Mean day over the training period; loaders guarantee days().
        days = raw.days()
        if not days:
            raise ValueError("dataset has no complete days")
        mean_day = np.mean([d for _, d in days], axis=0)      # (24, U)
        share = np.array([
            mean_day[:, grid_of_zone == z].sum() for z in range(n_grid)])
        share = share / max(share.sum(), 1e-12)

        # --- partition load buses into contiguous feeder groups ----------
        loads = net.load.sort_values("bus").reset_index(drop=True)
        p = loads.p_mw.to_numpy()
        cum_p = np.cumsum(p) / p.sum()
        cum_share = np.cumsum(share)
        # bus i goes to the first district whose cumulative share covers it
        self.group_of_load = np.searchsorted(cum_share, cum_p - 1e-12)
        self.group_of_load = np.clip(self.group_of_load, 0, n_grid - 1)
        # guarantee every district owns at least one bus (n_grid <= 32)
        for z in range(n_grid):
            if not np.any(self.group_of_load == z):
                donor = np.argmax(np.bincount(self.group_of_load,
                                              minlength=n_grid))
                idx = np.where(self.group_of_load == donor)[0][-1]
                self.group_of_load[idx] = z

        base_kw = np.array([
            p[self.group_of_load == z].sum() for z in range(n_grid)]) * 1e3
        self.base_kw = base_kw                                  # (Z,)

        # --- voltage-constrained hosting capacity per district -----------
        self.hosting_kw = np.array([
            self._hosting(net, loads, z, n_pf_iters) for z in range(n_grid)])

    # -------------------------------------------------------------------
    def _hosting(self, net, loads, z, iters):
        """Max total kW on district z's buses with min V >= vmin."""
        import copy as _copy
        pp = self._pp
        mask = (self.group_of_load == z)
        base = loads.p_mw.to_numpy().copy()
        baseq = loads.q_mvar.to_numpy().copy()
        netc = _copy.deepcopy(net)          # one working copy, mutated below
        order = netc.load.sort_values("bus").index

        def feasible(k):
            pm = base.copy(); qm = baseq.copy()
            pm[mask] *= k; qm[mask] *= k
            netc.load.loc[order, "p_mw"] = pm
            netc.load.loc[order, "q_mvar"] = qm
            try:
                pp.runpp(netc, numba=False)
            except Exception:
                return False
            return netc.res_bus.vm_pu.min() >= self.vmin

        lo, hi = 1.0, 2.0
        while feasible(hi) and hi < 64:
            lo, hi = hi, hi * 2.0
        if hi >= 64:
            lo = hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        return lo * self.base_kw[z]

    # -------------------------------------------------------------------
    def arrays(self, demand_day, grid_of_zone, n_grid, cfg, grid_scale=1.0):
        # peak charging demand per district for this day (magnitude proxy)
        dd = np.zeros((24, n_grid))
        for z in range(n_grid):
            dd[:, z] = demand_day[:, grid_of_zone == z].sum(axis=1)
        peak_total = dd.max(axis=0).sum()
        calib = peak_total / max(self.base_kw.sum(), 1e-12)

        bg = (DAILY_LOAD_SHAPE[:, None]
              * (calib * self.base_kw)[None, :])                # (24, Z)
        gcap = np.tile((calib * self.hosting_kw)[None, :], (24, 1))
        return _apply_grid_scale(bg, gcap, grid_scale)
