"""Reward function of the paper, plus a fast incremental evaluator.

Implements, for solution X and scenario omega:

    C_omega(X) = sum_t w_t sum_u d_{u,t} [1 - prod_{e in X} (1 - a_{u,e})]
    Y(X)       = sum_{e<f in X} b_{ef}
    P_omega(X) = sum_{t,z} max(0, l_{z,t} + sum_{e in X_z} zeta_t q_e - g_{z,t})^2
    U_omega(X) = sum_{t,u} max(0, d_{u,t} - s_{u,t}(X)),
                 s_{u,t}(X) = min(d_{u,t}, serve_eff * zeta_t * sum_e a_{u,e} q_e)

    F_omega = alpha C + beta Y - gamma P - eta U
    F_rob   = sum_omega p_omega F_omega - rho * CVaR_delta(L_omega)
    L_omega = gamma P_omega + eta U_omega

Performance design
------------------
IncrementalState caches the solution-dependent state (coverage product
Q_u, served capacity S_u, added grid load per region) and stacks all
scenario data into tensors:

    D    (M, T, U)   demand
    WD   (M, U)      sum_t w_t * demand  (coverage collapses over t!)
    ZETA (M, T)      utilization profiles
    BG   (M, T, Z)   background load
    GCAP (M, T, Z)   grid capacity

so one full evaluation is a handful of vectorized numpy ops instead of
a Python loop over scenarios. Coverage in particular reduces to a
single (M, U) x (U,) product because w_t and d factor out of the
solution-dependent term. This is what lets greedy, local exchange, and
SGTO explore large neighborhoods at budgets where solutions have
dozens of elements.
"""

import numpy as np

from config import ModelConfig
from model.instance import ProblemInstance, Scenario


def cvar(losses: np.ndarray, probs: np.ndarray, delta: float) -> float:
    """Discrete CVaR_delta: expected loss in the worst (1-delta) tail.

    Example: 10 equiprobable losses, delta=0.9 -> the single worst loss.
    """
    order = np.argsort(losses)[::-1]  # worst first
    l, p = losses[order], probs[order]
    tail = 1.0 - delta
    acc, val = 0.0, 0.0
    for li, pi in zip(l, p):
        take = min(pi, tail - acc)
        if take <= 0:
            break
        val += take * li
        acc += take
    return val / tail if tail > 0 else float(l[0])


class RewardModel:
    def __init__(self, inst: ProblemInstance, cfg: ModelConfig):
        self.inst = inst
        self.cfg = cfg
        # Per-element served-capacity contribution a_{u,e} * q_e
        self.AQ = inst.A * inst.qcap[None, :]          # (U, E)
        self.Bz = inst.B_zone

    # ------------------------------------------------------------------
    # Reference (non-incremental) evaluation, used by metrics
    # ------------------------------------------------------------------
    def components(self, X: set, scen: Scenario) -> dict:
        inst, cfg = self.inst, self.cfg
        idx = np.fromiter(X, dtype=int) if X else np.empty(0, dtype=int)

        if idx.size:
            Q = np.prod(1.0 - inst.A[:, idx], axis=1)
        else:
            Q = np.ones(inst.n_regions)
        cov_frac = 1.0 - Q
        C = float((inst.w_t[:, None] * scen.demand * cov_frac[None, :]).sum())

        Y = self._synergy(idx)

        load = scen.bg_load.copy()
        if idx.size:
            add = np.zeros(inst.n_grid)
            np.add.at(add, inst.grid_of[idx], inst.qcap[idx])
            load = load + scen.zeta[:, None] * add[None, :]
        over = np.maximum(0.0, load - scen.grid_cap)
        P = float((over ** 2).sum())

        S_u = self.AQ[:, idx].sum(axis=1) if idx.size else np.zeros(inst.n_regions)
        served_cap = cfg.serve_eff * scen.zeta[:, None] * S_u[None, :]
        unmet = np.maximum(0.0, scen.demand - served_cap)
        U = float(unmet.sum())

        ds = inst.demand_scale
        return {"C": C * ds, "Y": Y, "P": P * ds * ds, "U": U * ds}

    def _synergy(self, idx: np.ndarray) -> float:
        if idx.size < 2:
            return 0.0
        zs = self.inst.zone_of[idx]
        sub = self.Bz[np.ix_(zs, zs)]
        return float(np.triu(sub, k=1).sum())

    def f_omega(self, X: set, scen: Scenario) -> float:
        c = self.components(X, scen)
        cfg = self.cfg
        return (cfg.alpha * c["C"] + cfg.beta * c["Y"]
                - cfg.gamma * c["P"] - cfg.eta * c["U"])

    def loss_omega(self, X: set, scen: Scenario) -> float:
        c = self.components(X, scen)
        return self.cfg.gamma * c["P"] + self.cfg.eta * c["U"]

    def f_hat(self, X: set, scenarios: list) -> float:
        return float(np.mean([self.f_omega(X, s) for s in scenarios]))

    def f_rob(self, X: set, scenarios: list = None) -> float:
        scens = scenarios if scenarios is not None else self.inst.scen_train
        probs = np.array([s.prob for s in scens], dtype=float)
        probs = probs / probs.sum()
        f = np.array([self.f_omega(X, s) for s in scens])
        loss = np.array([self.loss_omega(X, s) for s in scens])
        return float((probs * f).sum()
                     - self.cfg.rho * cvar(loss, probs, self.cfg.delta))


class IncrementalState:
    """Cached solution state + stacked scenario tensors for fast gains.

    Marginal gains use component deltas on slices: a candidate element
    only changes (i) the coverage product in its neighborhood, (ii) the
    load of its own grid region, (iii) served capacity in the ~18 zones
    it covers (of 275). One gain costs O(M*T*n_affected) instead of
    O(M*T*U), roughly 15x fewer flops on UrbanEV.
    """

    def __init__(self, rm: RewardModel, scenarios: list, X: set = None):
        self.rm = rm
        self.inst = rm.inst
        self.cfg = rm.cfg
        self.scens = scenarios
        self.probs = np.array([s.prob for s in scenarios], dtype=float)
        self.probs /= self.probs.sum()

        inst = self.inst
        self.D = np.stack([s.demand for s in scenarios])          # (M,T,U)
        self.WD = np.einsum("t,mtu->mu", inst.w_t, self.D)        # (M,U)
        self.ZETA = np.stack([s.zeta for s in scenarios])         # (M,T)
        self.BG = np.stack([s.bg_load for s in scenarios])        # (M,T,Z)
        self.GCAP = np.stack([s.grid_cap for s in scenarios])     # (M,T,Z)
        self.WD_tot = self.WD.sum(axis=1)                         # (M,)

        # neighborhood (affected demand regions) per element, precomputed
        # once per RewardModel and shared
        if not hasattr(rm, "_nbrs"):
            rm._nbrs = [np.nonzero(inst.A[:, e])[0]
                        for e in range(inst.n_elements)]
        self.nbrs = rm._nbrs

        self.X = set()
        self.Q = np.ones(inst.n_regions)
        self.S = np.zeros(inst.n_regions)
        self.addload = np.zeros(inst.n_grid)
        self.syn = 0.0
        self._base = None  # cached (C, P, U) arrays, each (M,)
        if X:
            for e in X:
                self.add(e)

    # ---- state updates ------------------------------------------------
    def add(self, e: int):
        inst = self.inst
        self.Q *= (1.0 - inst.A[:, e])
        self.S += self.rm.AQ[:, e]
        self.addload[inst.grid_of[e]] += inst.qcap[e]
        self.syn += self._pair_syn_excl(e)
        self.X.add(e)
        self._base = None

    def remove(self, e: int):
        inst = self.inst
        self.X.discard(e)
        self.syn -= self._pair_syn_excl(e)
        self.addload[inst.grid_of[e]] -= inst.qcap[e]
        self.S -= self.rm.AQ[:, e]
        self.Q /= (1.0 - inst.A[:, e])  # a < 1 by construction
        self._base = None

    def _pair_syn_excl(self, e: int) -> float:
        """Synergy of e with X \\ {e}."""
        others = self.X - {e}
        if not others:
            return 0.0
        zs = self.inst.zone_of[np.fromiter(others, dtype=int)]
        return float(self.rm.Bz[self.inst.zone_of[e], zs].sum())

    # ---- base evaluation (vectorized over scenarios) ---------------------
    def _base_eval(self):
        if self._base is None:
            cfg, inst = self.cfg, self.inst
            ds = inst.demand_scale
            C = (self.WD_tot - self.WD @ self.Q) * ds              # (M,)
            load = self.BG + self.ZETA[:, :, None] * self.addload[None, None, :]
            over = np.maximum(0.0, load - self.GCAP)
            P = (over * over).sum(axis=(1, 2)) * ds * ds           # (M,)
            served = (cfg.serve_eff * self.ZETA)[:, :, None] * self.S[None, None, :]
            un = np.maximum(0.0, self.D - served)
            U = un.sum(axis=(1, 2)) * ds                           # (M,)
            self._base = (C, P, U)
        return self._base

    def _score(self, C, P, U, syn, risk_aware=True):
        cfg = self.cfg
        fs = cfg.alpha * C + cfg.beta * syn - cfg.gamma * P - cfg.eta * U
        if risk_aware:
            losses = cfg.gamma * P + cfg.eta * U
            return float((self.probs * fs).sum()
                         - cfg.rho * cvar(losses, self.probs, cfg.delta))
        return float((self.probs * fs).sum())

    def f_rob(self) -> float:
        C, P, U = self._base_eval()
        return self._score(C, P, U, self.syn, risk_aware=True)

    def f_mean(self) -> float:
        C, P, U = self._base_eval()
        return self._score(C, P, U, self.syn, risk_aware=False)

    # ---- marginal gains via slice deltas ---------------------------------
    def _deltas_add(self, e: int):
        """(dC, dP, dU) arrays (M,) for X -> X + e."""
        inst, cfg = self.inst, self.cfg
        ds = inst.demand_scale
        # coverage: C2 - C1 = WD @ (Q * a_e)
        dC = (self.WD @ (self.Q * inst.A[:, e])) * ds
        # grid: only region z changes
        z = inst.grid_of[e]
        lz1 = self.BG[:, :, z] + self.ZETA * self.addload[z]       # (M,T)
        lz2 = lz1 + self.ZETA * inst.qcap[e]
        gz = self.GCAP[:, :, z]
        o1 = np.maximum(0.0, lz1 - gz)
        o2 = np.maximum(0.0, lz2 - gz)
        dP = (o2 * o2 - o1 * o1).sum(axis=1) * ds * ds
        # unmet: only covered regions change
        nb = self.nbrs[e]
        d_sl = self.D[:, :, nb]                                    # (M,T,k)
        zs = (cfg.serve_eff * self.ZETA)[:, :, None]
        s1 = zs * self.S[nb][None, None, :]
        s2 = zs * (self.S[nb] + self.rm.AQ[nb, e])[None, None, :]
        dU = (np.maximum(0.0, d_sl - s2)
              - np.maximum(0.0, d_sl - s1)).sum(axis=(1, 2)) * ds
        return dC, dP, dU

    def gain_add(self, e: int, risk_aware: bool = True) -> float:
        """F(X + e) - F(X) without mutating state."""
        C, P, U = self._base_eval()
        dC, dP, dU = self._deltas_add(e)
        syn2 = self.syn + self._pair_syn_excl(e)
        return (self._score(C + dC, P + dP, U + dU, syn2, risk_aware)
                - self._score(C, P, U, self.syn, risk_aware))

    def gain_remove(self, e: int, risk_aware: bool = True) -> float:
        """F(X) - F(X - e), i.e. the removal weight w_k^-(e)."""
        self.remove(e)
        g = self.gain_add(e, risk_aware=risk_aware)
        self.add(e)
        return g

    def gains_add_all(self, elems, risk_aware: bool = True) -> np.ndarray:
        """gain_add for many elements; per-element cost is already the
        sliced delta, so this is a thin loop kept for API convenience."""
        return np.array([self.gain_add(int(e), risk_aware=risk_aware)
                         for e in elems])
