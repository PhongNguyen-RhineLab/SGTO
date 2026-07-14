"""Reward function of the paper, plus an incremental evaluator.

Implements, for solution X and scenario omega:

    C_omega(X) = sum_t w_t sum_u d_{u,t} [1 - prod_{e in X} (1 - a_{u,e})]
    Y(X)       = sum_{e<f in X} b_{ef}
    P_omega(X) = sum_{t,z} max(0, l_{z,t} + sum_{e in X_z} zeta_t q_e - g_{z,t})^2
    U_omega(X) = sum_{t,u} max(0, d_{u,t} - s_{u,t}(X)),
                 s_{u,t}(X) = min(d_{u,t}, serve_eff * zeta_t * sum_e a_{u,e} q_e)

    F_omega = alpha C - beta_note Y ... (see ModelConfig for weights)
    F_rob   = sum_omega p_omega F_omega - rho * CVaR_delta(L_omega)
    L_omega = gamma P_omega + eta U_omega

Design note: the coverage product and the grid load are modular in the
"state" quantities Q_u = prod(1-a), load_{z,t}, and served capacity
S_u = sum a q. The IncrementalState below caches these so a marginal
gain F(X + e) - F(X) costs O(U*T + Z*T) per scenario instead of a full
re-evaluation over X. This is what makes greedy and SGTO tractable at
E = 825 elements.
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
        # Precompute per-element served-capacity contribution a_{u,e} * q_e
        self.AQ = inst.A * inst.qcap[None, :]          # (U, E)
        # Element synergy lookup goes through zones
        self.Bz = inst.B_zone

    # ------------------------------------------------------------------
    # Full (non-incremental) evaluation, used for validation and metrics
    # ------------------------------------------------------------------
    def components(self, X: set, scen: Scenario) -> dict:
        inst, cfg = self.inst, self.cfg
        idx = np.fromiter(X, dtype=int) if X else np.empty(0, dtype=int)

        # coverage
        if idx.size:
            Q = np.prod(1.0 - inst.A[:, idx], axis=1)   # (U,)
        else:
            Q = np.ones(inst.n_regions)
        cov_frac = 1.0 - Q                               # (U,)
        C = float((inst.w_t[:, None] * scen.demand * cov_frac[None, :]).sum())

        # synergy (scenario independent)
        Y = self._synergy(idx)

        # grid penalty
        load = scen.bg_load.copy()                       # (T, Z)
        if idx.size:
            add = np.zeros(inst.n_grid)
            np.add.at(add, inst.grid_of[idx], inst.qcap[idx])
            load = load + scen.zeta[:, None] * add[None, :]
        over = np.maximum(0.0, load - scen.grid_cap)
        P = float((over ** 2).sum())

        # unmet demand
        S_u = self.AQ[:, idx].sum(axis=1) if idx.size else np.zeros(inst.n_regions)
        served_cap = cfg.serve_eff * scen.zeta[:, None] * S_u[None, :]  # (T, U)
        unmet = np.maximum(0.0, scen.demand - served_cap)
        U = float(unmet.sum())

        ds = inst.demand_scale
        return {
            "C": C * ds, "Y": Y, "P": P * ds * ds, "U": U * ds,
        }

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
        """Equal-weight sample average over a scenario list (Eq. F_hat)."""
        return float(np.mean([self.f_omega(X, s) for s in scenarios]))

    def f_rob(self, X: set, scenarios: list = None) -> float:
        """Expected reward minus rho * CVaR of the loss."""
        scens = scenarios if scenarios is not None else self.inst.scen_train
        probs = np.array([s.prob for s in scens], dtype=float)
        probs = probs / probs.sum()
        f = np.array([self.f_omega(X, s) for s in scens])
        loss = np.array([self.loss_omega(X, s) for s in scens])
        return float((probs * f).sum()
                     - self.cfg.rho * cvar(loss, probs, self.cfg.delta))


class IncrementalState:
    """Cached state for fast marginal gains of F over a fixed scenario list.

    Maintains, per scenario omega:
      Q[omega]     (U,)   prod_{e in X} (1 - a_{u,e})
      S[omega]     (U,)   sum_{e in X} a_{u,e} q_e
      addload      (Z,)   sum_{e in X} q_e per grid region (scenario indep.)
    and the current synergy sum.
    """

    def __init__(self, rm: RewardModel, scenarios: list, X: set = None):
        self.rm = rm
        self.inst = rm.inst
        self.cfg = rm.cfg
        self.scens = scenarios
        self.probs = np.array([s.prob for s in scenarios], dtype=float)
        self.probs /= self.probs.sum()
        self.X = set()
        U = self.inst.n_regions
        self.Q = np.ones(U)
        self.S = np.zeros(U)
        self.addload = np.zeros(self.inst.n_grid)
        self.syn = 0.0
        self._base = None  # cached (fs, losses) of current state
        if X:
            for e in X:
                self.add(e)

    # ---- state updates ------------------------------------------------
    def add(self, e: int):
        inst = self.inst
        self.Q *= (1.0 - inst.A[:, e])
        self.S += self.rm.AQ[:, e]
        self.addload[inst.grid_of[e]] += inst.qcap[e]
        self.syn += self._pair_syn(e)
        self.X.add(e)
        self._base = None

    def remove(self, e: int):
        inst = self.inst
        self.X.discard(e)
        self.syn -= self._pair_syn(e)
        self.addload[inst.grid_of[e]] -= inst.qcap[e]
        self.S -= self.rm.AQ[:, e]
        a = inst.A[:, e]
        # safe division: a < 1 by construction of coverage decay
        self.Q /= (1.0 - a)
        self._base = None

    def _pair_syn(self, e: int) -> float:
        if not self.X:
            return 0.0
        zs = self.inst.zone_of[np.fromiter(self.X - {e}, dtype=int)]
        return float(self.rm.Bz[self.inst.zone_of[e], zs].sum())

    # ---- evaluation from cached state ----------------------------------
    def _f_from_state(self, Q, S, addload, syn) -> tuple:
        """Returns (array of F_omega, array of loss_omega)."""
        cfg, inst = self.cfg, self.inst
        ds = inst.demand_scale
        cov_frac = 1.0 - Q
        fs, losses = [], []
        for s in self.scens:
            C = float((inst.w_t[:, None] * s.demand * cov_frac[None, :]).sum()) * ds
            load = s.bg_load + s.zeta[:, None] * addload[None, :]
            over = np.maximum(0.0, load - s.grid_cap)
            P = float((over ** 2).sum()) * ds * ds
            served = cfg.serve_eff * s.zeta[:, None] * S[None, :]
            Uu = float(np.maximum(0.0, s.demand - served).sum()) * ds
            f = cfg.alpha * C + cfg.beta * syn - cfg.gamma * P - cfg.eta * Uu
            fs.append(f)
            losses.append(cfg.gamma * P + cfg.eta * Uu)
        return np.array(fs), np.array(losses)

    def _base_eval(self):
        if self._base is None:
            self._base = self._f_from_state(self.Q, self.S, self.addload, self.syn)
        return self._base

    def f_rob(self) -> float:
        fs, losses = self._base_eval()
        return float((self.probs * fs).sum()
                     - self.cfg.rho * cvar(losses, self.probs, self.cfg.delta))

    def f_mean(self) -> float:
        fs, _ = self._base_eval()
        return float((self.probs * fs).sum())

    def gain_add(self, e: int, risk_aware: bool = True) -> float:
        """F(X + e) - F(X) without mutating state."""
        inst = self.inst
        Q2 = self.Q * (1.0 - inst.A[:, e])
        S2 = self.S + self.rm.AQ[:, e]
        al2 = self.addload.copy()
        al2[inst.grid_of[e]] += inst.qcap[e]
        syn2 = self.syn + self._pair_syn_incl(e)
        fs2, ls2 = self._f_from_state(Q2, S2, al2, syn2)
        fs1, ls1 = self._base_eval()
        if risk_aware:
            v2 = (self.probs * fs2).sum() - self.cfg.rho * cvar(ls2, self.probs, self.cfg.delta)
            v1 = (self.probs * fs1).sum() - self.cfg.rho * cvar(ls1, self.probs, self.cfg.delta)
            return float(v2 - v1)
        return float((self.probs * (fs2 - fs1)).sum())

    def _pair_syn_incl(self, e: int) -> float:
        if not self.X:
            return 0.0
        zs = self.inst.zone_of[np.fromiter(self.X, dtype=int)]
        return float(self.rm.Bz[self.inst.zone_of[e], zs].sum())

    def gain_remove(self, e: int, risk_aware: bool = True) -> float:
        """F(X) - F(X - e), i.e. the removal weight w_k^-(e)."""
        self.remove(e)
        g = self.gain_add(e, risk_aware=risk_aware)  # F(X) - F(X\e)
        self.add(e)
        return g
