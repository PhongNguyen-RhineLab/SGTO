"""Scenario-Based Global Trajectory Optimization (Algorithm 1).

Per iteration k:
  1. sample m scenarios Omega_k from the train pool
  2. semi-gradient weights: w_k(e) = mean marginal add (e not in X_k)
     or mean marginal removal (e in X_k) over Omega_k
  3. modular knapsack under partition + budget constraints
     (per-zone best level, then 0/1 knapsack by dynamic programming)
  4. local one-exchange on the knapsack solution
  5. accept only if F_hat on Omega_val improves by > eps, else stop

Flags allow the two ablation baselines from the paper:
  use_exchange=False   -> "SGTO without Local Exchange"
  risk_aware=False     -> "Risk-Neutral SGTO" (rho = 0 in all evaluations)
"""

import numpy as np

from algorithms.base import Solver
from algorithms.greedy import cost_aware_greedy
from algorithms.local_search import local_exchange
from model.reward import IncrementalState


def solve_modular_knapsack(inst, w: np.ndarray) -> set:
    """max sum w(e) s.t. one level per zone, total cost <= B.

    Costs are level costs (a handful of distinct values), so we scale to
    an integer grid for exact DP. Items = zones; per zone the choice is
    "no build" or one of the levels with positive weight.
    """
    B = inst.budget
    # integer cost grid: greatest common divisor of level costs
    costs = np.unique(inst.cost)
    step = float(np.gcd.reduce(np.round(costs).astype(int)))
    cap = int(B // step)

    n_zones = inst.n_zones
    # per zone: list of (int_cost, weight, element)
    zone_items = [[] for _ in range(n_zones)]
    for e in range(inst.n_elements):
        if w[e] > 0 and inst.cost[e] <= B:
            zone_items[inst.zone_of[e]].append(
                (int(round(inst.cost[e] / step)), w[e], e))

    NEG = -1.0
    dp = np.zeros(cap + 1)
    choice = [dict() for _ in range(cap + 1)]  # budget -> {zone: element}
    # group knapsack DP
    for z in range(n_zones):
        items = zone_items[z]
        if not items:
            continue
        new_dp = dp.copy()
        new_choice_src = np.arange(cap + 1)     # which old budget cell
        new_choice_e = np.full(cap + 1, -1)     # element added, -1 = none
        for c_int, wt, e in items:
            for b in range(cap, c_int - 1, -1):
                cand = dp[b - c_int] + wt
                if cand > new_dp[b]:
                    new_dp[b] = cand
                    new_choice_src[b] = b - c_int
                    new_choice_e[b] = e
        # rebuild choice maps for this layer
        new_choice = []
        for b in range(cap + 1):
            base = choice[new_choice_src[b]]
            if new_choice_e[b] >= 0:
                d = dict(base)
                d[z] = int(new_choice_e[b])
                new_choice.append(d)
            else:
                new_choice.append(choice[b])
        dp, choice = new_dp, new_choice

    b_star = int(np.argmax(dp))
    return set(choice[b_star].values())


class SGTO(Solver):
    name = "sgto"

    def __init__(self, inst, rm, algo_cfg, use_exchange=True,
                 risk_aware=True, name=None):
        super().__init__(inst, rm, algo_cfg)
        self.use_exchange = use_exchange
        self.risk_aware = risk_aware
        if name:
            self.name = name
        elif not use_exchange:
            self.name = "sgto_no_exchange"
        elif not risk_aware:
            self.name = "sgto_risk_neutral"

    def _f_val(self, X) -> float:
        """Validation objective F_hat on Omega_val (risk handling per flag)."""
        self.n_evals += len(self.inst.scen_val) * (2 if self.risk_aware else 1)
        st = IncrementalState(self.rm, self.inst.scen_val, X=X)
        return st.f_rob() if self.risk_aware else st.f_mean()

    def _solve(self):
        inst, rm, cfg = self.inst, self.rm, self.cfg
        rng = np.random.default_rng(cfg.seed)
        counter = [0]

        X_k = cost_aware_greedy(inst, rm, inst.scen_train,
                                risk_aware=self.risk_aware, counter=counter)
        f_val_k = self._f_val(X_k)
        history = [{"iter": -1, "stage": "init", "f_val": f_val_k,
                    "cost": inst.solution_cost(X_k), "size": len(X_k)}]
        rejects = 0

        for k in range(cfg.max_iters):
            # 1. sample scenarios
            m = min(cfg.n_sampled, len(inst.scen_train))
            idx = rng.choice(len(inst.scen_train), size=m, replace=False)
            omega_k = [inst.scen_train[i] for i in idx]

            # 2. semi-gradient weights
            state = IncrementalState(rm, omega_k, X=X_k)
            w = np.empty(inst.n_elements)
            for e in range(inst.n_elements):
                if e in X_k:
                    w[e] = state.gain_remove(e, risk_aware=False)
                else:
                    w[e] = state.gain_add(e, risk_aware=False)
                counter[0] += 2 * m

            # 3. modular knapsack
            X_tilde = solve_modular_knapsack(inst, w)

            # 4. local exchange on the training sample
            if self.use_exchange:
                X_hat = local_exchange(inst, rm, X_tilde, omega_k,
                                       eps=cfg.eps,
                                       max_passes=cfg.exchange_max_passes,
                                       counter=counter)
            else:
                X_hat = X_tilde

            # 5. validation-based acceptance
            f_val_new = self._f_val(X_hat)
            accepted = f_val_new > f_val_k + cfg.eps
            history.append({"iter": k, "f_val": f_val_new,
                            "accepted": accepted,
                            "cost": inst.solution_cost(X_hat),
                            "size": len(X_hat)})
            if accepted:
                X_k, f_val_k = X_hat, f_val_new
                rejects = 0
            else:
                rejects += 1
                if rejects >= getattr(cfg, "patience", 1):
                    break

        self.n_evals += counter[0]
        return X_k, history
