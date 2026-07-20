"""Scenario-Based Global Trajectory Optimization (Algorithm 1), with
iterated-local-search extensions.

Per iteration k:
  1. sample m scenarios Omega_k from the train pool
  2. semi-gradient weights at the current linearization point X_cur:
     w_k(e) = mean marginal add (e not in X_cur) or mean marginal
     removal (e in X_cur) over Omega_k
  3. modular knapsack under partition + budget (per-zone group DP)
  4. local one-exchange (plus drop-and-refill) on the knapsack solution
  5. validation gate on Omega_val against the best solution so far:
       accept  -> X_best updated, continue from it
       reject  -> perturb X_best (drop a random fraction, greedy refill
                  on the fresh sample) and continue from the perturbed
                  point; terminate after `patience` consecutive rejects
Finally (optional): one local-exchange polish of X_best on the full
training set, accepted through the same validation gate.

The perturbation and polish are extensions over the paper's Algorithm 1
(which terminates on the first rejection). Setting patience=1,
perturb_frac=0.0, final_polish=False recovers the paper exactly.
X_best is only ever replaced when validation improves, so the returned
solution is monotone in validation score regardless of the extensions.

Flags for the ablation baselines:
  use_exchange=False   -> "SGTO without Local Exchange"
  risk_aware=False     -> "Risk-Neutral SGTO" (rho = 0 in all evaluations)
"""

import numpy as np

from algorithms.base import Solver
from algorithms.greedy import cost_aware_greedy, greedy_fill
from algorithms.local_search import local_exchange
from model.reward import IncrementalState


def solve_modular_knapsack(inst, w: np.ndarray) -> set:
    """max sum w(e) s.t. one level per zone, total cost <= B.

    Costs are level costs (a handful of distinct values), so we scale to
    an integer grid for exact DP. Items = zones; per zone the choice is
    "no build" or one of the levels with positive weight.
    """
    B = inst.budget
    costs = np.unique(inst.cost)
    step = float(np.gcd.reduce(np.round(costs).astype(int)))
    cap = int(B // step)

    n_zones = inst.n_zones
    zone_items = [[] for _ in range(n_zones)]
    for e in range(inst.n_elements):
        if w[e] > 0 and inst.cost[e] <= B:
            zone_items[inst.zone_of[e]].append(
                (int(round(inst.cost[e] / step)), w[e], e))

    dp = np.zeros(cap + 1)
    choice = [dict() for _ in range(cap + 1)]
    for z in range(n_zones):
        items = zone_items[z]
        if not items:
            continue
        new_dp = dp.copy()
        new_choice_src = np.arange(cap + 1)
        new_choice_e = np.full(cap + 1, -1)
        for c_int, wt, e in items:
            for b in range(cap, c_int - 1, -1):
                cand = dp[b - c_int] + wt
                if cand > new_dp[b]:
                    new_dp[b] = cand
                    new_choice_src[b] = b - c_int
                    new_choice_e[b] = e
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
        self.n_evals += len(self.inst.scen_val) * (2 if self.risk_aware else 1)
        st = IncrementalState(self.rm, self.inst.scen_val, X=X)
        return st.f_rob() if self.risk_aware else st.f_mean()

    def _perturb(self, X: set, scenarios, rng, counter) -> set:
        """Drop a random fraction of X, then greedily respend the budget."""
        frac = getattr(self.cfg, "perturb_frac", 0.0)
        if frac <= 0 or not X:
            return set(X)
        elems = list(X)
        n_drop = max(1, int(round(frac * len(elems))))
        drop = set(rng.choice(elems, size=min(n_drop, len(elems)),
                              replace=False).tolist())
        kept = set(X) - drop
        return greedy_fill(self.inst, self.rm, scenarios, X0=kept,
                           risk_aware=self.risk_aware, counter=counter)

    def _solve(self):
        inst, rm, cfg = self.inst, self.rm, self.cfg
        rng = np.random.default_rng(cfg.seed)
        counter = [0]

        X_best = cost_aware_greedy(inst, rm, inst.scen_train,
                                   risk_aware=self.risk_aware,
                                   counter=counter)
        f_best = self._f_val(X_best)
        X_cur = set(X_best)
        history = [{"iter": -1, "stage": "init", "f_val": f_best,
                    "cost": inst.solution_cost(X_best), "size": len(X_best)}]
        rejects = 0

        for k in range(cfg.max_iters):
            # 1. sample scenarios
            m = min(cfg.n_sampled, len(inst.scen_train))
            idx = rng.choice(len(inst.scen_train), size=m, replace=False)
            omega_k = [inst.scen_train[i] for i in idx]

            # 2. semi-gradient weights at X_cur. The weights use the
            # SAME objective as the validation gate (risk-aware when the
            # solver is risk-aware): mean-only weights propose solutions
            # that the risk-aware gate then vetoes, which empirically
            # collapses the modular phase to a no-op. risk_in_weights
            # False restores the mean-only weights for the ablation.
            ra_w = self.risk_aware and getattr(cfg, "risk_in_weights", True)
            state = IncrementalState(rm, omega_k, X=X_cur)
            w = np.empty(inst.n_elements)
            for e in range(inst.n_elements):
                if e in X_cur:
                    w[e] = state.gain_remove(e, risk_aware=ra_w)
                else:
                    w[e] = state.gain_add(e, risk_aware=ra_w)
                counter[0] += 2 * m * (2 if ra_w else 1)

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

            # 5. validation gate against the best-so-far
            f_val_new = self._f_val(X_hat)
            accepted = f_val_new > f_best + cfg.eps
            history.append({"iter": k, "f_val": f_val_new,
                            "accepted": accepted,
                            "cost": inst.solution_cost(X_hat),
                            "size": len(X_hat)})
            if accepted:
                X_best, f_best = X_hat, f_val_new
                X_cur = set(X_hat)
                rejects = 0
            else:
                rejects += 1
                if rejects >= getattr(cfg, "patience", 1):
                    break
                X_cur = self._perturb(X_best, omega_k, rng, counter)

        # final polish on the full training set, same validation gate
        if getattr(cfg, "final_polish", False) and self.use_exchange:
            X_pol = local_exchange(inst, rm, X_best, inst.scen_train,
                                   eps=cfg.eps, max_passes=1,
                                   counter=counter)
            f_pol = self._f_val(X_pol)
            history.append({"iter": "polish", "f_val": f_pol,
                            "accepted": f_pol > f_best + cfg.eps,
                            "cost": inst.solution_cost(X_pol),
                            "size": len(X_pol)})
            if f_pol > f_best + cfg.eps:
                X_best, f_best = X_pol, f_pol

        self.n_evals += counter[0]
        return X_best, history
