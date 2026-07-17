"""Cost-aware greedy initialization, upgraded.

Two improvements over the plain benefit-to-cost rule:

1. Dual rule. Ratio greedy alone can be arbitrarily bad when one
   expensive element is worth more than many cheap ones (example:
   budget 900, a large station with gain 300 vs cheap stations with
   gain 25 at cost 60 each; ratio picks 15 smalls for 375 total, but
   if their gains overlap heavily the large may dominate). The classic
   remedy is to run greedy by gain/cost AND greedy by pure gain, then
   keep the better of the two; this carries the (1-1/e)/2-style
   guarantee from the budgeted maximum coverage literature.

2. Refill mode. greedy_fill() can start from an existing partial
   solution and spend the remaining budget. Used by the local-search
   drop-and-refill move and by SGTO's perturbation step.
"""

import numpy as np

from algorithms.base import Solver
from model.reward import IncrementalState


def greedy_fill(inst, rm, scenarios, X0=None, rule="ratio",
                risk_aware=True, counter=None, state=None):
    """Greedily extend X0 (default empty) under budget and partition.

    rule = "ratio" maximizes marginal gain / cost, "gain" maximizes
    marginal gain. Returns the final set; if `state` is passed it is
    mutated in place and reused (saves rebuilding for the caller).
    """
    if state is None:
        state = IncrementalState(rm, scenarios, X=set(X0) if X0 else None)
    remaining = float(inst.budget) - inst.solution_cost(state.X)
    blocked_zones = {inst.zone_of[e] for e in state.X}
    active = [e for e in range(inst.n_elements)
              if inst.zone_of[e] not in blocked_zones
              and inst.cost[e] <= remaining]

    while active:
        best_e, best_key, best_gain = None, 0.0, 0.0
        for e in active:
            g = state.gain_add(e, risk_aware=risk_aware)
            if counter is not None:
                counter[0] += len(scenarios)
            key = g / inst.cost[e] if rule == "ratio" else g
            if key > best_key:
                best_e, best_key, best_gain = e, key, g
        if best_e is None or best_gain <= 0:
            break
        state.add(best_e)
        remaining -= inst.cost[best_e]
        blocked_zones.add(inst.zone_of[best_e])
        active = [e for e in active
                  if inst.zone_of[e] not in blocked_zones
                  and inst.cost[e] <= remaining]
    return state.X


def cost_aware_greedy(inst, rm, scenarios, risk_aware=True,
                      counter=None) -> set:
    """Dual-rule greedy: best of ratio-greedy and gain-greedy."""
    X_ratio = greedy_fill(inst, rm, scenarios, rule="ratio",
                          risk_aware=risk_aware, counter=counter)
    X_gain = greedy_fill(inst, rm, scenarios, rule="gain",
                         risk_aware=risk_aware, counter=counter)
    if X_gain == X_ratio:
        return X_ratio
    f_r = IncrementalState(rm, scenarios, X=X_ratio).f_rob()
    f_g = IncrementalState(rm, scenarios, X=X_gain).f_rob()
    if counter is not None:
        counter[0] += 2 * len(scenarios)
    return X_ratio if f_r >= f_g else X_gain


class CostAwareGreedy(Solver):
    name = "cost_aware_greedy"

    def _solve(self):
        counter = [0]
        X = cost_aware_greedy(self.inst, self.rm, self.inst.scen_train,
                              risk_aware=True, counter=counter)
        self.n_evals = counter[0]
        return X, []
