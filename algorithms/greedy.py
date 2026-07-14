"""Cost-aware greedy: repeatedly add argmax marginal gain / cost.

Uses the IncrementalState so each candidate evaluation is O(U*T + Z*T)
per scenario. A lazy-greedy (priority queue) refinement can be added
later; for E = 825 the direct scan is fine.
"""

import numpy as np

from algorithms.base import Solver
from model.reward import IncrementalState


def cost_aware_greedy(inst, rm, scenarios, risk_aware=True,
                      counter=None) -> set:
    """Standalone routine, reused by SGTO as its initializer."""
    state = IncrementalState(rm, scenarios)
    remaining = float(inst.budget)
    blocked_zones = set()
    active = list(range(inst.n_elements))

    while True:
        best_e, best_ratio, best_gain = None, 0.0, 0.0
        for e in active:
            if inst.cost[e] > remaining or inst.zone_of[e] in blocked_zones:
                continue
            g = state.gain_add(e, risk_aware=risk_aware)
            if counter is not None:
                counter[0] += 2 * len(scenarios)
            ratio = g / inst.cost[e]
            if ratio > best_ratio:
                best_e, best_ratio, best_gain = e, ratio, g
        if best_e is None or best_gain <= 0:
            break
        state.add(best_e)
        remaining -= inst.cost[best_e]
        blocked_zones.add(inst.zone_of[best_e])
        active = [e for e in active
                  if inst.zone_of[e] not in blocked_zones
                  and inst.cost[e] <= remaining]
        if not active:
            break
    return state.X


class CostAwareGreedy(Solver):
    name = "cost_aware_greedy"

    def _solve(self):
        counter = [0]
        X = cost_aware_greedy(self.inst, self.rm, self.inst.scen_train,
                              risk_aware=True, counter=counter)
        self.n_evals = counter[0]
        return X, []
