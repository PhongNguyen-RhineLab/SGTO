"""Local one-exchange improvement (Eq. one_exchange / exchange_acceptance).

Moves considered, all kept feasible:
  add      X + e            (if budget allows and zone is free)
  swap     X - e- + e+      (different zones)
  relevel  (i, l) -> (i, l')

Accepts a move only if F_rob improves by more than eps; repeats until a
full pass yields no accepted move or max_passes is reached. First-improve
strategy keeps the number of reward evaluations bounded.
"""

import numpy as np

from algorithms.base import Solver
from algorithms.greedy import cost_aware_greedy
from model.reward import IncrementalState


def local_exchange(inst, rm, X: set, scenarios, eps: float,
                   max_passes: int = 3, counter=None) -> set:
    state = IncrementalState(rm, scenarios, X=X)
    E = inst.n_elements

    for _ in range(max_passes):
        improved = False
        cur_cost = inst.solution_cost(state.X)
        zones_used = {inst.zone_of[e]: e for e in state.X}

        # --- add moves ---------------------------------------------------
        for e in range(E):
            if inst.zone_of[e] in zones_used:
                continue
            if cur_cost + inst.cost[e] > inst.budget:
                continue
            g = state.gain_add(e)
            if counter is not None:
                counter[0] += 2 * len(scenarios)
            if g > eps:
                state.add(e)
                zones_used[inst.zone_of[e]] = e
                cur_cost += inst.cost[e]
                improved = True

        # --- swap and relevel moves ---------------------------------------
        for e_out in list(state.X):
            base = state.f_rob()
            if counter is not None:
                counter[0] += len(scenarios)
            state.remove(e_out)
            cost_wo = cur_cost - inst.cost[e_out]
            zone_out = inst.zone_of[e_out]
            best_e, best_val = e_out, base + eps
            for e_in in range(E):
                z_in = inst.zone_of[e_in]
                if e_in != e_out and z_in in zones_used and z_in != zone_out:
                    continue
                if cost_wo + inst.cost[e_in] > inst.budget:
                    continue
                val = state.f_rob() + state.gain_add(e_in)
                if counter is not None:
                    counter[0] += 3 * len(scenarios)
                if val > best_val:
                    best_e, best_val = e_in, val
            state.add(best_e)
            del zones_used[zone_out]
            zones_used[inst.zone_of[best_e]] = best_e
            cur_cost = cost_wo + inst.cost[best_e]
            if best_e != e_out:
                improved = True

        if not improved:
            break

    # --- drop-and-refill, once after exchange converges -----------------
    # One-exchange cannot trade one expensive element for several cheap
    # ones (or vice versa). Dropping the weakest elements and greedily
    # respending the freed budget covers 1-to-many moves. Runs once, on
    # the k weakest elements by removal gain, to bound cost (running it
    # inside every pass made large-budget instances intractable).
    from algorithms.greedy import greedy_fill  # local import, no cycle
    k_worst = 2
    if state.X:
        rem_gain = {e: state.gain_remove(e) for e in state.X}
        if counter is not None:
            counter[0] += 3 * len(scenarios) * len(state.X)
        worst = sorted(rem_gain, key=rem_gain.get)[:k_worst]
        for e in worst:
            if e not in state.X:
                continue
            base = state.f_rob()
            trial = set(state.X)
            trial.discard(e)
            trial = greedy_fill(inst, rm, scenarios, X0=trial,
                                counter=counter)
            val = IncrementalState(rm, scenarios, X=trial).f_rob()
            if counter is not None:
                counter[0] += len(scenarios)
            if val > base + eps:
                state = IncrementalState(rm, scenarios, X=trial)
    return state.X


class GreedyWithExchange(Solver):
    name = "greedy_one_exchange"

    def _solve(self):
        counter = [0]
        X = cost_aware_greedy(self.inst, self.rm, self.inst.scen_train,
                              counter=counter)
        X = local_exchange(self.inst, self.rm, X, self.inst.scen_train,
                           eps=self.cfg.eps,
                           max_passes=self.cfg.exchange_max_passes,
                           counter=counter)
        self.n_evals = counter[0]
        return X, []
