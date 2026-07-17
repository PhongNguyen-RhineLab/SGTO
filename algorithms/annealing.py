"""Simulated annealing baseline.

State: a feasible set X. Moves, chosen uniformly at random among the
applicable ones:
  add       random element in a free zone within remaining budget
  remove    random element of X
  relevel   change the level at a selected zone (if budget allows)
  swap      replace a random element with one in a different free zone

Acceptance: Metropolis on F_rob over a fixed training-scenario sample,
geometric cooling. The IncrementalState makes each move evaluation one
delta computation instead of a full re-evaluation.

Doubles as an independent check on SGTO: a long SA run exploring a
different neighborhood structure should not find substantially better
solutions if SGTO is near the instance's ceiling.
"""

import numpy as np

from algorithms.base import Solver
from model.reward import IncrementalState


class SimulatedAnnealing(Solver):
    name = "simulated_annealing"

    def __init__(self, inst, rm, algo_cfg, n_moves=15000,
                 t0=None, t_end_frac=1e-3, init_X=None):
        super().__init__(inst, rm, algo_cfg)
        self.n_moves = n_moves
        self.t0 = t0
        self.t_end_frac = t_end_frac
        self.init_X = init_X

    def _random_feasible(self, rng):
        from algorithms.random_search import random_feasible
        return random_feasible(self.inst, rng)

    def _solve(self):
        inst, rm = self.inst, self.rm
        rng = np.random.default_rng(self.cfg.seed)
        scens = inst.scen_train
        X0 = set(self.init_X) if self.init_X else self._random_feasible(rng)
        state = IncrementalState(rm, scens, X=X0)
        cur = state.f_rob()
        self.n_evals += 2 * len(scens)
        best_X, best_f = set(state.X), cur

        # auto temperature: sample a few random move deltas
        deltas = []
        for _ in range(30):
            e = int(rng.integers(inst.n_elements))
            if inst.zone_of[e] in {inst.zone_of[x] for x in state.X}:
                continue
            if inst.solution_cost(state.X) + inst.cost[e] <= inst.budget:
                deltas.append(abs(state.gain_add(e)))
                self.n_evals += len(scens)
        t0 = self.t0 or (np.median(deltas) if deltas else 1.0)
        t_end = max(t0 * self.t_end_frac, 1e-9)
        cool = (t_end / t0) ** (1.0 / max(1, self.n_moves))
        T = t0

        history = []
        for it in range(self.n_moves):
            zones_used = {inst.zone_of[x]: x for x in state.X}
            spent = inst.solution_cost(state.X)
            move = rng.integers(4)
            delta, apply_fn = None, None

            if move == 0:  # add
                free = [e for e in range(inst.n_elements)
                        if inst.zone_of[e] not in zones_used
                        and spent + inst.cost[e] <= inst.budget]
                if free:
                    e = int(rng.choice(free))
                    delta = state.gain_add(e)
                    apply_fn = lambda e=e: state.add(e)
            elif move == 1 and state.X:  # remove
                e = int(rng.choice(list(state.X)))
                delta = -state.gain_remove(e)
                apply_fn = lambda e=e: state.remove(e)
            elif move == 2 and state.X:  # relevel
                e_out = int(rng.choice(list(state.X)))
                z = inst.zone_of[e_out]
                opts = [e for e in range(z * 0, inst.n_elements)
                        if inst.zone_of[e] == z and e != e_out
                        and spent - inst.cost[e_out] + inst.cost[e]
                        <= inst.budget]
                if opts:
                    e_in = int(rng.choice(opts))
                    base = state.f_rob()
                    state.remove(e_out)
                    delta = state.gain_add(e_in) + state.f_rob() - base
                    state.add(e_out)  # undo for now

                    def apply_fn(e_out=e_out, e_in=e_in):
                        state.remove(e_out)
                        state.add(e_in)
            elif move == 3 and state.X:  # swap zones
                e_out = int(rng.choice(list(state.X)))
                free = [e for e in range(inst.n_elements)
                        if inst.zone_of[e] not in zones_used
                        and spent - inst.cost[e_out] + inst.cost[e]
                        <= inst.budget]
                if free:
                    e_in = int(rng.choice(free))
                    base = state.f_rob()
                    state.remove(e_out)
                    delta = state.gain_add(e_in) + state.f_rob() - base
                    state.add(e_out)

                    def apply_fn(e_out=e_out, e_in=e_in):
                        state.remove(e_out)
                        state.add(e_in)

            if delta is not None:
                self.n_evals += 3 * len(scens)
                if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
                    apply_fn()
                    cur += delta
                    if cur > best_f:
                        # re-evaluate exactly to avoid drift accumulation
                        exact = state.f_rob()
                        self.n_evals += 2 * len(scens)
                        if exact > best_f:
                            best_X, best_f = set(state.X), exact
                        cur = exact
            T *= cool
            if it % 2000 == 0:
                history.append({"iter": it, "T": float(T),
                                "cur": float(cur), "best": float(best_f)})

        return best_X, history
