"""Random search baseline: sample feasible solutions, keep the best.

The evaluation budget is matched roughly to what greedy consumes so the
comparison is fair in reward evaluations, not wall time alone.
"""

import numpy as np

from algorithms.base import Solver
from model.reward import IncrementalState


def random_feasible(inst, rng) -> set:
    zones = rng.permutation(inst.n_zones)
    X, spent = set(), 0.0
    n_levels = inst.n_elements // inst.n_zones
    for z in zones:
        l = int(rng.integers(n_levels))
        e = int(z * n_levels + l)
        if spent + inst.cost[e] <= inst.budget:
            X.add(e)
            spent += inst.cost[e]
        if rng.random() < 0.1:  # stop early sometimes for diverse sizes
            break
    return X


class RandomSearch(Solver):
    name = "random_search"

    def __init__(self, inst, rm, algo_cfg, n_samples=200):
        super().__init__(inst, rm, algo_cfg)
        self.n_samples = n_samples

    def _solve(self):
        rng = np.random.default_rng(self.cfg.seed)
        best_X, best_f = set(), -np.inf
        for _ in range(self.n_samples):
            X = random_feasible(self.inst, rng)
            st = IncrementalState(self.rm, self.inst.scen_train, X=X)
            f = st.f_rob()
            self.n_evals += 2 * len(self.inst.scen_train)
            if f > best_f:
                best_X, best_f = X, f
        return best_X, []
