"""Common solver interface so run_experiment can loop over methods."""

import time
from dataclasses import dataclass, field

from model.instance import ProblemInstance
from model.reward import RewardModel


@dataclass
class SolveResult:
    name: str
    X: set
    runtime_s: float = 0.0
    n_evals: int = 0          # number of scenario-level reward evaluations
    history: list = field(default_factory=list)  # per-iteration diagnostics


class Solver:
    name = "base"

    def __init__(self, inst: ProblemInstance, rm: RewardModel, algo_cfg):
        self.inst = inst
        self.rm = rm
        self.cfg = algo_cfg
        self.n_evals = 0

    def solve(self) -> SolveResult:
        t0 = time.perf_counter()
        X, history = self._solve()
        return SolveResult(name=self.name, X=X,
                           runtime_s=time.perf_counter() - t0,
                           n_evals=self.n_evals, history=history)

    def _solve(self):
        raise NotImplementedError
