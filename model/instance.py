"""ProblemInstance: everything the reward function and algorithms need.

The instance is dataset-agnostic. A loader (e.g. data_processing/urbanev.py)
is responsible for producing one; algorithms never touch raw files.

Index conventions
-----------------
u in [0, U)   demand regions
e in [0, E)   ground-set elements (zone, level) pairs
z in [0, Z)   grid regions
t in [0, T)   time periods within one scenario day
omega         index into a list of Scenario objects
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Scenario:
    """One demand / grid scenario omega."""

    name: str
    prob: float
    demand: np.ndarray      # (T, U) d_{u,t}^omega, kWh
    grid_cap: np.ndarray    # (T, Z) g_{z,t}^omega, kW
    bg_load: np.ndarray     # (T, Z) background load l_{z,t}^omega, kW
    zeta: np.ndarray        # (T,) utilization profile zeta_t^omega in [0,1]


@dataclass
class ProblemInstance:
    # Ground set E: element e = (zone_of[e], level_of[e])
    n_zones: int
    zone_of: np.ndarray     # (E,) int, candidate zone index of element e
    level_of: np.ndarray    # (E,) int
    cost: np.ndarray        # (E,) c_e
    qcap: np.ndarray        # (E,) q_e in kW
    budget: float

    # Coverage effectiveness a_{u,e}, static geometry x level factor
    A: np.ndarray           # (U, E), values in [0,1]

    # Synergy: symmetric zone-pair weights b (already includes kappa and
    # the D_max cutoff). Element pair (e,f) has synergy B_zone[z_e, z_f].
    B_zone: np.ndarray      # (n_zones, n_zones)

    # Grid membership
    grid_of: np.ndarray     # (E,) int, grid region of element e
    n_grid: int

    # Time weights
    w_t: np.ndarray         # (T,)

    # Scenario sets
    scen_train: list        # list[Scenario], sampling pool D
    scen_val: list          # Omega_val
    scen_test: list         # held-out evaluation

    # bookkeeping
    zone_ids: list = None       # original TAZ ids, for reporting
    level_names: list = None
    demand_scale: float = 1.0

    @property
    def n_elements(self) -> int:
        return len(self.cost)

    @property
    def n_regions(self) -> int:
        return self.A.shape[0]

    def feasible(self, X: set) -> bool:
        """Check partition (one level per zone) and budget constraints."""
        idx = np.fromiter(X, dtype=int) if X else np.empty(0, dtype=int)
        if idx.size == 0:
            return True
        zones = self.zone_of[idx]
        if len(np.unique(zones)) != len(zones):
            return False
        return float(self.cost[idx].sum()) <= self.budget + 1e-9

    def solution_cost(self, X: set) -> float:
        if not X:
            return 0.0
        return float(self.cost[np.fromiter(X, dtype=int)].sum())
