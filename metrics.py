"""Evaluation metrics from the Experimental Design section, computed on
the held-out test scenarios.
"""

import numpy as np

from model.reward import RewardModel, cvar


def evaluate(rm: RewardModel, X: set, scenarios=None) -> dict:
    inst, cfg = rm.inst, rm.cfg
    scens = scenarios if scenarios is not None else inst.scen_test
    probs = np.array([s.prob for s in scens], dtype=float)
    probs /= probs.sum()

    idx = np.fromiter(X, dtype=int) if X else np.empty(0, dtype=int)
    if idx.size:
        Q = np.prod(1.0 - inst.A[:, idx], axis=1)
        S_u = (inst.A[:, idx] * inst.qcap[idx][None, :]).sum(axis=1)
        addload = np.zeros(inst.n_grid)
        np.add.at(addload, inst.grid_of[idx], inst.qcap[idx])
    else:
        Q = np.ones(inst.n_regions)
        S_u = np.zeros(inst.n_regions)
        addload = np.zeros(inst.n_grid)
    cov_frac = 1.0 - Q

    peak = np.zeros(24, dtype=bool)
    peak[list(cfg.peak_hours)] = True

    f_l, loss_l = [], []
    served_tot = dem_tot = 0.0
    served_peak = dem_peak = 0.0
    over_tot, over_max, unmet_tot = 0.0, 0.0, 0.0

    for s in scens:
        comp = rm.components(X, s)
        f_l.append(cfg.alpha * comp["C"] + cfg.beta * comp["Y"]
                   - cfg.gamma * comp["P"] - cfg.eta * comp["U"])
        loss_l.append(cfg.gamma * comp["P"] + cfg.eta * comp["U"])

        served_cap = cfg.serve_eff * s.zeta[:, None] * S_u[None, :]
        served = np.minimum(s.demand, served_cap)
        served_tot += served.sum()
        dem_tot += s.demand.sum()
        # peak-hour coverage ratio: covered demand fraction in peak hours
        served_peak += (s.demand[peak] * cov_frac[None, :]).sum()
        dem_peak += s.demand[peak].sum()

        load = s.bg_load + s.zeta[:, None] * addload[None, :]
        over = np.maximum(0.0, load - s.grid_cap)
        over_tot += over.sum()
        over_max = max(over_max, float(over.max()))
        unmet_tot += np.maximum(0.0, s.demand - served_cap).sum()

    f = np.array(f_l)
    loss = np.array(loss_l)
    n = len(scens)
    f_rob = float((probs * f).sum() - cfg.rho * cvar(loss, probs, cfg.delta))
    # Baseline shift: unserved demand of the empty solution is a large
    # constant in F_rob; the gain over "build nothing" is the number to
    # compare and report.
    f0 = np.array([cfg.alpha * 0 + cfg.beta * 0
                   - cfg.gamma * rm.components(set(), s)["P"]
                   - cfg.eta * rm.components(set(), s)["U"] for s in scens])
    loss0 = -f0
    f_rob0 = float((probs * f0).sum() - cfg.rho * cvar(loss0, probs, cfg.delta))
    return {
        "F_rob": f_rob,
        "F_rob_gain": f_rob - f_rob0,
        "F_mean": float((probs * f).sum()),
        "F_worst": float(f.min()),
        "CVaR_loss": float(cvar(loss, probs, cfg.delta)),
        "fulfillment_ratio": served_tot / dem_tot if dem_tot else 0.0,
        "peak_coverage_ratio": served_peak / dem_peak if dem_peak else 0.0,
        "grid_overload_total_kW": over_tot / n,
        "grid_overload_max_kW": over_max,
        "unmet_demand_kWh": unmet_tot / n,
        "synergy": rm._synergy(idx),
        "cost": inst.solution_cost(X),
        "n_stations": len(X),
    }


def describe_solution(inst, X: set) -> list:
    """Human-readable list of (zone id, level name) for reporting."""
    out = []
    for e in sorted(X):
        out.append((inst.zone_ids[inst.zone_of[e]],
                    inst.level_names[inst.level_of[e]]))
    return out
