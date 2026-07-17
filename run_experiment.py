"""Run the UrbanEV Shenzhen experiment.

Usage:
    python run_experiment.py                 # all methods
    python run_experiment.py --quick         # small config, smoke test
    python run_experiment.py --methods sgto cost_aware_greedy
"""

import argparse
import json
import os

import numpy as np

from config import ExperimentConfig
from data_processing.urbanev import build_instance
from model.reward import RewardModel
from metrics import evaluate, describe_solution
from algorithms.greedy import CostAwareGreedy
from algorithms.local_search import GreedyWithExchange
from algorithms.random_search import RandomSearch
from algorithms.annealing import SimulatedAnnealing
from algorithms.sgto import SGTO
from algorithms.annealing import SimulatedAnnealing


def make_solvers(inst, rm, cfg, names=None):
    all_solvers = {
        "random_search": lambda: RandomSearch(inst, rm, cfg.algo),
        "simulated_annealing": lambda: SimulatedAnnealing(inst, rm, cfg.algo),
        "simulated_annealing": lambda: SimulatedAnnealing(inst, rm, cfg.algo),
        "cost_aware_greedy": lambda: CostAwareGreedy(inst, rm, cfg.algo),
        "greedy_one_exchange": lambda: GreedyWithExchange(inst, rm, cfg.algo),
        "sgto_no_exchange": lambda: SGTO(inst, rm, cfg.algo,
                                         use_exchange=False),
        "sgto_risk_neutral": lambda: SGTO(inst, rm, cfg.algo,
                                          risk_aware=False,
                                          name="sgto_risk_neutral"),
        "sgto": lambda: SGTO(inst, rm, cfg.algo),
    }
    if names:
        return {n: all_solvers[n] for n in names}
    return all_solvers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny config for a fast smoke test")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.quick:
        cfg.scenarios.n_train = 6
        cfg.scenarios.n_val = 3
        cfg.scenarios.n_test = 4
        cfg.algo.max_iters = 3
        cfg.algo.n_sampled = 3
        cfg.algo.exchange_max_passes = 1

    print("Building instance from", cfg.data_dir)
    inst = build_instance(cfg)
    rm = RewardModel(inst, cfg.model)
    print(f"  zones={inst.n_zones} elements={inst.n_elements} "
          f"grid regions={inst.n_grid} "
          f"scenarios train/val/test = {len(inst.scen_train)}/"
          f"{len(inst.scen_val)}/{len(inst.scen_test)}")
    print(f"  budget={inst.budget} cost units, "
          f"level costs={sorted(set(inst.cost.tolist()))}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    results = {}
    for name, mk in make_solvers(inst, rm, cfg, args.methods).items():
        print(f"\n=== {name} ===")
        res = mk().solve()
        m = evaluate(rm, res.X)
        m["runtime_s"] = round(res.runtime_s, 2)
        m["n_evals"] = res.n_evals
        results[name] = {
            "metrics": m,
            "solution": describe_solution(inst, res.X),
            "history": res.history,
        }
        print(f"  F_rob(test)={m['F_rob']:.2f}  FR={m['fulfillment_ratio']:.3f}  "
              f"peakCov={m['peak_coverage_ratio']:.3f}")
        print(f"  overload tot/max={m['grid_overload_total_kW']:.0f}/"
              f"{m['grid_overload_max_kW']:.0f} kW  "
              f"unmet={m['unmet_demand_kWh']:.0f} kWh")
        print(f"  stations={m['n_stations']}  cost={m['cost']:.0f}  "
              f"synergy={m['synergy']:.1f}  "
              f"time={m['runtime_s']}s  evals={m['n_evals']}")

    out_path = os.path.join(cfg.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved", out_path)

    # summary table
    print("\n{:<22} {:>10} {:>10} {:>8} {:>9} {:>10} {:>9}".format(
        "method", "F_rob", "gain", "FR", "CVaR", "cost", "time_s"))
    for name, r in results.items():
        m = r["metrics"]
        print("{:<22} {:>10.2f} {:>10.2f} {:>8.3f} {:>9.2f} {:>10.0f} {:>9.1f}"
              .format(name, m["F_rob"], m["F_rob_gain"],
                      m["fulfillment_ratio"], m["CVaR_loss"],
                      m["cost"], m["runtime_s"]))


if __name__ == "__main__":
    main()
