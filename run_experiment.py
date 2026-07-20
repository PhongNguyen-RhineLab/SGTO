"""Run the EV charging station planning experiment.

Datasets, grid models and road distances are selected by flags; every
combination reuses the same model and algorithms.

Usage examples
--------------
    python run_experiment.py                          # UrbanEV, all methods
    python run_experiment.py --quick                  # smoke test, ~2 min
    python run_experiment.py --dataset paris          # Paris Belib' instance
    python run_experiment.py --dataset urbanev --grid ieee33
    python run_experiment.py --dataset paris --roads osmnx
    python run_experiment.py --methods sgto cost_aware_greedy
    python run_experiment.py --budget 50000 --rho 0.5 --seed 7
    python run_experiment.py --list                   # datasets and methods
"""

import argparse
import json
import os

from config import ExperimentConfig
from data_processing.registry import DATASETS, apply_dataset_defaults
from model.reward import RewardModel
from metrics import evaluate, describe_solution
from algorithms.greedy import CostAwareGreedy
from algorithms.local_search import GreedyWithExchange
from algorithms.random_search import RandomSearch
from algorithms.annealing import SimulatedAnnealing
from algorithms.sgto import SGTO


def make_solvers(inst, rm, cfg, names=None):
    all_solvers = {
        "random_search": lambda: RandomSearch(inst, rm, cfg.algo),
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
        unknown = [n for n in names if n not in all_solvers]
        if unknown:
            raise SystemExit(f"unknown methods {unknown}; "
                             f"available: {sorted(all_solvers)}")
        return {n: all_solvers[n] for n in names}
    return all_solvers


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = ap.add_argument_group("dataset")
    g.add_argument("--dataset", default="urbanev",
                   choices=sorted(DATASETS),
                   help="which instance to build (default: urbanev)")
    g.add_argument("--data-dir", default=None,
                   help="override the dataset's default data directory")
    g.add_argument("--split-date", default=None,
                   help="train/held-out date boundary (YYYY-MM-DD; "
                        "default depends on dataset)")

    g = ap.add_argument_group("physical model")
    g.add_argument("--grid", default="synthetic",
                   choices=["synthetic", "ieee33"],
                   help="grid capacity model; ieee33 uses pandapower "
                        "voltage-constrained hosting capacity")
    g.add_argument("--roads", default="auto",
                   choices=["auto", "osmnx", "geodesic"],
                   help="road distances: dataset file / OSMnx / "
                        "great-circle x detour (default: auto)")

    g = ap.add_argument_group("optimization")
    g.add_argument("--methods", nargs="*", default=None,
                   help="subset of methods to run (default: all)")
    g.add_argument("--budget", type=float, default=None,
                   help="budget B in cost units (default per dataset)")
    g.add_argument("--rho", type=float, default=None,
                   help="risk aversion weight (default: config)")
    g.add_argument("--seed", type=int, default=None,
                   help="seed for scenarios AND algorithms")
    g.add_argument("--algo-seed", type=int, default=None,
                   help="algorithm seed only (scenarios stay fixed); "
                        "use for multi-seed ablations on one test set")
    g.add_argument("--no-risk-in-weights", action="store_true",
                   help="mean-only semi-gradient weights "
                        "(original behavior, for the ablation)")
    g.add_argument("--demand-growth", type=float, default=None,
                   help="paris only: scale demand by this factor "
                        "(EV adoption growth assumption)")
    g.add_argument("--n-train", type=int, default=None)
    g.add_argument("--n-val", type=int, default=None)
    g.add_argument("--n-test", type=int, default=None)

    g = ap.add_argument_group("run control")
    g.add_argument("--quick", action="store_true",
                   help="tiny config for a fast smoke test")
    g.add_argument("--out", default=None,
                   help="output directory (default: results/<dataset>)")
    g.add_argument("--tag", default=None,
                   help="suffix for the results file name")
    g.add_argument("--list", action="store_true",
                   help="list datasets and methods, then exit")
    return ap.parse_args()


def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig()

    overridden = set()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
        overridden.add("data_dir")
    if args.split_date is not None:
        cfg.scenarios.split_date = args.split_date
        overridden.add("split_date")
    if args.budget is not None:
        cfg.algo.budget = args.budget
        overridden.add("budget")
    apply_dataset_defaults(cfg, args.dataset, overridden)

    cfg.grid_model = args.grid
    cfg.roads = args.roads
    if args.rho is not None:
        cfg.model.rho = args.rho
    if args.seed is not None:
        cfg.scenarios.seed = args.seed
        cfg.algo.seed = args.seed
    if args.algo_seed is not None:
        cfg.algo.seed = args.algo_seed
    if args.no_risk_in_weights:
        cfg.algo.risk_in_weights = False
    if args.demand_growth is not None:
        cfg.paris.demand_growth = args.demand_growth
    for k in ("n_train", "n_val", "n_test"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg.scenarios, k, v)
    if args.quick:
        cfg.scenarios.n_train = 6
        cfg.scenarios.n_val = 3
        cfg.scenarios.n_test = 4
        cfg.algo.max_iters = 3
        cfg.algo.n_sampled = 3
        cfg.algo.exchange_max_passes = 1
    cfg.out_dir = args.out or os.path.join("results", args.dataset)
    return cfg


def main():
    args = parse_args()
    if args.list:
        print("datasets:")
        for name, d in DATASETS.items():
            print(f"  {name:<10} data_dir={d.data_dir}  budget={d.budget:.0f}"
                  f"  split={d.split_date}  ({d.notes})")
        print("methods:")
        for n in make_solvers(None, None, ExperimentConfig()):
            print(f"  {n}")
        return

    cfg = build_config(args)
    print(f"Building instance: dataset={cfg.dataset}  grid={cfg.grid_model}"
          f"  roads={cfg.roads}  data_dir={cfg.data_dir}")
    inst = DATASETS[cfg.dataset].build(cfg)
    rm = RewardModel(inst, cfg.model)
    print(f"  zones={inst.n_zones} elements={inst.n_elements} "
          f"grid regions={inst.n_grid} "
          f"scenarios train/val/test = {len(inst.scen_train)}/"
          f"{len(inst.scen_val)}/{len(inst.scen_test)}")
    print(f"  budget={inst.budget} cost units, "
          f"level costs={sorted(set(inst.cost.tolist()))}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    results = {"_config": {
        "dataset": cfg.dataset, "grid_model": cfg.grid_model,
        "roads": cfg.roads, "budget": cfg.algo.budget,
        "rho": cfg.model.rho, "seed": cfg.algo.seed,
        "demand_growth": cfg.paris.demand_growth,
        "scenario_seed": cfg.scenarios.seed,
        "risk_in_weights": cfg.algo.risk_in_weights,
        "split_date": cfg.scenarios.split_date,
    }}
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

    fname = "results.json" if not args.tag else f"results_{args.tag}.json"
    out_path = os.path.join(cfg.out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved", out_path)

    print("\n{:<22} {:>10} {:>10} {:>8} {:>9} {:>10} {:>9}".format(
        "method", "F_rob", "gain", "FR", "CVaR", "cost", "time_s"))
    for name, r in results.items():
        if name.startswith("_"):
            continue
        m = r["metrics"]
        print("{:<22} {:>10.2f} {:>10.2f} {:>8.3f} {:>9.2f} {:>10.0f} {:>9.1f}"
              .format(name, m["F_rob"], m["F_rob_gain"],
                      m["fulfillment_ratio"], m["CVaR_loss"],
                      m["cost"], m["runtime_s"]))


if __name__ == "__main__":
    main()
