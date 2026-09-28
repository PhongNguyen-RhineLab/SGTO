# SGTO: Risk-Aware EV Charging Station Planning

Code and experiments for the paper

> **Time-Aware and Risk-Aware Electric Vehicle Charging Station Planning via
> Scenario-Based Global Trajectory Optimization**
> Phong T.D. Nguyen, Dung T.K. Ha, Thai V. Nguyen, Uyen T. Nguyen, Doan T. Hoang.
> CSoNet 2026 (Springer LNCS).

The repository implements the planning model (time-dependent coverage,
corridor synergy, grid overload penalty, unmet-demand penalty and a CVaR risk
term), the SGTO algorithm, three baselines, and the scripts that produce every
table in the paper.

The appendix of the paper is in [`paper/SGTO_Appendix.pdf`](paper/SGTO_Appendix.pdf).
It contains the extended related work, the closure properties (P1)-(P3) and the
CVaR variational form used in the analysis, the curvature analysis of each term of
the global reward and why the BP structure does not extend to `F_rob`, and the
additional experiments (Table 3: budget sweep, Table 4: enlarged 32-day test set).

A Vietnamese walkthrough of the code is in [`Instruction_vn.md`](Instruction_vn.md).

---

## Problem in one paragraph

Choose where to build EV charging stations and at which capacity level
(small / medium / large), at most one level per location, under an investment
budget. A plan is scored on a set of daily demand and grid scenarios by

```
F_w(X)  = alpha*C_w(X) + beta*Y(X) - gamma*P_w(X) - eta*U_w(X)
F_rob(X) = E_w[F_w(X)] - rho * CVaR_delta( gamma*P_w(X) + eta*U_w(X) )
```

where `C` is probabilistic demand coverage (submodular), `Y` rewards adjacent
station pairs that form a corridor (supermodular), `P` is a quadratic grid
overload penalty and `U` is unmet demand. The expected part is a BP
(submodular + supermodular) function; the CVaR term breaks that structure, so
SGTO is a BP-inspired heuristic with no approximation guarantee. Improvement is
enforced by a validation gate on held-out scenarios.

## SGTO in five steps

1. **Initialization**: best of a gain-per-cost greedy and a pure-gain greedy.
2. **Scenario-based semi-gradient**: sample `m` training scenarios and compute
   modular weights from the robust estimate of `F_rob`.
3. **Modular knapsack**: maximize the weights under the budget and the
   one-level-per-location constraint (exact dynamic programming).
4. **Local exchange**: add, swap and re-level moves, plus one drop-and-refill
   pass that respends freed budget greedily.
5. **Validation gate and restarts**: accept a candidate only if it improves the
   validation objective; after a rejection, perturb the incumbent (drop a
   fraction `pi` of stations, refill greedily) and retry, stopping after `R`
   consecutive rejections. A final exchange polish runs on the full training pool.

---

## Quick start

Requirements: Python 3.10+, `numpy`, `pandas`. No GPU is needed.

```bash
pip install -r requirements.txt
python setup_data.py urbanev          # clones UrbanEV into UrbanEV/data
python run_experiment.py --quick      # smoke test, about 2 minutes
```

Optional extras: `matplotlib` for `make_plots.py`, `pandapower` for
`--grid ieee33`, `osmnx` for `--roads osmnx`.

## Reproducing the paper

All numbers in Tables 1 to 4 of the paper come from one driver script:

```bash
python review_runs.py            # 66 runs, resumable; NPROC=2 by default
python review_analyze.py         # prints every table and the statistical tests
```

| Paper element | What `review_runs.py` runs |
|---|---|
| Table 1 (main results) | all four methods at `B = 100,000`; SA and SGTO over 10 algorithm seeds (42, 1 to 9), greedy baselines once (deterministic) |
| Paired tests in Section 4 | per-scenario rewards on an enlarged held-out set of 32 test days, paired bootstrap and Wilcoxon test |
| Table 2 (ablation) | SGTO with one component removed, 5 seeds (42, 1 to 4) |
| Table 3 (budget sweep, appendix) | `B` in {20k, 30k, 40k}; SA and SGTO over 3 seeds |
| Table 4 (enlarged test set, appendix) | the Table 1 plans re-evaluated on 32 test days |

The scenario split is fixed (scenario seed 42); only the algorithm seed varies.
Raw results are appended to `results/review/runs.jsonl` (one JSON record per
run, including the selected plan and per-scenario rewards and losses), and the
script skips runs that are already recorded, so it can be stopped and resumed.
The full suite takes about 70 minutes of wall time on 2 cores; set `NPROC` to
use more.

## Running single experiments

```bash
python run_experiment.py                              # UrbanEV, all methods
python run_experiment.py --methods sgto cost_aware_greedy
python run_experiment.py --budget 30000 --rho 0.5
python run_experiment.py --algo-seed 3                # vary the algorithm only
python run_experiment.py --n-test 40                  # larger held-out test set
python run_experiment.py --no-risk-in-weights         # mean-only semi-gradient
python run_experiment.py --dataset paris              # secondary instance
python run_experiment.py --list                       # datasets and methods
```

`--seed` changes both the scenario split and the algorithm; `--algo-seed` keeps
the scenarios fixed. Results are written to `results/<dataset>/results[_<tag>].json`
with the configuration, test metrics, the selected plan as (zone, level) pairs,
and the SGTO iteration history.

Methods: `cost_aware_greedy`, `greedy_one_exchange`, `simulated_annealing`,
`random_search`, `sgto`, and the variants `sgto_no_exchange` and
`sgto_risk_neutral`.

### Ablation switches

All SGTO settings live in `AlgoConfig` in `config.py`. The defaults are the
full method used in the paper.

| Field | Default | Effect |
|---|---|---|
| `max_iters` (K) | 20 | outer iterations |
| `n_sampled` (m) | 8 | scenarios sampled per iteration |
| `eps` | 1e-3 | minimum improvement for any acceptance |
| `patience` (R) | 3 | consecutive validation rejections before stopping |
| `perturb_frac` (pi) | 0.34 | fraction of the incumbent dropped on a restart; 0 disables restarts |
| `exchange_max_passes` | 3 | local-exchange passes per iteration |
| `final_polish` | True | exchange pass on the full training pool at the end |
| `risk_in_weights` | True | robust (True) or mean-only (False) semi-gradient weights |
| `k_drop` | 2 | elements tried by drop-and-refill; 0 disables it |
| `use_knapsack` | True | False skips the semi-gradient and knapsack step |

`patience=1, perturb_frac=0` recovers the terminate-on-first-rejection rule.

---

## Instance and parameters (UrbanEV, Shenzhen)

| Model object | Source or value |
|---|---|
| Demand regions and candidates | 275 traffic zones, green-field (`volume.csv`) |
| Ground set | 275 zones x 3 levels = 825 configurations |
| Capacity levels | small 10 x 7 kW, medium 15 x 30 kW, large 20 x 120 kW |
| Level costs | 60 / 300 / 900 cost units (1 unit ~ USD 1000) |
| Budget | `B = 100,000` |
| Demand `d_{u,t}` | hourly charging volume (kWh), one scenario = one day, T = 24 |
| Coverage kernel | `exp(-dist / 2 km)`, truncated at 5 km road distance (`distance.csv`) |
| Level service factors `sigma_l` | 0.5 / 0.75 / 0.95 |
| Congestion saturation | `min(1, zeta_bar * q_e / att(i))`, `zeta_bar = 0.45`, training demand only |
| Utilization profile | city demand shape rescaled to [0.15, 0.75] |
| Synergy | `kappa = 1` per adjacent zone pair (`adj.csv`) within `D_max = 10 km` |
| Grid districts | 11, from `TAZID // 100` |
| Grid capacity | synthetic: `g = 1.05 x (peak background + reference load)`, reference = medium builds in 8% of a district's zones |
| Time weights | peak hours (7-9, 17-19) weight 1.5, otherwise 1.0 |
| Reward weights | `alpha = 1, beta = 0.5, gamma = 1, eta = 2` |
| Risk | `rho = 0.3`, `delta = 0.9` |
| Scenarios | 20 train (before 2023-01-15), 8 validation, 12 test (after); weekday / weekend / peak / perturbed mix |
| Perturbations | grid headroom x 0.7, or demand x 1.3 in one district |

Validation and test days are disjoint from training days, and from each other.

### Stated assumptions

1. Grid capacity is synthetic (see the table). An IEEE 33-bus provider is
   available with `--grid ieee33` (requires `pandapower`).
2. Level costs are literature ballparks for installed chargers.
3. Effective capacity aggregates additively across stations before the demand
   cap, `s = min(d, nu * zeta_t * sum_e a_{u,e} q_e)`, which keeps the
   unmet-demand term supermodular.
4. Reported `F_rob_gain` is `F_rob(X) - F_rob(empty plan)`, since the unmet
   demand of building nothing makes raw `F_rob` a large negative constant.

### Secondary instance (Paris Belib')

`--dataset paris` builds a 91-station instance from the Smarter Mobility data
challenge (`python setup_data.py paris`), with demand from hourly plug
occupancy and grid regions from arrondissements. It is not used in the paper.

---

## Repository layout

```
config.py                  every model, scenario and algorithm parameter
setup_data.py              dataset download (git clone)
run_experiment.py          single-run entry point
review_runs.py             reproduces all paper tables (multi-seed, ablation, budget)
review_analyze.py          aggregates runs.jsonl into tables and statistical tests
run_all.sh                 older staged suite (rho and weight sweeps, Paris calibration)
make_tables.py             LaTeX tables from results_*.json
make_plots.py              figures from results_*.json
metrics.py                 test-set metrics (gain, worst case, CVaR, FR, synergy, cost)
data_processing/
  registry.py              dataset name -> loader and defaults
  urbanev.py, paris.py     instance builders
  scenarios.py             train / validation / test scenario sampling
  common.py                load curve, synthetic grid
  grid_ieee33.py           IEEE 33-bus grid provider (optional)
  roads_osmnx.py           OSM road distances (optional)
model/
  instance.py              ProblemInstance and Scenario
  reward.py                F_omega, CVaR, F_rob, vectorized incremental gains
  reward_reference.py      slow reference implementation used for testing
paper/
  SGTO_Appendix.pdf    appendix of the paper (proofs and additional experiments)
algorithms/
  greedy.py                cost-aware greedy and greedy fill
  local_search.py          one-exchange, drop-and-refill, greedy + exchange baseline
  annealing.py             simulated annealing baseline
  random_search.py         random feasible plans
  sgto.py                  SGTO
results/review/            raw runs and summary behind the paper tables
```

Algorithms only see a `ProblemInstance`. Adding a dataset means one loader
exposing `build_instance(cfg)` and one entry in `data_processing/registry.py`.

## Known limitations

- Grid capacities are synthetic, so the case study tests the algorithm under a
  plausible grid proxy rather than giving an infrastructure recommendation.
- There is no exact or relaxation-based reference solution.
- The ablation shows the gain over the baselines comes mainly from the
  validation-gated perturbation restarts; the knapsack phase and
  drop-and-refill have effects within seed-to-seed noise on this instance.
- When the budget binds (B <= 40,000), SGTO ties greedy with one exchange.

## Citation

```bibtex
@inproceedings{nguyen2026sgto,
  title     = {Time-Aware and Risk-Aware Electric Vehicle Charging Station
               Planning via Scenario-Based Global Trajectory Optimization},
  author    = {Nguyen, Phong T.D. and Ha, Dung T.K. and Nguyen, Thai V. and
               Nguyen, Uyen T. and Hoang, Doan T.},
  booktitle = {Computational Data and Social Networks (CSoNet 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

The UrbanEV data is from Li et al., *UrbanEV: An open benchmark dataset for
urban electric vehicle charging demand prediction*, Scientific Data 12, 523
(2025); cite it if you use the instance.

## License

MIT, see [`LICENSE`](LICENSE).
