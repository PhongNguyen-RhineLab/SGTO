# SGTO Experiment: EV Charging Station Planning on UrbanEV (Shenzhen)

Code for the experiments of the risk-aware, scenario-based facility
location paper. Implements the full model (coverage, synergy, grid
penalty, unmet demand, CVaR) and the SGTO algorithm plus baselines,
on the UrbanEV Shenzhen dataset.

## Setup

```bash
pip install numpy pandas
git clone --depth 1 https://github.com/IntelligentSystemsLab/UrbanEV.git
python run_experiment.py --quick            # smoke test, ~2 min
python run_experiment.py                    # full run
python run_experiment.py --methods sgto cost_aware_greedy
```

Results land in `results/results.json` with metrics, solution
(zone id, level) lists, and SGTO iteration history.

## Structure

```
config.py                  all assumptions and hyperparameters in one place
data_processing/
  urbanev.py               loads UrbanEV csvs, builds the ProblemInstance
  scenarios.py             builds train/val/test scenario sets from real days
model/
  instance.py              ProblemInstance and Scenario dataclasses
  reward.py                F_omega, CVaR, F_rob, IncrementalState (fast gains)
algorithms/
  base.py                  Solver interface, SolveResult
  greedy.py                cost-aware greedy (also SGTO initializer)
  local_search.py          one-exchange improvement, greedy+exchange baseline
  sgto.py                  full SGTO: semi-gradient, modular knapsack DP,
                           exchange, validation acceptance; ablation flags
  random_search.py         random feasible baseline
metrics.py                 all paper metrics on held-out test scenarios
run_experiment.py          entry point
```

Algorithms only see a `ProblemInstance`, never raw files, so adding the
Paris instance later means writing one new loader.

## Mapping data to the model

| Model object | UrbanEV source |
|---|---|
| demand regions U (275) | volume.csv columns (traffic zones) |
| candidates V | same zones, green-field planning |
| ground set E (825) | zones x 3 capacity levels (config.py) |
| d_u,t | hourly charging volume (kWh), one scenario = one day, T=24 |
| coverage a_u,e | exp decay of road distance (distance.csv) x level factor |
| synergy pairs | adjacent zones (adj.csv) within D_max road distance |
| grid regions Z (9) | district groups from TAZID prefix |
| scenarios | real weekday/weekend/peak days + grid-cut and surge perturbations |

Train scenarios come from days before 2023-01-15; validation and test
from disjoint later days, so validation-based acceptance and final
evaluation never see training days.

## Stated assumptions (to cite in the paper)

1. Grid capacity is synthetic: background load per district follows a
   standard daily feeder curve scaled to district charging demand, and
   g_z,t = margin x (peak background + reference station load). Replace
   with IEEE 33-bus via pandapower later; hooks are in scenarios.py.
2. Costs per level use literature ballparks (config.py CapacityLevelConfig,
   1 unit = 1000 USD). Needs a citation row in the dataset table.
3. Served demand s_u,t = min(d, serve_eff x zeta_t x sum a_ue q_e),
   which satisfies the submodularity assumption (Assumption 1) of the
   theory section.
4. Utilization zeta_t follows the city demand shape rescaled to
   [zeta_min, zeta_max].

## Fixed after the first full run (session 2)

- Gridcut scenarios now scale headroom above background load instead of
  total capacity. The old rule created overload that existed even with
  zero stations, a constant loss that made the CVaR term unable to
  discriminate between solutions (risk-aware == risk-neutral).
- Effectiveness a_{u,e} is now congestion-aware (use_congestion in
  config): scaled by min(1, zeta_bar q_e / attracted demand), computed
  from training-period demand only. This implements the paper's
  "expected station congestion" dependency and makes the capacity-level
  choice non-degenerate: large builds appear in dense zones, small in
  sparse ones. Level economics also updated to reflect economies of
  scale (kW per cost unit rises with level).
- District mapping fixed: TAZID // 100 (11 districts), not the first
  character (which merged zone 1011 into district 1).
- SGTO acceptance has a patience parameter (default 3 rejections;
  patience=1 recovers the paper's stop-on-first-rejection rule). If
  keeping patience > 1, add one sentence to the algorithm section.
- Metrics report F_rob_gain = F_rob(X) - F_rob(empty), since unserved
  baseline demand makes raw F_rob a large negative constant.

## Known calibration items (still open)

- Risk-aware and risk-neutral SGTO still find the same solution at
  rho=0.3: the CVaR is dominated by the unavoidable unmet-demand mass,
  so the rho ablation needs either larger rho, a loss defined net of
  the empty-solution baseline, or tighter grid scenarios.
- Weight sweep over (alpha, beta, gamma, eta, budget) for the operating
  point of the main table; the ablation list in the paper covers these.
- Missing baselines: genetic algorithm, simulated annealing, original
  static GTO, and a MILP reference for small instances. The Solver
  interface in algorithms/base.py is ready.
- Routes currently use zone adjacency; swap in OSMnx corridors if
  reviewers want literal road routes.
