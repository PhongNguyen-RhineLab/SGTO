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

## Algorithm and performance upgrades (session 3)

- Dual-rule greedy: runs benefit-to-cost AND pure-gain greedy, keeps
  the better (guards against the known failure mode of ratio greedy
  with heterogeneous costs; carries the classic budgeted-max-coverage
  style guarantee).
- Local exchange gained a drop-and-refill move: drop one of the k=3
  weakest elements and greedily respend the freed budget, covering
  1-to-many trades that single exchanges cannot express.
- SGTO is now iterated local search: on validation rejection it
  perturbs the incumbent (drop perturb_frac of elements, greedy refill)
  instead of terminating, tracks the best-so-far validated solution,
  and finishes with a local-exchange polish on the full training set.
  patience=1, perturb_frac=0, final_polish=False recovers the paper's
  Algorithm 1 exactly; if the extensions are kept, the algorithm
  section needs a short paragraph.
- Simulated annealing baseline implemented (algorithms/annealing.py).
- Evaluator rewritten (model/reward.py): scenario tensors are stacked
  and marginal gains use component deltas on slices (a candidate only
  touches its own grid region and the ~18 zones it covers). ~6-7x
  faster per gain, verified bit-identical to the reference
  implementation kept in model/reward_reference.py. End to end: full
  SGTO 53s -> 9.5s, and budget-12000 runs that previously timed out
  finish in under a minute.

## Optimality evidence at the default operating point

At budget 5000 the instance appears ceiling-bound: aggressive SGTO
(40 iters, 50 percent perturbation, patience 12) and a 12000-move SA
run from random init both fail to beat the same solution SGTO finds in
10 seconds (val -3393.95). The greedy -> SGTO gap (gain 689 -> 736,
+6.9 percent) is the headline comparison. At budget 12000 SGTO
additionally cuts max grid overload from 2607 to 1841 kW vs greedy.

## Algorithm upgrades (session 3)

- Vectorized reward evaluation: scenario data is stacked into tensors
  in IncrementalState (coverage collapses to one (M,U) product since
  w_t and demand factor out of the solution term). ~6x faster per
  marginal gain, bit-identical to the reference implementation kept in
  model/reward_reference.py. This is what makes larger budgets (more
  stations, bigger neighborhoods) tractable.
- Dual-rule greedy: best of ratio-greedy and gain-greedy (classic fix
  for the budgeted-coverage failure mode of the pure ratio rule).
- Drop-and-refill move in local exchange: one-exchange cannot trade one
  expensive element for several cheap ones; dropping the weakest
  elements and greedily respending the budget covers 1-to-many moves.
  Runs once after the exchange passes converge (running it inside every
  pass made large budgets intractable).
- SGTO is now iterated local search: on validation rejection it
  perturbs the incumbent (drop perturb_frac of elements, greedy refill)
  instead of terminating, tracks the best-so-far validated solution,
  and optionally polishes on the full train set at the end. Setting
  patience=1, perturb_frac=0, final_polish=False recovers the paper's
  Algorithm 1 exactly; if the extensions are kept, the algorithm
  section needs a short paragraph describing them.
- Simulated annealing baseline implemented (algorithms/annealing.py);
  also used as an independent check.

Evidence at budget 12000 (single seed): greedy gain 1446.2, paper-rule
SGTO 1449.3 (stops at iteration 1), extended SGTO 1457.8 with max
overload reduced 2607 -> 1841 kW; the accepted improvements at
iterations 2/4/6/9 were all reached via perturbation restarts. At
budget 5000 the instance is effectively saturated: aggressive search
and an independent SA both fail to beat SGTO's solution, so method
differences there are small by nature; the main-table experiments
should include at least one larger-budget setting.

## Known calibration items (still open)

- Risk-aware and risk-neutral SGTO still find the same solution at
  rho=0.3: the CVaR is dominated by the unavoidable unmet-demand mass,
  so the rho ablation needs either larger rho, a loss defined net of
  the empty-solution baseline, or tighter grid scenarios.
- Weight sweep over (alpha, beta, gamma, eta, budget) for the operating
  point of the main table; the ablation list in the paper covers these.
- Missing baselines: genetic algorithm, original static GTO, and a
  MILP reference for small instances (simulated annealing done). The
  Solver interface in algorithms/base.py is ready.
- Synergy is near zero in optimized solutions (0-2 pairs): adjacency
  is a stricter condition than the paper's same-route-within-D_max.
  Building explicit corridors (shortest paths through the adjacency
  graph, or OSMnx arterials) would revive the Y term; this changes the
  route definition, so it is a modeling decision to make deliberately.
- Routes currently use zone adjacency; swap in OSMnx corridors if
  reviewers want literal road routes.
