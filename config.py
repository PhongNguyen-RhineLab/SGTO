"""Central configuration for the EV charging station planning experiment.

Every constant that the paper treats as an assumption lives here so it can
be cited and varied in ablations. Units are noted next to each field.
"""

from dataclasses import dataclass, field


@dataclass
class CapacityLevelConfig:
    """One admissible capacity level l in L_i.

    Example: level "medium" installs 20 chargers at 30 kW average,
    so q_e = 600 kW, at a cost of 20 * 25 = 500 cost units
    (1 cost unit = 1000 USD, per-charger cost from literature ballparks:
    AC level 2 ~ 5k USD, DC fast ~ 40-100k USD installed).
    """

    name: str
    n_chargers: int
    avg_power_kw: float      # average rated power per charger
    cost_per_charger: float  # in cost units (1 unit = 1000 USD)

    @property
    def capacity_kw(self) -> float:
        return self.n_chargers * self.avg_power_kw

    @property
    def cost(self) -> float:
        return self.n_chargers * self.cost_per_charger


@dataclass
class ModelConfig:
    """Parameters of the reward function F_omega and F_rob."""

    # Reward weights (Eq. scenario_reward)
    alpha: float = 1.0    # coverage weight
    beta: float = 0.5     # synergy weight
    gamma: float = 1.0    # grid penalty weight
    eta: float = 2.0      # unmet demand weight

    # Risk term (Eq. robust_global_reward)
    rho: float = 0.3      # risk aversion
    delta: float = 0.9    # CVaR confidence level

    # Coverage: a_{u,e} = lvl_factor * exp(-dist_km / cov_decay_km),
    # truncated to 0 beyond cov_radius_km
    cov_decay_km: float = 2.0
    cov_radius_km: float = 5.0
    # effectiveness factor per capacity level (index-aligned with levels)
    lvl_factors: tuple = (0.5, 0.75, 0.95)
    # Congestion-aware effectiveness (the paper's "expected station
    # congestion" dependency of a_{u,e}): effectiveness is scaled by
    # sat_e = min(1, zeta_bar * q_e / attracted_e), where attracted_e is
    # the geo-weighted mean hourly demand the station would draw. A small
    # build in a dense zone is congested and covers little; the same
    # build in a sparse zone is fully effective. Computed from training
    # period demand only.
    use_congestion: bool = True

    # Synergy: b_ef = kappa if zones adjacent and road dist <= d_max_km
    kappa: float = 1.0
    d_max_km: float = 10.0

    # IEEE 33-bus grid provider: min bus voltage (pu) defining the
    # voltage-constrained hosting capacity (only used with --grid ieee33)
    ieee33_vmin: float = 0.90

    # Grid: g_{z,t} = grid_margin * (background peak + reference station load)
    # where reference load assumes grid_ref_frac of a district's zones get
    # a medium build. Together these control how tight the grid is.
    grid_margin: float = 1.05
    grid_ref_frac: float = 0.08
    # utilization rate zeta_{e,t}: hourly profile scaled to [zeta_min, zeta_max]
    zeta_min: float = 0.15
    zeta_max: float = 0.75

    # Served demand: s_{u,t} = min(d_{u,t}, serve_eff * sum_e a_ue q_e * zeta_t)
    serve_eff: float = 1.0

    # Time weights w_t: peak hours get extra weight
    peak_hours: tuple = (7, 8, 9, 17, 18, 19)
    w_peak: float = 1.5
    w_offpeak: float = 1.0

    # Reward normalization: coverage/unmet are in kWh, synergy in pair units.
    # Scale demand-based terms so magnitudes are comparable.
    demand_scale: float = 1e-3  # kWh -> MWh


@dataclass
class ScenarioConfig:
    """How scenarios are built from the UrbanEV history."""

    n_train: int = 20          # scenarios used for semi-gradient sampling pool
    n_val: int = 8             # validation scenarios Omega_val
    n_test: int = 12           # held-out test scenarios
    # fraction of scenarios per type in each pool
    frac_weekday: float = 0.4
    frac_weekend: float = 0.2
    frac_peak: float = 0.2     # highest-demand days
    frac_perturbed: float = 0.2  # grid reduction / demand surge
    grid_reduction: float = 0.7  # multiply g_{z,t} by this in perturbed scenarios
    demand_surge: float = 1.3    # multiply d_{u,t} in one random district
    # date split: days before this go to train pool, after to val/test pools
    split_date: str = "2023-01-15"
    seed: int = 42


@dataclass
class AlgoConfig:
    """SGTO and baseline hyperparameters."""

    budget: float = 100000.0     # B, cost units (= 5M USD); forces level tradeoffs
    max_iters: int = 20        # K
    n_sampled: int = 8         # m, scenarios sampled per iteration
    eps: float = 1e-3          # epsilon, minimum improvement
    # consecutive validation rejections tolerated before termination.
    # patience = 1 is the paper's rule (stop on first rejection); higher
    # values resample scenarios and retry, keeping the best-so-far X_k.
    patience: int = 3
    # On rejection, perturb the incumbent (iterated local search): drop
    # a random fraction of elements and greedily respend the budget.
    # Set to 0.0 to disable and recover plain resampling.
    perturb_frac: float = 0.34
    # After the loop, run one local-exchange polish on the full training
    # scenario set, accepted through the same validation gate.
    final_polish: bool = True
    exchange_max_passes: int = 3
    seed: int = 42


@dataclass
class ParisConfig:
    """Assumptions specific to the Smarter Mobility (Paris Belib') data.

    The raw data is plug OCCUPANCY (Available/Charging/Passive/Other
    counts per station every 15 min), not energy. Demand is derived as
    d_{u,t} = mean(#Charging over the hour) * plug_power_kw, in kWh.
    """

    train_csv: str = "train.csv"     # relative to data_dir
    plug_power_kw: float = 7.4       # avg AC power per occupied plug
    detour_factor: float = 1.3       # road dist ~ detour * great-circle
    adjacency_km: float = 1.5        # stations closer than this are adjacent
    district_by: str = "postcode"    # "postcode" (arrondissement) | "area"
    max_gap_hours: int = 3           # interpolate occupancy gaps up to this
    min_day_coverage: float = 1.0    # keep only fully observed days


@dataclass
class ExperimentConfig:
    # Which dataset/loader to use: "urbanev" | "paris"
    dataset: str = "urbanev"
    # Grid model: "synthetic" | "ieee33" (pandapower IEEE 33-bus)
    grid_model: str = "synthetic"
    # Road distances: "auto" (dataset file or OSMnx, fallback geodesic),
    # "osmnx" (strict), "geodesic" (haversine * detour factor)
    roads: str = "auto"
    data_dir: str = "UrbanEV/data"
    out_dir: str = "results"
    paris: ParisConfig = field(default_factory=ParisConfig)
    # Economies of scale: kW per cost unit rises with level
    # (AC level-2 heavy "small" vs DC-fast heavy "large"):
    # small 70 kW / 60 = 1.17, medium 450/300 = 1.5, large 2400/900 = 2.67
    levels: tuple = field(default_factory=lambda: (
        CapacityLevelConfig("small", 10, 7.0, 6.0),
        CapacityLevelConfig("medium", 15, 30.0, 20.0),
        CapacityLevelConfig("large", 20, 120.0, 45.0),
    ))
    model: ModelConfig = field(default_factory=ModelConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
