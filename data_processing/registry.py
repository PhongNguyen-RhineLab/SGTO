"""Dataset registry: name -> loader plus per-dataset defaults.

Adding a dataset = writing one loader module with a build_instance(cfg)
function and registering it here. Defaults are applied to the config
BEFORE CLI overrides, so any flag the user passes explicitly wins.
"""

from dataclasses import dataclass, field


@dataclass
class DatasetDefaults:
    data_dir: str
    split_date: str          # train / held-out date boundary
    budget: float            # scaled to the instance size
    notes: str = ""
    build: object = None     # loader entry point, filled below
    configure: object = None # optional cfg hook for per-dataset scaling


def _urbanev(cfg):
    from data_processing.urbanev import build_instance
    return build_instance(cfg)


def _paris(cfg):
    from data_processing.paris import build_instance
    return build_instance(cfg)


def _paris_configure(cfg):
    """Scale the ground set to station-level (not TAZ-level) demand.

    UrbanEV zones aggregate hundreds of kWh/h, Belib' stations draw
    tens; keeping the Shenzhen levels makes even 'small' oversized and
    the instance degenerate. Costs stay in the same per-charger ballpark
    (AC ~6k USD, DC fast ~45k USD installed; 1 unit = 1000 USD).
    """
    from config import CapacityLevelConfig
    cfg.levels = (
        CapacityLevelConfig("small", 3, 7.4, 6.0),    # 22 kW,  cost 18
        CapacityLevelConfig("medium", 5, 22.0, 12.0), # 110 kW, cost 60
        CapacityLevelConfig("large", 6, 60.0, 45.0),  # 360 kW, cost 270
    )
    # station sites are points, not TAZ centroids: tighter coverage
    cfg.model.cov_decay_km = 1.0
    cfg.model.cov_radius_km = 3.0
    cfg.model.d_max_km = 3.0


DATASETS = {
    # 275 zones x 3 levels; all-medium build costs 82500 units
    "urbanev": DatasetDefaults(
        data_dir="UrbanEV/data", split_date="2023-01-15", budget=100000.0,
        notes="UrbanEV Shenzhen, primary instance", build=_urbanev),
    # 91 stations x 3 levels; all-medium build costs 5460 units
    "paris": DatasetDefaults(
        data_dir="smarter-mobility/data", split_date="2020-12-01",
        budget=7500.0,
        notes="Smarter Mobility Paris Belib', secondary instance",
        build=_paris, configure=_paris_configure),
}


def apply_dataset_defaults(cfg, name: str, overridden: set):
    """Set cfg fields from the dataset entry unless the CLI overrode them.

    overridden: set of field names the user set explicitly
    ("data_dir", "split_date", "budget").
    """
    if name not in DATASETS:
        raise SystemExit(
            f"unknown dataset '{name}'; available: {sorted(DATASETS)}")
    d = DATASETS[name]
    cfg.dataset = name
    if d.configure is not None:
        d.configure(cfg)
    if "data_dir" not in overridden:
        cfg.data_dir = d.data_dir
    if "split_date" not in overridden:
        cfg.scenarios.split_date = d.split_date
    if "budget" not in overridden:
        cfg.algo.budget = d.budget
    return d
