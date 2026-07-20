"""Aggregate results_*.json files into paper-ready LaTeX tables.

Modes
-----
table  one run -> methods x metrics table
    python make_tables.py table results/urbanev/results_main.json

seeds  several runs of the same setup -> mean +- std per method
    python make_tables.py seeds results/urbanev/results_main.json \
        results/urbanev/results_rho_seed*.json --methods sgto sgto_risk_neutral

sweep  runs differing in one config value -> value vs metrics
    python make_tables.py sweep results/urbanev/results_rho*.json --param rho

Notes: F_rob and F_rob_gain depend on the run's rho, so they are NOT
comparable across a rho sweep; sweep mode therefore defaults to
rho-independent metrics (F_mean, CVaR_loss, overload, cost). Repeated
values of the sweep parameter are aggregated as mean +- std.
"""

import argparse
import glob
import json
import sys

import numpy as np

DEFAULT_METRICS = ["F_rob", "CVaR_loss", "fulfillment_ratio",
                   "grid_overload_total_kW", "cost", "runtime_s"]
SWEEP_METRICS = ["F_mean", "CVaR_loss", "fulfillment_ratio",
                 "grid_overload_total_kW", "cost", "n_stations"]
HEADERS = {
    "F_rob": r"$F_{\mathrm{rob}}$", "F_mean": r"$\bar F$",
    "F_worst": r"$F_{\min}$", "CVaR_loss": r"CVaR$_\alpha$",
    "fulfillment_ratio": "FR", "peak_coverage_ratio": "peakCov",
    "grid_overload_total_kW": "overload (kW)",
    "grid_overload_max_kW": "max ovl (kW)",
    "unmet_demand_kWh": "unmet (kWh)", "synergy": "synergy",
    "cost": "cost", "n_stations": "\\#st", "runtime_s": "time (s)",
    "n_evals": "evals",
}
FMT = {"fulfillment_ratio": "{:.3f}", "peak_coverage_ratio": "{:.3f}",
       "runtime_s": "{:.1f}"}


def _fmt(metric, v):
    return FMT.get(metric, "{:.2f}" if abs(v) < 1000 else "{:.0f}").format(v)


def _load(paths):
    out = []
    for pat in paths:
        hits = sorted(glob.glob(pat)) or [pat]
        for p in hits:
            with open(p) as f:
                out.append((p, json.load(f)))
    if not out:
        sys.exit("no result files matched")
    return out


def _emit(headers, rows, caption):
    print(r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}")
    print(r"\toprule")
    print(" & ".join(headers) + r" \\")
    print(r"\midrule")
    for r in rows:
        print(" & ".join(r) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"% {caption}")


def mode_table(args):
    path, res = _load(args.files)[0]
    methods = args.methods or [k for k in res if not k.startswith("_")]
    rows = []
    for m in methods:
        met = res[m]["metrics"]
        rows.append([m.replace("_", r"\_")]
                    + [_fmt(k, met[k]) for k in args.metrics])
    _emit(["method"] + [HEADERS.get(k, k) for k in args.metrics], rows,
          f"source: {path}, config: {res.get('_config', {})}")


def mode_seeds(args):
    files = _load(args.files)
    methods = args.methods or sorted({m for _, r in files for m in r
                                      if not m.startswith("_")})
    rows = []
    for m in methods:
        vals = {k: [] for k in args.metrics}
        n = 0
        for _, r in files:
            if m in r:
                n += 1
                for k in args.metrics:
                    vals[k].append(r[m]["metrics"][k])
        if n == 0:
            continue
        cells = [m.replace("_", r"\_")]
        for k in args.metrics:
            a = np.array(vals[k], dtype=float)
            cells.append(f"{_fmt(k, a.mean())} $\\pm$ "
                         f"{_fmt(k, a.std(ddof=1) if len(a) > 1 else 0.0)}")
        rows.append(cells)
    _emit(["method"] + [HEADERS.get(k, k) for k in args.metrics], rows,
          f"mean +- std over {len(files)} runs "
          f"(seeds: {[r.get('_config', {}).get('seed') for _, r in files]})")


def mode_sweep(args):
    files = _load(args.files)
    groups = {}
    for p, r in files:
        cfg = r.get("_config", {})
        if args.param not in cfg:
            print(f"% skip {p}: no _config.{args.param}", file=sys.stderr)
            continue
        for m in (args.methods or [k for k in r if not k.startswith("_")]):
            if m in r:
                groups.setdefault((cfg[args.param], m), []).append(
                    r[m]["metrics"])
    rows = []
    for (val, m), mets in sorted(groups.items()):
        cells = [f"{val}" + (f" ({m.replace('_', chr(92)+'_')})"
                             if len({k[1] for k in groups}) > 1 else "")]
        for k in args.metrics:
            a = np.array([mm[k] for mm in mets], dtype=float)
            s = _fmt(k, a.mean())
            if len(a) > 1:
                s += f" $\\pm$ {_fmt(k, a.std(ddof=1))}"
            cells.append(s)
        rows.append(cells)
    _emit([args.param] + [HEADERS.get(k, k) for k in args.metrics], rows,
          f"sweep over _config.{args.param}, {len(files)} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["table", "seeds", "sweep"])
    ap.add_argument("files", nargs="+", help="result json files or globs")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--param", default="rho",
                    help="sweep mode: _config key to group by")
    args = ap.parse_args()
    if args.metrics is None:
        args.metrics = SWEEP_METRICS if args.mode == "sweep" \
            else DEFAULT_METRICS
    {"table": mode_table, "seeds": mode_seeds, "sweep": mode_sweep}[args.mode](args)


if __name__ == "__main__":
    main()
