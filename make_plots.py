"""Plot paper figures from results_*.json files (companion to make_tables.py).

Modes
-----
methods  bar chart comparing methods from one run
    python make_plots.py methods results/urbanev/results_main.json -o figures

sweep    parameter vs metrics curves (rho sweep, demand-growth sweep, ...)
    python make_plots.py sweep "results/urbanev/results_rho[0-9]*.json" \
        --param rho --methods sgto -o figures
    python make_plots.py sweep "results/paris/results_growth*.json" \
        --param demand_growth \
        --methods cost_aware_greedy sgto sgto_risk_neutral -o figures

history  SGTO validation value per iteration (accepted vs rejected)
    python make_plots.py history results/urbanev/results_main.json \
        --methods sgto sgto_risk_neutral -o figures

Notes
-----
- Figures are saved as both .pdf (for LaTeX) and .png.
- If _config lacks the sweep parameter (older result files), the value
  is recovered from the file name (e.g. results_growth3.json -> 3).
- F_rob / F_rob_gain are computed with the run's own rho, so sweep mode
  defaults to rho-independent metrics (CVaR_loss, F_mean, overload,
  cost); do not add F_rob to a rho sweep.
- history mode limits the y-axis to the range of the init and accepted
  values; wildly rejected proposals may fall outside the frame.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "F_rob": r"$F_{\mathrm{rob}}$", "F_rob_gain": r"$F_{\mathrm{rob}}$ gain",
    "F_mean": r"$\bar F$", "CVaR_loss": r"CVaR$_\alpha$ loss",
    "fulfillment_ratio": "fulfillment ratio",
    "peak_coverage_ratio": "peak coverage",
    "grid_overload_total_kW": "total overload (kW)",
    "grid_overload_max_kW": "max overload (kW)",
    "unmet_demand_kWh": "unmet demand (kWh)",
    "cost": "cost", "n_stations": "stations", "runtime_s": "time (s)",
}
NICE = {"cost_aware_greedy": "Greedy", "greedy_one_exchange": "Greedy+Exch",
        "sgto_no_exchange": "SGTO w/o exch", "sgto_risk_neutral": "SGTO (RN)",
        "sgto": "SGTO", "random_search": "Random",
        "simulated_annealing": "Sim. Annealing"}


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


def _param_value(path, cfg, param):
    if param in cfg and cfg[param] is not None:
        return float(cfg[param])
    # fall back to the file name: try the full param name, then its
    # last word ("demand_growth" -> "growth3" in results_growth3.json)
    base = os.path.basename(path)
    for key in (param, param.split("_")[-1]):
        m = re.search(re.escape(key) + r"[_-]?([0-9]+(?:\.[0-9]+)?)", base)
        if m:
            return float(m.group(1))
    sys.exit(f"cannot determine {param} for {path}: not in _config "
             "and not in the file name")


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("saved", path)
    plt.close(fig)


def mode_methods(args):
    path, res = _load(args.files)[0]
    methods = args.methods or [k for k in res if not k.startswith("_")]
    metrics = args.metrics or ["F_rob_gain", "CVaR_loss"]
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3.2 * len(metrics), 0.45 * len(methods) + 1.2))
    axes = np.atleast_1d(axes)
    names = [NICE.get(m, m) for m in methods]
    for ax, k in zip(axes, metrics):
        vals = [res[m]["metrics"][k] for m in methods]
        ax.barh(names, vals, color="#4878a8")
        ax.set_xlabel(LABELS.get(k, k))
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}",
                        (v, i), va="center",
                        ha="left" if v >= 0 else "right", fontsize=8)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    tag = res.get("_config", {}).get("dataset", "run")
    fig.suptitle(f"{tag}", fontsize=10)
    fig.tight_layout()
    _save(fig, args.out, f"methods_{os.path.splitext(os.path.basename(path))[0]}")


def mode_sweep(args):
    files = _load(args.files)
    metrics = args.metrics or ["CVaR_loss", "F_mean",
                               "grid_overload_total_kW", "cost"]
    # (param value, method) -> list of metric dicts
    groups = {}
    for p, r in files:
        val = _param_value(p, r.get("_config", {}), args.param)
        for m in (args.methods or [k for k in r if not k.startswith("_")]):
            if m in r:
                groups.setdefault(m, {}).setdefault(val, []).append(
                    r[m]["metrics"])
    ncol = 2
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.6 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, k in zip(axes, metrics):
        for m, byval in sorted(groups.items()):
            xs = sorted(byval)
            ys = np.array([np.mean([mm[k] for mm in byval[x]]) for x in xs])
            es = np.array([np.std([mm[k] for mm in byval[x]], ddof=1)
                           if len(byval[x]) > 1 else 0.0 for x in xs])
            ax.errorbar(xs, ys, yerr=(es if es.any() else None),
                        marker="o", ms=4, capsize=3, label=NICE.get(m, m))
        ax.set_xlabel(args.param.replace("_", " "))
        ax.set_ylabel(LABELS.get(k, k))
        ax.grid(alpha=0.3)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    if len(groups) > 1:
        axes[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, args.out, f"sweep_{args.param}")


def mode_history(args):
    path, res = _load(args.files)[0]
    methods = args.methods or [m for m in res if not m.startswith("_")
                               and res[m].get("history")]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    lo, hi = np.inf, -np.inf
    for m in methods:
        h = res[m].get("history") or []
        if not h:
            continue
        init = next((e for e in h if e.get("stage") == "init"), None)
        polish = next((e for e in h if e.get("stage") == "polish"), None)
        iters = [e for e in h if isinstance(e.get("iter"), (int, float))
                 and e.get("stage") not in ("init", "polish")]
        color = ax._get_lines.get_next_color()
        if init:
            ax.axhline(init["f_val"], ls="--", lw=0.8, color=color, alpha=0.6)
            lo, hi = min(lo, init["f_val"]), max(hi, init["f_val"])
        xs = [e["iter"] for e in iters]
        ys = [e["f_val"] for e in iters]
        acc = [bool(e.get("accepted")) for e in iters]
        ax.plot(xs, ys, lw=0.8, color=color, alpha=0.5,
                label=NICE.get(m, m))
        ax.scatter([x for x, a in zip(xs, acc) if a],
                   [y for y, a in zip(ys, acc) if a],
                   marker="o", s=22, color=color, zorder=3)
        ax.scatter([x for x, a in zip(xs, acc) if not a],
                   [y for y, a in zip(ys, acc) if not a],
                   marker="o", s=22, facecolors="none",
                   edgecolors=color, zorder=3)
        good = [init["f_val"]] if init else []
        good += [y for y, a in zip(ys, acc) if a]
        if polish and xs:
            ax.scatter([max(xs) + 1], [polish["f_val"]], marker="D", s=26,
                       color=color, zorder=3)
            good.append(polish["f_val"])
        if good:
            lo, hi = min(lo, min(good)), max(hi, max(good))
    if np.isfinite(lo):
        pad = 0.25 * max(abs(hi - lo), 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"validation $F_{\mathrm{rob}}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, args.out,
          f"history_{os.path.splitext(os.path.basename(path))[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["methods", "sweep", "history"])
    ap.add_argument("files", nargs="+", help="result json files or globs")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--param", default="rho",
                    help="sweep mode: _config key (or file-name token)")
    ap.add_argument("-o", "--out", default="figures",
                    help="output directory (default: figures/)")
    args = ap.parse_args()
    {"methods": mode_methods, "sweep": mode_sweep,
     "history": mode_history}[args.mode](args)


if __name__ == "__main__":
    main()
