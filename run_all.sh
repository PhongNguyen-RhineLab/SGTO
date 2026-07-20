#!/usr/bin/env bash
#
# Full experiment suite for the risk-aware EV charging paper.
#
# Usage
# -----
#   bash run_all.sh                 # run everything (several hours)
#   bash run_all.sh main            # only the main dataset x grid tables
#   bash run_all.sh main ablation   # a subset of stages, in this order
#   QUICK=1 bash run_all.sh         # smoke test: shrink every run (~min)
#   bash run_all.sh --list          # list stage names and exit
#
# Stages: setup main rho_seeds weights paris_calib rho_sweep
# Env vars:
#   QUICK=1        pass --quick to every run (fast pipeline check)
#   PY=python3     interpreter to use (default: python3)
#   SEEDS="1 2 3"  algorithm seeds for the rho_seeds stage
#   GROWTHS="1 2 3 5"   demand-growth factors for the paris_calib stage
#   RHOS="0.0 0.25 0.5 1.0 2.0"   rho values for the rho_sweep stage
#
# Results land in results/<dataset>/results_<tag>.json.

set -euo pipefail

PY="${PY:-python3}"
QUICK_FLAG=""
[ "${QUICK:-0}" = "1" ] && QUICK_FLAG="--quick"
SEEDS="${SEEDS:-1 2 3}"
GROWTHS="${GROWTHS:-1 2 3 5}"
RHOS="${RHOS:-0.0 0.25 0.5 1.0 2.0}"

ALL_STAGES="setup main rho_seeds weights paris_calib rho_sweep"

if [ "${1:-}" = "--list" ]; then
    echo "stages: $ALL_STAGES"
    exit 0
fi

# stages to run = args, or all of them
STAGES="${*:-$ALL_STAGES}"

# run <label> <cmd...>: echo the command, time it, keep going on failure
run() {
    local label="$1"; shift
    echo ""
    echo ">>> [$label] $*"
    local t0=$SECONDS
    if "$@"; then
        echo "<<< [$label] done in $((SECONDS - t0))s"
    else
        echo "!!! [$label] FAILED (exit $?), continuing" >&2
    fi
}

has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# ------------------------------------------------------------------
if has setup; then
    echo "=== stage: setup (fetch datasets) ==="
    run setup-urbanev "$PY" setup_data.py urbanev
    run setup-paris   "$PY" setup_data.py paris
    echo "note: --grid ieee33 needs pandapower, --roads osmnx needs osmnx"
    echo "      pip install pandapower osmnx"
fi

# ------------------------------------------------------------------
# Main tables: 4 dataset x grid combinations, all 7 methods each.
if has main; then
    echo ""
    echo "=== stage: main (dataset x grid, all methods) ==="
    run main-urbanev-synth "$PY" run_experiment.py \
        --dataset urbanev --grid synthetic --tag main $QUICK_FLAG
    run main-urbanev-ieee33 "$PY" run_experiment.py \
        --dataset urbanev --grid ieee33 --tag main $QUICK_FLAG
    run main-paris-synth "$PY" run_experiment.py \
        --dataset paris --grid synthetic --tag main $QUICK_FLAG
    run main-paris-osmnx "$PY" run_experiment.py \
        --dataset paris --grid synthetic --roads osmnx --tag osmnx $QUICK_FLAG
fi

# ------------------------------------------------------------------
# rho ablation across algorithm seeds (fixed test set): sgto pair only.
if has rho_seeds; then
    echo ""
    echo "=== stage: rho_seeds (multi-seed sgto vs sgto_risk_neutral) ==="
    for s in $SEEDS; do
        run "rho_seed-$s" "$PY" run_experiment.py --dataset urbanev \
            --methods sgto sgto_risk_neutral \
            --algo-seed "$s" --tag "rho_seed$s" $QUICK_FLAG
    done
fi

# ------------------------------------------------------------------
# Weight-alignment ablation: risk-aware weights vs mean-only weights.
if has weights; then
    echo ""
    echo "=== stage: weights (aligned vs mean-only semi-gradient) ==="
    run weights-aligned "$PY" run_experiment.py --dataset urbanev \
        --methods sgto_no_exchange sgto --tag aligned $QUICK_FLAG
    run weights-meanonly "$PY" run_experiment.py --dataset urbanev \
        --methods sgto_no_exchange sgto \
        --no-risk-in-weights --tag meanonly $QUICK_FLAG
fi

# ------------------------------------------------------------------
# Paris calibration: sweep demand-growth to find a binding budget.
if has paris_calib; then
    echo ""
    echo "=== stage: paris_calib (demand-growth sweep) ==="
    for g in $GROWTHS; do
        run "paris-growth-$g" "$PY" run_experiment.py --dataset paris \
            --demand-growth "$g" \
            --methods cost_aware_greedy sgto sgto_risk_neutral \
            --tag "growth$g" $QUICK_FLAG
    done
fi

# ------------------------------------------------------------------
# rho sweep for the risk-return curve (sgto only).
if has rho_sweep; then
    echo ""
    echo "=== stage: rho_sweep (risk-return curve) ==="
    for r in $RHOS; do
        run "rho-$r" "$PY" run_experiment.py --dataset urbanev \
            --methods sgto --rho "$r" --tag "rho$r" $QUICK_FLAG
    done
fi

echo ""
echo "All requested stages finished. Results under results/<dataset>/."
