#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="figures"

mkdir -p "$OUT_DIR"

echo "Generating main plots..."
python make_plots.py methods \
    results/urbanev/results_main.json \
    -o "$OUT_DIR"

echo "Generating rho sweep plots..."
python make_plots.py sweep \
    "results/urbanev/results_rho[0-9]*.json" \
    --param rho \
    --methods sgto \
    -o "$OUT_DIR"

echo "Generating demand growth sweep plots..."
python make_plots.py sweep \
    "results/paris/results_growth*.json" \
    --param demand_growth \
    --methods cost_aware_greedy sgto sgto_risk_neutral \
    -o "$OUT_DIR"

echo "Generating history plots..."
python make_plots.py history \
    results/urbanev/results_main.json \
    --methods sgto \
    -o "$OUT_DIR"

echo "All figures have been generated in: $OUT_DIR"