"""Fetch datasets into the directories run_experiment.py expects.

Usage
-----
    python setup_data.py urbanev        # UrbanEV Shenzhen -> UrbanEV/data
    python setup_data.py paris          # Smarter Mobility -> smarter-mobility/data
    python setup_data.py all

Both fetches are shallow git clones; nothing else is required. The
Paris step clones the official challenge repository from GitLab and
copies train.csv next to the expected path. If the clone fails (e.g.
GitLab blocked on your network), download train.csv manually from
https://gitlab.com/smarter-mobility-data-challenge/tutorials and place
it at smarter-mobility/data/train.csv.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

URBANEV_REPO = "https://github.com/IntelligentSystemsLab/UrbanEV.git"
PARIS_REPO = "https://gitlab.com/smarter-mobility-data-challenge/tutorials.git"


def sh(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_urbanev():
    if os.path.isdir("UrbanEV/data"):
        print("UrbanEV/data already present, skipping")
        return
    sh(["git", "clone", "--depth", "1", URBANEV_REPO])
    assert os.path.isdir("UrbanEV/data")
    print("ok: UrbanEV/data")


def fetch_paris():
    dst = os.path.join("smarter-mobility", "data")
    target = os.path.join(dst, "train.csv")
    if os.path.exists(target):
        print(f"{target} already present, skipping")
        return
    os.makedirs(dst, exist_ok=True)
    tmp = os.path.join("smarter-mobility", "_tutorials")
    if not os.path.isdir(tmp):
        sh(["git", "clone", "--depth", "1", PARIS_REPO, tmp])
    hits = (glob.glob(os.path.join(tmp, "**", "train.csv"), recursive=True)
            or glob.glob(os.path.join(tmp, "**", "train*.csv"),
                         recursive=True))
    if not hits:
        sys.exit(f"train.csv not found inside {tmp}; place it manually at "
                 f"{target}")
    shutil.copy(hits[0], target)
    print(f"ok: {target}  (from {hits[0]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["urbanev", "paris", "all"])
    args = ap.parse_args()
    if args.what in ("urbanev", "all"):
        fetch_urbanev()
    if args.what in ("paris", "all"):
        fetch_paris()


if __name__ == "__main__":
    main()
