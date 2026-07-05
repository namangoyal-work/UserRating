#!/usr/bin/env bash
# usage: scripts/train.sh <train_csv> <model_out>
set -euo pipefail
python3 -m userating train "$1" "$2"
