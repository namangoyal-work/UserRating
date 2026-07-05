#!/usr/bin/env bash
# usage: scripts/test.sh <model> <input_csv> <output_txt>
set -euo pipefail
python3 -m userating test "$1" "$2" "$3"
