#!/usr/bin/env bash
# Re-extract numbers.json from the result files and render all charts (light + dark, SVG + PNG) next to this script.
# Needs matplotlib and the Inter font; DeepSeek's qbench is read from a spark-bench checkout ($SPARK_BENCH,
# default: a sibling of this repo). Optional $1 = another output directory for the rendered files.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
"$PY" make_charts.py extract
"$PY" make_charts.py render "${1:-.}"
