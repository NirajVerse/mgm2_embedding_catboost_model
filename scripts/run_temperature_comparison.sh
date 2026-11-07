#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_temperature_comparison.sh RUN_ID [training options]" >&2
  echo "Example: bash scripts/run_temperature_comparison.sh temperature_cpu_01" >&2
  exit 2
fi

RUN_ID="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m mgm2_embedding_model train \
  --config configs/modeling.json \
  --run-id "$RUN_ID" \
  "$@"
