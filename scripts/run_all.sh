#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_smoke.sh"
bash "$SCRIPT_DIR/run_two_stage.sh"
bash "$SCRIPT_DIR/run_one_step.sh"
bash "$SCRIPT_DIR/run_eval_validation.sh"
bash "$SCRIPT_DIR/run_eval_test.sh"
