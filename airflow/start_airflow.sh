#!/usr/bin/env bash
set -euo pipefail

AIRFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AIRFLOW_HOME="$AIRFLOW_DIR"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_DIR/dags"
export MLFLOW_DISABLE_AGENT_HINT=1
export PATH="$AIRFLOW_DIR/.venv/bin:$PATH"

exec airflow standalone
