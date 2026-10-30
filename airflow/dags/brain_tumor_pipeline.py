import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PYTHON = os.getenv("PIPELINE_PYTHON", str(PROJECT_ROOT / ".venv" / "bin" / "python"))

QUICK = "{{ '--quick' if params.quick else '' }}"
FORCE_PREPARE = "{{ '--force' if params.force_prepare else '' }}"
FORCE_SAVE = "{{ '--force' if params.force_save else '' }}"

STEPS = [
    ("check_and_prepare_datasets", timedelta(minutes=30), FORCE_PREPARE),
    ("validate_data", timedelta(minutes=15), ""),
    ("train_classifier", timedelta(hours=3), QUICK),
    ("train_detector", timedelta(hours=3), QUICK),
    ("evaluate_and_log_mlflow", timedelta(minutes=30), ""),
    ("save_models_and_artifacts", timedelta(minutes=15), FORCE_SAVE),
]

logger = logging.getLogger(__name__)


def log_failure(context):
    task_instance = context["task_instance"]
    logger.error(
        "Échec de %s (essai %s) dans le run %s : %s",
        task_instance.task_id, task_instance.try_number, context["run_id"], context.get("exception"),
    )


with DAG(
    dag_id="brain_tumor_pipeline",
    description="Préparation des données, entraînement, évaluation et sauvegarde des modèles IRM",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["diagnostic-multimodal", "mlflow"],
    params={
        "quick": Param(False, type="boolean", description="Entraînement court (1 epoch) pour tester le pipeline"),
        "force_prepare": Param(False, type="boolean", description="Régénérer processed/ même s'il existe déjà"),
        "force_save": Param(False, type="boolean", description="Promouvoir les modèles même s'ils ne battent pas le champion"),
    },
    default_args={
        "owner": "diagnostic-multimodal",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
        "on_failure_callback": log_failure,
    },
) as dag:
    tasks = [
        BashOperator(
            task_id=step,
            bash_command=f"cd {PROJECT_ROOT} && {PYTHON} -m src.pipeline {step} {flags}",
            execution_timeout=timeout,
        )
        for step, timeout, flags in STEPS
    ]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
