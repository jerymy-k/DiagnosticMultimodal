import argparse
import importlib
import inspect
import logging
import subprocess
import sys

from src.utils.config import PROJECT_ROOT

STEPS = {
    "check_and_prepare_datasets": ["src.data.prepare:check_and_prepare_datasets"],
    "validate_data": ["src.data.validate:validate_data"],
    "train_classifier": ["src.classification.train:train_classifier"],
    "train_detector": ["src.detection.train:train_detector"],
    "evaluate_and_log_mlflow": [
        "src.classification.evaluate:evaluate_classifier",
        "src.detection.evaluate:evaluate_detector",
    ],
    "save_models_and_artifacts": [
        "src.classification.export:save_classifier",
        "src.detection.export:save_detector",
    ],
}

logger = logging.getLogger("pipeline")


def run_target(target: str, options: dict):
    module_name, function_name = target.split(":")
    function = getattr(importlib.import_module(module_name), function_name)
    accepted = inspect.signature(function).parameters
    result = function(**{k: v for k, v in options.items() if k in accepted})
    logger.info("%s terminé : %s", target, result)


def run_step(step: str, flags: list[str]):
    for target in STEPS[step]:
        logger.info("Lancement de %s", target)
        completed = subprocess.run([sys.executable, "-m", "src.pipeline", "--target", target, *flags], cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            raise SystemExit(f"{target} a échoué (code {completed.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Étapes du pipeline brain_tumor_pipeline")
    parser.add_argument("step", nargs="?", choices=list(STEPS))
    parser.add_argument("--target")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    options = {"quick": args.quick, "force": args.force}
    if args.target:
        run_target(args.target, options)
    elif args.step:
        run_step(args.step, [f"--{name}" for name, enabled in options.items() if enabled])
    else:
        parser.error("indiquer une étape ou --target")


if __name__ == "__main__":
    main()
