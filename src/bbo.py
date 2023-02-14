# Standard Library
import argparse
import subprocess
from dataclasses import dataclass
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import optuna

# First Party Library
from p2m.utils.bbo import NoStdoutException
from p2m.utils.bbo import ScoreSender

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class Args:
    name: str
    num_trials: int
    storage: str
    model: str


def get_args() -> Args:
    parser = argparse.ArgumentParser(description="BBO (black-box optimization)")
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--num_trials", required=True, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--storage", default="sqlite:///db.sqlite3", type=str)
    args = parser.parse_args()
    return Args(
        name=args.name,
        num_trials=args.num_trials,
        storage=args.storage,
        model=args.model,
    )


@dataclass
class Cmd:
    cmd: list[str]
    cwd: Path | None = None


def run_cmd(cmd: Cmd) -> bytes:
    # subprocess.run(cmd.cmd, stdout=cmd.stdout, stderr=cmd.stderr, cwd=cmd.cwd)
    with subprocess.Popen(cmd.cmd, stdout=subprocess.PIPE, cwd=cmd.cwd) as proc:
        if proc.stdout is None:
            raise NoStdoutException("proc.stdout is None")
        stdout, _stderr = proc.communicate()
        return stdout


def main() -> None:
    args = get_args()

    def objective(trial: optuna.trial.Trial) -> float:

        params_float_step = {
            # "loss.weights.chamfer_opposite": (0.005, 1.0),  # 0.55
            # "loss.weights.edge": (0.01, 1.0),  # 0.1
            "loss.weights.laplace": (0.1, 3.0),  # 0.5
        }
        params_float_log = {
            # "loss.weights.move": (1e-3, 0.1),  # 0.033
            # "loss.weights.normal": (1e-4, 0.1),  # 0.00016
            # "optim.lr": (1e-5, 1e-3),  # 0.0001",
        }

        suggest_params = {}

        for key, (low, high) in params_float_step.items():
            suggest_params[key] = trial.suggest_float(key, low, high)

        for key, (low, high) in params_float_log.items():
            suggest_params[key] = trial.suggest_float(key, low, high, log=True)

        cmd = Cmd(
            cmd=[
                "poetry",
                "run",
                "python",
                "src/train.py",
                f"model={args.model}",
                "dataset=resnet_with_template_airplane",
                "batch_size=32",
                "num_workers=16",
            ]
            + [f"{key}={val}" for key, val in suggest_params.items()],
            cwd=None,
        )

        score = ScoreSender.extract_score(ScoreSender.extract_path_from_stdout(run_cmd(cmd)))

        return score

    pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()  # optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name=args.name,
        sampler=sampler,
        direction=optuna.study.StudyDirection.MINIMIZE,
        storage=args.storage,
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.num_trials)

    trial = study.best_trial

    logger.info(f"{trial.value=}")
    logger.info(f"{trial.params=}")


if __name__ == "__main__":
    # Standard Library
    import logging

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
        level=logging.WARNING,
    )

    main()
