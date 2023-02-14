# Standard Library
import typing as t
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

# First Party Library
from p2m import get_module_set
from p2m.options import Options
from p2m.options import assert_mapping_config
from p2m.utils.bbo import ScoreSender
from p2m.utils.logger import reset_logging_config

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@hydra.main(
    version_base=None,
    config_path=f"{Path(__file__).parents[1] / 'conf'}",
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg = t.cast(DictConfig, OmegaConf.merge(OmegaConf.structured(Options), cfg))
    assert_mapping_config(cfg)
    options = t.cast(Options, cfg)

    log_root_path = Path(options.log_root_path)
    reset_logging_config(log_root_path / "run.log")

    seed_everything(options.random_seed, workers=True)

    dm, model = get_module_set(
        options.model.name,
        options=options,
    )

    logger_root_path = Path(options.log_root_path) / "lightning_logs"
    logger_root_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"{logger_root_path=}")

    pl_loggers: list[Logger] = [
        TensorBoardLogger(save_dir=logger_root_path / "tensorboard", name=options.model.name.name),
        WandbLogger(
            save_dir=logger_root_path,
            name=f"{options.model.name.name}_{options.datetime}",
            project=f"P2M-{options.model.name.name}",
        ),
    ]

    callbacks = {
        "model_checkpoint": ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            save_top_k=5,
            dirpath=logger_root_path / "model-checkpoint",
            filename="{val_loss:.8f}-{epoch}-{step}",
        ),
        "learning_rate_monitor": LearningRateMonitor(logging_interval="epoch"),
        "early_stopping": EarlyStopping(
            monitor="val_loss",
            patience=120,  # TODO: hyper parameter
        ),
    }

    trainer: pl.Trainer = pl.Trainer(
        default_root_dir=logger_root_path,
        logger=pl_loggers,
        max_epochs=options.num_epochs,
        callbacks=[c for c in callbacks.values()],
        auto_select_gpus=True,
        accelerator="gpu",
        devices=1,
        deterministic="warn",
    )

    # fit
    logger.info("Begin fit!")
    logger.info(f"checkpoint: {options.checkpoint_path}")
    trainer.fit(model=model, datamodule=dm, ckpt_path=options.checkpoint_path)

    # save score for BBO
    ScoreSender.save_score(log_root_path, score=callbacks["early_stopping"].best_score)


if __name__ == "__main__":
    main()
