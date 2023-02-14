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
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers import TensorBoardLogger

# First Party Library
from p2m import get_module_set
from p2m.options import Options
from p2m.options import assert_mapping_config
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
    logger.info(f"{logger_root_path=}")

    pl_loggers: list[Logger] = [
        TensorBoardLogger(save_dir=logger_root_path / "tensorboard", name=options.model.name.name)
    ]

    trainer: pl.Trainer = pl.Trainer(
        default_root_dir=logger_root_path,
        logger=pl_loggers,
        max_epochs=options.num_epochs,
        callbacks=None,
        auto_select_gpus=True,
        accelerator="gpu",
        devices=1,
        deterministic="warn",
    )

    # fit
    logger.info("Begin test!")
    logger.info(f"checkpoint: {options.checkpoint_path}")
    trainer.test(model=model, datamodule=dm, ckpt_path=options.checkpoint_path)


if __name__ == "__main__":
    main()
