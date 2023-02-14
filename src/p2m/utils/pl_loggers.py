# Standard Library
from logging import getLogger

# Third Party Library
import numpy as np
import numpy.typing as npt
import torch
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers import TensorBoardLogger

logger = getLogger(__name__)


def pl_log_scalar(pl_logger: Logger | list[Logger], tag: str, scalar: float | int, global_step: int) -> None:
    pl_loggers: list[Logger]
    if isinstance(pl_logger, Logger):
        pl_loggers = [pl_logger]
    else:
        pl_loggers = pl_logger

    for pl_l in pl_loggers:
        if isinstance(pl_l, TensorBoardLogger):
            # > add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
            # > <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_images>
            pl_l.experiment.add_scalar(tag=tag, scalar_value=scalar, global_step=global_step)
        else:
            error_msg: str = f"{pl_l} doesn't support scalar logging"
            logger.error(error_msg)


def pl_log_images(
    pl_logger: Logger | list[Logger],
    tag: str,
    imgs_arr: torch.Tensor | npt.NDArray[np.uint8],
    global_step: int,
    data_formats: str = "NCHW",
) -> None:
    pl_loggers: list[Logger]
    if isinstance(pl_logger, Logger):
        pl_loggers = [pl_logger]
    else:
        pl_loggers = pl_logger

    for pl_l in pl_loggers:
        if isinstance(pl_l, TensorBoardLogger):
            logger.debug(f"pl_logger = {pl_l}")
            logger.debug(f"imgs_arr.shape = {imgs_arr.shape}")
            # > add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
            # > <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_images>
            pl_l.experiment.add_images(
                tag=tag,
                img_tensor=imgs_arr,
                global_step=global_step,
                walltime=None,
                dataformats=data_formats,
            )
        else:
            error_msg: str = f"{pl_l} dones't support image logging"
            logger.warning(error_msg)
