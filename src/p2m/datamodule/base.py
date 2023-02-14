# Standard Library
import random
from logging import NullHandler
from logging import getLogger

# Third Party Library
import numpy as np
import pytorch_lightning as pl
import torch

# First Party Library
from p2m.options import Options

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        options: Options,
        batch_size: int,
        num_workers: int = 4,
    ):
        super().__init__()

        self.options = options
        self.batch_size = batch_size
        self.num_workers = num_workers
