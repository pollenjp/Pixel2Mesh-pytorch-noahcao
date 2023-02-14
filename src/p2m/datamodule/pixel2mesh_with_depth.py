# Standard Library
import json
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
from torch.utils.data import DataLoader

# First Party Library
from p2m.datasets.shapenet_with_depth import ShapeNetWithDepth
from p2m.datasets.shapenet_with_depth import get_shapenet_collate

# Local Library
from .base import DataModule

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class ShapeNetWithDepthDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        logger.info("prepare_data")

        with open(self.options.dataset.label_json_path, "rt") as fp:
            self.labels = sorted(list(json.load(fp).keys()))

        logger.info(f"labels: {[f'{i}: {label}' for i, label in enumerate(self.labels)]}")

    def prepare_data_per_node(self) -> None:
        logger.info("prepare_data_per_node")

    def setup(self, stage: str | None = None):
        if stage is not None:
            if stage not in ["fit", "test"]:
                raise ValueError(f"stage = {stage}")

    def train_dataloader(self):

        dataset = ShapeNetWithDepth(
            dataset_root_dirpath=Path(self.options.dataset.root_path).expanduser(),
            dataset_filepath_list_txt=Path(self.options.dataset.train_list_filepath).expanduser(),
            labels=self.labels,
            mesh_pos=self.options.dataset.mesh_pos,
            normalization=self.options.dataset.normalization,
            shapenet_options=self.options.dataset.shapenet,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=get_shapenet_collate(self.options.dataset.shapenet.num_points),
        )

        return dataloader

    def val_dataloader(self):
        dataset = ShapeNetWithDepth(
            dataset_root_dirpath=Path(self.options.dataset.root_path).expanduser(),
            dataset_filepath_list_txt=Path(self.options.dataset.val_list_filepath).expanduser(),
            labels=self.labels,
            mesh_pos=self.options.dataset.mesh_pos,
            normalization=self.options.dataset.normalization,
            shapenet_options=self.options.dataset.shapenet,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=get_shapenet_collate(self.options.dataset.shapenet.num_points),
        )

        return dataloader

    def test_dataloader(self):

        if self.options.dataset.test_list_filepath is None:
            raise ValueError("test_list_filepath is None")

        dataset = ShapeNetWithDepth(
            dataset_root_dirpath=Path(self.options.dataset.root_path).expanduser(),
            dataset_filepath_list_txt=Path(self.options.dataset.test_list_filepath).expanduser(),
            labels=self.labels,
            mesh_pos=self.options.dataset.mesh_pos,
            normalization=self.options.dataset.normalization,
            shapenet_options=self.options.dataset.shapenet,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=get_shapenet_collate(self.options.dataset.shapenet.num_points),
        )

        return dataloader
