# Standard Library
import typing as t
from dataclasses import dataclass

# Third Party Library
import pytorch_lightning as pl

# First Party Library
from p2m.datamodule.base import DataModule
from p2m.datamodule.pixel2mesh import ShapeNetDataModule
from p2m.datamodule.pixel2mesh_with_depth import ShapeNetWithDepthDataModule
from p2m.datamodule.pixel2mesh_with_template import ShapeNetWithTemplateDataModule
from p2m.lightningmodule.pixel2mesh import P2MModelModule
from p2m.lightningmodule.pixel2mesh_with_depth import P2MModelWithDepthModule
from p2m.lightningmodule.pixel2mesh_with_template import P2MModelWithTemplateModule
from p2m.options import ModelName
from p2m.options import Options


@dataclass
class ModuleSet:
    module: t.Type[pl.LightningModule]
    data_module: t.Type[DataModule]


name2module: dict[ModelName, ModuleSet] = {
    ModelName.P2M: ModuleSet(
        module=P2MModelModule,
        data_module=ShapeNetDataModule,
    ),
    ModelName.P2M_WITH_TEMPLATE: ModuleSet(
        module=P2MModelWithTemplateModule,
        data_module=ShapeNetWithTemplateDataModule,
    ),
    ModelName.P2M_WITH_DEPTH: ModuleSet(
        module=P2MModelWithDepthModule,
        data_module=ShapeNetWithDepthDataModule,
    ),
    ModelName.P2M_WITH_DEPTH_ONLY: ModuleSet(
        module=P2MModelWithDepthModule,
        data_module=ShapeNetWithDepthDataModule,
    ),
    ModelName.P2M_WITH_DEPTH_3D_CNN: ModuleSet(
        module=P2MModelWithDepthModule,
        data_module=ShapeNetWithDepthDataModule,
    ),
    ModelName.P2M_WITH_DEPTH_3D_CNN_CONCAT: ModuleSet(
        module=P2MModelWithDepthModule,
        data_module=ShapeNetWithDepthDataModule,
    ),
    ModelName.P2M_WITH_DEPTH_RESNET_3D_CNN: ModuleSet(
        module=P2MModelWithDepthModule,
        data_module=ShapeNetWithDepthDataModule,
    ),
}


def get_module_set(
    model_name: ModelName,
    options: Options,
) -> tuple[DataModule, pl.LightningModule]:
    match model_name:
        case ModelName.P2M | ModelName.P2M_WITH_TEMPLATE | ModelName.P2M_WITH_DEPTH_ONLY | ModelName.P2M_WITH_DEPTH | ModelName.P2M_WITH_DEPTH_3D_CNN | ModelName.P2M_WITH_DEPTH_RESNET_3D_CNN:
            module_set = name2module[model_name]
            return (
                module_set.data_module(
                    options=options,
                    batch_size=options.batch_size,
                    num_workers=options.num_workers,
                ),
                module_set.module(
                    options=options,
                ),
            )
        case ModelName.P2M_WITH_DEPTH_3D_CNN_CONCAT:
            raise NotImplementedError("Not implemented yet")
        case _:
            raise ValueError(f"Unknown module name: {model_name}")
