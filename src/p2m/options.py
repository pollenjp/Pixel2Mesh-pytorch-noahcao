# Standard Library
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from dataclasses import dataclass
from enum import Enum
from enum import unique
from logging import NullHandler
from logging import getLogger

# Third Party Library
import torch

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class OptionsDatasetShapenet:
    num_points: int
    resize_with_constant_border: bool


@dataclass
class OptionsDataset:
    name: str
    num_classes: int
    label_json_path: str
    root_path: str
    train_list_filepath: str
    val_list_filepath: str
    test_list_filepath: str | None
    camera_f: list[float]
    camera_c: list[float]
    mesh_pos: list[float]
    shapenet: OptionsDatasetShapenet

    normalization: bool = True


@unique
class ModelName(Enum):
    P2M = "p2m"
    P2M_WITH_TEMPLATE = "p2m_with_template"
    P2M_WITH_DEPTH = "p2m_with_depth"
    P2M_WITH_DEPTH_RESNET = "p2m_with_depth_resnet"
    P2M_WITH_DEPTH_ONLY = "p2m_with_depth_only"
    P2M_WITH_DEPTH_ONLY_3D_CNN = "p2m_with_depth_only_3d_cnn"
    P2M_WITH_DEPTH_3D_CNN = "p2m_with_depth_3d_cnn"
    P2M_WITH_DEPTH_PIX2VOX = "p2m_with_depth_pix2vox"
    P2M_WITH_DEPTH_3D_CNN_CONCAT = "p2m_with_depth_3d_cnn_concat"
    P2M_WITH_DEPTH_RESNET_3D_CNN = "p2m_with_depth_resnet_3d_cnn"


@unique
class ModelBackbone(Enum):
    RESNET50 = 0
    VGG16P2M = 1


@dataclass
class OptionsModel:
    name: ModelName
    hidden_dim: int
    last_hidden_dim: int
    coord_dim: int
    backbone: ModelBackbone
    gconv_activation: bool

    # provide a boundary for z, so that z will never be equal to 0, on denominator
    # if z is greater than 0, it will never be less than z;
    # if z is less than 0, it will never be greater than z.
    z_threshold: int

    # align with original tensorflow model
    # please follow experiments/tensorflow.yml
    align_with_tensorflow: bool


@dataclass
class OptionsLossWeights:
    normal: float
    edge: float
    laplace: float
    move: float
    constant: float
    chamfer: list[float]
    chamfer_opposite: float
    reconst: float


@dataclass
class OptionsLoss:
    weights: OptionsLossWeights


@unique
class OptimName(Enum):
    ADAM = 0
    SGD = 1


@dataclass
class OptionsOptim:
    name: OptimName
    lr: float
    # adam:
    #   - Adam beta1
    #   - Adam beta2
    #   - weight_decay
    # sgd:
    #   - SGD momentum
    params: list[float]


def generate_optimizer(
    model: torch.nn.Module,
    optim_data: OptionsOptim,
) -> torch.optim.Optimizer:
    match optim_data.name:
        case OptimName.ADAM:
            p1, p2, p3 = optim_data.params
            return torch.optim.Adam(params=model.parameters(), lr=optim_data.lr, betas=(p1, p2), weight_decay=p3)
        case OptimName.SGD:
            p1, p2 = optim_data.params
            return torch.optim.SGD(params=model.parameters(), lr=optim_data.lr, momentum=p1, weight_decay=p2)
        case _:
            raise ValueError(f"Unknown OptimName: {optim_data.name}")


@unique
class LearningRateSchedulerName(Enum):
    MULTI_STEP = 0
    COSINE_ANNEALING_WARM_RESTARTS = 1


@dataclass
class OptionsLRScheduler:
    name: LearningRateSchedulerName

    # MultiStep:
    params: list[float] | None
    list_params: list[list[float]] | None


def generate_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_data: OptionsLRScheduler,
):
    match scheduler_data.name:
        case LearningRateSchedulerName.MULTI_STEP:

            def parse_params_for_multi_step(
                params: list[float] | None = None,
                list_params: list[list[float]] | None = None,
            ) -> tuple[list[int], float]:
                milestones: list[int]
                gamma: float

                if params is None:
                    raise TypeError(f"Invalid params for {scheduler_data.name}: {params}")
                (gamma,) = params

                if list_params is None:
                    raise TypeError(f"Invalid list_params for {scheduler_data.name}: {list_params}")
                (pl1,) = list_params
                milestones = [int(p) for p in pl1]

                return (
                    milestones,
                    gamma,
                )

            milestones, gamma = parse_params_for_multi_step(scheduler_data.params, scheduler_data.list_params)
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma,
            )
        case LearningRateSchedulerName.COSINE_ANNEALING_WARM_RESTARTS:

            def parse_params_for_cosine_annealing_warm_restarts(
                params: list[int | float] | None = None,
                list_params: list[list[int | float]] | None = None,
            ) -> tuple[int, int, float]:
                if list_params is not None:
                    raise TypeError()

                if params is None:
                    raise TypeError(f"Invalid params for {scheduler_data.name}: {params}")

                p1, p2, p3 = params
                return (int(p1), int(p2), p3)

            T_0, T_mult, eta_min = parse_params_for_cosine_annealing_warm_restarts(
                scheduler_data.params, scheduler_data.list_params
            )
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
            )

        case _:
            raise ValueError(f"Unknown LearningRateSchedulerName: {scheduler_data.name}")


@dataclass
class Options:
    num_workers: int

    datetime: str  # %Y-%m-%dT%H%M%S
    log_root_path: str
    pretrained_weight_path: str | None
    checkpoint_path: str | None  # Checkpointへのpath
    dataset: OptionsDataset
    model: OptionsModel
    loss: OptionsLoss
    batch_size: int
    optim: OptionsOptim
    lr_scheduler: OptionsLRScheduler
    num_epochs: int
    random_seed: int

    batch_size_for_plot: int
    mtl_filepath: str
    usemtl_name: str


def assert_sequence_config(cfg: MutableSequence) -> None:
    for val in cfg:
        if isinstance(val, MutableMapping):
            assert_mapping_config(val)
        elif isinstance(val, MutableSequence):
            assert_sequence_config(val)


def assert_mapping_config(cfg: MutableMapping) -> None:
    for _, val in cfg.items():
        if isinstance(val, MutableMapping):
            assert_mapping_config(val)
        elif isinstance(val, MutableSequence):
            assert_sequence_config(val)
