# Standard Library
import typing as t
from enum import Enum
from enum import unique

# Third Party Library
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party Library
from p2m.datasets.shapenet_with_depth import P2MWithDepthBatchData
from p2m.models.backbones import get_backbone
from p2m.models.layers.gbottleneck import GBottleneck
from p2m.models.layers.gconv import GConv
from p2m.models.layers.gpooling import GUnpooling
from p2m.models.layers.gprojection import GProjection
from p2m.options import ModelBackbone
from p2m.options import ModelName
from p2m.options import OptionsModel
from p2m.utils.mesh import Ellipsoid


@unique
class MergeType(Enum):
    CONCAT_AND_REDUCTION = "concat_and_reduction"
    ADD = "add"


class Features2DMerger(nn.Module):
    def __init__(self, merge_type: MergeType, options: t.Any | None = None):
        super().__init__()

        self.merge_type = merge_type

        match self.merge_type:
            case MergeType.CONCAT_AND_REDUCTION:
                if options is None:
                    raise ValueError("options must be provided for concat_and_reduction")

                in_dims: list[int] = options["in_dims"]
                out_dims: list[int] = options["out_dims"]
                if len(in_dims) != len(out_dims):
                    raise ValueError(
                        f"in_dims and out_dims must have the same length: {len(in_dims)} != {len(out_dims)}"
                    )
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Conv2d(in_dims[i], out_dims[i], kernel_size=1) for i in range(len(in_dims))]
                )

    def forward(self, a_feature: t.Sequence[torch.Tensor], b_feature: t.Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """_summary_

        Args:
            a_feature (t.Sequence[torch.Tensor]):
                tensor size: (batch_size, c, h, w)
            b_feature (t.Sequence[torch.Tensor]):
                tensor size: (batch_size, c, h, w)

        Raises:
            NotImplementedError: _description_

        Returns:
            list[torch.Tensor]: _description_
        """
        vals = []
        match self.merge_type:
            case MergeType.CONCAT_AND_REDUCTION:
                # concatenate along the channel dimension and reduce with a 1x1 conv
                for i, (a, b) in enumerate(zip(a_feature, b_feature)):
                    x = torch.cat([a, b], dim=1)
                    vals.append(self.layers[i](x))

            case MergeType.ADD:
                for a, b in zip(a_feature, b_feature):
                    vals.append(a + b)
            case _:
                raise NotImplementedError(f"Merge type {self.merge_type} not implemented")
        return vals


class Features2DTo3DEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
        mid_channels: tuple[int, int, int, int] = (56, 18, 64, 256),
        out_channels: tuple[int, int, int, int] = (8, 16, 32, 64),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        if len(self.in_channels) != 4:
            raise ValueError(f"channels must have length 4: {len(self.in_channels)}")

        # scale1

        i = 0
        self.scale1_ch_dim_mid: int = mid_channels[i]
        self.scale1_ch_dim_last: int = out_channels[i]
        self.scale1_size: int = 56
        self.scale1_1x1conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels[i],
                self.scale1_size * self.scale1_ch_dim_mid,
                kernel_size=1,
            ),
            torch.nn.BatchNorm2d(num_features=self.scale1_size * self.scale1_ch_dim_mid),
            torch.nn.ReLU(),
        )
        self.scale1_conv3d = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                self.scale1_ch_dim_mid,
                self.scale1_ch_dim_last,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm3d(num_features=self.scale1_ch_dim_last),
            torch.nn.ReLU(),
        )

        # scale2

        i = 1
        self.scale2_ch_dim_mid: int = mid_channels[i]
        self.scale2_ch_dim_last: int = out_channels[i]
        self.scale2_size: int = 28
        self.scale2_1x1conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels[i],
                self.scale2_size * self.scale2_ch_dim_mid,
                kernel_size=1,
            ),
            torch.nn.BatchNorm2d(num_features=self.scale2_size * self.scale2_ch_dim_mid),
            torch.nn.ReLU(),
        )
        self.scale2_conv3d = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                self.scale2_ch_dim_mid,
                self.scale2_ch_dim_last,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm3d(num_features=self.scale2_ch_dim_last),
            torch.nn.ReLU(),
        )

        # scale3

        i = 2
        self.scale3_ch_dim_mid: int = mid_channels[i]
        self.scale3_ch_dim_last: int = out_channels[i]
        self.scale3_size: int = 14
        self.scale3_1x1conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels[i],
                self.scale3_size * self.scale3_ch_dim_mid,
                kernel_size=1,
            ),
            torch.nn.BatchNorm2d(num_features=self.scale3_size * self.scale3_ch_dim_mid),
            torch.nn.ReLU(),
        )
        self.scale3_conv3d = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                self.scale3_ch_dim_mid,
                self.scale3_ch_dim_last,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm3d(num_features=self.scale3_ch_dim_last),
            torch.nn.ReLU(),
        )

        # scale4

        i = 3
        self.scale4_ch_dim_mid: int = mid_channels[i]
        self.scale4_ch_dim_last: int = out_channels[i]
        self.scale4_size: int = 7
        self.scale4_1x1conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.in_channels[i],
                self.scale4_size * self.scale4_ch_dim_mid,
                kernel_size=1,
            ),
            torch.nn.BatchNorm2d(num_features=self.scale4_size * self.scale4_ch_dim_mid),
            torch.nn.ReLU(),
        )
        self.scale4_conv3d = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                self.scale4_ch_dim_mid,
                self.scale4_ch_dim_last,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm3d(num_features=self.scale4_ch_dim_last),
            torch.nn.ReLU(),
        )

    def forward(self, xs: list[torch.Tensor]):
        x1, x2, x3, x4 = xs
        batch_size: int = x1.size(0)

        features = []

        x = self.scale1_1x1conv(x1)
        x = x.view(batch_size, self.scale1_ch_dim_mid, self.scale1_size, self.scale1_size, self.scale1_size)
        x = self.scale1_conv3d(x)
        features.append(x)

        x = self.scale2_1x1conv(x2)
        x = x.view(batch_size, self.scale2_ch_dim_mid, self.scale2_size, self.scale2_size, self.scale2_size)
        x = self.scale2_conv3d(x)
        features.append(x)

        x = self.scale3_1x1conv(x3)
        x = x.view(batch_size, self.scale3_ch_dim_mid, self.scale3_size, self.scale3_size, self.scale3_size)
        x = self.scale3_conv3d(x)
        features.append(x)

        x = self.scale4_1x1conv(x4)
        x = x.view(batch_size, self.scale4_ch_dim_mid, self.scale4_size, self.scale4_size, self.scale4_size)
        x = self.scale4_conv3d(x)
        features.append(x)

        return features


class GProjection3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def calc_sample_points(pts: torch.Tensor, multi: float = 4.0) -> torch.Tensor:
        """

        Args:
            pts (torch.Tensor):
                Size(batch_size, num_points, 3)
                x*multi, y*multi, z*multi がそれぞれ (-1.0, 1.0) 範囲の座標であることを想定
                範囲外はclamp

        Returns: torch.Tensor
            Size(batch_size, num_points, 3)
        """
        # x = pts[:, :, 0] * 2
        # y = pts[:, :, 1] * 2
        # z = pts[:, :, 1] * 2
        # return torch.stack(
        #     [x, y, z],
        #     dim=-1,
        # )
        return torch.clamp(pts * multi, -1.0, 1.0)

    @staticmethod
    def project(feature: torch.Tensor, sample_points: torch.Tensor):
        """_summary_

        Args:
            feature (torch.Tensor): _description_
            sample_points (torch.Tensor):
                Size(batch_size, num_points, 3)

        Returns:
            _type_: _description_
        """

        # Size()
        output = F.grid_sample(
            feature,  # (batch_size, num_channels, depth, height, width)
            sample_points.unsqueeze(1).unsqueeze(1),  # (batch_size, 1, 1, num_points, 2) (N, D_out, H_out, W_out, 3)
            align_corners=True,
        )
        # (batch_size, num_points, num_channels)
        return torch.transpose(output.squeeze(2).squeeze(2), 1, 2)

    def forward(
        self,
        features: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:

        sample_points = self.calc_sample_points(points)

        feats = []
        for feat in features:  # each data in batch
            feats.append(
                self.project(
                    feature=feat,
                    sample_points=sample_points,
                ),
            )

        # Size(batch_size, num_points, num_channels)
        output = torch.cat(feats, 2)

        return output


class P2mWithDepth3dCNNEncoder(torch.nn.Module):
    def __init__(self, backbone: ModelBackbone, coord_dim: int) -> None:
        super().__init__()
        self.nn_encoder, _ = get_backbone(backbone)
        self.depth_nn_encoder, _ = get_backbone(backbone)
        if self.nn_encoder.features_dims != self.depth_nn_encoder.features_dims:
            raise ValueError("Features dim of rgb and depth encoder must be equal")
        block_dims = self.nn_encoder.features_dims
        self.features_merger = Features2DMerger(
            MergeType.CONCAT_AND_REDUCTION,
            options={
                "in_dims": [d * 2 for d in block_dims],
                "out_dims": block_dims,
            },
        )
        self.features_2d_to_3d_encoder = Features2DTo3DEncoder(in_channels=tuple(block_dims))

        self.features_dim: int = sum(self.features_2d_to_3d_encoder.out_channels)

    def forward(self, a_features: list[torch.Tensor], b_features: list[torch.Tensor]) -> torch.Tensor:
        a = self.nn_encoder(a_features)
        b = self.depth_nn_encoder(b_features)
        encoded_features = self.features_merger(a, b)
        return self.features_2d_to_3d_encoder(encoded_features)


class DepthOnly3dCNNEncoder(torch.nn.Module):
    def __init__(self, backbone: ModelBackbone, coord_dim: int) -> None:
        super().__init__()
        self.depth_nn_encoder, _ = get_backbone(backbone)
        block_dims = self.depth_nn_encoder.features_dims
        self.features_2d_to_3d_encoder = Features2DTo3DEncoder(in_channels=tuple(block_dims))

        self.features_dim: int = sum(self.features_2d_to_3d_encoder.out_channels)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x = self.depth_nn_encoder(x)
        return self.features_2d_to_3d_encoder(x)


class P2MModelWithDepth(nn.Module):
    def __init__(
        self,
        options: OptionsModel,
        ellipsoid: Ellipsoid,
        camera_f,
        camera_c,
        mesh_pos,
    ):
        super().__init__()

        self.options = options
        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.parameter.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.features_dim: int
        match self.options.name:
            case ModelName.P2M_WITH_DEPTH:
                self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold)
                self.nn_encoder, _ = get_backbone(options.backbone)
                self.depth_nn_encoder, _ = get_backbone(options.backbone)

                # self.merge_features = MergeFeatures(MergeType.ADD)
                block_dims = self.nn_encoder.features_dims

                # merge rgb-encoder and depth-encoder features
                self.features_merger = Features2DMerger(
                    MergeType.CONCAT_AND_REDUCTION,
                    options={
                        "in_dims": [d * 2 for d in block_dims],
                        "out_dims": block_dims,
                    },
                )

                self.features_dim = self.coord_dim + self.nn_encoder.features_dim
            case ModelName.P2M_WITH_DEPTH_RESNET:
                self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold)
                self.nn_encoder, _ = get_backbone(options.backbone)
                self.depth_nn_encoder, _ = get_backbone(ModelBackbone.RESNET50)
                self.features_dim = (
                    self.coord_dim + self.nn_encoder.features_dim + self.coord_dim + self.depth_nn_encoder.features_dim
                )

            case ModelName.P2M_WITH_DEPTH_ONLY:
                self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold)
                self.depth_nn_encoder, _ = get_backbone(options.backbone)
                self.features_dim = self.coord_dim + self.depth_nn_encoder.features_dim

            case ModelName.P2M_WITH_DEPTH_ONLY_3D_CNN:
                self.depth_nn_encoder = DepthOnly3dCNNEncoder(backbone=options.backbone, coord_dim=self.coord_dim)
                self.features_dim = self.coord_dim + self.depth_nn_encoder.features_dim
                self.projection = GProjection3D()

            case ModelName.P2M_WITH_DEPTH_3D_CNN:
                self.encoder = P2mWithDepth3dCNNEncoder(backbone=options.backbone, coord_dim=self.coord_dim)
                self.features_dim = self.coord_dim + self.encoder.features_dim
                self.projection = GProjection3D()

            case _:
                raise ValueError(f"Model name {self.options.name} not supported")

        self.gcns = nn.ModuleList(
            [
                GBottleneck(
                    6,
                    self.features_dim,
                    self.hidden_dim,
                    self.coord_dim,
                    ellipsoid.adj_mat[0],
                    activation=self.gconv_activation,
                ),
                GBottleneck(
                    6,
                    self.features_dim + self.hidden_dim,
                    self.hidden_dim,
                    self.coord_dim,
                    ellipsoid.adj_mat[1],
                    activation=self.gconv_activation,
                ),
                GBottleneck(
                    6,
                    self.features_dim + self.hidden_dim,
                    self.hidden_dim,
                    self.last_hidden_dim,
                    ellipsoid.adj_mat[2],
                    activation=self.gconv_activation,
                ),
            ]
        )

        self.unpooling = nn.ModuleList(
            [
                GUnpooling(ellipsoid.unpool_idx[0]),
                GUnpooling(ellipsoid.unpool_idx[1]),
            ],
        )

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim, adj_mat=ellipsoid.adj_mat[2])

    def forward(self, batch: P2MWithDepthBatchData):
        img = batch["images"]
        batch_size = img.size(0)
        init_pts: torch.Tensor = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        match self.options.name:
            case ModelName.P2M_WITH_DEPTH:
                img_shape = self.projection.image_feature_shape(img)
                img_features = self.nn_encoder(img)
                depth_img_feats = self.depth_nn_encoder(batch["depth_images"].repeat(1, 3, 1, 1))
                encoded_features = self.features_merger(img_features, depth_img_feats)
            case ModelName.P2M_WITH_DEPTH_RESNET:
                img_shape = self.projection.image_feature_shape(img)
                img_features = self.nn_encoder(img)
                depth_img_feats = self.depth_nn_encoder(batch["depth_images"].repeat(1, 3, 1, 1))
            case ModelName.P2M_WITH_DEPTH_ONLY:
                img_shape = self.projection.image_feature_shape(img)
                depth_img_feats = self.depth_nn_encoder(batch["depth_images"].repeat(1, 3, 1, 1))
                encoded_features = depth_img_feats
            case ModelName.P2M_WITH_DEPTH_ONLY_3D_CNN:
                depth_img_feats = self.depth_nn_encoder(batch["depth_images"].repeat(1, 3, 1, 1))
                encoded_features = depth_img_feats
            case ModelName.P2M_WITH_DEPTH_3D_CNN:
                encoded_features = self.encoder(img, batch["depth_images"].repeat(1, 3, 1, 1))
            case _:
                raise ValueError(f"Model name {self.options.name} not supported")
        # GCN Block 1
        match self.options.name:
            case ModelName.P2M_WITH_DEPTH | ModelName.P2M_WITH_DEPTH_ONLY:
                x = self.projection(img_shape, encoded_features, init_pts)
            case ModelName.P2M_WITH_DEPTH_ONLY_3D_CNN:
                x = self.projection(encoded_features, init_pts)
                x = torch.cat([init_pts, x], dim=2)
            case ModelName.P2M_WITH_DEPTH_RESNET:
                x_img = self.projection(img_shape, img_features, init_pts)
                x_depth = self.projection(img_shape, depth_img_feats, init_pts)
                x = torch.cat([x_img, x_depth], dim=2)
            case ModelName.P2M_WITH_DEPTH_3D_CNN:
                x = self.projection(encoded_features, init_pts)
                x = torch.cat([init_pts, x], dim=2)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        match self.options.name:
            case ModelName.P2M_WITH_DEPTH | ModelName.P2M_WITH_DEPTH_ONLY:
                x = self.projection(img_shape, encoded_features, x1)
            case ModelName.P2M_WITH_DEPTH_ONLY_3D_CNN:
                x = self.projection(encoded_features, x1)
                x = torch.cat([x1, x], dim=2)
            case ModelName.P2M_WITH_DEPTH_RESNET:
                x_img = self.projection(img_shape, img_features, x1)
                x_depth = self.projection(img_shape, depth_img_feats, x1)
                x = torch.cat([x_img, x_depth], dim=2)
            case ModelName.P2M_WITH_DEPTH_3D_CNN:
                x = self.projection(encoded_features, x1)
                x = torch.cat([x1, x], dim=2)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        match self.options.name:
            case ModelName.P2M_WITH_DEPTH | ModelName.P2M_WITH_DEPTH_ONLY:
                x = self.projection(img_shape, encoded_features, x2)
            case ModelName.P2M_WITH_DEPTH_ONLY_3D_CNN:
                x = self.projection(encoded_features, x2)
                x = torch.cat([x2, x], dim=2)
            case ModelName.P2M_WITH_DEPTH_RESNET:
                x_img = self.projection(img_shape, img_features, x2)
                x_depth = self.projection(img_shape, depth_img_feats, x2)
                x = torch.cat([x_img, x_depth], dim=2)
            case ModelName.P2M_WITH_DEPTH_3D_CNN:
                x = self.projection(encoded_features, x2)
                x = torch.cat([x2, x], dim=2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        return {"pred_coord": [x1, x2, x3], "pred_coord_before_deform": [init_pts, x1_up, x2_up], "reconst": None}


class P2MModelWithDepthForwardReturn(t.TypedDict):
    pred_coord: list[torch.Tensor]  # (3,) arary. Each element is (batch_size, num_points, 3)
    pred_coord_before_deform: list[torch.Tensor]  # [init_pts, x1_up, x2_up]
    reconst: torch.Tensor  # TODO: ???
