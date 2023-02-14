# Third Party Library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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

# Local Library
from .p2m_with_depth import Features2DTo3DEncoder
from .p2m_with_depth import GProjection3D
from .p2m_with_depth import MergeType
from .p2m_with_depth import P2MModelWithDepthForwardReturn


class DepthPix2VoxEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        resnet = torchvision.models.resnet50(pretrained=True)

        # torch.Size([batch_size, 512, 28, 28])
        self.resnet = torch.nn.Sequential(
            *[
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            ]
        )[:6]

        # torch.Size([batch_size, 512 * 28, 28, 28])
        self.encoder_dim: int = 28
        self.volume_dim: int = 28
        self.volume_ch_dim: int = 32
        ch_dim = self.volume_ch_dim * self.volume_dim
        assert self.volume_ch_dim * self.volume_dim**3 == ch_dim * self.encoder_dim**2
        self.encoder_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, ch_dim, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=ch_dim),
            # torch.nn.ReLU(),
        )

        # torch.Size([batch_size, 512, 28, 28, 28])

        self.channel_dim: int = 0

        # torch.Size([batch_size, ch, 56, 56, 56])
        ch_dim_pre, ch_dim_current = self.volume_ch_dim, 16
        self.decoder_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                ch_dim_pre,
                ch_dim_current,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm3d(num_features=ch_dim_current),
            torch.nn.ReLU(),
        )
        self.channel_dim += ch_dim_current

        # torch.Size([batch_size, ch, 112, 112, 112])
        ch_dim_pre, ch_dim_current = ch_dim_current, 8
        self.decoder_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                ch_dim_pre,
                ch_dim_current,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm3d(num_features=ch_dim_current),
            torch.nn.ReLU(),
        )
        self.channel_dim += ch_dim_current

        # torch.Size([batch_size, ch, 224, 224, 224])
        ch_dim_pre, ch_dim_current = ch_dim_current, 4
        self.decoder_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                ch_dim_pre,
                ch_dim_current,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm3d(num_features=ch_dim_current),
            torch.nn.ReLU(),
        )
        self.channel_dim += ch_dim_current

    def forward(self, x):
        x = self.resnet(x)
        x = self.encoder_layer1(x)

        num_batch = x.size(0)
        x = x.view(num_batch, self.volume_ch_dim, self.volume_dim, self.volume_dim, self.volume_dim)

        features = []

        x = self.decoder_layer1(x)
        features.append(x)
        x = self.decoder_layer2(x)
        features.append(x)
        x = self.decoder_layer3(x)
        features.append(x)
        assert (
            sum([feat.size(1) for feat in features]) == self.channel_dim
        ), f"{sum([feat.size(1) for feat in features])} != {self.channel_dim}"

        return features


class Depth3DEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_dim: int = 0

        # torch.Size([batch_size, 256, 56, 56])
        # torch.Size([batch_size, 512, 28, 28])
        # torch.Size([batch_size, 1024, 14, 14])
        # torch.Size([batch_size, 2048, 7, 7])
        self.resnet, _ = get_backbone(ModelBackbone.RESNET50)
        self.encoder = Features2DTo3DEncoder(in_channels=tuple(self.resnet.features_dims))

    def forward(self, x):
        x = self.resnet(x)
        x = self.encoder(x)
        return x


class ProjectedFeaturesMerger(nn.Module):
    def __init__(self, merge_type: MergeType, in_dim: int, out_dim: int):
        super().__init__()

        self.merge_type = merge_type

        match self.merge_type:
            case MergeType.CONCAT_AND_REDUCTION:
                self.layers = torch.nn.ModuleList([torch.nn.Conv2d(in_dim, out_dim, kernel_size=1)])
                raise NotImplementedError

    def forward(self, a_feature: torch.Tensor, b_feature: torch.Tensor) -> torch.Tensor:
        match self.merge_type:
            case MergeType.CONCAT_AND_REDUCTION:
                # concatenate along the channel dimension and reduce with a 1x1 conv
                # Size(batch_size, num_points, num_channels)
                x = torch.cat([a_feature.unsqueeze(2), b_feature.unsqueeze(2)], dim=2)
                return self.layers[0](x)
            case _:
                raise NotImplementedError(f"Merge type {self.merge_type} not implemented")


class P2MModelWithDepth3dCNN(nn.Module):
    def __init__(
        self,
        options: OptionsModel,
        ellipsoid: Ellipsoid,
        camera_f,
        camera_c,
        mesh_pos,
    ):
        super().__init__()

        self.options: OptionsModel = options
        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.parameter.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options.backbone)

        self.unpooling = nn.ModuleList(
            [
                GUnpooling(ellipsoid.unpool_idx[0]),
                GUnpooling(ellipsoid.unpool_idx[1]),
            ],
        )

        self.projection = GProjection(
            mesh_pos,
            camera_f,
            camera_c,
            bound=options.z_threshold,
        )

        self.depth_projection = GProjection3D()

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim, adj_mat=ellipsoid.adj_mat[2])

        match options.name:
            case ModelName.P2M_WITH_DEPTH_PIX2VOX:
                self.depth_nn_encoder = DepthPix2VoxEncoder()
                self.features_dim: int = (
                    self.coord_dim + self.nn_encoder.features_dim + self.depth_nn_encoder.channel_dim
                )
            case ModelName.P2M_WITH_DEPTH_RESNET_3D_CNN:
                self.depth_nn_encoder = Depth3DEncoder()
                self.features_dim: int = (
                    self.coord_dim + self.nn_encoder.features_dim + sum(self.depth_nn_encoder.encoder.out_channels)
                )
            case ModelName.P2M_WITH_DEPTH_3D_CNN_CONCAT:
                self.depth_nn_encoder = DepthPix2VoxEncoder()
                self.features_dim: int = self.coord_dim + self.nn_encoder.features_dim
                self.features_merger1 = ProjectedFeaturesMerger(
                    MergeType.CONCAT_AND_REDUCTION,
                    in_dim=self.coord_dim + self.nn_encoder.features_dim + self.depth_nn_encoder.channel_dim,
                    out_dim=self.nn_encoder.features_dim,
                )
                self.features_merger2 = ProjectedFeaturesMerger(
                    MergeType.CONCAT_AND_REDUCTION,
                    in_dim=self.coord_dim + self.nn_encoder.features_dim + self.depth_nn_encoder.channel_dim,
                    out_dim=self.nn_encoder.features_dim,
                )
                self.features_merger3 = ProjectedFeaturesMerger(
                    MergeType.CONCAT_AND_REDUCTION,
                    in_dim=self.coord_dim + self.nn_encoder.features_dim + self.depth_nn_encoder.channel_dim,
                    out_dim=self.nn_encoder.features_dim,
                )

            case _:
                raise NotImplementedError(f"Model {options.name} is not implemented yet.")

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

    def forward(self, batch: P2MWithDepthBatchData):
        img = batch["images"]
        batch_size = img.size(0)
        img_features = self.nn_encoder(img)
        depth_encoded_features = self.depth_nn_encoder(batch["depth_images"].repeat(1, 3, 1, 1))

        img_shape = self.projection.image_feature_shape(img)

        init_pts: torch.Tensor = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        # GCN Block 1

        # Size(batch_size, num_points, num_channels)
        match self.options.name:
            case ModelName.P2M_WITH_DEPTH_PIX2VOX | ModelName.P2M_WITH_DEPTH_RESNET_3D_CNN:
                x_img = self.projection(img_shape, img_features, init_pts)
                x_d = self.depth_projection(depth_encoded_features, init_pts)
                x = torch.cat([x_img, x_d], dim=2)
            case ModelName.P2M_WITH_DEPTH_3D_CNN_CONCAT:
                x = self.features_merger1(
                    self.projection(img_shape, img_features, init_pts),
                    self.depth_projection(depth_encoded_features, init_pts),
                )
                x = torch.concat([init_pts, x], dim=2)
            case _:
                raise NotImplementedError(f"Model {self.options.name} is not implemented yet.")
        assert x.size(2) == self.features_dim

        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2

        # Size(batch_size, num_points, num_channels)
        match self.options.name:
            case ModelName.P2M_WITH_DEPTH_PIX2VOX | ModelName.P2M_WITH_DEPTH_RESNET_3D_CNN:
                x_img = self.projection(img_shape, img_features, x1)
                x_d = self.depth_projection(depth_encoded_features, x1)
                x = torch.cat([x_img, x_d], dim=2)
            case ModelName.P2M_WITH_DEPTH_3D_CNN_CONCAT:
                x = self.features_merger1(
                    self.projection(img_shape, img_features, x1),
                    self.depth_projection(depth_encoded_features, x1),
                )
                x = torch.concat([x1, x], dim=2)
            case _:
                raise NotImplementedError(f"Model {self.options.name} is not implemented yet.")
        assert x.size(2) == self.features_dim

        x = self.unpooling[0](torch.cat([x, x_hidden], 2))

        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3

        # Size(batch_size, num_points, num_channels)
        match self.options.name:
            case ModelName.P2M_WITH_DEPTH_PIX2VOX | ModelName.P2M_WITH_DEPTH_RESNET_3D_CNN:
                x_img = self.projection(img_shape, img_features, x2)
                x_d = self.depth_projection(depth_encoded_features, x2)
                x = torch.cat([x_img, x_d], dim=2)
            case ModelName.P2M_WITH_DEPTH_3D_CNN_CONCAT:
                x = self.features_merger1(
                    self.projection(img_shape, img_features, x2),
                    self.depth_projection(depth_encoded_features, x2),
                )
                x = torch.concat([x2, x], dim=2)
            case _:
                raise NotImplementedError(f"Model {self.options.name} is not implemented yet.")
        assert x.size(2) == self.features_dim

        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(img_features)
        else:
            reconst = None

        return {"pred_coord": [x1, x2, x3], "pred_coord_before_deform": [init_pts, x1_up, x2_up], "reconst": reconst}


P2MModelWithDepth3DCNNForwardReturn = P2MModelWithDepthForwardReturn
