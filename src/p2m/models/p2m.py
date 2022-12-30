# Standard Library
import typing as t

# Third Party Library
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party Library
from p2m.datasets.shapenet import P2MBatchData
from p2m.models.backbones import get_backbone
from p2m.models.layers.gbottleneck import GBottleneck
from p2m.models.layers.gconv import GConv
from p2m.models.layers.gpooling import GUnpooling
from p2m.models.layers.gprojection import GProjection
from p2m.options import OptionsModel
from p2m.utils.mesh import Ellipsoid


class P2MModel(nn.Module):
    def __init__(
        self,
        options: OptionsModel,
        ellipsoid: Ellipsoid,
        camera_f,
        camera_c,
        mesh_pos,
    ):
        super().__init__()

        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.parameter.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

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

        self.projection = GProjection(
            mesh_pos, camera_f, camera_c, bound=options.z_threshold, tensorflow_compatible=options.align_with_tensorflow
        )

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim, adj_mat=ellipsoid.adj_mat[2])

    def forward(self, batch: P2MBatchData):
        img = batch["images"]
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        x = self.projection(img_shape, img_feats, init_pts)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, img_feats, x1)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, img_feats, x2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(img_feats)
        else:
            reconst = None

        return {"pred_coord": [x1, x2, x3], "pred_coord_before_deform": [init_pts, x1_up, x2_up], "reconst": reconst}


class P2MModelForwardReturn(t.TypedDict):
    pred_coord: list[torch.Tensor]  # (3,) arary. Each element is (batch_size, num_points, 3)
    pred_coord_before_deform: list[torch.Tensor]  # [init_pts, x1_up, x2_up]
    reconst: torch.Tensor  # TODO: ???
