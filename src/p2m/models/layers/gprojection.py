# Third Party Library
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold


class GProjection(nn.Module):
    """
    Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bi-linear interpolation to get the corresponding feature.
    """

    def __init__(self, mesh_pos, camera_f, camera_c, bound=0):
        super(GProjection, self).__init__()
        self.mesh_pos, self.camera_f, self.camera_c = mesh_pos, camera_f, camera_c
        self.threshold = None
        self.bound = 0
        if self.bound != 0:
            self.threshold = Threshold(bound, bound)

    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.threshold is None:
            return x

        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x

    @staticmethod
    def image_feature_shape(img: torch.Tensor) -> npt.NDArray[np.uint8]:
        w = img.size(-1)
        h = img.size(-2)
        return np.array([w, h])

    def calc_sample_points(self, resolution: npt.NDArray[np.uint8], inputs: torch.Tensor):
        """_summary_

        Args:
            resolution (npt.NDArray[np.uint8]): _description_
            inputs (torch.Tensor):
                (num_batch, num_vertices, 3)

        Returns:
            _type_: _description_
        """
        half_resolution = (resolution - 1) / 2
        camera_c_offset = np.array(self.camera_c) - half_resolution

        # map to [-1, 1]
        # 投射した際の画像内の座標を計算
        # mesh_pos で z軸方向 (奥行き) に対して, 移動させているがこれがカメラのz位置では？
        # 今回のデータセットの inputs が大体 +-0.5 に収まるくらいだからってことかな
        positions = inputs + torch.tensor(self.mesh_pos, device=inputs.device, dtype=torch.float)
        w = -self.camera_f[0] * (positions[:, :, 0] / self.bound_val(positions[:, :, 2])) + camera_c_offset[0]
        h = self.camera_f[1] * (positions[:, :, 1] / self.bound_val(positions[:, :, 2])) + camera_c_offset[1]

        # directly do clamping
        w /= half_resolution[0]
        h /= half_resolution[1]

        # clamp to [-1, 1]
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        return (w, h)

    def forward(
        self,
        resolution: npt.NDArray[np.uint8],
        img_features: torch.Tensor,
        inputs: torch.Tensor,
        insert_inputs: bool = True,
    ) -> torch.Tensor:

        w, h = self.calc_sample_points(resolution, inputs)

        feats = []
        if insert_inputs:
            feats.append(inputs)
        for img_feature in img_features:  # each scale features
            feats.append(
                self.project(
                    img_shape=resolution,
                    img_feat=img_feature,
                    sample_points=torch.stack([w, h], dim=-1),  # (batch_size, num_points, 2)
                ),
            )

        output = torch.cat(feats, 2)

        return output

    def project(self, img_shape, img_feat, sample_points: torch.Tensor):
        """
        :param img_shape: raw image shape
        :param img_feat: [batch_size x channel x h x w]
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x num_points x feat_dim]
        """
        # (batch_size, num_channels, 1, num_points)
        # num_channels is img_feat.size(1)
        output = F.grid_sample(
            img_feat,
            sample_points.unsqueeze(1),  # (batch_size, 1, num_points, 2)
            align_corners=True,
        )
        # torch/nn/functional.py:4215:
        # UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0.
        # Please specify align_corners=True if the old behavior is desired.
        # See the documentation of grid_sample for details.

        # (batch_size, num_points, num_channels)
        output = torch.transpose(output.squeeze(2), 1, 2)

        return output
