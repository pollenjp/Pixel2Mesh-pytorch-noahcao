# Standard Library
import typing as t

# Third Party Library
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.loss.chamfer import knn_points

# First Party Library
from p2m.models.p2m_with_template import P2MModelWithTemplateForwardReturn
from p2m.options import OptionsLoss
from p2m.utils.mesh import Ellipsoid


class P2MLoss(nn.Module):
    def __init__(self, options: OptionsLoss, ellipsoid: Ellipsoid):
        super().__init__()
        self.options = options
        self.l2_loss = nn.MSELoss(reduction="mean")
        # self.chamfer_dist = ChamferDist()
        self.laplace_idx = nn.ParameterList(
            [nn.parameter.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx]
        )

        # [
        #   [scale1_edges],
        #   [scale2_edges],
        #   [scale3_edges],
        # ]
        self.edges = nn.ParameterList([nn.parameter.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges])

    def edge_regularization(self, pred, edges) -> torch.Tensor:
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        return self.l2_loss(pred[:, edges[:, 0]], pred[:, edges[:, 1]]) * pred.size(-1)

    @staticmethod
    def edge_variance_regularization(pred, edges) -> torch.Tensor:
        num_edges = edges.size(0)

        # TODO: edge variance selection

        # Size(batch_size, num_edges, 3,)
        # '3' is (x,y,z) coordinates
        p1 = pred[:, edges[:, 0]]
        p2 = pred[:, edges[:, 1]]

        # Size(batch_size, num_edges,)
        edge_length_square = (p1 - p2).pow(2).sum(dim=2)

        # escape 0 division
        # TODO: 1e-8 or 1e-12?
        edge_length = (edge_length_square + 1e-12).sqrt()

        # Size(batch_size, 1,)
        edge_mean = edge_length.mean(dim=1, keepdim=True)

        # Size(batch_size, num_edges,)
        edge_variance = torch.nn.functional.mse_loss(
            edge_length,
            edge_mean.expand(-1, num_edges),
            # reduce=False,
            reduction="mean",
        )
        return edge_variance

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """

        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices

        vertices = inputs[:, all_valid_indices]
        vertices[:, invalid_mask] = 0
        neighbor_sum = torch.sum(vertices, 2)
        neighbor_count = lap_idx[:, -1].float()
        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]

        return laplace

    def laplace_regularization(
        self, input1: torch.Tensor, input2: torch.Tensor, block_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """

        lap1 = self.laplace_coord(input1, self.laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, self.laplace_idx[block_idx])
        laplace_loss: torch.Tensor = self.l2_loss(lap1, lap2) * lap1.size(-1)
        move_loss: torch.Tensor
        if block_idx > 0:
            move_loss = self.l2_loss(input1, input2) * input1.size(-1)
        else:
            move_loss = torch.tensor(0.0, device=input1.device)
        return laplace_loss, move_loss

    def normal_loss(self, gt_normal, indices, pred_points, adj_list):
        """
        最も近い点同士の法線ベクトルの差を計算する

        推論した頂点 p に一番近い教師データの頂点 q を見つける.
        教師の vertex normal を n_q とする.
        p の任意の隣接頂点を k とする.
        (p - k) ベクトルと n_q ベクトルとの内積を計算する.

        Args:
            gt_normal (_type_):
                Size (num_batch, num_gt_normals, 3).
            indices (_type_): _description_
            pred_points (_type_):
                Size (num_batch, num_points, 3).
            adj_list (_type_): edge list.
                Size (num_edges, 2).

        Returns:
            _type_: _description_
        """

        # size: (num_edges, 3,)
        # 辺上の単位ベクトル
        edges = F.normalize(pred_points[:, adj_list[:, 0]] - pred_points[:, adj_list[:, 1]], dim=2)
        nearest_normals = torch.index_select(gt_normal, dim=1, index=indices)
        normals = F.normalize(nearest_normals[:, adj_list[:, 0]], dim=2)
        cosine = torch.abs(torch.sum(edges * normals, 2))
        return torch.mean(cosine)

    def image_loss(self, gt_img, pred_img):
        rect_loss = F.binary_cross_entropy(pred_img, gt_img)
        return rect_loss

    def forward(
        self, outputs: P2MModelWithTemplateForwardReturn, targets
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """
        gt_coord, gt_normal, gt_images = targets["points"], targets["normals"], targets["images"]
        pred_coord, pred_coord_before_deform = outputs["pred_coord"], outputs["pred_coord_before_deform"]

        device = pred_coord[0].device

        chamfer_loss = torch.tensor(0.0, device=device)
        edge_loss = torch.tensor(0.0, device=device)
        normal_loss = torch.tensor(0.0, device=device)
        lap_loss = torch.tensor(0.0, device=device)
        move_loss = torch.tensor(0.0, device=device)
        lap_const = [0.2, 1.0, 1.0]

        image_loss = 0.0
        if outputs["reconst"] is not None and self.options.weights.reconst != 0:
            image_loss = self.image_loss(gt_images, outputs["reconst"])

        for i in range(3):
            knn = knn_points(pred_coord[i], gt_coord, K=1)

            chamfer_loss += self.options.weights.chamfer[i] * t.cast(
                torch.Tensor,
                chamfer_distance(gt_coord, pred_coord[i])[0],
            )
            normal_loss += self.normal_loss(gt_normal, knn.idx.flatten(), pred_coord[i], self.edges[i])
            edge_loss += self.edge_regularization(pred_coord[i], self.edges[i])
            lap, move = self.laplace_regularization(pred_coord_before_deform[i], pred_coord[i], i)
            lap_loss += lap_const[i] * lap
            move_loss += lap_const[i] * move

        loss = (
            chamfer_loss
            + self.options.weights.reconst * image_loss
            + self.options.weights.laplace * lap_loss
            + self.options.weights.move * move_loss
            + self.options.weights.edge * edge_loss
            + self.options.weights.normal * normal_loss
        )

        loss *= self.options.weights.constant

        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            "loss_move": move_loss,
            "loss_normal": normal_loss,
        }


class P2MLossForwardReturnSecondDict(t.TypedDict):
    loss: torch.Tensor
    loss_chamfer: torch.Tensor
    loss_edge: torch.Tensor
    loss_laplace: torch.Tensor
    loss_move: torch.Tensor
    loss_normal: torch.Tensor
