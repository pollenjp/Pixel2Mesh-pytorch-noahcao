# Third Party Library
import torch


def calc_f1_score(
    dis_to_pred: torch.Tensor,
    dis_to_gt: torch.Tensor,
    pred_length: int,
    gt_length: int,
    thresh: float,
) -> torch.Tensor:
    recall = (dis_to_gt < thresh).sum() / gt_length
    precision = (dis_to_pred < thresh).sum() / pred_length
    return 2 * precision * recall / (precision + recall + 1e-8)
