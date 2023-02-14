# Third Party Library
import torch
from pytorch3d.loss.chamfer import knn_points


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def main() -> None:
    """

      |
      |
    3 |          .
      |
    2 |  .   .
      |          .
    1 |  .   .
      |          .
      |
    0 -------------------------
      0  1   2   3   4

    """
    device = get_torch_device()
    p1 = torch.tensor(
        [
            [
                # [x, y, z]
                [1, 1, 1],
                [2, 1, 1],
                [2, 2, 1],
                [1, 2, 1],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    p2 = torch.tensor(
        [
            [
                # [x, y, z]
                [3.0, 0.5, 1],
                [3.0, 1.5, 1],
                [3.0, 3.0, 1],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    knn = knn_points(p1, p2, K=1)

    print(knn)

    print(knn.idx.flatten())
    print(torch.index_select(p2, dim=1, index=knn.idx.flatten()))


if __name__ == "__main__":
    main()
