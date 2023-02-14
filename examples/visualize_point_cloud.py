# Standard Library
import typing as t
from dataclasses import dataclass
from pathlib import Path

# Third Party Library
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_pointcloud_to_axes(
    points: torch.Tensor,
    ax: Axes3D,
) -> None:
    """_summary_

    Args:
        points (torch.Tensor):
            (N, 3) tensor
        ax (Axes):
            Axes3D
    """

    if ax.name != "3d":
        raise ValueError(f"ax.name must be '3d', but {ax.name}")

    # Sample points uniformly from the surface of the mesh.
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    ax.scatter3D(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@dataclass
class Args:
    # obj_path: Path
    out_path: Path


def get_args() -> Args:
    # Standard Library
    import argparse

    parser = argparse.ArgumentParser(description="args")

    # parser.add_argument("--obj_path", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument("--out_path", required=True, type=lambda x: Path(x).expanduser().absolute())

    args = parser.parse_args()

    return Args(
        # obj_path=args.obj_path,
        out_path=args.out_path,
    )


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
    args = get_args()
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

    points_list: list[torch.Tensor] = [p1, p2]

    num_plot = len(points_list)
    ncols = 2
    nrows = (num_plot // ncols) + (0 if num_plot % ncols == 0 else 1)
    figsize_unit = 2

    fig = plt.figure(
        facecolor="white",
        figsize=(figsize_unit * ncols, figsize_unit * nrows),
        tight_layout=True,
    )

    for i, p in enumerate(points_list):
        ax = t.cast(Axes3D, fig.add_subplot(nrows, ncols, i + 1, projection="3d"))
        plot_pointcloud_to_axes(p, ax)
        # 右手系にする
        ax.invert_xaxis()
        ax.invert_yaxis()
        view_elev, view_azim = 20, 135
        ax.view_init(elev=view_elev, azim=view_azim)

    # fig.subplots_adjust(wspace=0.5, hspace=0.3)
    space = 5.0
    fig.subplots_adjust(wspace=space, hspace=space)
    fig.savefig(f"{args.out_path}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
