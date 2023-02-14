# Standard Library
import pickle
import typing as t
from dataclasses import dataclass
from pathlib import Path

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D

AxisPoints = torch.Tensor | npt.NDArray[np.float32 | np.float64]


@dataclass
class Args:
    dat_path: Path
    out_path: Path


def get_args() -> Args:
    # Standard Library
    import argparse

    parser = argparse.ArgumentParser(description="args")

    parser.add_argument("--dat_path", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument("--out_path", required=True, type=lambda x: Path(x).expanduser().absolute())

    args = parser.parse_args()

    return Args(
        dat_path=args.dat_path,
        out_path=args.out_path,
    )


def main() -> None:
    args = get_args()

    pkl_path = args.dat_path
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp, encoding="latin1")

    pts: npt.NDArray[np.float32]
    pts, _ = data[:, :3], data[:, 3:]
    points_list: list[tuple[AxisPoints, AxisPoints, AxisPoints]] = [
        (pts[:, 0], pts[:, 1], pts[:, 2]),
    ]

    num_plot = len(points_list)
    ncols = 2
    nrows = (num_plot // ncols) + (0 if num_plot % ncols == 0 else 1)
    figsize_unit = 2

    fig = plt.figure(
        facecolor="white",
        figsize=(figsize_unit * ncols, figsize_unit * nrows),
        tight_layout=True,
    )

    for i, (x, y, z) in enumerate(points_list):
        ax = t.cast(Axes3D, fig.add_subplot(nrows, ncols, i + 1, projection="3d"))
        ax.scatter3D(x, y, z, s=0.05)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

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
