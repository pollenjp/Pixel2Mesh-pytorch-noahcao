# Standard Library
import contextlib
import io
import typing as t
from dataclasses import dataclass
from pathlib import Path

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.renderer import PointLights
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures.meshes import Meshes

# First Party Library
from p2m.utils.mesh import Ellipsoid
from p2m.utils.obj import load_objs_as_meshes
from p2m.utils.render import get_mesh_renderer


@dataclass
class Args:
    out_path: Path


def get_args() -> Args:
    # Standard Library
    import argparse

    parser = argparse.ArgumentParser(description="args")

    parser.add_argument("--out_path", required=True, type=lambda x: Path(x).expanduser().absolute())

    args = parser.parse_args()

    return Args(
        out_path=args.out_path,
    )


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@dataclass
class PlotData:
    array: np.ndarray
    title: t.Optional[str] = None
    axis: t.Optional[str] = None
    xlabel: t.Optional[str] = None
    ylabel: t.Optional[str] = None


def add_subplot_to_figure(
    plot_data_list: t.List[PlotData],
    fig: Figure,
    nrows: int,
    ncols: int,
    *,
    start_idx: int = 0,
) -> Figure:
    """template plot function

    Args:
        plot_list :
        nrows (int) : the number of rows
        ncols (int) : the number of columns
        idx (int) :
        fig (matpltlib.pyplot.figure) :

    Doc Style : google, http://www.sphinx-doc.org/ja/stable/ext/example_google.html#example-google
    """

    d: PlotData
    for d in plot_data_list:
        start_idx += 1
        ax = fig.add_subplot(nrows, ncols, start_idx)
        ax.imshow(d.array)
        if d.title is not None:
            ax.set_title(label=d.title)
        if d.xlabel is not None:
            ax.set_xlabel(xlabel=d.xlabel)
        if d.ylabel is not None:
            ax.set_ylabel(ylabel=d.ylabel)

    return fig


def write_obj_info(coords: torch.Tensor | list[list[float]], faces: torch.Tensor) -> t.ContextManager[io.StringIO]:
    f = io.StringIO("")
    mtl_filename = "rendering.mtl"
    usemtl_name = "Default_OBJ.004"

    f.write(f"mtllib {mtl_filename}\n")

    for coord in coords:
        f.write(f"v {coord[0]} {coord[1]} {coord[2]}\n")

    f.write(f"usemtl {usemtl_name}")

    for face in faces:
        f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    f.seek(0)

    return contextlib.closing(f)


def load_mesh(device: torch.device | str | None = None) -> Meshes:
    data_path = Path("datasets/data/ellipsoid/info_ellipsoid.dat")
    ellipsoid = Ellipsoid([0.0, 0.0, -0.8], data_path)

    print(ellipsoid.faces)
    print(ellipsoid.adj_mat)

    for i, (faces, adj_mat) in enumerate(zip(ellipsoid.faces, ellipsoid.adj_mat)):
        print(f"============{i}============")
        print(f"{faces.size()=}")
        print(f"{faces=}")
        print(f"{adj_mat.size()=}")
        print(f"{adj_mat=}")

    idx = 0
    with write_obj_info(
        ellipsoid.coord,
        ellipsoid.faces[idx] + 1,  # convert 0 origin to 1 origin (obj file format)
    ) as f:
        meshes = load_objs_as_meshes(
            files=[f],
            mtl_dirs=[Path("..")],
            device=device,
            load_textures=False,
            create_texture_atlas=True,
        )
        return meshes


def main() -> None:
    args = get_args()
    device = get_torch_device()

    batch_size: int = 20
    meshes = load_mesh(device=device)
    meshes = meshes.extend(batch_size)

    # Get a batch of viewing angles.
    elev = torch.linspace(0, 180, batch_size)
    azim = torch.linspace(-180, 180, batch_size)

    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = get_mesh_renderer(device=device)

    images = renderer(meshes, cameras=cameras, lights=lights)

    print(f"{images.size()=}")

    plot_data_list = [
        PlotData(
            array=img[..., :3],
            title=f"mesh: {i}",
        )
        for i, img in enumerate(images.detach().cpu().numpy())
    ]
    num_plot = len(plot_data_list)
    ncols = 5
    nrows = (num_plot // ncols) + (0 if num_plot % ncols == 0 else 1)
    figsize_unit = 2
    fig = plt.figure(
        facecolor="white",
        figsize=(figsize_unit * ncols, figsize_unit * nrows),
        tight_layout=True,
    )
    fig = add_subplot_to_figure(plot_data_list=plot_data_list, fig=fig, nrows=nrows, ncols=ncols)
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    fig.savefig(f"{args.out_path}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
