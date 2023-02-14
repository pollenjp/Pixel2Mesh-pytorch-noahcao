# Standard Library
import typing as t
from dataclasses import dataclass
from pathlib import Path

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer import PointLights
from pytorch3d.renderer import RasterizationSettings
from pytorch3d.renderer import SoftPhongShader
from pytorch3d.renderer import look_at_view_transform


@dataclass
class Args:
    obj_path: Path
    out_path: Path


def get_args() -> Args:
    # Standard Library
    import argparse

    parser = argparse.ArgumentParser(description="args")

    parser.add_argument("--obj_path", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument("--out_path", required=True, type=lambda x: Path(x).expanduser().absolute())

    args = parser.parse_args()

    return Args(
        obj_path=args.obj_path,
        out_path=args.out_path,
    )


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_mesh_renderer(device: torch.device = get_torch_device()) -> MeshRenderer:
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    R, T = look_at_view_transform(2.7, 0, 180)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    return renderer


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


def main() -> None:
    args = get_args()

    device = get_torch_device()

    # Load obj file
    batch_size = 20
    meshes = load_objs_as_meshes([args.obj_path], device=device, create_texture_atlas=True).extend(batch_size)

    # Get a batch of viewing angles.
    elev = torch.linspace(0, 180, batch_size)
    azim = torch.linspace(-180, 180, batch_size)

    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = get_mesh_renderer(device=device)

    images = renderer(meshes, cameras=cameras, lights=lights)

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
