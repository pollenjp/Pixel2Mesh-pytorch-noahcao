# Standard Library
import contextlib
import io
import typing as t
from pathlib import Path

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer import PointLights
from pytorch3d.renderer import RasterizationSettings
from pytorch3d.renderer import SoftPhongShader
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures.meshes import Meshes

# First Party Library
from p2m.utils.obj import load_objs_as_meshes


def plot_point_cloud(vertices: t.Sequence[torch.Tensor] | torch.Tensor, num_cols: int = 2) -> npt.NDArray[np.uint8]:
    """_summary_

    Args:
        vertices (torch.Tensor):
            Size (num_batch_size, num_vertices, 3)

    Returns:
        npt.NDArray[np.uint8]: _description_
    """

    num_plot = len(vertices)
    num_rows = (num_plot // num_cols) + (0 if num_plot % num_cols == 0 else 1)
    figure_size_unit = 2

    fig = plt.figure(
        facecolor="white",
        figsize=(figure_size_unit * num_cols, figure_size_unit * num_rows),
        tight_layout=True,
    )

    for i, vs in enumerate(vertices):
        # vs: (num_vertices, 3)
        x, y, z = vs.detach().cpu().squeeze().unbind(1)
        ax = t.cast(Axes3D, fig.add_subplot(num_rows, num_cols, i + 1, projection="3d"))
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
    fig.canvas.draw()
    fig.canvas = t.cast(
        FigureCanvasAgg,
        fig.canvas,
    )

    img: npt.NDArray = np.frombuffer(
        buffer=t.cast(bytes, fig.canvas.tostring_rgb()),
        dtype=np.uint8,
    )
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return img


def write_obj_info(
    f: io.TextIOBase,
    coords: torch.Tensor | list[list[float]],
    faces: torch.Tensor,
    mtl_filename: str,
    usemtl_name: str,
) -> None:
    f.write(f"mtllib {mtl_filename}\n")

    for coord in coords:
        f.write(f"v {coord[0]} {coord[1]} {coord[2]}\n")

    f.write(f"usemtl {usemtl_name}")

    for face in faces:
        f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def write_obj_info_context(
    coords: torch.Tensor | list[list[float]],
    faces: torch.Tensor,
    mtl_filename: str,
    usemtl_name: str,
) -> t.ContextManager[io.StringIO]:
    f = io.StringIO("")
    write_obj_info(f=f, coords=coords, faces=faces, mtl_filename=mtl_filename, usemtl_name=usemtl_name)
    f.seek(0)

    return contextlib.closing(f)


def get_mesh_renderer(device: Device) -> MeshRenderer:
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


def plot_pred_meshes(
    coords: torch.Tensor,
    faces: torch.Tensor,
    mtl_filepath: Path,
    usemtl_name: str,
    device: Device = "cpu",
) -> torch.Tensor:
    """_summary_

    Args:
        coords (torch.Tensor): _description_
        faces (torch.Tensor): _description_
        mtl_filepath (Path): _description_
        usemtl_name (str): _description_
        device (Device): _description_

    Returns:
        torch.Tensor: _description_
    """

    with write_obj_info_context(
        coords=coords,
        faces=faces,
        mtl_filename=mtl_filepath.name,
        usemtl_name=usemtl_name,
    ) as f:
        meshes: Meshes = load_objs_as_meshes(
            files=[f],
            mtl_dirs=[mtl_filepath.parent],
            device=device,
            load_textures=False,
            create_texture_atlas=True,
        )

    elev = 45.0
    azim = 45.0

    R, T = look_at_view_transform(dist=1.0, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = get_mesh_renderer(device=device)

    images: torch.Tensor = renderer(meshes, cameras=cameras, lights=lights)

    return images
