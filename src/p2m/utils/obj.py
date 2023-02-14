# Standard Library
import io
from pathlib import Path

# Third Party Library
from pytorch3d.common.datatypes import Device
from pytorch3d.io.obj_io import PathManager
from pytorch3d.io.obj_io import _load_obj
from pytorch3d.io.obj_io import _open_file
from pytorch3d.renderer import TexturesAtlas
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures.meshes import join_meshes_as_batch


def load_obj(
    f: Path | io.IOBase,
    mtl_dir: Path | None = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: str | None = "repeat",
    device: Device = "cpu",
    path_manager: PathManager | None = None,
):
    data_dir = Path("./")
    if isinstance(f, (str, bytes, Path)):
        data_dir = f.parent
    if mtl_dir is not None:
        data_dir = mtl_dir
    if path_manager is None:
        path_manager = PathManager()
    with _open_file(f, path_manager, "r") as _f:
        return _load_obj(
            _f,
            data_dir=f"{data_dir}",
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
            device=device,
        )


def load_objs_as_meshes(
    files: list,
    mtl_dirs: list[Path | None] | None = None,
    device: Device | None = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: str | None = "repeat",
    path_manager: PathManager | None = None,
) -> Meshes:
    """
    <https://github.com/facebookresearch/pytorch3d/blob/dba48fb4102dae3bd9dbb7048fc6632cdfeb4274/pytorch3d/io/obj_io.py#L234>

    Returns:
        Meshes: _description_
    """
    mesh_list = []
    _mtl_dirs = mtl_dirs or [None] * len(files)
    for f_obj, mtl_dir in zip(files, _mtl_dirs):
        verts, faces, aux = load_obj(
            f_obj,
            mtl_dir=mtl_dir,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
        )
        tex = None
        if create_texture_atlas:
            # TexturesAtlas type
            tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        else:
            # TexturesUV type
            tex_maps = aux.texture_images
            if tex_maps is not None and len(tex_maps) > 0:
                verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
                faces_uvs = faces.textures_idx.to(device)  # (F, 3)
                image = list(tex_maps.values())[0].to(device)[None]
                tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)

        mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex)
        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)
