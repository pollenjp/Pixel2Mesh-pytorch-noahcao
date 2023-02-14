# Standard Library
import math
import typing as t
from dataclasses import dataclass
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import bpy
import mathutils
import numpy as np
from omegaconf import OmegaConf

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class Renderer:
    def __init__(self):

        # Set up rendering
        self.context = bpy.context
        self.scene = bpy.context.scene
        bpy_cntx_scene_render = bpy.context.scene.render

        bpy_cntx_scene_render.engine = "CYCLES"
        bpy_cntx_scene_render.image_settings.color_mode = "RGBA"
        # render.image_settings.color_depth = args.color_depth  # ('8', '16')
        bpy_cntx_scene_render.image_settings.file_format = "PNG"
        bpy_cntx_scene_render.resolution_x = 528
        bpy_cntx_scene_render.resolution_y = 528
        bpy_cntx_scene_render.resolution_percentage = 100
        bpy_cntx_scene_render.film_transparent = True

        self.scene.use_nodes = True
        self.scene.view_layers["View Layer"].use_pass_normal = True
        self.scene.view_layers["View Layer"].use_pass_diffuse_color = True
        self.scene.view_layers["View Layer"].use_pass_object_index = True

        self.scene.world.color = (1, 1, 1)

        self.nodes: bpy.types.Nodes = bpy.context.scene.node_tree.nodes

        # Clear default nodes
        for n in self.nodes:
            self.nodes.remove(n)

        # Create input render layer node
        self.render_layers = self.nodes.new("CompositorNodeRLayers")

        self.max_depth_distance = 1.2
        self.render_depth: bool = False
        if self.render_depth:
            self.setup_depth_map()

        # Delete default cube
        self.context.active_object.select_set(True)
        bpy.ops.object.delete()

        self.init_lighting()

        # set camera
        self.init_camera()

    def setup_depth_map(self) -> None:
        # depth
        # Create depth output nodes
        self.depth_file_output = self.nodes.new(type="CompositorNodeOutputFile")
        self.depth_file_output.label = "Depth Output"
        self.depth_file_output.base_path = ""
        self.depth_file_output.file_slots[0].use_node_format = True
        self.depth_file_output.format.file_format = "PNG"
        self.depth_file_output.format.color_depth = "8"  # 8 bit per channel
        self.depth_file_output.format.color_mode = "BW"

        # Remap as other types can not represent the full range of depth.
        depth_map = self.nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        depth_map.offset = [-0.2]
        depth_map.size = [1.0]
        depth_map.use_min = True
        depth_map.min = [0]
        depth_map.use_max = True
        depth_map.max = [255]

        links = bpy.context.scene.node_tree.links
        links.new(self.render_layers.outputs["Depth"], depth_map.inputs[0])
        links.new(depth_map.outputs[0], self.depth_file_output.inputs[0])

    def init_lighting(self) -> None:
        #########
        # Light #
        #########

        # Make light just directional, disable shadows.
        obj_light_name: str = "Light1"
        light1_data: bpy.types.Light = bpy.data.lights.new(obj_light_name, type="POINT")
        light1_data.type = "POINT"
        light1_data.type = "SUN"
        light1_data.use_shadow = False
        # Possibly disable specular shading:
        # light1_data.specular_factor = 1.0
        light1_data.energy = 0.7
        self.light1_object = bpy.data.objects.new(name=obj_light_name, object_data=light1_data)

        # Add another light source so stuff facing away from light is not completely dark
        obj_light2_name: str = "Light2"
        light2_data = bpy.data.lights.new(obj_light2_name, type="POINT")
        light2_data.use_shadow = False
        # light2_data.specular_factor = 1.0
        light2_data.energy = 0.7
        self.light2_object = bpy.data.objects.new(name=obj_light2_name, object_data=light2_data)

        self.light1_object.rotation_euler = (math.radians(45), 0, math.radians(90))
        self.light2_object.rotation_euler = (-math.radians(45), 0, math.radians(90))
        self.light1_object.location = (0, -2, 2)
        self.light2_object.location = (0, 2, 2)

        bpy.context.scene.collection.objects.link(self.light1_object)
        bpy.context.scene.collection.objects.link(self.light2_object)

    def init_camera(self) -> None:
        # Place camera
        self.cam = self.scene.objects["Camera"]
        # self.cam.location = (0.5, 0, 0)
        self.cam.rotation_mode = "ZXY"
        self.cam.rotation_euler = (0, math.radians(90), math.radians(90))

        # self.cam.data.sensor_width = 32

        self.cam_rotation_axis = bpy.data.objects.new("RotCenter", None)
        self.cam_rotation_axis.location = (0, 0, 0)
        self.cam_rotation_axis.rotation_euler = (0, 0, 0)
        self.scene.collection.objects.link(self.cam_rotation_axis)
        self.cam.parent = self.cam_rotation_axis

        self.context.view_layer.objects.active = self.cam_rotation_axis

    def set_viewport(
        self,
        azimuth: float,
        elevation: float,
        yaw: float,
        distance_ratio: float,
        fov: float,
    ) -> None:
        """
        <https://github.com/chrischoy/3D-R2N2/blob/13a30e257cb2158c3bf5c2370d791073517ad22e/lib/blender_renderer.py#L132-L140>
        """

        self.cam.data.lens_unit = "FOV"
        self.cam.data.lens = fov

        self.cam_rotation_axis.rotation_euler = (0, 0, 0)

        # camera and light position
        cam_location: mathutils.Vector = (
            # distance_ratio * self.max_depth_distance,
            distance_ratio,
            0,
            0,
        )
        self.cam.location = cam_location
        self.light1_object.location = mathutils.Vector((distance_ratio * (2 + self.max_depth_distance), 0, 0))

        # camera axis rotation
        self.cam_rotation_axis.rotation_euler = (
            math.radians(-yaw),
            math.radians(-elevation),
            math.radians(-azimuth),
        )

    def load_object(self, object_filepath: Path, object_name: str = "Model") -> bpy.types.Object:
        if object_filepath.suffix == ".obj":
            obj = self.load_wavefront_obj(object_filepath, obj_name=object_name)
            obj.location = (0, 0, 0)
            logger.info(f"{obj.name=}, {obj.location=}, {obj.data.name=}")
            return obj
        else:
            raise ValueError(f"{object_filepath=} not supported format!")

    def render(self, filepath: Path) -> None:
        """save to f"{filepath}.<ext>". (<ext> is the file format)
        Args:
            filepath (_PathLike): [description]
        """
        self.scene.render.filepath = str(filepath)

        if self.render_depth:
            self.depth_file_output.file_slots[0].path = f"{filepath}_depth"

        bpy.ops.render.render(write_still=True)  # render still

    @staticmethod
    def load_wavefront_obj(obj_path: Path, obj_name: t.Optional[str] = None) -> bpy.types.Object:
        bpy.ops.object.select_all(action="DESELECT")  # deselect
        bpy.ops.import_scene.obj(filepath=str(obj_path))
        obj: bpy.types.Object = bpy.context.selected_objects[0]
        obj.name = obj_name
        # context.view_layer.objects.active = obj
        return obj


name2index = {
    # "category_id/object_id/num": (1, 2),
    #
    # "02691156/d3b9114df1d8a3388e415c6cf89025f0/00": (1, 0),
    # "02691156/d3b9114df1d8a3388e415c6cf89025f0/02": (3, 1),
    # "02691156/d4dac019726e980e203936772104a82d/02": (3, 3),
    # "02691156/d54ca25127a15d2b937ae00fead8910d/00": (1, 0),
    # "02691156/d59d75f52ac9b241ae0d772a1c85134a/02": (5, 2),
    # "02691156/d63daa9d1fd2ff5d575bf8a4b14be4f4/03": (2, 1),
    # "02691156/d605a53c0917acada80799ffaf21ea7d/00": (3, 2),
    #
    "02691156/d3dcf83f03c7ad2bbc0909d98a1ff2b4/00": (3, 2),
    "02691156/d4aec2680be68c813a116bc3efac4e3b/02": (2, 2),
    "02691156/d6bf9fb6149faabe36f3a2343a8221f2/04": (2, 3),
}


def main(
    obj_filepath: Path,
    output_dirpath: Path,
    output_name: str,
    debug_mode: bool = False,
) -> None:

    renderer = Renderer()
    _ = renderer.load_object(obj_filepath, object_name="TargetModel")

    def render(
        renderer: Renderer,
        azimuth_list: t.List[float],
        elevation_list: t.List[float],
    ) -> None:
        az, el = np.meshgrid(azimuth_list, elevation_list, indexing="ij")
        for i in range(len(azimuth_list)):
            for j in range(len(elevation_list)):
                renderer.set_viewport(
                    azimuth=az[i, j],
                    elevation=el[i, j],
                    yaw=0,
                    distance_ratio=0.658845958261,
                    # distance_ratio=0.3,
                    fov=25,
                )
                output_dirpath.mkdir(parents=True, exist_ok=True)
                renderer.render(filepath=output_dirpath / f"{output_name}_i{i}-j{j}")

    azimuth_list = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    elevation_list = [0.0, 30.0, 60.0, 90.0]

    #
    # category_id: str = obj_filepath.parents[2].name
    # object_id: str = obj_filepath.parents[1].name
    # i_label: str = obj_filepath.parents[0].name
    # i, j = name2index[f"{category_id}/{object_id}/{i_label}"]

    # debug
    # i = 3
    # j = 1

    # select
    # azimuth_list = azimuth_list[i : i + 1]
    # elevation_list = elevation_list[j : j + 1]

    render(renderer=renderer, azimuth_list=azimuth_list, elevation_list=elevation_list)

    # For debugging the workflow
    if debug_mode is True:
        bpy.ops.wm.save_as_mainfile(filepath=str(output_dirpath / "debug.blend"))


@dataclass
class ConfArgs:
    obj_filepath: str
    output_dirpath: str
    output_name: str


@dataclass
class Args:
    obj_filepath: Path
    output_dirpath: Path
    output_name: str


def get_args() -> Args:
    # Standard Library
    import sys

    args: t.List[str] = sys.argv[1:]
    custom_args: t.List[str] = []
    for i, arg in enumerate(args):
        if arg == "--":
            custom_args = args[i + 1 :]
            break

    conf = t.cast(
        ConfArgs,
        OmegaConf.merge(
            OmegaConf.structured(ConfArgs),
            OmegaConf.from_dotlist(custom_args),
        ),
    )

    return Args(
        obj_filepath=Path(conf.obj_filepath),
        output_dirpath=Path(conf.output_dirpath),
        output_name=conf.output_name,
    )


if __name__ == "__main__":
    args = get_args()
    main(args.obj_filepath, args.output_dirpath, output_name=args.output_name, debug_mode=True)
