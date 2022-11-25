# Standard Library
import json
import pickle
import typing as t
from pathlib import Path

# Third Party Library
import numpy as np
import numpy.typing as npt
import torch
from skimage import io
from skimage import transform
from torch.utils.data.dataloader import default_collate

# First Party Library
import config
from p2m.datasets.base_dataset import BaseDataset


def extract_coords_from_obj_file(obj_filepath: Path) -> t.List[t.List[float]]:
    coords: t.List[t.List[float]] = []
    with open(obj_filepath, "rt") as f:
        # for line in f:
        for i, line in enumerate(f):
            line_elem = line.strip().split(" ")
            # v, fn, s (?), f
            if len(line_elem) != 4 or line_elem[0] != "v":
                continue

            xyz = line_elem[1:]
            coords.append(list(map(float, xyz)))

    return coords


class ShapeNetWithTemplate(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(
        self,
        file_root: Path,
        file_list_name: str,
        mesh_pos,
        normalization: bool,
        shapenet_options: t.Any,
    ):
        super().__init__()
        self.file_root: Path = file_root
        with open(self.file_root / "meta" / "shapenet.json", "r") as fp:
            labels_map = sorted(list(json.load(fp).keys()))

        self.labels_map: t.Dict[str, int] = {k: i for i, k in enumerate(labels_map)}

        # Read file list
        with open(self.file_root / "meta" / f"{file_list_name}.txt", mode="rt") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.tensorflow = "_tf" in file_list_name  # tensorflow version of data
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border

    def __getitem__(self, index: int) -> dict[str, t.Any]:
        filename = self.file_names[index][17:]
        label = filename.split("/", maxsplit=1)[0]
        pkl_path = self.file_root / "data_tf" / filename
        img_path = pkl_path.parent / f"{pkl_path.stem}.png"
        template_obj_path = pkl_path.parent / f"{pkl_path.stem}_depth0001.obj"

        with open(pkl_path, "rb") as fp:
            data = pickle.load(fp, encoding="latin1")

        pts, normals = data[:, :3], data[:, 3:]
        img = io.imread(img_path)
        img[np.where(img[:, :, 3] == 0)] = 255
        if self.resize_with_constant_border:
            img = transform.resize(
                img,
                (config.IMG_SIZE, config.IMG_SIZE),
                mode="constant",
                anti_aliasing=False,
            )  # to match behavior of old versions
        else:
            img = transform.resize(
                img,
                (config.IMG_SIZE, config.IMG_SIZE),
            )
        img = img[:, :, :3].astype(np.float32)

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,  # torch.uint8
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": filename,
            "length": length,
            # template mesh's coordinates
            "init_pts": torch.tensor(extract_coords_from_obj_file(template_obj_path)),
        }

    def __len__(self):
        return len(self.file_names)


class P2MWithTemplateDataUnit(t.TypedDict):
    images: torch.Tensor  # (3, 224, 224)
    images_orig: torch.Tensor  # (3, 224, 224), torch.uint8
    points: npt.NDArray  # (num_points, 3)
    normals: npt.NDArray  # (num_points, 3)
    labels: torch.Tensor
    filename: str
    length: int

    # template mesh's coordinates
    # (num_points, 3)
    init_pts: torch.Tensor  # (num_points, 3)


def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """

    def shapenet_collate(batch: list[P2MWithTemplateDataUnit]):
        if len(batch) > 1:
            all_equal = True
            for b in batch:
                if b["length"] != batch[0]["length"]:
                    all_equal = False
                    break
            points_orig, normals_orig = [], []
            if not all_equal:
                for b in batch:
                    pts, normal = b["points"], b["normals"]
                    length = pts.shape[0]
                    choices = np.resize(np.random.permutation(length), num_points)
                    b["points"], b["normals"] = pts[choices], normal[choices]
                    points_orig.append(torch.from_numpy(pts))
                    normals_orig.append(torch.from_numpy(normal))
                ret = default_collate(batch)
                ret["points_orig"] = points_orig
                ret["normals_orig"] = normals_orig
                return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret

    return shapenet_collate


class P2MWithTemplateBatchData(t.TypedDict):
    images: torch.Tensor  # (batch_size, 3, 224, 224)
    images_orig: torch.Tensor  # (batch_size, 3, 224, 224)
    points: torch.Tensor  # (batch_size, num_points, 3)
    normals: torch.Tensor  # (batch_size, num_points, 3)
    points_orig: list[torch.Tensor] | torch.Tensor
    normals_orig: list[torch.Tensor] | torch.Tensor
    labels: torch.Tensor
    filename: list[str]
    length: list[int]
    init_pts: torch.Tensor  # (batch_size, num_points, 3)
