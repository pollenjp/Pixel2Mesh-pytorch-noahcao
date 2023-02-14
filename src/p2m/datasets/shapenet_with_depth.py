# Standard Library
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
from p2m import config
from p2m.datasets.base_dataset import BaseDataset


class ShapeNetWithDepth(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(
        self,
        dataset_filepath_list_txt: Path,
        dataset_root_dirpath: Path,
        labels: list[str],
        mesh_pos: list[float],
        normalization: bool,
        shapenet_options: t.Any,
    ):
        super().__init__()
        self.dataset_root_dirpath = dataset_root_dirpath

        self.labels_map: dict[str, int] = {k: i for i, k in enumerate(labels)}

        # Read file list
        self.relative_path_list: list[str] = []
        with open(dataset_filepath_list_txt, mode="rt") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                self.relative_path_list.append(line)
            # self.file_names = fp.read().split("\n")[:-1]

        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border

        # TODO: not measured value
        # self.depth_normalizer = Normalize(mean=[0.7], std=[1.0])

    def __getitem__(self, index: int):
        # self.dataset_root_dirpath / 04256520/1a4a8592046253ab5ff61a3a2a0e2484/rendering/00.dat
        pkl_path = self.dataset_root_dirpath / self.relative_path_list[index]
        assert pkl_path.exists(), f"{pkl_path} / {pkl_path.resolve()}"
        label = pkl_path.parents[2].name

        with open(pkl_path, "rb") as fp:
            data = pickle.load(fp, encoding="latin1")

        img_path = pkl_path.parent / f"{pkl_path.stem}.png"
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

        # TODO: need to be check
        # scaling
        # img = img / 255.0

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        depth_img_path = pkl_path.parent / f"{pkl_path.stem}_depth0001.png"

        # dim: (h, w)
        depth_img = io.imread(depth_img_path)
        # inverse gray scale
        depth_img = np.iinfo(np.uint8).max - depth_img

        depth_img = transform.resize(
            depth_img,
            (config.IMG_SIZE, config.IMG_SIZE),
        )

        depth_img = depth_img.astype(np.float32)

        # scaling
        depth_img = depth_img / 255.0

        depth_img = torch.from_numpy(depth_img).unsqueeze(0)
        # TODO: not normalized yet
        depth_img_normalized = depth_img

        return {
            "images": img_normalized,
            "images_orig": img,
            "depth_images": depth_img_normalized,
            "depth_images_orig": depth_img,
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": f"{pkl_path}",
            "length": length,
        }

    def __len__(self):
        return len(self.relative_path_list)


class P2MWithDepthDataUnit(t.TypedDict):
    images: torch.Tensor  # (3, 224, 224)
    images_orig: torch.Tensor  # (3, 224, 224), torch.uint8
    depth_images: torch.Tensor  # (1, 224, 224)
    depth_images_orig: torch.Tensor  # (1, 224, 224)
    points: npt.NDArray  # (num_points, 3)
    normals: npt.NDArray  # (num_points, 3)
    labels: torch.Tensor
    filename: str
    length: int


def get_shapenet_collate(num_points: int):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """

    def shapenet_collate(batch: list[P2MWithDepthDataUnit]):
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


class P2MWithDepthBatchData(t.TypedDict):
    images: torch.Tensor  # (batch_size, 3, 224, 224)
    images_orig: torch.Tensor  # (batch_size, 3, 224, 224)
    depth_images: torch.Tensor  # (batch_size, 1, 224, 224)
    depth_images_orig: torch.Tensor  # (batch_size, 1, 224, 224)
    points: torch.Tensor  # (batch_size, num_points, 3)
    normals: torch.Tensor  # (batch_size, num_points, 3)
    points_orig: list[torch.Tensor] | torch.Tensor
    normals_orig: list[torch.Tensor] | torch.Tensor
    labels: torch.Tensor
    filename: list[str]
    length: list[int]
