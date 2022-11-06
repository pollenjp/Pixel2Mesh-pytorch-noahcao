# Standard Library
import json
import os
import pickle
import typing as t
from pathlib import Path

# Third Party Library
import numpy as np
import torch
from PIL import Image
from skimage import io
from skimage import transform
from torch.utils.data.dataloader import default_collate

# First Party Library
import config
from p2m.datasets.base_dataset import BaseDataset


class ShapeNet(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(
        self,
        file_root: Path,
        file_list_name: str,
        mesh_pos,
        normalization,
        shapenet_options,
    ):
        super().__init__()
        self.file_root: Path = file_root
        with open(self.file_root / "meta" / "shapenet.json", "r") as fp:
            labels_map = sorted(list(json.load(fp).keys()))

        self.labels_map: t.Dict[str, int] = {
            k: i for i, k in enumerate(labels_map)
        }
        # Read file list
        with open(self.file_root / "meta" / f"{file_list_name}.txt", mode="rt") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.tensorflow = "_tf" in file_list_name  # tensorflow version of data
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border

    def __getitem__(self, index: int):
        if self.tensorflow:
            # self.file_names[index] = "Data/ShapeNetP2M/04256520/1a201d0a99d841ca684b7bc3f8a9aa55/rendering/04.dat"
            # filename = "04256520/1a201d0a99d841ca684b7bc3f8a9aa55/rendering/04.dat"
            # relative_path = self.file_names[index][17:]
            relative_path = Path(self.file_names[index][17:])
            pkl_path = self.file_root / "data_tf" / relative_path
            label = pkl_path.parents[2].name
            img_path = pkl_path.parent / f"{pkl_path.stem}.png"
            with open(pkl_path) as f:
                data = pickle.load(open(pkl_path, 'rb'), encoding="latin1")
            pts, normals = data[:, :3], data[:, 3:]
            img = io.imread(img_path)
            img[np.where(img[:, :, 3] == 0)] = 255
            if self.resize_with_constant_border:
                img = transform.resize(
                    img,
                    (config.IMG_SIZE, config.IMG_SIZE),
                    mode='constant',
                    anti_aliasing=False
                )  # to match behavior of old versions
            else:
                img = transform.resize(
                    img,
                    (config.IMG_SIZE, config.IMG_SIZE),
                )
            img = img[:, :, :3].astype(np.float32)
        else:
            label, fpath = self.file_names[index].split("_", maxsplit=1)
            relative_path = Path(fpath)
            with open(self.file_root / "data" / label / relative_path, mode="rb") as f:  # type: ignore
                data = pickle.load(f, encoding="latin1")  # type: ignore

            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": str(relative_path),
            "length": length
        }

    def __len__(self):
        return len(self.file_names)


class ShapeNetImageFolder(BaseDataset):

    def __init__(self, folder, normalization, shapenet_options):
        super().__init__()
        self.normalization = normalization
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.file_list = []
        for fl in os.listdir(folder):
            file_path = os.path.join(folder, fl)
            # check image before hand
            try:
                if file_path.endswith(".gif"):
                    raise ValueError("gif's are results. Not acceptable")
                Image.open(file_path)
                self.file_list.append(file_path)
            except (IOError, ValueError):
                print("=> Ignoring %s because it's not a valid image" % file_path)

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "filepath": self.file_list[item]
        }

    def __len__(self):
        return len(self.file_list)


def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    def shapenet_collate(batch):
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
