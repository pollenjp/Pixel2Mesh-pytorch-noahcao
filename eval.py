# Standard Library
import argparse
import concurrent.futures
import logging
import pickle
import typing as t
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

# Third Party Library
import numpy as np
import numpy.typing as npt
import torch
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.loss.chamfer import knn_points
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.structures.meshes import Meshes

# First Party Library
from p2m.utils.average_meter import AverageMeter
from p2m.utils.eval import calc_f1_score
from p2m.utils.obj import load_objs_as_meshes

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
    level=logging.WARNING,
)
logger = getLogger(__name__)
logger.setLevel(logging.INFO)


def sample_points_from_obj_file(obj_files: list[Path], num_samples: int) -> Pointclouds:
    meshes: Meshes = load_objs_as_meshes(
        files=obj_files,
        mtl_dirs=[Path(".") for i in range(len(obj_files))],
        device="cpu",
        load_textures=False,
        create_texture_atlas=True,
    )
    # Size(num_batch, num_vertices, 3)
    x = Pointclouds(t.cast(torch.Tensor, sample_points_from_meshes(meshes=meshes, num_samples=num_samples)))
    return x


def calc_chamfer_distance_score(
    pred_v: torch.Tensor | Pointclouds,
    gt_v: torch.Tensor | Pointclouds,
) -> torch.Tensor:
    """
    chamfer distanceを計算する
    """
    # chamfer distance
    cd = t.cast(
        torch.Tensor,
        chamfer_distance(pred_v, gt_v)[0],
    )
    return cd


def calc_f1_score_from_points(pred_v: torch.Tensor, gt_v: torch.Tensor, tau: float = 1e-4) -> tuple[float, float]:
    knn1 = knn_points(pred_v, gt_v, K=1)
    knn2 = knn_points(gt_v, pred_v, K=1)
    dis_to_pred = knn1.dists.squeeze(0).squeeze(1)  # (num_gt_points, )
    dis_to_gt = knn2.dists.squeeze(0).squeeze(1)
    pred_length = dis_to_pred.size(0)
    gt_length = dis_to_gt.size(0)
    f1_tau = calc_f1_score(dis_to_pred, dis_to_gt, pred_length, gt_length, thresh=tau)
    f1_2tau = calc_f1_score(dis_to_pred, dis_to_gt, pred_length, gt_length, thresh=2 * tau)

    return (float(f1_tau), float(f1_2tau))


def load_gt_data(pkl_path: Path) -> npt.NDArray[np.float32]:
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp, encoding="latin1")

    pts = data[:, :3]
    return pts


def calc_cd_and_f1_from_file(pred_fpath: Path, gt_fpath: Path, tau: float) -> tuple[float, float, float]:
    # pred
    logger.debug("before start")
    x = sample_points_from_obj_file([pred_fpath], 10000)
    logger.debug("end")

    # gt
    y_np = load_gt_data(gt_fpath)
    y_np -= np.array(
        [
            0.0,
            0.0,
            -0.8,
        ]
    )
    y = torch.tensor(np.array([y_np]))
    logger.debug(type(y))
    logger.debug(y.shape)
    logger.debug(y.dtype)

    cd = calc_chamfer_distance_score(x, y)
    f1_tau, f1_2tau = calc_f1_score_from_points(
        pred_v=x.points_list()[0].unsqueeze(0),
        gt_v=y,
        tau=tau,
    )
    return (
        float(cd),
        float(f1_tau),
        float(f1_2tau),
    )


def search_file_iter(dir: Path) -> t.Iterator[Path]:
    for p in dir.glob("**/*_2.obj"):
        yield p


@dataclass
class Info:
    obj_fpath: Path
    pkl_fpath: Path
    tau: float
    debug_info: str | None = None


TypeCdAndF1 = tuple[float, float, float]


def run_worker(info: Info) -> TypeCdAndF1:
    return calc_cd_and_f1_from_file(info.obj_fpath, info.pkl_fpath, tau=info.tau)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pred_dir", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument("--gt_dir", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument("--num_workers", default=1, type=int)
    args = parser.parse_args()
    return args


def recursive_calc(pred_dir: Path, gt_dir: Path, tau: float, num_workers: int = 1) -> TypeCdAndF1:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future: concurrent.futures.Future[TypeCdAndF1]
        future_to_fpath: t.Dict[concurrent.futures.Future[TypeCdAndF1], Info] = {}
        info: Info
        for i, obj_fpath in enumerate(search_file_iter(pred_dir)):
            # if i > 2:
            #     break
            i_label: str = obj_fpath.parents[0].name
            object_id: str = obj_fpath.parents[1].name
            category_id: str = obj_fpath.parents[2].name
            pkl_fpath: Path = gt_dir / category_id / object_id / "rendering" / f"{i_label}.dat"

            info = Info(
                obj_fpath=obj_fpath,
                pkl_fpath=pkl_fpath,
                tau=tau,
                debug_info=f"{category_id=}, {object_id=}, {i_label=}",
            )
            logger.info(f"{i:>5}: {info.obj_fpath}")
            future_to_fpath[executor.submit(run_worker, info)] = info

        total_cd = AverageMeter()
        total_f1_tau = AverageMeter()
        total_f1_2tau = AverageMeter()
        for future in concurrent.futures.as_completed(future_to_fpath):
            info = future_to_fpath[future]
            try:
                cd, f1_tau, f1_2tau = future.result()
                total_cd.update(cd)
                total_f1_tau.update(f1_tau)
                total_f1_2tau.update(f1_2tau)
            except Exception as exc:
                raise exc
            else:
                logger.info(f"Completed: {info.debug_info}")

        print(f"total_cd={total_cd.avg}")
        print(f"total_f1_tau={total_f1_tau.avg}")
        print(f"total_f1_2tau={total_f1_2tau.avg}")
        return (total_cd.avg, total_f1_tau.avg, total_f1_2tau.avg)


def main_debug() -> None:

    category_id: str = "02691156"
    object_id: str = "d3b9114df1d8a3388e415c6cf89025f0"
    i: int = 0
    i_label = f"{i:0>2}"
    pkl_fpath: Path = Path(
        f"/home/hiroakisugisaki/workdir/data_local/dataset/2022-11-10/shapenet_p2m_custom/{category_id}/{object_id}/rendering/{i_label}.dat"
    )
    obj_fpath: Path = Path(f"tested/2023-01-25T035025/output/{category_id}/{object_id}/{i_label}/mesh_pred_coord_2.obj")

    cd, f1_tau, f1_2tau = calc_cd_and_f1_from_file(obj_fpath, pkl_fpath, tau=1e-4)
    logger.info(f"{cd=} / {f1_tau=} / {f1_2tau=}")


if __name__ == "__main__":
    # args = get_args()
    # main(
    #     pred_dir=args.pred_dir,
    #     gt_dir=args.gt_dir,
    #     num_workers=args.num_workers,
    # )

    base_dir = Path("tested")
    dir_list = [
        base_dir / "2023-01-25T035025/",  # p2m (vgg16)
        base_dir / "2023-01-25T035857/",  # p2m (resnet50)
        base_dir / "2023-01-25T040743/",  # pixel2mesh_with_depth (vgg16+resnet50)
        base_dir / "2023-01-25T041623/",  # p2m_with_depth_3d_cnn
        base_dir / "2023-01-25T185041/",  # p2m_with_depth (vgg16)
        base_dir / "2023-01-25T191049/",  # p2m_with_depth (resnet50)
        base_dir / "2023-01-25T192844/",  # pixel2mesh_with_depth_resnet_3d_cnn
    ]
    gt_dir = Path("~/workdir/data_local/dataset/2022-11-10/shapenet_p2m_custom/").expanduser().absolute()
    for pred_dir in dir_list:
        pred_dir /= "output/02691156/d3b9114df1d8a3388e415c6cf89025f0"
        cd, f1_tau, f1_2tau = recursive_calc(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            tau=1e-4,
            num_workers=8,
        )

        with open("out.txt", "a") as f:
            f.write(f"{cd}\t{f1_tau}\t{f1_2tau}\t{pred_dir}\n")
