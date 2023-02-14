# Standard Library
import argparse
import concurrent.futures
import subprocess
import typing as t
from dataclasses import dataclass
from logging import NullHandler
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num_workers", default=1, type=int)
    args = parser.parse_args()
    return args


def search_file_iter(dir: Path) -> t.Iterator[Path]:
    for p in dir.glob("**/*_2.obj"):
        yield p


@dataclass
class Cmd:
    cmd: t.List[str]
    env: t.Optional[t.Dict[str, str]] = None
    stdout: t.Optional[t.Union[t.TextIO, int]] = None
    stderr: t.Optional[t.Union[t.TextIO, int]] = None


def run_cmd(cmd: Cmd) -> None:
    subprocess.run(cmd.cmd, stdout=cmd.stdout, stderr=cmd.stderr, env=cmd.env)


def main() -> None:
    args = get_args()

    cmd_args_list: list[list[str]] = [
        [  # p2m (vgg16)
            "model=pixel2mesh",
            "dataset=resnet_with_template_airplane",
            "model.backbone=VGG16P2M",
            "checkpoint_path='trained/shapenet_with_template/2023-01-03T154011/lightning_logs/model-checkpoint/val_loss=0.00182343-epoch=224-step=90900.ckpt'",
        ],
        [  # p2m (resnet50)
            "model=pixel2mesh",
            "dataset=resnet_with_template_airplane",
            "model.backbone=RESNET50",
            "checkpoint_path='trained/shapenet_with_template/2022-12-23T125418/lightning_logs/model-checkpoint/val_loss=0.00144993-epoch=91-step=37168.ckpt'",
        ],
        [  # p2m_with_depth (vgg16)
            "model=pixel2mesh_with_depth",
            "dataset=resnet_with_template_airplane",
            "model.backbone=VGG16P2M",
            "checkpoint_path='trained/shapenet_with_template/2022-12-30T170108/lightning_logs/model-checkpoint/val_loss=0.00176071-epoch=82-step=33532.ckpt'",
        ],
        [  # p2m_with_depth (resnet50)
            "model=pixel2mesh_with_depth",
            "dataset=resnet_with_template_airplane",
            "model.backbone=RESNET50",
            "checkpoint_path='trained/shapenet_with_template/2022-12-22T071729/lightning_logs/model-checkpoint/val_loss=0.00091946-epoch=199-step=80800.ckpt'",
        ],
        [  # p2m_with_depth (vgg16+resnet50)
            "model=pixel2mesh_with_depth_resnet",
            "dataset=resnet_with_template_airplane",
            "model.backbone=VGG16P2M",
            "checkpoint_path='trained/shapenet_with_template/2023-01-05T130620/lightning_logs/model-checkpoint/val_loss=0.00071044-epoch=98-step=80091.ckpt'",
        ],
        [  # p2m_with_depth_3d_cnn
            "model=pixel2mesh_with_depth_3d_cnn",
            "dataset=resnet_with_template_airplane",
            "model.backbone=RESNET50",
            "checkpoint_path='logs/train/P2M_WITH_DEPTH_3D_CNN/shapenet_with_template/2023-01-11T213742/lightning_logs/model-checkpoint/val_loss=0.00099826-epoch=26-step=10908.ckpt'",
        ],
        [  # pixel2mesh_with_depth_resnet_3d_cnn
            "model=pixel2mesh_with_depth_resnet_3d_cnn",
            "dataset=resnet_with_template_airplane",
            "model.backbone=VGG16P2M",
            "checkpoint_path='trained/shapenet_with_template/2023-01-10T122747/lightning_logs/model-checkpoint/val_loss=0.00102751-epoch=10-step=8899.ckpt'",
        ],
    ]

    # cmd_args_list = cmd_args_list[0:1]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_fpath: t.Dict[concurrent.futures.Future[None], Cmd] = {}
        cmd: Cmd
        for i, cmd_args in enumerate(cmd_args_list):
            # if i > 2:
            #     break
            logger.info(f"{i:>5}: {cmd_args}")
            cmd = Cmd(
                cmd=[
                    "./.venv/bin/python",
                    "src/test.py",
                    "batch_size=8",
                    "num_workers=8",
                    *cmd_args,
                ],
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL,
            )
            future_to_fpath[executor.submit(run_cmd, cmd)] = cmd

        future: concurrent.futures.Future[None]
        for future in concurrent.futures.as_completed(future_to_fpath):
            cmd = future_to_fpath[future]
            try:
                future.result()
            except Exception as exc:
                raise exc
            else:
                logger.info(f"Completed: {cmd.cmd}")


if __name__ == "__main__":
    # Standard Library
    import logging

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
        level=logging.INFO,
    )

    main()
