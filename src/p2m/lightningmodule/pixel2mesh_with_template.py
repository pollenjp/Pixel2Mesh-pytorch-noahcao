# Standard Library
import typing as t
from pathlib import Path

# Third Party Library
import numpy as np
import PIL.Image
import pytorch_lightning as pl
import torch
import torchvision
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.loss.chamfer import knn_points

# First Party Library
from p2m.datasets.shapenet_with_template import P2MWithTemplateBatchData
from p2m.models.losses.p2m_loss import P2MLoss
from p2m.models.losses.p2m_loss import P2MLossForwardReturnSecondDict
from p2m.models.p2m_with_template import P2MModelWithTemplate
from p2m.models.p2m_with_template import P2MModelWithTemplateForwardReturn
from p2m.options import Options
from p2m.options import generate_lr_scheduler
from p2m.options import generate_optimizer
from p2m.utils import pl_loggers
from p2m.utils.average_meter import AverageMeter
from p2m.utils.eval import calc_f1_score
from p2m.utils.mesh import Ellipsoid
from p2m.utils.render import plot_point_cloud
from p2m.utils.render import plot_pred_meshes
from p2m.utils.render import write_obj_info


class P2MModelWithTemplateModule(pl.LightningModule):
    def __init__(
        self,
        options: Options,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.options = options
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        self.model = P2MModelWithTemplate(
            self.options.model,
            self.ellipsoid,
            self.options.dataset.camera_f,
            self.options.dataset.camera_c,
            self.options.dataset.mesh_pos,
        )
        self.p2m_loss = P2MLoss(self.options.loss, self.ellipsoid).cuda()

        self.train_epoch_loss_avg_meters: dict[str, AverageMeter] = {
            "loss": AverageMeter(),
            "loss_chamfer": AverageMeter(),
            "loss_edge": AverageMeter(),
            "loss_laplace": AverageMeter(),
            "loss_move": AverageMeter(),
            "loss_normal": AverageMeter(),
        }
        self.val_epoch_loss_avg_meters: dict[str, AverageMeter] = {
            "loss": AverageMeter(),
            "loss_chamfer": AverageMeter(),
            "loss_edge": AverageMeter(),
            "loss_laplace": AverageMeter(),
            "loss_move": AverageMeter(),
            "loss_normal": AverageMeter(),
        }

        self.val_global_step: int = 0

    def forward(self, batch: P2MWithTemplateBatchData) -> P2MModelWithTemplateForwardReturn:
        x: P2MModelWithTemplateForwardReturn = self.model(batch)
        return x

    def training_epoch_start(self, *, phase_name: str = "train") -> None:
        """
        custom method
        """
        return

    def training_step(  # type: ignore # [override]
        self,
        batch: P2MWithTemplateBatchData,
        batch_idx: int,
        # optimizer_idx: int,
        *,
        phase_name: str = "train",
    ) -> None:
        batch_size: int = len(batch["images"])

        if batch_idx == 0:
            self.training_epoch_start(phase_name=phase_name)

        preds: P2MModelWithTemplateForwardReturn = self(batch)
        summary: P2MLossForwardReturnSecondDict
        loss, summary = self.p2m_loss(preds, batch)

        # Gradient decent
        self.model.zero_grad()

        # backward
        loss.backward()

        # step
        self.solver.step()

        self.log(name="train_loss", value=loss, batch_size=batch_size)

        for loss_name, avg_meter in self.train_epoch_loss_avg_meters.items():
            avg_meter.update(summary[loss_name])

        if batch_idx == 0:
            self.custom_batch_log(
                phase_name=phase_name,
                batch=batch,
                preds=preds,
                step=self.current_epoch,
            )

    def training_epoch_end(self, training_step_outputs, *, phase_name: str = "train") -> None:
        for loss_name, avg_meter in self.train_epoch_loss_avg_meters.items():
            pl_loggers.pl_log_scalar(
                pl_logger=self.loggers,
                tag=f"{phase_name}/epoch_{loss_name}",
                scalar=avg_meter.avg,
                global_step=self.current_epoch,
            )

    def validation_epoch_start(self, *, phase_name: str = "val"):

        return

    def custom_step(
        self,
        *,
        batch: P2MWithTemplateBatchData,
        batch_idx: int,
        phase_name: str,
    ) -> P2MModelWithTemplateForwardReturn:
        batch_size: int = len(batch["images"])

        if batch_idx == 0:
            self.validation_epoch_start(phase_name=phase_name)

        preds: P2MModelWithTemplateForwardReturn = self(batch)
        summary: P2MLossForwardReturnSecondDict
        _, summary = self.p2m_loss(preds, batch)

        # for monitor label
        self.log(name=f"{phase_name}_loss", value=summary["loss"], batch_size=batch_size)

        for loss_name, avg_meter in self.val_epoch_loss_avg_meters.items():
            avg_meter.update(summary[loss_name])
            self.log(name=f"{phase_name}/{loss_name}", value=summary[loss_name], batch_size=batch_size)

        # calculate evaluation metrics
        pred_vertices = preds["pred_coord"][-1]  # (batch_size, num_vertices, 3)
        gt_points = batch["points_orig"]  # (batch_size, ) array. torch.Tensor(num_points, 3)

        total_chamfer_distance: torch.Tensor | float = 0.0
        total_f1_tau: torch.Tensor | float = 0.0
        total_f1_2tau: torch.Tensor | float = 0.0

        for i, (label, pred_v, gt_v) in enumerate(zip(batch["labels"], pred_vertices, gt_points)):
            pred_v = pred_v.unsqueeze(0)  # (1, num_vertices, 3)
            gt_v = gt_v.unsqueeze(0)

            # chamfer distance
            cd = t.cast(
                torch.Tensor,
                chamfer_distance(pred_v, gt_v)[0],
            )
            self.log(name=f"{phase_name}_eval_cd/{label}", value=cd, batch_size=1)
            total_chamfer_distance += cd

            # f1 score
            knn1 = knn_points(pred_v, gt_v, K=1)
            knn2 = knn_points(gt_v, pred_v, K=1)
            dis_to_pred = knn1.dists.squeeze(0).squeeze(1)  # (num_gt_points, )
            dis_to_gt = knn2.dists.squeeze(0).squeeze(1)
            pred_length = dis_to_pred.size(0)
            gt_length = dis_to_gt.size(0)
            tau: float = 1e-4
            f1_tau = calc_f1_score(dis_to_pred, dis_to_gt, pred_length, gt_length, thresh=tau)
            f1_2tau = calc_f1_score(dis_to_pred, dis_to_gt, pred_length, gt_length, thresh=2 * tau)

            self.log(name=f"{phase_name}_eval_f1_tau/{label}", value=f1_tau, batch_size=1)
            self.log(name=f"{phase_name}_eval_f1_2tau/{label}", value=f1_2tau, batch_size=1)
            total_f1_tau += f1_tau
            total_f1_2tau += f1_2tau

        self.log(name=f"{phase_name}_eval_cd", value=total_chamfer_distance / batch_size, batch_size=batch_size)
        self.log(name=f"{phase_name}_eval_f1_tau", value=total_f1_tau / batch_size, batch_size=batch_size)
        self.log(name=f"{phase_name}_eval_f1_2tau", value=total_f1_2tau / batch_size, batch_size=batch_size)

        return preds

    def validation_step(
        self,
        batch: P2MWithTemplateBatchData,
        batch_idx: int,
        *,
        phase_name: str = "val",
    ):
        self.val_global_step += 1
        preds = self.custom_step(batch=batch, batch_idx=batch_idx, phase_name=phase_name)

        if batch_idx == 0:
            self.custom_batch_log(
                phase_name=phase_name,
                batch=batch,
                preds=preds,
                step=self.current_epoch,
            )

    def validation_epoch_end(
        self,
        validation_step_outputs,
        *,
        phase_name: str = "val",
    ) -> None:
        for loss_name, avg_meter in self.val_epoch_loss_avg_meters.items():
            pl_loggers.pl_log_scalar(
                pl_logger=self.loggers,
                tag=f"{phase_name}/epoch_{loss_name}",
                scalar=avg_meter.avg,
                global_step=self.current_epoch,
            )

    def test_epoch_start(self, *, phase_name: str = "test") -> None:
        """
        custom method
        """
        return

    def test_step(
        self,
        batch: P2MWithTemplateBatchData,
        batch_idx: int,
        *,
        phase_name: str = "test",
    ):
        if batch_idx == 0:
            self.test_epoch_start(phase_name=phase_name)

        preds = self.custom_step(batch=batch, batch_idx=batch_idx, phase_name=phase_name)

        # TODO: save predicted data
        output_dir_path = Path(self.options.log_root_path) / "output"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        for i_elem in range(len(batch["images"])):
            # 04256520/1a4a8592046253ab5ff61a3a2a0e2484/rendering/00.dat
            f = Path(batch["filename"][i_elem])
            label_id = f.parents[2].name
            instance_id = f.parents[1].name
            d = output_dir_path / label_id / instance_id / f.stem
            d.mkdir(parents=True, exist_ok=True)

            name: str

            name = "images_orig"
            torchvision.utils.save_image(batch[name][i_elem], d / f"{name}.png")

            # point cloud
            for name in ["points_orig", "init_pts"]:
                PIL.Image.fromarray(
                    plot_point_cloud(vertices=batch[name][i_elem : i_elem + 1], num_cols=1),
                ).save(d / f"point_cloud_{name}.png")
            for scale_i in range(len(preds["pred_coord"])):
                name = "pred_coord"
                PIL.Image.fromarray(
                    plot_point_cloud(vertices=preds[name][scale_i][i_elem : i_elem + 1], num_cols=1),
                ).save(d / f"point_cloud_{name}_{scale_i}.png")

            # mesh
            for scale_i in range(len(preds["pred_coord"])):
                name = "pred_coord"
                torchvision.utils.save_image(
                    plot_pred_meshes(
                        coords=preds[name][scale_i][i_elem],
                        faces=self.ellipsoid.faces[scale_i] + 1,  # 0-origin to 1-origin
                        mtl_filepath=Path(self.options.mtl_filepath),
                        usemtl_name=self.options.usemtl_name,
                    )
                    .squeeze(0)[..., :3]
                    .permute(2, 0, 1),
                    d / f"mesh_{name}_{scale_i}.png",
                )

            # save as obj file
            for scale_i in range(len(preds["pred_coord"])):
                name = "pred_coord"
                with open(d / f"mesh_{name}_{scale_i}.obj", "wt") as f:
                    write_obj_info(
                        f=f,
                        coords=preds[name][scale_i][i_elem],
                        faces=self.ellipsoid.faces[scale_i] + 1,  # 0-origin to 1-origin
                        mtl_filename=Path(self.options.mtl_filepath).name,
                        usemtl_name=self.options.usemtl_name,
                    )

    def test_epoch_end(self, test_step_outputs, *, phase_name: str = "test"):
        self.validation_epoch_end(validation_step_outputs=test_step_outputs, phase_name=phase_name)
        return

    def configure_optimizers(self):

        self.solver: torch.optim.Optimizer = generate_optimizer(
            model=self.model,
            optim_data=self.options.optim,
        )
        self.lr_scheduler = generate_lr_scheduler(optimizer=self.solver, scheduler_data=self.options.lr_scheduler)

        optimizers: list[torch.optim.Optimizer] = [
            self.solver,
        ]
        lr_schedulers: list = [
            self.lr_scheduler,
        ]

        return optimizers, lr_schedulers

    def custom_batch_log(
        self,
        phase_name: str,
        batch: P2MWithTemplateBatchData,
        preds: P2MModelWithTemplateForwardReturn,
        step: int,
    ) -> None:
        batch_size = min(self.options.batch_size_for_plot, len(batch["images_orig"]))

        pl_loggers.pl_log_images(
            pl_logger=self.loggers,
            tag=f"{phase_name}_imgs",
            imgs_arr=batch["images_orig"][:batch_size],
            global_step=step,
            data_formats="NCHW",
        )

        pl_loggers.pl_log_images(
            pl_logger=self.loggers,
            tag=f"{phase_name}_vertices/gt_pred",
            imgs_arr=np.array(
                [
                    plot_point_cloud(vertices=batch["points_orig"][:batch_size], num_cols=1),
                    plot_point_cloud(vertices=batch["init_pts"][:batch_size], num_cols=1),
                    plot_point_cloud(vertices=preds["pred_coord"][-1][:batch_size], num_cols=1),
                ]
            ),
            global_step=step,
            data_formats="NHWC",
        )

        pl_loggers.pl_log_images(
            pl_logger=self.loggers,
            tag=f"{phase_name}/pred_meshes",
            imgs_arr=torch.cat(  # size: (3, batch_size * h, w, c)
                [
                    torch.cat(
                        [
                            plot_pred_meshes(
                                coords=preds["pred_coord"][i][j],
                                faces=self.ellipsoid.faces[i] + 1,  # 0-origin to 1-origin
                                mtl_filepath=Path(self.options.mtl_filepath),
                                usemtl_name=self.options.usemtl_name,
                            ).squeeze(0)
                            for j in range(batch_size)
                        ],
                        dim=0,
                    ).unsqueeze(0)
                    for i in range(3)
                ]
            ),
            global_step=step,
            data_formats="NHWC",
        )
