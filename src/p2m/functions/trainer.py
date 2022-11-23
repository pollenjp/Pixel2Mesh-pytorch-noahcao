# Third Party Library
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader

# First Party Library
from p2m.functions.base import CheckpointRunner
from p2m.functions.evaluator import Evaluator
from p2m.models.classifier import Classifier
from p2m.models.losses.classifier import CrossEntropyLoss
from p2m.models.losses.p2m import P2MLoss
from p2m.models.p2m import P2MModel
from p2m.models.p2m_with_template import P2MModelWithTemplate
from p2m.utils.average_meter import AverageMeter
from p2m.utils.mesh import Ellipsoid
from p2m.utils.tensor import recursive_detach

# from p2m.utils.vis.renderer import MeshRenderer


class Trainer(CheckpointRunner):

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        self.renderer = None

        if self.options.model.name in ["pixel2mesh", "pixel2mesh_with_template"]:
            # Visualization renderer
            # self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
            #                              self.options.dataset.mesh_pos)
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)

        if shared_model is not None:
            self.model = shared_model
        else:
            if self.options.model.name == "pixel2mesh_with_template":
                # create model
                self.model = P2MModelWithTemplate(
                    self.options.model,
                    self.ellipsoid,
                    self.options.dataset.camera_f,
                    self.options.dataset.camera_c,
                    self.options.dataset.mesh_pos,
                )
            elif self.options.model.name == "pixel2mesh":
                # create model
                self.model = P2MModel(
                    self.options.model,
                    self.ellipsoid,
                    self.options.dataset.camera_f,
                    self.options.dataset.camera_c,
                    self.options.dataset.mesh_pos,
                )
            elif self.options.model.name == "classifier":
                self.model = Classifier(self.options.model, self.options.dataset.num_classes)
            else:
                raise NotImplementedError("Your model is not found")
            self.model = DataParallel(self.model, device_ids=self.gpus).cuda()

        # Setup a joint optimizer for the 2 models
        if self.options.optim.name == "adam":
            self.optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd,
            )
        elif self.options.optim.name == "sgd":
            self.optimizer = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd,
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        # Create loss functions
        if self.options.model.name in ["pixel2mesh", "pixel2mesh_with_template"]:
            self.criterion = P2MLoss(self.options.loss, self.ellipsoid).cuda()
        elif self.options.model.name == "classifier":
            self.criterion = CrossEntropyLoss()
        else:
            raise NotImplementedError("Your loss is not found")

        # Create AverageMeters for losses
        self.losses = AverageMeter()

        # Evaluators
        self.evaluators = [Evaluator(self.options, self.logger, self.summary_writer, shared_model=self.model)]

    def models_dict(self):
        return {"model": self.model}

    def optimizers_dict(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}

    def train_step(self, input_batch):
        self.model.train()

        # predict with model
        if self.options.model.name == "pixel2mesh_with_template":
            b = {
                "images": input_batch["images"],
                "init_pts": input_batch["init_pts"],
            }
        else:
            b = input_batch["images"]
        out = self.model(b)

        # compute loss
        loss, loss_summary = self.criterion(out, input_batch)
        self.losses.update(loss.detach().cpu().item())

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def train(self):
        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1

            # Create a new data loader for every epoch
            train_data_loader = DataLoader(
                self.dataset,
                batch_size=self.options.train.batch_size * self.options.num_gpus,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                shuffle=self.options.train.shuffle,
                collate_fn=self.dataset_collate_fn,
            )

            # Reset loss
            self.losses.reset()

            # Iterate over all batches in an epoch
            for step, batch in enumerate(train_data_loader):
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Run training step
                out = self.train_step(batch)

                self.step_count += 1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()

            # save checkpoint after each epoch
            self.dump_checkpoint()

            # Run validation every test_epochs
            if self.epoch_count % self.options.train.test_epochs == 0:
                self.test()

            # lr scheduler step
            self.lr_scheduler.step()

    def train_summaries(self, input_batch, out_summary, loss_summary):
        if self.renderer is not None:
            # Do visualization for the first 2 images of the batch
            render_mesh = self.renderer.p2m_batch_visualize(input_batch, out_summary, self.ellipsoid.faces)
            # p2m_batch_visualize
            self.summary_writer.add_image("render_mesh", render_mesh, self.step_count)
            self.summary_writer.add_histogram(
                "length_distribution", input_batch["length"].cpu().numpy(), self.step_count
            )

        # Debug info for filenames
        self.logger.debug(input_batch["filename"])

        # Save results in Tensorboard
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)

        # Save results to log
        self.logger.info(
            "Epoch {:03d}, Step {:06d}/{:06d}, Time elapsed {}, Loss {:.9f} ({:.9f})".format(
                self.epoch_count,
                self.step_count,
                self.options.train.num_epochs
                * len(self.dataset)
                // (self.options.train.batch_size * self.options.num_gpus),
                self.time_elapsed,
                self.losses.val,
                self.losses.avg,
            )
        )

    def test(self):
        for evaluator in self.evaluators:
            evaluator.evaluate()
