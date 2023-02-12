from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule

from src.evaluations import LEVEL_EPOCH, LEVEL_STAGE, PHASE_TEST, PHASE_TRAIN, PHASE_VALID
from src.evaluations import MetricGroup


class MNISTLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            metric_group: MetricGroup,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.metric_group = metric_group

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.metric_group.reset(level=LEVEL_STAGE)

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def train_step_end(self, outputs: Dict):
        """
        :param outputs: Anything returns by training_step
        :return:
        """

        """
        If using metrics in data parallel mode (dp),
        the metric update/logging should be done in the <mode>_step_end method
        see https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#common-pitfalls
        """

        # update and log metrics
        step_metrics = self.metric_group.batch_step(outputs, phase=PHASE_TRAIN)

        self.log_dict(step_metrics, prog_bar=True)

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def on_train_epoch_end(self):
        epoch_metrics = self.metric_group.epoch_step(phase=PHASE_TRAIN)
        self.log_dict(epoch_metrics)

        self.metric_group.reset(level=LEVEL_EPOCH, phases=PHASE_TRAIN)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step_end(self, outputs: Dict):
        """
        :param outputs: Anything returns by validation_step
        :return:
        """

        """
        If using metrics in data parallel mode (dp),
        the metric update/logging should be done in the <mode>_step_end method
        see https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#common-pitfalls
        """

        # update and log metrics
        step_metrics = self.metric_group.batch_step(outputs, phase=PHASE_VALID)

        self.log_dict(step_metrics, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric_group.epoch_step(phase=PHASE_VALID)
        self.log_dict(epoch_metrics)

        self.metric_group.reset(level=LEVEL_EPOCH, phases=PHASE_VALID)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step_end(self, outputs: Dict):
        """
        :param outputs: Anything returns by test_step
        :return:
        """

        """
        If using metrics in data parallel mode (dp),
        the metric update/logging should be done in the <mode>_step_end method
        see https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#common-pitfalls
        """

        # update and log metrics
        step_metrics = self.metric_group.batch_step(outputs, phase=PHASE_TEST)
        self.log_dict(step_metrics, prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_test_epoch_end(self):
        epoch_metrics = self.metric_group.epoch_step(phase=PHASE_TEST)
        self.log_dict(epoch_metrics)

        self.metric_group.reset(level=LEVEL_EPOCH, phases=PHASE_TEST)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
