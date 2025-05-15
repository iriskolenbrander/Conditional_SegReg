import torch
import lightning.pytorch as pl
from monai.losses import DiceLoss, GlobalMutualInformationLoss
from monai.losses.dice import one_hot
from torchmetrics import Metric

from Pytorch_lightning.lightning_metrics import MyDiceMultiLabel
from Registration.losses import Grad


class MyLightningModuleWeakSupervision(pl.LightningModule):
    def __init__(self, config, model):
        super(MyLightningModuleWeakSupervision, self).__init__()
        self.config = config
        self.model = model
        self.reg_weight = config.reg_weight
        self.seg_weight = config.seg_weight
        self.sim_weight = 1 - abs(self.seg_weight)
        self.stl_segmentation_maps = model.stl

        # Losses
        self.similarity_loss = GlobalMutualInformationLoss()
        self.seg_loss = DiceLoss(include_background=False,
                                 jaccard=True)
        self.smooth_loss = Grad(penalty='l2')

        # Validation metrics
        self.dice = MyDiceMultiLabel(max_label=1).to(self.config.device)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer

    def forward(self, input_channels):
        return self.model.forward(input_channels)

    def loss(self, source, target, dvf,
             source_map=None, target_map=None):
        loss_sim = self.sim_weight * self.similarity_loss.forward(source, target)
        loss_reg = self.reg_weight * self.smooth_loss.forward(dvf)
        loss_seg = self.seg_weight * self.seg_loss.forward(source_map, target_map) if source_map is not None else 0.0
        loss = loss_sim + loss_reg + loss_seg
        return loss

    def training_step(self, batch, batch_idx):
        # self.config
        source, target, source_map, target_map = batch
        if self.seg_weight == 0.0:
            source_map, target_map = None, None
        else:
            source_map_orig = source_map.clone()
            num_classes = int(target_map.max()) + 1
            source_map = one_hot(source_map, num_classes=num_classes)
            target_map = one_hot(target_map, num_classes=num_classes)

        if self.config.in_channels == 2:
            input_channels = [source, target]
        elif self.config.in_channels == 3:
            input_channels = [source, target, source_map_orig]

        dvf, source_warped = self.forward(input_channels)
        source_map_warped = self.stl_segmentation_maps(source_map, dvf) if source_map is not None else None
        loss = self.loss(source_warped, target, dvf, source_map_warped, target_map)
        self.log_dict({'train-loss': loss},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def val_test_step(self, batch, batch_idx, prefix, log=True, return_all=False):
        source, target, source_map, target_map = batch
        if self.config.in_channels == 2:
            input_channels = [source, target]
        elif self.config.in_channels == 3:
            input_channels = [source, target, source_map]
        dvf, source_warped = self.forward(input_channels)
        loss = self.loss(source_warped, target, dvf)
        source_map_warped = self.model.stl_binary(source_map, dvf)

        if log:
            dice = self.dice(source_map_warped, target_map)
            metrics_dict = {f'{prefix}loss': loss.item(),
                            f'{prefix}dice': dice.mean(),
                            f'{prefix}dice_median': dice.median()
                            }
            self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        if return_all:
            return (source, target, source_warped), dvf, metrics_dict
        else:
            return

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, prefix='val-')
        return

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, prefix='test-')
        return
