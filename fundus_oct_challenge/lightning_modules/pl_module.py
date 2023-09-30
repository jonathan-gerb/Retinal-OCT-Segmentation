import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.transforms.functional as TF
import torchmetrics
import wandb
from torchmetrics.classification import MulticlassAccuracy

from fundus_oct_challenge.utils import apply_colormap
from fundus_oct_challenge.utils.metrics import MeanEuclideanDistanceEdgeError, DiceCoefficient

class FundusOCTLightningModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super(FundusOCTLightningModule, self).__init__()
        self.model = model
        self.cfg = cfg
        
        # see the FundusOctDataset class for which class is which index.
        self.class_weights = torch.tensor([0, 0.4, 0.3, 0.3, 0, 0])
        self.dice_metric = DiceCoefficient(num_classes=cfg.MODEL.NUM_CLASSES, class_weights=self.class_weights)
        self.unweighted_dice_metric = DiceCoefficient(num_classes=cfg.MODEL.NUM_CLASSES)
        self.accuracy_metric = MulticlassAccuracy(num_classes=cfg.MODEL.NUM_CLASSES)
        self.med_metric = MeanEuclideanDistanceEdgeError()
        self.log_freq = cfg.TRAIN.LOG_FREQ

        # set how many images to log
        if cfg.TRAIN.N_LOG_IMAGES > cfg.TRAIN.TRAIN_BATCH_SIZE:
            self.n_log_images = cfg.TRAIN.TRAIN_BATCH_SIZE
        else:
            self.n_log_images = cfg.TRAIN.N_LOG_IMAGES

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # collapse target dimensions
        targets = targets.squeeze(dim=1)

        # forward pass of model
        outputs = self(inputs)

        loss = F.cross_entropy(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log images and segmentations every log_freq batches
        if batch_idx % self.log_freq == 0:
            self.log_images(inputs, targets, outputs)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # collapse target dimensions
        targets = targets.squeeze(dim=1)

        # forward pass of model
        outputs = self(inputs)
        
        # Compute and log metrics
        loss = F.cross_entropy(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        
        dice = self.dice_metric(outputs, targets)
        self.log('val_dice', dice, prog_bar=True)

        unweighted_dice = self.unweighted_dice_metric(outputs, targets)
        self.log('val_unweighted_dice', unweighted_dice, prog_bar=True)

        accuracy = self.accuracy_metric(outputs, targets)
        self.log('val_accuracy', accuracy, prog_bar=True)

        distance_error = self.med_metric(outputs.argmax(dim=1), targets)
        self.log('val_med', distance_error, prog_bar=True)

        # Log images and segmentations every log_freq batches
        if batch_idx % self.log_freq == 0:
            self.log_images(inputs, targets, outputs)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.TRAIN.LR)
        return optimizer
    
    def log_images(self, inputs, targets, outputs):
        self.logger.experiment.log({
                "input_images_val": [wandb.Image(img.float(), caption="Input Image") for img in inputs[:self.n_log_images]],
                "ground_truth_val": [wandb.Image(TF.to_pil_image(apply_colormap(img.int())), caption="Ground Truth") for img in targets[:self.n_log_images]],
                "predicted_segmentation_val": [wandb.Image(TF.to_pil_image(apply_colormap(img.int())), caption="Predicted Segmentation") for img in outputs.argmax(1)[:self.n_log_images]]
            })
