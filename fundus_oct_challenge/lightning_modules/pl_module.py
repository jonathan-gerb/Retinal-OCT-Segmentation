import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.transforms.functional as TF
import torchmetrics
import time
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from torchmetrics.classification import MulticlassAccuracy

from fundus_oct_challenge.utils import apply_colormap
from fundus_oct_challenge.utils.metrics import MeanEuclideanDistanceEdgeError, DiceCoefficient

class FundusOCTLightningModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super(FundusOCTLightningModule, self).__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters({k: v for k, v in cfg.items()})
        
        # see the FundusOctDataset class for which class is which index.
        self.class_weights = torch.tensor([0, 0.4, 0.3, 0.3, 0, 0])
        self.dice_metric = DiceCoefficient(num_classes=cfg.MODEL.NUM_CLASSES, class_weights=self.class_weights)
        self.unweighted_dice_metric = DiceCoefficient(num_classes=cfg.MODEL.NUM_CLASSES)
        self.accuracy_metric = MulticlassAccuracy(num_classes=cfg.MODEL.NUM_CLASSES)
        self.med_metric = MeanEuclideanDistanceEdgeError()
        self.log_freq = cfg.TRAIN.LOG_FREQ
        self.log_freq_img = cfg.TRAIN.LOG_FREQ_IMG

        # set how many images to log
        if cfg.TRAIN.N_LOG_IMAGES > cfg.TRAIN.TRAIN_BATCH_SIZE:
            self.n_log_images = cfg.TRAIN.TRAIN_BATCH_SIZE
        else:
            self.n_log_images = cfg.TRAIN.N_LOG_IMAGES

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, tasks = batch
        
        # all individual samples get a task from the dataloader
        # but its the same for all samples in a batch so just take the first
        assert len(set(tasks)) == 1, "not all the same tasks in batch! please check the dataset and sampler class"
        assert torch.max(targets) < 20, f"found class with number higher than 20, something is probably wrong. {torch.max(mask)}"

        task = tasks[0]

        self.model.task = task

        # collapse target dimensions
        targets = targets.squeeze(dim=1)

        # forward pass of model
        outputs = self(inputs)

        if task == "reconstruction":
            # we train all of the output layers to reconstruct the image
            # this way we don't have nonsensical outputs for the other logits
            # when finetuning on another task
            loss = 0
            for i in range(outputs.shape[1]):
                loss += F.mse_loss(inputs, outputs[:,i].unsqueeze(1))
            loss = loss / outputs.shape[1]

            self.log('train_loss_reconstruction', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.cfg.TRAIN.TRAIN_BATCH_SIZE)
            
            if batch_idx % self.log_freq_img == 0:
                outputs = outputs.mean(dim=1)
                self.logger.experiment.log({
                    "recon_input_val": [wandb.Image(img.float(), caption="Input Image") for img in inputs[:self.n_log_images]],
                    "recon_output_val": [wandb.Image(img, caption="Reconstructed Image") for img in outputs[:self.n_log_images]]
                })

        elif task == "segmentation":
            loss = F.cross_entropy(outputs, targets)

            # Log images and segmentations every log_freq batches  
            if batch_idx % self.log_freq_img == 0:
                self.log_images(inputs, targets, outputs)
            self.log('train_loss_segmentation', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.cfg.TRAIN.TRAIN_BATCH_SIZE)
        elif task == "classification":
            loss = F.cross_entropy(outputs, targets)
            self.log('train_loss_classification', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.cfg.TRAIN.TRAIN_BATCH_SIZE)

        return loss

    def validation_step(self, batch, batch_idx, is_test=False):
        split = 'val'
        if is_test:
            split = 'test'
        
        inputs, targets, tasks = batch

        # all individual samples get a task from the dataloader
        # but its the same for all samples in a batch so just take the first
        assert len(set(tasks)) == 1, "not all the same tasks in batch! please check the dataset and sampler class"
        task = tasks[0]

        # collapse target dimensions
        targets = targets.squeeze(dim=1)

        # forward pass of model
        outputs = self(inputs)

        if task == "reconstruction":
            # we train all of the output layers to reconstruct the image
            # this way we don't have nonsensical outputs for the other logits
            # when finetuning on another task
            loss = 0
            for i in range(outputs.shape[1]):
                loss += F.mse_loss(inputs, outputs[:,i].unsqueeze(1))
            loss = loss / outputs.shape[1]

            self.log(f'{split}_loss_reconstruction', loss, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)

            if batch_idx % self.log_freq == 0:
                outputs = outputs.mean(dim=1)
                self.logger.experiment.log({
                    f"recon_input_{split}": [wandb.Image(img.float(), caption="Input Image") for img in inputs[:self.n_log_images]],
                    f"recon_output_{split}": [wandb.Image(img, caption="Reconstructed Image") for img in outputs[:self.n_log_images]]
                })

        elif task == "segmentation":
            loss = F.cross_entropy(outputs, targets)
            # Log images and segmentations every log_freq batches
            if batch_idx % self.log_freq == 0:
                self.log_images(inputs, targets, outputs)

            start_time = time.time()
            self.log(f'{split}_loss_segmentation', loss, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
            loss_time = time.time()
            
            accuracy = self.accuracy_metric(outputs, targets)
            self.log(f'{split}_accuracy', accuracy, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
            accuracy_time = time.time()

            dice = self.dice_metric(outputs, targets)
            self.log(f'{split}_dice', dice, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
            dice_time = time.time()

            unweighted_dice = self.unweighted_dice_metric(outputs, targets)
            self.log(f'{split}_unweighted_dice', unweighted_dice, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
            dice_unweighted_time = time.time()

            # med takes super long like this (about 70x more than the others) so lets disable it for now
            # distance_error = self.med_metric(outputs.argmax(dim=1), targets)
            # self.log('val_med', distance_error, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
            med_time = time.time()

            # print(f"Time spend on all metrics: loss={loss_time - start_time:.4f}, acc={accuracy_time - loss_time:.4f}, dice={dice_time - accuracy_time:.4f}, dice_unweighted={dice_unweighted_time - dice_time:.4f}, med={med_time - dice_unweighted_time:.4f}")

        elif task == "classification":
            loss = F.cross_entropy(outputs, targets)
            self.log(f'{split}_loss_classification', loss, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)

            accuracy = self.accuracy_metric(outputs, targets)
            self.log(f'{split}_accuracy_classification', accuracy, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
        return loss
    

    def test_step(self, batch, batch_idx):
        """Test step is just a copy of validation_step, but with the names switched.
        To not have to update both calls, I just pass a special argument that changes the logging names.
        """
        return self.validation_step(batch, batch_idx, is_test=True)
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.TRAIN.LR)

        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.cfg.TRAIN.EPOCHS)
        if self.cfg.TRAIN.TASK == "reconstruction":
            monitor = "val_loss_reconstruction"
        elif self.cfg.TRAIN.TASK == "segmentation":
            monitor = "val_loss_segmentation"
        else:
            raise NotImplementedError(f"please specify the loss to monitor for the LR scheduler, not loss chosen for task: {self.cfg.TRAIN.TASK}")
        # Define the scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True),
            'monitor': monitor,
            'interval': 'epoch',
            'frequency': 1,
            'strict': True,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    
    def log_images(self, inputs, targets, outputs):
        self.logger.experiment.log({
                "seg_input_val": [wandb.Image(img.float(), caption="Input Image") for img in inputs[:self.n_log_images]],
                "seg_ground_truth_val": [wandb.Image(TF.to_pil_image(apply_colormap(img.int())), caption="Ground Truth") for img in targets[:self.n_log_images]],
                "seg_predicted_val": [wandb.Image(TF.to_pil_image(apply_colormap(img.int())), caption="Predicted Segmentation") for img in outputs.argmax(1)[:self.n_log_images]]
            })
