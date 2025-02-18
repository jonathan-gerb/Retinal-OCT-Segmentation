import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.transforms.functional as TF
import torchmetrics
from pathlib import Path
from PIL import Image
import numpy as np
import time
import csv
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
        self.log_freq_val = cfg.TRAIN.LOG_FREQ_VAL
        self.log_freq_train = cfg.TRAIN.LOG_FREQ_TRAIN

        # set how many images to log
        if cfg.TRAIN.N_LOG_IMAGES > cfg.TRAIN.TRAIN_BATCH_SIZE:
            self.n_log_images = cfg.TRAIN.TRAIN_BATCH_SIZE
        else:
            self.n_log_images = cfg.TRAIN.N_LOG_IMAGES

        self.mapping_dict = {
                0: 255, # background to png background
                1: 0,
                2: 80,
                3: 160,
                4: 255,
                5: 255
            }
        
        # variables for save_segmentations
        self.lookup_table = torch.arange(256)
        for k, v in self.mapping_dict.items():
            self.lookup_table[k] = v
        
        # for reference
        self.label2id = {
            "background_above": 0,
            "retinal nerve fiber layer": 1,
            "ganglion cell-inner plexiform layer": 2,
            "choroidal layer": 3,
            "background_between": 4,
            "background_below": 5,
        }
        # self.id2label = {v: k for k, v in self.label2id.items()}
        # hardcoded all backgrounds to be the same, no longer seperating backgrounds
        self.id2label = {
            0: "background",
            1: "retinal nerve fiber layer",
            2: "ganglion cell-inner plexiform layer",
            3: "choroidal layer",
            4: "background",
            5: "background",
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, tasks, img_path = batch
        
        # all individual samples get a task from the dataloader
        # but its the same for all samples in a batch so just take the first
        assert len(set(tasks)) == 1, "not all the same tasks in batch! please check the dataset and sampler class"
        assert torch.max(targets) < 20, f"found class with number higher than 20, something is probably wrong. {torch.max(targets)}"

        task = tasks[0]
        self.set_task(task)

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
            
            if batch_idx % self.log_freq_train == 0:
                outputs = outputs.mean(dim=1)
                self.logger.experiment.log({
                    "recon_input_train": [wandb.Image(img.float(), caption="Input Image") for img in inputs[:self.n_log_images]],
                    "recon_output_train": [wandb.Image(img, caption="Reconstructed Image") for img in outputs[:self.n_log_images]]
                })

        elif task == "segmentation":
            loss = F.cross_entropy(outputs, targets)

            # Log images and segmentations every log_freq batches  
            if batch_idx % self.log_freq_train == 0:
                self.log_images(inputs, targets, outputs, 'train')
            self.log('train_loss_segmentation', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.cfg.TRAIN.TRAIN_BATCH_SIZE)
        elif task == "classification":
            loss = F.cross_entropy(outputs, targets)
            self.log('train_loss_classification', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.cfg.TRAIN.TRAIN_BATCH_SIZE)

        return loss

    def validation_step(self, batch, batch_idx, is_test=False):
        split = 'val'
        if is_test:
            split = 'test'
        
        inputs, targets, tasks, img_path = batch

        # all individual samples get a task from the dataloader
        # but its the same for all samples in a batch so just take the first
        assert len(set(tasks)) == 1, "not all the same tasks in batch! please check the dataset and sampler class"
        task = tasks[0]
        
        # because of the wrapped 
        self.set_task(task)

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

            if batch_idx % self.log_freq_val == 0:
                outputs = outputs.mean(dim=1)
                self.logger.experiment.log({
                    f"recon_input_{split}": [wandb.Image(img.float(), caption="Input Image") for img in inputs[:self.n_log_images]],
                    f"recon_output_{split}": [wandb.Image(img, caption="Reconstructed Image") for img in outputs[:self.n_log_images]]
                })

        elif task == "segmentation":
            loss = F.cross_entropy(outputs, targets)
            # Log images and segmentations every log_freq batches
            if batch_idx % self.log_freq_val == 0:
                self.log_images(inputs, targets, outputs, "val")

            start_time = time.time()
            self.log(f'{split}_loss_segmentation', loss, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
            loss_time = time.time()
            
            accuracy = self.accuracy_metric(outputs, targets)
            self.log(f'{split}_accuracy', accuracy, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
            accuracy_time = time.time()

            dice, dice_per_class = self.dice_metric(outputs, targets)
            self.log(f'{split}_dice', dice, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)
            dice_time = time.time()

            for id, labelname in self.id2label.items():
                class_dice = dice_per_class[int(id)]
                self.log(f'{split}_dice_{labelname}', class_dice, prog_bar=True, batch_size=self.cfg.TRAIN.VAL_BATCH_SIZE)

            unweighted_dice, _  = self.unweighted_dice_metric(outputs, targets)
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
    

    def set_task(self, task):
        """Because of the model wrapping we need to set set the task in .model.model instead of just .model

        Args:
            task (str): task to set in the model
        """
        try:
            self.model.model.task = task
        except AttributeError:
            self.model.task = task
    

    def configure_optimizers(self):
        # add parameters to model, but only those that require gradients,
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.TRAIN.LR)

        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.cfg.TRAIN.EPOCHS)
        if self.cfg.TRAIN.TASK == "reconstruction":
            monitor = "val_loss_reconstruction"
        elif self.cfg.TRAIN.TASK == "segmentation":
            monitor = "val_loss_segmentation"
        else:
            raise NotImplementedError(f"please specify the loss to monitor for the LR scheduler, not loss chosen for task: {self.cfg.TRAIN.TASK}")
        # Define the scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, factor=self.cfg.TRAIN.LR_REDUCER.FACTOR, patience=self.cfg.TRAIN.LR_REDUCER.PATIENCE, verbose=True),
            'monitor': monitor,
            'interval': 'epoch',
            'frequency': 1,
            'strict': True,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    
    def log_images(self, inputs, targets, outputs, mode):
        self.logger.experiment.log({
                f"seg_input_{mode}": [wandb.Image(img.float(), caption="Input Image") for img in inputs[:self.n_log_images]],
                f"seg_ground_truth_{mode}": [wandb.Image(TF.to_pil_image(apply_colormap(img.int())), caption="Ground Truth") for img in targets[:self.n_log_images]],
                f"seg_predicted_{mode}": [wandb.Image(TF.to_pil_image(apply_colormap(img.int())), caption="Predicted Segmentation") for img in outputs.argmax(1)[:self.n_log_images]]
            })


    def map_values(self, tensor):
        # Use the tensor values as indices to get the mapped values
        return torch.index_select(self.lookup_table, 0, tensor.reshape(-1)).reshape(tensor.shape)


    def save_segmentations(self, dataloader, output_path, channel_first=True):
        self.eval()
        csv_filename = output_path.parents[1] / "segmentation_scores.csv"
        
        # set task (function because the model might be wrapped)
        self.set_task("segmentation")
        dice_scores = []
        filepaths = []

        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write header to CSV
            csvwriter.writerow(["Image Name", "Dice Score"])

            for batch in dataloader:
                inputs, targets, _, img_paths = batch
                inputs = inputs.to(self.device)

                img_path = img_paths[0]
                img_name = Path(img_path).name
                segmentation_output_path = output_path / img_name

                # collapse target dimensions
                targets = targets.squeeze(dim=1)

                # forward pass of model
                with torch.no_grad():
                    outputs = self(inputs).cpu()

                dice, _ = self.dice_metric(outputs.to('cuda:0'), targets.to('cuda:0'))
                dice_scores.append(float(dice))
                filepaths.append(img_path)

                # collapse dimensions
                if channel_first:
                    outputs = outputs.argmax(dim=-3)
                else:
                    outputs = outputs.argmax(dim=-1)

                # convert label values to rgb values
                # outputs = self.map_values(outputs)

                # Convert the tensor to a numpy array and then to a PIL Image
                outputs = outputs.cpu().squeeze()
                outputs_np = apply_colormap(outputs.int()).numpy().astype(np.uint8)

                # Ensure it's 2D before converting to an image
                if len(outputs_np.shape) > 3:
                    raise ValueError("Unexpected shape for output array:", outputs_np.shape)
                
                img = Image.fromarray(outputs_np.transpose(1,2,0))
                img.save(str(segmentation_output_path))

                # targets_np = apply_colormap(targets.squeeze().int()).numpy().astype(np.uint8)
                # img = Image.fromarray(targets_np.transpose(1,2,0))
                # img.save(str(segmentation_output_path))


                # Write to CSV
                csvwriter.writerow([img_name, float(dice)])

        # Pair up the dice scores with their corresponding filepaths
        pairs = list(zip(dice_scores, filepaths))

        # Sort the pairs based on dice scores
        sorted_pairs = sorted(pairs, key=lambda x: x[0])

        # Get the filepaths corresponding to the k lowest dice scores
        print(f"mean dice: {np.mean(dice_scores)}")
        print(f"The files with scores are: ")
        for i in range(len(sorted_pairs)):
            score, fp = sorted_pairs[i][0], sorted_pairs[i][1]
            print(f"file: '{str(Path(fp).name)}' dice: {score:.4f}")





            

