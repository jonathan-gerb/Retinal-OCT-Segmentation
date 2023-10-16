import argparse
import os
import pytorch_lightning as pl
import wandb

# replace with your own username to get your own version
# we can also create a shared project
# os.getenv('API_USER')

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from .config.defaults import get_cfg_defaults
from .lightning_modules.pl_module import FundusOCTLightningModule
from .datasets.fundus_oct_dataset import FundusOctDataset
from .datasets.fundus_oct_cur_dataset import FundusOctCurDataset, TaskAwareSampler
from .datasets.combined_dataset import CombinedOCTDataset
from .transforms.oct_transforms import get_transforms
from .models.model_setup import get_model
from torch.utils.data import DataLoader


def main():
    cfg = parse_arguments()
    model = get_model(cfg)

    # manually move the model to gpu, pt lightning should do this but there was some error with the wrapper
    # so now we have to move it manually
    # model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')


    if cfg.MODEL.RESUME_PATH != "":
        # load model that was previously trained using the codebase. In this case any settings
        # in the config about the model size etc are ignored because we just load the saved lightning module.
        kwargs = {
            
        }
        lightning_module = FundusOCTLightningModule.load_from_checkpoint(cfg.MODEL.RESUME_PATH, model=model, cfg=cfg)
        # We can also load models from the weights and biases registry as follows:
        # artifact = wandb_run.use_artifact('entity/your-project-name/model:v0', type='model')
        # artifact_dir = artifact.download()

    else:
        # initialize lightning module with model and config
        lightning_module = FundusOCTLightningModule(model, cfg)

    # Create a logger
    wandb_logger = WandbLogger(log_model=True, 
                               project='oct_fundus', 
                               entity='rockfor', 
                               dir="/mnt/mass_storage/master_ai/medical_ai_training_logs")
    
    # add callbacks
    # model checkpointing, uploads model checkpoints to wandb
    if cfg.TRAIN.TASK == "segmentation":
        checkpoint_callback = ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=3)
    elif cfg.TRAIN.TASK == "reconstruction":
        checkpoint_callback = ModelCheckpoint(monitor="val_loss_reconstruction", mode="min", save_top_k=3)
    elif cfg.TRAIN.TASK == "curriculum":
        raise NotImplementedError("please implement which loss should be monitored to select the best performing model")
    else:
        raise NotImplementedError(f"{cfg.TRAIN.TASK} task not yet implemented")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # initialize pl trainer, with all the arguments
    trainer = pl.Trainer(max_epochs=cfg.TRAIN.EPOCHS,
                         accelerator='gpu',
                         log_every_n_steps=cfg.TRAIN.LOG_FREQ,
                         precision=16,
                         accumulate_grad_batches=cfg.TRAIN.ACCUMUALTE_GRAD_BATCHES,
                         devices=1,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, lr_monitor])

    # load datasets
    if cfg.TRAIN.DO_TRAIN:
        transforms_train = get_transforms(mode='train')
        transforms_val = get_transforms(mode='val')
        transforms_test = get_transforms(mode='test')

        if cfg.DATA.DATASET.lower() ==  "goals":
            if cfg.TRAIN.TASK == "segmentation" or cfg.TRAIN.TASK == "reconstruction":
                train_dataset = FundusOctDataset(cfg.DATA.BASEPATH, "train", transforms=transforms_train, task=cfg.TRAIN.TASK)
                val_dataset = FundusOctDataset(cfg.DATA.BASEPATH, "val", transforms=transforms_val, task=cfg.TRAIN.TASK)
                test_dataset = FundusOctDataset(cfg.DATA.BASEPATH, "test", transforms=transforms_test)

                train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS)
                val_dataloader = DataLoader(val_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS)
                test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS)

            elif cfg.TRAIN.TASK == "curriculum":
                train_dataset = FundusOctCurDataset(cfg.DATA.BASEPATH, "train", transforms=transforms_train, tasks=["segmentation","reconstruction"], batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE)
                val_dataset = FundusOctCurDataset(cfg.DATA.BASEPATH, "val", transforms=transforms_val, tasks=["segmentation","reconstruction"], batch_size=cfg.TRAIN.VAL_BATCH_SIZE)
                test_dataset = FundusOctCurDataset(cfg.DATA.BASEPATH, "test", transforms=transforms_test)
                train_sampler = TaskAwareSampler(train_dataset, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE)
                val_sampler = TaskAwareSampler(val_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE)
                test_sampler = TaskAwareSampler(test_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE)

                # shuffle is ignored with a custom sampler
                train_dataloader = DataLoader(train_dataset, num_workers=cfg.DATA.NUM_WORKERS, batch_sampler=train_sampler)
                val_dataloader = DataLoader(val_dataset, num_workers=cfg.DATA.NUM_WORKERS, batch_sampler=val_sampler)
                test_dataloader = DataLoader(val_dataset, num_workers=cfg.DATA.NUM_WORKERS, batch_sampler=test_sampler)
            else:
                raise NotImplementedError(f"{cfg.TRAIN.TASK} task not yet implemented")
        
        elif cfg.DATA.DATASET.lower() == "combined":
            train_dataset = CombinedOCTDataset(cfg.DATA.BASEPATH, "train", transforms=transforms_train, task=cfg.TRAIN.TASK, 
                                               datasets=["GOALS", "kermany2018", "neh_ut_2021", "2015_BOE_CHIU", "OCTID"],
                                               img_size=cfg.DATA.IMG_SIZE)
            val_dataset = CombinedOCTDataset(cfg.DATA.BASEPATH, "val", transforms=transforms_val, task=cfg.TRAIN.TASK, 
                                             datasets=["GOALS", "kermany2018", "neh_ut_2021", "2015_BOE_CHIU", "OCTID"],
                                             img_size=cfg.DATA.IMG_SIZE)
            test_dataset = CombinedOCTDataset(cfg.DATA.BASEPATH, "test", transforms=transforms_test, task=cfg.TRAIN.TASK, 
                                              datasets=["GOALS", "kermany2018", "neh_ut_2021", "2015_BOE_CHIU", "OCTID"],
                                              img_size=cfg.DATA.IMG_SIZE)

            train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS)
            val_dataloader = DataLoader(val_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS)
            test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS)
        else:
            raise NotImplementedError(f"no dataset: {cfg.DATA.DATASET}")

        # train model
        try:
            trainer.fit(lightning_module, train_dataloader, val_dataloader)
        except KeyboardInterrupt:
            print("\nDetected Control-C. Moving on to testing...")

    # load the best model and do eval
    if cfg.EVALUATE.DO_EVAL:
        trainer.test(lightning_module, test_dataloader, ckpt_path='best')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fundus OCT Challenge")
    parser.add_argument("--cfg", required=True, help="Path to the configuration file", type=str)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    return cfg

if __name__ == '__main__':
    main()
