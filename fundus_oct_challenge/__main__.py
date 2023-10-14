import argparse
import os
import pytorch_lightning as pl
import wandb

# replace with your own username to get your own version
# we can also create a shared project
os.getenv('API_USER')

wandb.init(project='oct_fundus', entity='rockfor')
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .config.defaults import get_cfg_defaults
from .lightning_modules.pl_module import FundusOCTLightningModule
from .datasets.fundus_oct_dataset import FundusOctDataset
from .datasets.fundus_oct_cur_dataset import FundusOctCurDataset, TaskAwareSampler
from .transforms.oct_transforms import get_transforms
from .models.model_setup import get_model
from torch.utils.data import DataLoader


def main():
    cfg = parse_arguments()
    model = get_model(cfg)

    # manually move the model to gpu, pt lightning should do this but there was some error with the wrapper
    # so now we have to move it manually
    # model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # initialize lightning module with model and config
    lightning_module = FundusOCTLightningModule(model, cfg)

    # Create a logger
    wandb_logger = WandbLogger()
    # model checkpointing, uploads model checkpoints to wandb
    checkpoint_callback = ModelCheckpoint()
    
    # initialize pl trainer, with all the arguments
    trainer = pl.Trainer(max_epochs=cfg.TRAIN.EPOCHS, 
                         accelerator='gpu',
                         log_every_n_steps=cfg.TRAIN.LOG_FREQ,
                         precision=16,
                         devices=1, 
                         logger=wandb_logger, 
                         callbacks=[checkpoint_callback])

    # load datasets
    if cfg.TRAIN.DO_TRAIN:
        transforms_train = get_transforms(mode='train')
        transforms_val = get_transforms(mode='val')

        if cfg.TRAIN.TASK == "segmentation" or cfg.TRAIN.TASK == "reconstruction":
            train_dataset = FundusOctDataset(cfg.DATA.BASEPATH, "train", transforms=transforms_train, task=cfg.TRAIN.TASK)
            val_dataset = FundusOctDataset(cfg.DATA.BASEPATH, "val", transforms=transforms_val, task=cfg.TRAIN.TASK)
            train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS)
            val_dataloader = DataLoader(val_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS)

        elif cfg.TRAIN.TASK == "curriculum":
            train_dataset = FundusOctCurDataset(cfg.DATA.BASEPATH, "train", transforms=transforms_train, tasks=["segmentation","reconstruction"], batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE)
            val_dataset = FundusOctCurDataset(cfg.DATA.BASEPATH, "val", transforms=transforms_val, tasks=["segmentation","reconstruction"], batch_size=cfg.TRAIN.VAL_BATCH_SIZE)
            train_sampler = TaskAwareSampler(train_dataset, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE)
            val_sampler = TaskAwareSampler(val_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE)
            
            # shuffle is ignored with a custom sampler
            train_dataloader = DataLoader(train_dataset, num_workers=cfg.DATA.NUM_WORKERS, batch_sampler=train_sampler)
            val_dataloader = DataLoader(val_dataset, num_workers=cfg.DATA.NUM_WORKERS, batch_sampler=val_sampler)
        else:
            raise NotImplementedError(f"{cfg.TRAIN.TASK} task not yet implemented")

    
        # train model
        trainer.fit(lightning_module, train_dataloader, val_dataloader)

    elif cfg.EVALUATE.DO_EVAL:
        transforms_test = get_transforms(mode='test')
        test_dataset = FundusOctDataset(cfg.DATA.BASEPATH, "test", transforms=transforms_test)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.TRAIN.TEST_BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS)
        trainer.validate(lightning_module, test_dataloader)

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
