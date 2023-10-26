import argparse
import os
import pytorch_lightning as pl
from pathlib import Path

# replace with your own username to get your own version
# we can also create a shared project
# os.getenv('API_USER')
import torch
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from .config.defaults import get_cfg_defaults
from .lightning_modules.pl_module import FundusOCTLightningModule
from .datasets.fundus_oct_dataset import FundusOctDataset
from .datasets.wrapper_datasets import OversampledDataset
from .datasets.fundus_oct_cur_dataset import FundusOctCurDataset, TaskAwareSampler
from .datasets.combined_dataset import CombinedOCTDataset
from .transforms.oct_transforms import get_transforms
from .models.model_setup import get_model
from .utils.model_utils import get_limited_dataset
from torch.utils.data import DataLoader, Subset


def main():
    cfg = parse_arguments()
    assert (
        cfg.USERNAME != ""
    ), "please specify you weights adn biases username in teh yaml config as: USERNAME: 'username'"

    model = get_model(cfg)

    # manually move the model to gpu, pt lightning should do this but there was some error with the wrapper
    # so now we have to move it manually
    # model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')

    if cfg.MODEL.RESUME_PATH != "":
        # load model that was previously trained using the codebase. In this case any settings
        # in the config about the model size etc are ignored because we just load the saved lightning module.
        lightning_module = FundusOCTLightningModule.load_from_checkpoint(
            cfg.MODEL.RESUME_PATH, model=model, cfg=cfg
        )
        print(f"loaded weights from: {cfg.MODEL.RESUME_PATH}")
        # We can also load models from the weights and biases registry as follows:
        # artifact = wandb_run.use_artifact('entity/your-project-name/model:v0', type='model')
        # artifact_dir = artifact.download()

    else:
        # initialize lightning module with model and config
        lightning_module = FundusOCTLightningModule(model, cfg)

    # Create a logger
    wandb.init(dir="/mnt/mass_storage/master_ai/medical_ai_training_logs/", name=cfg.TRAIN.RUN_NAME, project="oct_fundus_final")
    wandb_logger = WandbLogger(
        log_model=True,
        entity=cfg.USERNAME,
        dir="/mnt/mass_storage/master_ai/medical_ai_training_logs/"
    )

    # add callbacks
    # model checkpointing, uploads model checkpoints to wandb
    if cfg.TRAIN.TASK == "segmentation":
        checkpoint_callback = ModelCheckpoint(
            monitor="val_dice", mode="max", save_top_k=3, dirpath="/mnt/mass_storage/master_ai/medical_ai_training_logs/"
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss_segmentation',
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='min'
        )
    elif cfg.TRAIN.TASK == "reconstruction":
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_reconstruction", mode="min", save_top_k=3, dirpath="/mnt/mass_storage/master_ai/medical_ai_training_logs/model_checkpoints"
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss_reconstruction',
            min_delta=0.00,
            patience=4,
            verbose=False,
            mode='min'
        )
    elif cfg.TRAIN.TASK == "curriculum":
        raise NotImplementedError(
            "please implement which loss should be monitored to select the best performing model"
        )
    else:
        raise NotImplementedError(f"{cfg.TRAIN.TASK} task not yet implemented")
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")


    # initialize pl trainer, with all the arguments
    trainer = pl.Trainer(
        max_epochs=cfg.TRAIN.EPOCHS,
        default_root_dir="/mnt/mass_storage/master_ai/medical_ai_training_logs/lightning_logs",
        accelerator="gpu",
        log_every_n_steps=cfg.TRAIN.LOG_FREQ_TRAIN,
        precision=16,
        accumulate_grad_batches=cfg.TRAIN.ACCUMUALTE_GRAD_BATCHES,
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
    )

    # load datasets
    if cfg.TRAIN.DO_TRAIN:
        transforms_train = get_transforms(mode="train", no_sptransforms=cfg.TRAIN.NO_SPTRANSFORMS, no_transforms=cfg.TRAIN.NO_TRANSFORMS)
        transforms_val = get_transforms(mode="val")
        transforms_test = get_transforms(mode="test")

        if cfg.DATA.DATASET.lower() == "goals":
            if cfg.TRAIN.TASK == "segmentation" or cfg.TRAIN.TASK == "reconstruction":
                train_dataset = FundusOctDataset(
                    cfg.DATA.BASEPATH,
                    "train",
                    transforms=transforms_train,
                    task=cfg.TRAIN.TASK,
                    separate_bottom_bg=cfg.TRAIN.SEPARATE_BOTTOM_BG,
                )
                val_dataset = FundusOctDataset(
                    cfg.DATA.BASEPATH,
                    "val",
                    transforms=transforms_val,
                    task=cfg.TRAIN.TASK,
                    separate_bottom_bg=cfg.TRAIN.SEPARATE_BOTTOM_BG,
                )
                test_dataset = FundusOctDataset(
                    cfg.DATA.BASEPATH,
                    "test",
                    transforms=transforms_test,
                    separate_bottom_bg=cfg.TRAIN.SEPARATE_BOTTOM_BG,
                )

                if cfg.DATA.TRAIN_DATASET_SUBSET_K > 0:
                    print(f"limiting dataset to {cfg.DATA.TRAIN_DATASET_SUBSET_K} samples")
                    train_dataset = get_limited_dataset(
                        train_dataset, cfg.DATA.TRAIN_DATASET_SUBSET_K
                    )
                if cfg.DATA.OVERSAMPLE_TRAIN_DATASET_FACTOR > 1:
                    print(f"Oversamplign dataset with factor: {cfg.DATA.OVERSAMPLE_TRAIN_DATASET_FACTOR}")
                    train_dataset = OversampledDataset(
                        train_dataset,
                        oversampling_factor=cfg.DATA.OVERSAMPLE_TRAIN_DATASET_FACTOR,
                    )

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
                    shuffle=True,
                    num_workers=cfg.DATA.NUM_WORKERS,
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=cfg.DATA.NUM_WORKERS,
                )
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=cfg.DATA.NUM_WORKERS,
                )

            elif cfg.TRAIN.TASK == "curriculum":
                train_dataset = FundusOctCurDataset(
                    cfg.DATA.BASEPATH,
                    "train",
                    transforms=transforms_train,
                    tasks=["segmentation", "reconstruction"],
                    batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
                )
                val_dataset = FundusOctCurDataset(
                    cfg.DATA.BASEPATH,
                    "val",
                    transforms=transforms_val,
                    tasks=["segmentation", "reconstruction"],
                    batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
                )
                test_dataset = FundusOctCurDataset(
                    cfg.DATA.BASEPATH, "test", transforms=transforms_test
                )

                if cfg.DATA.TRAIN_DATASET_SUBSET_K > 0:
                    train_dataset = get_limited_dataset(
                        train_dataset, cfg.DATA.TRAIN_DATASET_SUBSET_K
                    )

                if cfg.DATA.OVERSAMPLE_TRAIN_DATASET_FACTOR > 1:
                    train_dataset = OversampledDataset(
                        train_dataset,
                        oversampling_factor=cfg.DATA.OVERSAMPLE_TRAIN_DATASET_FACTOR,
                    )

                train_sampler = TaskAwareSampler(
                    train_dataset, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE
                )
                val_sampler = TaskAwareSampler(
                    val_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE
                )
                test_sampler = TaskAwareSampler(
                    test_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE
                )

                # shuffle is ignored with a custom sampler
                train_dataloader = DataLoader(
                    train_dataset,
                    num_workers=cfg.DATA.NUM_WORKERS,
                    batch_sampler=train_sampler,
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    num_workers=cfg.DATA.NUM_WORKERS,
                    batch_sampler=val_sampler,
                )
                test_dataloader = DataLoader(
                    val_dataset,
                    num_workers=cfg.DATA.NUM_WORKERS,
                    batch_sampler=test_sampler,
                )
            else:
                raise NotImplementedError(f"{cfg.TRAIN.TASK} task not yet implemented")

        elif cfg.DATA.DATASET.lower() == "combined":
            train_dataset = CombinedOCTDataset(
                cfg.DATA.BASEPATH,
                "train",
                transforms=transforms_train,
                task=cfg.TRAIN.TASK,
                datasets=[
                    "GOALS",
                    "kermany2018",
                    "neh_ut_2021",
                    "2015_BOE_CHIU",
                    "OCTID",
                ],
                img_size=cfg.DATA.IMG_SIZE,
                max_ds_size=cfg.TRAIN.MAX_EPOCH_LENGTH,
            )
            val_dataset = CombinedOCTDataset(
                cfg.DATA.BASEPATH,
                "val",
                transforms=transforms_val,
                task=cfg.TRAIN.TASK,
                datasets=[
                    "GOALS",
                    "kermany2018",
                    "neh_ut_2021",
                    "2015_BOE_CHIU",
                    "OCTID",
                ],
                img_size=cfg.DATA.IMG_SIZE,
                max_ds_size=int(cfg.TRAIN.MAX_EPOCH_LENGTH / 2),
            )
            test_dataset = CombinedOCTDataset(
                cfg.DATA.BASEPATH,
                "test",
                transforms=transforms_test,
                task=cfg.TRAIN.TASK,
                datasets=[
                    "GOALS",
                    "kermany2018",
                    "neh_ut_2021",
                    "2015_BOE_CHIU",
                    "OCTID",
                ],
                img_size=cfg.DATA.IMG_SIZE,
                max_ds_size=int(cfg.TRAIN.MAX_EPOCH_LENGTH / 2),
            )

            if cfg.DATA.TRAIN_DATASET_SUBSET_K > 0:
                train_dataset = get_limited_dataset(
                    train_dataset, cfg.DATA.TRAIN_DATASET_SUBSET_K
                )

            if cfg.DATA.OVERSAMPLE_TRAIN_DATASET_FACTOR > 1:
                train_dataset = OversampledDataset(
                    train_dataset,
                    oversampling_factor=cfg.DATA.OVERSAMPLE_TRAIN_DATASET_FACTOR,
                )

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.DATA.NUM_WORKERS,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.DATA.NUM_WORKERS,
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=cfg.TRAIN.VAL_BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.DATA.NUM_WORKERS,
            )
        else:
            raise NotImplementedError(f"no dataset: {cfg.DATA.DATASET}")

        # train model
        try:
            trainer.fit(lightning_module, train_dataloader, val_dataloader)
        except KeyboardInterrupt:
            print("\nDetected Control-C. Moving on to testing...")

        # log_dir = trainer.logger.log_dir
        # print(f"saving to: {str(Path(log_dir) / 'model_last.ckpt')}")
        # torch.save(trainer, str(Path(log_dir) / "model_last.ckpt"))

    # load the best model and do eval
    if cfg.EVALUATE.DO_EVAL:
        try:
            trainer.test(lightning_module, test_dataloader, ckpt_path="best")
        except ValueError as e:
            print(f"intercepted error in testing: {e}")

    if cfg.EVALUATE.GENERATE_OUTPUT_SEGMENTATIONS:
        lightning_module = lightning_module.to("cuda:0")
        transforms_val = get_transforms(mode="val")
        test_dataset = FundusOctDataset(
            cfg.DATA.BASEPATH,
            "test",
            transforms=transforms_val,
            task="segmentation",
            separate_bottom_bg=False,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS
        )

        output_path = (
            Path(cfg.DATA.BASEPATH)
            / "Validation"
            / "Layer_Segmentations"
            / "Layer_Segmentations"
        )
        os.makedirs(str(output_path), exist_ok=True)
        lightning_module.save_segmentations(test_dataloader, output_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fundus OCT Challenge")
    parser.add_argument(
        "--cfg", required=True, help="Path to the configuration file", type=str
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    main()
