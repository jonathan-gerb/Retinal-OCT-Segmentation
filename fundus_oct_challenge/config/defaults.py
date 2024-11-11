# default values for the run configs

from yacs.config import CfgNode as CN

_C = CN()

# Dataset configurations
_C.DATA = CN()
_C.DATA.BASEPATH = "./data"
_C.DATA.DATASET = "GOALS"
_C.DATA.NUM_WORKERS = 4
_C.DATA.IMG_SIZE = (800, 1104)
_C.DATA.TRAIN_DATASET_SUBSET_K = -1
_C.DATA.OVERSAMPLE_TRAIN_DATASET_FACTOR = 1
# ... Add more default data configurations as required

# user config
_C.USERNAME = ""

# Model configurations
_C.MODEL = CN()
_C.MODEL.NAME = "deeplab"  # default model
_C.MODEL.PRETRAINED = True
_C.MODEL.RESUME_PATH = ""
_C.MODEL.NUM_CLASSES = 6 # for segmentation
_C.MODEL.NUM_CLASSES_CLASSIFICATION = 4
_C.MODEL.FREEZE_ENCODER = False
# ... Add more default model configurations

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.RUN_NAME = ""
_C.TRAIN.DO_TRAIN = True
_C.TRAIN.MAX_EPOCH_LENGTH = -1 # for some datasets that are huge we would like to limit the size of the training dataset, -1 uses the entire dataset

# can be segmentation/reconstruction/classification/curriculum
_C.TRAIN.TASK = "segmentation" 
_C.TRAIN.EPOCHS = 10
_C.TRAIN.TRAIN_BATCH_SIZE = 2
_C.TRAIN.ACCUMUALTE_GRAD_BATCHES = 1
_C.TRAIN.LR = 0.0001
_C.TRAIN.VAL_BATCH_SIZE = 2
_C.TRAIN.NO_TRANSFORMS = False
_C.TRAIN.NO_SPTRANSFORMS = False
_C.TRAIN.LOG_FREQ_VAL = 25 # log every LOG_FREQ steps
_C.TRAIN.LOG_FREQ_TRAIN = 25
_C.TRAIN.SEPARATE_BOTTOM_BG = True
_C.TRAIN.N_LOG_IMAGES = 1 # how many images from the batch to log, if higher than batch_size is set to batch_size
# ... Add more default training configurations
_C.TRAIN.LR_REDUCER = CN()
_C.TRAIN.LR_REDUCER.PATIENCE = 3
_C.TRAIN.LR_REDUCER.FACTOR = 0.5

# ... Add other configurations like optimizer, evaluator, etc.
_C.EVALUATE = CN()
_C.EVALUATE.DO_EVAL = True
_C.EVALUATE.GENERATE_OUTPUT_SEGMENTATIONS = False
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
