# default values for the run configs

from yacs.config import CfgNode as CN

_C = CN()

# Dataset configurations
_C.DATA = CN()
_C.DATA.BASEPATH = "./data"
_C.DATA.NUM_WORKERS = 4
_C.DATA.NAME = "fundus"
# ... Add more default data configurations as required

# Model configurations
_C.MODEL = CN()
_C.MODEL.NAME = "deeplab"  # default model
_C.MODEL.PRETRAINED = True
_C.MODEL.NUM_CLASSES = 6 # for segmentation
_C.MODEL.NUM_CLASSES_CLASSIFICATION = 4
# ... Add more default model configurations

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.DO_TRAIN = True
# can be segmentation/reconstruction/classification/curriculum
_C.TRAIN.TASK = "segmentation" 
_C.TRAIN.EPOCHS = 10
_C.TRAIN.TRAIN_BATCH_SIZE = 2
_C.TRAIN.LR = 0.0001
_C.TRAIN.VAL_BATCH_SIZE = 2
_C.TRAIN.LOG_FREQ = 25 # log every LOG_FREQ steps
_C.TRAIN.N_LOG_IMAGES = 1 # how many images from the batch to log, if higher than batch_size is set to batch_size
# ... Add more default training configurations

# ... Add other configurations like optimizer, evaluator, etc.
_C.EVALUATE = CN()
_C.EVALUATE.DO_EVAL = True
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
