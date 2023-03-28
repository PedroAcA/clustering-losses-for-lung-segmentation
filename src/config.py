#!/usr/bin/env python

#!/usr/bin/env python

from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()

# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 0
_C.SYSTEM.DEVICE = 'cuda:0'
_C.SYSTEM.LOG_FREQ = 10
_C.SYSTEM.EXP_NAME = 'default'

# Dataset setting
_C.DATA = CN()
_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.ROOT = '../datasets/JSRT_dataset/pre_processed_2048_2048/'
_C.DATA.TRAIN.LIST = '../datasets/JSRT_dataset/train_no_labels.lst'
_C.DATA.TEST = CN()
_C.DATA.TEST.ROOT = '../datasets/JSRT_dataset/pre_processed_2048_2048/'
_C.DATA.TEST.LIST = '../datasets/JSRT_dataset/test.lst'
_C.DATA.TEST.GT_ROOT = '../datasets/JSRT_dataset/masks/lungs/'

# Optimizer
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = 'SGD'
_C.SOLVER.FUZZY_FACTOR = 2
_C.SOLVER.REGULARIZER = 0.0016
_C.SOLVER.LOSS = 'RFCM'
_C.SOLVER.EPOCHS = 20
_C.SOLVER.IMG_SIZE = (256, 256)
# Optimizer Settings
_C.SOLVER.BETAS = (0.9, 0.99)
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.USE_POLY_DECAY = True
_C.SOLVER.BATCH_SIZE = 2
_C.SOLVER.ITER_SIZE = 1 # how many batches should be accumulated before updating weights

#Model
_C.MODEL = CN()
_C.MODEL.CLASSES = 3
_C.MODEL.TIME_STEPS = 3

# Miscellaneous
_C.SAVE_ROOT = '../experiments/'


cfg = _C  # users can `from config import cfg`