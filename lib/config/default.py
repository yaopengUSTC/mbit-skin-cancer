from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "default"
_C.OUTPUT_DIR = "./output/derm_7pt"
_C.VALID_STEP = 5
_C.SAVE_STEP = 5
_C.SHOW_STEP = 20
_C.PIN_MEMORY = True
_C.INPUT_SIZE = (224, 224)    # (h, w)
_C.COLOR_SPACE = "RGB"
_C.RESUME_MODEL = ""
_C.RESUME_MODE = "all"
_C.CPU_MODE = False
_C.EVAL_MODE = False
_C.GPUS = [0, 1]

# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.DATASET = ""
_C.DATASET.ROOT = ""
_C.DATASET.DATA_TYPE = "jpg"
_C.DATASET.TRAIN_JSON = ""
_C.DATASET.VALID_JSON = ""
_C.DATASET.TEST_JSON = ""
_C.DATASET.CLASS_NAME = ['BCC', 'NV', 'MEL', 'MISC', 'SK']
_C.DATASET.VALID_ADD_ONE_CLASS = False    # for ISIC_2019 valid and test, the number of class is increased from 8 to 9.
_C.DATASET.ADD_CLASS_NAME = "UNK"
_C.DATASET.IMBALANCECIFAR = CN()
_C.DATASET.IMBALANCECIFAR.RATIO = 0.01
_C.DATASET.IMBALANCECIFAR.RANDOM_SEED = 0

# ----- BACKBONE BUILDER -----
_C.BACKBONE = CN()
_C.BACKBONE.TYPE = "RegNetY_800MF"      # refer to lib/backbone/all_models.py
_C.BACKBONE.BBN = False
_C.BACKBONE.FREEZE = False
_C.BACKBONE.PRE_FREEZE = False
_C.BACKBONE.PRE_FREEZE_EPOCH = 5
_C.BACKBONE.PRETRAINED = True
_C.BACKBONE.PRETRAINED_MODEL = ""

# if using drop block, below are drop block parameter
_C.BACKBONE.DROP = CN()
_C.BACKBONE.DROP.BLOCK_PROB = 0.1
_C.BACKBONE.DROP.BLOCK_SIZE = 5
_C.BACKBONE.DROP.NR_STEPS = 50000
# dropout parameter to the last FC layer
_C.BACKBONE.DROP.OUT_PROB = 0.1

# ----- MODULE BUILDER -----
_C.MODULE = CN()
_C.MODULE.TYPE = "GAP"      # "GAP", "Identity"

# ----- CLASSIFIER BUILDER -----
_C.CLASSIFIER = CN()
_C.CLASSIFIER.TYPE = "FC"   # "FC", "FCNorm"
_C.CLASSIFIER.BIAS = True

# ----- LOSS BUILDER -----
_C.LOSS = CN()
_C.LOSS.WEIGHT_POWER = 1.1
_C.LOSS.EXTRA_WEIGHT = [1.0, 1.0, 1.0, 1.0, 1.0]
_C.LOSS.LOSS_TYPE = "CrossEntropy"  # "CrossEntropy", "LDAMLoss", "FocalLoss", "LOWLoss", "GHMCLoss", "CCELoss", "MWNLoss"
_C.LOSS.SCHEDULER = "default"       # "default"--the weights of all classes are "1.0",
                                    # "re_weight"--re-weighting by the power of inverse class frequency at all train stage,
                                    # "drw"--two-stage strategy using re-weighting at the second stage,
                                    # "cls"--cumulative learning strategy to set loss weight.
# For drw scheduler
_C.LOSS.DRW_EPOCH = 50
# For cls scheduler
_C.LOSS.CLS_EPOCH_MIN = 20
_C.LOSS.CLS_EPOCH_MAX = 60

# For LDAMLoss
_C.LOSS.LDAM = CN()
_C.LOSS.LDAM.MAX_MARGIN = 0.5
# For FocalLoss
_C.LOSS.FOCAL = CN()
_C.LOSS.FOCAL.GAMMA = 2.0
_C.LOSS.FOCAL.TYPE = "sigmoid"      # "cross_entropy", "sigmoid", "ldam"
_C.LOSS.FOCAL.SIGMOID = "normal"    # "normal", "enlarge"
# For LOWLoss
_C.LOSS.LOW = CN()
_C.LOSS.LOW.LAMB = 0.01
# For GHMCLoss
_C.LOSS.GHMC = CN()
_C.LOSS.GHMC.BINS = 10
_C.LOSS.GHMC.MOMENTUM = 0.0
# For MWNLoss
_C.LOSS.MWNL = CN()
_C.LOSS.MWNL.GAMMA = 2.0
_C.LOSS.MWNL.BETA = 0.1
_C.LOSS.MWNL.TYPE = "fix"      # "zero", "fix", "decrease"
_C.LOSS.MWNL.SIGMOID = "normal"    # "normal", "enlarge"

# ----- TRAIN BUILDER -----
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32            # for every gpu
_C.TRAIN.MAX_EPOCH = 70
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.TENSORBOARD = CN()
_C.TRAIN.TENSORBOARD.ENABLE = True

# ----- SAMPLER BUILDER -----
_C.TRAIN.SAMPLER = CN()
_C.TRAIN.SAMPLER.TYPE = "default"       # "default", "weighted sampler", "oversample"
_C.TRAIN.SAMPLER.IMAGE_TYPE = "derm"    # "derm", "clinic". For derm_7pt dataset used.

_C.TRAIN.SAMPLER.BORDER_CROP = "pixel"      # "pixel", "ratio"
_C.TRAIN.SAMPLER.BORDER_CROP_PIXEL = 0      # An integer specifying how many pixels to crop at the image border. Useful if images contain a black boundary.
_C.TRAIN.SAMPLER.BORDER_CROP_RATIO = 0.0    # the ratio of edge of the image to be cropped.
_C.TRAIN.SAMPLER.IMAGE_RESIZE = True        # whether the input image needs to be resized to a fix size
_C.TRAIN.SAMPLER.IMAGE_RESIZE_SHORT = 450   # the need size of the short side of the input image
_C.TRAIN.SAMPLER.COLOR_CONSTANCY = False
_C.TRAIN.SAMPLER.CONSTANCY_POWER = 6.0
_C.TRAIN.SAMPLER.CONSTANCY_GAMMA = 0.0

# For Modified RandAugment
_C.TRAIN.SAMPLER.AUGMENT = CN()
_C.TRAIN.SAMPLER.AUGMENT.NEED_AUGMENT = False
_C.TRAIN.SAMPLER.AUGMENT.AUG_METHOD = "v1_0"    # the method of Modified RandAugment ('v0_0' to 'v3_1') or RandAugment ('rand') (refer to: lib/data_transform/modified_randaugment.py)
_C.TRAIN.SAMPLER.AUGMENT.AUG_PROB = 0.7         # the probability parameter 'P' of Modified RandAugment (0.1 -- 0.9)
_C.TRAIN.SAMPLER.AUGMENT.AUG_MAG = 10           # the magnitude parameter 'M' of Modified RandAugment (1 -- 20)
_C.TRAIN.SAMPLER.AUGMENT.AUG_LAYER_NUM = 1      # the number of transformations applied to a training image if AUG_METHOD = 'rand'

# for BBN sampler
_C.TRAIN.SAMPLER.DUAL_SAMPLER = CN()
_C.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE = "reversed"  # "balance", "reverse", "uniform"
# for other sampler
_C.TRAIN.SAMPLER.WEIGHTED_SAMPLER = CN()
_C.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE = "balance"  # "balance", "reverse"

# for multi crop
_C.TRAIN.SAMPLER.MULTI_CROP = CN()
_C.TRAIN.SAMPLER.MULTI_CROP.ENABLE = False      # Should the crops be order or random for evaluation
_C.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM = 16       # Number of crops to use during evaluation (must be N^2)
_C.TRAIN.SAMPLER.MULTI_CROP.L_REGION = 1.0      # Only crop within a certain range of the central area (along the long side of the image)
_C.TRAIN.SAMPLER.MULTI_CROP.S_REGION = 1.0      # Only crop within a certain range of the central area (along the short side of the image)
_C.TRAIN.SAMPLER.MULTI_CROP.SCHEME = 'average'  # Averaging or voting over the crop predictions ("vote", "average")
# for multi transformation of the center crop
_C.TRAIN.SAMPLER.MULTI_SCALE = CN()
_C.TRAIN.SAMPLER.MULTI_SCALE.ENABLE = False     # whether to perform multi transformation on the central crop
_C.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NUM = 12     # Number of scales to use during evaluation (must be less than or equal to the length of SCALE_NAME)
_C.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NAME = ["scale_+00", "flip_x_+00", "rotate_90_+00", "rotate_270_+00",
                                           "scale_+10", "flip_x_+10", "rotate_90_+10", "rotate_270_+10",
                                           "scale_+20", "flip_x_+20", "rotate_90_+20", "rotate_270_+20",
                                           "scale_+30", "flip_x_+30", "rotate_90_+30", "rotate_270_+30",
                                           "scale_-10", "flip_x_-10", "rotate_90_-10", "rotate_270_-10",
                                           "flip_y_+00", "flip_y_+10", "flip_y_-10", "flip_y_+20"]

_C.TRAIN.SAMPLER.FIX_MEAN_VAR = CN()
_C.TRAIN.SAMPLER.FIX_MEAN_VAR.ENABLE = True     # Normalize using the mean and variance of each image, or using fixed values
# A fixed set mean (input image will be subtracted from the mean, processing variance)
_C.TRAIN.SAMPLER.FIX_MEAN_VAR.SET_MEAN = [0.485, 0.456, 0.406]
# A fixed set variance
_C.TRAIN.SAMPLER.FIX_MEAN_VAR.SET_VAR = [0.229, 0.224, 0.225]

# ----- OPTIMIZER -----
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = "SGD"         # 'SGD', 'ADAM', 'NADAM', 'RMSPROP'
_C.TRAIN.OPTIMIZER.BASE_LR = 0.001
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

# ----- LR_SCHEDULER -----
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = "multistep"        # "steplr", "multistep", "cosine", "warmup"
_C.TRAIN.LR_SCHEDULER.LR_LOWER_STEP = 20        # for 'steplr'
_C.TRAIN.LR_SCHEDULER.LR_STEP = [40, 50]        # for 'multistep'
_C.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5            # for 'warmup'
_C.TRAIN.LR_SCHEDULER.COSINE_DECAY_END = 0

# For valid or test
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128         # for every gpu
_C.TEST.NUM_WORKERS = 8
_C.TEST.MODEL_FILE = "best_model.pth"


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


def update_cfg_name(cfg):
    '''
        modify the cfg.NAME
    :param cfg:
    :return:
    '''
    cfg.defrost()
    cfg_name = cfg.DATASET.DATASET + "." + cfg.BACKBONE.TYPE + (
        "_BBN." if cfg.BACKBONE.BBN else ".") + cfg.LOSS.LOSS_TYPE + cfg.NAME
    cfg.merge_from_list(['NAME', cfg_name])
    cfg.freeze()
