#! /usr/bin/env python
from reface import utils
from reface.faceshifter.train_AEI import ModelManager

# language=YAML
cfg_text = """
RANDOM_SEED: 1
INPUT:
    FACE_BOX_EXTENSION_FACTOR: 1.4
    IMAGE_SIZE: 256
    MIN_FACE_SIZE: 120
    N_CHANNELS: 3
    TRAIN:
        BATCH_SIZE: 8
        SAME_PERSON_PAIRS_PER_BATCH: 2
    LOADER:
        NUM_WORKERS: 16
GENERATOR:
    DIMS: [32, 64, 128, 256, 512, 1024, 1024]
    # DIMS: [32, 64, 128, 256, 512, 512] #, 1024]
    AAD_NORM: batch
    # LR: 0.0006
    LR: 0.00004
DISCRIMINATOR:
    LR: 0.0004
    CONV_SIZE: 4
    N_SCALES: 3
    N_LAYERS: 6
    BASE_DIM: 64
    MAX_DIM: 512
    NORM_LAYER: batch
    USE_SIGMOID: false
    LEAKY_RELU_SLOPE: 0.1
TRAINING:
    CHECKPOINT_PERIOD: 1000
    CHECKPOINTS_MAX_LAST: 10
    CHECKPOINTS_KEEP_PERIOD: 10000
    OPT_STEP_PERIOD: 1
    LOG_PERIOD: 50
    VIS_PERIOD: 100
    VIS_MAX_IMAGES: 5
    # INIT_CHECKPOINT: 
    INIT_CHECKPOINT: .models/imsz256-60_bs8-2-1_G_L6e-04_full_D_L4e-04_lr6_sc3/model_0096000.ckpt
TEST:
    VIS_PERIOD: 100
    VIS_MAX_IMAGES: 4
    TEST_PERIOD: 1000
    N_TEST_BATCHES: 100
"""
cfg = utils.load_config_from_yaml_str(cfg_text)
model_dir = (
    f".models/"
    f"imsz{cfg.INPUT.IMAGE_SIZE}-{cfg.INPUT.MIN_FACE_SIZE}_"
    f"bs{cfg.INPUT.TRAIN.BATCH_SIZE}-{cfg.INPUT.TRAIN.SAME_PERSON_PAIRS_PER_BATCH}-{cfg.TRAINING.OPT_STEP_PERIOD}_"
    f"G_L{cfg.GENERATOR.LR:.0e}_full_"
    f"D_L{cfg.DISCRIMINATOR.LR:.0e}_"
    f"lr{cfg.DISCRIMINATOR.N_LAYERS}_"
    f"sc{cfg.DISCRIMINATOR.N_SCALES}"
    "_finetune"
)
print(model_dir)
ModelManager.create_model_dir(cfg, model_dir, strict=True)
