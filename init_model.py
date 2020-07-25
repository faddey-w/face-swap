#! /usr/bin/env python
from reface import utils
from reface.faceshifter.train_AEI import ModelManager

# language=YAML
cfg_text = """
RANDOM_SEED: 1
INPUT:
    FACE_BOX_EXTENSION_FACTOR: 1.4
    IMAGE_SIZE: 64
    MIN_FACE_SIZE: 60
    N_CHANNELS: 3
    TRAIN:
        BATCH_SIZE: 8
        SAME_PERSON_PAIRS_PER_BATCH: 2
    LOADER:
        NUM_WORKERS: 16
GENERATOR:
    DIMS: [32, 64, 128, 256, 512]#, 1024, 1024]
    LR: 0.0001
DISCRIMINATOR:
    LR: 0.001
    CONV_SIZE: 4
    N_SCALES: 4
    N_LAYERS: 3
    BASE_DIM: 64
    MAX_DIM: 512
    NORM_LAYER: AdaIN
    USE_SIGMOID: false
    LEAKY_RELU_SLOPE: 0.2
TRAINING:
    CHECKPOINT_PERIOD: 1000
    CHECKPOINTS_MAX_LAST: 10
    LOG_PERIOD: 50
    VIS_PERIOD: 100
    VIS_MAX_IMAGES: 6
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
    f"bs{cfg.INPUT.TRAIN.BATCH_SIZE}-{cfg.INPUT.TRAIN.SAME_PERSON_PAIRS_PER_BATCH}_"
    f"G_L{cfg.GENERATOR.LR:.0e}_"
    f"D_L{cfg.DISCRIMINATOR.LR:.0e}_"
    f"lr{cfg.DISCRIMINATOR.N_LAYERS}_"
    f"sc{cfg.DISCRIMINATOR.N_SCALES}"
)
print(model_dir)
ModelManager.create_model_dir(cfg, model_dir, strict=True)
