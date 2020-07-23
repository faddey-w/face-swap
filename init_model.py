from reface import utils
from reface.faceshifter.train_AEI import ModelManager


model_dir = ".models/batch-8-2_imsz-128_G_L1e-4_D_L1e-3_layers-3_scales-4"
cfg = utils.load_config_from_yaml_str("""
RANDOM_SEED: 1
INPUT:
    FACE_BOX_EXTENSION_FACTOR: 1.4
    IMAGE_SIZE: 128
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
    NORM_LAYER: instance
    USE_SIGMOID: false
    LEAKY_RELU_SLOPE: 0.2
TRAINING:
    CHECKPOINT_PERIOD: 1000
    CHECKPOINTS_MAX_LAST: 10
    LOG_PERIOD: 50
    VIS_PERIOD: 50
    VIS_MAX_IMAGES: 6
TEST:
    VIS_PERIOD: 100
    VIS_MAX_IMAGES: 4
""")

ModelManager.create_model_dir(cfg, model_dir, strict=True)
