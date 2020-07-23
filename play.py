from reface import utils
from reface.data_lib import Dataset
from reface.faceshifter.train_AEI import ModelManager, Trainer


model_dir = ".models/try2"
cfg = utils.load_config_from_yaml_str("""
RANDOM_SEED: 1
INPUT:
    FACE_BOX_EXTENSION_FACTOR: 1.4
    IMAGE_SIZE: 128
    N_CHANNELS: 3
    TRAIN:
        BATCH_SIZE: 32
        SAME_PERSON_PAIRS_PER_BATCH: 8
    LOADER:
        NUM_WORKERS: 16
GENERATOR:
    DIMS: [32, 64, 128, 256, 512]#, 1024, 1024]
    LR: 0.0004
DISCRIMINATOR:
    LR: 0.00004
    CONV_SIZE: 4
    N_SCALES: 3
    N_LAYERS: 6
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
mmgr = ModelManager(model_dir)
ds_train = Dataset("train")
ds_test = Dataset("test")
trainer = Trainer(mmgr, ds_train, ds_test)

trainer.train()
