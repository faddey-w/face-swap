from reface import utils
model_dir = ".models/try0"
cfg = utils.load_config_from_yaml_str("""
RANDOM_SEED: 1
INPUT:
    FACE_BOX_EXTENSION_FACTOR: 1.4
    IMAGE_SIZE: 128
    N_CHANNELS: 3
    TRAIN:
        BATCH_SIZE: 2
        SAME_PERSON_PAIRS_PER_BATCH: 1
    LOADER:
        NUM_WORKERS: 0
GENERATOR:
    DIMS: [32, 64, 128, 256, 512]#, 1024, 1024]
    LR: 0.0001
DISCRIMINATOR:
    CONV_SIZE: 4
    N_SCALES: 3
    N_LAYERS: 6
    BASE_DIM: 64
    MAX_DIM: 512
    NORM_LAYER: instance
    USE_SIGMOID: false
    LEAKY_RELU_SLOPE: 0.2
    LR: 0.0004
TRAINING:
    CHECKPOINT_PERIOD: 1000
    CHECKPOINTS_MAX_LAST: 5
    LOG_PERIOD: 1
    VIS_PERIOD: 1
    VIS_MAX_IMAGES: 4
""")


from reface.faceshifter.train_AEI import ModelManager, Trainer
from reface.data_lib import Dataset
ModelManager.create_model_dir(cfg, model_dir, strict=False)
mmgr = ModelManager(model_dir)
ds = Dataset("train")
trainer = Trainer(mmgr, ds)

def runtrain():
    import cv2

    for event in trainer.train():
        if event.is_metrics:
            print(event.message)
        if event.is_visualization:
            # import pdb; pdb.set_trace()
            cv2.imshow("vis", event.image[:, :, ::-1])
            cv2.waitKey(100)
            # print(event.image.shape, event.image.dtype)
            # trainer.stop_train()
runtrain()
