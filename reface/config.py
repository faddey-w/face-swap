from typing import List, Union


class Config:
    RANDOM_SEED: int

    class INPUT:
        FACE_BOX_EXTENSION_FACTOR: float
        IMAGE_SIZE: int
        N_CHANNELS: int

        class TRAIN:
            # this is amount of *pairs*, not individual images per batch
            BATCH_SIZE: int
            SAME_PERSON_PAIRS_PER_BATCH: int

        class LOADER:
            NUM_WORKERS: int

    class GENERATOR:
        DIMS: list

        LR: float

    class DISCRIMINATOR:
        CONV_SIZE: int
        N_SCALES: int
        N_LAYERS: int
        BASE_DIM: int
        MAX_DIM: int
        NORM_LAYER: str
        USE_SIGMOID: bool
        LEAKY_RELU_SLOPE: float

        LR: float

    class TRAINING:
        CHECKPOINT_PERIOD: int
        CHECKPOINTS_MAX_LAST: int
        LOG_PERIOD: float
        VIS_PERIOD: int
        VIS_MAX_IMAGES: int

