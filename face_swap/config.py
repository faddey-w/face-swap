from typing import List, Union, Optional


class Config:
    RANDOM_SEED: int

    class INPUT:
        FACE_BOX_EXTENSION_FACTOR: float
        IMAGE_SIZE: int
        N_CHANNELS: int
        MIN_FACE_SIZE: int

        class TRAIN:
            # this is amount of *pairs*, not individual images per batch
            BATCH_SIZE: int
            SAME_PERSON_PAIRS_PER_BATCH: int

        class LOADER:
            NUM_WORKERS: int

    class GENERATOR:
        DIMS: list
        AAD_NORM: str

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
        CHECKPOINTS_KEEP_PERIOD: int
        OPT_STEP_PERIOD: int
        LOG_PERIOD: float
        VIS_PERIOD: int
        VIS_MAX_IMAGES: int

        INIT_CHECKPOINT: Optional[str]

    class TEST:
        VIS_PERIOD: int
        VIS_MAX_IMAGES: int

        TEST_PERIOD: int
        N_TEST_BATCHES: int


