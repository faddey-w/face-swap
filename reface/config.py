from typing import List, Union


class Config:
    RANDOM_SEED: int

    class INPUT:
        FACE_BOX_EXTENSION_FACTOR: float
        IMAGE_SIZE: int

        class TRAIN:
            # this is amount of *pairs*, not individual images per batch
            BATCH_SIZE: int
            SAME_PERSON_PAIRS_PER_BATCH: int

        class LOADER:
            NUM_WORKERS: int

    class MODEL:
        LR_D: float
        LR_G: float
        CHECKPOINT_PERIOD: int
        CHECKPOINTS_MAX_LAST: int

    class LOGGING:
        PERIOD: float
        VIS_PERIOD: int
        VIS_MAX_IMAGES: int

