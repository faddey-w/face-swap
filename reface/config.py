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

