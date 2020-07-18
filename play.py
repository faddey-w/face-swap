from reface.config import Config
from reface.utils import load_config_from_dict
from reface.face_recognizer import get_face_recognizer
from reface.data_lib import Dataset, build_data_loader
from PIL import Image
import numpy as np


cfg = load_config_from_dict(
    {
        "RANDOM_SEED": 1,
        "INPUT": {
            "FACE_BOX_EXTENSION_FACTOR": 1.4,
            "IMAGE_SIZE": 256,
            "TRAIN": {"BATCH_SIZE": 2, "SAME_PERSON_PAIRS_PER_BATCH": 1},
            "LOADER": {"NUM_WORKERS": 0},
        },
    }
)

recog = get_face_recognizer()
ds = Dataset("test")
from torchvision import transforms

loader = build_data_loader(ds, cfg, True)
ldr_it = iter(loader)

def cosim(v1, v2):
    return (v1 * v2).sum() / (v1.norm() * v2.norm())


def test_sim(use_transf=False):
    if use_transf:
        v11 = recog(transf(im11))
        v12 = recog(transf(im12))
        v21 = recog(transf(im21))
        v31 = recog(transf(im31))
    else:
        v11 = recog(im11)
        v12 = recog(im12)
        v21 = recog(im21)
        v31 = recog(im31)
    return (
        cosim(v11, v12),
        cosim(v21, v31),
        cosim(v11, v21),
        cosim(v11, v31),
        cosim(v12, v21),
        cosim(v12, v31),
    )


b1 = next(ldr_it)

print(b1[0][0]["id"], b1[0][1]["id"], b1[1][0]["id"], b1[1][1]["id"])

transf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
im11 = b1[0][0]["image"].unsqueeze(0)
im12 = b1[0][1]["image"].unsqueeze(0)
im21 = b1[1][0]["image"].unsqueeze(0)
im31 = b1[1][1]["image"].unsqueeze(0)
