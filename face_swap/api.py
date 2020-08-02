import numpy as np
import torch
from PIL import Image
from face_swap import env
from face_swap.data_lib import InputLayer
from face_swap.face_recognizer import FaceRecognizer
from face_swap.faceshifter.train_AEI import ModelManager


class FaceSwapper:
    def __init__(self, model_dir):
        self.model_manager = ModelManager(model_dir)
        self.generator, discriminator = self.model_manager.build_model()
        del discriminator
        self.model_manager.load_from_checkpoint(self.generator)
        self.generator.eval()
        self.face_recognizer = FaceRecognizer()
        self.input_layer = InputLayer(self.model_manager.cfg, env.device)

    def __call__(self, image_source, image_target):
        # TODO introduce also face detector, so it will do all the job together:
        #  find faces, swap them, and insert the result into the full target image
        image_source = _load_image(image_source)
        image_target = _load_image(image_target)

        with torch.no_grad():
            image_src = self.input_layer(image_source).unsqueeze(0)
            image_tgt = self.input_layer(image_target).unsqueeze(0)

            face_embed = self.face_recognizer(image_src)
            image_result, target_attrs = self.generator(image_tgt, face_embed)

            image_result = image_result.cpu().numpy()[0]
        image_result = (1 + image_result) * 255 / 2
        image_result = image_result.astype("uint8")
        image_result = np.transpose(image_result, [1, 2, 0])

        return image_result


def _load_image(image):
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    return image
