from torch import nn
from reface import env
from facenet_pytorch import InceptionResnetV1


class FaceRecognizer:
    embedding_dimension = 512
    best_input_size = 112, 112

    def __init__(self):
        self.recognizer = InceptionResnetV1(pretrained="vggface2", device=env.device)
        self.recognizer.eval()

    def __call__(self, image):
        size = tuple(image.shape[1:3])
        if size != self.best_input_size:
            image = nn.functional.interpolate(
                image, self.best_input_size, mode="bilinear", align_corners=True
            )
        return self.recognizer(image)
