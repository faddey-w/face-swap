from reface.env import device
from facenet_pytorch import InceptionResnetV1


def get_face_recognizer():
    return InceptionResnetV1(pretrained="vggface2", device=device).eval()
