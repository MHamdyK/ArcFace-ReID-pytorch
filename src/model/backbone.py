from facenet_pytorch import InceptionResnetV1
import torch


def get_pretrained_model(device='cpu'):
    """
    Returns Inception‑ResNet V1 initialised with VGGFace2 weights.
    """
    model = InceptionResnetV1(pretrained='vggface2', classify=False)
    return model.to(device)
