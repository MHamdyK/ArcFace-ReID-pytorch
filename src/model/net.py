import torch.nn as nn
import torch.nn.functional as F
from .arc_margin import ArcMarginProduct


class FaceReIDModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.arcface = ArcMarginProduct(512, num_classes)

    def forward(self, x, labels=None):
        embeddings = F.normalize(self.backbone(x))
        if labels is not None:
            return self.arcface(embeddings, labels)
        return embeddings
