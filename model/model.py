import torch
import torch.nn as nn
from torchvision import models

class DefectClassifier(nn.Module):
    """
    Transfer-learning classifier built on a pretrained ResNet18 backbone.
    Freezes the backbone, fine-tunes a new classification head.
    """
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def load_model(checkpoint_path: str, num_classes: int, device: str = "cpu"):
    model = DefectClassifier(num_classes=num_classes, freeze_backbone=True)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state.get("class_names", [])