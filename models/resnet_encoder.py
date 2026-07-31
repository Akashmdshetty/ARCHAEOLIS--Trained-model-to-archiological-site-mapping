import torch
import torch.nn as nn
from torchvision import models

class ResNetEncoder(nn.Module):
    """
    ResNet18 backbone for feature extraction. 
    Uses 'self.model' to match state_dict keys in SSL checkpoints.
    """
    def __init__(self, pretrained=False):
        super(ResNetEncoder, self).__init__()
        # Attribute name MUST be 'model' to match checkpoint keys
        self.model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.embedding_dim = 512

    def forward(self, x):
        # x: (Batch, 3, H, W)
        # Extract intermediate features manually from self.model layers
        x0 = self.model.conv1(x)
        x0 = self.model.bn1(x0)
        x0 = self.model.relu(x0)
        x0 = self.model.maxpool(x0)
        
        x1 = self.model.layer1(x0)    # (Batch, 64, H/4, W/4)
        x2 = self.model.layer2(x1)    # (Batch, 128, H/8, W/8)
        x3 = self.model.layer3(x2)    # (Batch, 256, H/16, W/16)
        x4 = self.model.layer4(x3)    # (Batch, 512, H/32, W/32)
        
        pooled = self.model.avgpool(x4)
        embedding = torch.flatten(pooled, 1)
        
        return [x1, x2, x3, x4], embedding

def get_resnet_encoder(pretrained=False):
    return ResNetEncoder(pretrained=pretrained)
