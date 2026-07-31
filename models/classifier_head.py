import torch.nn as nn

class ClassifierHead(nn.Module):
    """
    Linear classification head to be used on top of the frozen/pretrained encoder.
    """
    def __init__(self, input_dim=512, num_classes=10, hidden_dim=512):
        super(ClassifierHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
