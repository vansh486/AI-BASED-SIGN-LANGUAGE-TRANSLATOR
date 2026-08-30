import torch
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, in_channels=258, num_classes=128):
        super().__init__()
        self.spatial_fc = nn.Linear(in_channels, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [Batch, Frames, 258]
        return self.relu(self.spatial_fc(x))