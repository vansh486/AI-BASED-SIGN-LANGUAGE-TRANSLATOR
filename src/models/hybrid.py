import torch
import torch.nn as nn
from src.models.stgcn import STGCN
from src.models.transformer import CustomTransformer

class HybridModel(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=128):
        super().__init__()
        self.stgcn = STGCN(in_channels=258, num_classes=hidden_dim)
        self.transformer = CustomTransformer(d_model=hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, Frames, 258]
        x = self.stgcn(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Temporal pooling across frames
        return self.fc(x)