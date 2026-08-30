import torch
import torch.nn as nn
from src.models.stgcn import STGCN

class HybridModel(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=128):
        super().__init__()
        # Set in_channels to 258 to match your input landmark features
        self.stgcn = STGCN(in_channels=258, num_classes=hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, Frames, 258]
        x = self.stgcn(x) if hasattr(self, 'stgcn') else x 
        x = self.transformer(x)
        x = x.mean(dim=1)  # Temporal pooling across frames
        return self.fc(x)