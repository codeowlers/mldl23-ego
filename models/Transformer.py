import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, num_classes=..., input_size=..., hidden_size=..., num_layers=..., num_heads=..., dropout=0.1):
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Transformer Encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size,
                dropout=self.dropout
            ),
            num_layers=self.num_layers
        )
        
        # Fully-connected layer
        self.fc = nn.Linear(self.input_size, self.num_classes)
        
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.fc(x)
        return x

