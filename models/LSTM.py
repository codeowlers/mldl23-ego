import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_classes=8,  input_size= 1024, hidden_size=256, num_layers=5, dropout=0.2):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)

        return out