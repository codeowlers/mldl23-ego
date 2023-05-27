import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, num_classes=..., input_size=..., hidden_size=..., num_layers=...):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fc1 = nn.Linear(self.hidden_size, ...)
        self.fc2 = nn.Linear(..., self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # hidden state
        out, h_n = self.gru(x, h_0)  # gru with input and hidden state
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
