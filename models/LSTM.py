import torch
import torch.nn as nn


class LSTM(nn.Module):
    def init(self, num_classes=..., input_size=..., hidden_size=..., num_layers=..., dropout=0.2):
        super(LSTM, self).init()
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = True

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size * 2 if self.bidirectional else self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)

        return out


# Example usage:
# Define the hyperparameters
input_size = 1024
hidden_size = 256
num_layers = 1
num_classes = 10
dropout = 0.2
batch_size = 32
sequence_length = 100

# Create the advanced LSTM model
model = LSTM(num_classes, input_size, hidden_size, num_layers, dropout)

# Generate random input data for testing
x = torch.randn(batch_size, sequence_length, input_size)

# Forward pass
output = model(x)
print(output.shape)  # Shape: (batch_size, num_classes)