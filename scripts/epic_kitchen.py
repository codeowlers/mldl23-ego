import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models import LSTM
import sys
import pickle
import numpy as np
from utils.utils import load_features_labels

def main():
    if len(sys.argv) != 7:
        print("Usage: python train_lstm.py <path_train> <path_test> <learning_rate> <momentum> <epochs>")
        sys.exit(1)

    path_train = sys.argv[1]
    path_test = sys.argv[2]
    learning_rate = float(sys.argv[3])
    momentum = float(sys.argv[4])
    epochs = int(sys.argv[5])
    model_description = sys.argv[6]
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # importing datasets #

    # Load and preprocess your dataset
    train_loader, test_loader = load_data(path_train, path_test)

    # Initialize your LSTM model

    # MODEL initialization
    if model_description == "LSTM":
        model = LSTM()
    else:
        raise NotImplementedError("This model is not implemented")

    # Move model to the specified device (cpu or cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=epochs, device=device)

    # Validate the model
    validate(model, test_loader, device=device)


def load_data(path_train, path_test):

    # Load your dataset
    train_dataset = load_features_labels(path_train)
    test_dataset = load_features_labels(path_test)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradient buffers
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy:.2f}%")

def validate(model, test_loader, device, num_classes=8):
    model.eval()  # Set the model to evaluation mode
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(num_classes):
                class_labels = (labels == i)
                class_predicted = (predicted == i)
                correct_per_class[i] += torch.sum(class_labels & class_predicted).item()
                total_per_class[i] += torch.sum(class_labels).item()

    accuracy_per_class = [100 * correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0.0 for i in range(num_classes)]
    overall_accuracy = 100 * sum(correct_per_class) / sum(total_per_class) if sum(total_per_class) > 0 else 0.0

    for i in range(num_classes):
        print(f"Class {i} Accuracy: {accuracy_per_class[i]:.2f}%")

    print(f"Overall Validation Accuracy: {overall_accuracy:.2f}%")


if __name__ == '__main__':
    main()
