import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.logger import logger
from models import LSTM
from sklearn.utils import shuffle
import sys
import numpy as np
import pickle
import time

# Set random seed for reproducibility
np.random.seed(13696641)
torch.manual_seed(13696641)

class CustomDataLoader():
    def __init__(self, data):
        self.data = np.array(data["features_RGB"])
        self.labels = np.array(data["labels"])
        self.shuffle_data()
    
    def __len__(self):
        return len(self.data)

    def shuffle_data(self):
        self.data, self.labels = shuffle(self.data, self.labels)

    def __getitem__(self, index):
        record = self.data[index]
        label = self.labels[index]

        return record, label

def main():
    path_train = sys.argv[1]
    path_val = sys.argv[2]
    learning_rate = float(sys.argv[3])
    epochs = int(sys.argv[4])

    # Load, unpickle and prepare data
    train_loader, val_loader = load_and_prepare_data(path_train, path_val, batch_size=32)

    # Initialize LSTM model
    model = LSTM()

    # Move model to the specified device (cpu or cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=epochs, device=device)

    # Validate the model
    validate(model, val_loader, device=device)


def load_and_prepare_data(path_train, path_val, batch_size=32):
    # Load the data from the pickle file
    with open(path_train, 'rb') as file:
        train_unpickled = pickle.load(file)

    # Load the data from the pickle file
    with open(path_val, 'rb') as file:
        val_unpickled = pickle.load(file)  

    # Create CustomDataLoader instances for training and validation
    train_data = CustomDataLoader(train_unpickled)
    val_data = CustomDataLoader(val_unpickled)

    # Create DataLoader instances for training and validation
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()  # Set the model to training mode
    start_time = time.time()  # Record the start time of the epoch

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

        end_time = time.time()  # Record the end time of the epoch
        epoch_time = end_time - start_time  # Calculate the time taken for this epoch
        
        accuracy = 100 * correct / total
        logger.info(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy:.2f}%, Epoch Time: {epoch_time:.2f} seconds")
    total_training_time = time.time() - start_time  # Calculate the total training time
    logger.info(f"Total Training Time: {total_training_time:.2f} seconds")

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
        logger.info(f"Class {i} Accuracy: {accuracy_per_class[i]:.2f}%")

    logger.info(f"Overall Validation Accuracy: {overall_accuracy:.2f}%")


if __name__ == '__main__':
    main()
