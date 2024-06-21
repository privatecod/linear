import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Data Preprocessing Module
def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace and newlines
        if line:  # Skip empty lines
            parts = line.split()  # Split by whitespace
            if len(parts) == 3:
                value, time_str = parts[1:]
                time_obj = datetime.strptime(time_str, '%H:%M')  # Convert time to datetime object
                hour = time_obj.hour
                minute = time_obj.minute
                data.append((float(value), hour, minute))  # Convert value to float and add time features
            else:
                print(f"Ignoring invalid line: {line}")

    return data

# Function to identify if a number is high or low
def classify_flight(number):
    if number > 5:
        return "high"
    else:
        return "low"

# Define PyTorch model
class SimpleRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Train PyTorch model
def train_model(X_train, y_train, X_val, y_val):
    input_dim = X_train.shape[1]
    model = SimpleRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_outputs = model(X_val_tensor)
    val_loss = criterion(val_outputs, y_val_tensor).item()
    
    return model, val_loss

# Function to cap and round predictions
def process_predictions(predictions):
    processed_predictions = []
    for pred in predictions:
        # Cap the predictions between 1.00 and 9000.00
        if pred < 1.00:
            pred = 1.00
        elif pred > 9000.00:
            pred = 9000.00
        processed_predictions.append(round(pred, 3))
    return processed_predictions
