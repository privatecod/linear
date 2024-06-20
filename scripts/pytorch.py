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
                letter, value, time_str = parts
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
        # Cap the predictions between 1.00 and 3000.00
        if pred < 1.00:
            pred = 1.00
        elif pred > 3000.00:
            pred = 3000.00
        processed_predictions.append(round(pred, 3))
    return processed_predictions

# Main Application (example usage)
if __name__ == "__main__":
    # Load and preprocess data
    data = preprocess_data('data.txt')  # Ensure 'data.txt' is in the scripts folder
    
    # Convert data to DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=['value', 'hour', 'minute'])

    # Prepare features and targets
    df['previous_value'] = df['value'].shift(1)  # Create a lag feature
    df = df.dropna()  # Drop the first row which will have a NaN in 'previous_value'
    
    X = df[['previous_value', 'hour', 'minute']]
    y = df['value']
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model, mean_mse = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate the model on the validation data
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    val_predictions = model(X_val_tensor).detach().numpy().flatten()
    val_predictions = process_predictions(val_predictions)  # Process predictions to ensure they are within the desired range
    print(f'Mean Squared Error: {mean_mse:.3f}')  # Round MSE to 3 decimal places
    
    # Classify predictions as 'high' or 'low'
    classified_predictions = [classify_flight(pred) for pred in val_predictions]
    print(f'Predictions: {classified_predictions}')
    
    # Save the model
    torch.save(model.state_dict(), 'model.pth')
