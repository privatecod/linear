import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import joblib
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

# Train Linear Regression Model with Cross-Validation
def train_model(X, y):
    model = LinearRegression()
    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    model.fit(X_poly, y)
    return model, mean_mse

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

# Main Application
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
    
    # Train the model
    model, mean_mse = train_model(X, y)
    
    # Evaluate the model on the same training data (for simplicity, ideally use a separate test set)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    predictions = model.predict(X_poly)
    predictions = process_predictions(predictions)  # Process predictions to ensure they are within the desired range
    print(f'Mean Squared Error: {mean_mse:.3f}')  # Round MSE to 3 decimal places
    
    # Classify predictions as 'high' or 'low'
    classified_predictions = [classify_flight(pred) for pred in predictions]
    print(f'Predictions: {classified_predictions}')
    
    # Save the model
    joblib.dump(model, 'model.pkl')
