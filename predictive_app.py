import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Data Preprocessing Module
def preprocess_data(file_path):
    # Read data from text file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse lines and convert alphabets to numerical values
    data = []
    for line in lines:
        values = line.strip().split(',')
        numerical_values = [parse_value(val) for val in values]
        data.append(numerical_values)
    print(data)

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(len(data[0]))])
    return df

# Function to parse values, considering alphabets as numerical values
def parse_value(val):
    if val.strip().replace('.', '', 1).isdigit():
        return float(val.strip())  # Convert to float if the string contains only digits and at most one dot
    else:
        # If the value is not numeric, assume it's a symbolic representation
        # You can define your own mapping here
        symbol_mapping = {'a': 205, 'b': 306, 'c': 407}  # Example mapping
        return symbol_mapping.get(val.strip().lower(), val.strip())  # Return mapped value if exists, else return as is

# Model Training Module
def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    
    joblib.dump(model, 'model.pkl')
    return model

# Prediction Module
def make_prediction(input_data):
    model = joblib.load('model.pkl')
    prediction = model.predict([input_data])
    return prediction

# Main Application
if __name__ == "__main__":
    data = preprocess_data('data.txt')  # Ensure 'data.txt' is in the scripts folder
    data['target'] = data['feature_1']  # Assigning a target for demonstration purpose, replace this with your actual target column
    
    model = train_model(data)
    
    # Example input data for prediction
    input_data = [1, 2, 3]  # Replace with actual input features
    prediction = make_prediction(input_data)
    print(f'Prediction: {prediction}')
