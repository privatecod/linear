from flask import Flask, render_template, request, jsonify
import joblib
from werkzeug.urls import url_quote

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
from pytorch_module import preprocess_data, train_model, process_predictions, SimpleRegressionModel, classify_flight

app = Flask(__name__)

DATA_FILE = 'data.txt'
MODEL_FILE = 'trained_model_rf.pkl'
PYTORCH_MODEL_FILE = 'model.pth'

model_rf = None
model_pytorch = None

# Load Random Forest model
try:
    model_rf = joblib.load(MODEL_FILE)
except FileNotFoundError:
    # Handle the case where the model file is not found
    pass
except Exception as e:
    print(f"Error loading Random Forest model: {e}")

# Load PyTorch model
try:
    model_pytorch = SimpleRegressionModel(input_dim=3)  # Assuming input_dim matches your features
    model_pytorch.load_state_dict(torch.load(PYTORCH_MODEL_FILE))
    model_pytorch.eval()
except FileNotFoundError:
    # Handle the case where the PyTorch model file is not found
    pass
except Exception as e:
    print(f"Error loading PyTorch model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save-data', methods=['POST'])
def save_data():
    try:
        data = request.get_json(force=True)
        if not all(key in data for key in ['previous_values', 'hour', 'minute']):
            return jsonify({'error': 'Invalid request data. Ensure all required fields are provided.'}), 400
        
        previous_values = data['previous_values']
        hour = data['hour']
        minute = data['minute']

        with open(DATA_FILE, 'a') as file:
            for previous_value in previous_values:
                # Encode each previous_value before saving
                encoded_previous_value = url_quote(previous_value)
                file.write(f'{encoded_previous_value} {hour:02}:{minute:02}\n')

        return jsonify({'message': 'Data saved successfully'})
    except Exception as e:
        print(f"Error saving data: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not all(key in data for key in ['previous_values', 'hour', 'minute']):
            return jsonify({'error': 'Invalid request data. Ensure all required fields are provided.'}), 400
        
        previous_values = data['previous_values']
        start_hour = data['hour']
        start_minute = data['minute']

        # Load all historical values
        with open(DATA_FILE, 'r') as file:
            lines = file.readlines()
        historical_values = [float(line.split()[0]) for line in lines]

        # Add the new data to historical values
        new_data = [(value, start_hour, start_minute) for value in previous_values]
        historical_values.extend([value for value, _, _ in new_data])

        mean = np.mean(historical_values)
        std = np.std(historical_values)

        # Prepare data for prediction
        time = datetime(2000, 1, 1, start_hour, start_minute)
        last_value = previous_values[-1]

        # Predict using Random Forest model
        rf_prediction = model_rf.predict([[last_value, start_hour, start_minute]])
        
        # Predict using PyTorch model
        pytorch_prediction = None
        if model_pytorch is not None:
            pytorch_prediction_tensor = torch.tensor([[last_value, start_hour, start_minute]], dtype=torch.float32)
            pytorch_prediction = model_pytorch(pytorch_prediction_tensor).item()
            pytorch_prediction = process_predictions([pytorch_prediction])[0]  # Process PyTorch prediction

        # Calculate probability for RF prediction
        rf_probability = calculate_probability(rf_prediction, mean, std)
        
        # Prepare response
        predictions = {
            'rf_prediction': round(rf_prediction[0], 3),
            'rf_probability': round(rf_probability, 2),
            'pytorch_prediction': round(pytorch_prediction, 3) if pytorch_prediction is not None else None
        }

        return jsonify({'predictions': predictions})
    
    except Exception as e:
        print(f"Error during prediction request: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

def calculate_probability(prediction, mean, std):
    # Define your method to calculate probability
    return np.exp(-((prediction - mean) ** 2) / (2 * (std ** 2))) / (std * np.sqrt(2 * np.pi))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
