from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime, timedelta
from model2 import preprocess_data, load_and_preprocess_data, train_random_forest, evaluate_model, predict_value

app = Flask(__name__)

DATA_FILE = 'data.txt'
MODEL_FILE = 'trained_model_rf.pkl'

model = None

def calculate_probability(value, mean, std):
    if std == 0:
        return 0  # Avoid division by zero
    z_score = (value - mean) / std
    probability = (1.0 - np.abs(z_score) / 3) * 100  # Simple linear scaling
    probability = max(0, min(100, probability))  # Clamp to [0, 100]
    print(f"Value: {value}, Mean: {mean}, Std: {std}, Z-score: {z_score}, Probability: {probability}")
    return probability

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    X, y = load_and_preprocess_data()
    model = train_random_forest(X, y)
    joblib.dump(model, MODEL_FILE)
except Exception as e:
    print(f"Error loading model: {e}")

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
                file.write(f'{previous_value} {hour:02}:{minute:02}\n')

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

        predictions = []
        for i in range(3):  # Generate 3 predictions
            time += timedelta(minutes=1)
            prediction = predict_value(last_value, time.hour, time.minute)
            probability = calculate_probability(prediction, mean, std)
            predictions.append({
                'previous_value': last_value,
                'predicted_value': round(prediction, 3),
                'probability': round(probability, 2),
                'time': time.strftime('%H:%M:%S')
            })

        if not predictions:
            return jsonify({'message': 'No predictions available.'})

        return jsonify({'predictions': predictions})
    except Exception as e:
        print(f"Error during prediction request: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
