from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime, timedelta
import random

app = Flask(__name__)

DATA_FILE = 'data.txt'
MODEL_FILE = 'trained_model.pkl'

# Declare model as global at the top level
model = None

# Data Preprocessing Module
def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) == 3:
                value, time_str = parts[1], parts[2]
                time_obj = datetime.strptime(time_str, '%H:%M')
                hour = time_obj.hour
                minute = time_obj.minute
                data.append((float(value), hour, minute))
            else:
                print(f"Ignoring invalid line: {line}")

    return data

# Train Linear Regression Model
def train_model(X, y):
    global model  # Declare model as global here
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

# Function to cap and round predictions
def process_predictions(predictions):
    processed_predictions = []
    for pred in predictions:
        if pred < 1.00:
            pred = 1.00
        elif pred > 2000000.00:
            pred = 2000000.00
        processed_predictions.append(round(pred, 3))
    return processed_predictions

# Load and preprocess data
def load_and_preprocess_data():
    data = preprocess_data(DATA_FILE)
    df = pd.DataFrame(data, columns=['value', 'hour', 'minute'])

    df['previous_value'] = df['value'].shift(1)
    df = df.dropna()

    X = df[['previous_value', 'hour', 'minute']]
    y = df['value']
    return X, y

X, y = load_and_preprocess_data()

# Try loading the model
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    model = train_model(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model  # Declare model as global here
        data = request.get_json(force=True)
        previous_values = data['previous_values']
        start_hour = data['hour']
        start_minute = data['minute']
        
        predictions = []
        probabilities = []

        time = datetime(2000, 1, 1, start_hour, start_minute)

        all_data = preprocess_data(DATA_FILE)
        df = pd.DataFrame(all_data, columns=['value', 'hour', 'minute'])
        df['previous_value'] = df['value'].shift(1)
        df = df.dropna()

        for previous_value in previous_values:
            hour = time.hour
            minute = time.minute
            X_new = pd.concat([df[['previous_value', 'hour', 'minute']], pd.DataFrame([[previous_value, hour, minute]], columns=['previous_value', 'hour', 'minute'])], ignore_index=True)
            y_new = pd.concat([df['value'], pd.Series([previous_value])], ignore_index=True)

            model.fit(X_new, y_new)
            prediction = model.predict([[previous_value, hour, minute]])
            processed_prediction = process_predictions(prediction)
            predictions.append(processed_prediction[0])

            # Generating random probabilities for the predictions
            probability = random.uniform(0, 1)
            probabilities.append(probability)

            time += timedelta(minutes=1)

        result = []

        for pred, prob in zip(predictions, probabilities):
            result.append({'predicted_value': pred, 'probability': round(prob*100, 2), 'time': time.strftime('%H:%M')})

        return jsonify({'predictions': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)



