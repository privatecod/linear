import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

DATA_FILE = 'data.txt'
MODEL_FILE = 'trained_model_rf.pkl'

def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) == 2:  # Assuming each line contains value and time (HH:MM)
                value, time_str = parts
                time_obj = datetime.strptime(time_str, '%H:%M')
                hour = time_obj.hour
                minute = time_obj.minute
                data.append((float(value), hour, minute))
            else:
                print(f"Ignoring invalid line: {line}")

    return data

def load_and_preprocess_data():
    data = preprocess_data(DATA_FILE)
    df = pd.DataFrame(data, columns=['value', 'hour', 'minute'])
    df['previous_value'] = df['value'].shift(1)
    df = df.dropna()
    X = df[['previous_value', 'hour', 'minute']]
    y = df['value']
    return X, y

def train_random_forest(X, y):
    return RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

def evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    X, y = load_and_preprocess_data()
    model = train_random_forest(X, y)
    joblib.dump(model, MODEL_FILE)
except Exception as e:
    print(f"Error loading model: {e}")

def predict_value(last_value, hour, minute):
    prediction = model.predict([[last_value, hour, minute]])[0]
    return prediction
