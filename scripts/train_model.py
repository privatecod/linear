# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime

DATA_FILE = 'data.txt'
MODEL_FILE = 'trained_model.pkl'

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
                letter, value, time_str = parts
                time_obj = datetime.strptime(time_str, '%H:%M:%S')
                hour = time_obj.hour
                minute = time_obj.minute
                second = time_obj.second
                data.append((float(value), hour, minute, second))
            else:
                print(f"Ignoring invalid line: {line}")

    return data

# Load and preprocess data
def load_and_preprocess_data():
    data = preprocess_data(DATA_FILE)
    df = pd.DataFrame(data, columns=['value', 'hour', 'minute', 'second'])

    df['previous_value'] = df['value'].shift(1)
    df = df.dropna()

    X = df[['previous_value', 'hour', 'minute', 'second']]
    y = df['value']
    return X, y

# Train the model with additional analysis
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    return model

if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    model = train_model(X, y)
    joblib.dump(model, MODEL_FILE)
    print("Model training complete and saved to", MODEL_FILE)
