from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


def preprocess_data(weather_pred):
    weather_pred['date'] = pd.to_datetime(weather_pred['dt_iso'].str[:10])
    weather_pred['time'] = pd.to_datetime(weather_pred['dt_iso'].str[11:19])
    weather_pred['month'] = weather_pred['date'].dt.month
    weather_pred['year'] = weather_pred['date'].dt.year
    weather_pred['hour'] = weather_pred['time'].dt.hour
    weather_pred['day'] = weather_pred['date'].dt.day
    weather_pred['weekday'] = weather_pred['date'].dt.weekday
    weather_pred['quarter'] = weather_pred['date'].dt.quarter

    weather_pred.drop(
        ['dt_iso', 'city_name', 'weather_description', 'date', 'time'],
        axis=1,
        inplace=True
    )

    le = LabelEncoder()
    weather_pred['weather_main'] = (
        le.fit_transform(weather_pred['weather_main'])
    )
    weather_pred['weather_icon'] = (
        le.fit_transform(weather_pred['weather_icon'])
    )

    z = np.abs(zscore(weather_pred))
    weather_pred_new = weather_pred[(z < 3).all(axis=1)]

    return weather_pred_new


def train_model(X, y):
    dt = DecisionTreeRegressor()
    dt.fit(X, y)
    return dt


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    weather_data = pd.DataFrame(data['weather_data'])
    weather_data = preprocess_data(weather_data)
    X = weather_data.drop('temp', axis=1)
    y = weather_data['temp']
    X_train, y_train = (
            train_test_split(X, y, test_size=0.2, random_state=42)
        )
    model = train_model(X_train, y_train)
    prediction = model.predict(X)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
