import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
import numpy as np
import pickle
import unittest


def preprocess_data(weather_pred):
    # Convert 'date' column to datetime
    weather_pred['date'] = 0
    weather_pred['date'] = pd.to_datetime(weather_pred['date'])
    # Preprocessing steps
    weather_pred['month'] = weather_pred['date'].dt.month
    weather_pred['year'] = weather_pred['date'].dt.year
    weather_pred['hour'] = weather_pred['date'].dt.hour
    weather_pred['day'] = weather_pred['date'].dt.day
    weather_pred['weekday'] = weather_pred['date'].dt.weekday
    weather_pred['quarter'] = weather_pred['date'].dt.quarter
    # Drop unnecessary columns
    weather_pred.drop(['dt_iso', 'city_name', 'weather_description'],
                      axis=1, inplace=True)
    # Label encoding
    le = LabelEncoder()
    weather_pred['weather_main'] = le.fit_transform(weather_pred['weather_main'])
    weather_pred['weather_icon'] = le.fit_transform(weather_pred['weather_icon'])
    # Remove outliers
    numeric_columns = weather_pred.select_dtypes(include=[np.number]).columns
    weather_pred_numeric = weather_pred[numeric_columns]

    # Remove outliers using z-score normalization
    z = np.abs(zscore(weather_pred_numeric))
    weather_pred_new = weather_pred[(z < 3).all(axis=1)]
    return weather_pred_new


def train_model(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model


def main():
    # Read data
    weather_pred = pd.read_csv('weather_features.csv')
    weather_pred_sample = weather_pred.sample(n=1000, random_state=42)
    # Preprocess data
    weather_pred_new = preprocess_data(weather_pred_sample)
    # Split data into features and target
    X = weather_pred_new.drop('temp', axis=1)
    y = weather_pred_new['temp']
    # Calculate test_size based on the proportion of the dataset
    test_size = min(0.2, len(weather_pred_new) / len(weather_pred_sample))
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )
    # Train model
    model = train_model(X_train, y_train)
    # Evaluate model
    r2 = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # Print results
    print('Random Forest Regression Results:')
    print('R2 score:', r2)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)


class TestWeatherPrediction(unittest.TestCase):
    def setUp(self):
        # Load data
        self.weather_pred = pd.read_csv('weather_features.csv')
        # Preprocess data
        self.weather_pred_sample = (
            self.weather_pred.sample(n=1000, random_state=42)
        )
        self.weather_pred_new = preprocess_data(self.weather_pred_sample)
        # Split data
        self.X = self.weather_pred_new.drop('temp', axis=1)
        self.y = self.weather_pred_new['temp']
        # Fix the line below to remove the line break
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        )
        # Train model
        self.model = train_model(self.X_train, self.y_train)

    def test_model_performance(self):
        # Evaluate model
        r2 = self.model.score(self.X_test, self.y_test)
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)

        self.assertTrue(r2 > 0.5,
                        "R2 score should be greater than 0.5")
        self.assertTrue(mae < 2,
                        "Mean Absolute Error should be less than 2")
        self.assertTrue(rmse < 3,
                        "Root Mean Squared Error should be less than 3")

    def test_model_pickle(self):
        # Serialize model
        filename = 'weather_model.pkl'
        pickle.dump(self.model, open(filename, 'wb'))
        # Load model
        loaded_model = pickle.load(open(filename, 'rb'))
        self.assertIsNotNone(loaded_model, "Model should be loaded success")


if __name__ == '__main__':
    print('Executing main function...')
    main()
    print('Execution completed.')
    print('Done')
    print('Done')
    unittest.main()
    print('Done')
    print('Done')
    print('Done')
