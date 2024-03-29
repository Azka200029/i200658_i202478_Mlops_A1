import unittest
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


class TestWeatherPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weather_pred = pd.read_csv('weather_features.csv')
        cls.weather_pred = cls.weather_pred.sample(n=1000, random_state=42)
        cls.weather_pred['date'] = pd.to_datetime(
            (cls.weather_pred['dt_iso'].str[:10])
        )
        cls.weather_pred['time'] = pd.to_datetime(
            (cls.weather_pred['dt_iso'].str[11:19])
        )
        cls.weather_pred['month'] = cls.weather_pred['date'].dt.month
        cls.weather_pred['year'] = cls.weather_pred['date'].dt.year
        cls.weather_pred['hour'] = cls.weather_pred['time'].dt.hour
        cls.weather_pred['day'] = cls.weather_pred['date'].dt.day
        cls.weather_pred['weekday'] = cls.weather_pred['date'].dt.weekday
        cls.weather_pred['quarter'] = cls.weather_pred['date'].dt.quarter
        cls.weather_pred.drop(
            ['dt_iso', 'city_name', 'weather_description', 'date', 'time'],
            axis=1,
            inplace=True
        )
        cls.le = LabelEncoder()
        cls.weather_pred['weather_main'] = cls.le.fit_transform(
            (cls.weather_pred['weather_main'])
        )
        cls.weather_pred['weather_icon'] = cls.le.fit_transform(
            (cls.weather_pred['weather_icon'])
        )
        cls.z = np.abs(zscore(cls.weather_pred))
        cls.weather_pred_new = cls.weather_pred[(cls.z < 3).all(axis=1)]

    def test_train_test_split(self):
        X = self.weather_pred_new.drop('temp', axis=1)
        y = self.weather_pred_new['temp']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))

    def test_model_performance(self):
        X = self.weather_pred_new.drop('temp', axis=1)
        y = self.weather_pred_new['temp']
        X_train, X_test, y_train, y_test = (
            train_test_split(X, y, test_size=0.2, random_state=42)
        )
        dt = DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        r2 = r2_score(y_test, y_pred_dt)
        mae = mean_absolute_error(y_test, y_pred_dt)
        mse = mean_squared_error(y_test, y_pred_dt)
        rmse = mse ** 0.5
        self.assertTrue(r2 > 0.5, "R2 score should be greater than 0.5")
        self.assertTrue(mae < 2, "Mean Absolute Error should be less than 2")
        self.assertTrue(
            rmse < 3,
            "Root Mean Squared Error should be less than 3"
        )


if __name__ == '__main__':
    unittest.main()
    print("Everything passed")
    print("Congratulations! You have successfully completed this project.")
