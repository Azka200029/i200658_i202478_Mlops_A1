import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV


def preprocess_data(weather_pred):
    # Preprocessing steps
    weather_pred['date'] = pd.to_datetime(weather_pred['dt_iso'].str[:10])
    weather_pred['time'] = pd.to_datetime(weather_pred['dt_iso'].str[11:19])
    weather_pred['month'] = weather_pred['date'].dt.month
    weather_pred['year'] = weather_pred['date'].dt.year
    weather_pred['hour'] = weather_pred['time'].dt.hour
    weather_pred['day'] = weather_pred['date'].dt.day
    weather_pred['weekday'] = weather_pred['date'].dt.weekday
    weather_pred['quarter'] = weather_pred['date'].dt.quarter
    weather_pred.drop(['dt_iso', 'city_name', 'weather_description',
                       'date', 'time'], axis=1, inplace=True)
    # Label encoding
    le = LabelEncoder()
    weather_pred['weather_main'] = le.fit_transform
    (weather_pred['weather_main'])
    weather_pred['weather_icon'] = le.fit_transform
    (weather_pred['weather_icon'])
    # Remove outliers
    z = np.abs(zscore(weather_pred))
    weather_pred_new = weather_pred[(z < 3).all(axis=1)]
    return weather_pred_new


def train_model(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return r2, mae, mse, rmse


def optimize_decision_tree(X_train, y_train):
    dt_op = DecisionTreeRegressor()
    # Define the hyperparameter grid
    parameters = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'criterion': ['mse', 'friedman_mse', 'mae']
    }
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=dt_op, param_grid=parameters, cv=5,
                               n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    # Get the best parameters and best model
    best_model = grid_search.best_estimator_
    return best_model


def main():
    # Read data
    weather_pred = pd.read_csv('weather_features.csv')
    weather_pred_sample = weather_pred.sample(n=1000, random_state=42)
    # Preprocess data
    weather_pred_new = preprocess_data(weather_pred_sample)
    # Split data into features and target
    X = weather_pred_new.drop('temp', axis=1)
    y = weather_pred_new['temp']
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    # Train model
    model = train_model(X_train, y_train)
    # Evaluate model
    r2, mae, mse, rmse = evaluate_model(model, X_test, y_test)
    # Optimize decision tree
    best_model = optimize_decision_tree(X_train, y_train)
    # Serialize model
    filename_dt = 'weather_pred_dt.pkl'
    pickle.dump(best_model, open(filename_dt, 'wb'))
    # Load model
    loaded_model_dt = pickle.load(open(filename_dt, 'rb'))
    result_dt = loaded_model_dt.score(X_test, y_test)
    # Print results
    print('Random Forest Regression Results:')
    print('R2 score:', r2)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    print('\nDecision Tree Regression Results:')
    print('Accuracy score:', result_dt)


if __name__ == "__main__":
    print('Executing main function...')
    main()
    print('Execution completed.')
    print('Done')
