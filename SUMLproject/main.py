import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from geopy.distance import geodesic


def preprocess_data(df):
    # Conversion of data types
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype('float64')
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype('float64')

    # Replace 'NaN' with np.nan
    df.replace('NaN', np.nan, inplace=True)

    # Handle null values (if necessary, replace with your own logic)
    df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median(), inplace=True)
    df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median(), inplace=True)

    # Feature Engineering: Distance calculation
    df['distance'] = df.apply(lambda row: geodesic(
        (row['Restaurant_latitude'], row['Restaurant_longitude']),
        (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    ).km, axis=1)

    # Dropping unnecessary columns
    df.drop(['ID', 'Delivery_person_ID', 'Type_of_order', 'Type_of_vehicle'], axis=1, inplace=True)

    return df


def train_model(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
    model.fit(X_train, y_train, verbose=True)
    return model


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions


if __name__ == '__main__':
    df = pd.read_csv('deliverytime.csv')
    df_processed = preprocess_data(df)
    X = df_processed.drop('Time_taken(min)', axis=1)
    y = df_processed['Time_taken(min)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    save_model(model, 'delivery_time_predictor.pkl')
    loaded_model = load_model('delivery_time_predictor.pkl')
    predictions = make_predictions(loaded_model, X_test)
