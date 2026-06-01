import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load the energy consumption dataset and perform preprocessing.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Convert timestamp to datetime and extract features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day

    # Encode categorical variables
    categorical_cols = ['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Select features and target
    feature_cols = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
                   'HVACUsage', 'LightingUsage', 'RenewableEnergy', 'DayOfWeek',
                   'Holiday', 'Hour', 'Month', 'Day']

    X = df[feature_cols]
    y = df['EnergyConsumption']

    # Scale numerical features
    numerical_cols = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
                     'RenewableEnergy', 'Hour', 'Month', 'Day']

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, feature_cols


