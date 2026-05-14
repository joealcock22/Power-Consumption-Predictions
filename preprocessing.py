import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    data = pd.read_csv('Energy_Consumption.csv')
    data = data.dropna()
    X = data.drop('Energy_Consumption', axis=1)
    y = data['Energy_Consumption']
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def preprcess_new_data(new_data, scaler):
    new_data_scaled = scaler.transform(new_data)
    return new_data_scaled

