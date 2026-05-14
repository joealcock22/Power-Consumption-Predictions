import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import load_and_preprocess_data

#Training Model on the Dataset 
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)
    return model

#Evaluationn of the Model performance on the test set
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")
    return predictions

#Visualisation of actual vs predicted values
def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha = 0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()

#Main Function
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)
    plot_predictions(y_test, predictions)


