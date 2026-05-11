# Power-Consumption-Predictions

A machine learning algorithm to predict energy consumption based on various environmental and operational factors.

## Dataset

The dataset contains hourly energy consumption data with the following features:
- **Timestamp**: Date and time of the reading
- **Temperature**: Ambient temperature
- **Humidity**: Relative humidity percentage
- **SquareFootage**: Building area in square feet
- **Occupancy**: Number of occupants
- **HVACUsage**: HVAC system status (On/Off)
- **LightingUsage**: Lighting system status (On/Off)
- **RenewableEnergy**: Renewable energy contribution
- **DayOfWeek**: Day of the week
- **Holiday**: Whether it's a holiday (Yes/No)
- **EnergyConsumption**: Target variable - energy consumption in kWh

## Algorithm

The prediction algorithm uses a Random Forest Regressor trained on the preprocessed dataset. Key preprocessing steps include:

1. **Feature Engineering**: Extract hour, month, and day from timestamp
2. **Categorical Encoding**: Convert categorical variables to numerical using Label Encoding
3. **Feature Scaling**: Standardize numerical features
4. **Train/Test Split**: 80/20 split for model evaluation

## Installation

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Run the prediction algorithm:

```bash
python predictions.py
```

This will:
1. Load and preprocess the data
2. Train the Random Forest model
3. Evaluate model performance
4. Generate feature importance and prediction plots

## Model Performance

The model is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Files

- `predictions.py`: Main script for training and evaluating the model
- `preprocessing.py`: Data preprocessing functions
- `data/Energy_consumption.csv`: Input dataset
- `feature_importance.png`: Feature importance visualization
- `predictions_vs_actual.png`: Model predictions vs actual values plot 
