import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Assuming the dataset has already been preprocessed
df = pd.read_csv('data_reduced.csv')

# Enhanced pest risk calculation
def calculate_pest_risk_enhanced(row):
    temp_score = 100 - min(abs(row['temperature'] - 25) * 5, 100)
    humidity_score = 100 - min(abs(row['humidity'] - 80) * 3, 100)
    ph_score = 100 - min(abs(row['ph'] - 6.8) * 20, 100)
    temp_humidity_interaction = (100 - abs(temp_score - humidity_score)) * 0.1
    pest_score = (0.35 * temp_score + 
                  0.35 * humidity_score + 
                  0.2 * ph_score + 
                  0.1 * temp_humidity_interaction)
    variation = np.random.uniform(-2.5, 2.5)
    return max(0, min(100, pest_score + variation))

df['pest_level'] = df.apply(calculate_pest_risk_enhanced, axis=1)

# Prepare features and target
X = df[['temperature', 'humidity', 'ph']]
y = df['pest_level']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
xgb_model.fit(X_train_scaled, y_train)

# Function for prediction
def predict_pest_level_xgb(temperature, humidity, ph):
    input_data = np.array([[temperature, humidity, ph]])
    input_scaled = scaler.transform(input_data)
    prediction = xgb_model.predict(input_scaled)[0]
    if prediction < 20:
        risk = "Very Low"
    elif prediction < 40:
        risk = "Low"
    elif prediction < 60:
        risk = "Moderate"
    elif prediction < 80:
        risk = "High"
    else:
        risk = "Very High"
    return prediction, risk

# User input and prediction
def get_user_input_and_predict():
    print("\nPest Infestation Prediction System\n")
    try:
        temperature = float(input("Enter temperature (°C): "))
        humidity = float(input("Enter humidity (%): "))
        ph = float(input("Enter pH value: "))
        prediction, risk = predict_pest_level_xgb(temperature, humidity, ph)
        print(f"\nPredicted Pest Level: {prediction:.2f}")
        print(f"Risk Category: {risk}")
    except ValueError:
        print("\nInvalid input. Please enter numerical values.")

# Run the system
if __name__ == "__main__":
    get_user_input_and_predict()