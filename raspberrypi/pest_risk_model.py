import numpy as np
import xgboost as xgb
from joblib import load

scaler = load("model/scaler.joblib")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("model/xgb_pest_model.json")

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

if __name__ == "__main__":
    get_user_input_and_predict()