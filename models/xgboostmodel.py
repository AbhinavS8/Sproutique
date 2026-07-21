import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load and preprocess dataset
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

# Calculate optimal grid points
def calculate_optimized_grid(field_length, field_width, sensor_radius):
    num_x = int(np.ceil(field_length / (2 * sensor_radius)))
    num_y = int(np.ceil(field_width / (2 * sensor_radius)))
    return num_x, num_y

# Collect data points from the user
def collect_user_data(grid_points):
    data_points = []
    print("\nEnter data for each grid point:")
    for idx, (x, y) in enumerate(grid_points):
        print(f"\nGrid Point {idx + 1}: (X={x:.2f}, Y={y:.2f})")
        try:
            temperature = float(input("  Temperature (°C): "))
            humidity = float(input("  Humidity (%): "))
            ph = float(input("  pH: "))
            data_points.append({'X': x, 'Y': y, 'Temperature': temperature, 'Humidity': humidity, 'pH': ph})
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return None
    return data_points

# Predict pest levels for the field
def predict_field_pest_levels(field_length, field_width, sensor_radius):
    num_x, num_y = calculate_optimized_grid(field_length, field_width, sensor_radius)
    sensor_diameter = 2 * sensor_radius
    grid_points = [(i * sensor_diameter + sensor_radius, j * sensor_diameter + sensor_radius)
                   for i in range(num_x) for j in range(num_y)]

    print(f"\nOptimized Sensor Placement: {len(grid_points)} sensors")
    user_data = collect_user_data(grid_points)
    if not user_data:
        return

    # Prediction for each data point
    results = []
    for data in user_data:
        prediction, risk = predict_pest_level_xgb(data['Temperature'], data['Humidity'], data['pH'])
        results.append({
            'X': data['X'],
            'Y': data['Y'],
            'Temperature': data['Temperature'],
            'Humidity': data['Humidity'],
            'pH': data['pH'],
            'Pest Level': prediction,
            'Risk': risk
        })

    results_df = pd.DataFrame(results)
    print("\nPest Prediction Results:")
    print(results_df)
    results_df.to_csv('field_pest_predictions.csv', index=False)
    print("\nResults saved to 'field_pest_predictions.csv'.")

# User input for field dimensions
def get_field_input_and_predict():
    print("\nPest Infestation Prediction for Field\n")
    try:
        field_length = float(input("Enter field length (in meters): "))
        field_width = float(input("Enter field width (in meters): "))
        sensor_radius = 10.0  # Fixed radius for DHT sensor
        print(f"\nUsing sensor radius: {sensor_radius} meters")
        predict_field_pest_levels(field_length, field_width, sensor_radius)
    except ValueError:
        print("\nInvalid input. Please enter numerical values.")

# Run the system
get_field_input_and_predict()
