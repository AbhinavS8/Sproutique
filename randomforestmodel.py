import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV data
df = pd.read_csv('data_reduced.csv')

# Generate target variable based on domain knowledge
def calculate_pest_risk(row):
    """
    Calculate pest risk based on known favorable conditions:
    - Temperature: Pests thrive between 20-30°C
    - Humidity: Higher humidity (75-85%) favors pest growth
    - pH: Most pests prefer slightly acidic to neutral soil (6-7.5)
    """
    temp_score = 100 - min(abs(row['temperature'] - 25) * 5, 100)  # Optimal around 25°C
    humidity_score = 100 - min(abs(row['humidity'] - 80) * 3, 100)  # Optimal around 80%
    ph_score = 100 - min(abs(row['ph'] - 6.8) * 20, 100)  # Optimal around 6.8
    
    # Combine scores with weights
    pest_score = (0.4 * temp_score + 0.4 * humidity_score + 0.2 * ph_score)
    
    # Add some random variation (±10%)
    variation = np.random.uniform(-5, 5)
    pest_score = max(0, min(100, pest_score + variation))
    
    return pest_score

# Add pest_level column based on environmental conditions
df['pest_level'] = df.apply(calculate_pest_risk, axis=1)

# Prepare data for modeling
X = df[['temperature', 'humidity', 'ph']]
y = df['pest_level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Function to predict pest level
def predict_pest_level(temperature, humidity, ph):
    """
    Predict pest infestation level given environmental conditions
    Returns a value between 0-100 where:
    0-20: Very Low
    21-40: Low
    41-60: Moderate
    61-80: High
    81-100: Very High
    """
    input_data = np.array([[temperature, humidity, ph]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    # Get feature importance
    feature_importance = dict(zip(['Temperature', 'Humidity', 'pH'], 
                                model.feature_importances_))
    
    # Determine risk level
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
    
    return {
        'pest_level': round(prediction, 2),
        'risk_category': risk,
        'feature_importance': feature_importance
    }

# Print some example data points
print("Sample of input data with calculated pest levels:")
print(df[['temperature', 'humidity', 'ph', 'pest_level']].head())

# Model evaluation
print("\nModel Performance Metrics:")
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MSE: {mse:.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
print(f"Cross-Validation R²: Mean = {cv_mean:.3f}, Std = {cv_std:.3f}")

# Example predictions for different conditions
test_conditions = [
    {'temperature': 25.0, 'humidity': 80.0, 'ph': 7.0},  # Optimal conditions
    {'temperature': 20.0, 'humidity': 75.0, 'ph': 6.0},  # Suboptimal conditions
    {'temperature': 30.0, 'humidity': 85.0, 'ph': 8.0}   # Extreme conditions
]

print("\nExample Predictions:")
for conditions in test_conditions:
    result = predict_pest_level(**conditions)
    print(f"\nConditions: Temp={conditions['temperature']}°C, "
          f"Humidity={conditions['humidity']}%, "
          f"pH={conditions['ph']}")
    print(f"Pest Level: {result['pest_level']}")
    print(f"Risk Category: {result['risk_category']}")

# Print feature importance
importance = model.feature_importances_
features = ['Temperature', 'Humidity', 'pH']
print("\nFeature Importance:")
for feature, imp in zip(features, importance):
    print(f"{feature}: {imp:.3f}")

# Save predictions to CSV
df['predicted_pest_level'] = model.predict(scaler.transform(X))
df.to_csv('agriculture_with_pest_predictions.csv', index=False)
print("\nPredictions have been saved to 'agriculture_with_pest_predictions.csv'")
