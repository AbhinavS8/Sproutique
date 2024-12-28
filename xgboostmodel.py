import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Read and prepare the data
df = pd.read_csv('data_reduced.csv')

# Enhanced pest risk calculation with additional considerations
def calculate_pest_risk_enhanced(row):
    """
    Calculate pest risk with enhanced logic:
    - Temperature: Pests thrive between 20-30°C with peak at 25°C
    - Humidity: Higher humidity (75-85%) favors pest growth with peak at 80%
    - pH: Most pests prefer slightly acidic to neutral soil (6-7.5) with peak at 6.8
    - Added interaction effects between variables
    """
    temp_score = 100 - min(abs(row['temperature'] - 25) * 5, 100)
    humidity_score = 100 - min(abs(row['humidity'] - 80) * 3, 100)
    ph_score = 100 - min(abs(row['ph'] - 6.8) * 20, 100)
    
    # Add interaction effects
    temp_humidity_interaction = (100 - abs(temp_score - humidity_score)) * 0.1
    
    # Combine scores with weights including interaction
    pest_score = (0.35 * temp_score + 
                  0.35 * humidity_score + 
                  0.2 * ph_score + 
                  0.1 * temp_humidity_interaction)
    
    # Add controlled random variation (±5%)
    variation = np.random.uniform(-2.5, 2.5)
    pest_score = max(0, min(100, pest_score + variation))
    
    return pest_score

# Generate target variable
df['pest_level'] = df.apply(calculate_pest_risk_enhanced, axis=1)

# Prepare features and target
X = df[['temperature', 'humidity', 'ph']]
y = df['pest_level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train XGBoost model with optimized parameters
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'  # Explicitly set objective for regression
)

# Train the model with corrected parameters
eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
xgb_model.fit(
    X_train_scaled, 
    y_train,
    eval_set=eval_set,
    verbose=False
)

def predict_pest_level_xgb(temperature, humidity, ph):
    """
    Predict pest infestation level using XGBoost model
    Returns detailed prediction information including confidence metrics
    """
    input_data = np.array([[temperature, humidity, ph]])
    input_scaled = scaler.transform(input_data)
    
    # Get prediction
    prediction = xgb_model.predict(input_scaled)[0]
    
    # Get feature importance
    feature_importance = dict(zip(['Temperature', 'Humidity', 'pH'], 
                                xgb_model.feature_importances_))
    
    # Calculate prediction confidence based on feature ranges
    temp_range = abs(temperature - X_train['temperature'].mean()) / X_train['temperature'].std()
    humid_range = abs(humidity - X_train['humidity'].mean()) / X_train['humidity'].std()
    ph_range = abs(ph - X_train['ph'].mean()) / X_train['ph'].std()
    confidence_score = 100 * (1 - np.mean([temp_range, humid_range, ph_range]) / 3)
    confidence_score = max(0, min(100, confidence_score))
    
    # Determine risk level with more granular categories
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
        'confidence_score': round(confidence_score, 2),
        'feature_importance': feature_importance
    }

# Model evaluation
def evaluate_model():
    """
    Comprehensive model evaluation including cross-validation
    """
    # Predictions on test set
    y_pred = xgb_model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, 
                              cv=5, scoring='r2')
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std()
    }

# Print evaluation results
evaluation = evaluate_model()
print("\nModel Performance Metrics:")
print(f"R² Score: {evaluation['r2_score']:.3f}")
print(f"RMSE: {evaluation['rmse']:.3f}")
print(f"MAE: {evaluation['mae']:.3f}")
print(f"Cross-validation R² (mean ± std): {evaluation['cv_scores_mean']:.3f} ± {evaluation['cv_scores_std']:.3f}")

# Example predictions
test_conditions = [
    {'temperature': 25.0, 'humidity': 80.0, 'ph': 7.0},  # Optimal conditions
    {'temperature': 20.0, 'humidity': 75.0, 'ph': 6.0},  # Suboptimal conditions
    {'temperature': 30.0, 'humidity': 85.0, 'ph': 8.0}   # Extreme conditions
]

print("\nExample Predictions:")
for conditions in test_conditions:
    result = predict_pest_level_xgb(**conditions)
    print(f"\nConditions: Temp={conditions['temperature']}°C, "
          f"Humidity={conditions['humidity']}%, "
          f"pH={conditions['ph']}")
    print(f"Pest Level: {result['pest_level']}")
    print(f"Risk Category: {result['risk_category']}")
    print(f"Prediction Confidence: {result['confidence_score']}%")

# Save predictions to CSV
df['predicted_pest_level_xgb'] = xgb_model.predict(scaler.transform(X))
df.to_csv('agriculture_with_xgboost_predictions.csv', index=False)
print("\nPredictions have been saved to 'agriculture_with_xgboost_predictions.csv'")