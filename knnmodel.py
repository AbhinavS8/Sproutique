import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('agriculture_data_reduced.csv')

# Calculate pest risk (similar logic as before)
def calculate_pest_risk_enhanced(row):
    temp_score = 100 - min(abs(row['temperature'] - 25) * 5, 100)
    humidity_score = 100 - min(abs(row['humidity'] - 80) * 3, 100)
    ph_score = 100 - min(abs(row['ph'] - 6.8) * 20, 100)
    temp_humidity_interaction = (100 - abs(temp_score - humidity_score)) * 0.1
    pest_score = (0.35 * temp_score + 0.35 * humidity_score + 
                  0.2 * ph_score + 0.1 * temp_humidity_interaction)
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

# Initialize and train k-NN model
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_model.fit(X_train_scaled, y_train)

# Evaluate k-NN model
def evaluate_knn():
    y_pred_knn = knn_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_knn)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    mae = mean_absolute_error(y_test, y_pred_knn)
    cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5, scoring='r2')
    return {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_scores_mean': cv_scores.mean(),
        'cv_scores_std': cv_scores.std()
    }

# Hyperparameter tuning for k-NN
def tune_knn():
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, y_train)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Example prediction function
def predict_pest_level_knn(temperature, humidity, ph):
    input_data = np.array([[temperature, humidity, ph]])
    input_scaled = scaler.transform(input_data)
    prediction = knn_model.predict(input_scaled)[0]
    return round(prediction, 2)

# Visualization function
def create_visualizations():
    y_pred_knn = knn_model.predict(X_test_scaled)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred_knn, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Pest Level')
    plt.ylabel('Predicted Pest Level')
    plt.title('Actual vs Predicted Pest Levels (k-NN)')
    plt.subplot(2, 2, 2)
    correlation_matrix = df[['temperature', 'humidity', 'ph', 'pest_level']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    plt.subplot(2, 2, 3)
    errors = y_test - y_pred_knn
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors (k-NN)')
    plt.subplot(2, 2, 4)
    feature_corr = abs(correlation_matrix['pest_level'])[:-1]
    feature_corr.plot(kind='bar')
    plt.title('Feature Importance (Correlation)')
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation with Pest Level')
    plt.tight_layout()
    plt.show()

# Evaluate and display results
evaluation = evaluate_knn()
print("\nModel Performance Metrics (k-NN):")
print(f"R² Score: {evaluation['r2_score']:.3f}")
print(f"RMSE: {evaluation['rmse']:.3f}")
print(f"MAE: {evaluation['mae']:.3f}")
print(f"Cross-validation R² (mean ± std): {evaluation['cv_scores_mean']:.3f} ± {evaluation['cv_scores_std']:.3f}")

# Example predictions
test_conditions = [{'temperature': 25.0, 'humidity': 80.0, 'ph': 7.0},
                   {'temperature': 20.0, 'humidity': 75.0, 'ph': 6.0},
                   {'temperature': 30.0, 'humidity': 85.0, 'ph': 8.0}]
print("\nExample Predictions:")
for conditions in test_conditions:
    pest_level = predict_pest_level_knn(conditions['temperature'], 
                                        conditions['humidity'], 
                                        conditions['ph'])
    print(f"Conditions: Temp={conditions['temperature']}°C, "
          f"Humidity={conditions['humidity']}%, pH={conditions['ph']}")
    print(f"Pest Level Prediction: {pest_level}")

# Create visualizations
create_visualizations()

# Save predictions to CSV
df['predicted_pest_level_knn'] = knn_model.predict(scaler.transform(X))
df.to_csv('agriculture_with_knn_predictions.csv', index=False)
print("\nPredictions have been saved to 'agriculture_with_knn_predictions.csv'")
