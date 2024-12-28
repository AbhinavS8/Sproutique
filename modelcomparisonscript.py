import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                           precision_score, recall_score, accuracy_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('data_reduced.csv')

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

# Create binary classification targets for precision/recall metrics
df['high_risk'] = (df['pest_level'] > df['pest_level'].mean()).astype(int)

# Prepare features and targets
X = df[['temperature', 'humidity', 'ph']]
y_reg = df['pest_level']  # For regression
y_class = df['high_risk']  # For classification metrics

# Split the data
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
    X, y_reg, y_class, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM': SVR(kernel='rbf', C=10.0, epsilon=0.1),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

def evaluate_model(name, model, X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test):
    # Train model
    model.fit(X_train, y_reg_train)
    
    # Regression predictions
    y_reg_pred = model.predict(X_test)
    
    # Convert regression predictions to binary for classification metrics
    y_class_pred = (y_reg_pred > y_reg_train.mean()).astype(int)
    
    # Regression metrics
    r2 = r2_score(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    
    # Classification metrics
    accuracy = accuracy_score(y_class_test, y_class_pred)
    precision = precision_score(y_class_test, y_class_pred)
    recall = recall_score(y_class_test, y_class_pred)
    f1 = f1_score(y_class_test, y_class_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_reg_train, cv=5, scoring='r2')
    
    return {
        'Model': name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'CV R² (mean)': cv_scores.mean(),
        'CV R² (std)': cv_scores.std()
    }

# Evaluate all models
results = []
for name, model in models.items():
    metrics = evaluate_model(name, model, X_train_scaled, X_test_scaled,
                           y_reg_train, y_reg_test, y_class_train, y_class_test)
    results.append(metrics)

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Model')

# Print results
print("\nDetailed Model Comparison:")
print(results_df.round(3))

# Create visualization
plt.figure(figsize=(15, 10))

# Plot regression metrics
plt.subplot(2, 1, 1)
regression_metrics = ['R²', 'RMSE', 'MAE']
results_df[regression_metrics].plot(kind='bar', ax=plt.gca())
plt.title('Regression Metrics Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(title='Metrics')

# Plot classification metrics
plt.subplot(2, 1, 2)
classification_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
results_df[classification_metrics].plot(kind='bar', ax=plt.gca())
plt.title('Classification Metrics Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(title='Metrics')

plt.tight_layout()
plt.show()

# Save results to CSV
results_df.to_csv('model_comparison_results.csv')
print("\nResults have been saved to 'model_comparison_results.csv'")

# Print best model for each metric
print("\nBest Models by Metric:")
for column in results_df.columns:
    if column != 'CV R² (std)':
        best_model = results_df[column].idxmax()
        best_score = results_df[column].max()
        print(f"{column}: {best_model} ({best_score:.3f})")