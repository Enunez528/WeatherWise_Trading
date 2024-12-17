import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import os
import joblib

# ----------------------------
# 1. Load Processed Data
# ----------------------------
file_path = 'outputs/processed_weather_financial_data.csv'
output_dir = 'outputs/model_outputs'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = pd.read_csv(file_path)
data['timestamp'] = pd.to_datetime(data['timestamp'])

# ----------------------------
# 2. Feature Engineering
# ----------------------------
features = ['price_rsi', 'humidity_700']
target = 'WTI_Close'
data = data[features + [target]].dropna()

# Train-test split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# ----------------------------
# 3. Hyperparameter Tuning with GridSearchCV
# ----------------------------
param_grid = {
    'n_estimators': [10, 20, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1.0]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

print("\nStarting hyperparameter tuning...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,  # Cross-validation folds
    scoring='neg_mean_squared_error',
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
print(f"\nBest Hyperparameters: {best_params}")

best_model = grid_search.best_estimator_

# ----------------------------
# 4. Evaluate the Best Model
# ----------------------------
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance with Best Parameters:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model and predictions
joblib.dump(best_model, os.path.join(output_dir, 'xgboost_best_model.pkl'))
predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}, index=y_test.index)
predictions.to_csv(os.path.join(output_dir, 'xgboost_best_predictions.csv'), index=False)