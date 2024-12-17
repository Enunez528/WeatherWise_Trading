import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib  # To save the model
import os
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Processed Data
# ----------------------------
file_path = 'outputs/processed_weather_financial_data.csv'
output_dir = 'outputs/model_outputs'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}.")
    exit()

data['timestamp'] = pd.to_datetime(data['timestamp'])

# ----------------------------
# 2. Feature Engineering
# ----------------------------
print("\nFeature engineering in progress...")
data['sma_5'] = data['WTI_Close'].rolling(window=5).mean()

# Features to use
features = ['price_rsi', 'humidity_700', 'temp_surface', 'max_wind_surface', 'lag_WTI_Close', 'sma_5']
target = 'WTI_Close'

# Drop rows with NaNs in selected features and target
data = data[features + [target]].dropna()
print(f"Final dataset shape: {data.shape}")

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ----------------------------
# 4. Train LightGBM Model
# ----------------------------
model = lgb.LGBMRegressor(
    n_estimators=50,          # Fewer iterations for small data
    learning_rate=0.05,       # Step size
    max_depth=3,              # Depth of the trees
    num_leaves=20,            # Number of leaves in a tree
    reg_alpha=0.5,            # L1 regularization
    reg_lambda=0.5,           # L2 regularization
    random_state=42
)

print("\nTraining LightGBM model...")
model.fit(X_train, y_train)

# ----------------------------
# 5. Make Predictions and Evaluate
# ----------------------------
y_pred = model.predict(X_test)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save predictions to a CSV file
predictions = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
}, index=y_test.index)

predictions_path = os.path.join(output_dir, 'lightgbm_predictions.csv')
predictions.to_csv(predictions_path, index=False)
print(f"\nPredictions saved to: {predictions_path}")

# Save the trained LightGBM model
model_path = os.path.join(output_dir, 'lightgbm_model.pkl')
joblib.dump(model, model_path)
print(f"Trained model saved to: {model_path}")

# ----------------------------
# 6. Visualize Predictions
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Actual WTI Prices', marker='o', linewidth=2)
plt.plot(range(len(y_test)), y_pred, label='Predicted WTI Prices', linestyle='--', marker='x', linewidth=2)
plt.title('LightGBM Predictions vs Actual WTI Prices')
plt.xlabel('Index')
plt.ylabel('WTI Closing Price (USD)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Feature Importance
# ----------------------------
print("\nPlotting feature importance...")
lgb.plot_importance(model, importance_type='split', max_num_features=5)
plt.title('Top 5 Feature Importance')
plt.tight_layout()
plt.show()
