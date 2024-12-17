import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import os
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Processed Data
# ----------------------------
file_path = 'outputs/processed_weather_financial_data.csv'
output_dir = 'outputs/model_outputs'
os.makedirs(output_dir, exist_ok=True)

try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}.")
    exit()

# Convert 'timestamp' to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# ----------------------------
# 2. Feature Engineering
# ----------------------------
print("\nFeature engineering in progress...")

# Compute the 5-day simple moving average (sma_5)
data['sma_5'] = data['WTI_Close'].rolling(window=5).mean()

# Selected Features: Adding back key features
features = ['price_rsi', 'humidity_700', 'temp_surface', 'lag_WTI_Close', 'sma_5']
target = 'WTI_Close'

# Drop rows with NaNs in selected features and target
data = data[features + [target]].dropna()

# Check dataset shape
print(f"Final dataset shape: {data.shape}")

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X = data[features]
y = data[target]

# Ensure reproducibility with shuffle=False for time series data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ----------------------------
# 4. Train Ridge Regression Model
# ----------------------------
model = Ridge(alpha=1.0)  # Regularized Linear Regression
print("\nTraining Ridge Regression model...")
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

# Save predictions to CSV
predictions = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
}, index=y_test.index)

predictions_path = os.path.join(output_dir, 'ridge_predictions.csv')
predictions.to_csv(predictions_path, index=False)
print(f"\nPredictions saved to: {predictions_path}")

# ----------------------------
# 6. Visualize Predictions
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual WTI Prices', marker='o', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted WTI Prices', linestyle='--', marker='x', linewidth=2)
plt.title('Ridge Regression Predictions vs Actual WTI Prices')
plt.xlabel('Index')
plt.ylabel('WTI Closing Price (USD)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
