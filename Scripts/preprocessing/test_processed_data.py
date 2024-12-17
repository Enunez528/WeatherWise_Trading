import pandas as pd
import os

# ----------------------------
# 1. Load the Dataset
# ----------------------------

# Load the processed dataset
file_path = os.path.join('../../outputs', 'processed_weather_financial_data.csv')
try:
    data = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from {file_path}.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}.")
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Display the structure and data types
print("\nDataset Info:")
print(data.info())

# ----------------------------
# 2. Check for Missing Values
# ----------------------------

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

# ----------------------------
# 3. Basic Statistics
# ----------------------------

# Summarize numerical columns
print("\nSummary statistics for numerical columns:")
print(data.describe())

# ----------------------------
# 4. Validate Lagged and Forecasted Features
# ----------------------------

def validate_features(data):
    print("\nValidating lagged and forecasted features...")
    for row in range(1, 3):  # Validate first 2 rows dynamically
        try:
            print(f"\nRow {row} Validation:")
            print(f"Lagged Price: {data.loc[row, 'lag_WTI_Close']} | Previous Price: {data.loc[row-1, 'WTI_Close']}")
            print(f"Forecast Temp (Row {row-1}): {data.loc[row-1, 'forecast_temp_surface']} | Next Day Temp (Row {row}): {data.loc[row, 'temp_surface']}")
        except KeyError as e:
            print(f"Error validating features: {e}")
        except IndexError as e:
            print(f"Index error: {e}")

validate_features(data)

# ----------------------------
# 5. Test Dataset for Model Readiness
# ----------------------------

def test_model_readiness(data):
    print("\nTesting dataset readiness for modeling...")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # Select a few features for testing
    features = ['temp_surface', 'WTI_Volume', 'lag_WTI_Close']
    target = 'WTI_Close'
    
    # Check if all features are in the dataset
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Error: Missing required features {missing_features}.")
        return

    # Prepare data
    X = data[features].dropna()  # Drop rows with missing values
    y = data[target][X.index]   # Align target with features

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Check model performance
    score = model.score(X_test, y_test)
    print(f"Model R^2 Score: {score:.2f}")

test_model_readiness(data)

# ----------------------------
# 6. Check File Size and Memory Usage
# ----------------------------

print("\nMemory usage (human-readable):")
memory_usage = data.memory_usage(deep=True).sum()
print(f"Total Memory Usage: {memory_usage / (1024 ** 2):.2f} MB")

print("\nNumber of rows:", data.shape[0])
print("Number of columns:", data.shape[1] )

# ----------------------------
# Testing Completed
# ----------------------------
print("\nTesting completed. Review the outputs above.")
