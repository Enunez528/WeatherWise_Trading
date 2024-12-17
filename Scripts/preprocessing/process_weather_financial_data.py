import pygrib
import pandas as pd
import numpy as np
import os

# ----------------------------
# 1. Load and Process Weather Data
# ----------------------------

def process_grib_files(grib_files_dir, variable_names):
    """
    Processes GRIB files for specified weather variables and aggregates to daily data.
    Args:
        grib_files_dir (str): Directory containing GRIB files.
        variable_names (list): List of weather variables to extract from GRIB files.
    Returns:
        pd.DataFrame: Daily aggregated weather data.
    """
    daily_weather_data = []

    for file_name in sorted(os.listdir(grib_files_dir)):
        if file_name.endswith('.grib'):
            file_path = os.path.join(grib_files_dir, file_name)
            grbs = pygrib.open(file_path)
            valid_time = grbs.message(1).validDate.date()
            daily_record = {'date': valid_time}

            # Extract specified variables and compute daily means
            for var_name in variable_names:
                try:
                    messages = grbs.select(name=var_name)
                    values = [msg.values for msg in messages]
                    daily_record[var_name] = np.mean(values)
                except Exception as e:
                    print(f"Warning: Variable '{var_name}' not found in {file_name}. Skipping.")
                    daily_record[var_name] = np.nan

            daily_weather_data.append(daily_record)
            grbs.close()

    daily_weather_df = pd.DataFrame(daily_weather_data)
    return daily_weather_df

# Define the GRIB file directory and weather variables
grib_files_dir = os.path.join('../../data', 'grib_files_dir')
weather_variables = [
    '2 metre temperature',        # Surface temperature
    'Temperature',                # Temperature at pressure levels
    '10 metre U wind component',  # Surface U-component wind
    '10 metre V wind component',  # Surface V-component wind
    'U component of wind',        # Wind component at 500 hPa
    'V component of wind',        # Wind component at 500 hPa
    'Mean sea level pressure',    # Mean sea-level pressure
    'Specific humidity'           # Specific humidity at 700 hPa
]

# Process GRIB files and aggregate weather data
daily_weather = process_grib_files(grib_files_dir, weather_variables)

# Compute derived features for wind speed
daily_weather['max_wind_surface'] = np.sqrt(
    daily_weather['10 metre U wind component']**2 + daily_weather['10 metre V wind component']**2
)
daily_weather['max_wind_500'] = np.sqrt(
    daily_weather['U component of wind']**2 + daily_weather['V component of wind']**2
)

# Rename weather columns for consistency
daily_weather.rename(columns={
    '2 metre temperature': 'temp_surface',
    'Temperature': 'temp_500',
    'Mean sea level pressure': 'msl_surface',
    'Specific humidity': 'humidity_700'
}, inplace=True)

# Drop NaN rows caused by missing weather data
daily_weather.dropna(inplace=True)
print("Processed Weather Data:")
print(daily_weather.head())

# ----------------------------
# 2. Load and Process Financial Data
# ----------------------------

# Load financial data
financial_data = pd.read_csv(os.path.join('../../data', 'Financial Data.csv'))

# Rename columns for consistency
financial_data.rename(columns={
    'Date': 'timestamp',
    'Price': 'WTI_Close',
    'Open': 'WTI_Open',
    'High': 'WTI_High',
    'Low': 'WTI_Low',
    'Vol.': 'WTI_Volume',
    'Change %': 'WTI_Change_Percent'
}, inplace=True)

# Convert timestamp to datetime and sort
financial_data['timestamp'] = pd.to_datetime(financial_data['timestamp'])
financial_data.sort_values(by='timestamp', inplace=True)

# Ensure WTI_Close is numeric
financial_data['WTI_Close'] = pd.to_numeric(financial_data['WTI_Close'], errors='coerce')

# Clean and convert WTI_Volume and WTI_Change_Percent
financial_data['WTI_Volume'] = (
    financial_data['WTI_Volume']
    .str.replace('K', '', regex=False).astype(float) * 1000
)
financial_data['WTI_Change_Percent'] = (
    financial_data['WTI_Change_Percent']
    .str.replace('%', '', regex=False).astype(float) / 100
)

# Add derived financial features: SMA, RSI, and price range
financial_data['price_sma_20'] = financial_data['WTI_Close'].rolling(window=20, min_periods=20).mean()

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

financial_data['price_rsi'] = compute_rsi(financial_data['WTI_Close'])
financial_data['price_range'] = financial_data['WTI_High'] - financial_data['WTI_Low']

# Reset index after sorting
financial_data.reset_index(drop=True, inplace=True)

# Debugging outputs
print("Processed Financial Data:")
print(financial_data.head(25))  # First 25 rows to see initial NaN values
print("\nSummary Statistics:")
print(financial_data.describe())

# Check for missing dates or data integrity
print("\nDate Range in Financial Data:")
print(f"Start Date: {financial_data['timestamp'].min()} | End Date: {financial_data['timestamp'].max()}")
print(f"Total Rows: {financial_data.shape[0]}")
print(f"Unique Dates: {financial_data['timestamp'].nunique()}")

# ----------------------------
# 3. Merge Weather and Financial Data
# ----------------------------

# Ensure 'date' in daily_weather is in datetime format
daily_weather['date'] = pd.to_datetime(daily_weather['date'], errors='coerce')

# Ensure 'timestamp' in financial_data is in datetime format
financial_data['timestamp'] = pd.to_datetime(financial_data['timestamp'], errors='coerce')

# Validate for unexpected missing timestamps or date issues
print("\nValidating 'timestamp' and 'date' columns:")
print(f"Financial Data Null Timestamps: {financial_data['timestamp'].isnull().sum()}")
print(f"Weather Data Null Dates: {daily_weather['date'].isnull().sum()}")

# Drop any rows with null datetime values if present
financial_data.dropna(subset=['timestamp'], inplace=True)
daily_weather.dropna(subset=['date'], inplace=True)

# Merge weather and financial data
combined_data = pd.merge(
    financial_data,
    daily_weather,
    left_on='timestamp',
    right_on='date',
    how='inner'
)

# Drop the redundant 'date' column
combined_data.drop(columns=['date'], inplace=True)

# Filter the final dataset for January 1–10, 2023
combined_data = combined_data[
    (combined_data['timestamp'] >= '2023-01-01') &
    (combined_data['timestamp'] <= '2023-01-10')
]

# ----------------------------
# 4. Add Lagged and Forecasted Features
# ----------------------------

print("\nAdding lagged and forecasted features...")

# Add lagged financial features
combined_data['lag_WTI_Close'] = combined_data['WTI_Close'].shift(1)
combined_data['lag_volume'] = combined_data['WTI_Volume'].shift(1)

# Add lagged weather features
combined_data['lag_temp_surface'] = combined_data['temp_surface'].shift(1)
combined_data['lag_max_wind_surface'] = combined_data['max_wind_surface'].shift(1)

# Add forecasted weather features
combined_data['forecast_temp_surface'] = combined_data['temp_surface'].shift(-1)

# ----------------------------
# 5. Impute Missing Values Instead of Dropping
# ----------------------------

print("\nHandling missing values with forward and backward fill...")

# Forward fill followed by backward fill for any remaining NaNs
combined_data.fillna(method='ffill', inplace=True)
combined_data.fillna(method='bfill', inplace=True)

# Validate that no NaN values remain
print("\nFinal Check for Missing Values:")
print(combined_data.isnull().sum())

# Display a final summary of the combined dataset
print("\nFinal Combined Dataset Summary:")
print(combined_data.head())
print(f"\nDataset shape (rows, columns): {combined_data.shape}")

# Save the final combined dataset to a CSV
output_path = os.path.join('../../outputs', 'processed_weather_financial_data.csv')
combined_data.to_csv(output_path, index=False)

print(f"\nFinal combined dataset successfully saved to: {output_path}")
