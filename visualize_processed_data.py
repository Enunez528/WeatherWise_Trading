import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# ----------------------------
# 2. Define Visualization Functions
# ----------------------------

def plot_data(data, column, title, ylabel):
    """Plot a single column over time."""
    plt.figure(figsize=(10, 6))
    try:
        plt.plot(pd.to_datetime(data['timestamp']), data[column], label=column)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.show()
    except KeyError:
        print(f"Column '{column}' not found in the dataset.")

def overlay_trends(data, col1, col2, title, ylabel1, ylabel2):
    """Overlay two trends on the same plot with dual axes."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Date')
    ax1.set_ylabel(ylabel1, color='tab:blue')
    ax1.plot(pd.to_datetime(data['timestamp']), data[col1], color='tab:blue', label=col1)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel2, color='tab:orange')
    ax2.plot(pd.to_datetime(data['timestamp']), data[col2], color='tab:orange', label=col2)
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title(title)
    plt.show()

def plot_histogram(data, column, bins=20):
    """Plot histogram of a column."""
    plt.figure(figsize=(8, 5))
    plt.hist(data[column].dropna(), bins=bins, color='gray', alpha=0.7)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

def plot_correlation_heatmap(data):
    """Plot correlation heatmap for numerical features."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# ----------------------------
# 3. Generate Plots
# ----------------------------

print("\nGenerating visualizations...")

# Plot key trends
plot_data(data, 'WTI_Close', 'WTI Closing Prices Over Time', 'Price (USD)')
plot_data(data, 'temp_surface', 'Surface Temperature Over Time', 'Temperature (K)')
plot_data(data, 'WTI_Volume', 'WTI Trading Volume Over Time', 'Volume')

# Overlay trends
overlay_trends(data, 'WTI_Close', 'temp_surface', 
               'WTI Close Price vs Surface Temperature', 'Price (USD)', 'Temperature (K)')

# Plot distributions
plot_histogram(data, 'WTI_Close')
plot_histogram(data, 'temp_surface')

# Plot correlation heatmap
plot_correlation_heatmap(data)

# ----------------------------
# Visualization Completed
# ----------------------------
print("\nVisualizations completed. Review the plots above.")