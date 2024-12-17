# **Weather-Driven WTI Price Prediction**

## **Project Overview**
This project integrates weather data and financial data to predict short-term WTI (West Texas Intermediate) oil prices using machine learning models. The key steps involve data processing, feature engineering, testing, visualization, and model training (XGBoost, LightGBM, Linear Regression).

--- 

## **Project Structure**
The project is organized as follows:

```plaintext
project_folder/
│
├── data/                       # Input raw data files
│   ├── grib_files_dir/         # GRIB weather data files
│   └── Financial Data.csv      # Historical financial data
│
├── outputs/                    # Generated outputs
│   ├── processed_weather_financial_data.csv  # Final combined dataset
│   └── model_outputs/          # Model predictions and saved models
│       ├── xgboost_predictions.csv
│       ├── xgboost_model.pkl
│       ├── lightgbm_predictions.csv
│       └── lightgbm_model.pkl
│
├── scripts/
│   ├── preprocessing/          # Preprocessing and testing scripts
│   │   ├── process_data.py     # Combines weather and financial data
│   │   ├── test_data.py        # Tests the processed dataset
│   │   └── visualize_data.py   # Generates key visualizations
│   │
│   └── models/                 # Model training scripts
│       ├── train_xgboost.py    # XGBoost model training
│       ├── train_lightgbm.py   # LightGBM model training
│       └── train_linear_regression.py  # Linear Regression training
│
└── README.md                   # Project documentation
```
---

## **Requirements**
To set up the environment and run this project, install the following dependencies:

1. Python Libraries
Install the required libraries using pip:
```
pip install pandas numpy scikit-learn matplotlib xgboost lightgbm pygrib joblib
```
Note: pygrib requires additional setup. Install eccodes beforehand:
- Mac: brew install eccodes
- Linux: Use apt or a package manager for your distribution.

2. Tools
- Python 3.8+
- Anaconda (Optional) for environment management.

---

## **Data Sources**
- Weather Data: GRIB files containing weather forecasts (temperature, wind, pressure, humidity).
- Financial Data: Historical WTI oil price data, including volume and percentage change.

---

## **Step-by-Step Instructions**
1. Data Preprocessing
Combine and clean weather and financial data.
```
cd scripts/preprocessing
python process_data.py
```
Output: outputs/processed_weather_financial_data.csv

2. Dataset Testing
Verify the processed data integrity and model readiness.
```
python test_data.py
```
 Checks for:
    Missing values
    Summary statistics
    Lagged/forecasted features validation

3. Data Visualization
Generate visualizations to explore key trends.
```
python visualize_data.py
```

Outputs:
- WTI closing price trends
- Surface temperature trends
- Correlation heatmap

4. Train Machine Learning Models
Train different models to predict WTI prices.

XGBoost
```
cd ../models
python train_xgboost.py
```
LightGBM
```
python train_lightgbm.py
```
Linear Regression
```
python train_linear_regression.py
```
Outputs:
- Predictions CSV files
- Saved model files in outputs/model_outputs

---

## **Results**
The models are evaluated using:
   - RMSE: Root Mean Squared Error
   - R² Score: Model accuracy in predicting WTI prices.

Outputs and feature importance are visualized and saved for further analysis.

---

## **Future Improvements**
   - Incorporate larger datasets to improve model performance.
   - Fine-tune hyperparameters for better accuracy.
   - Explore additional machine learning algorithms, e.g., Random Forest or LSTM (deep learning).

---

## **License**
This project is licensed under the MIT License.
