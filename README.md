# ShadowFox - Boston House Price Prediction

A machine learning project to predict Boston house prices using various regression models. This project implements data preprocessing, multiple model training (Linear, Ridge, Lasso, Random Forest, and Gradient Boosting), and comprehensive evaluation with visualizations.

## 🎯 Project Overview

The project uses the Boston House Prices dataset to build and compare multiple regression models that predict house prices based on features such as:
- Crime rates (CRIM)
- Number of rooms (RM)
- Age of the property (AGE)
- Distance to employment centers (DIS)
- Tax rates (TAX)
- Lower status population percentage (LSTAT)
- And more...

## 📋 Features

- **Data Preprocessing**: Handle missing values, data cleaning
- **Exploratory Data Analysis**: Statistical analysis and visualizations
- **Multiple Model Comparison**: 
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Model Evaluation**: RMSE, MAE, and R² score metrics
- **Feature Importance Analysis**: Understanding which features matter most
- **Visualizations**: Comprehensive graphs and charts

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone or download the repository**
   ```bash
   cd ShadowFox
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**
   - Place the `boston_house_prices.csv` file in the `data/` directory
   - The script will automatically handle the data loading

## 💻 How to Run

### Method 1: Run Python Script
```bash
python boston_housing_regression.py
```

### Method 2: Run in Jupyter Notebook
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Create a new notebook and copy the code from `boston_housing_regression.py`

## 📊 Output

The script will generate the following:

1. **Console Output**:
   - Dataset information
   - Preprocessing results
   - Model training progress
   - Performance metrics for each model
   - Best model identification

2. **Results Folder**:
   - `exploratory_analysis.png` - Data distribution and correlation analysis
   - `model_predictions.png` - Actual vs Predicted values for all models
   - `model_comparison.png` - Performance comparison across models
   - `feature_importance.png` - Feature importance analysis

## 📁 Project Structure

```
ShadowFox/
│
├── data/
│   └── boston_house_prices.csv    # Dataset file
│
├── results/                        # Generated visualizations
│   ├── exploratory_analysis.png
│   ├── model_predictions.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── boston_housing_regression.py   # Main script
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🔍 Model Metrics Explained

- **RMSE (Root Mean Squared Error)**: Lower is better. Measures prediction accuracy.
- **MAE (Mean Absolute Error)**: Lower is better. Average prediction error.
- **R² Score**: Higher is better (max 1.0). Proportion of variance explained.

## 🛠️ Customization

You can modify the script to:
- Adjust hyperparameters for models
- Add new regression models
- Change train/test split ratio
- Modify visualization settings

## 📝 Notes

- The dataset should be in CSV format
- The script automatically creates the `results/` directory if it doesn't exist
- Random seed is set to 42 for reproducibility

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available for educational purposes.
