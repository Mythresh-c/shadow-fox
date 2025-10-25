# Car Selling Price Prediction

A machine learning project to predict car selling prices based on various features such as fuel type, years of service, engine size, mileage, and other relevant factors.

## 📋 Project Overview

This project implements a comprehensive ML solution to predict car selling prices. The system provides users with an approximate selling price for their cars based on several features including:
- Fuel type (Petrol, Diesel, Hybrid)
- Years of service (based on manufacture year)
- Engine size
- Mileage (kilometers driven)
- Manufacturer and Model information

## 🚀 Features

- **Data Preprocessing**: Handle missing values, feature engineering
- **Multiple Model Comparison**: 
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Model Evaluation**: RMSE, MAE, and R² score metrics
- **Visualizations**: Model performance comparison charts

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Navigate to the project directory**
   ```bash
   cd Project2
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 How to Run

### Run the Main Script
```bash
python car_price_prediction.py
```

The script will:
1. Load and explore the data
2. Preprocess the dataset
3. Train multiple regression models
4. Evaluate and compare models
5. Generate visualizations and save them to the `results/` directory
6. Run an example prediction on a car from the dataset

### Run Prediction Examples
```bash
python predict_example.py
```

This will demonstrate how to predict car prices with 3 different examples.

## 💡 Model Prediction Results

Based on testing, the **Gradient Boosting** model performs best with:
- **R² Score**: 0.9575 (95.75% accuracy)
- **RMSE**: $3,397

Example prediction:
- **Car**: 2002 Ford Fiesta, 1.0L Petrol, 127,300 km
- **Actual Price**: $3,074
- **Random Forest Prediction**: $3,244 (5.53% error)
- **Gradient Boosting Prediction**: $3,967

## 📊 Output

The script generates:

1. **Console Output**:
   - Dataset information and statistics
   - Model training progress with timestamps
   - Performance metrics for each model
   - Best model identification

2. **Results Folder** (automatically created):
   - `model_comparison.png` - Performance comparison across models

## 📁 Project Structure

```
Project2/
│
├── car_sales_data.csv           # Dataset file
├── car_price_prediction.py      # Main script
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore file
├── README.md                     # Project documentation
└── results/                      # Generated visualizations
    └── model_comparison.png
```

## 🔍 Features Explained

### Dataset Features:
- **Manufacturer**: Car manufacturer (Ford, Toyota, BMW, etc.)
- **Model**: Car model name
- **Engine size**: Engine capacity (in liters)
- **Fuel type**: Type of fuel (Petrol, Diesel, Hybrid)
- **Year of manufacture**: Manufacturing year
- **Mileage**: Distance traveled (in kilometers)
- **Price**: Selling price (target variable)

### Engineered Features:
- **Years_of_service**: Calculated from manufacture year (2024 - Year of manufacture)
- **Fuel_type_encoded**: Encoded fuel type using LabelEncoder
- **Log_Mileage**: Log-transformed mileage for better distribution

## 📈 Model Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Lower is better. Measures prediction accuracy.
- **MAE (Mean Absolute Error)**: Lower is better. Average prediction error.
- **R² Score**: Higher is better (max 1.0). Proportion of variance explained.

## 🛠️ Customization

You can modify the script to:
- Adjust hyperparameters for models
- Add new regression models
- Change train/test split ratio (default: 80/20)
- Modify visualization settings
- Adjust current year for years of service calculation

## 📝 Notes

- The dataset should be in CSV format
- The script automatically creates the `results/` directory if it doesn't exist
- Random seed is set to 42 for reproducibility
- The script uses optimized hyperparameters for faster training
- Progress tracking with timestamps for monitoring execution

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available for educational purposes.
