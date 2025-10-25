"""
Car Selling Price Prediction
A machine learning model to predict car selling prices based on various features
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import sys
import time
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def log(message):
    """Print message with timestamp for progress tracking"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")
    sys.stdout.flush()


def load_data(file_path):
    """
    Load the car sales dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        The loaded dataset
    """
    df = pd.read_csv(file_path)
    return df


def explore_data(df):
    """Explore the dataset with basic statistics."""
    log("Exploring data...")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample data:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nStatistics:\n{df.describe()}")


def preprocess_data(df):
    """
    Preprocess the data: handle missing values, encode categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataset
    """
    log("Preprocessing data...")
    df_processed = df.copy()
    
    # Feature engineering
    current_year = 2024
    if 'Year of manufacture' in df_processed.columns:
        df_processed['Years_of_service'] = current_year - df_processed['Year of manufacture']
    
    if 'Fuel type' in df_processed.columns:
        fuel_encoder = LabelEncoder()
        df_processed['Fuel_type_encoded'] = fuel_encoder.fit_transform(df_processed['Fuel type'])
    
    if 'Mileage' in df_processed.columns:
        df_processed['Log_Mileage'] = np.log1p(df_processed['Mileage'])
    
    log("Preprocessing completed!")
    return df_processed


def prepare_features(df):
    """
    Prepare features for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataset
    
    Returns:
    --------
    X, y : Feature matrix and target vector
    feature_names : List of feature names
    """
    features_to_drop = ['Manufacturer', 'Model', 'Fuel type', 'Year of manufacture']
    features_to_use = [col for col in df.columns if col not in features_to_drop and col != 'Price']
    
    X = df[features_to_use]
    y = df['Price']
    
    log(f"Features: {list(X.columns)}")
    return X, y, list(X.columns)


def train_and_evaluate_models(X, y, feature_names):
    """
    Train multiple regression models and evaluate their performance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature variables
    y : pd.Series
        Target variable
    feature_names : list
        List of feature names
    
    Returns:
    --------
    dict
        Dictionary containing trained models and their metrics
    """
    log("Starting model training...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with optimized parameters
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0, max_iter=1000),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)
    }
    
    results = {}
    
    for name, model in models.items():
        log(f"Training {name}...")
        
        try:
            if 'Regression' in name:
                model.fit(X_train_scaled, y_train)
                y_test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            results[name] = {
                'model': model,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'predictions': y_test_pred
            }
            
            log(f"{name} - RMSE: {test_rmse:.2f}, R2: {test_r2:.4f}")
            
        except Exception as e:
            log(f"Error training {name}: {str(e)}")
            continue
    
    # Find best model
    if results:
        best_model = min(results, key=lambda x: results[x]['test_rmse'])
        log(f"Best Model: {best_model}")
        log(f"Best Model RMSE: {results[best_model]['test_rmse']:.2f}")
        log(f"Best Model R2: {results[best_model]['test_r2']:.4f}")
    
    return results, scaler, X_test, y_test


def create_visualizations(results, y_test):
    """
    Create visualizations for model comparison.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    y_test : pd.Series
        True test values
    """
    log("Creating visualizations...")
    
    import os
    os.makedirs('results', exist_ok=True)
    
    # Model comparison
    if len(results) > 0:
        model_names = list(results.keys())
        test_rmse = [results[name]['test_rmse'] for name in model_names]
        test_r2 = [results[name]['test_r2'] for name in model_names]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, test_rmse, width, label='RMSE', color='skyblue')
        ax2 = ax.twinx()
        ax2.bar(x + width/2, test_r2, width, label='R²', color='coral')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('RMSE', color='skyblue', fontsize=12)
        ax2.set_ylabel('R² Score', color='coral', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        log("Visualization saved to results/model_comparison.png")
    else:
        log("No results to visualize")


def predict_car_price(manufacturer, model_name, engine_size, fuel_type, year, mileage, trained_results, scaler, fuel_encoder):
    """
    Predict the price of a car based on its features.
    
    Parameters:
    -----------
    manufacturer : str
        Car manufacturer
    model_name : str
        Car model name
    engine_size : float
        Engine size in liters
    fuel_type : str
        Fuel type (Petrol, Diesel, Hybrid)
    year : int
        Year of manufacture
    mileage : int
        Mileage in kilometers
    trained_results : dict
        Dictionary containing trained models
    scaler : StandardScaler
        Fitted scaler
    fuel_encoder : LabelEncoder
        Fitted fuel type encoder
    
    Returns:
    --------
    dict
        Dictionary with predictions from each model
    """
    # Create feature vector
    current_year = 2024
    years_of_service = current_year - year
    fuel_encoded = fuel_encoder.transform([fuel_type])[0]
    log_mileage = np.log1p(mileage)
    
    # Create feature array
    features = np.array([[engine_size, mileage, years_of_service, fuel_encoded, log_mileage]])
    
    predictions = {}
    
    for name, result in trained_results.items():
        model = result['model']
        
        try:
            if 'Regression' in name:
                # Scale features for regression models
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)[0]
            else:
                pred = model.predict(features)[0]
            
            predictions[name] = round(pred, 2)
        except Exception as e:
            log(f"Error predicting with {name}: {str(e)}")
            continue
    
    return predictions


def example_prediction(df_processed, trained_results, scaler):
    """Run an example prediction using a car from the dataset."""
    log("\n" + "="*80)
    log("EXAMPLE PREDICTION")
    log("="*80)
    
    # Get a sample car from the dataset
    sample_car = df_processed.iloc[0]
    
    print("\nCar Details:")
    print(f"Manufacturer: {sample_car['Manufacturer']}")
    print(f"Model: {sample_car['Model']}")
    print(f"Engine Size: {sample_car['Engine size']} L")
    print(f"Fuel Type: {sample_car['Fuel type']}")
    print(f"Year of Manufacture: {int(sample_car['Year of manufacture'])}")
    print(f"Mileage: {int(sample_car['Mileage'])} km")
    print(f"Actual Price: ${int(sample_car['Price'])}")
    
    # Prepare fuel encoder
    fuel_encoder = LabelEncoder()
    fuel_encoder.fit(df_processed['Fuel type'])
    
    # Make prediction
    predictions = predict_car_price(
        manufacturer=sample_car['Manufacturer'],
        model_name=sample_car['Model'],
        engine_size=sample_car['Engine size'],
        fuel_type=sample_car['Fuel type'],
        year=int(sample_car['Year of manufacture']),
        mileage=int(sample_car['Mileage']),
        trained_results=trained_results,
        scaler=scaler,
        fuel_encoder=fuel_encoder
    )
    
    print("\n" + "="*80)
    print("PREDICTED PRICES:")
    print("="*80)
    for model_name, price in predictions.items():
        error = abs(price - sample_car['Price'])
        error_percent = (error / sample_car['Price']) * 100
        print(f"{model_name}: ${price:.2f} (Error: ${error:.2f}, {error_percent:.2f}%)")
    
    print(f"\nActual Price: ${sample_car['Price']:.2f}")
    
    # Best prediction
    if predictions:
        best_model = min(predictions, key=lambda x: abs(predictions[x] - sample_car['Price']))
        print(f"\nBest Prediction: {best_model} with ${predictions[best_model]:.2f}")
    
    log("="*80)


def main():
    """Main function to run the car price prediction pipeline."""
    print("="*80)
    print("CAR SELLING PRICE PREDICTION")
    print("="*80)
    
    try:
        # Load data
        log("Loading data...")
        df = load_data('car_sales_data.csv')
        log(f"Data loaded: {df.shape}")
        
        # Explore data
        explore_data(df)
        
        # Preprocess
        df_processed = preprocess_data(df)
        
        # Prepare features
        X, y, feature_names = prepare_features(df_processed)
        log(f"Features: {len(feature_names)}")
        
        # Train models
        results, scaler, X_test, y_test = train_and_evaluate_models(X, y, feature_names)
        
        # Create visualizations
        create_visualizations(results, y_test)
        
        # Example prediction
        example_prediction(df_processed, results, scaler)
        
        log("Analysis complete!")
        print("="*80)
        
    except Exception as e:
        log(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
