"""
Example: Predict Car Price
This script demonstrates how to predict car prices using the trained model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pickle
import os

def load_and_prepare_data(file_path='car_sales_data.csv'):
    """Load and prepare data for training."""
    df = pd.read_csv(file_path)
    
    # Feature engineering
    current_year = 2024
    df['Years_of_service'] = current_year - df['Year of manufacture']
    
    fuel_encoder = LabelEncoder()
    df['Fuel_type_encoded'] = fuel_encoder.fit_transform(df['Fuel type'])
    
    df['Log_Mileage'] = np.log1p(df['Mileage'])
    
    return df, fuel_encoder

def train_model(df):
    """Train a model for predictions."""
    # Prepare features
    features_to_drop = ['Manufacturer', 'Model', 'Fuel type', 'Year of manufacture']
    features_to_use = [col for col in df.columns if col not in features_to_drop and col != 'Price']
    
    X = df[features_to_use]
    y = df['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model (using Gradient Boosting - best performing model)
    model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    print(f"Model trained! R² Score: {model.score(X_test, y_test):.4f}")
    
    return model, X_train.columns.tolist()

def predict_price(manufacturer, model_name, engine_size, fuel_type, year, mileage, 
                  trained_model, feature_names, fuel_encoder):
    """
    Predict car price based on features.
    
    Parameters:
    -----------
    manufacturer : str - Car manufacturer (not used in prediction, but shown for reference)
    model_name : str - Car model (not used in prediction, but shown for reference)
    engine_size : float - Engine size in liters
    fuel_type : str - Fuel type (Petrol, Diesel, Hybrid)
    year : int - Year of manufacture
    mileage : int - Mileage in kilometers
    trained_model : trained model object
    feature_names : list of feature column names
    fuel_encoder : fitted LabelEncoder for fuel type
    
    Returns:
    --------
    float : Predicted price in dollars
    """
    # Create feature vector
    current_year = 2024
    years_of_service = current_year - year
    fuel_encoded = fuel_encoder.transform([fuel_type])[0]
    log_mileage = np.log1p(mileage)
    
    # Create feature array in the same order as training
    features = np.array([[engine_size, mileage, years_of_service, fuel_encoded, log_mileage]])
    
    # Make prediction
    predicted_price = trained_model.predict(features)[0]
    
    return predicted_price

# Example usage
if __name__ == "__main__":
    print("="*80)
    print("CAR PRICE PREDICTION EXAMPLE")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading data...")
    df, fuel_encoder = load_and_prepare_data()
    print(f"Data loaded: {len(df)} cars")
    
    # Train model
    print("\nTraining model...")
    model, feature_names = train_model(df)
    
    # Example 1: Predict price for a specific car
    print("\n" + "="*80)
    print("EXAMPLE 1: Predicting Price for a Car")
    print("="*80)
    
    # Car details from the dataset (Ford Fiesta)
    manufacturer = "Ford"
    model_name = "Fiesta"
    engine_size = 1.0
    fuel_type = "Petrol"
    year = 2002
    mileage = 127300
    
    print(f"\nCar Details:")
    print(f"  Manufacturer: {manufacturer}")
    print(f"  Model: {model_name}")
    print(f"  Engine Size: {engine_size} L")
    print(f"  Fuel Type: {fuel_type}")
    print(f"  Year: {year}")
    print(f"  Mileage: {mileage:,} km")
    
    # Predict price
    predicted_price = predict_price(
        manufacturer=manufacturer,
        model_name=model_name,
        engine_size=engine_size,
        fuel_type=fuel_type,
        year=year,
        mileage=mileage,
        trained_model=model,
        feature_names=feature_names,
        fuel_encoder=fuel_encoder
    )
    
    print(f"\nPredicted Price: ${predicted_price:,.2f}")
    
    # Find actual price for comparison
    actual_car = df[(df['Manufacturer'] == manufacturer) & 
                    (df['Model'] == model_name) & 
                    (df['Engine size'] == engine_size) &
                    (df['Fuel type'] == fuel_type) &
                    (df['Year of manufacture'] == year) &
                    (df['Mileage'] == mileage)]
    
    if not actual_car.empty:
        actual_price = actual_car['Price'].iloc[0]
        print(f"Actual Price: ${actual_price:,.2f}")
        error = abs(predicted_price - actual_price)
        error_percent = (error / actual_price) * 100
        print(f"Error: ${error:,.2f} ({error_percent:.2f}%)")
    
    # Example 2: Predict price for a new car
    print("\n" + "="*80)
    print("EXAMPLE 2: Predicting Price for a New Car")
    print("="*80)
    
    # New car details
    manufacturer2 = "Toyota"
    model_name2 = "Camry"
    engine_size2 = 2.5
    fuel_type2 = "Hybrid"
    year2 = 2020
    mileage2 = 50000
    
    print(f"\nCar Details:")
    print(f"  Manufacturer: {manufacturer2}")
    print(f"  Model: {model_name2}")
    print(f"  Engine Size: {engine_size2} L")
    print(f"  Fuel Type: {fuel_type2}")
    print(f"  Year: {year2}")
    print(f"  Mileage: {mileage2:,} km")
    
    # Predict price
    predicted_price2 = predict_price(
        manufacturer=manufacturer2,
        model_name=model_name2,
        engine_size=engine_size2,
        fuel_type=fuel_type2,
        year=year2,
        mileage=mileage2,
        trained_model=model,
        feature_names=feature_names,
        fuel_encoder=fuel_encoder
    )
    
    print(f"\nPredicted Price: ${predicted_price2:,.2f}")
    
    # Example 3: Predict price for an older car
    print("\n" + "="*80)
    print("EXAMPLE 3: Predicting Price for an Older Car")
    print("="*80)
    
    manufacturer3 = "BMW"
    model_name3 = "Z4"
    engine_size3 = 2.0
    fuel_type3 = "Petrol"
    year3 = 2016
    mileage3 = 80000
    
    print(f"\nCar Details:")
    print(f"  Manufacturer: {manufacturer3}")
    print(f"  Model: {model_name3}")
    print(f"  Engine Size: {engine_size3} L")
    print(f"  Fuel Type: {fuel_type3}")
    print(f"  Year: {year3}")
    print(f"  Mileage: {mileage3:,} km")
    
    # Predict price
    predicted_price3 = predict_price(
        manufacturer=manufacturer3,
        model_name=model_name3,
        engine_size=engine_size3,
        fuel_type=fuel_type3,
        year=year3,
        mileage=mileage3,
        trained_model=model,
        feature_names=feature_names,
        fuel_encoder=fuel_encoder
    )
    
    print(f"\nPredicted Price: ${predicted_price3:,.2f}")
    
    print("\n" + "="*80)
    print("All predictions complete!")
    print("="*80)
