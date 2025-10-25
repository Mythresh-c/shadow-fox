"""
Boston House Price Prediction
A regression model to predict Boston house prices using features such as
crime rates, number of rooms, and other relevant factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(file_path):
    """
    Load the Boston house prices dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        The loaded dataset
    """
    # Skip first row as it contains row count and dimensions
    df = pd.read_csv(file_path, skiprows=1)
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


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
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # Display basic information
    print("\nDataset Shape:", df.shape)
    print("\nColumn Names:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Fill missing values with median for numerical columns
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
    
    print("\n✅ Data preprocessing completed!")
    
    return df


def explore_data(df):
    """
    Explore the dataset with visualizations and statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataset
    """
    print("\n" + "=" * 80)
    print("DATA EXPLORATION")
    print("=" * 80)
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Correlation matrix
    print("\nCorrelation with target variable (MEDV):")
    if 'MEDV' in df.columns:
        correlations = df.corr()['MEDV'].sort_values(ascending=False)
        print(correlations)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution of target variable
    axes[0, 0].hist(df['MEDV'], bins=30, edgecolor='black', color='skyblue')
    axes[0, 0].set_title('Distribution of House Prices (MEDV)')
    axes[0, 0].set_xlabel('Price in $1000s')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Correlation heatmap
    if len(df.columns) > 1:
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, ax=axes[0, 1])
        axes[0, 1].set_title('Correlation Heatmap')
    
    # 3. Scatter plot: RM vs MEDV
    if 'RM' in df.columns:
        axes[1, 0].scatter(df['RM'], df['MEDV'], alpha=0.6, color='coral')
        axes[1, 0].set_title('Number of Rooms vs Price')
        axes[1, 0].set_xlabel('Average Number of Rooms (RM)')
        axes[1, 0].set_ylabel('Price (MEDV)')
    
    # 4. Scatter plot: LSTAT vs MEDV
    if 'LSTAT' in df.columns:
        axes[1, 1].scatter(df['LSTAT'], df['MEDV'], alpha=0.6, color='green')
        axes[1, 1].set_title('Lower Status Population vs Price')
        axes[1, 1].set_xlabel('LSTAT (%)')
        axes[1, 1].set_ylabel('Price (MEDV)')
    
    plt.tight_layout()
    plt.savefig('results/exploratory_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Exploratory analysis saved to results/exploratory_analysis.png")
    
    plt.close()


def train_and_evaluate_models(X, y):
    """
    Train multiple regression models and evaluate their performance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature variables
    y : pd.Series
        Target variable
    
    Returns:
    --------
    dict
        Dictionary containing trained models and their metrics
    """
    print("\n" + "=" * 80)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {name}")
        print(f"{'=' * 60}")
        
        # Train model
        if 'Regression' in name:
            model.fit(X_train_scaled, y_train)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_test_pred
        }
        
        # Print metrics
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Train MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
    
    # Find best model
    best_model_name = min(results, key=lambda x: results[x]['test_rmse'])
    print(f"\n{'=' * 60}")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
    print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"{'=' * 60}")
    
    return results, scaler


def visualize_results(results, y_test):
    """
    Visualize model predictions and performance.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    y_test : pd.Series
        True test values
    """
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Models to plot
    model_names = list(results.keys())
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Predictions vs Actual
        predictions = result['predictions']
        ax.scatter(y_test, predictions, alpha=0.6, s=50)
        ax.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title(f'{name}\nTest R²: {result["test_r2"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('results/model_predictions.png', dpi=300, bbox_inches='tight')
    print("\n✅ Model predictions visualization saved to results/model_predictions.png")
    
    plt.close()
    
    # Model comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    model_names = list(results.keys())
    test_rmse = [results[name]['test_rmse'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_rmse, width, label='RMSE', color='skyblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, test_r2, width, label='R²', color='coral')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE', color='skyblue')
    ax2.set_ylabel('R² Score', color='coral')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Model comparison saved to results/model_comparison.png")
    
    plt.close()


def feature_importance_analysis(results, feature_names):
    """
    Analyze feature importance for tree-based models.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    feature_names : list
        List of feature names
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest Feature Importance
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[0].barh(range(len(importances)), importances[indices])
        axes[0].set_yticks(range(len(importances)))
        axes[0].set_yticklabels([feature_names[i] for i in indices])
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Random Forest - Feature Importance')
        axes[0].invert_yaxis()
    
    # Gradient Boosting Feature Importance
    if 'Gradient Boosting' in results:
        gb_model = results['Gradient Boosting']['model']
        importances = gb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[1].barh(range(len(importances)), importances[indices])
        axes[1].set_yticks(range(len(importances)))
        axes[1].set_yticklabels([feature_names[i] for i in indices])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Gradient Boosting - Feature Importance')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Feature importance analysis saved to results/feature_importance.png")
    
    plt.close()


def main():
    """Main function to run the Boston house price prediction pipeline."""
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "=" * 80)
    print("BOSTON HOUSE PRICE PREDICTION")
    print("=" * 80)
    
    # Load data
    print("\n📁 Loading data...")
    file_path = 'data/boston_house_prices.csv'
    
    # Check if file exists, if not, use the provided path
    if not os.path.exists(file_path):
        file_path = r'C:\Users\Mythresh Chandra\Downloads\boston_house_prices.csv'
    
    df = load_data(file_path)
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    df = preprocess_data(df)
    
    # Explore data
    print("\n🔍 Exploring data...")
    explore_data(df)
    
    # Prepare features and target
    print("\n🎯 Preparing features and target...")
    
    # Define target variable
    target = 'MEDV'
    
    # Define features (all except target)
    features = df.columns.drop([target])
    
    X = df[features]
    y = df[target]
    
    print(f"\nFeatures: {list(features)}")
    print(f"Target: {target}")
    print(f"Number of samples: {len(X)}")
    
    # Train and evaluate models
    print("\n🤖 Training models...")
    results, scaler = train_and_evaluate_models(X, y)
    
    # Visualize results
    print("\n📊 Creating visualizations...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    visualize_results(results, y_test)
    
    # Feature importance analysis
    print("\n📈 Analyzing feature importance...")
    feature_importance_analysis(results, list(features))
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nResults saved in the 'results' directory:")
    print("  - exploratory_analysis.png")
    print("  - model_predictions.png")
    print("  - model_comparison.png")
    print("  - feature_importance.png")
    print("=" * 80)


if __name__ == "__main__":
    main()

