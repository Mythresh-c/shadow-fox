"""
Boston House Price Prediction
==============================

This script implements a machine learning solution to predict Boston house prices
using various features such as number of rooms, crime rates, and other relevant factors.

Author: ShadowFox Team
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class BostonHousePricePredictor:
    """
    A comprehensive class for predicting Boston house prices using multiple algorithms.
    """
    
    def __init__(self, data_path='boston_house_prices.csv'):
        """Initialize the predictor with data path."""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and preprocess the Boston housing dataset."""
        print("Loading data...")
        
        # Read CSV file, skip the first row (contains 506,13,,,,,,,)
        self.data = pd.read_csv(self.data_path, skiprows=1)
        
        # Check if data was loaded correctly
        if self.data.shape[0] == 0:
            # If no data, try reading without skiprows
            self.data = pd.read_csv(self.data_path)
            
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nFirst few rows:")
        print(self.data.head())
        print("\nDataset info:")
        print(self.data.info())
        print("\nBasic statistics:")
        print(self.data.describe())
        
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Check for missing values
        print("\n1. Missing Values:")
        print(self.data.isnull().sum())
        
        # Check for duplicate rows
        print(f"\n2. Duplicate rows: {self.data.duplicated().sum()}")
        
        # Correlation analysis
        print("\n3. Correlation with target variable (MEDV):")
        if 'MEDV' in self.data.columns:
            correlations = self.data.corr()['MEDV'].sort_values(ascending=False)
            print(correlations)
            
            # Plot correlation heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Heatmap of Features', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")
            plt.close()
        
        # Distribution of target variable
        if 'MEDV' in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data['MEDV'], kde=True, bins=30)
            plt.title('Distribution of House Prices (MEDV)', fontsize=14, fontweight='bold')
            plt.xlabel('Price (in $1000s)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
            print("Price distribution plot saved as 'price_distribution.png'")
            plt.close()
    
    def prepare_data(self):
        """Prepare data for modeling."""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        # Separate features and target
        self.y = self.data['MEDV']
        self.X = self.data.drop('MEDV', axis=1)
        
        print(f"\nFeatures shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"\nFeatures: {list(self.X.columns)}")
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\nData preparation completed!")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self):
        """Initialize multiple regression models for comparison."""
        print("\n" + "="*60)
        print("INITIALIZING MODELS")
        print("="*60)
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        print("\nModels initialized:")
        for name in self.models.keys():
            print(f"  - {name}")
        
    def train_models(self):
        """Train all models and evaluate performance."""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train_scaled)
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # Store results
            result = {
                'Model': name,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R²': train_r2,
                'Test R²': test_r2
            }
            results.append(result)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='r2')
            
            print(f"  Train RMSE: {train_rmse:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store model
            self.models[name] = model
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(self.results_df.to_string(index=False))
        
        return self.results_df
    
    def select_best_model(self):
        """Select the best performing model."""
        print("\n" + "="*60)
        print("SELECTING BEST MODEL")
        print("="*60)
        
        # Sort by Test R² score
        best_model_name = self.results_df.loc[self.results_df['Test R²'].idxmax(), 'Model']
        best_model = self.models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Test R² Score: {self.results_df['Test R²'].max():.4f}")
        print(f"Test RMSE: {self.results_df.loc[self.results_df['Test R²'].idxmax(), 'Test RMSE']:.4f}")
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        
        return best_model
    
    def visualize_predictions(self):
        """Visualize predictions vs actual values for the best model."""
        print("\n" + "="*60)
        print("VISUALIZING PREDICTIONS")
        print("="*60)
        
        # Make predictions with best model
        y_train_pred = self.best_model.predict(self.X_train_scaled)
        y_test_pred = self.best_model.predict(self.X_test_scaled)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set
        axes[0].scatter(self.y_train, y_train_pred, alpha=0.5)
        axes[0].plot([self.y_train.min(), self.y_train.max()], 
                     [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price', fontsize=12)
        axes[0].set_ylabel('Predicted Price', fontsize=12)
        axes[0].set_title(f'{self.best_model_name} - Training Set', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Testing set
        axes[1].scatter(self.y_test, y_test_pred, alpha=0.5, color='green')
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Price', fontsize=12)
        axes[1].set_ylabel('Predicted Price', fontsize=12)
        axes[1].set_title(f'{self.best_model_name} - Testing Set', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
        print("Predictions comparison plot saved as 'predictions_comparison.png'")
        plt.close()
        
    def feature_importance_analysis(self):
        """Analyze and visualize feature importance for tree-based models."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Check if the best model is tree-based
        tree_models = ['Random Forest', 'Gradient Boosting']
        
        if self.best_model_name in tree_models:
            # Get feature importances
            importances = self.best_model.feature_importances_
            feature_names = self.X.columns
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            print(importance_df.to_string(index=False))
            
            # Visualize feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
            plt.title(f'Feature Importance - {self.best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance plot saved as 'feature_importance.png'")
            plt.close()
        else:
            print(f"\nFeature importance not available for {self.best_model_name}.")
            print("Feature importance is available for tree-based models only.")
    
    def generate_report(self):
        """Generate a comprehensive report of the analysis."""
        print("\n" + "="*60)
        print("GENERATING FINAL REPORT")
        print("="*60)
        
        report = f"""
BOSTON HOUSE PRICE PREDICTION - FINAL REPORT
============================================

DATASET INFORMATION
-------------------
Total Samples: {len(self.data)}
Features: {len(self.X.columns)}
Target Variable: MEDV (Median House Price in $1000s)

DATA SPLIT
----------
Training Set: {len(self.X_train)} samples (80%)
Testing Set: {len(self.X_test)} samples (20%)

MODEL COMPARISON
----------------
{self.results_df.to_string(index=False)}

BEST MODEL
----------
Model: {self.best_model_name}

Performance Metrics:
- Test R² Score: {self.results_df['Test R²'].max():.4f}
- Test RMSE: {self.results_df.loc[self.results_df['Test R²'].idxmax(), 'Test RMSE']:.4f}
- Test MAE: {self.results_df.loc[self.results_df['Test R²'].idxmax(), 'Test MAE']:.4f}

CONCLUSION
----------
The {self.best_model_name} model performed best with an R² score of {self.results_df['Test R²'].max():.4f},
indicating that approximately {self.results_df['Test R²'].max()*100:.2f}% of the variance 
in house prices can be explained by the selected features.

Generated visualizations:
1. correlation_heatmap.png - Feature correlations
2. price_distribution.png - Distribution of house prices
3. predictions_comparison.png - Actual vs Predicted prices
4. feature_importance.png - Most important features (if applicable)

============================================
Report generated successfully!
"""
        
        print(report)
        
        # Save report to file
        with open('prediction_report.txt', 'w') as f:
            f.write(report)
        
        print("Report saved to 'prediction_report.txt'")
        
        return report

def main():
    """Main function to run the Boston House Price Prediction pipeline."""
    print("="*60)
    print("BOSTON HOUSE PRICE PREDICTION PROJECT")
    print("="*60)
    
    # Initialize predictor
    predictor = BostonHousePricePredictor('boston_house_prices.csv')
    
    # Load data
    predictor.load_data()
    
    # Explore data
    predictor.explore_data()
    
    # Prepare data
    predictor.prepare_data()
    
    # Initialize models
    predictor.initialize_models()
    
    # Train models
    predictor.train_models()
    
    # Select best model
    predictor.select_best_model()
    
    # Visualize predictions
    predictor.visualize_predictions()
    
    # Feature importance analysis
    predictor.feature_importance_analysis()
    
    # Generate report
    predictor.generate_report()
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
