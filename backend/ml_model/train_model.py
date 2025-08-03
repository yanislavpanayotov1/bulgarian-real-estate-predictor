import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Try to import XGBoost, but don't fail if it's not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEstatePricePredictor:
    """
    Machine Learning pipeline for Bulgarian real estate price prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, file_path: str = "../data/raw_properties.csv") -> pd.DataFrame:
        """Load and return property data"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} properties from {file_path}")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            # Create sample data for demonstration
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration purposes"""
        logger.info("Creating sample data for demonstration...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data that mimics Bulgarian real estate
        cities = ['Sofia', 'Plovdiv', 'Varna', 'Burgas', 'Stara Zagora']
        neighborhoods = ['Center', 'Mladost', 'Lyulin', 'Vitosha', 'Boyana', 'Lozenets']
        
        # Generate realistic sizes first
        sizes = np.random.normal(80, 30, n_samples)
        sizes = np.clip(sizes, 25, 180)  # Ensure reasonable sizes
        
        # Generate base prices that correlate with size (realistic Bulgarian prices)
        # Base price per sqm: 800-2000 BGN/sqm depending on location
        base_price_per_sqm = np.random.uniform(800, 2000, n_samples)
        base_prices = sizes * base_price_per_sqm
        
        data = {
            'price': base_prices,
            'size': sizes,
            'rooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.35, 0.2, 0.05]),
            'floor': np.random.randint(1, 15, n_samples),
            'year_built': np.random.randint(1960, 2024, n_samples),
            'city': np.random.choice(cities, n_samples),
            'neighborhood': np.random.choice(neighborhoods, n_samples),
            'coordinates': [{'lat': 42.7 + np.random.normal(0, 0.1), 
                           'lng': 23.3 + np.random.normal(0, 0.1)} for _ in range(n_samples)],
            'features': [['parking', 'balcony'] if np.random.random() > 0.5 else ['elevator'] 
                        for _ in range(n_samples)]
        }
        
        df = pd.DataFrame(data)
        
        # Add some realistic correlations
        df.loc[df['city'] == 'Sofia', 'price'] *= 1.4  # Sofia is more expensive
        df.loc[df['neighborhood'] == 'Center', 'price'] *= 1.2  # Center is more expensive
        
        # Adjust for property age (newer properties are more expensive)
        age_factor = 1 + (2024 - df['year_built']) * -0.005  # 0.5% per year
        df['price'] = df['price'] * age_factor
        
        # Add some random variation
        df['price'] = df['price'] * np.random.uniform(0.9, 1.1, n_samples)
        
        # Clean unrealistic values
        df = df[(df['size'] > 20) & (df['size'] < 200)]
        df = df[(df['price'] > 20000) & (df['price'] < 500000)]  # 20k-500k BGN is realistic range
        
        # Reset index after filtering
        df = df.reset_index(drop=True)
        
        logger.info(f"Created {len(df)} sample properties")
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from raw data"""
        logger.info("Performing feature engineering...")
        
        df = df.copy()
        
        # Extract coordinates if they exist
        if 'coordinates' in df.columns:
            try:
                df['latitude'] = df['coordinates'].apply(
                    lambda x: eval(x)['lat'] if isinstance(x, str) else x.get('lat', None) if isinstance(x, dict) else None
                )
                df['longitude'] = df['coordinates'].apply(
                    lambda x: eval(x)['lng'] if isinstance(x, str) else x.get('lng', None) if isinstance(x, dict) else None
                )
            except:
                # Default coordinates for Bulgaria
                df['latitude'] = 42.7339
                df['longitude'] = 25.4858
        else:
            df['latitude'] = 42.7339
            df['longitude'] = 25.4858
        
        # Price per square meter (only if price column exists - not in prediction mode)
        if 'price' in df.columns:
            df['price_per_sqm'] = df['price'] / df['size']
        
        # Age of property
        current_year = 2024
        df['property_age'] = current_year - df['year_built']
        
        # Room density (rooms per 100 sqm)
        df['room_density'] = (df['rooms'] / df['size']) * 100
        
        # City encoding with price-based ordering (only during training)
        if 'price' in df.columns and 'city' in df.columns:
            city_price_map = df.groupby('city')['price'].median().to_dict()
            df['city_price_rank'] = df['city'].map(city_price_map)
        else:
            # Use default city rankings for prediction
            default_city_ranks = {
                'Sofia': 150000, 'Plovdiv': 100000, 'Varna': 120000, 
                'Burgas': 110000, 'Stara Zagora': 80000
            }
            df['city_price_rank'] = df['city'].map(default_city_ranks).fillna(100000)
        
        # Feature count
        if 'features' in df.columns:
            df['feature_count'] = df['features'].apply(
                lambda x: len(eval(x)) if isinstance(x, str) else len(x) if isinstance(x, list) else 0
            )
        else:
            df['feature_count'] = 0
        
        # Floor category
        df['floor_category'] = pd.cut(df['floor'], bins=[0, 1, 5, 10, float('inf')], 
                                    labels=['Ground', 'Low', 'Mid', 'High'])
        
        # Size category
        df['size_category'] = pd.cut(df['size'], bins=[0, 50, 80, 120, float('inf')], 
                                   labels=['Small', 'Medium', 'Large', 'XLarge'])
        
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training"""
        logger.info("Preparing features for model training...")
        
        # Select numeric features
        numeric_features = ['size', 'rooms', 'floor', 'year_built', 'latitude', 'longitude', 
                          'property_age', 'room_density', 'city_price_rank', 'feature_count']
        
        # Select categorical features
        categorical_features = ['city', 'neighborhood', 'floor_category', 'size_category']
        
        # Prepare feature matrix
        X = df[numeric_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in df.columns:
                # Handle NaN values by converting to string first
                feature_values = df[feature].astype(str).fillna('Unknown')
                
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    X[f'{feature}_encoded'] = self.encoders[feature].fit_transform(feature_values)
                else:
                    # Handle unknown categories in transform
                    try:
                        X[f'{feature}_encoded'] = self.encoders[feature].transform(feature_values)
                    except ValueError:
                        # If there are unknown categories, fit again
                        X[f'{feature}_encoded'] = self.encoders[feature].fit_transform(feature_values)
        
        # Target variable (only if price column exists - not in prediction mode)
        if 'price' in df.columns:
            y = df['price'].values
            logger.info(f"Features prepared. X shape: {X.shape}, y shape: {y.shape}")
            return X.values, y
        else:
            logger.info(f"Features prepared for prediction. X shape: {X.shape}")
            return X.values, None
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train multiple models and compare performance"""
        logger.info("Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'Linear Regression': LinearRegression()
        }
        
        # Try to add XGBoost if available
        if XGBOOST_AVAILABLE:
            try:
                models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                logger.info("XGBoost added to model list")
            except Exception as e:
                logger.warning(f"XGBoost failed to initialize: {str(e)}")
                logger.info("Continuing without XGBoost...")
        else:
            logger.info("XGBoost not installed, continuing without it...")
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Use scaled features for linear models
            if name in ['Ridge Regression', 'Linear Regression']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_model)
            y_pred_test = model.predict(X_test_model)
            
            # Metrics
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_score': cross_val_score(model, X_train_model, y_train, cv=5, 
                                          scoring='neg_mean_absolute_error').mean()
            }
            
            # Store model
            self.models[name] = model
            self.model_scores[name] = results[name]
            
            logger.info(f"{name} - Test MAE: {test_mae:.2f}, Test R²: {test_r2:.3f}")
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        logger.info(f"Best model: {best_model_name}")
        
        return results
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Optimize hyperparameters for the best model"""
        if self.best_model_name == 'Random Forest':
            logger.info("Optimizing Random Forest hyperparameters...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
            )
            
            grid_search.fit(X, y)
            self.best_model = grid_search.best_estimator_
            self.models[self.best_model_name] = self.best_model
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
    
    def analyze_feature_importance(self, X: pd.DataFrame):
        """Analyze and visualize feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
            importance = self.best_model.feature_importances_
            
            self.feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 most important features:")
            for feature, importance in sorted_features[:10]:
                logger.info(f"{feature}: {importance:.4f}")
            
            return sorted_features
        
        return None
    
    def save_model(self, output_dir: str = "../models"):
        """Save trained model and preprocessing objects"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, output_path / "best_model.joblib")
        
        # Save scalers and encoders
        joblib.dump(self.scalers, output_path / "scalers.joblib")
        joblib.dump(self.encoders, output_path / "encoders.joblib")
        
        # Save feature importance
        if self.feature_importance:
            joblib.dump(self.feature_importance, output_path / "feature_importance.joblib")
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'model_scores': self.model_scores,
            'feature_names': list(self.feature_importance.keys()) if self.feature_importance else []
        }
        
        joblib.dump(metadata, output_path / "model_metadata.joblib")
        
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_dir: str = "../models"):
        """Load trained model and preprocessing objects"""
        model_path = Path(model_dir)
        
        try:
            self.best_model = joblib.load(model_path / "best_model.joblib")
            self.scalers = joblib.load(model_path / "scalers.joblib")
            self.encoders = joblib.load(model_path / "encoders.joblib")
            
            metadata = joblib.load(model_path / "model_metadata.joblib")
            self.best_model_name = metadata['best_model_name']
            self.model_scores = metadata['model_scores']
            
            if (model_path / "feature_importance.joblib").exists():
                self.feature_importance = joblib.load(model_path / "feature_importance.joblib")
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make price prediction for a single property"""
        if not self.best_model:
            raise ValueError("Model not trained or loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([property_data])
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        # Scale if needed
        if self.best_model_name in ['Ridge Regression', 'Linear Regression']:
            X = self.scalers['standard'].transform(X)
        
        # Predict
        prediction = self.best_model.predict(X)[0]
        
        # Calculate confidence interval (rough estimate)
        if hasattr(self.best_model, 'estimators_'):
            # For ensemble methods
            predictions = [estimator.predict(X)[0] for estimator in self.best_model.estimators_]
            std_dev = np.std(predictions)
            confidence_lower = prediction - 1.96 * std_dev
            confidence_upper = prediction + 1.96 * std_dev
        else:
            # Simple estimate
            std_dev = prediction * 0.15  # Assume 15% uncertainty
            confidence_lower = prediction - 1.96 * std_dev
            confidence_upper = prediction + 1.96 * std_dev
        
        return {
            'predicted_price': float(prediction),
            'confidence_lower': float(max(0, confidence_lower)),
            'confidence_upper': float(confidence_upper),
            'model_used': self.best_model_name
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive model performance report"""
        report = "\n" + "="*50 + "\n"
        report += "REAL ESTATE PRICE PREDICTION MODEL REPORT\n"
        report += "="*50 + "\n\n"
        
        if self.model_scores:
            report += "MODEL PERFORMANCE COMPARISON:\n"
            report += "-"*30 + "\n"
            
            for name, scores in self.model_scores.items():
                report += f"\n{name}:\n"
                report += f"  Test MAE: {scores['test_mae']:.2f} BGN\n"
                report += f"  Test RMSE: {scores['test_rmse']:.2f} BGN\n"
                report += f"  Test R²: {scores['test_r2']:.3f}\n"
                report += f"  CV Score: {-scores['cv_score']:.2f}\n"
        
        if self.best_model_name:
            report += f"\nBEST MODEL: {self.best_model_name}\n"
            report += "-"*20 + "\n"
            
            best_scores = self.model_scores[self.best_model_name]
            report += f"Mean Absolute Error: {best_scores['test_mae']:.2f} BGN\n"
            report += f"Root Mean Square Error: {best_scores['test_rmse']:.2f} BGN\n"
            report += f"R² Score: {best_scores['test_r2']:.3f}\n"
        
        if self.feature_importance:
            report += "\nTOP 10 FEATURE IMPORTANCE:\n"
            report += "-"*25 + "\n"
            
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                report += f"{i:2d}. {feature}: {importance:.4f}\n"
        
        return report

def main():
    """Main training pipeline"""
    predictor = RealEstatePricePredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Feature engineering
    df = predictor.feature_engineering(df)
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Optimize best model
    predictor.optimize_hyperparameters(X, y)
    
    # Analyze feature importance
    feature_df = df[[col for col in df.columns if col != 'price']]
    predictor.analyze_feature_importance(feature_df.select_dtypes(include=[np.number]))
    
    # Save model
    predictor.save_model()
    
    # Generate and print report
    report = predictor.generate_report()
    print(report)
    
    # Test prediction
    sample_property = {
        'size': 75,
        'rooms': 2,
        'floor': 3,
        'year_built': 2010,
        'city': 'Sofia',
        'neighborhood': 'Center',
        'coordinates': {'lat': 42.7, 'lng': 23.3},
        'features': ['parking', 'balcony']
    }
    
    prediction = predictor.predict(sample_property)
    print(f"\nSample Prediction:")
    print(f"Property: 75m², 2 rooms, 3rd floor, built 2010, Sofia Center")
    print(f"Predicted Price: {prediction['predicted_price']:.2f} BGN")
    print(f"Confidence Interval: {prediction['confidence_lower']:.2f} - {prediction['confidence_upper']:.2f} BGN")

if __name__ == "__main__":
    main() 