"""
ML Model Module: Trains and predicts expected returns and volatility
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import sys
sys.path.append('..')
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class MLPredictor:
    """Machine Learning predictor for returns and volatility"""
    
    def __init__(self, n_estimators: int = config.N_ESTIMATORS, random_state: int = config.RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        
    def train_models(
        self,
        features: pd.DataFrame,
        future_returns: pd.DataFrame,
        future_volatility: pd.DataFrame
    ) -> Dict:
        """
        Train ML models for each asset
        
        Args:
            features: Feature DataFrame
            future_returns: Target returns
            future_volatility: Target volatility
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training models for {len(future_returns.columns)} assets...")
        
        metrics = {}
        
        for ticker in future_returns.columns:
            logger.info(f"Training models for {ticker}...")
            
            # Get features for this ticker
            ticker_features = [col for col in features.columns if col.startswith(ticker)]
            
            if not ticker_features:
                logger.warning(f"No features found for {ticker}, skipping...")
                continue
            
            X = features[ticker_features]
            
            # Train return prediction model
            y_return = future_returns[ticker]
            return_metrics = self._train_single_model(
                X, y_return, f"{ticker}_return"
            )
            
            # Train volatility prediction model
            y_volatility = future_volatility[ticker]
            volatility_metrics = self._train_single_model(
                X, y_volatility, f"{ticker}_volatility"
            )
            
            metrics[ticker] = {
                'return': return_metrics,
                'volatility': volatility_metrics
            }
        
        logger.info(f"Training completed for {len(self.models)} models")
        return metrics
    
    def _train_single_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str
    ) -> Dict:
        """
        Train a single Random Forest model
        
        Args:
            X: Features
            y: Target
            model_name: Name for the model
            
        Returns:
            Dictionary with performance metrics
        """
        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Remove NaNs
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            logger.warning(f"Insufficient data for {model_name}: {len(X)} samples")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=self.random_state, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'n_samples': len(X)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        metrics['cv_mse'] = -cv_scores.mean()
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.feature_importances[model_name] = dict(zip(X.columns, model.feature_importances_))
        
        logger.info(f"{model_name}: Test R2={metrics['test_r2']:.4f}, Test MSE={metrics['test_mse']:.6f}")
        
        return metrics
    
    def predict(
        self,
        features: pd.DataFrame,
        tickers: list
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Predict expected returns and volatility for next period
        
        Args:
            features: Current features
            tickers: List of tickers to predict
            
        Returns:
            Tuple of (expected_returns, expected_volatility)
        """
        logger.info(f"Making predictions for {len(tickers)} tickers...")
        
        expected_returns = {}
        expected_volatility = {}
        
        for ticker in tickers:
            # Get latest features
            ticker_features = [col for col in features.columns if col.startswith(ticker)]
            
            if not ticker_features:
                logger.warning(f"No features for {ticker}")
                continue
            
            X = features[ticker_features].iloc[-1:].values
            
            # Predict return
            return_model_name = f"{ticker}_return"
            if return_model_name in self.models:
                X_scaled = self.scalers[return_model_name].transform(X)
                pred_return = self.models[return_model_name].predict(X_scaled)[0]
                expected_returns[ticker] = pred_return
            else:
                # Fallback to historical mean
                expected_returns[ticker] = 0.0005  # Small positive return
            
            # Predict volatility
            vol_model_name = f"{ticker}_volatility"
            if vol_model_name in self.models:
                X_scaled = self.scalers[vol_model_name].transform(X)
                pred_vol = self.models[vol_model_name].predict(X_scaled)[0]
                expected_volatility[ticker] = max(pred_vol, 0.01)  # Ensure positive
            else:
                # Fallback to historical volatility
                expected_volatility[ticker] = 0.02
        
        expected_returns_series = pd.Series(expected_returns)
        expected_volatility_series = pd.Series(expected_volatility)
        
        # Annualize predictions
        expected_returns_series = expected_returns_series * config.TRADING_DAYS_PER_YEAR
        expected_volatility_series = expected_volatility_series * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        
        logger.info("Predictions completed")
        logger.info(f"Expected Returns:\n{expected_returns_series}")
        logger.info(f"Expected Volatility:\n{expected_volatility_series}")
        
        return expected_returns_series, expected_volatility_series
    
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance for a model
        
        Args:
            model_name: Name of the model
            top_n: Number of top features
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.feature_importances:
            return pd.DataFrame()
        
        importances = self.feature_importances[model_name]
        importance_df = pd.DataFrame({
            'Feature': list(importances.keys()),
            'Importance': list(importances.values())
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'feature_importances': self.feature_importances
            }, f)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scalers = data['scalers']
            self.feature_importances = data['feature_importances']
        logger.info(f"Models loaded from {filepath}")


if __name__ == "__main__":
    # Test the module
    from data_module import DataFetcher
    from feature_engineering import FeatureEngineer
    from datetime import datetime, timedelta
    
    # Fetch data
    fetcher = DataFetcher()
    tickers = ["AAPL", "MSFT"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    prices = fetcher.fetch_data(tickers, start_date, end_date)
    returns = fetcher.calculate_returns()
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(prices, returns)
    future_returns, future_volatility = engineer.create_target_variables(returns)
    
    # Train models
    predictor = MLPredictor()
    metrics = predictor.train_models(features, future_returns, future_volatility)
    
    print("\nTraining Metrics:")
    for ticker, metric in metrics.items():
        print(f"\n{ticker}:")
        print(f"  Return R2: {metric['return'].get('test_r2', 'N/A')}")
        print(f"  Volatility R2: {metric['volatility'].get('test_r2', 'N/A')}")
    
    # Make predictions
    pred_returns, pred_vol = predictor.predict(features, tickers)
    print(f"\nPredicted Returns:\n{pred_returns}")
    print(f"\nPredicted Volatility:\n{pred_vol}")
