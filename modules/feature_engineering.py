"""
Feature Engineering Module: Creates features for ML models
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple
import sys
sys.path.append('..')
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for ML-based return prediction"""
    
    def __init__(self):
        self.feature_columns = []
        
    def create_features(
        self, 
        prices: pd.DataFrame,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create technical and statistical features for each asset
        
        Args:
            prices: Historical price data
            returns: Historical returns data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating features...")
        
        features_list = []
        
        for ticker in prices.columns:
            ticker_features = self._create_ticker_features(
                prices[ticker],
                returns[ticker],
                ticker
            )
            features_list.append(ticker_features)
        
        # Combine all features
        features = pd.concat(features_list, axis=1)
        features = features.dropna()
        
        self.feature_columns = features.columns.tolist()
        logger.info(f"Created {len(self.feature_columns)} features")
        
        return features
    
    def _create_ticker_features(
        self, 
        price_series: pd.Series,
        return_series: pd.Series,
        ticker: str
    ) -> pd.DataFrame:
        """
        Create features for a single ticker
        
        Args:
            price_series: Price time series
            return_series: Return time series
            ticker: Ticker symbol
            
        Returns:
            DataFrame with features for the ticker
        """
        features = pd.DataFrame(index=price_series.index)
        
        # 1. Moving Averages
        for period in config.LOOKBACK_PERIODS:
            features[f'{ticker}_MA_{period}'] = price_series.rolling(window=period).mean()
            features[f'{ticker}_MA_ratio_{period}'] = price_series / features[f'{ticker}_MA_{period}']
        
        # 2. Momentum indicators
        features[f'{ticker}_momentum_5'] = price_series.pct_change(5)
        features[f'{ticker}_momentum_10'] = price_series.pct_change(10)
        features[f'{ticker}_momentum_20'] = price_series.pct_change(20)
        
        # 3. Volatility features
        features[f'{ticker}_volatility_{config.VOLATILITY_WINDOW}'] = \
            return_series.rolling(window=config.VOLATILITY_WINDOW).std()
        
        # 4. Price-based features
        features[f'{ticker}_price_change'] = price_series.pct_change()
        features[f'{ticker}_price_acceleration'] = features[f'{ticker}_price_change'].diff()
        
        # 5. Statistical features
        features[f'{ticker}_return_mean_20'] = return_series.rolling(window=20).mean()
        features[f'{ticker}_return_std_20'] = return_series.rolling(window=20).std()
        features[f'{ticker}_return_skew_20'] = return_series.rolling(window=20).skew()
        features[f'{ticker}_return_kurt_20'] = return_series.rolling(window=20).kurt()
        
        # 6. RSI (Relative Strength Index)
        features[f'{ticker}_RSI_14'] = self._calculate_rsi(price_series, 14)
        
        # 7. Bollinger Bands
        ma_20 = price_series.rolling(window=20).mean()
        std_20 = price_series.rolling(window=20).std()
        features[f'{ticker}_BB_upper'] = ma_20 + (std_20 * 2)
        features[f'{ticker}_BB_lower'] = ma_20 - (std_20 * 2)
        features[f'{ticker}_BB_position'] = (price_series - features[f'{ticker}_BB_lower']) / \
                                            (features[f'{ticker}_BB_upper'] - features[f'{ticker}_BB_lower'])
        
        # 8. Volume-based features (if available - using returns as proxy)
        features[f'{ticker}_return_volume'] = return_series.abs().rolling(window=10).mean()
        
        # 9. Lag features
        for lag in [1, 2, 3, 5]:
            features[f'{ticker}_return_lag_{lag}'] = return_series.shift(lag)
        
        # 10. Rolling correlations with market (using average of all returns as market proxy)
        features[f'{ticker}_market_correlation'] = \
            return_series.rolling(window=20).corr(return_series.rolling(window=20).mean())
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def create_target_variables(
        self, 
        returns: pd.DataFrame,
        forecast_horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create target variables for prediction
        
        Args:
            returns: Historical returns
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Tuple of (future_returns, future_volatility)
        """
        logger.info(f"Creating target variables with horizon={forecast_horizon}")
        
        # Future returns (shifted backwards)
        future_returns = returns.shift(-forecast_horizon)
        
        # Future volatility (rolling std of future returns)
        future_volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for col in returns.columns:
            rolling_vol = returns[col].rolling(window=forecast_horizon).std().shift(-forecast_horizon)
            future_volatility[col] = rolling_vol
        
        # Remove NaN values
        future_returns = future_returns.dropna()
        future_volatility = future_volatility.dropna()
        
        logger.info(f"Target variables created. Returns shape: {future_returns.shape}, Volatility shape: {future_volatility.shape}")
        
        return future_returns, future_volatility
    
    def prepare_ml_dataset(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for ML training
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
            target_col: Column name for target variable
            
        Returns:
            Tuple of (X, y) ready for ML
        """
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        
        X = features.loc[common_index]
        y = targets.loc[common_index, target_col]
        
        # Remove any remaining NaNs
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"ML dataset prepared: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def get_feature_importance_report(
        self, 
        feature_names: list,
        importances: np.ndarray,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Create feature importance report
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df


if __name__ == "__main__":
    # Test the module
    from data_module import DataFetcher
    from datetime import datetime, timedelta
    
    # Fetch some test data
    fetcher = DataFetcher()
    tickers = ["AAPL", "MSFT"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    prices = fetcher.fetch_data(tickers, start_date, end_date)
    returns = fetcher.calculate_returns()
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(prices, returns)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"\nFeature columns (first 10): {features.columns[:10].tolist()}")
    print(f"\nFeatures sample:\n{features.head()}")
    
    # Create targets
    future_returns, future_volatility = engineer.create_target_variables(returns)
    print(f"\nFuture returns shape: {future_returns.shape}")
    print(f"\nFuture volatility shape: {future_volatility.shape}")
