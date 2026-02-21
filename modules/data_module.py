"""
Data Module: Handles fetching and managing historical stock data
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import config

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and manages historical stock data"""
    
    def __init__(self):
        self.data = None
        self.tickers = None
        
    def fetch_data(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            DataFrame with adjusted close prices
        """
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        try:
            # Download data
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            # Extract 'Close' prices
            if len(tickers) == 1:
                prices = data[['Close']].copy()
                prices.columns = tickers
            else:
                prices = data['Close'].copy()
            
            # Handle missing data
            prices = self._clean_data(prices)
            
            self.data = prices
            self.tickers = list(prices.columns)
            
            logger.info(f"Successfully fetched {len(prices)} rows for {len(self.tickers)} tickers")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers
        
        Args:
            data: Raw price data
            
        Returns:
            Cleaned price data
        """
        logger.info("Cleaning data...")
        
        # Remove columns with too many missing values (>20%)
        threshold = len(data) * 0.20
        data = data.dropna(axis=1, thresh=len(data) - threshold)
        
        # Forward fill then backward fill remaining NaNs
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with NaNs
        data = data.dropna()
        
        # Remove outliers (prices that change more than 50% in a day)
        returns = data.pct_change()
        mask = (returns.abs() < 0.5).all(axis=1)
        data = data[mask]
        
        logger.info(f"Data cleaned. Shape: {data.shape}")
        return data
    
    def calculate_returns(
        self, 
        prices: Optional[pd.DataFrame] = None,
        method: str = "simple"
    ) -> pd.DataFrame:
        """
        Calculate returns from prices
        
        Args:
            prices: Price data (uses self.data if None)
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame of returns
        """
        if prices is None:
            prices = self.data
            
        if prices is None:
            raise ValueError("No price data available")
        
        if method == "simple":
            returns = prices.pct_change()
        elif method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        returns = returns.dropna()
        logger.info(f"Calculated {method} returns. Shape: {returns.shape}")
        
        return returns
    
    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get latest prices for tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary of ticker: latest_price
        """
        try:
            latest_data = yf.download(
                tickers,
                period="1d",
                interval="1d",
                progress=False,
                auto_adjust=True
            )
            
            if len(tickers) == 1:
                latest_prices = {tickers[0]: float(latest_data['Close'].iloc[-1])}
            else:
                latest_prices = latest_data['Close'].iloc[-1].to_dict()
            
            logger.info(f"Retrieved latest prices for {len(latest_prices)} tickers")
            return latest_prices
            
        except Exception as e:
            logger.error(f"Error getting latest prices: {str(e)}")
            raise
    
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """
        Validate that tickers exist and have data
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            List of valid tickers
        """
        valid_tickers = []
        
        for ticker in tickers:
            try:
                # Try to fetch 1 day of data
                test_data = yf.download(
                    ticker,
                    period="5d",
                    progress=False,
                    auto_adjust=True
                )
                
                if not test_data.empty:
                    valid_tickers.append(ticker)
                else:
                    logger.warning(f"Ticker {ticker} returned no data")
                    
            except Exception as e:
                logger.warning(f"Ticker {ticker} is invalid: {str(e)}")
        
        logger.info(f"Validated {len(valid_tickers)}/{len(tickers)} tickers")
        return valid_tickers
    
    def get_summary_statistics(
        self, 
        returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate summary statistics for returns
        
        Args:
            returns: Returns data
            
        Returns:
            DataFrame with summary statistics
        """
        if returns is None:
            if self.data is None:
                raise ValueError("No data available")
            returns = self.calculate_returns()
        
        stats = pd.DataFrame({
            'Mean': returns.mean() * config.TRADING_DAYS_PER_YEAR,
            'Volatility': returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR),
            'Sharpe': (returns.mean() * config.TRADING_DAYS_PER_YEAR - config.DEFAULT_RISK_FREE_RATE) / 
                      (returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)),
            'Min': returns.min(),
            'Max': returns.max(),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis()
        })
        
        return stats


if __name__ == "__main__":
    # Test the module
    fetcher = DataFetcher()
    
    # Test with a few tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    prices = fetcher.fetch_data(tickers, start_date, end_date)
    print(f"\nPrices shape: {prices.shape}")
    print(f"\nFirst 5 rows:\n{prices.head()}")
    
    returns = fetcher.calculate_returns()
    print(f"\nReturns shape: {returns.shape}")
    
    stats = fetcher.get_summary_statistics(returns)
    print(f"\nSummary Statistics:\n{stats}")
