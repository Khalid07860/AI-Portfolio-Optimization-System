"""
Configuration file for Portfolio Optimization System
"""
import logging
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Logging configuration
LOG_FILE = LOGS_DIR / "portfolio_system.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Portfolio optimization parameters
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252

# ML Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# Optimization constraints
DEFAULT_MAX_WEIGHT = 0.40  # 40% max per asset
DEFAULT_MIN_WEIGHT = 0.0   # No short selling by default
ALLOW_SHORT_SELLING = False

# Monte Carlo simulation
N_PORTFOLIOS = 10000

# Stress test scenarios
STRESS_SCENARIOS = {
    "market_crash": -0.10,      # -10% shock
    "volatility_spike": 1.5,     # 1.5x volatility increase
    "sector_rotation": 0.05      # 5% sector shift
}

# Feature engineering parameters
LOOKBACK_PERIODS = [5, 10, 20, 50]  # Days for moving averages
VOLATILITY_WINDOW = 20

# Default stock universe (can be overridden)
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "V", "WMT"]

# Risk profiles
RISK_PROFILES = {
    "conservative": {"max_volatility": 0.12, "min_return": 0.08},
    "moderate": {"max_volatility": 0.18, "min_return": 0.12},
    "aggressive": {"max_volatility": 0.30, "min_return": 0.18}
}
