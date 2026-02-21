"""
API Server: FastAPI REST endpoints for portfolio optimization
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_module import DataFetcher
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import MLPredictor
from modules.optimizer import PortfolioOptimizer
import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Portfolio Optimization System",
    description="Production-ready portfolio optimization with ML predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class OptimizationRequest(BaseModel):
    tickers: List[str] = Field(..., example=["AAPL", "MSFT", "GOOGL", "AMZN"])
    start_date: str = Field(..., example="2023-01-01")
    end_date: str = Field(..., example="2024-01-01")
    investment_amount: float = Field(..., example=10000.0, gt=0)
    risk_profile: str = Field("moderate", example="moderate")
    max_weight: float = Field(0.4, example=0.4, ge=0, le=1)
    allow_short: bool = Field(False, example=False)


class StressTestRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    scenario: str = Field("market_crash", example="market_crash")


# Response models
class PortfolioResponse(BaseModel):
    max_sharpe_portfolio: Dict
    min_volatility_portfolio: Dict
    capital_allocation: Dict
    ml_predictions: Dict
    summary_statistics: Dict
    timestamp: str


# Global state (in production, use database or cache)
class SystemState:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.ml_predictor = MLPredictor()
        self.optimizer = PortfolioOptimizer()
        self.last_trained = None
        

state = SystemState()


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "AI Portfolio Optimization System API",
        "version": "1.0.0",
        "endpoints": {
            "optimize": "/api/optimize",
            "stress_test": "/api/stress-test",
            "validate": "/api/validate-tickers",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/validate-tickers")
def validate_tickers(tickers: List[str]):
    """
    Validate ticker symbols
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Dictionary with valid and invalid tickers
    """
    try:
        valid_tickers = state.data_fetcher.validate_tickers(tickers)
        invalid_tickers = list(set(tickers) - set(valid_tickers))
        
        return {
            "valid_tickers": valid_tickers,
            "invalid_tickers": invalid_tickers,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error validating tickers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize", response_model=PortfolioResponse)
def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio based on ML predictions
    
    Args:
        request: Optimization parameters
        
    Returns:
        Optimized portfolio results
    """
    try:
        logger.info(f"Optimization request received: {request.tickers}")
        
        # Step 1: Fetch data
        logger.info("Step 1: Fetching historical data...")
        prices = state.data_fetcher.fetch_data(
            request.tickers,
            request.start_date,
            request.end_date
        )
        returns = state.data_fetcher.calculate_returns()
        
        # Step 2: Feature engineering
        logger.info("Step 2: Creating features...")
        features = state.feature_engineer.create_features(prices, returns)
        future_returns, future_volatility = state.feature_engineer.create_target_variables(returns)
        
        # Step 3: Train ML models
        logger.info("Step 3: Training ML models...")
        training_metrics = state.ml_predictor.train_models(
            features, future_returns, future_volatility
        )
        
        # Step 4: Make predictions
        logger.info("Step 4: Making predictions...")
        expected_returns, expected_volatility = state.ml_predictor.predict(
            features, request.tickers
        )
        
        # Step 5: Calculate covariance matrix
        logger.info("Step 5: Calculating covariance matrix...")
        cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
        
        # Step 6: Optimize portfolios
        logger.info("Step 6: Optimizing portfolios...")
        state.optimizer.set_parameters(expected_returns, cov_matrix)
        
        max_sharpe = state.optimizer.optimize_max_sharpe(
            max_weight=request.max_weight,
            allow_short=request.allow_short
        )
        
        min_vol = state.optimizer.optimize_min_volatility(
            max_weight=request.max_weight,
            allow_short=request.allow_short
        )
        
        # Step 7: Get latest prices for allocation
        logger.info("Step 7: Calculating capital allocation...")
        latest_prices = state.data_fetcher.get_latest_prices(request.tickers)
        
        max_sharpe_allocation = state.optimizer.calculate_capital_allocation(
            max_sharpe['weights'],
            request.investment_amount,
            latest_prices
        )
        
        min_vol_allocation = state.optimizer.calculate_capital_allocation(
            min_vol['weights'],
            request.investment_amount,
            latest_prices
        )
        
        # Step 8: Summary statistics
        summary_stats = state.data_fetcher.get_summary_statistics(returns)
        
        # Prepare response
        response = {
            "max_sharpe_portfolio": {
                "weights": max_sharpe['weights'],
                "expected_return": float(max_sharpe['expected_return']),
                "volatility": float(max_sharpe['volatility']),
                "sharpe_ratio": float(max_sharpe['sharpe_ratio']),
                "allocation": max_sharpe_allocation.to_dict('records') if not max_sharpe_allocation.empty else []
            },
            "min_volatility_portfolio": {
                "weights": min_vol['weights'],
                "expected_return": float(min_vol['expected_return']),
                "volatility": float(min_vol['volatility']),
                "sharpe_ratio": float(min_vol['sharpe_ratio']),
                "allocation": min_vol_allocation.to_dict('records') if not min_vol_allocation.empty else []
            },
            "ml_predictions": {
                "expected_returns": expected_returns.to_dict(),
                "expected_volatility": expected_volatility.to_dict(),
                "training_metrics": training_metrics
            },
            "summary_statistics": summary_stats.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Optimization completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stress-test")
def stress_test_portfolio(request: StressTestRequest):
    """
    Perform stress testing on portfolio
    
    Args:
        request: Stress test parameters
        
    Returns:
        Stress test results
    """
    try:
        logger.info(f"Stress test request: {request.scenario}")
        
        # Fetch recent data for covariance
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        prices = state.data_fetcher.fetch_data(request.tickers, start_date, end_date)
        returns = state.data_fetcher.calculate_returns()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
        
        # Mock expected returns (in production, use latest ML predictions)
        expected_returns = returns.mean() * config.TRADING_DAYS_PER_YEAR
        
        # Set optimizer parameters
        state.optimizer.set_parameters(expected_returns, cov_matrix)
        
        # Run stress test
        stress_results = state.optimizer.stress_test(request.weights, request.scenario)
        
        return {
            "stress_test_results": stress_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stress test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/efficient-frontier")
def generate_efficient_frontier(
    tickers: List[str],
    start_date: str,
    end_date: str,
    n_portfolios: int = 5000
):
    """
    Generate efficient frontier
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        n_portfolios: Number of portfolios to simulate
        
    Returns:
        Efficient frontier data
    """
    try:
        logger.info(f"Generating efficient frontier for {len(tickers)} tickers")
        
        # Fetch data
        prices = state.data_fetcher.fetch_data(tickers, start_date, end_date)
        returns = state.data_fetcher.calculate_returns()
        
        # Calculate parameters
        expected_returns = returns.mean() * config.TRADING_DAYS_PER_YEAR
        cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
        
        # Set optimizer parameters
        state.optimizer.set_parameters(expected_returns, cov_matrix)
        
        # Generate frontier
        frontier = state.optimizer.generate_efficient_frontier(n_portfolios=n_portfolios)
        
        return {
            "efficient_frontier": frontier.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Efficient frontier error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting API server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "api_server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )
