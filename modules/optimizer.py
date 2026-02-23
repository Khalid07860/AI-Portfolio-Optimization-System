"""
Fixed Optimizer Module - Correct calculations for returns and risk metrics
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Portfolio optimization with CORRECTED calculations"""
    
    def __init__(self, risk_free_rate: float = 0.06):  # 6% for India
        self.risk_free_rate = risk_free_rate
        self.expected_returns = None
        self.cov_matrix = None
        self.tickers = None
        
    def set_parameters(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ):
        """Set optimization parameters with validation"""
        # Validate inputs
        if expected_returns.isna().any():
            logger.warning("NaN values in expected returns, filling with mean")
            expected_returns = expected_returns.fillna(expected_returns.mean())
        
        if (expected_returns < -0.5).any() or (expected_returns > 2.0).any():
            logger.warning("Extreme return values detected, capping...")
            expected_returns = expected_returns.clip(-0.5, 2.0)
        
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.tickers = list(expected_returns.index)
        
        logger.info(f"Parameters set for {len(self.tickers)} assets")
        logger.info(f"Expected returns range: {expected_returns.min():.2%} to {expected_returns.max():.2%}")
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics
        
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        # Portfolio return
        portfolio_return = np.sum(weights * self.expected_returns.values)
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        )
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe(self, weights: np.ndarray) -> float:
        """Objective function to minimize (negative Sharpe ratio)"""
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return self.portfolio_performance(weights)[1]
    
    def optimize_max_sharpe(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
        allow_short: bool = False
    ) -> Dict:
        """
        Optimize for maximum Sharpe ratio
        """
        logger.info("Optimizing for maximum Sharpe ratio...")
        
        n_assets = len(self.tickers)
        
        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        if allow_short:
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        else:
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")
        
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        # Validation
        if ret < -0.5 or ret > 2.0:
            logger.error(f"Invalid return: {ret:.2%}")
            # Use equal weights as fallback
            weights = np.array([1/n_assets] * n_assets)
            ret, vol, sharpe = self.portfolio_performance(weights)
        
        optimization_result = {
            'weights': dict(zip(self.tickers, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }
        
        logger.info(f"Max Sharpe - Return: {ret:.2%}, Vol: {vol:.2%}, Sharpe: {sharpe:.3f}")
        
        return optimization_result
    
    def optimize_min_volatility(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
        allow_short: bool = False
    ) -> Dict:
        """
        Optimize for minimum volatility
        """
        logger.info("Optimizing for minimum volatility...")
        
        n_assets = len(self.tickers)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if allow_short:
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        else:
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            self.portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        optimization_result = {
            'weights': dict(zip(self.tickers, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }
        
        logger.info(f"Min Vol - Return: {ret:.2%}, Vol: {vol:.2%}, Sharpe: {sharpe:.3f}")
        
        return optimization_result
    
    def generate_efficient_frontier(
        self,
        n_portfolios: int = 5000,
        max_weight: float = 0.40
    ) -> pd.DataFrame:
        """
        Generate efficient frontier using Monte Carlo simulation
        """
        logger.info(f"Generating efficient frontier with {n_portfolios} portfolios...")
        
        n_assets = len(self.tickers)
        results = []
        
        np.random.seed(42)
        
        for _ in range(n_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)
            
            # Ensure max weight constraint
            while np.any(weights > max_weight):
                weights = np.random.random(n_assets)
                weights = weights / np.sum(weights)
            
            # Calculate performance
            ret, vol, sharpe = self.portfolio_performance(weights)
            
            # Only include reasonable portfolios
            if -0.5 < ret < 2.0 and vol < 1.0:
                results.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    **{f'weight_{ticker}': w for ticker, w in zip(self.tickers, weights)}
                })
        
        efficient_frontier = pd.DataFrame(results)
        logger.info(f"Generated {len(efficient_frontier)} valid portfolios")
        
        return efficient_frontier
    
    def calculate_capital_allocation(
        self,
        weights: Dict[str, float],
        investment_amount: float,
        latest_prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Calculate capital allocation in shares and rupees
        """
        logger.info(f"Calculating capital allocation for ₹{investment_amount:,.2f}")
        
        allocation = []
        
        for ticker, weight in weights.items():
            if weight > 0.001:  # Filter negligible weights
                dollar_amount = investment_amount * weight
                
                if ticker in latest_prices:
                    price = latest_prices[ticker]
                    shares = int(dollar_amount / price)
                    actual_amount = shares * price
                else:
                    shares = 0
                    actual_amount = 0
                    price = 0
                
                allocation.append({
                    'Ticker': ticker,
                    'Weight': weight,
                    'Target_Amount': dollar_amount,
                    'Latest_Price': price,
                    'Shares': shares,
                    'Actual_Amount': actual_amount
                })
        
        allocation_df = pd.DataFrame(allocation)
        
        if not allocation_df.empty:
            allocation_df = allocation_df.sort_values('Weight', ascending=False)
            
            # Calculate actual weights
            total_actual = allocation_df['Actual_Amount'].sum()
            if total_actual > 0:
                allocation_df['Actual_Weight'] = allocation_df['Actual_Amount'] / total_actual
            
            # Cash remainder
            cash_remainder = investment_amount - total_actual
            logger.info(f"Cash remainder: ₹{cash_remainder:,.2f}")
        
        return allocation_df
    
    def stress_test(
        self,
        weights: Dict[str, float],
        scenario: str = "market_crash"
    ) -> Dict:
        """
        Perform stress testing on portfolio
        """
        logger.info(f"Running stress test: {scenario}")
        
        weights_array = np.array([weights[ticker] for ticker in self.tickers])
        
        # Original performance
        original_return, original_vol, original_sharpe = self.portfolio_performance(weights_array)
        
        if scenario == "market_crash":
            # Apply -10% shock to all returns
            stressed_returns = self.expected_returns * 0.90
            stressed_return = np.sum(weights_array * stressed_returns.values)
            stressed_vol = original_vol
            
        elif scenario == "volatility_spike":
            # Increase volatility by 50%
            stressed_cov = self.cov_matrix * 1.5
            stressed_vol = np.sqrt(
                np.dot(weights_array.T, np.dot(stressed_cov.values, weights_array))
            )
            stressed_return = original_return
            
        else:
            stressed_return = original_return
            stressed_vol = original_vol
        
        stressed_sharpe = (stressed_return - self.risk_free_rate) / stressed_vol
        
        stress_results = {
            'scenario': scenario,
            'original_return': original_return,
            'stressed_return': stressed_return,
            'return_change': stressed_return - original_return,
            'original_volatility': original_vol,
            'stressed_volatility': stressed_vol,
            'volatility_change': stressed_vol - original_vol,
            'original_sharpe': original_sharpe,
            'stressed_sharpe': stressed_sharpe,
            'sharpe_change': stressed_sharpe - original_sharpe
        }
        
        logger.info(f"Stress test completed: Return change = {stress_results['return_change']:.2%}")
        
        return stress_results
