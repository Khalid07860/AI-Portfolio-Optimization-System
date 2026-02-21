"""
Optimizer Module: Performs portfolio optimization and simulations
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import sys
sys.path.append('..')
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Portfolio optimization using mean-variance optimization"""
    
    def __init__(self, risk_free_rate: float = config.DEFAULT_RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.expected_returns = None
        self.cov_matrix = None
        self.tickers = None
        
    def set_parameters(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ):
        """
        Set optimization parameters
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.tickers = list(expected_returns.index)
        logger.info(f"Parameters set for {len(self.tickers)} assets")
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(weights * self.expected_returns.values)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        )
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
        max_weight: float = config.DEFAULT_MAX_WEIGHT,
        min_weight: float = config.DEFAULT_MIN_WEIGHT,
        allow_short: bool = config.ALLOW_SHORT_SELLING
    ) -> Dict:
        """
        Optimize for maximum Sharpe ratio
        
        Args:
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            allow_short: Allow short selling
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Optimizing for maximum Sharpe ratio...")
        
        n_assets = len(self.tickers)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
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
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")
        
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        optimization_result = {
            'weights': dict(zip(self.tickers, weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }
        
        logger.info(f"Max Sharpe Portfolio - Return: {ret:.4f}, Vol: {vol:.4f}, Sharpe: {sharpe:.4f}")
        
        return optimization_result
    
    def optimize_min_volatility(
        self,
        max_weight: float = config.DEFAULT_MAX_WEIGHT,
        min_weight: float = config.DEFAULT_MIN_WEIGHT,
        allow_short: bool = config.ALLOW_SHORT_SELLING
    ) -> Dict:
        """
        Optimize for minimum volatility
        
        Args:
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            allow_short: Allow short selling
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Optimizing for minimum volatility...")
        
        n_assets = len(self.tickers)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        if allow_short:
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        else:
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
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
        
        logger.info(f"Min Vol Portfolio - Return: {ret:.4f}, Vol: {vol:.4f}, Sharpe: {sharpe:.4f}")
        
        return optimization_result
    
    def generate_efficient_frontier(
        self,
        n_portfolios: int = config.N_PORTFOLIOS,
        max_weight: float = config.DEFAULT_MAX_WEIGHT
    ) -> pd.DataFrame:
        """
        Generate efficient frontier using Monte Carlo simulation
        
        Args:
            n_portfolios: Number of random portfolios
            max_weight: Maximum weight per asset
            
        Returns:
            DataFrame with portfolio simulations
        """
        logger.info(f"Generating efficient frontier with {n_portfolios} portfolios...")
        
        n_assets = len(self.tickers)
        results = []
        
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
            
            results.append({
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                **{f'weight_{ticker}': w for ticker, w in zip(self.tickers, weights)}
            })
        
        efficient_frontier = pd.DataFrame(results)
        logger.info("Efficient frontier generated")
        
        return efficient_frontier
    
    def calculate_capital_allocation(
        self,
        weights: Dict[str, float],
        investment_amount: float,
        latest_prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Calculate capital allocation in dollars and shares
        
        Args:
            weights: Portfolio weights
            investment_amount: Total investment amount
            latest_prices: Latest prices for each asset
            
        Returns:
            DataFrame with allocation details
        """
        logger.info(f"Calculating capital allocation for ${investment_amount:,.2f}")
        
        allocation = []
        
        for ticker, weight in weights.items():
            if weight > 0.001:  # Filter out negligible weights
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
            allocation_df['Actual_Weight'] = allocation_df['Actual_Amount'] / total_actual
            
            # Add cash remainder
            cash_remainder = investment_amount - total_actual
            logger.info(f"Cash remainder: ${cash_remainder:,.2f}")
        
        return allocation_df
    
    def stress_test(
        self,
        weights: Dict[str, float],
        scenario: str = "market_crash"
    ) -> Dict:
        """
        Perform stress testing on portfolio
        
        Args:
            weights: Portfolio weights
            scenario: Stress test scenario
            
        Returns:
            Dictionary with stress test results
        """
        logger.info(f"Running stress test: {scenario}")
        
        weights_array = np.array([weights[ticker] for ticker in self.tickers])
        
        # Original performance
        original_return, original_vol, original_sharpe = self.portfolio_performance(weights_array)
        
        if scenario == "market_crash":
            # Apply -10% shock to all returns
            stressed_returns = self.expected_returns * (1 + config.STRESS_SCENARIOS['market_crash'])
            stressed_return = np.sum(weights_array * stressed_returns.values)
            stressed_vol = original_vol
            
        elif scenario == "volatility_spike":
            # Increase volatility by 50%
            stressed_cov = self.cov_matrix * config.STRESS_SCENARIOS['volatility_spike']
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
        
        logger.info(f"Stress test completed: Return change = {stress_results['return_change']:.4f}")
        
        return stress_results


if __name__ == "__main__":
    # Test the optimizer
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Mock data
    expected_returns = pd.Series([0.12, 0.10, 0.15], index=tickers)
    returns_data = pd.DataFrame(
        np.random.randn(252, 3) * 0.02,
        columns=tickers
    )
    cov_matrix = returns_data.cov() * 252
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    optimizer.set_parameters(expected_returns, cov_matrix)
    
    # Optimize
    max_sharpe = optimizer.optimize_max_sharpe()
    min_vol = optimizer.optimize_min_volatility()
    
    print("\nMax Sharpe Portfolio:")
    print(f"Return: {max_sharpe['expected_return']:.4f}")
    print(f"Volatility: {max_sharpe['volatility']:.4f}")
    print(f"Sharpe: {max_sharpe['sharpe_ratio']:.4f}")
    print(f"Weights: {max_sharpe['weights']}")
    
    print("\nMin Volatility Portfolio:")
    print(f"Return: {min_vol['expected_return']:.4f}")
    print(f"Volatility: {min_vol['volatility']:.4f}")
    print(f"Sharpe: {min_vol['sharpe_ratio']:.4f}")
    print(f"Weights: {min_vol['weights']}")
    
    # Stress test
    stress_result = optimizer.stress_test(max_sharpe['weights'], "market_crash")
    print(f"\nStress Test (Market Crash):")
    print(f"Return change: {stress_result['return_change']:.4f}")
