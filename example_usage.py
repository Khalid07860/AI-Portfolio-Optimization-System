"""
Example Usage Script: Practical examples of using the portfolio optimization system
"""
import sys
from datetime import datetime, timedelta

# Import modules
from modules.data_module import DataFetcher
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import MLPredictor
from modules.optimizer import PortfolioOptimizer
import config

def example_1_basic_optimization():
    """
    Example 1: Basic portfolio optimization
    Simple use case with tech stocks
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Portfolio Optimization")
    print("="*70)
    
    # Define portfolio
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    investment_amount = 10000
    
    # Date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"\nTickers: {tickers}")
    print(f"Investment: ${investment_amount:,}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Step 1: Fetch data
    print("\n1. Fetching data...")
    data_fetcher = DataFetcher()
    prices = data_fetcher.fetch_data(tickers, start_date, end_date)
    returns = data_fetcher.calculate_returns()
    print(f"   Fetched {len(prices)} days of data")
    
    # Step 2: Simple optimization (no ML)
    print("\n2. Running optimization...")
    expected_returns = returns.mean() * config.TRADING_DAYS_PER_YEAR
    cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
    
    optimizer = PortfolioOptimizer()
    optimizer.set_parameters(expected_returns, cov_matrix)
    
    # Maximum Sharpe
    max_sharpe = optimizer.optimize_max_sharpe(max_weight=0.4, allow_short=False)
    
    print("\n" + "-"*70)
    print("RESULTS - Maximum Sharpe Portfolio:")
    print("-"*70)
    print(f"Expected Return:  {max_sharpe['expected_return']*100:.2f}%")
    print(f"Volatility:       {max_sharpe['volatility']*100:.2f}%")
    print(f"Sharpe Ratio:     {max_sharpe['sharpe_ratio']:.3f}")
    print("\nWeights:")
    for ticker, weight in sorted(max_sharpe['weights'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            print(f"  {ticker}: {weight*100:>6.2f}%")
    
    # Calculate allocation
    latest_prices = data_fetcher.get_latest_prices(tickers)
    allocation = optimizer.calculate_capital_allocation(
        max_sharpe['weights'],
        investment_amount,
        latest_prices
    )
    
    print("\nCapital Allocation:")
    print(allocation.to_string(index=False))
    
    return max_sharpe, allocation


def example_2_ml_enhanced_optimization():
    """
    Example 2: ML-enhanced optimization
    Uses ML predictions for expected returns
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: ML-Enhanced Portfolio Optimization")
    print("="*70)
    
    # Define portfolio
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD"]
    
    # Date range (need more data for ML)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    print(f"\nTickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Initialize components
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()
    ml_predictor = MLPredictor()
    optimizer = PortfolioOptimizer()
    
    # Step 1: Fetch data
    print("\n1. Fetching data...")
    prices = data_fetcher.fetch_data(tickers, start_date, end_date)
    returns = data_fetcher.calculate_returns()
    print(f"   Fetched {len(prices)} days of data")
    
    # Step 2: Feature engineering
    print("\n2. Creating features...")
    features = feature_engineer.create_features(prices, returns)
    future_returns, future_volatility = feature_engineer.create_target_variables(returns)
    print(f"   Created {features.shape[1]} features")
    
    # Step 3: Train ML models
    print("\n3. Training ML models...")
    training_metrics = ml_predictor.train_models(features, future_returns, future_volatility)
    
    print("\n   Training Results:")
    for ticker, metrics in training_metrics.items():
        if 'return' in metrics and 'test_r2' in metrics['return']:
            print(f"   {ticker}: Return R² = {metrics['return']['test_r2']:.3f}")
    
    # Step 4: Make predictions
    print("\n4. Making predictions...")
    expected_returns, expected_volatility = ml_predictor.predict(features, tickers)
    
    print("\n   ML Predictions:")
    for ticker in tickers:
        print(f"   {ticker}: Return = {expected_returns[ticker]*100:.2f}%, Vol = {expected_volatility[ticker]*100:.2f}%")
    
    # Step 5: Optimize with ML predictions
    print("\n5. Optimizing portfolio...")
    cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
    optimizer.set_parameters(expected_returns, cov_matrix)
    
    max_sharpe = optimizer.optimize_max_sharpe(max_weight=0.35, allow_short=False)
    
    print("\n" + "-"*70)
    print("RESULTS - ML-Enhanced Maximum Sharpe Portfolio:")
    print("-"*70)
    print(f"Expected Return:  {max_sharpe['expected_return']*100:.2f}%")
    print(f"Volatility:       {max_sharpe['volatility']*100:.2f}%")
    print(f"Sharpe Ratio:     {max_sharpe['sharpe_ratio']:.3f}")
    print("\nWeights:")
    for ticker, weight in sorted(max_sharpe['weights'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            print(f"  {ticker}: {weight*100:>6.2f}%")
    
    return max_sharpe


def example_3_stress_testing():
    """
    Example 3: Stress testing portfolios
    Test portfolio under adverse conditions
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Portfolio Stress Testing")
    print("="*70)
    
    # Use results from example 1
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    weights = {"AAPL": 0.25, "MSFT": 0.40, "GOOGL": 0.20, "AMZN": 0.15}
    
    print(f"\nTesting portfolio: {weights}")
    
    # Fetch data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    data_fetcher = DataFetcher()
    prices = data_fetcher.fetch_data(tickers, start_date, end_date)
    returns = data_fetcher.calculate_returns()
    
    # Set up optimizer
    expected_returns = returns.mean() * config.TRADING_DAYS_PER_YEAR
    cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
    
    optimizer = PortfolioOptimizer()
    optimizer.set_parameters(expected_returns, cov_matrix)
    
    # Run stress tests
    scenarios = ["market_crash", "volatility_spike"]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario.upper().replace('_', ' ')}")
        print('='*70)
        
        stress_results = optimizer.stress_test(weights, scenario)
        
        print(f"\nOriginal Portfolio:")
        print(f"  Return:     {stress_results['original_return']*100:.2f}%")
        print(f"  Volatility: {stress_results['original_volatility']*100:.2f}%")
        print(f"  Sharpe:     {stress_results['original_sharpe']:.3f}")
        
        print(f"\nStressed Portfolio:")
        print(f"  Return:     {stress_results['stressed_return']*100:.2f}%")
        print(f"  Volatility: {stress_results['stressed_volatility']*100:.2f}%")
        print(f"  Sharpe:     {stress_results['stressed_sharpe']:.3f}")
        
        print(f"\nChange:")
        print(f"  Return:     {stress_results['return_change']*100:+.2f}%")
        print(f"  Volatility: {stress_results['volatility_change']*100:+.2f}%")
        print(f"  Sharpe:     {stress_results['sharpe_change']:+.3f}")


def example_4_efficient_frontier():
    """
    Example 4: Generate efficient frontier
    Visualize risk-return tradeoffs
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Efficient Frontier Generation")
    print("="*70)
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"\nTickers: {tickers}")
    print("Generating 1000 random portfolios...")
    
    # Fetch data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    data_fetcher = DataFetcher()
    prices = data_fetcher.fetch_data(tickers, start_date, end_date)
    returns = data_fetcher.calculate_returns()
    
    # Set up optimizer
    expected_returns = returns.mean() * config.TRADING_DAYS_PER_YEAR
    cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
    
    optimizer = PortfolioOptimizer()
    optimizer.set_parameters(expected_returns, cov_matrix)
    
    # Generate frontier
    frontier = optimizer.generate_efficient_frontier(n_portfolios=1000, max_weight=0.5)
    
    # Find key portfolios
    max_sharpe_idx = frontier['sharpe_ratio'].idxmax()
    min_vol_idx = frontier['volatility'].idxmin()
    max_return_idx = frontier['return'].idxmax()
    
    print("\n" + "-"*70)
    print("Key Portfolios on Efficient Frontier:")
    print("-"*70)
    
    print("\n1. Maximum Sharpe Portfolio:")
    print(f"   Return:  {frontier.loc[max_sharpe_idx, 'return']*100:.2f}%")
    print(f"   Vol:     {frontier.loc[max_sharpe_idx, 'volatility']*100:.2f}%")
    print(f"   Sharpe:  {frontier.loc[max_sharpe_idx, 'sharpe_ratio']:.3f}")
    
    print("\n2. Minimum Volatility Portfolio:")
    print(f"   Return:  {frontier.loc[min_vol_idx, 'return']*100:.2f}%")
    print(f"   Vol:     {frontier.loc[min_vol_idx, 'volatility']*100:.2f}%")
    print(f"   Sharpe:  {frontier.loc[min_vol_idx, 'sharpe_ratio']:.3f}")
    
    print("\n3. Maximum Return Portfolio:")
    print(f"   Return:  {frontier.loc[max_return_idx, 'return']*100:.2f}%")
    print(f"   Vol:     {frontier.loc[max_return_idx, 'volatility']*100:.2f}%")
    print(f"   Sharpe:  {frontier.loc[max_return_idx, 'sharpe_ratio']:.3f}")
    
    # Statistics
    print("\n" + "-"*70)
    print("Frontier Statistics:")
    print("-"*70)
    print(f"Average Return:    {frontier['return'].mean()*100:.2f}%")
    print(f"Average Volatility: {frontier['volatility'].mean()*100:.2f}%")
    print(f"Average Sharpe:    {frontier['sharpe_ratio'].mean():.3f}")
    print(f"Max Sharpe:        {frontier['sharpe_ratio'].max():.3f}")


def example_5_diversified_portfolio():
    """
    Example 5: Diversified multi-sector portfolio
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Diversified Multi-Sector Portfolio")
    print("="*70)
    
    # Diversified portfolio across sectors
    tickers = [
        "AAPL",   # Tech
        "JPM",    # Finance
        "JNJ",    # Healthcare
        "XOM",    # Energy
        "WMT",    # Consumer
        "BA"      # Industrial
    ]
    
    investment_amount = 50000
    
    print(f"\nSector-Diversified Portfolio:")
    print("  AAPL - Technology")
    print("  JPM  - Financial Services")
    print("  JNJ  - Healthcare")
    print("  XOM  - Energy")
    print("  WMT  - Consumer Staples")
    print("  BA   - Industrials")
    print(f"\nInvestment Amount: ${investment_amount:,}")
    
    # Fetch data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    data_fetcher = DataFetcher()
    prices = data_fetcher.fetch_data(tickers, start_date, end_date)
    returns = data_fetcher.calculate_returns()
    
    # Optimize with diversification constraint (max 25% per asset)
    expected_returns = returns.mean() * config.TRADING_DAYS_PER_YEAR
    cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
    
    optimizer = PortfolioOptimizer()
    optimizer.set_parameters(expected_returns, cov_matrix)
    
    max_sharpe = optimizer.optimize_max_sharpe(max_weight=0.25, allow_short=False)
    min_vol = optimizer.optimize_min_volatility(max_weight=0.25, allow_short=False)
    
    # Display both portfolios
    print("\n" + "-"*70)
    print("MAXIMUM SHARPE PORTFOLIO (Aggressive)")
    print("-"*70)
    print(f"Return:  {max_sharpe['expected_return']*100:.2f}%")
    print(f"Vol:     {max_sharpe['volatility']*100:.2f}%")
    print(f"Sharpe:  {max_sharpe['sharpe_ratio']:.3f}")
    print("\nAllocation:")
    for ticker, weight in sorted(max_sharpe['weights'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            amount = investment_amount * weight
            print(f"  {ticker}: {weight*100:>6.2f}% (${amount:>10,.2f})")
    
    print("\n" + "-"*70)
    print("MINIMUM VOLATILITY PORTFOLIO (Conservative)")
    print("-"*70)
    print(f"Return:  {min_vol['expected_return']*100:.2f}%")
    print(f"Vol:     {min_vol['volatility']*100:.2f}%")
    print(f"Sharpe:  {min_vol['sharpe_ratio']:.3f}")
    print("\nAllocation:")
    for ticker, weight in sorted(min_vol['weights'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            amount = investment_amount * weight
            print(f"  {ticker}: {weight*100:>6.2f}% (${amount:>10,.2f})")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("AI PORTFOLIO OPTIMIZATION SYSTEM - EXAMPLE USAGE")
    print("="*70)
    print("\nThis script demonstrates various use cases of the system.")
    print("Each example is self-contained and can be run independently.")
    
    try:
        # Run examples
        example_1_basic_optimization()
        
        input("\n\nPress Enter to continue to Example 2...")
        example_2_ml_enhanced_optimization()
        
        input("\n\nPress Enter to continue to Example 3...")
        example_3_stress_testing()
        
        input("\n\nPress Enter to continue to Example 4...")
        example_4_efficient_frontier()
        
        input("\n\nPress Enter to continue to Example 5...")
        example_5_diversified_portfolio()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED!")
        print("="*70)
        print("\nNext steps:")
        print("1. Try modifying the tickers in these examples")
        print("2. Adjust the constraints (max_weight, etc.)")
        print("3. Experiment with different date ranges")
        print("4. Build your own custom portfolio!")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
