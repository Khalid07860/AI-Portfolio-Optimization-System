"""
Simple launcher script for the Portfolio Optimization System
"""
import sys
import subprocess
from typing import Optional

def print_menu():
    """Print main menu"""
    print("\n" + "="*60)
    print("🚀 AI Portfolio Optimization System")
    print("="*60)
    print("\nSelect an option:")
    print("1. Launch Interactive Dashboard (Streamlit)")
    print("2. Launch API Server (FastAPI)")
    print("3. Run Quick Test")
    print("4. Exit")
    print("="*60)

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("\n🎯 Launching Streamlit Dashboard...")
    print("📍 URL: http://localhost:8501")
    print("⌨️  Press Ctrl+C to stop\n")
    try:
        subprocess.run(["streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")

def launch_api():
    """Launch FastAPI server"""
    print("\n🎯 Launching FastAPI Server...")
    print("📍 URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("⌨️  Press Ctrl+C to stop\n")
    try:
        subprocess.run(["python", "api_server.py"])
    except KeyboardInterrupt:
        print("\n✅ API Server stopped")

def run_test():
    """Run a quick test"""
    print("\n🧪 Running Quick Test...")
    print("="*60)
    
    try:
        from modules.data_module import DataFetcher
        from modules.optimizer import PortfolioOptimizer
        from datetime import datetime, timedelta
        import pandas as pd
        import numpy as np
        
        # Test data fetching
        print("\n1. Testing Data Fetcher...")
        fetcher = DataFetcher()
        
        # Validate tickers
        test_tickers = ["AAPL", "MSFT", "GOOGL"]
        print(f"   Validating tickers: {test_tickers}")
        valid_tickers = fetcher.validate_tickers(test_tickers)
        print(f"   ✅ Valid tickers: {valid_tickers}")
        
        # Fetch data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        print(f"\n2. Fetching data from {start_date} to {end_date}...")
        prices = fetcher.fetch_data(valid_tickers, start_date, end_date)
        print(f"   ✅ Fetched {len(prices)} days of data")
        print(f"   Latest prices:\n{prices.tail(1)}")
        
        # Calculate returns
        print("\n3. Calculating returns...")
        returns = fetcher.calculate_returns()
        print(f"   ✅ Returns shape: {returns.shape}")
        
        # Test optimizer
        print("\n4. Testing Optimizer...")
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        optimizer = PortfolioOptimizer()
        optimizer.set_parameters(expected_returns, cov_matrix)
        
        max_sharpe = optimizer.optimize_max_sharpe(max_weight=0.4, allow_short=False)
        
        print(f"   ✅ Max Sharpe Portfolio:")
        print(f"      Return: {max_sharpe['expected_return']*100:.2f}%")
        print(f"      Volatility: {max_sharpe['volatility']*100:.2f}%")
        print(f"      Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
        print(f"      Weights: {max_sharpe['weights']}")
        
        print("\n" + "="*60)
        print("✅ All tests passed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            launch_dashboard()
        elif choice == "2":
            launch_api()
        elif choice == "3":
            run_test()
            input("\nPress Enter to continue...")
        elif choice == "4":
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
