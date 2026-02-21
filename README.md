# 🚀 AI-Powered Portfolio Optimization System

A production-ready portfolio optimization system that uses machine learning to predict expected returns and volatility, then performs constrained mean-variance optimization to generate optimal portfolio allocations.

## 📋 Features

### Core Functionality
- **Dynamic Data Fetching**: Fetches historical stock data via Yahoo Finance API
- **Data Cleaning**: Handles missing values, outliers, and data quality issues
- **Feature Engineering**: Creates 30+ technical and statistical features per asset
- **ML Predictions**: Random Forest models predict next-period returns and volatility
- **Mean-Variance Optimization**: Scipy-based constrained optimization
- **Efficient Frontier**: Monte Carlo simulation with 5,000+ portfolios
- **Capital Allocation**: Calculates exact shares and dollar amounts
- **Stress Testing**: Simulates market crash and volatility spike scenarios

### User Interfaces
- **REST API**: FastAPI endpoints for programmatic access
- **Interactive Dashboard**: Streamlit-based web interface
- **Modular Architecture**: Clean separation of concerns

### Constraints & Features
- Weight bounds (max 40% per asset by default)
- No short selling (optional toggle)
- Risk profile support (conservative, moderate, aggressive)
- Real-time logging and error handling

## 🏗️ Architecture

```
portfolio_optimization_system/
├── config.py                      # System-wide configuration
├── requirements.txt               # Python dependencies
├── api_server.py                  # FastAPI REST endpoints
├── dashboard.py                   # Streamlit interactive dashboard
├── modules/
│   ├── __init__.py
│   ├── data_module.py            # Data fetching and management
│   ├── feature_engineering.py    # Feature creation
│   ├── ml_model.py               # ML prediction models
│   └── optimizer.py              # Portfolio optimization
├── logs/                         # System logs
└── data/                         # Cached data (optional)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone/Download the Project

```bash
cd portfolio_optimization_system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import yfinance, sklearn, scipy, fastapi, streamlit; print('All dependencies installed successfully!')"
```

## 🚀 Usage

### Option 1: Interactive Dashboard (Recommended for Beginners)

Launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

This will open a web browser with an interactive interface where you can:
1. Enter stock tickers
2. Select date range
3. Set investment amount
4. Configure risk profile
5. View optimized portfolios with visualizations

**Default URL**: `http://localhost:8501`

### Option 2: REST API Server

Start the FastAPI server:

```bash
python api_server.py
```

**Default URL**: `http://localhost:8000`

**API Documentation**: `http://localhost:8000/docs` (Swagger UI)

#### API Endpoints

##### 1. Optimize Portfolio
```bash
POST /api/optimize

# Example request:
curl -X POST "http://localhost:8000/api/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "investment_amount": 10000,
    "risk_profile": "moderate",
    "max_weight": 0.4,
    "allow_short": false
  }'
```

##### 2. Validate Tickers
```bash
POST /api/validate-tickers

# Example:
curl -X POST "http://localhost:8000/api/validate-tickers" \
  -H "Content-Type: application/json" \
  -d '["AAPL", "INVALID", "MSFT"]'
```

##### 3. Stress Test
```bash
POST /api/stress-test

# Example:
curl -X POST "http://localhost:8000/api/stress-test" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT"],
    "weights": {"AAPL": 0.6, "MSFT": 0.4},
    "scenario": "market_crash"
  }'
```

##### 4. Efficient Frontier
```bash
POST /api/efficient-frontier

# Example:
curl -X POST "http://localhost:8000/api/efficient-frontier" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "n_portfolios": 5000
  }'
```

### Option 3: Python Module (For Developers)

Use the system programmatically:

```python
from modules.data_module import DataFetcher
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import MLPredictor
from modules.optimizer import PortfolioOptimizer

# Initialize components
data_fetcher = DataFetcher()
feature_engineer = FeatureEngineer()
ml_predictor = MLPredictor()
optimizer = PortfolioOptimizer()

# Fetch data
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
prices = data_fetcher.fetch_data(tickers, "2023-01-01", "2024-01-01")
returns = data_fetcher.calculate_returns()

# Create features
features = feature_engineer.create_features(prices, returns)
future_returns, future_volatility = feature_engineer.create_target_variables(returns)

# Train ML models
training_metrics = ml_predictor.train_models(features, future_returns, future_volatility)

# Make predictions
expected_returns, expected_volatility = ml_predictor.predict(features, tickers)

# Optimize portfolio
cov_matrix = returns.cov() * 252
optimizer.set_parameters(expected_returns, cov_matrix)

max_sharpe = optimizer.optimize_max_sharpe(max_weight=0.4, allow_short=False)
min_vol = optimizer.optimize_min_volatility(max_weight=0.4, allow_short=False)

print("Max Sharpe Portfolio:", max_sharpe)
print("Min Volatility Portfolio:", min_vol)
```

## 📊 Example Workflows

### Workflow 1: Quick Optimization

```bash
# Start dashboard
streamlit run dashboard.py

# In the web interface:
# 1. Enter tickers: AAPL, MSFT, GOOGL, AMZN
# 2. Investment: $10,000
# 3. Click "Optimize Portfolio"
# 4. View results and download allocation
```

### Workflow 2: API Integration

```python
import requests

# Optimize portfolio via API
response = requests.post(
    "http://localhost:8000/api/optimize",
    json={
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "investment_amount": 10000,
        "risk_profile": "moderate",
        "max_weight": 0.4,
        "allow_short": False
    }
)

results = response.json()
print("Max Sharpe Weights:", results['max_sharpe_portfolio']['weights'])
print("Expected Return:", results['max_sharpe_portfolio']['expected_return'])
```

### Workflow 3: Stress Testing

```python
from modules.optimizer import PortfolioOptimizer

# Assume optimizer is set up with parameters
weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}

# Run stress test
stress_results = optimizer.stress_test(weights, "market_crash")
print("Return change:", stress_results['return_change'])
print("Volatility change:", stress_results['volatility_change'])
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Risk-free rate
DEFAULT_RISK_FREE_RATE = 0.02  # 2%

# Trading days per year
TRADING_DAYS_PER_YEAR = 252

# ML model parameters
N_ESTIMATORS = 100
RANDOM_STATE = 42

# Optimization constraints
DEFAULT_MAX_WEIGHT = 0.40  # 40%
DEFAULT_MIN_WEIGHT = 0.0
ALLOW_SHORT_SELLING = False

# Monte Carlo simulations
N_PORTFOLIOS = 10000

# Stress test scenarios
STRESS_SCENARIOS = {
    "market_crash": -0.10,
    "volatility_spike": 1.5
}
```

## 📈 Output Examples

### Max Sharpe Portfolio
```
Expected Return: 15.23%
Volatility: 18.45%
Sharpe Ratio: 0.716

Weights:
- AAPL: 35%
- MSFT: 40%
- GOOGL: 15%
- AMZN: 10%
```

### Capital Allocation ($10,000)
```
Ticker | Weight | Amount   | Price  | Shares | Actual Amount
-------|--------|----------|--------|--------|---------------
MSFT   | 40%    | $4,000   | $380   | 10     | $3,800
AAPL   | 35%    | $3,500   | $175   | 20     | $3,500
GOOGL  | 15%    | $1,500   | $140   | 10     | $1,400
AMZN   | 10%    | $1,000   | $150   | 6      | $900

Cash Remainder: $400
```

## 🧪 Testing

Test individual modules:

```bash
# Test data fetching
cd modules
python data_module.py

# Test feature engineering
python feature_engineering.py

# Test ML models
python ml_model.py

# Test optimizer
python optimizer.py
```

## 📝 Logging

Logs are stored in `logs/portfolio_system.log`:

```bash
# View logs
cat logs/portfolio_system.log

# Monitor logs in real-time
tail -f logs/portfolio_system.log
```

## 🔧 Troubleshooting

### Issue: "Module not found"
**Solution**: Ensure you're in the correct directory and virtual environment is activated

```bash
pwd  # Should show portfolio_optimization_system directory
which python  # Should show virtual environment path
```

### Issue: "API connection refused"
**Solution**: Ensure API server is running

```bash
python api_server.py
```

### Issue: "Data fetching failed"
**Solution**: Check internet connection and ticker validity

```python
from modules.data_module import DataFetcher
fetcher = DataFetcher()
valid_tickers = fetcher.validate_tickers(["AAPL", "INVALID"])
print(valid_tickers)  # Should only show valid tickers
```

### Issue: "Optimization fails"
**Solution**: Ensure sufficient historical data (minimum 1 year recommended)

## 🚀 Deployment Options

### Option 1: Local Deployment
Already covered in the usage section above.

### Option 2: Docker Deployment
```dockerfile
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Run both API and dashboard
CMD ["bash", "-c", "python api_server.py & streamlit run dashboard.py"]
```

### Option 3: Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Connect repository
4. Deploy `dashboard.py`

### Option 4: Heroku Deployment
```bash
# Create Procfile
web: uvicorn api_server:app --host 0.0.0.0 --port $PORT

# Deploy
heroku create portfolio-optimizer
git push heroku main
```

## 📚 Technical Details

### Machine Learning Models
- **Algorithm**: Random Forest Regressor
- **Features**: 30+ per asset (moving averages, RSI, Bollinger Bands, momentum indicators)
- **Targets**: Next-period returns and volatility
- **Training**: 80/20 train-test split with cross-validation

### Optimization Algorithm
- **Method**: Sequential Least Squares Programming (SLSQP)
- **Objective Functions**:
  - Max Sharpe: Minimize negative Sharpe ratio
  - Min Volatility: Minimize portfolio variance
- **Constraints**: Weights sum to 1, bounded weights

### Performance Metrics
- **Expected Return**: Annualized predicted return
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: (Return - Risk-free rate) / Volatility

## 🤝 Contributing

This is a production-ready system. To extend:

1. Add new features in `feature_engineering.py`
2. Implement new optimizers in `optimizer.py`
3. Add API endpoints in `api_server.py`
4. Enhance dashboard in `dashboard.py`

## 📄 License

This project is provided as-is for educational and professional use.

## ⚠️ Disclaimer

This system is for educational and informational purposes only. It should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

## 📧 Support

For issues or questions:
- Check logs in `logs/portfolio_system.log`
- Review documentation in this README
- Test individual modules using the test scripts

## 🎯 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch dashboard (easiest)
streamlit run dashboard.py

# OR launch API server
python api_server.py

# 3. Access in browser
# Dashboard: http://localhost:8501
# API: http://localhost:8000/docs
```

---

**Built with**: Python, scikit-learn, scipy, FastAPI, Streamlit, yfinance, pandas, numpy

**Version**: 1.0.0

**Status**: Production-ready ✅
