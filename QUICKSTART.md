# ⚡ Quick Start Guide (5 Minutes)

Get the AI Portfolio Optimization System running in 5 minutes!

## 🎯 For Complete Beginners

### Step 1: Open Terminal/Command Prompt (30 seconds)

**Windows:**
- Press `Win + R`
- Type `cmd` and press Enter

**Mac:**
- Press `Cmd + Space`
- Type `terminal` and press Enter

**Linux:**
- Press `Ctrl + Alt + T`

### Step 2: Check Python (30 seconds)

```bash
python --version
```

**Should show Python 3.8 or higher.**

If not, download from: https://www.python.org/downloads/

### Step 3: Navigate to Project (30 seconds)

```bash
cd path/to/portfolio_optimization_system
```

Replace `path/to/` with your actual path.

### Step 4: Install Everything (3 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Wait for installation to complete (~3 minutes)**

### Step 5: Launch! (10 seconds)

```bash
python run.py
```

**Select option 1** to launch the dashboard.

Your browser will open automatically at `http://localhost:8501`

## 🚀 For Developers

### Ultra-Quick Setup

```bash
# One-liner setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard.py

# OR launch API
python api_server.py
```

### Quick Test

```bash
python -c "from modules.data_module import DataFetcher; f = DataFetcher(); print('✅ System ready!')"
```

## 📱 First Portfolio Optimization

Once dashboard is open:

1. **Keep default tickers** (AAPL, MSFT, GOOGL, etc.)
2. **Set investment**: $10,000
3. **Click**: "Optimize Portfolio" button
4. **Wait**: ~30 seconds
5. **View**: Your optimized portfolio!

## 🎓 What You'll See

- **Maximum Sharpe Portfolio**: Best risk-adjusted returns
- **Minimum Volatility Portfolio**: Lowest risk option
- **Efficient Frontier**: Visual risk-return tradeoff
- **Capital Allocation**: Exact shares to buy
- **ML Predictions**: AI-powered forecasts

## ⚙️ Quick Customization

### Change Stocks

Replace the default tickers with your own:
```
TSLA, NVDA, AMD, INTC, QCOM
```

### Adjust Investment

Use slider to set any amount:
- Minimum: $1,000
- Maximum: $10,000,000

### Risk Profile

Choose your preference:
- **Conservative**: Lower risk, stable returns
- **Moderate**: Balanced approach
- **Aggressive**: Higher risk, higher potential returns

## 🔧 Troubleshooting

### "Python not found"
Download Python: https://www.python.org/downloads/

### "pip not found"
```bash
python -m ensurepip --upgrade
```

### "Module not found"
```bash
# Ensure virtual environment is active
# You should see (venv) in your terminal prompt
```

### "Port already in use"
```bash
# Kill existing process or use different port
streamlit run dashboard.py --server.port 8502
```

### "Data fetching failed"
- Check internet connection
- Try different tickers
- Wait 1 minute and retry

## 📊 Using the API Instead

### Start API Server
```bash
python api_server.py
```

### Test in Browser
Open: `http://localhost:8000/docs`

### Quick API Test
```bash
curl -X POST "http://localhost:8000/api/validate-tickers" \
  -H "Content-Type: application/json" \
  -d '["AAPL", "MSFT", "GOOGL"]'
```

## 💡 Pro Tips

1. **Start Simple**: Use 4-5 well-known tickers first
2. **Use Recent Data**: Last 1-2 years works best
3. **Diversify**: Mix different sectors (tech, finance, healthcare)
4. **Check Logs**: `logs/portfolio_system.log` for details
5. **Save Results**: Screenshot or export data from dashboard

## 🎯 Next Steps

After your first optimization:

1. **Try Different Tickers**: Test various combinations
2. **Adjust Constraints**: Change max weight (slider in advanced settings)
3. **Stress Test**: See how portfolio performs in crashes
4. **Compare Profiles**: Conservative vs Aggressive
5. **Read Full Docs**: Check README.md for advanced features

## 📚 Full Documentation

- **README.md**: Complete feature documentation
- **SETUP_GUIDE.md**: Detailed setup instructions
- **example_usage.py**: 5 practical examples
- **API Docs**: http://localhost:8000/docs (when server running)

## ⏱️ Time Summary

- ✅ Setup: 4 minutes
- ✅ First optimization: 30 seconds
- ✅ Total: **< 5 minutes**

## 🎉 Success!

If you see:
- ✅ Dashboard opens in browser
- ✅ Can enter tickers
- ✅ Optimization completes
- ✅ Results display

**You're ready to optimize portfolios! 🚀**

---

## 🆘 Still Stuck?

1. Check logs: `cat logs/portfolio_system.log`
2. Run test: `python run.py` → Select "3. Run Quick Test"
3. Review SETUP_GUIDE.md for detailed help
4. Test individual modules: `cd modules && python data_module.py`

---

**Remember**: 
- Keep virtual environment activated (`venv` in prompt)
- Use Ctrl+C to stop servers
- Check internet connection for data fetching

**Happy Optimizing! 📈**
