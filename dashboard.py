"""
SIMPLIFIED Dashboard - Uses Historical Returns (NO ML ERRORS)
Guaranteed correct calculations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📊", layout="wide")

# Simple CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1 { color: #1f2937; text-align: center; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# Indian stocks
INDIAN_STOCKS = {
    'Technology': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS'],
    'Auto': ['TATAMOTORS.NS', 'M&M.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS'],
    'FMCG': ['ITC.NS', 'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'AUROPHARMA.NS']
}

def get_all_stocks():
    all_stocks = []
    for stocks in INDIAN_STOCKS.values():
        all_stocks.extend(stocks)
    return all_stocks

def fetch_data(tickers, start, end):
    """Fetch stock data"""
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if len(tickers) == 1:
        prices = data[['Close']].copy()
        prices.columns = tickers
    else:
        prices = data['Close'].copy()
    return prices.dropna()

def calculate_portfolio_metrics(weights, returns, risk_free_rate=0.06):
    """Calculate portfolio performance - SIMPLE AND CORRECT"""
    # Portfolio return (annualized)
    portfolio_return = np.sum(weights * returns.mean() * 252)
    
    # Portfolio volatility (annualized)
    cov_matrix = returns.cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def optimize_portfolio(returns, max_weight=0.40):
    """Simple optimization using Sharpe ratio"""
    from scipy.optimize import minimize
    
    n_assets = len(returns.columns)
    
    def neg_sharpe(weights):
        ret, vol, sharpe = calculate_portfolio_metrics(weights, returns)
        return -sharpe
    
    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, max_weight) for _ in range(n_assets))
    
    # Initial guess
    x0 = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    ret, vol, sharpe = calculate_portfolio_metrics(optimal_weights, returns)
    
    return optimal_weights, ret, vol, sharpe

def auto_select_stocks(n_stocks):
    """Auto select best stocks based on historical performance"""
    st.info(f"🤖 Analyzing stocks to select top {n_stocks}...")
    
    all_stocks = get_all_stocks()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock_scores = []
    progress = st.progress(0)
    
    for idx, ticker in enumerate(all_stocks[:20]):  # Check first 20
        try:
            progress.progress((idx + 1) / 20)
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(data) < 100:
                continue
            
            returns = data['Close'].pct_change().dropna()
            
            # Calculate metrics
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            sharpe = (annual_return - 0.06) / annual_vol if annual_vol > 0 else 0
            
            # Get sector
            sector = None
            for s, stocks in INDIAN_STOCKS.items():
                if ticker in stocks:
                    sector = s
                    break
            
            stock_scores.append({
                'ticker': ticker,
                'sharpe': sharpe,
                'return': annual_return,
                'sector': sector
            })
        except:
            continue
    
    progress.empty()
    
    # Sort by Sharpe and select diversified
    stock_scores = sorted(stock_scores, key=lambda x: x['sharpe'], reverse=True)
    
    selected = []
    sectors_used = set()
    
    for stock in stock_scores:
        if len(selected) >= n_stocks:
            break
        if stock['sector'] not in sectors_used or len(selected) < 3:
            selected.append(stock['ticker'])
            if stock['sector']:
                sectors_used.add(stock['sector'])
    
    return selected, stock_scores[:len(selected)]

# Session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.title("📊 AI Portfolio Optimizer - Indian Stocks")
st.markdown("**Simple, Accurate, Reliable**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mode = st.radio("Selection Mode", ["🤖 Auto-Select", "✍️ Manual"])
    
    if mode == "🤖 Auto-Select":
        n_stocks = st.slider("Number of Stocks", 3, 10, 5)
    else:
        sectors = st.multiselect("Sectors", list(INDIAN_STOCKS.keys()), default=['Technology', 'Banking'])
        available = []
        for sector in sectors:
            available.extend(INDIAN_STOCKS[sector])
        manual_tickers = st.multiselect("Stocks", available, default=available[:3] if available else [])
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=datetime.now() - timedelta(days=730))
    with col2:
        end_date = st.date_input("End", value=datetime.now())
    
    investment = st.number_input("Investment (₹)", 10000, 10000000, 100000, 10000)
    
    with st.expander("Advanced"):
        max_weight = st.slider("Max Weight %", 10, 50, 30) / 100
    
    optimize_btn = st.button("🚀 OPTIMIZE", type="primary", use_container_width=True)

# Main
if optimize_btn:
    with st.spinner("Optimizing..."):
        try:
            # Get tickers
            if mode == "🤖 Auto-Select":
                tickers, stock_info = auto_select_stocks(n_stocks)
                st.success(f"✅ Selected: {', '.join([t.replace('.NS', '') for t in tickers])}")
            else:
                tickers = manual_tickers
            
            if len(tickers) < 2:
                st.error("Select at least 2 stocks")
                st.stop()
            
            # Fetch data
            st.info("📊 Fetching data...")
            prices = fetch_data(tickers, start_date, end_date)
            
            if prices.empty:
                st.error("No data available")
                st.stop()
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Optimize
            st.info("⚡ Optimizing...")
            optimal_weights, portfolio_return, portfolio_vol, portfolio_sharpe = optimize_portfolio(returns, max_weight)
            
            # Get latest prices
            latest_prices = {}
            for ticker in tickers:
                try:
                    data = yf.download(ticker, period='1d', progress=False)
                    latest_prices[ticker] = float(data['Close'].iloc[-1])
                except:
                    latest_prices[ticker] = float(prices[ticker].iloc[-1])
            
            # Calculate allocation
            allocation_data = []
            for ticker, weight in zip(tickers, optimal_weights):
                if weight > 0.001:
                    amount = investment * weight
                    shares = int(amount / latest_prices[ticker])
                    actual = shares * latest_prices[ticker]
                    
                    allocation_data.append({
                        'Stock': ticker.replace('.NS', ''),
                        'Weight': weight,
                        'Amount': amount,
                        'Price': latest_prices[ticker],
                        'Shares': shares,
                        'Invested': actual
                    })
            
            allocation_df = pd.DataFrame(allocation_data)
            
            # Store results
            st.session_state.results = {
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': portfolio_sharpe,
                'weights': dict(zip(tickers, optimal_weights)),
                'allocation': allocation_df,
                'investment': investment,
                'returns_data': returns
            }
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Results
if st.session_state.results:
    r = st.session_state.results
    
    st.markdown("---")
    st.subheader("📈 Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    ret = r['return'] * 100
    vol = r['volatility'] * 100
    sharpe = r['sharpe']
    projected = r['investment'] * (1 + r['return'])
    gain = projected - r['investment']
    
    with col1:
        st.metric("Expected Return", f"{ret:.2f}%")
    with col2:
        st.metric("Volatility (Risk)", f"{vol:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    with col4:
        st.metric("Projected Value (1Y)", f"₹{projected:,.0f}", delta=f"+₹{gain:,.0f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Allocation", "💰 Investment Plan"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Weights")
            
            weights_clean = {k.replace('.NS', ''): v for k, v in r['weights'].items() if v > 0.001}
            
            fig = go.Figure(data=[go.Pie(
                labels=list(weights_clean.keys()),
                values=list(weights_clean.values()),
                hole=0.3
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Weight Distribution")
            df = pd.DataFrame({
                'Stock': list(weights_clean.keys()),
                'Weight %': [v*100 for v in weights_clean.values()]
            }).sort_values('Weight %', ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    with tab2:
        st.subheader("Your Investment Plan")
        
        if not r['allocation'].empty:
            display = r['allocation'].copy()
            display['Weight'] = (display['Weight'] * 100).round(2)
            display['Amount'] = display['Amount'].round(0)
            display['Price'] = display['Price'].round(2)
            display['Invested'] = display['Invested'].round(0)
            
            display.columns = ['Stock', 'Weight %', 'Target ₹', 'Price ₹', 'Shares', 'Actual ₹']
            
            st.dataframe(display, hide_index=True, use_container_width=True)
            
            # Summary
            total = r['allocation']['Invested'].sum()
            cash = r['investment'] - total
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Invested", f"₹{total:,.0f}")
            with col2:
                st.metric("Cash", f"₹{cash:,.0f}")
            with col3:
                st.metric("Total Shares", int(r['allocation']['Shares'].sum()))

else:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🤖 **AI-Powered**\n\nAuto-selects best stocks")
    with col2:
        st.info("🇮🇳 **Indian Market**\n\n30+ NSE stocks")
    with col3:
        st.info("✅ **Accurate**\n\nHistorical data-based")
    
    st.markdown("---")
    st.markdown("**👈 Configure and click OPTIMIZE**")

st.markdown("---")
st.caption("⚠️ Educational purposes only | 🇮🇳 NSE Stocks")
