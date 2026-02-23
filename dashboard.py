"""
Professional AI Portfolio Optimizer - Fixed Version
Clean design, correct calculations, Indian stocks
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_module import DataFetcher
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import MLPredictor
from modules.optimizer import PortfolioOptimizer
import config

# Page config
st.set_page_config(
    page_title="AI Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

# Simple, clean CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f2937;
        text-align: center;
        padding: 20px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Indian Stock Universe (NSE symbols with .NS suffix for yfinance)
INDIAN_STOCKS = {
    'Technology': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'NTPC.NS'],
    'Auto': ['TATAMOTORS.NS', 'M&M.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS'],
    'FMCG': ['ITC.NS', 'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'AUROPHARMA.NS'],
    'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'COALINDIA.NS'],
    'Cement': ['ULTRACEMCO.NS', 'AMBUJACEM.NS', 'SHREECEM.NS', 'ACC.NS', 'JKCEMENT.NS']
}

def get_all_indian_stocks():
    """Get all Indian stocks"""
    all_stocks = []
    for sector_stocks in INDIAN_STOCKS.values():
        all_stocks.extend(sector_stocks)
    return all_stocks

def auto_select_indian_stocks(n_stocks):
    """Auto select best Indian stocks using ML"""
    st.info("🤖 Analyzing top Indian stocks...")
    
    all_stocks = get_all_indian_stocks()
    
    # Date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    fetcher = DataFetcher()
    
    # Validate stocks
    st.write("Validating stocks...")
    valid_stocks = fetcher.validate_tickers(all_stocks[:20])  # Check first 20 for speed
    
    if len(valid_stocks) < n_stocks:
        st.warning(f"Only {len(valid_stocks)} stocks are available")
        n_stocks = len(valid_stocks)
    
    # Quick analysis
    stock_scores = []
    progress = st.progress(0)
    
    for idx, ticker in enumerate(valid_stocks):
        try:
            progress.progress((idx + 1) / len(valid_stocks))
            
            prices = fetcher.fetch_data([ticker], start_date, end_date)
            returns = fetcher.calculate_returns()
            
            # Calculate annualized metrics
            mean_return = returns.mean().iloc[0]
            std_return = returns.std().iloc[0]
            
            annual_return = mean_return * 252
            annual_vol = std_return * np.sqrt(252)
            sharpe = (annual_return - 0.06) / annual_vol if annual_vol > 0 else 0  # 6% risk-free rate for India
            
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
                'volatility': annual_vol,
                'sector': sector
            })
        except Exception as e:
            continue
    
    progress.empty()
    
    # Sort by Sharpe ratio
    stock_scores = sorted(stock_scores, key=lambda x: x['sharpe'], reverse=True)
    
    # Select diversified portfolio
    selected = []
    sectors_used = set()
    
    for stock in stock_scores:
        if len(selected) >= n_stocks:
            break
        
        # Ensure sector diversification
        if stock['sector'] not in sectors_used or len(selected) < 3:
            selected.append(stock['ticker'])
            if stock['sector']:
                sectors_used.add(stock['sector'])
    
    return selected, stock_scores[:n_stocks]

# Session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.title("📊 AI Portfolio Optimizer - Indian Stocks")
st.markdown("**Optimize your portfolio with Machine Learning**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mode = st.radio(
        "Selection Mode",
        ["🤖 Auto-Select (AI)", "✍️ Manual Selection"]
    )
    
    if mode == "🤖 Auto-Select (AI)":
        n_stocks = st.slider("Number of Stocks", 3, 10, 5)
        st.info(f"AI will select top {n_stocks} Indian stocks")
    else:
        sectors = st.multiselect(
            "Select Sectors",
            list(INDIAN_STOCKS.keys()),
            default=['Technology', 'Banking']
        )
        
        available = []
        for sector in sectors:
            available.extend(INDIAN_STOCKS[sector])
        
        manual_tickers = st.multiselect(
            "Select Stocks",
            available,
            default=available[:3] if available else []
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=730)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    investment = st.number_input(
        "Investment Amount (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    risk_profile = st.selectbox(
        "Risk Profile",
        ["Conservative", "Moderate", "Aggressive"]
    )
    
    with st.expander("Advanced Settings"):
        max_weight = st.slider("Max Weight per Stock (%)", 10, 50, 30) / 100
        n_simulations = st.slider("Simulations", 1000, 10000, 3000)
    
    st.markdown("---")
    optimize_btn = st.button("🚀 OPTIMIZE PORTFOLIO", type="primary")

# Main content
if optimize_btn:
    with st.spinner("Optimizing portfolio..."):
        try:
            # Get tickers
            if mode == "🤖 Auto-Select (AI)":
                tickers, stock_info = auto_select_indian_stocks(n_stocks)
                
                st.success(f"✅ Selected {len(tickers)} stocks")
                
                # Display selected stocks
                cols = st.columns(len(tickers))
                for idx, ticker in enumerate(tickers):
                    with cols[idx]:
                        clean_name = ticker.replace('.NS', '')
                        sector = stock_info[idx]['sector'] if idx < len(stock_info) else 'N/A'
                        st.metric(clean_name, sector)
            else:
                tickers = manual_tickers
            
            if len(tickers) < 2:
                st.error("Please select at least 2 stocks")
                st.stop()
            
            # Progress
            progress = st.progress(0)
            status = st.empty()
            
            # Initialize
            fetcher = DataFetcher()
            engineer = FeatureEngineer()
            predictor = MLPredictor()
            optimizer = PortfolioOptimizer()
            
            # Step 1: Fetch data
            status.text("📊 Fetching data...")
            progress.progress(20)
            
            prices = fetcher.fetch_data(
                tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            returns = fetcher.calculate_returns()
            
            # Step 2: Features
            status.text("🔧 Creating features...")
            progress.progress(40)
            
            features = engineer.create_features(prices, returns)
            future_returns, future_vol = engineer.create_target_variables(returns)
            
            # Step 3: Train
            status.text("🤖 Training models...")
            progress.progress(60)
            
            predictor.train_models(features, future_returns, future_vol)
            expected_returns, expected_vol = predictor.predict(features, tickers)
            
            # Step 4: Optimize
            status.text("⚡ Optimizing...")
            progress.progress(80)
            
            cov_matrix = returns.cov() * 252
            optimizer.set_parameters(expected_returns, cov_matrix)
            
            max_sharpe = optimizer.optimize_max_sharpe(max_weight=max_weight)
            min_vol = optimizer.optimize_min_volatility(max_weight=max_weight)
            frontier = optimizer.generate_efficient_frontier(n_portfolios=n_simulations)
            
            # Step 5: Allocation
            status.text("💰 Calculating allocation...")
            progress.progress(95)
            
            latest_prices = fetcher.get_latest_prices(tickers)
            allocation = optimizer.calculate_capital_allocation(
                max_sharpe['weights'],
                investment,
                latest_prices
            )
            
            summary = fetcher.get_summary_statistics(returns)
            
            progress.progress(100)
            status.empty()
            progress.empty()
            
            # Store results
            st.session_state.results = {
                'max_sharpe': max_sharpe,
                'min_vol': min_vol,
                'frontier': frontier,
                'allocation': allocation,
                'summary': summary,
                'tickers': tickers,
                'investment': investment
            }
            
            st.success("✅ Optimization complete!")
            st.balloons()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display results
if st.session_state.results:
    r = st.session_state.results
    
    st.markdown("---")
    st.subheader("📈 Optimization Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    ret = r['max_sharpe']['expected_return'] * 100
    vol = r['max_sharpe']['volatility'] * 100
    sharpe = r['max_sharpe']['sharpe_ratio']
    proj = r['investment'] * (1 + r['max_sharpe']['expected_return'])
    gain = proj - r['investment']
    
    with col1:
        st.metric("Expected Return", f"{ret:.2f}%", delta=f"{ret-10:.1f}% vs avg")
    with col2:
        st.metric("Volatility", f"{vol:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.3f}", delta="Good" if sharpe > 1 else "Fair")
    with col4:
        st.metric("Projected Value (1Y)", f"₹{proj:,.0f}", delta=f"+₹{gain:,.0f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Allocation", "💰 Investment Plan", "📈 Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Weights")
            
            weights = r['max_sharpe']['weights']
            
            # Clean ticker names for display
            clean_weights = {k.replace('.NS', ''): v for k, v in weights.items()}
            
            fig = go.Figure(data=[go.Pie(
                labels=list(clean_weights.keys()),
                values=list(clean_weights.values()),
                hole=0.3,
                textinfo='label+percent'
            )])
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Weight Distribution")
            
            df = pd.DataFrame({
                'Stock': [k.replace('.NS', '') for k in weights.keys()],
                'Weight (%)': [v*100 for v in weights.values()]
            }).sort_values('Weight (%)', ascending=False)
            
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    with tab2:
        st.subheader("Your Investment Plan")
        
        if not r['allocation'].empty:
            alloc = r['allocation'].copy()
            
            # Clean ticker names
            alloc['Ticker'] = alloc['Ticker'].str.replace('.NS', '')
            
            # Format the dataframe
            display_df = alloc[['Ticker', 'Shares', 'Latest_Price', 'Actual_Amount', 'Weight']].copy()
            display_df.columns = ['Stock', 'Shares', 'Price (₹)', 'Amount (₹)', 'Weight (%)']
            display_df['Weight (%)'] = (display_df['Weight (%)'] * 100).round(2)
            display_df['Price (₹)'] = display_df['Price (₹)'].round(2)
            display_df['Amount (₹)'] = display_df['Amount (₹)'].round(0)
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # Summary
            total = alloc['Actual_Amount'].sum()
            cash = r['investment'] - total
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Invested", f"₹{total:,.0f}")
            with col2:
                st.metric("Cash", f"₹{cash:,.0f}")
            with col3:
                st.metric("Shares", f"{int(alloc['Shares'].sum())}")
    
    with tab3:
        st.subheader("Efficient Frontier")
        
        fig = go.Figure()
        
        # All portfolios
        fig.add_trace(go.Scatter(
            x=r['frontier']['volatility']*100,
            y=r['frontier']['return']*100,
            mode='markers',
            marker=dict(
                size=5,
                color=r['frontier']['sharpe_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe")
            ),
            name='Portfolios'
        ))
        
        # Optimal
        fig.add_trace(go.Scatter(
            x=[vol],
            y=[ret],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Optimal'
        ))
        
        fig.update_layout(
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Return (%)",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome screen
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🤖 **AI-Powered**\n\nMachine learning selects best stocks")
    
    with col2:
        st.info("🇮🇳 **Indian Stocks**\n\n40+ stocks from NSE")
    
    with col3:
        st.info("⚡ **Fast**\n\nResults in 60 seconds")
    
    st.markdown("---")
    st.markdown("**👈 Configure your portfolio and click OPTIMIZE**")

# Footer
st.markdown("---")
st.caption("⚠️ For educational purposes only. Not financial advice. | 🇮🇳 Indian Stock Market")
