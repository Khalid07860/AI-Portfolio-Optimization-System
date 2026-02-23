"""
Professional AI Portfolio Optimizer Dashboard
Industry-ready with auto-stock selection and interactive features
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

# Page configuration
st.set_page_config(
    page_title="AI Portfolio Optimizer Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Cards */
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Title */
    .main-title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        animation: fadeIn 1s ease-in;
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    
    .badge-success { background: #10b981; color: white; }
    .badge-warning { background: #f59e0b; color: white; }
    .badge-info { background: #3b82f6; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Stock universe for auto-selection
STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'ADBE', 'CRM'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SCHW', 'USB'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN'],
    'Consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'COST', 'PG'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'Industrial': ['BA', 'CAT', 'GE', 'UPS', 'HON', 'RTX', 'LMT', 'MMM', 'DE', 'EMR']
}

def get_all_stocks():
    """Get all stocks from universe"""
    all_stocks = []
    for sector_stocks in STOCK_UNIVERSE.values():
        all_stocks.extend(sector_stocks)
    return all_stocks

def auto_select_stocks(n_stocks, investment_amount, risk_profile):
    """
    Automatically select best stocks using ML
    Returns diversified portfolio across sectors
    """
    st.info("🤖 AI is analyzing 60+ stocks across 6 sectors to find the best opportunities...")
    
    # Get all available stocks
    all_stocks = get_all_stocks()
    
    # Fetch recent data for quick analysis
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    fetcher = DataFetcher()
    
    # Validate stocks
    valid_stocks = fetcher.validate_tickers(all_stocks)
    st.success(f"✅ Found {len(valid_stocks)} valid stocks")
    
    # Quick score calculation
    stock_scores = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(valid_stocks[:30]):  # Analyze top 30 for speed
        try:
            status_text.text(f"Analyzing {ticker}... ({idx+1}/30)")
            progress_bar.progress((idx + 1) / 30)
            
            prices = fetcher.fetch_data([ticker], start_date, end_date)
            returns = fetcher.calculate_returns()
            
            # Calculate metrics
            annual_return = returns.mean().iloc[0] * 252
            annual_vol = returns.std().iloc[0] * np.sqrt(252)
            sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
            
            # Get sector
            sector = None
            for s, stocks in STOCK_UNIVERSE.items():
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
        except:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort by Sharpe ratio
    stock_scores = sorted(stock_scores, key=lambda x: x['sharpe'], reverse=True)
    
    # Select diversified portfolio
    selected = []
    sectors_used = set()
    
    # Ensure sector diversification
    for stock in stock_scores:
        if len(selected) >= n_stocks:
            break
        
        # Prefer stocks from unused sectors
        if stock['sector'] not in sectors_used or len(selected) < 3:
            selected.append(stock['ticker'])
            if stock['sector']:
                sectors_used.add(stock['sector'])
    
    return selected, stock_scores[:n_stocks]

# Initialize session state
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []

# Header
st.markdown('<h1 class="main-title">🤖 AI Portfolio Optimizer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Machine Learning | Industry-Grade Portfolio Management</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Mode selection
    mode = st.radio(
        "📊 Selection Mode",
        ["🤖 Auto-Select (AI Powered)", "✍️ Manual Selection"],
        help="Let AI choose the best stocks or select manually"
    )
    
    if mode == "🤖 Auto-Select (AI Powered)":
        st.markdown("---")
        st.markdown("#### 🎯 AI Configuration")
        
        n_stocks = st.slider(
            "Number of Stocks",
            min_value=3,
            max_value=15,
            value=8,
            help="AI will select the best stocks from 60+ options"
        )
        
        st.info(f"🤖 AI will analyze **60+ stocks** across 6 sectors and select the **top {n_stocks}** for your portfolio")
        
    else:
        st.markdown("---")
        st.markdown("#### 📝 Manual Selection")
        
        # Sector-based selection
        selected_sectors = st.multiselect(
            "Choose Sectors",
            list(STOCK_UNIVERSE.keys()),
            default=['Technology', 'Finance', 'Healthcare']
        )
        
        available_stocks = []
        for sector in selected_sectors:
            available_stocks.extend(STOCK_UNIVERSE[sector])
        
        manual_tickers = st.multiselect(
            "Select Stocks",
            available_stocks,
            default=available_stocks[:5] if available_stocks else []
        )
    
    st.markdown("---")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "📅 Start",
            value=datetime.now() - timedelta(days=730)
        )
    with col2:
        end_date = st.date_input(
            "📅 End",
            value=datetime.now()
        )
    
    # Investment amount
    investment_amount = st.number_input(
        "💰 Investment Amount ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000,
        format="%d"
    )
    
    # Risk profile
    risk_profile = st.selectbox(
        "⚖️ Risk Profile",
        ["Conservative 🛡️", "Moderate ⚖️", "Aggressive 🚀"],
        index=1
    )
    
    risk_profile_clean = risk_profile.split()[0].lower()
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        max_weight = st.slider(
            "Max Weight per Asset (%)",
            10, 100, 40, 5
        ) / 100
        
        allow_short = st.checkbox("Allow Short Selling", value=False)
        
        n_simulations = st.number_input(
            "Monte Carlo Simulations",
            1000, 20000, 5000, 1000
        )
    
    st.markdown("---")
    
    # Optimize button
    optimize_btn = st.button(
        "🚀 OPTIMIZE PORTFOLIO",
        use_container_width=True,
        type="primary"
    )

# Main content
if optimize_btn:
    with st.spinner("🔮 AI is working its magic..."):
        try:
            # Determine tickers
            if mode == "🤖 Auto-Select (AI Powered)":
                st.markdown("### 🤖 AI Stock Selection")
                selected_tickers, stock_analysis = auto_select_stocks(
                    n_stocks, investment_amount, risk_profile_clean
                )
                st.session_state.selected_stocks = selected_tickers
                
                # Display selected stocks
                st.success(f"✅ AI selected {len(selected_tickers)} optimal stocks")
                
                cols = st.columns(len(selected_tickers))
                for idx, ticker in enumerate(selected_tickers):
                    with cols[idx]:
                        st.markdown(f"""
                        <div style='background:white; padding:15px; border-radius:10px; text-align:center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h3 style='color:#667eea; margin:0;'>{ticker}</h3>
                            <p style='color:#666; font-size:12px; margin:5px 0 0 0;'>{stock_analysis[idx]['sector']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                tickers = selected_tickers
            else:
                tickers = manual_tickers
                st.session_state.selected_stocks = tickers
            
            if len(tickers) < 2:
                st.error("❌ Please select at least 2 stocks")
                st.stop()
            
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Initialize components
            data_fetcher = DataFetcher()
            feature_engineer = FeatureEngineer()
            ml_predictor = MLPredictor()
            optimizer = PortfolioOptimizer()
            
            # Fetch data
            status_text.markdown("### 📊 Step 1/5: Fetching market data...")
            progress_bar.progress(20)
            
            prices = data_fetcher.fetch_data(
                tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            returns = data_fetcher.calculate_returns()
            
            # Feature engineering
            status_text.markdown("### 🔧 Step 2/5: Engineering features...")
            progress_bar.progress(40)
            
            features = feature_engineer.create_features(prices, returns)
            future_returns, future_volatility = feature_engineer.create_target_variables(returns)
            
            # Train ML models
            status_text.markdown("### 🤖 Step 3/5: Training AI models...")
            progress_bar.progress(60)
            
            training_metrics = ml_predictor.train_models(features, future_returns, future_volatility)
            expected_returns, expected_volatility = ml_predictor.predict(features, tickers)
            
            # Optimize
            status_text.markdown("### ⚡ Step 4/5: Optimizing portfolio...")
            progress_bar.progress(80)
            
            cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
            optimizer.set_parameters(expected_returns, cov_matrix)
            
            max_sharpe = optimizer.optimize_max_sharpe(max_weight=max_weight, allow_short=allow_short)
            min_vol = optimizer.optimize_min_volatility(max_weight=max_weight, allow_short=allow_short)
            frontier = optimizer.generate_efficient_frontier(n_portfolios=n_simulations, max_weight=max_weight)
            
            # Calculate allocation
            status_text.markdown("### 💰 Step 5/5: Calculating allocation...")
            progress_bar.progress(95)
            
            latest_prices = data_fetcher.get_latest_prices(tickers)
            
            max_sharpe_allocation = optimizer.calculate_capital_allocation(
                max_sharpe['weights'],
                investment_amount,
                latest_prices
            )
            
            summary_stats = data_fetcher.get_summary_statistics(returns)
            
            progress_bar.progress(100)
            status_text.markdown("### ✅ Optimization Complete!")
            
            # Store results
            st.session_state.results = {
                'max_sharpe': max_sharpe,
                'min_vol': min_vol,
                'frontier': frontier,
                'max_sharpe_allocation': max_sharpe_allocation,
                'summary_stats': summary_stats,
                'expected_returns': expected_returns,
                'training_metrics': training_metrics,
                'tickers': tickers
            }
            st.session_state.optimization_complete = True
            
            # Clear progress
            progress_container.empty()
            
            st.balloons()
            st.success("🎉 Portfolio optimized successfully!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display results
if st.session_state.optimization_complete and st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📈 Expected Return",
            f"{results['max_sharpe']['expected_return']*100:.2f}%",
            delta=f"+{(results['max_sharpe']['expected_return']-0.10)*100:.1f}% vs market"
        )
    
    with col2:
        st.metric(
            "📊 Volatility",
            f"{results['max_sharpe']['volatility']*100:.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "⭐ Sharpe Ratio",
            f"{results['max_sharpe']['sharpe_ratio']:.3f}",
            delta="Excellent" if results['max_sharpe']['sharpe_ratio'] > 1 else "Good"
        )
    
    with col4:
        projected_value = investment_amount * (1 + results['max_sharpe']['expected_return'])
        st.metric(
            "💵 Projected Value (1Y)",
            f"${projected_value:,.0f}",
            delta=f"+${(projected_value - investment_amount):,.0f}"
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Overview",
        "💰 Investment Plan",
        "📈 Performance Analysis",
        "🧪 Stress Testing"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥧 Optimal Allocation")
            
            # Pie chart
            weights = results['max_sharpe']['weights']
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='label+percent',
                textfont=dict(size=14, color='white'),
                hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>'
            )])
            
            fig.update_layout(
                showlegend=True,
                height=400,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Weight Distribution")
            
            weights_df = pd.DataFrame({
                'Stock': list(weights.keys()),
                'Weight': [w*100 for w in weights.values()]
            }).sort_values('Weight', ascending=True)
            
            fig = go.Figure(go.Bar(
                y=weights_df['Stock'],
                x=weights_df['Weight'],
                orientation='h',
                marker=dict(
                    color=weights_df['Weight'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Weight %")
                ),
                text=[f"{w:.1f}%" for w in weights_df['Weight']],
                textposition='auto'
            ))
            
            fig.update_layout(
                xaxis_title="Weight (%)",
                yaxis_title="",
                height=400,
                margin=dict(t=30, b=30, l=80, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 💰 Your Investment Plan")
        
        if not results['max_sharpe_allocation'].empty:
            allocation = results['max_sharpe_allocation']
            
            # Display as cards
            cols = st.columns(3)
            for idx, row in allocation.iterrows():
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div style='background:white; padding:20px; border-radius:15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom:20px;'>
                        <h2 style='color:#667eea; margin:0 0 10px 0;'>{row['Ticker']}</h2>
                        <p style='font-size:24px; font-weight:bold; color:#333; margin:5px 0;'>{int(row['Shares'])} shares</p>
                        <p style='color:#666; margin:5px 0;'>@ ${row['Latest_Price']:.2f} each</p>
                        <p style='font-size:20px; font-weight:bold; color:#10b981; margin:10px 0 0 0;'>${row['Actual_Amount']:,.2f}</p>
                        <p style='color:#999; font-size:12px;'>{row['Weight']*100:.1f}% of portfolio</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Summary
            total = allocation['Actual_Amount'].sum()
            cash = investment_amount - total
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💵 Total Invested", f"${total:,.2f}")
            with col2:
                st.metric("💰 Cash Remaining", f"${cash:,.2f}")
            with col3:
                st.metric("📊 Total Shares", f"{int(allocation['Shares'].sum())}")
    
    with tab3:
        st.markdown("### 📈 Efficient Frontier")
        
        # Efficient frontier plot
        fig = go.Figure()
        
        # All portfolios
        fig.add_trace(go.Scatter(
            x=results['frontier']['volatility'] * 100,
            y=results['frontier']['return'] * 100,
            mode='markers',
            marker=dict(
                size=6,
                color=results['frontier']['sharpe_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe<br>Ratio")
            ),
            text=[f"Sharpe: {s:.3f}" for s in results['frontier']['sharpe_ratio']],
            hovertemplate='<b>Return:</b> %{y:.2f}%<br><b>Vol:</b> %{x:.2f}%<br>%{text}<extra></extra>',
            name='Portfolios'
        ))
        
        # Max Sharpe
        fig.add_trace(go.Scatter(
            x=[results['max_sharpe']['volatility'] * 100],
            y=[results['max_sharpe']['expected_return'] * 100],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
            name='Max Sharpe',
            hovertemplate='<b>Optimal Portfolio</b><br>Return: %{y:.2f}%<br>Vol: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk-Return Tradeoff",
            xaxis_title="Volatility (Risk) %",
            yaxis_title="Expected Return %",
            height=500,
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.9)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🧪 Stress Test Your Portfolio")
        
        scenario = st.selectbox(
            "Select Scenario",
            ["Market Crash (-10%)", "Volatility Spike (+50%)", "Recession", "Bull Market"]
        )
        
        if st.button("Run Stress Test"):
            scenario_map = {
                "Market Crash (-10%)": "market_crash",
                "Volatility Spike (+50%)": "volatility_spike"
            }
            
            if scenario in scenario_map:
                stress = optimizer.stress_test(
                    results['max_sharpe']['weights'],
                    scenario_map[scenario]
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    change = stress['return_change'] * 100
                    st.metric(
                        "Return Impact",
                        f"{change:+.2f}%",
                        delta=f"{change:+.2f}%"
                    )
                
                with col2:
                    vol_change = stress['volatility_change'] * 100
                    st.metric(
                        "Volatility Change",
                        f"{vol_change:+.2f}%",
                        delta=f"{vol_change:+.2f}%"
                    )
                
                with col3:
                    sharpe_change = stress['sharpe_change']
                    st.metric(
                        "Sharpe Change",
                        f"{sharpe_change:+.3f}",
                        delta=f"{sharpe_change:+.3f}"
                    )

else:
    # Welcome screen
    st.markdown("""
    <div style='background:white; padding:40px; border-radius:20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin:40px 0;'>
        <h2 style='color:#667eea; text-align:center;'>Welcome to the Future of Portfolio Management</h2>
        <p style='text-align:center; font-size:18px; color:#666; margin:20px 0;'>
            Our AI analyzes 60+ stocks across 6 sectors to build your perfect portfolio
        </p>
        
        <div style='display:grid; grid-template-columns: 1fr 1fr 1fr; gap:20px; margin:30px 0;'>
            <div style='text-align:center; padding:20px;'>
                <div style='font-size:48px; margin-bottom:10px;'>🤖</div>
                <h3 style='color:#667eea; margin:10px 0;'>AI-Powered</h3>
                <p style='color:#666;'>Machine learning selects optimal stocks</p>
            </div>
            
            <div style='text-align:center; padding:20px;'>
                <div style='font-size:48px; margin-bottom:10px;'>⚡</div>
                <h3 style='color:#667eea; margin:10px 0;'>Lightning Fast</h3>
                <p style='color:#666;'>Results in under 60 seconds</p>
            </div>
            
            <div style='text-align:center; padding:20px;'>
                <div style='font-size:48px; margin-bottom:10px;'>🎯</div>
                <h3 style='color:#667eea; margin:10px 0;'>Precision</h3>
                <p style='color:#666;'>Optimized for your risk profile</p>
            </div>
        </div>
        
        <p style='text-align:center; color:#999; margin-top:30px;'>
            👈 Configure your portfolio in the sidebar and click <strong>OPTIMIZE PORTFOLIO</strong> to begin
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:white; padding:20px;'>
    <p style='margin:5px 0;'>🤖 Powered by Advanced Machine Learning</p>
    <p style='margin:5px 0; font-size:12px;'>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
