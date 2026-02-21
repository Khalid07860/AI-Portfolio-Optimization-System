"""
Dashboard: Streamlit interactive dashboard for portfolio optimization
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
    page_title="AI Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None


def plot_efficient_frontier(frontier_data, max_sharpe_point, min_vol_point):
    """Plot efficient frontier"""
    fig = go.Figure()
    
    # Scatter plot of all portfolios
    fig.add_trace(go.Scatter(
        x=frontier_data['volatility'],
        y=frontier_data['return'],
        mode='markers',
        marker=dict(
            size=5,
            color=frontier_data['sharpe_ratio'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        text=[f"Sharpe: {s:.3f}" for s in frontier_data['sharpe_ratio']],
        name='Portfolios'
    ))
    
    # Max Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe_point['volatility']],
        y=[max_sharpe_point['expected_return']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Max Sharpe',
        text=[f"Sharpe: {max_sharpe_point['sharpe_ratio']:.3f}"]
    ))
    
    # Min Volatility portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol_point['volatility']],
        y=[min_vol_point['expected_return']],
        mode='markers',
        marker=dict(size=15, color='green', symbol='diamond'),
        name='Min Volatility',
        text=[f"Sharpe: {min_vol_point['sharpe_ratio']:.3f}"]
    ))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        hovermode='closest',
        height=500
    )
    
    return fig


def plot_allocation_pie(weights, title):
    """Plot portfolio allocation pie chart"""
    # Filter out negligible weights
    filtered_weights = {k: v for k, v in weights.items() if v > 0.01}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(filtered_weights.keys()),
        values=list(filtered_weights.values()),
        hole=0.3,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig


def plot_returns_comparison(summary_stats):
    """Plot returns comparison bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=summary_stats.index,
            y=summary_stats['Mean'] * 100,
            name='Expected Return (%)',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title='Expected Annual Returns by Asset',
        xaxis_title='Asset',
        yaxis_title='Expected Return (%)',
        height=400
    )
    
    return fig


def plot_risk_return_scatter(summary_stats):
    """Plot risk-return scatter"""
    fig = go.Figure(data=[
        go.Scatter(
            x=summary_stats['Volatility'] * 100,
            y=summary_stats['Mean'] * 100,
            mode='markers+text',
            marker=dict(size=12, color='blue'),
            text=summary_stats.index,
            textposition='top center'
        )
    ])
    
    fig.update_layout(
        title='Risk-Return Profile of Assets',
        xaxis_title='Volatility (Risk) %',
        yaxis_title='Expected Return %',
        height=400
    )
    
    return fig


def main():
    st.title("🚀 AI-Powered Portfolio Optimization System")
    st.markdown("*Production-ready portfolio optimization with machine learning predictions*")
    
    # Sidebar inputs
    st.sidebar.header("Portfolio Configuration")
    
    # Ticker input
    default_tickers = ", ".join(config.DEFAULT_TICKERS)
    tickers_input = st.sidebar.text_input(
        "Enter Stock Tickers (comma-separated)",
        value=default_tickers
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
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
    
    # Investment amount
    investment_amount = st.sidebar.number_input(
        "Investment Amount ($)",
        min_value=1000.0,
        max_value=10000000.0,
        value=10000.0,
        step=1000.0
    )
    
    # Risk profile
    risk_profile = st.sidebar.selectbox(
        "Risk Profile",
        options=["conservative", "moderate", "aggressive"],
        index=1
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        max_weight = st.slider(
            "Max Weight per Asset (%)",
            min_value=10,
            max_value=100,
            value=40,
            step=5
        ) / 100
        
        allow_short = st.checkbox("Allow Short Selling", value=False)
        
        n_portfolios = st.number_input(
            "Monte Carlo Simulations",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000
        )
    
    # Optimize button
    optimize_button = st.sidebar.button("🎯 Optimize Portfolio", type="primary", use_container_width=True)
    
    # Main content
    if optimize_button:
        with st.spinner("🔄 Fetching data and training models..."):
            try:
                # Initialize components
                data_fetcher = DataFetcher()
                feature_engineer = FeatureEngineer()
                ml_predictor = MLPredictor()
                optimizer = PortfolioOptimizer()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Fetch data
                status_text.text("📊 Fetching historical data...")
                progress_bar.progress(10)
                
                prices = data_fetcher.fetch_data(
                    tickers,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                returns = data_fetcher.calculate_returns()
                
                # Step 2: Feature engineering
                status_text.text("🔧 Engineering features...")
                progress_bar.progress(30)
                
                features = feature_engineer.create_features(prices, returns)
                future_returns, future_volatility = feature_engineer.create_target_variables(returns)
                
                # Step 3: Train models
                status_text.text("🤖 Training ML models...")
                progress_bar.progress(50)
                
                training_metrics = ml_predictor.train_models(features, future_returns, future_volatility)
                
                # Step 4: Make predictions
                status_text.text("🔮 Making predictions...")
                progress_bar.progress(60)
                
                expected_returns, expected_volatility = ml_predictor.predict(features, tickers)
                
                # Step 5: Optimize
                status_text.text("⚡ Optimizing portfolios...")
                progress_bar.progress(70)
                
                cov_matrix = returns.cov() * config.TRADING_DAYS_PER_YEAR
                optimizer.set_parameters(expected_returns, cov_matrix)
                
                max_sharpe = optimizer.optimize_max_sharpe(max_weight=max_weight, allow_short=allow_short)
                min_vol = optimizer.optimize_min_volatility(max_weight=max_weight, allow_short=allow_short)
                
                # Step 6: Generate efficient frontier
                status_text.text("📈 Generating efficient frontier...")
                progress_bar.progress(85)
                
                frontier = optimizer.generate_efficient_frontier(n_portfolios=n_portfolios, max_weight=max_weight)
                
                # Step 7: Calculate allocation
                status_text.text("💰 Calculating allocation...")
                progress_bar.progress(95)
                
                latest_prices = data_fetcher.get_latest_prices(tickers)
                
                max_sharpe_allocation = optimizer.calculate_capital_allocation(
                    max_sharpe['weights'],
                    investment_amount,
                    latest_prices
                )
                
                min_vol_allocation = optimizer.calculate_capital_allocation(
                    min_vol['weights'],
                    investment_amount,
                    latest_prices
                )
                
                summary_stats = data_fetcher.get_summary_statistics(returns)
                
                progress_bar.progress(100)
                status_text.text("✅ Optimization complete!")
                
                # Store results
                st.session_state.results = {
                    'max_sharpe': max_sharpe,
                    'min_vol': min_vol,
                    'frontier': frontier,
                    'max_sharpe_allocation': max_sharpe_allocation,
                    'min_vol_allocation': min_vol_allocation,
                    'summary_stats': summary_stats,
                    'expected_returns': expected_returns,
                    'expected_volatility': expected_volatility,
                    'training_metrics': training_metrics
                }
                st.session_state.optimization_complete = True
                
                st.success("✅ Portfolio optimization completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.optimization_complete = False
    
    # Display results
    if st.session_state.optimization_complete and st.session_state.results:
        results = st.session_state.results
        
        st.markdown("---")
        st.header("📊 Optimization Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Max Sharpe Return",
                f"{results['max_sharpe']['expected_return']*100:.2f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "Max Sharpe Volatility",
                f"{results['max_sharpe']['volatility']*100:.2f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "Max Sharpe Ratio",
                f"{results['max_sharpe']['sharpe_ratio']:.3f}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Min Vol Return",
                f"{results['min_vol']['expected_return']*100:.2f}%",
                delta=None
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Efficient Frontier",
            "🥧 Allocations",
            "💰 Capital Breakdown",
            "📊 Asset Analysis",
            "🧪 Stress Testing"
        ])
        
        with tab1:
            st.subheader("Efficient Frontier")
            fig_frontier = plot_efficient_frontier(
                results['frontier'],
                results['max_sharpe'],
                results['min_vol']
            )
            st.plotly_chart(fig_frontier, use_container_width=True)
            
            st.info("""
            **Interpretation:** 
            - Each point represents a different portfolio combination
            - The color indicates the Sharpe ratio (higher is better)
            - 🔴 Red star: Maximum Sharpe Ratio portfolio
            - 🟢 Green diamond: Minimum Volatility portfolio
            """)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Maximum Sharpe Portfolio")
                fig_max_sharpe = plot_allocation_pie(
                    results['max_sharpe']['weights'],
                    "Max Sharpe Allocation"
                )
                st.plotly_chart(fig_max_sharpe, use_container_width=True)
                
                st.dataframe(
                    pd.DataFrame(results['max_sharpe']['weights'].items(), 
                                columns=['Ticker', 'Weight'])
                    .sort_values('Weight', ascending=False)
                    .style.format({'Weight': '{:.2%}'})
                )
            
            with col2:
                st.subheader("Minimum Volatility Portfolio")
                fig_min_vol = plot_allocation_pie(
                    results['min_vol']['weights'],
                    "Min Volatility Allocation"
                )
                st.plotly_chart(fig_min_vol, use_container_width=True)
                
                st.dataframe(
                    pd.DataFrame(results['min_vol']['weights'].items(), 
                                columns=['Ticker', 'Weight'])
                    .sort_values('Weight', ascending=False)
                    .style.format({'Weight': '{:.2%}'})
                )
        
        with tab3:
            st.subheader("Capital Allocation Breakdown")
            
            portfolio_choice = st.radio(
                "Select Portfolio",
                ["Maximum Sharpe", "Minimum Volatility"],
                horizontal=True
            )
            
            if portfolio_choice == "Maximum Sharpe":
                allocation_df = results['max_sharpe_allocation']
            else:
                allocation_df = results['min_vol_allocation']
            
            if not allocation_df.empty:
                st.dataframe(
                    allocation_df.style.format({
                        'Weight': '{:.2%}',
                        'Target_Amount': '${:,.2f}',
                        'Latest_Price': '${:,.2f}',
                        'Actual_Amount': '${:,.2f}',
                        'Actual_Weight': '{:.2%}'
                    }),
                    use_container_width=True
                )
                
                total_invested = allocation_df['Actual_Amount'].sum()
                cash_remainder = investment_amount - total_invested
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Invested", f"${total_invested:,.2f}")
                with col2:
                    st.metric("Cash Remainder", f"${cash_remainder:,.2f}")
                with col3:
                    st.metric("Total Shares", f"{allocation_df['Shares'].sum():,}")
            else:
                st.warning("No allocation data available")
        
        with tab4:
            st.subheader("Asset Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_returns = plot_returns_comparison(results['summary_stats'])
                st.plotly_chart(fig_returns, use_container_width=True)
            
            with col2:
                fig_risk_return = plot_risk_return_scatter(results['summary_stats'])
                st.plotly_chart(fig_risk_return, use_container_width=True)
            
            st.subheader("ML Predictions")
            
            pred_df = pd.DataFrame({
                'Expected Return': results['expected_returns'] * 100,
                'Expected Volatility': results['expected_volatility'] * 100
            })
            
            st.dataframe(
                pred_df.style.format('{:.2f}%'),
                use_container_width=True
            )
            
            st.subheader("Summary Statistics")
            st.dataframe(
                results['summary_stats'].style.format({
                    'Mean': '{:.4f}',
                    'Volatility': '{:.4f}',
                    'Sharpe': '{:.4f}',
                    'Min': '{:.4f}',
                    'Max': '{:.4f}',
                    'Skewness': '{:.4f}',
                    'Kurtosis': '{:.4f}'
                }),
                use_container_width=True
            )
        
        with tab5:
            st.subheader("Stress Testing")
            
            scenario = st.selectbox(
                "Select Stress Test Scenario",
                ["market_crash", "volatility_spike"]
            )
            
            portfolio_type = st.radio(
                "Select Portfolio to Test",
                ["Maximum Sharpe", "Minimum Volatility"],
                horizontal=True
            )
            
            if st.button("Run Stress Test"):
                with st.spinner("Running stress test..."):
                    if portfolio_type == "Maximum Sharpe":
                        weights = results['max_sharpe']['weights']
                    else:
                        weights = results['min_vol']['weights']
                    
                    # Get optimizer (re-initialize with current parameters)
                    optimizer = PortfolioOptimizer()
                    cov_matrix = results['summary_stats'].index.to_series().apply(
                        lambda x: results['expected_volatility'][x]**2
                    )
                    optimizer.set_parameters(results['expected_returns'], 
                                           pd.DataFrame(np.diag(cov_matrix)))
                    
                    stress_results = optimizer.stress_test(weights, scenario)
                    
                    st.success("Stress test completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Return Change",
                            f"{stress_results['return_change']*100:.2f}%",
                            delta=f"{stress_results['return_change']*100:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Volatility Change",
                            f"{stress_results['volatility_change']*100:.2f}%",
                            delta=f"{stress_results['volatility_change']*100:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Sharpe Change",
                            f"{stress_results['sharpe_change']:.3f}",
                            delta=f"{stress_results['sharpe_change']:.3f}"
                        )
                    
                    st.json(stress_results)
    
    else:
        # Welcome screen
        st.info("👈 Configure your portfolio in the sidebar and click **Optimize Portfolio** to begin")
        
        st.markdown("""
        ### Features:
        
        - 🤖 **Machine Learning Predictions**: Uses Random Forest to predict returns and volatility
        - 📊 **Mean-Variance Optimization**: Finds optimal portfolio allocations
        - 📈 **Efficient Frontier**: Visualizes risk-return tradeoffs
        - 💰 **Capital Allocation**: Calculates exact shares to buy
        - 🧪 **Stress Testing**: Simulates market crash and volatility scenarios
        - 🎯 **Risk Profiles**: Conservative, Moderate, or Aggressive
        
        ### How to Use:
        
        1. Enter stock tickers (comma-separated)
        2. Select date range for historical data
        3. Set your investment amount
        4. Choose risk profile and constraints
        5. Click "Optimize Portfolio"
        6. Review results and allocations
        
        ### Example Tickers:
        - Tech: AAPL, MSFT, GOOGL, AMZN, NVDA
        - Finance: JPM, BAC, GS, MS, C
        - Healthcare: JNJ, PFE, UNH, ABBV, MRK
        - Consumer: WMT, PG, KO, PEP, COST
        """)


if __name__ == "__main__":
    main()
