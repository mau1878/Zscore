import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
    page_title="📈 Pairs Trading Backtester with Z-Score (Long-Only)",
    layout="wide",
)

# Title and Description
st.title("📈 Pairs Trading Backtester with Z-Score (Long-Only)")
st.markdown("""
A backtesting tool for a modified pairs trading strategy suitable for markets where short selling is not permitted.
""")

# Sidebar for User Inputs
st.sidebar.header("Select Parameters")

# Function to fetch stock data with caching and error handling
@st.cache_data(ttl=60*60)  # Cache data for 1 hour
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f"Failed to fetch data for `{ticker}`. Please check the ticker symbol.")
            return None
        return data['Adj Close']
    except Exception as e:
        st.error(f"Error fetching data for `{ticker}`: {e}")
        return None

# User Inputs
default_ticker1 = 'GGAL.BA'  # Example Argentine stocks
default_ticker2 = 'BMA.BA'

ticker1 = st.sidebar.text_input("First Stock Ticker", value=default_ticker1).upper()
ticker2 = st.sidebar.text_input("Second Stock Ticker", value=default_ticker2).upper()

start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

zscore_window = st.sidebar.number_input("Z-Score Window (Days)", min_value=5, max_value=100, value=20, step=1)
entry_zscore = st.sidebar.number_input("Entry Z-Score Entry Threshold", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
exit_zscore = st.sidebar.number_input("Exit Z-Score Exit Threshold", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

# Validate Date Inputs
if start_date >= end_date:
    st.sidebar.error("⚠️ **Start date must be before end date.**")

# Fetch Stock Data
with st.spinner("🔄 Fetching stock data..."):
    stock1 = get_stock_data(ticker1, start_date, end_date)
    stock2 = get_stock_data(ticker2, start_date, end_date)

# Stop execution if data fetching failed
if stock1 is None or stock2 is None:
    st.stop()

# Combine Data into a DataFrame
data = pd.DataFrame({ticker1: stock1, ticker2: stock2}).dropna()

if data.empty:
    st.error("❌ **No overlapping data between the selected dates.** Please adjust the date range or tickers.")
    st.stop()

# Calculate Spread and Z-Score
data['Spread'] = data[ticker1] - data[ticker2]
data['Spread_Mean'] = data['Spread'].rolling(window=zscore_window).mean()
data['Spread_STD'] = data['Spread'].rolling(window=zscore_window).std()
data['Z-Score'] = (data['Spread'] - data['Spread_Mean']) / data['Spread_STD']

# Backtesting Logic (Long-Only Strategy)
def backtest_pairs_long_only(data, entry_threshold, exit_threshold):
    position = 0  # 1: Long Position, 0: No Position
    returns = []
    daily_returns = data[ticker1].pct_change().fillna(0)
    
    for i in range(len(data)):
        z = data['Z-Score'].iloc[i]
        if position == 0:
            if z < -entry_threshold:
                position = 1  # Enter Long Position on ticker1
        elif position == 1:
            if z > -exit_threshold:
                position = 0  # Exit Long Position
                
        # Calculate daily return based on position
        if position == 1:
            daily_ret = daily_returns.iloc[i]
        else:
            daily_ret = 0
        returns.append(daily_ret)
    
    strategy_returns = pd.Series(returns, index=data.index)
    cumulative_strategy = (1 + strategy_returns).cumprod()
    
    # Benchmark: Holding ticker1
    benchmark_returns = daily_returns
    cumulative_buy_hold = (1 + benchmark_returns).cumprod()
    
    return strategy_returns, cumulative_strategy, cumulative_buy_hold, benchmark_returns

# Execute Backtest
strategy_returns, cumulative_strategy, cumulative_buy_hold, benchmark_returns = backtest_pairs_long_only(
    data, entry_zscore, exit_zscore
)

# Generate Trade Signals
def generate_signals_long_only(data, entry_threshold, exit_threshold):
    position = 0
    signals = []
    for z in data['Z-Score']:
        if position == 0:
            if z < -entry_threshold:
                position = 1
                signals.append('Buy')
            else:
                signals.append(None)
        elif position == 1:
            if z > -exit_threshold:
                position = 0
                signals.append('Sell')
            else:
                signals.append(None)
        else:
            signals.append(None)
    return signals

data['Signal'] = generate_signals_long_only(data, entry_zscore, exit_zscore)
signal_df = data[data['Signal'].notnull()][['Signal']]

# Visualization with Plotly

# 1. Stock Prices Plot
st.header("📊 Stock Prices")
fig_prices = make_subplots(rows=1, cols=1, shared_xaxes=True)

fig_prices.add_trace(go.Scatter(
    x=data.index,
    y=data[ticker1],
    mode='lines',
    name=ticker1
))

fig_prices.update_layout(
    title=f"Adjusted Close Prices - {ticker1}",
    xaxis_title="Date",
    yaxis_title="Price (ARS)",
    hovermode='x unified',
    width=1000,
    height=600
)

st.plotly_chart(fig_prices, use_container_width=True)

# 2. Z-Score with Trade Signals Plot
st.header("📈 Z-Score of Spread with Trade Signals")

fig_zscore = make_subplots(rows=1, cols=1, shared_xaxes=True)

# Plot Z-Score
fig_zscore.add_trace(go.Scatter(
    x=data.index,
    y=data['Z-Score'],
    mode='lines',
    name='Z-Score',
    line=dict(color='blue')
))

# Plot Thresholds
fig_zscore.add_trace(go.Scatter(
    x=data.index,
    y=[-entry_zscore]*len(data),
    mode='lines',
    name='Entry Threshold (-)',
    line=dict(color='red', dash='dash')
))
fig_zscore.add_trace(go.Scatter(
    x=data.index,
    y=[-exit_zscore]*len(data),
    mode='lines',
    name='Exit Threshold (-)',
    line=dict(color='green', dash='dash')
))

# Plot Trade Signals
buy_signals = data[data['Signal'] == 'Buy']
sell_signals = data[data['Signal'] == 'Sell']

fig_zscore.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals['Z-Score'],
    mode='markers',
    name='Buy Signal',
    marker=dict(symbol='triangle-up', color='green', size=12),
    hovertemplate='Date: %{x}<br>Signal: Buy'
))

fig_zscore.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals['Z-Score'],
    mode='markers',
    name='Sell Signal',
    marker=dict(symbol='triangle-down', color='red', size=12),
    hovertemplate='Date: %{x}<br>Signal: Sell'
))

fig_zscore.update_layout(
    title="Z-Score of Spread with Trade Signals",
    xaxis_title="Date",
    yaxis_title="Z-Score",
    hovermode='x unified',
    width=1000,
    height=600
)

st.plotly_chart(fig_zscore, use_container_width=True)

# 3. Cumulative Returns Plot
st.header("📈 Strategy vs. Benchmark Cumulative Returns")

fig_cum_returns = make_subplots(rows=1, cols=1, shared_xaxes=True)

fig_cum_returns.add_trace(go.Scatter(
    x=cumulative_strategy.index,
    y=cumulative_strategy,
    mode='lines',
    name='Strategy',
    line=dict(color='purple')
))

fig_cum_returns.add_trace(go.Scatter(
    x=cumulative_buy_hold.index,
    y=cumulative_buy_hold,
    mode='lines',
    name=f'Buy and Hold {ticker1}',
    line=dict(color='grey')
))

fig_cum_returns.update_layout(
    title="Cumulative Returns",
    xaxis_title="Date",
    yaxis_title="Cumulative Returns",
    hovermode='x unified',
    width=1000,
    height=600
)

st.plotly_chart(fig_cum_returns, use_container_width=True)

# Performance Metrics
st.header("📊 Performance Metrics")

def calculate_metrics(strategy_returns, benchmark_returns, cumulative_strategy, cumulative_buy_hold):
    # Strategy Metrics
    total_return = cumulative_strategy.iloc[-1] - 1
    annual_return = strategy_returns.mean() * 252
    annual_vol = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan
    max_drawdown = (cumulative_strategy / cumulative_strategy.cummax() - 1).min()
    
    # Benchmark Metrics
    benchmark_total = cumulative_buy_hold.iloc[-1] - 1
    benchmark_annual = benchmark_returns.mean() * 252
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)
    benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol != 0 else np.nan
    benchmark_max_dd = (cumulative_buy_hold / cumulative_buy_hold.cummax() -1).min()
    
    metrics = {
        "Strategy": {
            "Total Return": f"{total_return:.2%}",
            "Annualized Return": f"{annual_return:.2%}",
            "Annualized Volatility": f"{annual_vol:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
        },
        f"Buy and Hold {ticker1}": {
            "Total Return": f"{benchmark_total:.2%}",
            "Annualized Return": f"{benchmark_annual:.2%}",
            "Annualized Volatility": f"{benchmark_vol:.2%}",
            "Sharpe Ratio": f"{benchmark_sharpe:.2f}",
            "Max Drawdown": f"{benchmark_max_dd:.2%}",
        }
    }
    return metrics

# Calculate Metrics
metrics = calculate_metrics(strategy_returns, benchmark_returns, cumulative_strategy, cumulative_buy_hold)

# Display Metrics in a Table
metrics_df = pd.DataFrame(metrics).T
st.table(metrics_df)

# Trade Signals Table
st.header("📋 Trade Signals")
st.write(signal_df)

# Footer Disclaimer
st.markdown("""
---
**Disclaimer:** This tool is for educational purposes only and should not be considered as financial advice. Always do your own research before making investment decisions.
""")
