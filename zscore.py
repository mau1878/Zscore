import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
  page_title="📈 Adaptive Pairs Trading Backtester",
  layout="wide",
)

# Title and Description
st.title("📈 Adaptive Pairs Trading Backtester")
st.markdown("""
Welcome to the **Adaptive Pairs Trading Backtester**!

This tool allows you to backtest a pairs trading strategy between two stocks, incorporating advanced allocation strategies and cash positions. It's designed to be useful even in markets where short selling isn't possible.

---

**How to Use This Tool:**

1. **Select Two Stocks:** Choose two stocks that you believe are correlated or have historically moved together.
2. **Set Parameters:**
 - **Z-Score Window:** The number of days to calculate the rolling mean and standard deviation of the price spread.
 - **Entry/Exit Thresholds:** The z-score levels at which the strategy will adjust allocations.
 - **Maximum Allocation (%):** The maximum percentage of the portfolio to allocate to any single stock.
3. **Run the Backtest:** The tool will calculate the strategy performance and display interactive charts and metrics.

**Understanding the Strategy:**

- **Objective:** To profit from the relative movements of two stocks by adjusting portfolio allocations based on their price spread's z-score.
- **Cash Position:** When the spread doesn't indicate a strong signal, the strategy can move a portion of the portfolio into cash to reduce market exposure.
- **Advanced Allocation:** Allocations to each stock (and cash) are proportional to the magnitude of the z-score.

---

""")

# Sidebar for User Inputs
st.sidebar.header("Select Parameters")

# Function to fetch stock data with caching and error handling
@st.cache_data(ttl=60*60)  # Cache data for 1 hour
def get_stock_data(ticker, start, end):
  try:
      data = yf.download(ticker, start=start, end=end, progress=False)
      if data.empty:
          st.error(f"Failed to fetch data for `{ticker}`. Please check the ticker symbol.")
          return None
      return data['Adj Close']
  except Exception as e:
      st.error(f"Error fetching data for `{ticker}`: {e}")
      return None

# User Inputs
default_ticker1 = 'AAPL'  # Default stocks
default_ticker2 = 'MSFT'

ticker1 = st.sidebar.text_input("First Stock Ticker", value=default_ticker1).upper()
ticker2 = st.sidebar.text_input("Second Stock Ticker", value=default_ticker2).upper()

start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

zscore_window = st.sidebar.number_input("Z-Score Window (Days)", min_value=5, max_value=252, value=30, step=1)
entry_zscore = st.sidebar.number_input("Entry Threshold (Z-Score)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
exit_zscore = st.sidebar.number_input("Exit Threshold (Z-Score)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

max_allocation = st.sidebar.number_input("Maximum Allocation to a Single Stock (%)", min_value=10, max_value=100, value=50, step=5)
max_allocation /= 100  # Convert to decimal

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

# Backtesting Logic with Cash Position and Advanced Allocation
def backtest_pairs_adaptive(data, entry_threshold, exit_threshold, max_alloc):
  # Initialize positions and returns
  positions = []
  weights_t1 = []
  weights_t2 = []
  weights_cash = []
  returns = []
  daily_returns = data[[ticker1, ticker2]].pct_change().fillna(0)

  for i in range(len(data)):
      z = data['Z-Score'].iloc[i]

      # Calculate allocation weights based on z-score
      z_abs = abs(z)
      if z_abs >= entry_threshold:
          # Proportional allocation based on z-score magnitude
          weight = min((z_abs - exit_threshold) / (entry_threshold - exit_threshold), 1) * max_alloc
          weight = min(weight, max_alloc)  # Cap at maximum allocation
          if z > 0:
              # Overweight ticker2
              w_t1 = (1 - weight) * (1 - max_alloc)  # Remaining allocation
              w_t2 = weight
          else:
              # Overweight ticker1
              w_t1 = weight
              w_t2 = (1 - weight) * (1 - max_alloc)
          w_cash = 1 - (w_t1 + w_t2)
      else:
          # No strong signal, move to cash
          w_t1 = 0
          w_t2 = 0
          w_cash = 1

      # Record weights
      weights_t1.append(w_t1)
      weights_t2.append(w_t2)
      weights_cash.append(w_cash)

      # Calculate daily portfolio return
      daily_ret = (w_t1 * daily_returns[ticker1].iloc[i] +
                   w_t2 * daily_returns[ticker2].iloc[i] +
                   w_cash * 0)  # Cash earns zero return
      returns.append(daily_ret)

  strategy_returns = pd.Series(returns, index=data.index)
  cumulative_strategy = (1 + strategy_returns).cumprod()

  # Benchmark: Equal-weighted portfolio held throughout
  benchmark_weights = np.array([0.5, 0.5])
  benchmark_returns = daily_returns.dot(benchmark_weights)
  cumulative_benchmark = (1 + benchmark_returns).cumprod()

  # Store weights in DataFrame
  weights_df = pd.DataFrame({
      'Weight_' + ticker1: weights_t1,
      'Weight_' + ticker2: weights_t2,
      'Weight_Cash': weights_cash
  }, index=data.index)

  return strategy_returns, cumulative_strategy, cumulative_benchmark, benchmark_returns, weights_df

# Execute Backtest
results = backtest_pairs_adaptive(
  data, entry_zscore, exit_zscore, max_allocation
)
strategy_returns, cumulative_strategy, cumulative_benchmark, benchmark_returns, weights_df = results

# Combine weights with data
data = pd.concat([data, weights_df], axis=1)

# Generate Trade Signals for Display
def generate_signals_adaptive(weights_df):
  signals = []
  prev_w_t1 = 0
  prev_w_t2 = 0
  prev_w_cash = 1

  for idx, row in weights_df.iterrows():
      w_t1 = row['Weight_' + ticker1]
      w_t2 = row['Weight_' + ticker2]
      w_cash = row['Weight_Cash']

      signal = None
      if w_t1 > prev_w_t1:
          signal = 'Increase ' + ticker1
      elif w_t2 > prev_w_t2:
          signal = 'Increase ' + ticker2
      elif w_cash > prev_w_cash:
          signal = 'Increase Cash Position'
      elif w_t1 < prev_w_t1:
          signal = 'Decrease ' + ticker1
      elif w_t2 < prev_w_t2:
          signal = 'Decrease ' + ticker2

      signals.append(signal)
      prev_w_t1, prev_w_t2, prev_w_cash = w_t1, w_t2, w_cash

  return signals

data['Signal'] = generate_signals_adaptive(weights_df)
signal_df = data[['Signal']].dropna()

# Visualization with Plotly

# 1. Stock Prices Plot
st.header("📊 Stock Prices")
st.markdown("""
This chart displays the adjusted closing prices of the two selected stocks over the chosen time period.
""")
fig_prices = make_subplots(rows=1, cols=1, shared_xaxes=True)

fig_prices.add_trace(go.Scatter(
  x=data.index,
  y=data[ticker1],
  mode='lines',
  name=ticker1
))

fig_prices.add_trace(go.Scatter(
  x=data.index,
  y=data[ticker2],
  mode='lines',
  name=ticker2
))

fig_prices.update_layout(
  title=f"Adjusted Close Prices - {ticker1} & {ticker2}",
  xaxis_title="Date",
  yaxis_title="Price",
  hovermode='x unified',
  width=1000,
  height=500
)

st.plotly_chart(fig_prices, use_container_width=True)

# 2. Z-Score with Trade Signals Plot
st.header("📈 Z-Score of Spread with Trade Signals")
st.markdown("""
This chart shows the z-score of the price spread between the two stocks, along with entry and exit thresholds. It illustrates how the strategy adjusts allocations based on the z-score.
""")

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
  y=[entry_zscore]*len(data),
  mode='lines',
  name='Entry Threshold',
  line=dict(color='red', dash='dash')
))
fig_zscore.add_trace(go.Scatter(
  x=data.index,
  y=[-entry_zscore]*len(data),
  mode='lines',
  name='-Entry Threshold',
  line=dict(color='red', dash='dash')
))
fig_zscore.add_trace(go.Scatter(
  x=data.index,
  y=[exit_zscore]*len(data),
  mode='lines',
  name='Exit Threshold',
  line=dict(color='green', dash='dash')
))
fig_zscore.add_trace(go.Scatter(
  x=data.index,
  y=[-exit_zscore]*len(data),
  mode='lines',
  name='-Exit Threshold',
  line=dict(color='green', dash='dash')
))

# Plot Allocation Changes
allocation_changes = data[data['Signal'].notnull()]
fig_zscore.add_trace(go.Scatter(
  x=allocation_changes.index,
  y=allocation_changes['Z-Score'],
  mode='markers',
  name='Allocation Change',
  marker=dict(symbol='circle', color='purple', size=10),
  hovertemplate='Date: %{x}<br>Signal: %{text}',
  text=allocation_changes['Signal']
))

fig_zscore.update_layout(
  title="Z-Score with Allocation Signals",
  xaxis_title="Date",
  yaxis_title="Z-Score",
  hovermode='x unified',
  width=1000,
  height=500
)

st.plotly_chart(fig_zscore, use_container_width=True)

# 3. Portfolio Allocation Over Time
st.header("📈 Portfolio Allocation Over Time")
st.markdown("""
This chart displays how the portfolio allocations to each stock and cash change over time based on the z-score signals.
""")

fig_alloc = make_subplots(rows=1, cols=1, shared_xaxes=True)

fig_alloc.add_trace(go.Scatter(
  x=data.index,
  y=data['Weight_' + ticker1],
  mode='lines',
  name='Allocation to ' + ticker1,
  stackgroup='one'
))

fig_alloc.add_trace(go.Scatter(
  x=data.index,
  y=data['Weight_' + ticker2],
  mode='lines',
  name='Allocation to ' + ticker2,
  stackgroup='one'
))

fig_alloc.add_trace(go.Scatter(
  x=data.index,
  y=data['Weight_Cash'],
  mode='lines',
  name='Allocation to Cash',
  stackgroup='one'
))

fig_alloc.update_layout(
  title="Portfolio Allocation Over Time",
  xaxis_title="Date",
  yaxis_title="Allocation Percentage",
  yaxis=dict(range=[0, 1]),
  hovermode='x unified',
  width=1000,
  height=500
)

st.plotly_chart(fig_alloc, use_container_width=True)

# 4. Cumulative Returns Plot
st.header("📈 Strategy vs. Benchmark Cumulative Returns")
st.markdown("""
This chart compares the cumulative returns of the adaptive strategy against a benchmark of holding an equal-weighted portfolio of the two stocks continuously.
""")

fig_cum_returns = make_subplots(rows=1, cols=1, shared_xaxes=True)

fig_cum_returns.add_trace(go.Scatter(
  x=cumulative_strategy.index,
  y=cumulative_strategy,
  mode='lines',
  name='Adaptive Strategy',
  line=dict(color='purple')
))

fig_cum_returns.add_trace(go.Scatter(
  x=cumulative_benchmark.index,
  y=cumulative_benchmark,
  mode='lines',
  name='Benchmark (Equal-weighted)',
  line=dict(color='grey')
))

fig_cum_returns.update_layout(
  title="Cumulative Returns",
  xaxis_title="Date",
  yaxis_title="Cumulative Returns",
  hovermode='x unified',
  width=1000,
  height=500
)

st.plotly_chart(fig_cum_returns, use_container_width=True)

# Performance Metrics
st.header("📊 Performance Metrics")
st.markdown("""
The table below summarizes key performance metrics for the adaptive strategy and the benchmark.
""")

def calculate_metrics(strategy_returns, benchmark_returns, cumulative_strategy, cumulative_benchmark):
  # Strategy Metrics
  total_return = cumulative_strategy.iloc[-1] - 1
  annual_return = strategy_returns.mean() * 252
  annual_vol = strategy_returns.std() * np.sqrt(252)
  sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan
  max_drawdown = (cumulative_strategy / cumulative_strategy.cummax() - 1).min()

  # Benchmark Metrics
  benchmark_total = cumulative_benchmark.iloc[-1] - 1
  benchmark_annual = benchmark_returns.mean() * 252
  benchmark_vol = benchmark_returns.std() * np.sqrt(252)
  benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol != 0 else np.nan
  benchmark_max_dd = (cumulative_benchmark / cumulative_benchmark.cummax() -1).min()

  metrics = {
      "Adaptive Strategy": {
          "Total Return": f"{total_return:.2%}",
          "Annualized Return": f"{annual_return:.2%}",
          "Annualized Volatility": f"{annual_vol:.2%}",
          "Sharpe Ratio": f"{sharpe_ratio:.2f}",
          "Max Drawdown": f"{max_drawdown:.2%}",
      },
      "Benchmark (Equal-weighted)": {
          "Total Return": f"{benchmark_total:.2%}",
          "Annualized Return": f"{benchmark_annual:.2%}",
          "Annualized Volatility": f"{benchmark_vol:.2%}",
          "Sharpe Ratio": f"{benchmark_sharpe:.2f}",
          "Max Drawdown": f"{benchmark_max_dd:.2%}",
      }
  }
  return metrics

# Calculate Metrics
metrics = calculate_metrics(strategy_returns, benchmark_returns, cumulative_strategy, cumulative_benchmark)

# Display Metrics in a Table
metrics_df = pd.DataFrame(metrics).T
st.table(metrics_df)

# Trade Signals Table
st.header("📋 Allocation Signals")
st.markdown("""
The table below details the points in time where the portfolio allocations changed based on the z-score signals.
""")
st.write(signal_df.dropna())

# Footer Disclaimer
st.markdown("""
---
**Disclaimer:** This tool is for educational purposes only and should not be considered financial advice. Always do your own research before making investment decisions.
""")
