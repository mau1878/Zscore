import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set page configuration
st.set_page_config(
  page_title="Pairs Trading Backtester",
  layout="wide",
)

st.title("📈 Pairs Trading Backtester with Z-Score")
st.markdown("""
A simple backtesting tool for a pairs trading strategy using z-score to decide when to switch between two stocks.
""")

# Sidebar for user inputs
st.sidebar.header("Select Parameters")

# Function to get stock data
@st.cache_data
def get_stock_data(ticker, start, end):
  data = yf.download(ticker, start=start, end=end)
  if data.empty:
      st.error(f"Failed to fetch data for {ticker}. Please check the ticker symbol.")
  return data['Adj Close']

# User inputs
default_ticker1 = 'AAPL'
default_ticker2 = 'MSFT'

ticker1 = st.sidebar.text_input("First Stock Ticker", value=default_ticker1)
ticker2 = st.sidebar.text_input("Second Stock Ticker", value=default_ticker2)

start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

zscore_window = st.sidebar.number_input("Z-Score Window", min_value=5, max_value=100, value=20, step=1)
entry_zscore = st.sidebar.number_input("Entry Z-Score Threshold", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
exit_zscore = st.sidebar.number_input("Exit Z-Score Threshold", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

if start_date >= end_date:
  st.sidebar.error("End date must be after start date.")

# Fetch data
with st.spinner("Fetching data..."):
  stock1 = get_stock_data(ticker1, start_date, end_date)
  stock2 = get_stock_data(ticker2, start_date, end_date)

# Combine into DataFrame
data = pd.DataFrame({ticker1: stock1, ticker2: stock2}).dropna()

if data.empty:
  st.error("No overlapping data between the selected dates. Please adjust the date range or tickers.")
  st.stop()

# Calculate spread and z-score
data['Spread'] = data[ticker1] - data[ticker2]
data['Z-Score'] = (data['Spread'] - data['Spread'].rolling(window=zscore_window).mean()) / data['Spread'].rolling(window=zscore_window).std()

# Backtesting logic
def backtest_pairs(data, entry_threshold, exit_threshold):
  position = 0  # 1: Long Spread, -1: Short Spread, 0: No Position
  returns = []
  daily_returns = data[[ticker1, ticker2]].pct_change().fillna(0)
  
  for i in range(len(data)):
      z = data['Z-Score'].iloc[i]
      if position == 0:
          if z > entry_threshold:
              position = -1  # Short Spread
          elif z < -entry_threshold:
              position = 1   # Long Spread
      elif position == 1:
          if z > -exit_threshold:
              position = 0
      elif position == -1:
          if z < exit_threshold:
              position = 0
      
      # Calculate daily return based on position
      if position == 1:
          # Long spread: Long ticker1, Short ticker2
          daily_ret = daily_returns[ticker1].iloc[i] - daily_returns[ticker2].iloc[i]
      elif position == -1:
          # Short spread: Short ticker1, Long ticker2
          daily_ret = -daily_returns[ticker1].iloc[i] + daily_returns[ticker2].iloc[i]
      else:
          daily_ret = 0
      returns.append(daily_ret)
  
  strategy_returns = pd.Series(returns, index=data.index)
  cumulative_strategy = (1 + strategy_returns).cumprod()
  cumulative_buy_hold = (1 + daily_returns.mean(axis=1)).cumprod()
  
  return strategy_returns, cumulative_strategy, cumulative_buy_hold

strategy_returns, cumulative_strategy, cumulative_buy_hold = backtest_pairs(data, entry_zscore, exit_zscore)

# Display data and plots
st.header("Stock Prices")
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(data.index, data[ticker1], label=ticker1)
ax1.plot(data.index, data[ticker2], label=ticker2)
ax1.set_title("Adjusted Close Prices")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

st.header("Z-Score of Spread")
fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(data.index, data['Z-Score'], label='Z-Score')
ax2.axhline(entry_zscore, color='red', linestyle='--', label='Entry Threshold')
ax2.axhline(-entry_zscore, color='red', linestyle='--')
ax2.axhline(exit_zscore, color='green', linestyle='--', label='Exit Threshold')
ax2.axhline(-exit_zscore, color='green', linestyle='--')
ax2.set_title("Z-Score")
ax2.set_xlabel("Date")
ax2.set_ylabel("Z-Score")
ax2.legend()
st.pyplot(fig2)

st.header("Strategy Performance")
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(cumulative_strategy.index, cumulative_strategy, label='Pairs Trading Strategy')
ax3.plot(cumulative_buy_hold.index, cumulative_buy_hold, label='Buy and Hold (Average)')
ax3.set_title("Cumulative Returns")
ax3.set_xlabel("Date")
ax3.set_ylabel("Cumulative Returns")
ax3.legend()
st.pyplot(fig3)

# Display metrics
st.header("Performance Metrics")

def calculate_metrics(strategy_returns, benchmark_returns):
  total_return = strategy_returns.sum()
  annual_return = strategy_returns.mean() * 252
  annual_vol = strategy_returns.std() * np.sqrt(252)
  sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan
  max_drawdown = (cumulative_strategy / cumulative_strategy.cummax() - 1).min()
  
  benchmark_total = benchmark_returns.sum()
  benchmark_annual = benchmark_returns.mean() * 252
  benchmark_vol = benchmark_returns.std() * np.sqrt(252)
  benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol !=0 else np.nan
  benchmark_max_dd = (cumulative_buy_hold / cumulative_buy_hold.cummax() -1).min()
  
  metrics = {
      "Strategy": {
          "Total Return": f"{total_return:.2%}",
          "Annualized Return": f"{annual_return:.2%}",
          "Annualized Volatility": f"{annual_vol:.2%}",
          "Sharpe Ratio": f"{sharpe_ratio:.2f}",
          "Max Drawdown": f"{max_drawdown:.2%}",
      },
      "Benchmark (Average Buy & Hold)": {
          "Total Return": f"{benchmark_total:.2%}",
          "Annualized Return": f"{benchmark_annual:.2%}",
          "Annualized Volatility": f"{benchmark_vol:.2%}",
          "Sharpe Ratio": f"{benchmark_sharpe:.2f}",
          "Max Drawdown": f"{benchmark_max_dd:.2%}",
      }
  }
  return metrics

metrics = calculate_metrics(strategy_returns, benchmark_returns)

# Create dataframe for metrics
metrics_df = pd.DataFrame(metrics).T

st.table(metrics_df)

# Optional: Show trade signals
st.header("Trade Signals")
def generate_signals(data, entry_threshold, exit_threshold):
  position = 0
  signals = []
  for z in data['Z-Score']:
      if position == 0:
          if z > entry_threshold:
              position = -1
              signals.append('Short Spread')
          elif z < -entry_threshold:
              position = 1
              signals.append('Long Spread')
          else:
              signals.append(None)
      elif position == 1:
          if z > -exit_threshold:
              position = 0
              signals.append('Exit Long Spread')
          else:
              signals.append(None)
      elif position == -1:
          if z < exit_threshold:
              position = 0
              signals.append('Exit Short Spread')
          else:
              signals.append(None)
      else:
          signals.append(None)
  return signals

data['Signal'] = generate_signals(data, entry_zscore, exit_zscore)
signal_df = data[data['Signal'].notnull()][['Signal']]

st.write(signal_df)

# Footer
st.markdown("""
---
**Disclaimer:** This tool is for educational purposes only and should not be considered as financial advice. Always do your own research before making investment decisions.
""")
