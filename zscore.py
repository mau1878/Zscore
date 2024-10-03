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
Bienvenido al **Adaptive Pairs Trading Backtester**!

Esta herramienta te permite realizar un back-testing de una estrategia de trading de pares entre dos acciones, incorporando estrategias de asignación avanzadas y posiciones en efectivo. Está diseñada para ser útil incluso en mercados donde no se permite la venta en corto.

---

**Cómo usar esta herramienta:**

1. **Selecciona dos acciones:** Elige dos acciones que creas que están correlacionadas o que históricamente se han movido juntas.
2. **Configura los parámetros:**
- **Ventana de Z-Score:** El número de días para calcular la media móvil y la desviación estándar del spread de precios.
- **Umbrales de Entrada/Salida:** Los niveles de z-score en los que la estrategia ajustará las asignaciones.
- **Asignación Máxima (%):** El porcentaje máximo del portafolio que se puede asignar a cualquier acción individual.
3. **Ejecuta el Back-Test:** La herramienta calculará el rendimiento de la estrategia y mostrará gráficos interactivos y métricas.

**Entendiendo la estrategia:**

- **Objetivo:** Obtener ganancias de los movimientos relativos de dos acciones ajustando las asignaciones del portafolio en función del z-score de su spread de precios.
- **Posición en Efectivo:** Cuando el spread no indica una señal fuerte, la estrategia puede mover una porción del portafolio a efectivo para reducir la exposición al mercado.
- **Asignación Avanzada:** Las asignaciones a cada acción (y efectivo) son proporcionales a la magnitud del z-score.

---
""")

# Sidebar for User Inputs
st.sidebar.header("Selecciona Parámetros")

# Function to fetch stock data with caching and error handling
@st.cache_data(ttl=60*60)  # Cache data for 1 hour
def get_stock_data(ticker, start, end):
  try:
      data = yf.download(ticker, start=start, end=end, progress=False)
      if data.empty:
          st.error(f"No se pudo obtener datos para `{ticker}`. Por favor, verifica el símbolo de la acción.")
          return None
      return data['Adj Close']
  except Exception as e:
      st.error(f"Error al obtener datos para `{ticker}`: {e}")
      return None

# User Inputs
default_ticker1 = 'AAPL'  # Default stocks
default_ticker2 = 'MSFT'

# Add a selection for strategy type
strategy_type = st.sidebar.selectbox("Selecciona el Tipo de Estrategia", ["Trading de Pares", "Estrategia de Acción Única"])

if strategy_type == "Trading de Pares":
  ticker1 = st.sidebar.text_input("Símbolo de la Primera Acción", value=default_ticker1).upper()
  ticker2 = st.sidebar.text_input("Símbolo de la Segunda Acción", value=default_ticker2).upper()

  start_date = st.sidebar.date_input("Fecha de Inicio", value=datetime(2020, 1, 1))
  end_date = st.sidebar.date_input("Fecha de Fin", value=datetime.today())

  zscore_window = st.sidebar.number_input("Ventana de Z-Score (Días)", min_value=5, max_value=252, value=30, step=1)
  entry_zscore = st.sidebar.number_input("Umbral de Entrada (Z-Score)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
  exit_zscore = st.sidebar.number_input("Umbral de Salida (Z-Score)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

  max_allocation = st.sidebar.number_input("Asignación Máxima a una Acción Individual (%)", min_value=10, max_value=100, value=50, step=5)
  max_allocation /= 100  # Convert to decimal

  # Validate Date Inputs
  if start_date >= end_date:
      st.sidebar.error("⚠️ **La fecha de inicio debe ser anterior a la fecha de fin.**")

  # Fetch Stock Data
  with st.spinner("🔄 Obteniendo datos de acciones..."):
      stock1 = get_stock_data(ticker1, start_date, end_date)
      stock2 = get_stock_data(ticker2, start_date, end_date)

  # Stop execution if data fetching failed
  if stock1 is None or stock2 is None:
      st.stop()

  # Combine Data into a DataFrame
  data = pd.DataFrame({ticker1: stock1, ticker2: stock2}).dropna()

  if data.empty:
      st.error("❌ **No hay datos superpuestos entre las fechas seleccionadas.** Por favor, ajusta el rango de fechas o los símbolos de las acciones.")
      st.stop()

  # Calculate Spread and Z-Score
  data['Spread'] = data[ticker1] - data[ticker2]
  data['Spread_Mean'] = data['Spread'].rolling(window=zscore_window).mean()
  data['Spread_STD'] = data['Spread'].rolling(window=zscore_window).std()
  data['Z-Score'] = (data['Spread'] - data['Spread_Mean']) / data['Spread_STD']

  # Backtesting Logic with Cash Position and Advanced Allocation
  def backtest_pairs_adaptive(data, entry_threshold, exit_threshold, max_alloc):
      # Initialize positions and returns
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
      prev_signal = None  # Track the last signal

      for idx, row in weights_df.iterrows():
          w_t1 = row['Weight_' + ticker1]
          w_t2 = row['Weight_' + ticker2]

          # Determine the current signal based on weights
          if w_t1 > 0 and (prev_signal != 'Comprar'):
              signals.append('Comprar')
              prev_signal = 'Comprar'
          elif w_t2 > 0 and (prev_signal != 'Vender'):
              signals.append('Vender')
              prev_signal = 'Vender'
          else:
              signals.append('Mantener')  # No change

      return signals

  data['Signal'] = generate_signals_adaptive(weights_df)
  signal_df = data[['Signal']].dropna()

  # Visualization with Plotly

  # 1. Stock Prices Plot
  st.header("📊 Precios de Acciones")
  st.markdown("""
  Este gráfico muestra los precios de cierre ajustados de las dos acciones seleccionadas a lo largo del período elegido.
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
      title=f"Precios de Cierre Ajustados - {ticker1} & {ticker2}",
      xaxis_title="Fecha",
      yaxis_title="Precio",
      hovermode='x unified',
      width=1000,
      height=500
  )

  st.plotly_chart(fig_prices, use_container_width=True)

  # 2. Z-Score with Trade Signals Plot
  st.header("📈 Z-Score del Spread con Señales de Asignación")
  st.markdown("""
  Este gráfico muestra el z-score del spread entre las dos acciones, junto con los umbrales de entrada y salida. Ilustra cómo la estrategia ajusta las asignaciones en función del z-score.
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
      name='Umbral de Entrada',
      line=dict(color='red', dash='dash')
  ))
  fig_zscore.add_trace(go.Scatter(
      x=data.index,
      y=[-entry_zscore]*len(data),
      mode='lines',
      name='-Umbral de Entrada',
      line=dict(color='red', dash='dash')
  ))
  fig_zscore.add_trace(go.Scatter(
      x=data.index,
      y=[exit_zscore]*len(data),
      mode='lines',
      name='Umbral de Salida',
      line=dict(color='green', dash='dash')
  ))
  fig_zscore.add_trace(go.Scatter(
      x=data.index,
      y=[-exit_zscore]*len(data),
      mode='lines',
      name='-Umbral de Salida',
      line=dict(color='green', dash='dash')
  ))

  # Plot Allocation Changes
  allocation_changes = data[data['Signal'].notnull()]
  fig_zscore.add_trace(go.Scatter(
      x=allocation_changes.index,
      y=allocation_changes['Z-Score'],
      mode='markers',
      name='Cambio de Asignación',
      marker=dict(symbol='circle', color='purple', size=10),
      hovertemplate='Fecha: %{x}<br>Señal: %{text}',
      text=allocation_changes['Signal']
  ))

  fig_zscore.update_layout(
      title="Z-Score con Señales de Asignación",
      xaxis_title="Fecha",
      yaxis_title="Z-Score",
      hovermode='x unified',
      width=1000,
      height=500
  )

  st.plotly_chart(fig_zscore, use_container_width=True)

  # 3. Asignación de Portafolio a lo Largo del Tiempo
  st.header("📈 Asignación de Portafolio a lo Largo del Tiempo")
  st.markdown("""
  Este gráfico muestra cómo cambian las asignaciones del portafolio a cada acción y a efectivo a lo largo del tiempo, basándose en las señales del z-score.
  """)

  fig_alloc = make_subplots(rows=1, cols=1, shared_xaxes=True)

  fig_alloc.add_trace(go.Scatter(
      x=data.index,
      y=data['Weight_' + ticker1],
      mode='lines',
      name='Asignación a ' + ticker1,
      stackgroup='one'
  ))

  fig_alloc.add_trace(go.Scatter(
      x=data.index,
      y=data['Weight_' + ticker2],
      mode='lines',
      name='Asignación a ' + ticker2,
      stackgroup='one'
  ))

  fig_alloc.add_trace(go.Scatter(
      x=data.index,
      y=data['Weight_Cash'],
      mode='lines',
      name='Asignación a Efectivo',
      stackgroup='one'
  ))

  fig_alloc.update_layout(
      title="Asignación de Portafolio a lo Largo del Tiempo",
      xaxis_title="Fecha",
      yaxis_title="Porcentaje de Asignación",
      yaxis=dict(range=[0, 1]),
      hovermode='x unified',
      width=1000,
      height=500
  )

  st.plotly_chart(fig_alloc, use_container_width=True)

  # 4. Cumulative Returns Plot
  st.header("📈 Rendimiento Acumulado de la Estrategia vs. Benchmark")
  st.markdown("""
  Este gráfico compara los rendimientos acumulados de la estrategia adaptativa contra un benchmark de mantener un portafolio de igual ponderación de las dos acciones de forma continua.
  """)

  fig_cum_returns = make_subplots(rows=1, cols=1, shared_xaxes=True)

  fig_cum_returns.add_trace(go.Scatter(
      x=cumulative_strategy.index,
      y=cumulative_strategy,
      mode='lines',
      name='Estrategia Adaptativa',
      line=dict(color='purple')
  ))

  fig_cum_returns.add_trace(go.Scatter(
      x=cumulative_benchmark.index,
      y=cumulative_benchmark,
      mode='lines',
      name='Benchmark (Igual Ponderación)',
      line=dict(color='grey')
  ))

  fig_cum_returns.update_layout(
      title="Rendimiento Acumulado",
      xaxis_title="Fecha",
      yaxis_title="Rendimiento Acumulado",
      hovermode='x unified',
      width=1000,
      height=500
  )

  st.plotly_chart(fig_cum_returns, use_container_width=True)

  # Performance Metrics
  st.header("📊 Métricas de Rendimiento")
  st.markdown("""
  La tabla a continuación resume las métricas clave de rendimiento para la estrategia adaptativa y el benchmark.
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
      benchmark_max_dd = (cumulative_benchmark / cumulative_benchmark.cummax() - 1).min()

      metrics = {
          "Estrategia Adaptativa": {
              "Rendimiento Total": f"{total_return:.2%}",
              "Rendimiento Anualizado": f"{annual_return:.2%}",
              "Volatilidad Anualizada": f"{annual_vol:.2%}",
              "Ratio de Sharpe": f"{sharpe_ratio:.2f}",
              "Máxima Caída": f"{max_drawdown:.2%}",
          },
          "Benchmark (Igual Ponderación)": {
              "Rendimiento Total": f"{benchmark_total:.2%}",
              "Rendimiento Anualizado": f"{benchmark_annual:.2%}",
              "Volatilidad Anualizada": f"{benchmark_vol:.2%}",
              "Ratio de Sharpe": f"{benchmark_sharpe:.2f}",
              "Máxima Caída": f"{benchmark_max_dd:.2%}",
          }
      }
      return metrics

  # Calculate Metrics
  metrics = calculate_metrics(strategy_returns, benchmark_returns, cumulative_strategy, cumulative_benchmark)

  # Display Metrics in a Table
  metrics_df = pd.DataFrame(metrics).T
  st.table(metrics_df)

  # Trade Signals Table
  st.header("📋 Señales de Asignación")
  st.markdown("""
  La tabla a continuación detalla los momentos en que las asignaciones del portafolio cambiaron en función de las señales del z-score.
  """)
  st.write(signal_df.dropna())

elif strategy_type == "Estrategia de Acción Única":
  # User Inputs for single stock strategy
  single_ticker = st.sidebar.text_input("Símbolo de la Acción", value=default_ticker1).upper()

  start_date = st.sidebar.date_input("Fecha de Inicio", value=datetime(2020, 1, 1))
  end_date = st.sidebar.date_input("Fecha de Fin", value=datetime.today())

  zscore_window = st.sidebar.number_input("Ventana de Z-Score (Días)", min_value=5, max_value=252, value=30, step=1)
  entry_zscore = st.sidebar.number_input("Umbral de Entrada (Z-Score)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
  exit_zscore = st.sidebar.number_input("Umbral de Salida (Z-Score)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

  # Validate Date Inputs
  if start_date >= end_date:
      st.sidebar.error("⚠️ **La fecha de inicio debe ser anterior a la fecha de fin.**")

  # Fetch Stock Data for Single Stock
  with st.spinner("🔄 Obteniendo datos de la acción..."):
      single_stock_data = get_stock_data(single_ticker, start_date, end_date)

  # Stop execution if data fetching failed
  if single_stock_data is None:
      st.stop()

  # Prepare DataFrame for single stock
  single_stock_df = pd.DataFrame(single_stock_data).dropna()
  single_stock_df = single_stock_df.reset_index()  # Reset index to have a date column

  # Generate Signals for Single Stock
  def generate_signals_single_stock(stock_data, zscore_window, entry_threshold, exit_threshold):
      # Calculate Z-Score
      stock_data['Mean'] = stock_data['Adj Close'].rolling(window=zscore_window).mean()
      stock_data['STD'] = stock_data['Adj Close'].rolling(window=zscore_window).std()
      stock_data['Z-Score'] = (stock_data['Adj Close'] - stock_data['Mean']) / stock_data['STD']

      signals = []
      prev_signal = None  # Track the last signal

      for z in stock_data['Z-Score']:
          if z >= entry_threshold and prev_signal != 'Comprar':
              signals.append('Comprar')
              prev_signal = 'Comprar'
          elif z <= -entry_threshold and prev_signal != 'Vender':
              signals.append('Vender')
              prev_signal = 'Vender'
          else:
              signals.append('Mantener')  # No change

      stock_data['Signal'] = signals
      return stock_data[['Adj Close', 'Z-Score', 'Signal']]

  # Generate signals for the single stock
  single_stock_signals = generate_signals_single_stock(single_stock_df, zscore_window, entry_zscore, exit_zscore)

  # Display the signals
  st.header(f"📈 Señales para {single_ticker}")
  st.write(single_stock_signals)

  # Visualization for Single Stock
  st.header("📊 Gráficos para la Acción Única")
  
  # 1. Adjusted Close Price Plot
  st.subheader("📈 Precio Ajustado de la Acción")
  fig_price = go.Figure()
  fig_price.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Adj Close'],
      mode='lines',
      name='Precio Ajustado'
  ))
  fig_price.update_layout(
      title=f"Precio Ajustado de {single_ticker}",
      xaxis_title="Fecha",
      yaxis_title="Precio Ajustado",
      hovermode='x unified'
  )
  st.plotly_chart(fig_price, use_container_width=True)

  # 2. Z-Score Plot with Buy/Sell Signals
  st.subheader("📈 Z-Score de la Acción con Señales de Compra/Venta")
  fig_zscore_single = go.Figure()
  fig_zscore_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Z-Score'],
      mode='lines',
      name='Z-Score',
      line=dict(color='blue')
  ))
  fig_zscore_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=[entry_zscore]*len(single_stock_df),
      mode='lines',
      name='Umbral de Entrada',
      line=dict(color='red', dash='dash')
  ))
  fig_zscore_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=[-entry_zscore]*len(single_stock_df),
      mode='lines',
      name='-Umbral de Entrada',
      line=dict(color='red', dash='dash')
  ))
  fig_zscore_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=[exit_zscore]*len(single_stock_df),
      mode='lines',
      name='Umbral de Salida',
      line=dict(color='green', dash='dash')
  ))
  fig_zscore_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=[-exit_zscore]*len(single_stock_df),
      mode='lines',
      name='-Umbral de Salida',
      line=dict(color='green', dash='dash')
  ))

  # Add Buy and Sell Signals
  buy_signals = single_stock_df[single_stock_df['Signal'] == 'Comprar']
  sell_signals = single_stock_df[single_stock_df['Signal'] == 'Vender']
  
  fig_zscore_single.add_trace(go.Scatter(
      x=buy_signals['Date'],
      y=buy_signals['Z-Score'],
      mode='markers',
      name='Señales de Compra',
      marker=dict(symbol='triangle-up', color='green', size=10),
      hovertemplate='Fecha: %{x}<br>Señal: Comprar'
  ))
  
  fig_zscore_single.add_trace(go.Scatter(
      x=sell_signals['Date'],
      y=sell_signals['Z-Score'],
      mode='markers',
      name='Señales de Venta',
      marker=dict(symbol='triangle-down', color='red', size=10),
      hovertemplate='Fecha: %{x}<br>Señal: Vender'
  ))

  fig_zscore_single.update_layout(
      title=f"Z-Score de {single_ticker} con Señales de Compra/Venta",
      xaxis_title="Fecha",
      yaxis_title="Z-Score",
      hovermode='x unified'
  )
  st.plotly_chart(fig_zscore_single, use_container_width=True)

  # 3. Asignación de Portafolio a lo Largo del Tiempo
  st.header("📈 Asignación de Portafolio a lo Largo del Tiempo")
  st.markdown("""
  Este gráfico muestra cómo cambian las asignaciones del portafolio a la acción y a efectivo a lo largo del tiempo, basándose en las señales del z-score.
  """)

  # Calculate Portfolio Allocation
  single_stock_df['Weight'] = np.where(single_stock_df['Signal'] == 'Comprar', 1, 0)  # 100% in stock when buying
  single_stock_df['Weight'] = np.where(single_stock_df['Signal'] == 'Vender', 0, single_stock_df['Weight'])  # 0% when selling
  single_stock_df['Weight'] = single_stock_df['Weight'].ffill().fillna(0)  # Forward fill to maintain position

  fig_alloc = go.Figure()
  fig_alloc.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Weight'],
      mode='lines',
      name='Asignación a ' + single_ticker,
      line=dict(color='blue')
  ))

  fig_alloc.update_layout(
      title="Asignación de Portafolio a lo Largo del Tiempo",
      xaxis_title="Fecha",
      yaxis_title="Porcentaje de Asignación",
      yaxis=dict(range=[0, 1]),
      hovermode='x unified'
  )

  st.plotly_chart(fig_alloc, use_container_width=True)

  # 4. Cumulative Returns Plot
  st.header("📈 Rendimiento Acumulado de la Estrategia")
  st.markdown("""
  Este gráfico compara los rendimientos acumulados de la estrategia de trading contra un benchmark de compra y mantenimiento.
  """)

  # Calculate Buy and Hold Returns
  single_stock_df['Buy_Hold_Returns'] = single_stock_df['Adj Close'] / single_stock_df['Adj Close'].iloc[0] - 1

  # Calculate Trading Strategy Returns
  trading_returns = []
  position = 0  # 1 for holding the stock, 0 for not holding
  for i in range(len(single_stock_df)):
      if single_stock_df['Signal'].iloc[i] == 'Comprar':
          position = 1  # Buy
      elif single_stock_df['Signal'].iloc[i] == 'Vender':
          position = 0  # Sell

      if position == 1:
          trading_returns.append(single_stock_df['Adj Close'].iloc[i] / single_stock_df['Adj Close'].iloc[0] - 1)
      else:
          trading_returns.append(trading_returns[-1] if trading_returns else 0)  # Hold previous return

  single_stock_df['Trading_Returns'] = trading_returns

  # Create the comparison figure
  fig_comparison = go.Figure()
  fig_comparison.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Buy_Hold_Returns'],
      mode='lines',
      name='Compra y Mantenimiento',
      line=dict(color='blue')
  ))
  fig_comparison.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Trading_Returns'],
      mode='lines',
      name='Estrategia de Trading',
      line=dict(color='purple')
  ))

  fig_comparison.update_layout(
      title=f"Comparación de Estrategia de Compra y Mantenimiento vs Estrategia de Trading para {single_ticker}",
      xaxis_title="Fecha",
      yaxis_title="Rendimiento Acumulado",
      hovermode='x unified'
  )
  st.plotly_chart(fig_comparison, use_container_width=True)

# Footer Disclaimer
st.markdown("""
---
**Disclaimer:** Esta herramienta es solo para fines educativos y no debe considerarse como asesoramiento financiero. Siempre realiza tu propia investigación antes de tomar decisiones de inversión.
""")
