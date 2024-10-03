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

Esta herramienta te permite realizar un back-testing de una estrategia de trading de pares entre dos acciones, así como una estrategia de acción única, incorporando estrategias de asignación avanzadas y posiciones en efectivo. Está diseñada para ser útil incluso en mercados donde no se permite la venta en corto.

---

**Cómo usar esta herramienta:**

1. **Selecciona el Tipo de Estrategia:** Elige entre Trading de Pares o Estrategia de Acción Única.
2. **Configura los Parámetros:**
 - **Ventana de Z-Score:** Número de días para calcular la media móvil y la desviación estándar del spread de precios.
 - **Umbrales de Entrada/Salida:** Niveles de z-score en los que la estrategia ajustará las asignaciones.
 - **Asignación Máxima (%):** Porcentaje máximo del portafolio que se puede asignar a cualquier acción individual.
3. **Ejecuta el Back-Test:** La herramienta calculará el rendimiento de la estrategia y mostrará gráficos interactivos y métricas.

**Entendiendo la estrategia:**

- **Objetivo:** Obtener ganancias de los movimientos relativos de dos acciones (en el caso de Trading de Pares) o de una sola acción ajustando las asignaciones del portafolio en función del z-score de su spread de precios.
- **Posición en Efectivo:** Cuando el spread o movimiento no indica una señal fuerte, la estrategia puede mover una porción del portafolio a efectivo para reducir la exposición al mercado.
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
      data = data['Adj Close'].dropna()
      data.name = 'Adj_Close'  # Rename the series to 'Adj_Close'
      return data
  except Exception as e:
      st.error(f"Error al obtener datos para `{ticker}`: {e}")
      return None

# User Inputs
default_ticker1 = 'AAPL'  # Default stocks
default_ticker2 = 'MSFT'

# Add a selection for strategy type
strategy_type = st.sidebar.selectbox("Selecciona el Tipo de Estrategia", ["Trading de Pares", "Estrategia de Acción Única"])

# Function to calculate performance metrics
def calculate_metrics(strategy_returns, benchmark_returns=None):
  metrics = {}
  
  # Strategy Metrics
  total_return = (strategy_returns + 1).prod() - 1
  annual_return = strategy_returns.mean() * 252
  annual_vol = strategy_returns.std() * np.sqrt(252)
  sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan
  rolling_max = (strategy_returns + 1).cumprod().cummax()
  drawdown = (strategy_returns + 1).cumprod() / rolling_max - 1
  max_drawdown = drawdown.min()
  
  metrics["Estrategia Adaptativa"] = {
      "Rendimiento Total": f"{total_return:.2%}",
      "Rendimiento Anualizado": f"{annual_return:.2%}",
      "Volatilidad Anualizada": f"{annual_vol:.2%}",
      "Ratio de Sharpe": f"{sharpe_ratio:.2f}",
      "Máxima Caída": f"{max_drawdown:.2%}",
  }
  
  if benchmark_returns is not None:
      # Benchmark Metrics
      benchmark_total = (benchmark_returns + 1).prod() - 1
      benchmark_annual = benchmark_returns.mean() * 252
      benchmark_vol = benchmark_returns.std() * np.sqrt(252)
      benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol != 0 else np.nan
      benchmark_rolling_max = (benchmark_returns + 1).cumprod().cummax()
      benchmark_drawdown = (benchmark_returns + 1).cumprod() / benchmark_rolling_max - 1
      benchmark_max_dd = benchmark_drawdown.min()
  
      metrics["Benchmark (Igual Ponderación)"] = {
          "Rendimiento Total": f"{benchmark_total:.2%}",
          "Rendimiento Anualizado": f"{benchmark_annual:.2%}",
          "Volatilidad Anualizada": f"{benchmark_vol:.2%}",
          "Ratio de Sharpe": f"{benchmark_sharpe:.2f}",
          "Máxima Caída": f"{benchmark_max_dd:.2%}",
      }
  
  return metrics

if strategy_type == "Trading de Pares":
  # ---------- Trading de Pares ----------
  ticker1 = st.sidebar.text_input("Símbolo de la Primera Acción", value=default_ticker1).upper()
  ticker2 = st.sidebar.text_input("Símbolo de la Segunda Acción", value=default_ticker2).upper()

  start_date = st.sidebar.date_input("Fecha de Inicio", value=datetime(2020, 1, 1))
  end_date = st.sidebar.date_input("Fecha de Fin", value=datetime.today())

  zscore_window = st.sidebar.number_input("Ventana de Z-Score (Días)", min_value=5, max_value=252, value=30, step=1)
  entry_zscore = st.sidebar.number_input("Umbral de Entrada (Z-Score)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
  exit_zscore = st.sidebar.number_input("Umbral de Salida (Z-Score)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

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
  data = pd.concat([stock1, stock2], axis=1).dropna()

  if data.empty:
      st.error("❌ **No hay datos superpuestos entre las fechas seleccionadas.** Por favor, ajusta el rango de fechas o los símbolos de las acciones.")
      st.stop()

  # Calculate Spread and Z-Score
  data['Spread'] = data[ticker1] - data[ticker2]
  data['Spread_Mean'] = data['Spread'].rolling(window=zscore_window, min_periods=1).mean()
  data['Spread_STD'] = data['Spread'].rolling(window=zscore_window, min_periods=1).std()
  data['Z-Score'] = (data['Spread'] - data['Spread_Mean']) / data['Spread_STD']

  # Backtesting Logic with Cash Position and Advanced Allocation (No Short Selling)
  def backtest_pairs_adaptive_no_short(data, entry_threshold, exit_threshold, max_alloc):
      # Initialize allocations
      allocations = pd.DataFrame(index=data.index, columns=['Weight_' + ticker1, 'Weight_' + ticker2, 'Weight_Cash'])
      allocations.iloc[:] = 0  # Start with all cash

      # Conditions for entering and exiting positions
      entry_condition = data['Z-Score'].abs() >= entry_threshold
      exit_condition = data['Z-Score'].abs() <= exit_threshold

      # Signals
      data['Signal'] = None
      current_position = None  # None, 'Overweight ' ticker1', 'Overweight ' ticker2'

      for i in range(len(data)):
          if entry_condition.iloc[i]:
              z = data['Z-Score'].iloc[i]
              if z > 0 and current_position != f'Overweight {ticker2}':
                  # Overweight ticker2, Underweight ticker1
                  allocations.iloc[i] = [(1 - max_alloc), max_alloc, 0]  # No short, adjust allocations
                  data['Signal'].iloc[i] = f'Overweight {ticker2}'
                  current_position = f'Overweight {ticker2}'
              elif z < 0 and current_position != f'Overweight {ticker1}':
                  # Overweight ticker1, Underweight ticker2
                  allocations.iloc[i] = [max_alloc, (1 - max_alloc), 0]
                  data['Signal'].iloc[i] = f'Overweight {ticker1}'
                  current_position = f'Overweight {ticker1}'
          elif exit_condition.iloc[i]:
              # Exit to Cash
              allocations.iloc[i] = [0, 0, 1]
              if current_position is not None:
                  data['Signal'].iloc[i] = 'Exit to Cash'
                  current_position = None
          else:
              if i > 0:
                  allocations.iloc[i] = allocations.iloc[i-1]
      
      # Fill initial positions if any
      allocations.fillna(method='ffill', inplace=True)
      allocations.fillna(0, inplace=True)  # If still NaN, set to 0

      # Calculate daily portfolio returns
      daily_returns = data[[ticker1, ticker2]].pct_change().fillna(0)
      strategy_returns = (allocations.shift(1) * daily_returns).sum(axis=1)
      strategy_returns.fillna(0, inplace=True)
      cumulative_strategy = (1 + strategy_returns).cumprod()

      # Benchmark: Equal-weighted portfolio held throughout
      benchmark_weights = np.array([0.5, 0.5])
      benchmark_returns = daily_returns.dot(benchmark_weights)
      cumulative_benchmark = (1 + benchmark_returns).cumprod()

      return strategy_returns, cumulative_strategy, cumulative_benchmark, benchmark_returns, allocations

  # Execute Backtest
  strategy_returns, cumulative_strategy, cumulative_benchmark, benchmark_returns, allocations = backtest_pairs_adaptive_no_short(
      data, entry_zscore, exit_zscore, max_allocation
  )

  # Performance Metrics
  st.header("📊 Métricas de Rendimiento")
  st.markdown("""
  La tabla a continuación resume las métricas clave de rendimiento para la estrategia adaptativa y el benchmark.
  """)

  metrics = calculate_metrics(strategy_returns, benchmark_returns)

  metrics_df = pd.DataFrame(metrics).T
  st.table(metrics_df)

  # Trade Signals Table
  st.header("📋 Señales de Asignación")
  st.markdown("""
  La tabla a continuación detalla los momentos en que las asignaciones del portafolio cambiaron en función de las señales del z-score.
  """)
  signal_df = data[['Signal']].dropna()
  st.write(signal_df)

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

  # Plot Trade Signals
  signals = data[['Signal']].dropna()
  for idx, row in signals.iterrows():
      if 'Overweight' in row['Signal']:
          fig_zscore.add_trace(go.Scatter(
              x=[idx],
              y=[data.loc[idx, 'Z-Score']],
              mode='markers',
              name=row['Signal'],
              marker=dict(symbol='triangle-up', color='green', size=12),
              showlegend=False
          ))
      elif row['Signal'] == 'Exit to Cash':
          fig_zscore.add_trace(go.Scatter(
              x=[idx],
              y=[data.loc[idx, 'Z-Score']],
              mode='markers',
              name='Exit to Cash',
              marker=dict(symbol='circle', color='black', size=8),
              showlegend=False
          ))

  fig_zscore.update_layout(
      title="Z-Score con Señales de Asignación",
      xaxis_title="Fecha",
      yaxis_title="Z-Score",
      hovermode='x unified',
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
      y=allocations['Weight_' + ticker1],
      mode='lines',
      name='Asignación a ' + ticker1,
      stackgroup='one',
      fill='tonexty'
  ))

  fig_alloc.add_trace(go.Scatter(
      x=data.index,
      y=allocations['Weight_' + ticker2],
      mode='lines',
      name='Asignación a ' + ticker2,
      stackgroup='one',
      fill='tonexty'
  ))

  fig_alloc.add_trace(go.Scatter(
      x=data.index,
      y=allocations['Weight_Cash'],
      mode='lines',
      name='Asignación a Efectivo',
      stackgroup='one',
      fill='tonexty'
  ))

  fig_alloc.update_layout(
      title="Asignación de Portafolio a lo Largo del Tiempo",
      xaxis_title="Fecha",
      yaxis_title="Porcentaje de Asignación",
      yaxis=dict(range=[0, 1]),
      hovermode='x unified',
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
      height=500
  )

  st.plotly_chart(fig_cum_returns, use_container_width=True)

elif strategy_type == "Estrategia de Acción Única":
  # ---------- Estrategia de Acción Única ----------
  default_ticker = default_ticker1  # Reuse default_ticker1 as the default single stock

  single_ticker = st.sidebar.text_input("Símbolo de la Acción", value=default_ticker).upper()

  start_date = st.sidebar.date_input("Fecha de Inicio", value=datetime(2020, 1, 1), key="single_start")
  end_date = st.sidebar.date_input("Fecha de Fin", value=datetime.today(), key="single_end")

  zscore_window = st.sidebar.number_input("Ventana de Z-Score (Días)", min_value=5, max_value=252, value=30, step=1)
  entry_zscore = st.sidebar.number_input("Umbral de Entrada (Z-Score)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
  exit_zscore = st.sidebar.number_input("Umbral de Salida (Z-Score)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

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
  single_stock_df = pd.DataFrame(single_stock_data).reset_index()
  single_stock_df.rename(columns={'Adj_Close': 'Adj_Close'}, inplace=True)  # Already named correctly

  # Generate Signals for Single Stock
  def generate_signals_single_stock(df, z_window, entry_thresh, exit_thresh):
      df['Mean'] = df['Adj_Close'].rolling(window=z_window, min_periods=1).mean()
      df['STD'] = df['Adj_Close'].rolling(window=z_window, min_periods=1).std()
      df['Z-Score'] = (df['Adj_Close'] - df['Mean']) / df['STD']

      # Initialize Signal and Position
      df['Signal'] = 'Mantener'
      position = 0  # 1 for holding the stock, 0 for holding cash

      for i in range(len(df)):
          z = df['Z-Score'].iloc[i]
          if position == 0:
              if z >= entry_thresh:
                  df.at[i, 'Signal'] = 'Comprar'
                  position = 1
              elif z <= -entry_thresh:
                  df.at[i, 'Signal'] = 'Comprar'
                  position = 1
          elif position == 1:
              if abs(z) <= exit_thresh:
                  df.at[i, 'Signal'] = 'Vender'
                  position = 0
      return df

  # Generate signals for the single stock
  single_stock_df = generate_signals_single_stock(single_stock_df, zscore_window, entry_zscore, exit_zscore)

  # Calculate Positions (No Short Selling)
  single_stock_df['Position'] = 0
  single_stock_df['Position'] = np.where(single_stock_df['Signal'] == 'Comprar', 1, single_stock_df['Position'])
  single_stock_df['Position'] = np.where(single_stock_df['Signal'] == 'Vender', 0, single_stock_df['Position'])
  single_stock_df['Position'] = single_stock_df['Position'].ffill().fillna(0)

  # Calculate Daily Returns
  single_stock_df['Daily_Return'] = single_stock_df['Adj_Close'].pct_change().fillna(0)
  single_stock_df['Strategy_Return'] = single_stock_df['Position'].shift(1) * single_stock_df['Daily_Return']
  single_stock_df['Strategy_Return'].fillna(0, inplace=True)

  # Calculate Cumulative Returns
  single_stock_df['Cumulative_Strategy'] = (1 + single_stock_df['Strategy_Return']).cumprod()
  single_stock_df['Cumulative_Buy_Hold'] = (1 + single_stock_df['Daily_Return']).cumprod()

  # Performance Metrics
  st.header("📊 Métricas de Rendimiento")
  st.markdown("""
  La tabla a continuación resume las métricas clave de rendimiento para la estrategia de acción única y el benchmark de compra y mantenimiento.
  """)

  metrics_strategy = single_stock_df['Strategy_Return']
  metrics_buy_hold = single_stock_df['Daily_Return']

  metrics = calculate_metrics(metrics_strategy, metrics_buy_hold)

  metrics_df = pd.DataFrame(metrics).T
  st.table(metrics_df)

  # Trade Signals Table
  st.header("📋 Señales de Trading")
  st.markdown("""
  La tabla a continuación detalla los momentos en que las señales de compra y venta fueron generadas en función del z-score.
  """)
  signals_df = single_stock_df[single_stock_df['Signal'] != 'Mantener'][['Date', 'Signal']].reset_index(drop=True)
  st.write(signals_df)

  # Visualization with Plotly

  # 1. Adjusted Close Price Plot
  st.header("📊 Precio Ajustado de la Acción")
  st.markdown("""
  Este gráfico muestra el precio ajustado de cierre de la acción seleccionada a lo largo del período elegido.
  """)
  fig_price = go.Figure()

  fig_price.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Adj_Close'],
      mode='lines',
      name='Precio Ajustado'
  ))

  fig_price.update_layout(
      title=f"Precio Ajustado de {single_ticker}",
      xaxis_title="Fecha",
      yaxis_title="Precio Ajustado",
      hovermode='x unified',
      height=500
  )

  st.plotly_chart(fig_price, use_container_width=True)

  # 2. Z-Score Plot with Buy/Sell Signals
  st.header("📈 Z-Score de la Acción con Señales de Compra/Venta")
  st.markdown("""
  Este gráfico muestra el z-score de la acción, junto con los umbrales de entrada y salida. Ilustra cómo la estrategia genera señales de compra y venta en función del z-score.
  """)

  fig_zscore_single = make_subplots(rows=1, cols=1, shared_xaxes=True)

  # Plot Z-Score
  fig_zscore_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Z-Score'],
      mode='lines',
      name='Z-Score',
      line=dict(color='blue')
  ))

  # Plot Thresholds
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
      marker=dict(symbol='triangle-up', color='green', size=12),
      hovertemplate='Fecha: %{x}<br>Señal: Comprar'
  ))
  
  fig_zscore_single.add_trace(go.Scatter(
      x=sell_signals['Date'],
      y=sell_signals['Z-Score'],
      mode='markers',
      name='Señales de Venta',
      marker=dict(symbol='triangle-down', color='red', size=12),
      hovertemplate='Fecha: %{x}<br>Señal: Vender'
  ))

  fig_zscore_single.update_layout(
      title=f"Z-Score de {single_ticker} con Señales de Compra/Venta",
      xaxis_title="Fecha",
      yaxis_title="Z-Score",
      hovermode='x unified',
      height=500
  )
  st.plotly_chart(fig_zscore_single, use_container_width=True)

  # 3. Asignación de Portafolio a lo Largo del Tiempo
  st.header("📈 Asignación de Portafolio a lo Largo del Tiempo")
  st.markdown("""
  Este gráfico muestra cómo cambia la asignación del portafolio a la acción y a efectivo a lo largo del tiempo, basándose en las señales del z-score.
  """)

  fig_alloc_single = go.Figure()

  fig_alloc_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Position'],
      mode='lines',
      name='Posición en Acción',
      line=dict(color='blue')
  ))

  fig_alloc_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=1 - single_stock_df['Position'],
      mode='lines',
      name='Posición en Efectivo',
      line=dict(color='orange')
  ))

  fig_alloc_single.update_layout(
      title="Asignación de Portafolio a lo Largo del Tiempo",
      xaxis_title="Fecha",
      yaxis_title="Porcentaje de Asignación",
      yaxis=dict(range=[0, 1]),
      hovermode='x unified',
      height=500
  )

  st.plotly_chart(fig_alloc_single, use_container_width=True)

  # 4. Cumulative Returns Plot
  st.header("📈 Rendimiento Acumulado de la Estrategia vs. Benchmark")
  st.markdown("""
  Este gráfico compara los rendimientos acumulados de la estrategia de trading contra un benchmark de compra y mantenimiento.
  """)

  fig_cum_returns_single = make_subplots(rows=1, cols=1, shared_xaxes=True)

  fig_cum_returns_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Cumulative_Strategy'],
      mode='lines',
      name='Estrategia de Trading',
      line=dict(color='purple')
  ))

  fig_cum_returns_single.add_trace(go.Scatter(
      x=single_stock_df['Date'],
      y=single_stock_df['Cumulative_Buy_Hold'],
      mode='lines',
      name='Compra y Mantenimiento',
      line=dict(color='grey')
  ))

  fig_cum_returns_single.update_layout(
      title="Rendimiento Acumulado",
      xaxis_title="Fecha",
      yaxis_title="Rendimiento Acumulado",
      hovermode='x unified',
      height=500
  )

  st.plotly_chart(fig_cum_returns_single, use_container_width=True)

# Footer Disclaimer
st.markdown("""
---
**Disclaimer:** Esta herramienta es solo para fines educativos y no debe considerarse como asesoramiento financiero. Siempre realiza tu propia investigación antes de tomar decisiones de inversión.
""")
