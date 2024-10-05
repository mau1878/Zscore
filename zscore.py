import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# Configuración de la página de Streamlit
st.set_page_config(
page_title="📈 Adaptive Pairs Trading Backtester",
layout="wide",
)

# Título y Descripción
st.title("📈 Adaptive Pairs Trading Backtester")
st.markdown("""
Bienvenido al **Adaptive Pairs Trading Backtester**!

Esta herramienta te permite realizar un backtesting de una estrategia de trading de pares entre dos acciones, así como una estrategia de acción única, incorporando estrategias de asignación avanzadas y posiciones en efectivo. **Nota:** La venta en corto no está disponible en Argentina, por lo que todas las posiciones son largas o en efectivo.

---

### **Cómo usar esta herramienta:**

1. **Selecciona el Tipo de Estrategia:** Elige entre Trading de Pares o Estrategia de Acción Única.
2. **Configura los Parámetros:**
- **Ventana de Z-Score:** Número de días para calcular la media móvil y la desviación estándar del spread de precios.
- **Umbrales de Entrada/Salida:** Niveles de z-score en los que la estrategia ajustará las asignaciones.
- **Asignación Máxima (%):** Porcentaje máximo del portafolio que se puede asignar a cualquier acción individual.
3. **Ejecuta el Backtest:** La herramienta calculará el rendimiento de la estrategia y mostrará gráficos interactivos y métricas.

### **Entendiendo la estrategia:**

- **Objetivo:** Obtener ganancias de los movimientos relativos de dos acciones (en el caso de Trading de Pares) o de una sola acción ajustando las asignaciones del portafolio en función del z-score de su spread de precios o precio individual.
- **Asignación Parcial:** Las asignaciones a cada acción (y efectivo) son proporcionales a la magnitud del z-score dentro de los umbrales definidos, respetando siempre el límite de asignación máxima y sin permitir ventas en corto.
- **Posición en Efectivo:** Cuando el spread o movimiento no indica una señal fuerte, la estrategia puede mover una porción del portafolio a efectivo para reducir la exposición al mercado.

---
""")

# Barra lateral para Entradas del Usuario
st.sidebar.header("📌 Configuración de Parámetros")

# Función para obtener datos de acciones con caché y manejo de errores
@st.cache_data(ttl=60*60)  # Cache de datos por 1 hora
def get_stock_data(ticker, start, end, name=None):
  """
  Descarga datos históricos de una acción utilizando yfinance.

  Parameters:
      ticker (str): Símbolo de la acción.
      start (datetime): Fecha de inicio.
      end (datetime): Fecha de fin.
      name (str, optional): Nombre personalizado para la columna.

  Returns:
      pd.Series: Serie de precios ajustados de cierre.
  """
  try:
      data = yf.download(ticker, start=start, end=end, progress=False)
      if data.empty:
          st.error(f"No se pudo obtener datos para `{ticker}`. Por favor, verifica el símbolo de la acción.")
          return None
      adj_close = data['Adj Close'].dropna()
      adj_close.name = name if name else ticker
      return adj_close
  except Exception as e:
      st.error(f"Error al obtener datos para `{ticker}`: {e}")
      return None

# Entradas predeterminadas
default_ticker1 = 'AAPL'  # Primera acción por defecto
default_ticker2 = 'MSFT'  # Segunda acción por defecto

# Selección del tipo de estrategia
strategy_type = st.sidebar.selectbox(
  "Selecciona el Tipo de Estrategia",
  ["Trading de Pares", "Estrategia de Acción Única"]
)

# Función para calcular métricas de rendimiento
def calculate_metrics(strategy_returns, benchmark_returns=None):
  """
  Calcula métricas clave de rendimiento para la estrategia y el benchmark.

  Parameters:
      strategy_returns (pd.Series): Retornos diarios de la estrategia.
      benchmark_returns (pd.Series, optional): Retornos diarios del benchmark.

  Returns:
      dict: Diccionario con las métricas calculadas.
  """
  metrics = {}
  
  # Métricas de la Estrategia
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
      # Métricas del Benchmark
      benchmark_total = (benchmark_returns + 1).prod() - 1
      benchmark_annual = benchmark_returns.mean() * 252
      benchmark_vol = benchmark_returns.std() * np.sqrt(252)
      benchmark_sharpe = benchmark_annual / benchmark_vol if benchmark_vol != 0 else np.nan
      benchmark_rolling_max = (benchmark_returns + 1).cumprod().cummax()
      benchmark_drawdown = (benchmark_returns + 1).cumprod() / benchmark_rolling_max - 1
      benchmark_max_dd = benchmark_drawdown.min()
  
      metrics["Benchmark"] = {
          "Rendimiento Total": f"{benchmark_total:.2%}",
          "Rendimiento Anualizado": f"{benchmark_annual:.2%}",
          "Volatilidad Anualizada": f"{benchmark_vol:.2%}",
          "Ratio de Sharpe": f"{benchmark_sharpe:.2f}",
          "Máxima Caída": f"{benchmark_max_dd:.2%}",
      }
  
  return metrics

if strategy_type == "Trading de Pares":
  # ---------- Trading de Pares ----------
  st.header("📈 Estrategia de Trading de Pares")
  
  # Entradas de los usuarios para los símbolos de las acciones
  ticker1 = st.sidebar.text_input(
      "Símbolo de la Primera Acción",
      value=default_ticker1,
      help="Ingresa el ticker de la primera acción (ejemplo: AAPL)."
  ).upper()
  
  ticker2 = st.sidebar.text_input(
      "Símbolo de la Segunda Acción",
      value=default_ticker2,
      help="Ingresa el ticker de la segunda acción (ejemplo: MSFT)."
  ).upper()

  # Selección de fechas
  start_date = st.sidebar.date_input(
      "Fecha de Inicio",
      value=datetime(2020, 1, 1),
      help="Selecciona la fecha de inicio para el backtest."
  )
  end_date = st.sidebar.date_input(
      "Fecha de Fin",
      value=datetime.today(),
      help="Selecciona la fecha de fin para el backtest."
  )

  # Parámetros de la estrategia
  zscore_window = st.sidebar.number_input(
      "Ventana de Z-Score (Días)",
      min_value=5,
      max_value=252,
      value=30,
      step=1,
      help="Número de días para calcular la media y desviación estándar del spread."
  )
  entry_zscore = st.sidebar.number_input(
      "Umbral de Entrada (Z-Score)",
      min_value=0.1,
      max_value=5.0,
      value=1.0,
      step=0.1,
      help="Nivel de z-score para comenzar a ajustar asignaciones."
  )
  exit_zscore = st.sidebar.number_input(
      "Umbral de Salida (Z-Score)",
      min_value=0.0,
      max_value=5.0,
      value=0.0,
      step=0.1,
      help="Nivel de z-score para cerrar posiciones completamente."
  )

  max_allocation = st.sidebar.number_input(
      "Asignación Máxima a una Acción Individual (%)",
      min_value=10,
      max_value=100,
      value=100,
      step=5,
      help="Porcentaje máximo del portafolio que se puede asignar a una acción individual."
  )
  max_allocation /= 100  # Convertir a decimal

  # Validar Entradas de Fechas
  if start_date >= end_date:
      st.sidebar.error("⚠️ **La fecha de inicio debe ser anterior a la fecha de fin.**")

  # Obtener Datos de Acciones
  with st.spinner("🔄 Obteniendo datos de las acciones..."):
      stock1 = get_stock_data(ticker1, start_date, end_date)
      stock2 = get_stock_data(ticker2, start_date, end_date)

  # Detener ejecución si la obtención de datos falló
  if stock1 is None or stock2 is None:
      st.stop()

  # Combinar Datos en un DataFrame
  data = pd.concat([stock1, stock2], axis=1).dropna()

  if data.empty:
      st.error("❌ **No hay datos superpuestos entre las fechas seleccionadas.** Por favor, ajusta el rango de fechas o los símbolos de las acciones.")
      st.stop()

  # Calcular Spread y Z-Score
  data['Spread'] = data[ticker1] - data[ticker2]
  data['Spread_Mean'] = data['Spread'].rolling(window=zscore_window, min_periods=1).mean()
  data['Spread_STD'] = data['Spread'].rolling(window=zscore_window, min_periods=1).std()
  data['Z-Score'] = (data['Spread'] - data['Spread_Mean']) / data['Spread_STD']

  # Lógica de Backtesting con Asignación Parcial (Sin Venta en Corto)
  def backtest_pairs_adaptive_partial_alloc(data, ticker1, ticker2, entry_threshold, exit_threshold, max_alloc):
      """
      Realiza el backtesting de una estrategia de pares con asignaciones parciales sin venta en corto.

      Parameters:
          data (pd.DataFrame): Datos de precios y z-score.
          ticker1 (str): Primer ticker.
          ticker2 (str): Segundo ticker.
          entry_threshold (float): Umbral de entrada para el z-score.
          exit_threshold (float): Umbral de salida para el z-score.
          max_alloc (float): Asignación máxima por acción.

      Returns:
          tuple: Retornos de la estrategia, retornos acumulados, benchmark acumulado, retornos del benchmark, asignaciones.
      """
      # Inicializar asignaciones: ticker1, ticker2 y efectivo
      allocations = pd.DataFrame(index=data.index, columns=[ticker1, ticker2, 'Cash'])

      # Calcular asignaciones basadas en el z-score
      for i in range(len(data)):
          idx = data.index[i]
          z = data['Z-Score'].iloc[i]
          alloc1 = 0
          alloc2 = 0

          # Calcular asignación para ticker1
          if z <= -entry_threshold:
              alloc1 = max_alloc
          elif -entry_threshold < z < -exit_threshold:
              alloc1 = max_alloc * (entry_threshold - abs(z)) / (entry_threshold - exit_threshold)
          elif -exit_threshold <= z <= exit_threshold:
              alloc1 = 0
          elif exit_threshold < z < entry_threshold:
              alloc2 = max_alloc * (z - exit_threshold) / (entry_threshold - exit_threshold)
          elif z >= entry_threshold:
              alloc2 = max_alloc

          alloc1 = min(max_alloc, max(0, alloc1))
          alloc2 = min(max_alloc, max(0, alloc2))
          cash_alloc = 1 - (alloc1 + alloc2)
          allocations.at[idx, ticker1] = alloc1
          allocations.at[idx, ticker2] = alloc2
          allocations.at[idx, 'Cash'] = cash_alloc

      # Calcular retornos diarios del portafolio
      daily_returns = data[[ticker1, ticker2]].pct_change().fillna(0)

      # Calcular retornos de la estrategia
      strategy_returns = (allocations.shift(1)[[ticker1, ticker2]] * daily_returns).sum(axis=1)

      strategy_returns.fillna(0, inplace=True)  # Asegurar que no haya NaN en los retornos

      # Calcular retornos acumulados
      cumulative_strategy = (1 + strategy_returns).cumprod()

      # Benchmark: Portafolio igual ponderado mantenido constantemente
      benchmark_weights = np.array([0.5, 0.5])
      benchmark_returns = daily_returns.dot(benchmark_weights)
      cumulative_benchmark = (1 + benchmark_returns).cumprod()

      return strategy_returns, cumulative_strategy, cumulative_benchmark, benchmark_returns, allocations

  # Ejecutar Backtest
  with st.spinner("⏳ Ejecutando backtest de la estrategia..."):
      strategy_returns, cumulative_strategy, cumulative_benchmark, benchmark_returns, allocations = backtest_pairs_adaptive_partial_alloc(
          data, ticker1, ticker2, entry_zscore, exit_zscore, max_allocation
      )

  # Métricas de Rendimiento
  st.header("📊 Métricas de Rendimiento")
  st.markdown("""
  La tabla a continuación resume las métricas clave de rendimiento para la **estrategia adaptativa** y el **benchmark**.
  """)

  metrics = calculate_metrics(strategy_returns, benchmark_returns)

  metrics_df = pd.DataFrame(metrics).T
  st.table(metrics_df)

  # Visualizaciones con Plotly
  st.header("📈 Visualizaciones de la Estrategia")

  # 1. Gráfico de Precios de Acciones
  with st.expander("🔍 Ver Precios de Acciones"):
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
          title=f"📈 Precios de Cierre Ajustados - {ticker1} & {ticker2}",
          xaxis_title="Fecha",
          yaxis_title="Precio (USD)",
          hovermode='x unified',
          hoverlabel=dict(
              bgcolor="white",  # Background color of the hover label
              font_size=16,     # Increase font size for better visibility
              font_family="Arial",  # Font family of hover label
              bordercolor="gray",  # Optional: Add a border to the hover label
              namelength=-1,       # Show full name in hover text
          ),
          height=500
      )

      st.plotly_chart(fig_prices, use_container_width=True)

  # 2. Gráfico de Z-Score
  with st.expander("🔍 Ver Z-Score"):
      st.markdown("""
      Este gráfico muestra el z-score del spread entre las dos acciones, junto con los umbrales de entrada y salida.
      """)
      fig_zscore = make_subplots(rows=1, cols=1, shared_xaxes=True)

      # Graficar Z-Score
      fig_zscore.add_trace(go.Scatter(
          x=data.index,
          y=data['Z-Score'],
          mode='lines',
          name='Z-Score',
          line=dict(color='blue')
      ))

      # Graficar Umbrales
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

      fig_zscore.update_layout(
          title="📊 Z-Score del Spread",
          xaxis_title="Fecha",
          yaxis_title="Z-Score",
          hovermode='x unified',
          height=500
      )

      st.plotly_chart(fig_zscore, use_container_width=True)

  # 3. Gráfico de Asignación de Portafolio a lo Largo del Tiempo
  with st.expander("🔍 Ver Asignación de Portafolio"):
      st.markdown("""
      Este gráfico muestra cómo cambian las asignaciones del portafolio para cada acción y efectivo a lo largo del tiempo, basándose en las señales del z-score.
      """)
      fig_alloc = make_subplots(rows=1, cols=1, shared_xaxes=True)

      fig_alloc.add_trace(go.Scatter(
          x=allocations.index,
          y=allocations[ticker1],
          mode='lines',
          name=f'Asignación a {ticker1}',
          stackgroup='one',
          fill='tonexty'
      ))

      fig_alloc.add_trace(go.Scatter(
          x=allocations.index,
          y=allocations[ticker2],
          mode='lines',
          name=f'Asignación a {ticker2}',
          stackgroup='one',
          fill='tonexty'
      ))

      fig_alloc.add_trace(go.Scatter(
          x=allocations.index,
          y=allocations['Cash'],
          mode='lines',
          name='Asignación a Efectivo',
          stackgroup='one',
          fill='tonexty'
      ))

      fig_alloc.update_layout(
          title="📈 Asignación de Portafolio a lo Largo del Tiempo",
          xaxis_title="Fecha",
          yaxis_title="Porcentaje de Asignación",
          yaxis=dict(range=[0, 1]),
          hovermode='x unified',
          height=500
      )

      st.plotly_chart(fig_alloc, use_container_width=True)

  # 4. Gráfico de Rendimiento Acumulado
  with st.expander("🔍 Ver Rendimiento Acumulado"):
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
          title="📈 Rendimiento Acumulado",
          xaxis_title="Fecha",
          yaxis_title="Rendimiento Acumulado",
          hovermode='x unified',
          height=500
      )

      st.plotly_chart(fig_cum_returns, use_container_width=True)

elif strategy_type == "Estrategia de Acción Única":
  # ---------- Estrategia de Acción Única ----------
  st.header("📈 Estrategia de Acción Única")
  
  # Entradas de los usuarios para el símbolo de la acción
  default_ticker = default_ticker1  # Usar ticker1 como predeterminado para acción única

  single_ticker = st.sidebar.text_input(
      "Símbolo de la Acción",
      value=default_ticker,
      help="Ingresa el ticker de la acción (ejemplo: AAPL)."
  ).upper()

  # Selección de fechas
  start_date = st.sidebar.date_input(
      "Fecha de Inicio",
      value=datetime(2020, 1, 1),
      key="single_start",
      help="Selecciona la fecha de inicio para el backtest."
  )
  end_date = st.sidebar.date_input(
      "Fecha de Fin",
      value=datetime.today(),
      key="single_end",
      help="Selecciona la fecha de fin para el backtest."
  )

  # Parámetros de la estrategia
  zscore_window = st.sidebar.number_input(
      "Ventana de Z-Score (Días)",
      min_value=5,
      max_value=252,
      value=30,
      step=1,
      help="Número de días para calcular la media y desviación estándar del precio."
  )
  entry_zscore = st.sidebar.number_input(
      "Umbral de Entrada (Z-Score)",
      min_value=0.1,
      max_value=5.0,
      value=1.0,
      step=0.1,
      help="Nivel de z-score para comenzar a ajustar asignaciones."
  )
  exit_zscore = st.sidebar.number_input(
      "Umbral de Salida (Z-Score)",
      min_value=0.0,
      max_value=5.0,
      value=0.0,
      step=0.1,
      help="Nivel de z-score para cerrar posiciones completamente."
  )

  max_allocation = st.sidebar.number_input(
      "Asignación Máxima a la Acción (%)",
      min_value=10,
      max_value=100,
      value=100,
      step=5,
      help="Porcentaje máximo del portafolio que se puede asignar a la acción."
  )
  max_allocation /= 100  # Convertir a decimal

  # Validar Entradas de Fechas
  if start_date >= end_date:
      st.sidebar.error("⚠️ **La fecha de inicio debe ser anterior a la fecha de fin.**")

  # Obtener Datos de la Acción Única
  with st.spinner("🔄 Obteniendo datos de la acción..."):
      single_stock_data = get_stock_data(single_ticker, start_date, end_date)

  # Detener ejecución si la obtención de datos falló
  if single_stock_data is None:
      st.stop()

  # Preparar DataFrame para la acción única
  single_stock_df = pd.DataFrame(single_stock_data).reset_index()
  single_stock_df.rename(columns={single_stock_data.name: 'Adj_Close'}, inplace=True)  # Renombrar la columna correctamente

  # Generar Asignaciones para la Acción Única
  def calculate_allocations_single_stock(df, z_window, entry_thresh, exit_thresh, max_alloc):
      """
      Calcula las asignaciones basadas en el z-score del precio de la acción.

      Parameters:
          df (pd.DataFrame): Datos de precios.
          z_window (int): Ventana para calcular el z-score.
          entry_thresh (float): Umbral para comenzar a ajustar asignaciones.
          exit_thresh (float): Umbral para cerrar posiciones completamente.
          max_alloc (float): Asignación máxima permitida.

      Returns:
          pd.DataFrame: DataFrame con asignaciones calculadas.
      """
      df['Mean'] = df['Adj_Close'].rolling(window=z_window, min_periods=1).mean()
      df['STD'] = df['Adj_Close'].rolling(window=z_window, min_periods=1).std()
      df['Z-Score'] = (df['Adj_Close'] - df['Mean']) / df['STD']

      # Inicializar asignaciones
      allocations = pd.DataFrame(index=df.index, columns=[single_ticker, 'Cash'])

      for i in range(len(df)):
          z = df['Z-Score'].iloc[i]
          if z <= -entry_thresh:
              alloc = max_alloc
          elif -entry_thresh < z < -exit_thresh:
              alloc = max_alloc * (entry_thresh - abs(z)) / (entry_thresh - exit_thresh)
          elif -exit_thresh <= z <= exit_thresh:
              alloc = 0
          elif exit_thresh < z < entry_thresh:
              alloc = 0  # No asignamos posición corta
          elif z >= entry_thresh:
              alloc = 0  # No asignamos posición corta

          alloc = min(max_alloc, max(0, alloc))
          cash_alloc = 1 - alloc
          allocations.at[i, single_ticker] = alloc
          allocations.at[i, 'Cash'] = cash_alloc

      df = pd.concat([df, allocations], axis=1)
      return df

  # Calcular asignaciones
  single_stock_df = calculate_allocations_single_stock(single_stock_df, zscore_window, entry_zscore, exit_zscore, max_allocation)

  # Calcular Retornos Diarios
  single_stock_df['Daily_Return'] = single_stock_df['Adj_Close'].pct_change().fillna(0)

  # Calcular Retornos de la Estrategia
  single_stock_df['Strategy_Return'] = single_stock_df[single_ticker].shift(1) * single_stock_df['Daily_Return']

  single_stock_df['Strategy_Return'].fillna(0, inplace=True)

  # Calcular Retornos Acumulados
  single_stock_df['Cumulative_Strategy'] = (1 + single_stock_df['Strategy_Return']).cumprod()
  single_stock_df['Cumulative_Buy_Hold'] = (1 + single_stock_df['Daily_Return']).cumprod()

  # Métricas de Rendimiento
  st.header("📊 Métricas de Rendimiento")
  st.markdown("""
  La tabla a continuación resume las métricas clave de rendimiento para la **estrategia de acción única** y el **benchmark de compra y mantenimiento**.
  """)

  metrics_strategy = single_stock_df['Strategy_Return']
  metrics_buy_hold = single_stock_df['Daily_Return']

  metrics = calculate_metrics(metrics_strategy, metrics_buy_hold)

  metrics_df = pd.DataFrame(metrics).T
  st.table(metrics_df)

  # Visualizaciones con Plotly
  st.header("📈 Visualizaciones de la Estrategia")

  # 1. Gráfico de Precio Ajustado de la Acción
  with st.expander("🔍 Ver Precio Ajustado"):
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
          title=f"📈 Precio Ajustado de {single_ticker}",
          xaxis_title="Fecha",
          yaxis_title="Precio Ajustado (USD)",
          hovermode='x unified',
          height=500
      )

      st.plotly_chart(fig_price, use_container_width=True)

  # 2. Gráfico de Z-Score
  with st.expander("🔍 Ver Z-Score"):
      st.markdown("""
      Este gráfico muestra el z-score de la acción, junto con los umbrales de entrada y salida.
      """)
      fig_zscore_single = make_subplots(rows=1, cols=1, shared_xaxes=True)

      # Graficar Z-Score
      fig_zscore_single.add_trace(go.Scatter(
          x=single_stock_df['Date'],
          y=single_stock_df['Z-Score'],
          mode='lines',
          name='Z-Score',
          line=dict(color='blue')
      ))

      # Graficar Umbrales
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

      fig_zscore_single.update_layout(
          title=f"📊 Z-Score de {single_ticker}",
          xaxis_title="Fecha",
          yaxis_title="Z-Score",
          hovermode='x unified',
          height=500
      )
      st.plotly_chart(fig_zscore_single, use_container_width=True)

  # 3. Gráfico de Asignación de Portafolio a lo Largo del Tiempo
  with st.expander("🔍 Ver Asignación de Portafolio"):
      st.markdown("""
      Este gráfico muestra cómo cambia la asignación del portafolio entre la acción y efectivo a lo largo del tiempo, basándose en las señales del z-score.
      """)
      fig_alloc_single = make_subplots(rows=1, cols=1, shared_xaxes=True)

      fig_alloc_single.add_trace(go.Scatter(
          x=single_stock_df['Date'],
          y=single_stock_df[single_ticker],
          mode='lines',
          name='Asignación a Acción',
          stackgroup='one',
          fill='tonexty'
      ))

      fig_alloc_single.add_trace(go.Scatter(
          x=single_stock_df['Date'],
          y=single_stock_df['Cash'],
          mode='lines',
          name='Asignación a Efectivo',
          stackgroup='one',
          fill='tonexty'
      ))

      fig_alloc_single.update_layout(
          title="📈 Asignación de Portafolio a lo Largo del Tiempo",
          xaxis_title="Fecha",
          yaxis_title="Porcentaje de Asignación",
          yaxis=dict(range=[0, 1]),
          hovermode='x unified',
          height=500
      )

      st.plotly_chart(fig_alloc_single, use_container_width=True)

  # 4. Gráfico de Rendimiento Acumulado
  with st.expander("🔍 Ver Rendimiento Acumulado"):
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
          title="📈 Rendimiento Acumulado",
          xaxis_title="Fecha",
          yaxis_title="Rendimiento Acumulado",
          hovermode='x unified',
          height=500
      )

      st.plotly_chart(fig_cum_returns_single, use_container_width=True)

# Aviso Legal en el Footer
st.markdown("""
---
**Disclaimer:** Esta herramienta es solo para fines educativos y no debe considerarse como asesoramiento financiero. Siempre realiza tu propia investigación antes de tomar decisiones de inversión.
""")
