import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
import lightgbm as lgb
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import logging
import io
import hashlib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import openpyxl

# Set page config as the first Streamlit command
st.set_page_config(page_title="Forecasting Pipeline", layout="wide")

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {background-color: #1e1e1e; color: #ffffff;}
    .stButton>button {background-color: #1e90ff; color: #ffffff; border-radius: 5px; border: none;}
    .stButton>button:hover {background-color: #4682b4; color: #ffffff;}
    .stSidebar {background-color: #2c2c2c; color: #ffffff;}
    .stDataFrame {background-color: #2c2c2c; color: #ffffff; border: 1px solid #444444; border-radius: 5px;}
    h1, h2, h3, h4, h5, h6 {color: #ffffff;}
    .stMarkdown, .stText, .stSelectbox, .stMultiselect, .stCheckbox, .stFileUploader {color: #ffffff;}
    .stSelectbox>div>div, .stMultiselect>div>div {background-color: #2c2c2c; color: #ffffff;}
    .css-1d391kg {color: #ffffff;} /* Label color */
    table {color: #ffffff; background-color: #2c2c2c;}
    th, td {border: 1px solid #444444; color: #ffffff;}
    .stPlotlyChart {background-color: #2c2c2c; color: #ffffff;}
    </style>
""", unsafe_allow_html=True)

# Setup logging
logging.basicConfig(filename='forecasting_errors.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Metrics Function
def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Calculate error metrics."""
    epsilon = 1e-10
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    mape = np.mean(np.abs((actual - pred) / (actual + epsilon))) * 100
    smape = np.mean(2.0 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + epsilon)) * 100
    bias = np.mean(actual - pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'SMAPE': smape, 'Bias': bias}

# 2. Split Series
def split_series(ts: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Devuelve train, validation y hold_out (3 últimos meses)."""
    if len(ts) < 24:  # Minimum 24 months
        raise ValueError("Serie demasiado corta")
    hold_out = ts.iloc[-3:]      # Last 3 months
    validation = ts.iloc[-6:-3]  # Months -6 to -4
    train = ts.iloc[:-6]         # Rest
    return train, validation, hold_out

# 3. Model Definitions
def model_naive(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    preds = np.full(3, train.iloc[-1])
    rmse = np.sqrt(mean_squared_error(hold_out, preds))
    return preds, rmse

def model_ets(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    try:
        ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
        preds = ets_model.forecast(steps=3)
        rmse = np.sqrt(mean_squared_error(hold_out, preds))
        return preds, rmse
    except:
        preds = np.full(3, train.iloc[-1])
        rmse = np.sqrt(mean_squared_error(hold_out, preds))
        return preds, rmse

def model_arima(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    try:
        model = pm.auto_arima(train, seasonal=False, error_action='ignore', suppress_warnings=True)
        preds = model.predict(n_periods=3)
        rmse = np.sqrt(mean_squared_error(hold_out, preds))
        return preds, rmse
    except:
        preds = np.full(3, train.iloc[-1])
        rmse = np.sqrt(mean_squared_error(hold_out, preds))
        return preds, rmse

def model_sarima(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    try:
        model = pm.auto_arima(train, seasonal=True, m=12, error_action='ignore', suppress_warnings=True)
        preds = model.predict(n_periods=3)
        rmse = np.sqrt(mean_squared_error(hold_out, preds))
        return preds, rmse
    except:
        preds = np.full(3, train.iloc[-1])
        rmse = np.sqrt(mean_squared_error(hold_out, preds))
        return preds, rmse

def model_prophet(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    train_df = pd.DataFrame({'ds': train.index, 'y': train.values})
    try:
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m.fit(train_df)
        future = pd.date_range(start=train_df['ds'].iloc[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')
        future_df = pd.DataFrame({'ds': future})
        forecast = m.predict(future_df)
        preds = forecast['yhat'].values
        rmse = np.sqrt(mean_squared_error(hold_out, preds))
        return preds, rmse
    except:
        preds = np.full(3, train.iloc[-1])
        rmse = np.sqrt(mean_squared_error(hold_out, preds))
        return preds, rmse

def model_rf(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    def create_lags(series, n=3):
        df = pd.DataFrame({'y': series})
        for i in range(1, n+1):
            df[f'lag_{i}'] = df['y'].shift(i)
        return df.dropna()
    
    df_lags = create_lags(ts.values, n=3)
    train_size = len(train)
    df_train = df_lags.iloc[:train_size-3]
    df_test = df_lags.iloc[train_size-3:train_size]
    X_train = df_train.drop('y', axis=1)
    y_train = df_train['y']
    X_test = df_test.drop('y', axis=1)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(hold_out, preds))
    return preds, rmse

def model_xgb(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    def create_lags(series, n=3):
        df = pd.DataFrame({'y': series})
        for i in range(1, n+1):
            df[f'lag_{i}'] = df['y'].shift(i)
        return df.dropna()
    
    df_lags = create_lags(ts.values, n=3)
    train_size = len(train)
    df_train = df_lags.iloc[:train_size-3]
    df_test = df_lags.iloc[train_size-3:train_size]
    X_train = df_train.drop('y', axis=1)
    y_train = df_train['y']
    X_test = df_test.drop('y', axis=1)
    
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(hold_out, preds))
    return preds, rmse

def model_lstm(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    n_lags = 3
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1,1))
    
    def create_supervised(data, n_lags):
        df = pd.DataFrame(data)
        for i in range(1, n_lags+1):
            df[f'lag_{i}'] = df[0].shift(i)
        df.dropna(inplace=True)
        return df.values
    
    supervised = create_supervised(train_scaled, n_lags)
    X = supervised[:, 1:]
    y = supervised[:, 0]
    X_train = X.reshape((X.shape[0], n_lags, 1))
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y, epochs=50, verbose=0, callbacks=[early_stopping])
    
    # Predict
    last_n = ts.iloc[-n_lags:].values.reshape(-1,1)
    last_n_scaled = scaler.transform(last_n)
    X_test = last_n_scaled[-n_lags:].reshape((1, n_lags, 1))
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    preds = np.repeat(preds, 3)[:3]  # Repeat for 3 months
    rmse = np.sqrt(mean_squared_error(hold_out, preds))
    return preds, rmse

def model_kalman(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    kf = KalmanFilter(initial_state_mean=train.iloc[0], n_dim_obs=1, n_dim_state=1,
                      transition_matrices=[1], observation_matrices=[1],
                      observation_covariance=1, transition_covariance=0.1)
    state_means, _ = kf.filter(train.values.reshape(-1,1))
    preds = np.full(3, state_means[-1])
    rmse = np.sqrt(mean_squared_error(hold_out, preds))
    return preds, rmse

def model_lgb(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    def create_lags_df(series, n=3):
        df = pd.DataFrame({'y': series})
        for i in range(1, n+1):
            df[f'lag_{i}'] = df['y'].shift(i)
        return df.dropna()
    
    df_lags = create_lags_df(ts.values, n=3)
    train_size = len(train)
    df_train = df_lags.iloc[:train_size-3]
    df_test = df_lags.iloc[train_size-3:train_size]
    X_train = df_train.drop('y', axis=1)
    y_train = df_train['y']
    X_test = df_test.drop('y', axis=1)
    
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    lgb_model.fit(X_train, y_train)
    preds = lgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(hold_out, preds))
    return preds, rmse

def model_cat(ts: pd.Series) -> Tuple[np.ndarray, float]:
    train, _, hold_out = split_series(ts)
    def create_lags_df(series, n=3):
        df = pd.DataFrame({'y': series})
        for i in range(1, n+1):
            df[f'lag_{i}'] = df['y'].shift(i)
        return df.dropna()
    
    df_lags = create_lags_df(ts.values, n=3)
    train_size = len(train)
    df_train = df_lags.iloc[:train_size-3]
    df_test = df_lags.iloc[train_size-3:train_size]
    X_train = df_train.drop('y', axis=1)
    y_train = df_train['y']
    X_test = df_test.drop('y', axis=1)
    
    cat_model = CatBoostRegressor(iterations=100, verbose=0, random_state=42)
    cat_model.fit(X_train, y_train)
    preds = cat_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(hold_out, preds))
    return preds, rmse

# Model Dictionary
modelos = {
    'Naive': model_naive,
    'ETS': model_ets,
    'ARIMA': model_arima,
    'SARIMA': model_sarima,
    'Prophet': model_prophet,
    'RandomForest': model_rf,
    'XGBoost': model_xgb,
    'LSTM': model_lstm,
    'Kalman': model_kalman,
    'LightGBM': model_lgb,
    'CatBoost': model_cat
}

# 4. Preprocessing
def preprocess_df(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Preprocess the input DataFrame with column mapping."""
    try:
        df = df.rename(columns=column_mapping)
    except Exception as e:
        st.error(f"Error renaming columns: {e}")
        logging.error(f"Error renaming columns: {e}")
        return None
    
    required_columns = ['YEAR', 'MONTH', 'QUANTITY', 'ITEM']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
        logging.error(f"Faltan columnas: {missing}")
        return None
    
    cols_to_convert = ['YEAR', 'MONTH', 'QUANTITY']
    if 'DAY' in df.columns:
        cols_to_convert.append('DAY')
    cols_present = [col for col in cols_to_convert if col in df.columns]
    
    df[cols_present] = df[cols_present].fillna(0).astype(int)
    if df['QUANTITY'].lt(0).any():
        st.warning("Se encontraron valores negativos en QUANTITY. Se convirtieron a 0.")
        df['QUANTITY'] = df['QUANTITY'].clip(lower=0)
    
    df['YEAR'] = df['YEAR'].fillna(method='ffill').astype(int)
    df['MONTH'] = df['MONTH'].fillna(method='ffill').astype(int)
    df['DAY'] = 1  # Set DAY to 1 for monthly forecast
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')
    df.dropna(subset=['DATE'], inplace=True)
    return df

# 5. Series Creation
def create_series(df: pd.DataFrame) -> Dict[Tuple, pd.Series]:
    """Create time series grouped by AREA and ITEM or just ITEM."""
    series_dict = {}
    has_area = 'AREA' in df.columns
    group_cols = ['AREA', 'ITEM'] if has_area else ['ITEM']
    
    for group_key, group_data in df.groupby(group_cols):
        group_data = group_data.set_index('DATE').sort_index()
        if len(group_data.index) < 3:
            continue
        
        date_diffs = group_data.index.to_series().diff().dropna()
        date_diffs_days = date_diffs.dt.days
        if not date_diffs_days.isin([28, 29, 30, 31]).all():
            st.warning(f"Series {group_key} tiene huecos en las fechas. Se rellenará con frecuencia mensual.")
        
        try:
            freq = pd.infer_freq(group_data.index)
        except:
            freq = None
        
        group_data = group_data.asfreq(freq if freq else 'MS', method='ffill')
        group_data['QUANTITY'] = pd.to_numeric(group_data['QUANTITY'], errors='coerce')
        group_data.dropna(subset=['QUANTITY'], inplace=True)
        
        if len(group_data) < 24:  # Minimum 24 months
            continue
            
        ts = group_data['QUANTITY']
        Q1, Q3 = ts.quantile(0.25), ts.quantile(0.75)
        IQR = Q3 - Q1
        ts_corrected = ts.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        series_dict[group_key if has_area else (group_key,)] = ts_corrected
    
    return series_dict

# 6. Cached Model Evaluation
@st.cache_data
def evaluate_models_cached(_ts: pd.Series, _columns: str) -> Dict:
    """Devuelve modelo ganador + métricas hold_out + pred future."""
    ts_hash = hashlib.md5(pd.util.hash_pandas_object(_ts).to_numpy().tobytes() + _columns.encode()).hexdigest()
    resultados_modelos = {}
    _, _, hold_out = split_series(_ts)
    
    for nombre_modelo, func_modelo in modelos.items():
        try:
            preds, rmse = func_modelo(_ts)
            resultados_modelos[nombre_modelo] = {'preds': preds, 'rmse': rmse}
        except Exception as e:
            resultados_modelos[nombre_modelo] = {'preds': None, 'rmse': np.inf}
    
    mejor_modelo = min(resultados_modelos, key=lambda m: resultados_modelos[m]['rmse'])
    full_preds_plus3 = modelos[mejor_modelo](_ts)[0]
    r2 = r2_score(hold_out, resultados_modelos[mejor_modelo]['preds'])
    metrics = compute_metrics(hold_out, resultados_modelos[mejor_modelo]['preds'])
    
    return {
        'best_model': mejor_modelo,
        'hold_actual': hold_out,
        'hold_pred': resultados_modelos[mejor_modelo]['preds'],
        'rmse': resultados_modelos[mejor_modelo]['rmse'],
        'r2': r2,
        'metrics': metrics,
        'future_pred': full_preds_plus3
    }

# 7. Main Processing
def process_forecasting(df: pd.DataFrame, force_rerun: bool, selected_models: List[str], column_mapping: Dict[str, str]) -> Tuple[Dict, Dict]:
    """Main forecasting process."""
    df_processed = preprocess_df(df, column_mapping)
    if df_processed is None:
        return None, None
    
    series_dict = create_series(df_processed)
    seleccion_modelos = {}
    global_r2_sum, series_ok = 0.0, 0
    
    active_modelos = {k: v for k, v in modelos.items() if k in selected_models}
    
    for key, ts in series_dict.items():
        if len(ts) < 24:
            st.warning(f"Series {key} demasiado corta para procesar.")
            continue
            
        if force_rerun:
            st.cache_data.clear()
        
        try:
            results = evaluate_models_cached(ts, ','.join(df_processed.columns))
        except ValueError:
            continue
        
        seleccion_modelos[key] = results
        global_r2_sum += results['r2']
        series_ok += 1
    
    global_metrics = {'Global_R2': global_r2_sum / series_ok if series_ok else np.nan}
    return seleccion_modelos, global_metrics

# 8. Visualization
def plot_series(ts: pd.Series, actuals: np.ndarray, preds: np.ndarray, future_preds: np.ndarray, 
                last_date: pd.Timestamp, series_key: Tuple, model_name: str) -> go.Figure:
    """Plot historical, validation, and future predictions."""
    historical_dates = ts.index
    validation_dates = ts.index[-3:]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=3, freq='MS')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_dates, y=ts.values, mode='lines', name='Historical', line=dict(color='#1e90ff')))
    fig.add_trace(go.Scatter(x=validation_dates, y=actuals, mode='markers', name='Actual (Hold-out)', marker=dict(color='#32cd32', size=10)))
    fig.add_trace(go.Scatter(x=validation_dates, y=preds, mode='markers', name='Predicted (Hold-out)', marker=dict(color='#ff4500', size=10, symbol='x')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', name='Future Forecast', line=dict(color='#ffa500', dash='dash')))
    
    title = f"Forecast for {series_key} using {model_name}"
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Quantity", template="plotly_dark",
                      paper_bgcolor='#2c2c2c', plot_bgcolor='#2c2c2c', font_color='#ffffff')
    return fig

def plot_model_comparison(metrics_dict: Dict, metric: str = 'RMSE') -> go.Figure:
    """Plot comparison of model metrics."""
    models = [m for m in metrics_dict.keys() if metrics_dict[m]['metrics'] is not None]
    values = [metrics_dict[m]['metrics'][metric] for m in models]
    
    fig = go.Figure(data=[go.Bar(x=models, y=values, name=metric, marker_color='#1e90ff')])
    fig.update_layout(title=f"Model Comparison by {metric}", xaxis_title="Model", yaxis_title=metric, template="plotly_dark",
                      paper_bgcolor='#2c2c2c', plot_bgcolor='#2c2c2c', font_color='#ffffff')
    return fig

def plot_metric_heatmap(seleccion_modelos: Dict) -> go.Figure:
    """Plot a heatmap of metrics across series."""
    if not seleccion_modelos:
        return None
    metrics_list = []
    series_names = []
    for key, info in seleccion_modelos.items():
        if info['metrics']:
            metrics_list.append([info['metrics'][m] for m in ['MAE', 'RMSE', 'R2', 'MAPE', 'SMAPE', 'Bias']])
            series_names.append(str(key))
    
    if not metrics_list:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=metrics_list,
        x=['MAE', 'RMSE', 'R2', 'MAPE', 'SMAPE', 'Bias'],
        y=series_names,
        colorscale='Viridis'
    ))
    fig.update_layout(title="Metric Heatmap Across Series", xaxis_title="Metric", yaxis_title="Series", template="plotly_dark",
                      paper_bgcolor='#2c2c2c', plot_bgcolor='#2c2c2c', font_color='#ffffff')
    return fig

def plot_error_distribution(seleccion_modelos: Dict) -> go.Figure:
    """Plot histogram of RMSE across series."""
    rmses = [info['rmse'] for info in seleccion_modelos.values() if info['rmse'] != np.inf]
    if not rmses:
        return None
    
    fig = go.Figure(data=[go.Histogram(x=rmses, nbinsx=20, marker_color='#1e90ff')])
    fig.update_layout(title="RMSE Distribution Across Series", xaxis_title="RMSE", yaxis_title="Count", template="plotly_dark",
                      paper_bgcolor='#2c2c2c', plot_bgcolor='#2c2c2c', font_color='#ffffff')
    return fig

def plot_actual_vs_predicted(seleccion_modelos: Dict) -> go.Figure:
    """Plot actual vs predicted values for hold-out period."""
    actuals = []
    preds = []
    for info in seleccion_modelos.values():
        if info['hold_actual'] is not None and info['hold_pred'] is not None:
            actuals.extend(info['hold_actual'])
            preds.extend(info['hold_pred'])
    
    if not actuals:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actuals, y=preds, mode='markers', name='Actual vs Predicted', marker=dict(color='#1e90ff')))
    fig.add_trace(go.Scatter(x=[min(actuals), max(actuals)], y=[min(actuals), max(actuals)], mode='lines', name='Ideal',
                             line=dict(color='#ff4500', dash='dash')))
    fig.update_layout(title="Actual vs Predicted (Hold-out Period)", xaxis_title="Actual", yaxis_title="Predicted",
                      template="plotly_dark", paper_bgcolor='#2c2c2c', plot_bgcolor='#2c2c2c', font_color='#ffffff')
    return fig

# 9. Main Streamlit App
def main():
    st.title("📈 Time Series Forecasting Pipeline")
    st.markdown("**Upload an Excel file and map columns to forecast quantities for the next 3 months.**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"], help="Excel with columns for YEAR, MONTH, QUANTITY, ITEM, and optional AREA, DAY")
        
        # Column mapping
        column_mapping = {}
        if uploaded_file:
            try:
                df_temp = pd.read_excel(uploaded_file, engine='openpyxl')
                columns = [''] + list(df_temp.columns)
                st.subheader("Map Columns")
                column_mapping['YEAR'] = st.selectbox("Select column for YEAR", columns, index=0)
                column_mapping['MONTH'] = st.selectbox("Select column for MONTH", columns, index=0)
                column_mapping['QUANTITY'] = st.selectbox("Select column for QUANTITY", columns, index=0)
                column_mapping['ITEM'] = st.selectbox("Select column for ITEM", columns, index=0)
                column_mapping['AREA'] = st.selectbox("Select column for AREA (optional)", columns, index=0)
                column_mapping['DAY'] = st.selectbox("Select column for DAY (optional)", columns, index=0)
                
                column_mapping = {k: v for k, v in column_mapping.items() if v}
                
                if uploaded_file and all(k in column_mapping for k in ['YEAR', 'MONTH', 'QUANTITY', 'ITEM']):
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.column_mapping = column_mapping
                else:
                    st.warning("Please map all required columns (YEAR, MONTH, QUANTITY, ITEM).")
            except Exception as e:
                st.error(f"Error reading Excel for column mapping: {e}")
                logging.error(f"Error reading Excel for column mapping: {e}")
        
        force_rerun = st.checkbox("Force Model Re-evaluation", value=False, help="Clear cache and rerun all models")
        selected_models = st.multiselect("Select Models", list(modelos.keys()), default=list(modelos.keys()))

    if 'uploaded_file' not in st.session_state or not st.session_state.uploaded_file or 'column_mapping' not in st.session_state:
        st.info("Please upload an Excel file and map all required columns to begin.")
        return

    # Read data
    try:
        df = pd.read_excel(st.session_state.uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        logging.error(f"Error reading Excel: {e}")
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Description", "📊 Metrics", "📈 Forecasts", "📉 Visualizations"])

    with tab1:
        st.subheader("Process Description")
        st.markdown("""
        This application forecasts quantities for the next 3 months using historical data. Here's how it works:

        1. **Data Input**: Upload an Excel file and map columns for `YEAR`, `MONTH`, `QUANTITY`, `ITEM`, and optionally `AREA` and `DAY`. Each `ITEM` (or `AREA`+`ITEM`) is treated as a separate time series. The `DAY` column is set to 1 for monthly forecasting.
        2. **Preprocessing**: Data is cleaned, missing values are filled, and dates are created from `YEAR`, `MONTH`, and `DAY=1`. Outliers are clipped using the IQR method. Negative quantities are set to 0.
        3. **Splitting**: Each series (min. 24 months) is split into train (up to -6 months), validation (-6 to -4 months), and hold-out (-3 to -1 months).
        4. **Validation**: Models are trained on train data and evaluated on the hold-out set using RMSE.
        5. **Model Selection**: The model with the lowest RMSE on the hold-out set is selected for each series.
        6. **Future Prediction**: The best model is trained on all historical data to predict the next 3 months.
        7. **Output**:
           - **Metrics**: MAE, RMSE, R2, MAPE, SMAPE, and Bias for the hold-out period, plus Global R2 across series.
           - **Forecasts**: Predictions for the next 3 months, downloadable as Excel.
           - **Visualizations**: Plots for historical data, hold-out results, model comparisons, metric heatmaps, and error distributions.
        8. **Caching**: Results are cached unless "Force Model Re-evaluation" is checked.

        Use the tabs to explore results and download forecasts as Excel.
        """)

    with tab2:
        st.subheader("Validation Metrics (Hold-out Period)")
        with st.spinner("Processing data..."):
            seleccion_modelos, global_metrics = process_forecasting(df, force_rerun, selected_models, st.session_state.column_mapping)
        
        if global_metrics:
            st.markdown("**Global Metrics Across All Series**")
            metrics_df = pd.DataFrame([global_metrics]).round(2)
            st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)
        
        if seleccion_modelos:
            st.markdown("**Per-Series Metrics**")
            results_list = []
            for key, info in seleccion_modelos.items():
                results_list.append({
                    'Series': str(key),
                    'Best Model': info['best_model'],
                    **{k: round(v, 2) for k, v in info['metrics'].items()}
                })
            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df.style.format({k: "{:.2f}" for k in ['MAE', 'RMSE', 'R2', 'MAPE', 'SMAPE', 'Bias']})
                         .background_gradient(cmap='Blues', subset=['MAE', 'RMSE', 'MAPE', 'SMAPE', 'Bias'])
                         .background_gradient(cmap='Greens', subset=['R2']),
                         use_container_width=True)
            
            excel_buffer = io.BytesIO()
            results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download Metrics (Excel)",
                data=excel_buffer,
                file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            with open('forecasting_errors.log', 'r') as f:
                log_content = f.read()
            if log_content:
                st.download_button(
                    label="📥 Download Error Log",
                    data=log_content,
                    file_name=f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    mime="text/plain"
                )
        else:
            st.warning("No valid series found for forecasting.")

    with tab3:
        st.subheader("Future Forecasts (Next 3 Months)")
        if seleccion_modelos:
            forecast_list = []
            for key, info in seleccion_modelos.items():
                last_date = info['hold_actual'].index[-1]
                future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=3, freq='MS')
                for i, (date, pred) in enumerate(zip(future_dates, info['future_pred'])):
                    forecast_list.append({
                        'Series': str(key),
                        'Model': info['best_model'],
                        'Date': date,
                        'Forecast': round(pred, 2)
                    })
            forecast_df = pd.DataFrame(forecast_list)
            st.dataframe(forecast_df, use_container_width=True)
            
            excel_buffer = io.BytesIO()
            forecast_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download Forecasts (Excel)",
                data=excel_buffer,
                file_name=f"forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No forecasts available.")

    with tab4:
        st.subheader("Visualizations")
        if seleccion_modelos:
            series_keys = list(seleccion_modelos.keys())
            selected_series = st.selectbox("Select Series to Visualize", series_keys, format_func=lambda x: str(x))
            metric = st.selectbox("Select Metric for Comparison", ['MAE', 'RMSE', 'R2', 'MAPE', 'SMAPE', 'Bias'])
            
            series_data = create_series(preprocess_df(df, st.session_state.column_mapping))[selected_series]
            info = seleccion_modelos[selected_series]
            
            st.markdown("**Historical and Forecast Plot**")
            fig_series = plot_series(series_data, info['hold_actual'], info['hold_pred'], info['future_pred'], 
                                    info['hold_actual'].index[-1], selected_series, info['best_model'])
            st.plotly_chart(fig_series, use_container_width=True)
            
            st.markdown("**Model Comparison by Metric**")
            ts_hash = hashlib.md5(pd.util.hash_pandas_object(series_data).to_numpy().tobytes() + ','.join(df.columns).encode()).hexdigest()
            results = evaluate_models_cached(series_data, ','.join(df.columns))
            fig_comparison = plot_model_comparison(results['metrics'], metric)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.markdown("**Metric Heatmap Across Series**")
            fig_heatmap = plot_metric_heatmap(seleccion_modelos)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("No data available for heatmap.")
            
            st.markdown("**RMSE Distribution Across Series**")
            fig_error_dist = plot_error_distribution(seleccion_modelos)
            if fig_error_dist:
                st.plotly_chart(fig_error_dist, use_container_width=True)
            else:
                st.warning("No data available for error distribution.")
            
            st.markdown("**Actual vs Predicted (Hold-out Period)**")
            fig_scatter = plot_actual_vs_predicted(seleccion_modelos)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("No data available for actual vs predicted scatter.")
        else:
            st.warning("No series available for visualization.")

if __name__ == "__main__":
    main()