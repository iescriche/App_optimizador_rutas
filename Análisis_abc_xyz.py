import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime
import io
import json
import os
from typing import Dict, Tuple, Optional
import uuid

# Set Altair data transformer to default with no max rows
alt.data_transformers.enable('default', max_rows=None)

# File for storing column mappings
COLMAP_FILE = "col_map_abc_xyz.json"

def load_col_map() -> dict:
    """Load column mappings from a JSON file."""
    if os.path.exists(COLMAP_FILE):
        try:
            with open(COLMAP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_col_map(col_map: dict):
    """Save column mappings to a JSON file."""
    with open(COLMAP_FILE, "w", encoding="utf-8") as f:
        json.dump(col_map, f, ensure_ascii=False, indent=2)

def clean_string(s: str) -> str:
    """Clean string by removing invisible characters and stripping whitespace."""
    if pd.isna(s):
        return ""
    s = str(s)
    for ch in ['\xa0', '\u00A0', '\u200B', '\u200C', '\u200D', '\uFEFF']:
        s = s.replace(ch, '')
    return s.strip()

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame column names by stripping whitespace and removing special characters."""
    df.columns = [clean_string(col).replace(' ', '_') for col in df.columns]
    return df

def analyze_abc_xyz(
    df_pv: pd.DataFrame,
    df_maestro: pd.DataFrame,
    ref_col: str,
    qty_col: str,
    order_col: str,
    price_col: Optional[str],
    abc_thresholds: Tuple[float, float],
    xyz_thresholds: Tuple[float, float],
    by_quantity: bool = True,
    by_value: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform ABC-XYZ analysis based on quantity and/or monetary value.
    
    Parameters:
    - df_pv: DataFrame with sales data.
    - df_maestro: DataFrame with master data (references and descriptions).
    - ref_col, qty_col, order_col, price_col: Column names for reference, quantity, order, and price.
    - abc_thresholds: Tuple of (A, B) thresholds for ABC classification (e.g., (80, 95)).
    - xyz_thresholds: Tuple of (X, Y) thresholds for XYZ classification based on CV (e.g., (0.5, 1.0)).
    - by_quantity: Perform analysis by quantity if True.
    - by_value: Perform analysis by monetary value if True and price_col is provided.
    
    Returns:
    - abc_df_qty: ABC analysis by quantity.
    - abc_df_value: ABC analysis by monetary value (empty if by_value=False or no price_col).
    - xyz_df: XYZ analysis based on coefficient of variation.
    - contingency: Contingency table (XYZ vs ABC).
    - contingency_pct: Contingency table in percentages.
    - products_df: Detailed product data with ABC-XYZ classifications.
    
    Raises:
    - ValueError: If required columns are missing in df_pv or df_maestro.
    """
    # Validate required columns in df_pv
    required_cols = [ref_col, qty_col, order_col]
    if price_col and by_value:
        required_cols.append(price_col)
    missing_cols = [col for col in required_cols if col not in df_pv.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en Pedidos.xlsx: {', '.join(missing_cols)}")

    # Validate maestro columns if provided
    if not df_maestro.empty and 'maestro_ref' in st.session_state.col_map and 'maestro_desc' in st.session_state.col_map:
        maestro_required = [st.session_state.col_map['maestro_ref'], st.session_state.col_map['maestro_desc']]
        if 'maestro_price' in st.session_state.col_map and st.session_state.col_map['maestro_price'] != 'Ninguna':
            maestro_required.append(st.session_state.col_map['maestro_price'])
        missing_maestro_cols = [col for col in maestro_required if col not in df_maestro.columns]
        if missing_maestro_cols:
            raise ValueError(f"Columnas faltantes en Maestro.xlsx: {', '.join(missing_maestro_cols)}")

    # Clean object-type columns, excluding date column
    date_col = st.session_state.col_map.get('pv_date')
    for col in df_pv.select_dtypes(include="object").columns:
        if col != date_col:
            df_pv[col] = df_pv[col].apply(clean_string)

    # Convert quantity and price columns to numeric types
    df_pv[qty_col] = pd.to_numeric(df_pv[qty_col], errors='coerce')
    if price_col and by_value:
        df_pv[price_col] = pd.to_numeric(df_pv[price_col], errors='coerce')

    # Handle NaN values introduced during conversion
    df_pv = df_pv.dropna(subset=[qty_col])
    if price_col and by_value:
        df_pv = df_pv.dropna(subset=[price_col])

    # Initialize output DataFrames
    abc_df_qty = pd.DataFrame()
    abc_df_value = pd.DataFrame()
    xyz_df = pd.DataFrame()
    contingency = pd.DataFrame()
    contingency_pct = pd.DataFrame()
    products_df = pd.DataFrame()

    # Description mapping from maestro
    desc_dict = {}
    if not df_maestro.empty and 'maestro_ref' in st.session_state.col_map and 'maestro_desc' in st.session_state.col_map:
        df_maestro = df_maestro.dropna(subset=[st.session_state.col_map['maestro_ref']])
        desc_dict = dict(zip(
            df_maestro[st.session_state.col_map['maestro_ref']].apply(clean_string),
            df_maestro[st.session_state.col_map['maestro_desc']].apply(clean_string)
        ))

    # ABC Analysis by Quantity
    if by_quantity:
        total_sold = df_pv.groupby(ref_col)[qty_col].sum().reset_index()
        total_sold = total_sold.sort_values(by=qty_col, ascending=False)
        total_sold['Porcentaje_Acumulado'] = total_sold[qty_col].cumsum() / total_sold[qty_col].sum() * 100
        a_threshold, b_threshold = abc_thresholds
        total_sold['ABC'] = np.where(total_sold['Porcentaje_Acumulado'] / 100 <= a_threshold / 100, 'A',
                                     np.where(total_sold['Porcentaje_Acumulado'] / 100 <= b_threshold / 100, 'B', 'C'))
        abc_df_qty = total_sold[[ref_col, qty_col, 'Porcentaje_Acumulado', 'ABC']].rename(columns={qty_col: 'Demanda_Total'})
        if desc_dict:
            abc_df_qty['Descripción'] = abc_df_qty[ref_col].map(desc_dict)

    # ABC Analysis by Monetary Value
    if by_value and price_col:
        df_pv['Value'] = df_pv[qty_col] * df_pv[price_col]
        total_value = df_pv.groupby(ref_col)['Value'].sum().reset_index()
        total_value = total_value.sort_values(by='Value', ascending=False)
        total_value['Porcentaje_Acumulado'] = total_value['Value'].cumsum() / total_value['Value'].sum() * 100
        total_value['ABC'] = np.where(total_value['Porcentaje_Acumulado'] / 100 <= a_threshold / 100, 'A',
                                      np.where(total_value['Porcentaje_Acumulado'] / 100 <= b_threshold / 100, 'B', 'C'))
        abc_df_value = total_value[[ref_col, 'Value', 'Porcentaje_Acumulado', 'ABC']].rename(columns={'Value': 'Valor_Total'})
        if desc_dict:
            abc_df_value['Descripción'] = abc_df_value[ref_col].map(desc_dict)

    # XYZ Analysis based on Coefficient of Variation
    demand_stats = df_pv.groupby(ref_col)[qty_col].agg(['mean', 'std']).reset_index()
    demand_stats['CV'] = demand_stats['std'] / demand_stats['mean'].replace(0, np.nan)
    demand_stats['CV'] = demand_stats['CV'].fillna(0)
    x_threshold, y_threshold = xyz_thresholds
    demand_stats['XYZ'] = np.where(demand_stats['CV'] <= x_threshold, 'X',
                                   np.where(demand_stats['CV'] <= y_threshold, 'Y', 'Z'))
    xyz_df = demand_stats[[ref_col, 'mean', 'std', 'CV', 'XYZ']].rename(columns={'mean': 'Demanda_Media', 'std': 'Desviación_Estándar'})
    if desc_dict:
        xyz_df['Descripción'] = xyz_df[ref_col].map(desc_dict)

    # Combine ABC and XYZ
    if by_quantity:
        products_df = total_sold.merge(demand_stats[[ref_col, 'CV', 'XYZ']], on=ref_col)
        products_df = products_df[[ref_col, qty_col, 'ABC', 'CV', 'XYZ']].rename(columns={qty_col: 'Demanda_Total'})
        if by_value and price_col:
            products_df = products_df.merge(total_value[[ref_col, 'Value', 'ABC']], on=ref_col, suffixes=('_Qty', '_Value'))
            products_df = products_df.rename(columns={'Value': 'Valor_Total', 'ABC_Value': 'ABC_Valor', 'ABC_Qty': 'ABC_Cantidad'})
        if desc_dict:
            products_df['Descripción'] = products_df[ref_col].map(desc_dict)
        
        # Contingency Table
        if by_value and price_col:
            contingency_qty = pd.crosstab(products_df['XYZ'], products_df['ABC_Cantidad'])
            contingency_value = pd.crosstab(products_df['XYZ'], products_df['ABC_Valor'])
            contingency = pd.concat({'Cantidad': contingency_qty, 'Valor': contingency_value}, axis=1)
            contingency_pct_qty = (contingency_qty / contingency_qty.sum().sum() * 100).round(2)
            contingency_pct_value = (contingency_value / contingency_value.sum().sum() * 100).round(2)
            contingency_pct = pd.concat({'Cantidad': contingency_pct_qty, 'Valor': contingency_pct_value}, axis=1)
        else:
            contingency = pd.crosstab(products_df['XYZ'], products_df['ABC'])
            contingency_pct = (contingency / contingency.sum().sum() * 100).round(2)

    return abc_df_qty, abc_df_value, xyz_df, contingency, contingency_pct, products_df

def download_chart(chart: alt.Chart, filename: str) -> tuple[io.BytesIO, str]:
    """Returns (buffer, filename) with the chart as interactive HTML."""
    buf = io.BytesIO()
    out_name = filename.rsplit('.', 1)[0] + '.html'
    buf.write(chart.to_html().encode())
    buf.seek(0)
    return buf, out_name

def plot_abc_distribution(df: pd.DataFrame, value_col: str, title: str) -> alt.Chart:
    """Plot bar chart of ABC classification distribution with percentages."""
    count_df = df['ABC'].value_counts().reset_index()
    count_df.columns = ['ABC', 'Count']
    total = count_df['Count'].sum()
    count_df['Percentage'] = (count_df['Count'] / total * 100).round(2)
    
    chart = alt.Chart(count_df).mark_bar().encode(
        x=alt.X('ABC:N', title='Categoría ABC', sort=['A', 'B', 'C']),
        y=alt.Y('Count:Q', title='Número de Productos'),
        color=alt.Color('ABC:N', scale=alt.Scale(domain=['A', 'B', 'C'], range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(orient='right')),
        tooltip=['ABC', 'Count', alt.Tooltip('Percentage:Q', title='Porcentaje', format='.2f')]
    ).properties(
        title=title,
        width=350,
        height=250
    )
    
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, fontSize=12).encode(
        text=alt.Text('Percentage:Q', format='.1f')
    )
    
    combined_chart = (chart + text).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def plot_xyz_distribution(df: pd.DataFrame, title: str) -> alt.Chart:
    """Plot bar chart of XYZ classification distribution with percentages."""
    count_df = df['XYZ'].value_counts().reset_index()
    count_df.columns = ['XYZ', 'Count']
    total = count_df['Count'].sum()
    count_df['Percentage'] = (count_df['Count'] / total * 100).round(2)
    
    chart = alt.Chart(count_df).mark_bar().encode(
        x=alt.X('XYZ:N', title='Categoría XYZ', sort=['X', 'Y', 'Z']),
        y=alt.Y('Count:Q', title='Número de Productos'),
        color=alt.Color('XYZ:N', scale=alt.Scale(domain=['X', 'Y', 'Z'], range=['#9467bd', '#8c564b', '#e377c2']),
                        legend=alt.Legend(orient='right')),
        tooltip=['XYZ', 'Count', alt.Tooltip('Percentage:Q', title='Porcentaje', format='.2f')]
    ).properties(
        title=title,
        width=350,
        height=250
    )
    
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, fontSize=12).encode(
        text=alt.Text('Percentage:Q', format='.1f')
    )
    
    combined_chart = (chart + text).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def plot_abc_pie(df: pd.DataFrame, title: str) -> alt.Chart:
    """Plot pie chart of ABC classification distribution with percentages."""
    count_df = df['ABC'].value_counts().reset_index()
    count_df.columns = ['ABC', 'Count']
    total = count_df['Count'].sum()
    count_df['Percentage'] = (count_df['Count'] / total * 100).round(2)
    
    chart = alt.Chart(count_df).mark_arc().encode(
        theta=alt.Theta('Count:Q', stack=True),
        color=alt.Color('ABC:N', scale=alt.Scale(domain=['A', 'B', 'C'], range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(orient='right')),
        tooltip=['ABC', 'Count', alt.Tooltip('Percentage:Q', title='Porcentaje', format='.2f')]
    ).properties(
        title=title,
        width=250,
        height=250
    )
    
    text = chart.mark_text(radius=120, size=12).encode(
        text=alt.Text('Percentage:Q', format='.1f'),
        color=alt.value('black')
    )
    
    combined_chart = (chart + text).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def plot_xyz_pie(df: pd.DataFrame, title: str) -> alt.Chart:
    """Plot pie chart of XYZ classification distribution with percentages."""
    count_df = df['XYZ'].value_counts().reset_index()
    count_df.columns = ['XYZ', 'Count']
    total = count_df['Count'].sum()
    count_df['Percentage'] = (count_df['Count'] / total * 100).round(2)
    
    chart = alt.Chart(count_df).mark_arc().encode(
        theta=alt.Theta('Count:Q', stack=True),
        color=alt.Color('XYZ:N', scale=alt.Scale(domain=['X', 'Y', 'Z'], range=['#9467bd', '#8c564b', '#e377c2']),
                        legend=alt.Legend(orient='right')),
        tooltip=['XYZ', 'Count', alt.Tooltip('Percentage:Q', title='Porcentaje', format='.2f')]
    ).properties(
        title=title,
        width=250,
        height=250
    )
    
    text = chart.mark_text(radius=120, size=12).encode(
        text=alt.Text('Percentage:Q', format='.1f'),
        color=alt.value('black')
    )
    
    combined_chart = (chart + text).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def plot_contingency_heatmap(contingency_pct: pd.DataFrame, title: str) -> alt.Chart:
    """Plot heatmap of ABC-XYZ contingency table with percentages."""
    if isinstance(contingency_pct.columns, pd.MultiIndex):
        contingency_pct = contingency_pct['Cantidad']  # Default to quantity if multi-index
    contingency_melt = contingency_pct.reset_index().melt(id_vars='XYZ', var_name='ABC', value_name='Porcentaje')
    
    chart = alt.Chart(contingency_melt).mark_rect().encode(
        x=alt.X('ABC:N', title='Categoría ABC', sort=['A', 'B', 'C']),
        y=alt.Y('XYZ:N', title='Categoría XYZ', sort=['X', 'Y', 'Z']),
        color=alt.Color('Porcentaje:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['ABC', 'XYZ', alt.Tooltip('Porcentaje:Q', format='.2f')]
    ).properties(
        title=title,
        width=350,
        height=250
    )
    
    text = chart.mark_text(baseline='middle', fontSize=12).encode(
        text=alt.Text('Porcentaje:Q', format='.1f'),
        color=alt.condition(
            alt.datum.Porcentaje > 10,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    combined_chart = (chart + text).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def plot_pareto_curve(df: pd.DataFrame, value_col: str, title: str, top_n=200) -> alt.Chart:
    """Plot Pareto curve for cumulative demand, limited to top N items with 'Others' category, with descending percentage."""
    df = df.nlargest(top_n, value_col).copy()
    total = df[value_col].sum()
    df['Porcentaje_Acumulado'] = (total - df[value_col].cumsum()) / total * 100  # Reverse cumulative percentage
    resto = df['Porcentaje_Acumulado'].iloc[-1]
    if resto > 0:
        df.loc['Otros'] = {value_col: 0, 'Porcentaje_Acumulado': 0}
    
    chart = alt.Chart(df.reset_index()).mark_area(
        line={'color': '#1f77b4'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='#1f77b4', offset=0),
                   alt.GradientStop(color='#4ca8d9', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('index:O', title='Productos (Ordenados por Demanda)', axis=alt.Axis(labels=False)),
        y=alt.Y('Porcentaje_Acumulado:Q', title='Porcentaje Acumulado (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=['index', value_col, alt.Tooltip('Porcentaje_Acumulado:Q', format='.2f')]
    ).properties(
        title=title,
        width=450,
        height=300
    )
    
    combined_chart = chart.configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def plot_demand_cv_scatter(products_df: pd.DataFrame, demand_col: str, title: str) -> alt.Chart:
    """Plot scatter of demand vs coefficient of variation."""
    chart = alt.Chart(products_df).mark_circle(size=80, opacity=0.6).encode(
        x=alt.X(f'{demand_col}:Q', title='Demanda Total',
                scale=alt.Scale(zero=False, type='log')),
        y=alt.Y('CV:Q', title='Coeficiente de Variación',
                scale=alt.Scale(zero=False)),
        color=alt.Color('XYZ:N',
                        scale=alt.Scale(domain=['X', 'Y', 'Z'],
                                        range=['#9467bd', '#8c564b', '#e377c2']),
                        legend=alt.Legend(orient='right')),
        size=alt.Size('Demanda_Total:Q', scale=alt.Scale(range=[50, 200])),
        tooltip=[st.session_state.col_map['pv_ref'],
                 demand_col, 'CV', 'ABC', 'XYZ']
    ).properties(
        title=title,
        width=450,
        height=350
    )
    
    combined_chart = chart.configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def plot_abc_xyz_stacked_bar(products_df: pd.DataFrame, title: str) -> alt.Chart:
    """Plot stacked bar chart of ABC-XYZ combinations with percentages."""
    combo_counts = products_df.groupby(['ABC', 'XYZ']).size().reset_index(name='Count')
    total = combo_counts['Count'].sum()
    combo_counts['Percentage'] = (combo_counts['Count'] / total * 100).round(2)
    
    chart = alt.Chart(combo_counts).mark_bar().encode(
        x=alt.X('ABC:N', title='Categoría ABC', sort=['A', 'B', 'C']),
        y=alt.Y('Count:Q', title='Número de Productos'),
        color=alt.Color('XYZ:N', scale=alt.Scale(domain=['X', 'Y', 'Z'], range=['#9467bd', '#8c564b', '#e377c2']),
                        legend=alt.Legend(orient='right')),
        tooltip=['ABC', 'XYZ', 'Count', alt.Tooltip('Percentage:Q', title='Porcentaje', format='.2f')]
    ).properties(
        title=title,
        width=350,
        height=250
    )
    
    text = chart.mark_text(align='center', baseline='middle', dy=-5, fontSize=12).encode(
        text=alt.Text('Count:Q'),
        color=alt.value('black')
    )
    
    combined_chart = (chart + text).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def plot_demand_box_plot(products_df: pd.DataFrame, demand_col: str, title: str) -> alt.Chart:
    """Plot box plot of demand distribution by ABC category."""
    if products_df.empty:
        st.warning("No data available for box plot. Check data merging and column mappings.")
        return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data"))
    
    # Debug: Show DataFrame columns
    st.write(f"Columns in products_df: {list(products_df.columns)}")
    
    if demand_col not in products_df.columns or 'ABC' not in products_df.columns:
        st.error(f"Required columns ({demand_col}, ABC) not found in DataFrame. Available columns: {list(products_df.columns)}")
        st.write("Sample of products_df:", products_df.head())
        return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("Invalid Data"))
    
    chart = alt.Chart(products_df).mark_boxplot(size=50).encode(
        x=alt.X('ABC:N', title='Categoría ABC', sort=['A', 'B', 'C']),
        y=alt.Y(demand_col + ':Q', title='Demanda Total', scale=alt.Scale(zero=False)),
        color=alt.Color('ABC:N', scale=alt.Scale(domain=['A', 'B', 'C'], range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(orient='right')),
        tooltip=['ABC', alt.Tooltip(demand_col + ':Q', title='Demanda Total', format='.2f')]
    ).properties(
        title=title,
        width=350,
        height=250
    )
    
    combined_chart = chart.configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='middle'
    )
    
    return combined_chart

def download_excel(df: pd.DataFrame, filename: str) -> io.BytesIO:
    """Generate Excel file for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    output.seek(0)
    return output

def main():
    st.set_page_config(page_title="ABC-XYZ Inventory Analysis", layout="wide")
    
    # Initialize session state
    if "col_map" not in st.session_state:
        st.session_state.col_map = load_col_map()
        st.session_state.form_submitted = bool(st.session_state.col_map)
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False

    st.title("📊 Análisis ABC-XYZ de Inventario")
    st.markdown("Clasifica productos según su demanda (cantidad o valor) y variabilidad para optimizar la gestión de inventario.")

    # File upload
    with st.sidebar:
        st.header("📂 Carga de Datos")
        uploaded_pv = st.file_uploader("Pedidos.xlsx", type="xlsx")
        uploaded_maestro = st.file_uploader("Maestro.xlsx", type="xlsx")

    if not uploaded_pv:
        st.error("Sube el archivo Pedidos.xlsx para continuar.")
        return

    # Load data and clean column names
    df_pv = pd.read_excel(uploaded_pv)
    df_pv = clean_column_names(df_pv)
    df_maestro = pd.read_excel(uploaded_maestro) if uploaded_maestro else pd.DataFrame()
    if not df_maestro.empty:
        df_maestro = clean_column_names(df_maestro)

    # Convert date column to datetime, normalize, and extract year
    if ("pv_date" in st.session_state.col_map and
        st.session_state.col_map["pv_date"] in df_pv.columns):
        date_col = st.session_state.col_map["pv_date"]
        df_pv[date_col] = pd.to_datetime(df_pv[date_col], dayfirst=True, errors="coerce").dt.normalize()
        df_pv["Year"] = df_pv[date_col].dt.year

    # Data validation and numeric conversion for price columns
    if (not df_maestro.empty and
        'maestro_price' in st.session_state.col_map and
        st.session_state.col_map['maestro_price'] != 'Ninguna' and
        st.session_state.col_map['maestro_price'] in df_maestro.columns):
        price_col_maestro = st.session_state.col_map['maestro_price']
        df_maestro[price_col_maestro] = pd.to_numeric(
            df_maestro[price_col_maestro], errors='coerce'
        )

    if ('pv_price' in st.session_state.col_map and
        st.session_state.col_map['pv_price'] != 'Ninguna' and
        st.session_state.col_map['pv_price'] in df_pv.columns):
        price_col_pv = st.session_state.col_map['pv_price']
        df_pv[price_col_pv] = pd.to_numeric(df_pv[price_col_pv], errors='coerce')

    # Display data preview
    st.subheader("Vista Previa de los Datos")
    st.write("**Pedidos.xlsx** (primeras 5 filas):")
    st.dataframe(df_pv.head(), use_container_width=True)
    if not df_maestro.empty:
        st.write("**Maestro.xlsx** (primeras 5 filas):")
        st.dataframe(df_maestro.head(), use_container_width=True)

    # Column mapping form
    if not st.session_state.form_submitted:
        with st.sidebar.form("col_mapping"):
            st.write("### Mapeo de Columnas (Pedidos)")
            pv_columns = list(df_pv.columns)
            st.session_state.col_map["pv_ref"] = st.selectbox(
                "Referencia", pv_columns, 
                index=pv_columns.index('Referencia') if 'Referencia' in pv_columns else 0, 
                key="pv_ref"
            )
            st.session_state.col_map["pv_qty"] = st.selectbox(
                "Cantidad", pv_columns, 
                index=pv_columns.index('Cantidad') if 'Cantidad' in pv_columns else 0, 
                key="pv_qty"
            )
            st.session_state.col_map["pv_order"] = st.selectbox(
                "Nº Pedido", pv_columns, 
                index=pv_columns.index('Pedido') if 'Pedido' in pv_columns else 0, 
                key="pv_order"
            )
            date_candidates = [c for c in pv_columns if 'fecha' in c.lower() or 'date' in c.lower()]
            st.session_state.col_map["pv_date"] = st.selectbox(
                "Fecha del pedido", pv_columns,
                index=pv_columns.index(date_candidates[0]) if date_candidates else 0, 
                key="pv_date"
            )
            price_candidates = [col for col in pv_columns if 'precio' in col.lower() or 'price' in col.lower() or 'importe' in col.lower()]
            st.session_state.col_map["pv_price"] = st.selectbox(
                "Precio Unitario (Opcional)", ['Ninguna'] + pv_columns, 
                index=pv_columns.index(price_candidates[0]) + 1 if price_candidates else 0, 
                key="pv_price"
            )
            if not df_maestro.empty:
                st.write("### Mapeo de Columnas (Maestro)")
                maestro_columns = list(df_maestro.columns)
                st.session_state.col_map["maestro_ref"] = st.selectbox(
                    "Referencia (Maestro)", maestro_columns, 
                    index=maestro_columns.index('Referencia') if 'Referencia' in maestro_columns else 0, 
                    key="maestro_ref"
                )
                st.session_state.col_map["maestro_desc"] = st.selectbox(
                    "Descripción", maestro_columns, 
                    index=maestro_columns.index('Descripción') if 'Descripción' in maestro_columns else 0, 
                    key="maestro_desc"
                )
                maestro_price_candidates = [col for col in maestro_columns if 'precio' in col.lower() or 'price' in col.lower()]
                st.session_state.col_map["maestro_price"] = st.selectbox(
                    "Precio Unitario (Opcional)", ['Ninguna'] + maestro_columns,
                    index=maestro_columns.index(maestro_price_candidates[0]) + 1 if maestro_price_candidates else 0,
                    key="maestro_price"
                )
            
            if st.form_submit_button("Confirmar Mapeo"):
                if (st.session_state.col_map["pv_ref"] not in df_pv.columns or
                    st.session_state.col_map["pv_qty"] not in df_pv.columns or
                    st.session_state.col_map["pv_order"] not in df_pv.columns or
                    st.session_state.col_map["pv_date"] not in df_pv.columns):
                    st.error("Una o más columnas seleccionadas para Pedidos.xlsx no son válidas. Por favor, verifica las selecciones.")
                    return
                if (not df_maestro.empty and
                    (st.session_state.col_map["maestro_ref"] not in df_maestro.columns or
                     st.session_state.col_map["maestro_desc"] not in df_maestro.columns or
                     (st.session_state.col_map["maestro_price"] != 'Ninguna' and
                      st.session_state.col_map["maestro_price"] not in df_maestro.columns))):
                    st.error("Una o más columnas seleccionadas para Maestro.xlsx no son válidas. Por favor, verifica las selecciones.")
                    return
                st.session_state.form_submitted = True
                save_col_map(st.session_state.col_map)
                st.success("Mapeo confirmado.")
    else:
        st.sidebar.success("✅ Mapeo cargado")

        # Analysis configuration
        st.sidebar.header("⚙️ Configuración del Análisis")
        
        # Year range filtering
        if ("pv_date" in st.session_state.col_map and
            st.session_state.col_map["pv_date"] in df_pv.columns):
            date_col = st.session_state.col_map["pv_date"]
            years = sorted(df_pv["Year"].dropna().unique())
            if years:
                min_y, max_y = int(years[0]), int(years[-1])
                year_range = st.sidebar.slider(
                    "Rango de años",
                    min_value=min_y,
                    max_value=max_y,
                    value=(min_y, max_y),
                    step=1
                )
                mask = df_pv["Year"].between(*year_range)
                df_pv = df_pv.loc[mask].copy()
                st.sidebar.info(f"⚡ Filtrado: {mask.sum():,} registros entre {year_range[0]} y {year_range[1]}")
            else:
                st.sidebar.warning("⚠️ No hay años válidos en la columna de fecha.")

        a_threshold = st.sidebar.number_input("Umbral A (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        b_threshold = st.sidebar.number_input("Umbral B (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
        x_threshold = st.sidebar.number_input("Umbral X (CV)", min_value=0.0, value=0.5, step=0.1, help="CV bajo indica demanda estable")
        y_threshold = st.sidebar.number_input("Umbral Y (CV)", min_value=0.0, value=1.0, step=0.1, help="CV medio indica demanda moderada")
        
        # Validate thresholds
        if a_threshold >= b_threshold:
            st.sidebar.error("El umbral A debe ser menor que el umbral B.")
            return
        if x_threshold >= y_threshold:
            st.sidebar.error("El umbral X debe ser menor que el umbral Y.")
            return

        # Determine price column
        price_col = None
        if st.session_state.col_map.get("pv_price") != 'Ninguna' and st.session_state.col_map["pv_price"] in df_pv.columns:
            price_col = st.session_state.col_map["pv_price"]
        elif (not df_maestro.empty and
              st.session_state.col_map.get("maestro_price") != 'Ninguna' and
              st.session_state.col_map["maestro_price"] in df_maestro.columns):
            df_pv = df_pv.merge(
                df_maestro[[st.session_state.col_map["maestro_ref"], st.session_state.col_map["maestro_price"]]],
                left_on=st.session_state.col_map["pv_ref"],
                right_on=st.session_state.col_map["maestro_ref"],
                how='left'
            )
            price_col = st.session_state.col_map["maestro_price"]
        
        # Analysis options
        by_quantity = st.sidebar.checkbox("Analizar por Cantidad", value=True)
        by_value = st.sidebar.checkbox("Analizar por Valor Monetario", value=bool(price_col))
        if by_value and not price_col:
            st.sidebar.warning("No se proporcionó columna de precio. El análisis por valor se omitirá.")
            by_value = False

        if not (by_quantity or by_value):
            st.error("Selecciona al menos un tipo de análisis (Cantidad o Valor).")
            return

        if st.sidebar.button("Calcular Análisis"):
            with st.spinner("Procesando análisis ABC-XYZ..."):
                try:
                    abc_df_qty, abc_df_value, xyz_df, contingency, contingency_pct, products_df = analyze_abc_xyz(
                        df_pv,
                        df_maestro,
                        st.session_state.col_map["pv_ref"],
                        st.session_state.col_map["pv_qty"],
                        st.session_state.col_map["pv_order"],
                        price_col,
                        abc_thresholds=(a_threshold, b_threshold),
                        xyz_thresholds=(x_threshold, y_threshold),
                        by_quantity=by_quantity,
                        by_value=by_value
                    )
                except ValueError as e:
                    st.error(str(e))
                    return

                # Display Results
                header_text = f"📈 Resultados del Análisis ABC-XYZ ({year_range[0]}–{year_range[1]})" if years else "📈 Resultados del Análisis ABC-XYZ"
                st.header(header_text)

                # Create tabs for different analysis sections
                tab1, tab2, tab3, tab4 = st.tabs(["ABC por Cantidad", "ABC por Valor", "XYZ", "Consolidado"])

                with tab1:
                    if by_quantity and not abc_df_qty.empty:
                        st.subheader("Análisis ABC por Cantidad")
                        st.dataframe(abc_df_qty.round(2), use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            chart = plot_abc_distribution(abc_df_qty, 'Demanda_Total', 'Distribución ABC por Cantidad')
                            st.altair_chart(chart, use_container_width=True)
                            buf, fname = download_chart(chart, "abc_distribution_qty.png")
                            st.download_button(
                                label="📥 Descargar Gráfico (HTML)",
                                data=buf,
                                file_name=fname,
                                mime="text/html"
                            )
                        with col2:
                            chart = plot_abc_pie(abc_df_qty, 'Proporción ABC por Cantidad')
                            st.altair_chart(chart, use_container_width=True)
                            buf, fname = download_chart(chart, "abc_pie_qty.png")
                            st.download_button(
                                label="📥 Descargar Gráfico (HTML)",
                                data=buf,
                                file_name=fname,
                                mime="text/html"
                            )
                        
                        chart = plot_pareto_curve(abc_df_qty.reset_index(), 'Demanda_Total', 'Curva de Pareto por Cantidad')
                        st.altair_chart(chart, use_container_width=True)
                        buf, fname = download_chart(chart, "pareto_qty.png")
                        st.download_button(
                            label="📥 Descargar Gráfico (HTML)",
                            data=buf,
                            file_name=fname,
                            mime="text/html"
                        )
                        
                        # Debug merge for box plot
                        st.write("**Debugging Box Plot Data (Quantity)**")
                        st.write(f"abc_df_qty shape: {abc_df_qty.shape}")
                        st.write(f"abc_df_qty columns: {list(abc_df_qty.columns)}")
                        st.write(f"products_df shape: {products_df.shape}")
                        st.write(f"products_df columns: {list(products_df.columns)}")
                        
                        # Use abc_df_qty directly since it has Demanda_Total and ABC
                        merged_df = abc_df_qty[[st.session_state.col_map['pv_ref'], 'Demanda_Total', 'ABC']].copy()
                        st.write(f"Merged DataFrame shape: {merged_df.shape}")
                        st.write(f"Merged DataFrame columns: {list(merged_df.columns)}")
                        st.write("Merged DataFrame sample:", merged_df.head())
                        
                        chart = plot_demand_box_plot(merged_df, 'Demanda_Total', 'Distribución de Demanda por Categoría ABC')
                        st.altair_chart(chart, use_container_width=True)
                        buf, fname = download_chart(chart, "box_plot_qty.png")
                        st.download_button(
                            label="📥 Descargar Gráfico (HTML)",
                            data=buf,
                            file_name=fname,
                            mime="text/html"
                        )

                with tab2:
                    if by_value and not abc_df_value.empty:
                        st.subheader("Análisis ABC por Valor Monetario")
                        st.dataframe(abc_df_value.round(2), use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            chart = plot_abc_distribution(abc_df_value, 'Valor_Total', 'Distribución ABC por Valor')
                            st.altair_chart(chart, use_container_width=True)
                            buf, fname = download_chart(chart, "abc_distribution_value.png")
                            st.download_button(
                                label="📥 Descargar Gráfico (HTML)",
                                data=buf,
                                file_name=fname,
                                mime="text/html"
                            )
                        with col2:
                            chart = plot_abc_pie(abc_df_value, 'Proporción ABC por Valor')
                            st.altair_chart(chart, use_container_width=True)
                            buf, fname = download_chart(chart, "abc_pie_value.png")
                            st.download_button(
                                label="📥 Descargar Gráfico (HTML)",
                                data=buf,
                                file_name=fname,
                                mime="text/html"
                            )
                        
                        chart = plot_pareto_curve(abc_df_value.reset_index(), 'Valor_Total', 'Curva de Pareto por Valor')
                        st.altair_chart(chart, use_container_width=True)
                        buf, fname = download_chart(chart, "pareto_value.png")
                        st.download_button(
                            label="📥 Descargar Gráfico (HTML)",
                            data=buf,
                            file_name=fname,
                            mime="text/html"
                        )
                        
                        # Debug merge for box plot (value)
                        st.write("**Debugging Box Plot Data (Value)**")
                        st.write(f"abc_df_value shape: {abc_df_value.shape}")
                        st.write(f"abc_df_value columns: {list(abc_df_value.columns)}")
                        st.write(f"products_df shape: {products_df.shape}")
                        st.write(f"products_df columns: {list(products_df.columns)}")
                        
                        # Use abc_df_value directly since it has Valor_Total and ABC
                        merged_df_value = abc_df_value[[st.session_state.col_map['pv_ref'], 'Valor_Total', 'ABC']].copy()
                        st.write(f"Merged DataFrame shape: {merged_df_value.shape}")
                        st.write(f"Merged DataFrame columns: {list(merged_df_value.columns)}")
                        st.write("Merged DataFrame sample:", merged_df_value.head())
                        
                        chart = plot_demand_box_plot(merged_df_value, 'Valor_Total', 'Distribución de Valor por Categoría ABC')
                        st.altair_chart(chart, use_container_width=True)
                        buf, fname = download_chart(chart, "box_plot_value.png")
                        st.download_button(
                            label="📥 Descargar Gráfico (HTML)",
                            data=buf,
                            file_name=fname,
                            mime="text/html"
                        )

                with tab3:
                    if not xyz_df.empty:
                        st.subheader("Análisis XYZ por Variabilidad de Demanda")
                        st.dataframe(xyz_df.round(2), use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            chart = plot_xyz_distribution(xyz_df, 'Distribución XYZ')
                            st.altair_chart(chart, use_container_width=True)
                            buf, fname = download_chart(chart, "xyz_distribution.png")
                            st.download_button(
                                label="📥 Descargar Gráfico (HTML)",
                                data=buf,
                                file_name=fname,
                                mime="text/html"
                            )
                        with col2:
                            chart = plot_xyz_pie(xyz_df, 'Proporción XYZ')
                            st.altair_chart(chart, use_container_width=True)
                            buf, fname = download_chart(chart, "xyz_pie.png")
                            st.download_button(
                                label="📥 Descargar Gráfico (HTML)",
                                data=buf,
                                file_name=fname,
                                mime="text/html"
                            )

                with tab4:
                    if not contingency.empty:
                        st.subheader("Tabla de Contingencia ABC-XYZ")
                        st.write("**Valores Absolutos**")
                        st.dataframe(contingency, use_container_width=True)
                        st.write("**Porcentajes**")
                        st.dataframe(contingency_pct, use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            chart = plot_contingency_heatmap(contingency_pct, 'Mapa de Calor ABC-XYZ (Porcentajes)')
                            st.altair_chart(chart, use_container_width=True)
                            buf, fname = download_chart(chart, "contingency_heatmap.png")
                            st.download_button(
                                label="📥 Descargar Gráfico (HTML)",
                                data=buf,
                                file_name=fname,
                                mime="text/html"
                            )
                        with col2:
                            chart = plot_abc_xyz_stacked_bar(products_df, 'Combinaciones ABC-XYZ')
                            st.altair_chart(chart, use_container_width=True)
                            buf, fname = download_chart(chart, "abc_xyz_stacked.png")
                            st.download_button(
                                label="📥 Descargar Gráfico (HTML)",
                                data=buf,
                                file_name=fname,
                                mime="text/html"
                            )

                    if not products_df.empty:
                        st.subheader("Productos con Categorías ABC-XYZ")
                        st.dataframe(products_df.round(2), use_container_width=True)
                        chart = plot_demand_cv_scatter(products_df, 'Demanda_Total', 'Demanda vs Variabilidad')
                        st.altair_chart(chart, use_container_width=True)
                        buf, fname = download_chart(chart, "demand_cv_scatter.png")
                        st.download_button(
                            label="📥 Descargar Gráfico (HTML)",
                            data=buf,
                            file_name=fname,
                            mime="text/html"
                        )
                        excel_file = download_excel(products_df, f"ABC_XYZ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                        st.download_button(
                            label="📥 Descargar Resultados en Excel",
                            data=excel_file,
                            file_name=f"ABC_XYZ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()