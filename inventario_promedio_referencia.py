import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import io
import plotly.express as px
from typing import Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constantes
COLMAP_FILE = "col_map.json"

# Funciones
def load_col_map() -> dict:
    logger.debug("Intentando cargar col_map.json")
    if not os.path.exists(COLMAP_FILE):
        logger.error(f"Archivo col_map.json no encontrado en {os.path.abspath(COLMAP_FILE)}")
        return {}
    try:
        with open(COLMAP_FILE, "r", encoding="utf-8") as f:
            col_map = json.load(f)
            logger.debug(f"col_map.json cargado: {col_map}")
            return col_map
    except Exception as e:
        logger.error(f"Error al cargar col_map.json: {e}")
        return {}

def calculate_avg_inventory(df_rec: pd.DataFrame, df_pv: pd.DataFrame, fecha_col: str, ref_col: str, qty_col: str) -> pd.DataFrame:
    logger.debug("Calculando inventario promedio time-weighted")
    try:
        # Validar columnas en los DataFrames
        for df, name in [(df_rec, "Recepciones"), (df_pv, "Pedidos")]:
            missing = [col for col in [fecha_col, ref_col, qty_col] if col not in df.columns]
            if missing:
                logger.error(f"Faltan columnas en {name}.xlsx: {missing}")
                raise ValueError(f"Faltan columnas en {name}.xlsx: {missing}")
        
        # Preparar datos
        df_rec = df_rec[[fecha_col, ref_col, qty_col]].copy()
        df_pv = df_pv[[fecha_col, ref_col, qty_col]].copy()
        
        # Crear columna Evento y ajustar Cantidad
        df_rec['Evento'] = 'Compra'
        df_pv['Evento'] = 'Venta'
        df_rec[qty_col] = df_rec[qty_col].abs()  # Compras: positivo
        df_pv[qty_col] = -df_pv[qty_col].abs()   # Ventas: negativo
        
        # Combinar eventos
        df_events = pd.concat([
            df_rec.rename(columns={fecha_col: 'Fecha', ref_col: 'Referencia', qty_col: 'Cantidad'}),
            df_pv.rename(columns={fecha_col: 'Fecha', ref_col: 'Referencia', qty_col: 'Cantidad'})
        ], ignore_index=True)
        
        # Convertir Fecha a datetime
        df_events['Fecha'] = pd.to_datetime(df_events['Fecha'], errors='coerce')
        if df_events['Fecha'].isna().any():
            logger.error("Fechas inválidas detectadas en los datos")
            raise ValueError("Una o más fechas en los archivos Excel son inválidas")
        
        df_events = df_events.sort_values(['Referencia', 'Fecha'])
        
        # Calcular inventario promedio por referencia
        results = []
        for ref in df_events['Referencia'].unique():
            df_ref = df_events[df_events['Referencia'] == ref].copy()
            if df_ref.empty:
                continue
                
            # Calcular inventario acumulado
            df_ref['Inventario'] = df_ref['Cantidad'].cumsum()
            
            # Calcular días entre eventos
            df_ref['Fecha_Anterior'] = df_ref['Fecha'].shift(1)
            df_ref['Días'] = (df_ref['Fecha'] - df_ref['Fecha_Anterior']).dt.days.fillna(0)
            
            # Usar el inventario del evento anterior para Inventario × Días
            df_ref['Inventario_Anterior'] = df_ref['Inventario'].shift(1).fillna(0)
            df_ref['Inventario × Días'] = df_ref['Inventario_Anterior'] * df_ref['Días']
            
            # Calcular promedio
            total_inventory_days = df_ref['Inventario × Días'].sum()
            total_days = (df_ref['Fecha'].max() - df_ref['Fecha'].min()).days
            avg_inventory = total_inventory_days / total_days if total_days > 0 else 0
            
            results.append({
                'Referencia': ref,
                'Inventario Promedio': avg_inventory,
                'Entradas': df_ref[df_ref['Evento'] == 'Compra']['Cantidad'].sum(),
                'Salidas': -df_ref[df_ref['Evento'] == 'Venta']['Cantidad'].sum()
            })
        
        result_df = pd.DataFrame(results)
        logger.debug(f"Inventario promedio calculado: {result_df.shape[0]} filas")
        return result_df[['Referencia', 'Inventario Promedio', 'Entradas', 'Salidas']]
    except Exception as e:
        logger.error(f"Error en calculate_avg_inventory: {e}")
        return pd.DataFrame()

def download_excel(df: pd.DataFrame, filename: str) -> io.BytesIO:
    logger.debug("Generando archivo Excel")
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Inventario_Promedio')
        output.seek(0)
        logger.debug("Archivo Excel generado")
        return output
    except Exception as e:
        logger.error(f"Error al generar Excel: {e}")
        return io.BytesIO()

# Configuración de la interfaz de Streamlit
logger.debug("Iniciando configuración de Streamlit")
st.set_page_config(page_title="Cálculo de Inventario Promedio", layout="wide")
st.title("Cálculo de Inventario Promedio")

# Barra lateral para carga de archivos
st.sidebar.header("Cargar Archivos Excel")
uploaded_pv = st.sidebar.file_uploader("Pedidos.xlsx", type=["xlsx"])
uploaded_rec = st.sidebar.file_uploader("Recepciones.xlsx", type=["xlsx"])

# Estado de los archivos cargados
st.sidebar.header("Estado de Archivos")
st.sidebar.text(f"Pedidos: {'✅ Subido' if uploaded_pv else '❌ No subido'}")
st.sidebar.text(f"Recepciones: {'✅ Subido' if uploaded_rec else '❌ No subido'}")

# Configuración de visualización
st.sidebar.header("Configuración de Visualización")
n_references = st.sidebar.slider("Número de referencias a mostrar", min_value=1, max_value=50, value=10)

# Cargar col_map.json
logger.debug("Cargando col_map.json")
col_map = load_col_map()
if not col_map:
    st.error(
        f"Error: No se pudo cargar col_map.json. Crea un archivo 'col_map.json' en {os.path.abspath(COLMAP_FILE)} con el siguiente contenido:\n"
        "```json\n"
        "{\n"
        "    \"fecha_col\": \"Fecha\",\n"
        "    \"ref_col\": \"Referencia\",\n"
        "    \"qty_col\": \"Cantidad\"\n"
        "}\n"
        "```"
    )
    st.stop()

# Validar columnas requeridas
required_cols = ["fecha_col", "ref_col", "qty_col"]
missing_cols = [col for col in required_cols if col not in col_map or not col_map[col]]
if missing_cols:
    st.error(
        f"Faltan las siguientes columnas en col_map.json: {', '.join(missing_cols)}. Asegúrate de que el archivo contenga:\n"
        "```json\n"
        "{\n"
        "    \"fecha_col\": \"Fecha\",\n"
        "    \"ref_col\": \"Referencia\",\n"
        "    \"qty_col\": \"Cantidad\"\n"
        "}\n"
        "```"
    )
    st.stop()

# Cargar datos
logger.debug("Cargando datos de Excel")
try:
    df_pv = pd.read_excel(uploaded_pv) if uploaded_pv else pd.DataFrame()
    df_rec = pd.read_excel(uploaded_rec) if uploaded_rec else pd.DataFrame()
    logger.debug(f"Datos cargados - Pedidos: {df_pv.shape}, Recepciones: {df_rec.shape}")
except Exception as e:
    logger.error(f"Error al cargar archivos Excel: {e}")
    st.error(f"Error al cargar archivos Excel: {e}. Verifica que los archivos sean válidos y contengan las columnas especificadas en col_map.json.")
    st.stop()

# Vista previa de los datos cargados
st.header("Vista Previa de Datos")
if not df_pv.empty:
    st.subheader("Pedidos.xlsx (primeras 5 filas)")
    st.dataframe(df_pv.head(), use_container_width=True)
if not df_rec.empty:
    st.subheader("Recepciones.xlsx (primeras 5 filas)")
    st.dataframe(df_rec.head(), use_container_width=True)

# Botón para calcular
if st.sidebar.button("Calcular Inventario Promedio"):
    logger.debug("Botón 'Calcular Inventario Promedio' presionado")
    if not (uploaded_pv and uploaded_rec):
        st.error("Por favor, carga los archivos Pedidos.xlsx y Recepciones.xlsx.")
    else:
        # Calcular inventario promedio
        avg_inv_df = calculate_avg_inventory(
            df_rec, df_pv,
            col_map.get("fecha_col", "Fecha"),
            col_map.get("ref_col", "Referencia"),
            col_map.get("qty_col", "Cantidad")
        )

        if avg_inv_df.empty:
            st.error("No se generaron resultados. Verifica que los archivos contengan datos válidos (fechas válidas, cantidades numéricas, referencias coincidentes).")
            logger.warning("avg_inv_df está vacío")
        else:
            # Mostrar resultados
            st.success("Cálculo completado!")
            st.header("Resultados de Inventario Promedio")
            st.dataframe(avg_inv_df.round(2), use_container_width=True)

            # Visualización: Gráfico de barras con las top N referencias
            st.header("Visualización: Top Referencias por Inventario Promedio")
            top_n_df = avg_inv_df.sort_values("Inventario Promedio", ascending=False).head(n_references)
            fig = px.bar(
                top_n_df,
                x="Referencia",
                y="Inventario Promedio",
                text="Inventario Promedio",
                hover_data=["Entradas", "Salidas"],
                title=f"Top {n_references} Referencias por Inventario Promedio",
                labels={"Inventario Promedio": "Inventario Promedio (unidades)"}
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
            fig.update_layout(
                xaxis_title="Referencia",
                yaxis_title="Inventario Promedio (unidades)",
                xaxis_tickangle=45,
                showlegend=False,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            # Descargar resultados
            output_file = f"Inventario_Promedio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            excel_file = download_excel(avg_inv_df, output_file)
            st.download_button(
                label="Descargar resultados en Excel",
                data=excel_file,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )