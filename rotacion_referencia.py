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
    if os.path.exists(COLMAP_FILE):
        try:
            with open(COLMAP_FILE, "r", encoding="utf-8") as f:
                col_map = json.load(f)
                logger.debug(f"col_map.json cargado: {col_map}")
                return col_map
        except Exception as e:
            logger.error(f"Error al cargar col_map.json: {e}")
            return {}
    logger.warning("col_map.json no encontrado")
    return {}

def calculate_avg_inventory(df_rec: pd.DataFrame, df_pv: pd.DataFrame, ref_col_rec: str, ref_col_pv: str, qty_col_rec: str, qty_col_pv: str) -> pd.DataFrame:
    logger.debug("Calculando inventario promedio")
    try:
        rec_sum = df_rec.groupby(ref_col_rec)[qty_col_rec].sum().reset_index(name='Entradas')
        pv_sum = df_pv.groupby(ref_col_pv)[qty_col_pv].sum().reset_index(name='Salidas')
        inventory = rec_sum.merge(pv_sum, left_on=ref_col_rec, right_on=ref_col_pv, how='outer').fillna(0)
        inventory['Inventario Promedio'] = np.maximum((inventory['Entradas'] - inventory['Salidas']) / 2, 0)
        logger.debug(f"Inventario promedio calculado: {inventory.shape[0]} filas")
        return inventory[[ref_col_rec, 'Inventario Promedio', 'Entradas', 'Salidas']].rename(columns={ref_col_rec: 'Referencia'})
    except Exception as e:
        logger.error(f"Error en calculate_avg_inventory: {e}")
        return pd.DataFrame()

def calculate_rotation(df_pv: pd.DataFrame, df_rec: pd.DataFrame, ref_col_pv: str, ref_col_rec: str, qty_col_pv: str, qty_col_rec: str) -> pd.DataFrame:
    logger.debug("Calculando rotación")
    try:
        demand = df_pv.groupby(ref_col_pv)[qty_col_pv].sum().reset_index(name='Demanda Anual')
        avg_inventory = calculate_avg_inventory(df_rec, df_pv, ref_col_rec, ref_col_pv, qty_col_rec, qty_col_pv)
        rotation = demand.merge(avg_inventory, left_on=ref_col_pv, right_on=ref_col_rec)
        rotation['Rotación'] = rotation['Demanda Anual'] / rotation['Inventario Promedio'].replace(0, np.nan)
        rotation['Rotación'] = rotation['Rotación'].clip(lower=0)
        logger.debug(f"Rotación calculada: {rotation.shape[0]} filas")
        return rotation[['Referencia', 'Demanda Anual', 'Inventario Promedio', 'Entradas', 'Salidas', 'Rotación']]
    except Exception as e:
        logger.error(f"Error en calculate_rotation: {e}")
        return pd.DataFrame()

def download_excel(df: pd.DataFrame, filename: str) -> io.BytesIO:
    logger.debug("Generando archivo Excel")
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Rotación')
        output.seek(0)
        logger.debug("Archivo Excel generado")
        return output
    except Exception as e:
        logger.error(f"Error al generar Excel: {e}")
        return io.BytesIO()

# Configuración de la interfaz de Streamlit
logger.debug("Iniciando configuración de Streamlit")
st.set_page_config(page_title="Cálculo de Rotación de Inventario", layout="wide")
st.title("Cálculo de Rotación de Inventario")

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
    st.error("Error: No se pudo cargar col_map.json o está vacío. Crea un archivo col_map.json con el mapeo de columnas.")
    st.stop()

# Validar columnas requeridas
required_cols = ["pv_ref", "pv_qty", "rec_ref", "rec_qty"]
missing_cols = [col for col in required_cols if col not in col_map or not col_map[col]]
if missing_cols:
    st.error(f"Faltan las siguientes columnas en col_map.json: {', '.join(missing_cols)}")
    st.stop()

# Cargar datos
logger.debug("Cargando datos de Excel")
try:
    df_pv = pd.read_excel(uploaded_pv) if uploaded_pv else pd.DataFrame()
    df_rec = pd.read_excel(uploaded_rec) if uploaded_rec else pd.DataFrame()
    logger.debug(f"Datos cargados - Pedidos: {df_pv.shape}, Recepciones: {df_rec.shape}")
except Exception as e:
    logger.error(f"Error al cargar archivos Excel: {e}")
    st.error(f"Error al cargar archivos Excel: {e}")
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
if st.sidebar.button("Calcular Rotación"):
    logger.debug("Botón 'Calcular Rotación' presionado")
    if not (uploaded_pv and uploaded_rec):
        st.error("Por favor, carga los archivos Pedidos.xlsx y Recepciones.xlsx.")
    else:
        # Calcular rotación
        rotation_df = calculate_rotation(
            df_pv, df_rec,
            col_map.get("pv_ref", "Referencia"),
            col_map.get("rec_ref", "Referencia"),
            col_map.get("pv_qty", "Cantidad"),
            col_map.get("rec_qty", "Cantidad")
        )

        if rotation_df.empty:
            st.error("No se generaron resultados. Verifica que los archivos contengan datos válidos.")
            logger.warning("rotation_df está vacío")
        else:
            # Mostrar resultados
            st.success("Cálculo completado!")
            st.header("Resultados de Rotación")
            st.dataframe(rotation_df.round(2), use_container_width=True)

            # Descargar resultados
            output_file = f"Rotacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            excel_file = download_excel(rotation_df, output_file)
            st.download_button(
                label="Descargar resultados en Excel",
                data=excel_file,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )