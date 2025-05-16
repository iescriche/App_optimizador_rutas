import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from typing import Dict, List, Tuple, Optional

# Funciones necesarias para el cálculo del stock mínimo
def clean_string(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    for ch in ['\xa0', '\u00A0', '\u200B', '\u200C', '\u200D', '\uFEFF']:
        s = s.replace(ch, '')
    return s.strip()

def validate_date(val) -> Optional[datetime.date]:
    try:
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.date()
        if isinstance(val, str):
            s = str(val).strip().replace("FECHA ALBARAN", "")
            for ch in ['\u00A0', '\u200B', '\u200C', '\u200D', '\uFEFF']:
                s = s.replace(ch, '')
            s = s.strip()
            dt = pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            if not pd.isna(dt):
                return dt.date()
            dt = pd.to_datetime(s, format='%d/%m/%Y', dayfirst=True, errors='coerce')
            if not pd.isna(dt):
                return dt.date()
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d']:
                dt = pd.to_datetime(s, format=fmt, errors='coerce')
                if not pd.isna(dt):
                    return dt.date()
            st.warning(f"Fecha no válida: '{s}'")
            return None
        st.warning(f"Valor no procesable: '{val}'")
        return None
    except Exception as e:
        st.warning(f"Error en validación: {e}, valor: '{val}'")
        return None

def months_between(start: datetime, end: datetime) -> int:
    return (end.year - start.year) * 12 + end.month - start.month + 1

def calculate_inventory_stats(
    df_rec: pd.DataFrame,
    df_ord: pd.DataFrame,
    df_pv: pd.DataFrame,
    df_maestro: pd.DataFrame,
    col_map: Dict[str, str],
    start_date: datetime,
    end_date: datetime,
    z_factor: float
) -> Tuple[pd.DataFrame, List[str]]:
    debug_log = []
    required_cols = [
        "rec_ped", "rec_date", "rec_prov", "rec_ref", "rec_qty",
        "ord_ped", "ord_prov", "ord_ref", "ord_qty", "ord_date",
        "pv_ref", "pv_qty", "pv_date"
    ]
    missing_cols = [col for col in required_cols if col not in col_map or not col_map[col]]
    if missing_cols:
        debug_log.append(f"Faltan columnas mapeadas: {', '.join(missing_cols)}")
        return pd.DataFrame(), debug_log

    for df in [df_rec, df_ord, df_pv]:
        for col in df.columns:
            df[col] = df[col].apply(clean_string)

    desc_dict = {}
    if not df_maestro.empty and col_map.get("maestro_ref") and col_map.get("maestro_desc"):
        df_maestro = df_maestro.dropna(subset=[col_map["maestro_ref"]])
        for _, row in df_maestro.iterrows():
            ref = clean_string(row[col_map["maestro_ref"]])
            desc = clean_string(row.get(col_map["maestro_desc"], ""))
            if ref:
                desc_dict[ref] = desc

    orders_dict = {}
    demand_dict = {}
    lead_count = {}
    lead_sum = {}
    lead_sum_sq = {}

    if df_ord.empty:
        debug_log.append("El archivo Compras.xlsx está vacío.")
    else:
        for _, row in df_ord.iterrows():
            order_no = clean_string(row.get(col_map["ord_ped"], ""))
            ref = clean_string(row.get(col_map["ord_ref"], ""))
            prov = clean_string(row.get(col_map["ord_prov"], ""))
            qty = pd.to_numeric(row.get(col_map["ord_qty"], 0), errors="coerce")
            order_date = validate_date(row.get(col_map["ord_date"], ""))
            if pd.isna(qty):
                qty = 0
            if not order_no or not ref:
                continue
            if order_date is None:
                debug_log.append(f"Fecha inválida en Pedidos de Compra, fila {row.name + 2}: '{row.get(col_map['ord_date'], '')}'")
                continue
            key = f"{order_no}|{ref}"
            if key not in orders_dict:
                orders_dict[key] = [order_date, prov, qty]
            else:
                orders_dict[key][2] += qty

    total_interval_months = months_between(start_date, end_date)
    if total_interval_months <= 0:
        debug_log.append("El rango de fechas es inválido.")
        return pd.DataFrame(), debug_log
    debug_log.append(f"Intervalo de meses: {total_interval_months}")

    if df_rec.empty:
        debug_log.append("El archivo Recepciones.xlsx está vacío.")
    else:
        for _, row in df_rec.iterrows():
            order_no = clean_string(row.get(col_map["rec_ped"], ""))
            ref = clean_string(row.get(col_map["rec_ref"], ""))
            qty = pd.to_numeric(row.get(col_map["rec_qty"], 0), errors="coerce")
            receipt_date = validate_date(row.get(col_map["rec_date"], ""))
            if pd.isna(qty) or qty <= 0 or receipt_date is None:
                if receipt_date is None:
                    debug_log.append(f"Fecha inválida en Recepciones, fila {row.name + 2}: '{row.get(col_map['rec_date'], '')}'")
                continue
            if not (start_date <= receipt_date <= end_date):
                continue
            if not order_no or not ref:
                continue
            key = f"{order_no}|{ref}"
            if key in orders_dict and orders_dict[key][0] is not None:
                order_date = orders_dict[key][0]
                diff_days = (receipt_date - order_date).days
                if diff_days >= 0:
                    if ref not in lead_count:
                        lead_count[ref] = 0
                        lead_sum[ref] = 0.0
                        lead_sum_sq[ref] = 0.0
                    lead_count[ref] += 1
                    lead_sum[ref] += diff_days
                    lead_sum_sq[ref] += diff_days ** 2
                else:
                    debug_log.append(f"Lead time negativo para referencia {ref}, pedido {order_no}: {diff_days} días")

    if df_pv.empty:
        debug_log.append("El archivo Pedidos.xlsx está vacío.")
    else:
        for _, row in df_pv.iterrows():
            ref = clean_string(row.get(col_map["pv_ref"], ""))
            qty = pd.to_numeric(row.get(col_map["pv_qty"], 0), errors="coerce")
            pv_date = validate_date(row.get(col_map["pv_date"], ""))
            if not ref or qty <= 0 or pv_date is None:
                if pv_date is None:
                    debug_log.append(f"Fecha inválida en Pedidos de Venta, fila {row.name + 2}: '{row.get(col_map['pv_date'], '')}'")
                continue
            if not (start_date <= pv_date <= end_date):
                continue
            month_key = pv_date.strftime("%Y-%m")
            if ref not in demand_dict:
                demand_dict[ref] = {}
            demand_dict[ref][month_key] = demand_dict[ref].get(month_key, 0) + qty

    all_refs = set(demand_dict.keys()).union(lead_count.keys())
    if not all_refs:
        debug_log.append("No se encontraron referencias válidas.")
        return pd.DataFrame(), debug_log

    results = []
    for ref in all_refs:
        total_demand = sum(demand_dict[ref].values()) if ref in demand_dict else 0
        month_count = len(demand_dict[ref]) if ref in demand_dict else 0
        avg_demand = max(0, total_demand / total_interval_months if total_interval_months > 0 else 0)

        sum_dev = 0
        if month_count > 0:
            for qty in demand_dict[ref].values():
                sum_dev += (qty - avg_demand) ** 2
        demand_var = sum_dev / total_interval_months if total_interval_months > 0 else 0

        lt_avg = lead_sum[ref] / lead_count[ref] if ref in lead_count and lead_count[ref] > 0 else 0
        lt_var = 0
        if ref in lead_count and lead_count[ref] > 1:
            lt_var = max(0, lead_sum_sq[ref] / lead_count[ref] - lt_avg ** 2)

        lt_month = lt_avg / 30 if lt_avg > 0 else 0
        var_lt_month = lt_var / (30 ** 2) if lt_var > 0 else 0
        daily_demand = avg_demand / 30 if avg_demand > 0 else 0

        var_dem_during_lt = (lt_month * demand_var) + (daily_demand ** 2 * var_lt_month)
        var_dem_during_lt = max(0, var_dem_during_lt)

        cycle_stock = daily_demand * lt_avg
        safety_stock = z_factor * (var_dem_during_lt ** 0.5) if var_dem_during_lt > 0 else 0
        min_stock = int(cycle_stock + safety_stock) + 1

        results.append({
            "Referencia": ref,
            "Descripción": desc_dict.get(ref, ""),
            "Stock mínimo": min_stock
        })

    result_df = pd.DataFrame(results).sort_values("Referencia")
    return result_df, debug_log

def download_excel(df, filename):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Stock Mínimo')
    output.seek(0)
    return output

def run_module(df_rec: pd.DataFrame, df_ord: pd.DataFrame, df_pv: pd.DataFrame, df_maestro: pd.DataFrame, col_map: Dict[str, str]):
    """Ejecuta la interfaz Streamlit del módulo."""
    st.title("Cálculo de Stock Mínimo")

    # Validar columnas requeridas
    required_cols = [
        "rec_ped", "rec_date", "rec_prov", "rec_ref", "rec_qty",
        "ord_ped", "ord_prov", "ord_ref", "ord_qty", "ord_date",
        "pv_ref", "pv_qty", "pv_date"
    ]
    missing_cols = [col for col in required_cols if col not in col_map or not col_map[col]]
    if missing_cols:
        st.error(f"Faltan las siguientes columnas en col_map.json: {', '.join(missing_cols)}")
        st.stop()

    # Vista previa de los datos cargados
    st.header("Vista Previa de Datos")
    if not df_rec.empty:
        st.subheader("Recepciones.xlsx (primeras 5 filas)")
        st.dataframe(df_rec.head(), use_container_width=True)
    if not df_ord.empty:
        st.subheader("Compras.xlsx (primeras 5 filas)")
        st.dataframe(df_ord.head(), use_container_width=True)
    if not df_pv.empty:
        st.subheader("Pedidos.xlsx (primeras 5 filas)")
        st.dataframe(df_pv.head(), use_container_width=True)
    if not df_maestro.empty:
        st.subheader("Maestro.xlsx (primeras 5 filas)")
        st.dataframe(df_maestro.head(), use_container_width=True)

    # Configuración de parámetros
    st.sidebar.header("Configuración")
    date_option = st.sidebar.selectbox("Rango de Fechas", ["Últimos N Meses", "Rango Personalizado"])
    if date_option == "Últimos N Meses":
        n_months = st.sidebar.number_input("Número de Meses", min_value=1, value=12)
        end_date = datetime.today().replace(day=1).date()
        start_date = (end_date - pd.offsets.MonthBegin(n_months)).date()
    else:
        start_date = st.sidebar.date_input("Fecha Inicio", value=datetime.today().replace(day=1, month=1))
        end_date = st.sidebar.date_input("Fecha Fin", value=datetime.today())
    service_level = st.sidebar.selectbox("Nivel de Servicio (%)", [90, 95, 99], index=1)

    # Botón para calcular
    if st.sidebar.button("Calcular Stock Mínimo"):
        if df_rec.empty or df_ord.empty or df_pv.empty:
            st.error("Faltan datos de Recepciones.xlsx, Compras.xlsx o Pedidos.xlsx.")
        else:
            # Validar fechas
            if start_date >= end_date:
                st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
                st.stop()

            # Mapear nivel de servicio a z_factor
            z_factors = {90: 1.28, 95: 1.65, 99: 2.33}
            z_factor = z_factors[service_level]

            # Calcular estadísticas
            result_df, debug_log = calculate_inventory_stats(
                df_rec, df_ord, df_pv, df_maestro, col_map, start_date, end_date, z_factor
            )

            if result_df.empty:
                st.error("No se generaron resultados. Revisa el log de depuración.")
                st.text_area("Log de depuración", "\n".join(debug_log), height=200)
            else:
                # Mostrar resultados
                st.success("Cálculo completado!")
                st.header("Resultados del Stock Mínimo")
                st.dataframe(result_df.round(2), use_container_width=True)

                # Descargar resultados
                output_file = f"Stock_Minimo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                excel_file = download_excel(result_df, output_file)
                st.download_button(
                    label="Descargar resultados en Excel",
                    data=excel_file,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.text_area("Detalles", "\n".join(debug_log), height=200)