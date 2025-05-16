
from __future__ import annotations
import io
import math
import datetime
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import uuid

import altair as alt
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from streamlit_folium import st_folium
from tenacity import retry, stop_after_attempt, wait_fixed

# ⏱️ Límite legal
MAX_H_REG = 8  # Horas regulares

# =============================================================
# UTILITIES
# =============================================================

def clean_addr(s: str) -> str:
    """Convierte a mayúsculas, elimina tildes y normaliza espacios."""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.upper().split())

def time_to_minutes(t: str) -> int:
    """Convierte HH:MM a minutos desde medianoche."""
    try:
        h, m = map(int, t.split(":"))
    except Exception as e:
        raise ValueError(f"Formato hora inválido: {t}") from e
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"Hora fuera de rango: {t}")
    return h * 60 + m

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000  # m
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlat / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def validate_coords(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180

# =============================================================
# ORS
# =============================================================

def validate_ors_key(api_key: str) -> bool:
    body = {"coordinates": [[8.681495, 49.41461], [8.686507, 49.41943]]}
    try:
        r = requests.post(
            "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
            json=body,
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            timeout=10,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def ors_distance_matrix_block(
    origins: List[Tuple[float, float]],
    dests: List[Tuple[float, float]],
    api_key: str,
    mode: str = "driving-car",
) -> Tuple[List[List[float]], List[List[int]]]:
    coords = [[lon, lat] for lat, lon in origins + dests]
    body = {
        "locations": coords,
        "sources": list(range(len(origins))),
        "destinations": list(range(len(origins), len(origins) + len(dests))),
        "metrics": ["distance", "duration"],
    }
    r = requests.post(
        f"https://api.openrouteservice.org/v2/matrix/{mode}",
        json=body,
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    dist = data["distances"]
    dur = [[int(x / 60) for x in row] for row in data["durations"]]
    return dist, dur

def _compute_matrix_chunk(
    coords: List[Tuple[float, float]],
    api_key: str,
    block: int = 10,
    mode: str = "driving-car",
):
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    dur = [[0] * n for _ in range(n)]

    for r0 in range(0, n, block):
        for c0 in range(0, n, block):
            sub_orig = coords[r0 : r0 + block]
            sub_dest = coords[c0 : c0 + block]
            try:
                d_sub, t_sub = ors_distance_matrix_block(sub_orig, sub_dest, api_key, mode)
            except Exception:
                d_sub = [[haversine(*o, *d) for d in sub_dest] for o in sub_orig]
                speed_kmh = 30 if mode == "driving-car" else 15 if mode.startswith("cycling") else 5
                t_sub = [[int(d / 1000 / speed_kmh * 60) for d in row] for row in d_sub]

            for i, ri in enumerate(range(r0, min(r0 + block, n))):
                for j, cj in enumerate(range(c0, min(c0 + block, n))):
                    dist[ri][cj] = d_sub[i][j]
                    dur[ri][cj] = t_sub[i][j]

    for i in range(n):
        for j in range(i + 1, n):
            dist[j][i] = dist[i][j]
            dur[j][i] = dur[i][j]
    return dist, dur

@st.cache_data(show_spinner=False, max_entries=10)
def ors_matrix_chunk(coords, api_key, block=10, mode="driving-car"):
    """Parte cacheada (sin UI)."""
    return _compute_matrix_chunk(coords, api_key, block, mode)

# =============================================================
# VRP helpers
# =============================================================

def build_cost_matrix(
    dist_m: List[List[float]],      # metros
    time_m: List[List[int]],        # minutos
    fuel_price: float,              # €/litro
    fuel_consumption: float,        # litros/100 km
    driver_price_h: float           # €/hora
) -> List[List[int]]:
    """Devuelve matriz de coste en céntimos (int) para evitar floats."""
    n = len(dist_m)
    cost = [[0]*n for _ in range(n)]
    # factores
    c_km = fuel_price * fuel_consumption / 100          # €/km
    for i in range(n):
        for j in range(n):
            km   = dist_m[i][j] / 1_000                 # metros → km
            hrs  = time_m[i][j] / 60                    # minutos → horas
            euro = km*c_km + hrs*driver_price_h
            cost[i][j] = int(round(euro*100))           # céntimos
    return cost

def build_routing_model(
    arc_cost_m: List[List[int]],        # céntimos
    time_m: List[List[int]],            # minutos
    vehs: int,
    depot: int,
    service_time: int,
    start_min: int,                     # Minutos desde medianoche
    balance: bool = True,
    max_stops_per_vehicle: int | None = None,
    price_per_hour: float = 0.0,        # €/hora, para coste fijo
    max_minutes: int = 1440             # Máximo tiempo permitido por ruta
):
    """Crea y devuelve (manager, routing, time_dim)."""
    n = len(arc_cost_m)
    man = pywrapcp.RoutingIndexManager(n, vehs, depot)
    rout = pywrapcp.RoutingModel(man)

    dist_cb = rout.RegisterTransitCallback(
        lambda i, j: arc_cost_m[man.IndexToNode(i)][man.IndexToNode(j)]
    )
    rout.SetArcCostEvaluatorOfAllVehicles(dist_cb)

    # Coste fijo por vehículo: 2 horas de sueldo del conductor en céntimos
    if price_per_hour > 0:
        fixed_cents = int(round(2 * price_per_hour * 100))  # 2 horas base
        for v in range(vehs):
            rout.SetFixedCostOfVehicle(fixed_cents, v)

    max_time = max(max(r) for r in time_m) * n
    time_cb = rout.RegisterTransitCallback(
        lambda i, j: time_m[man.IndexToNode(i)][man.IndexToNode(j)] + (
            service_time if man.IndexToNode(j) != depot else 0
        )
    )
    rout.AddDimension(time_cb, 0, max(max_minutes, max_time), False, "Time")
    time_dim = rout.GetDimensionOrDie("Time")

    # Aplicar límite de tiempo máximo por vehículo
    for v in range(vehs):
        start = rout.Start(v)
        end = rout.End(v)
        time_dim.CumulVar(start).SetRange(start_min, start_min)  # Fija inicio en start_min
        time_dim.CumulVar(end).SetRange(start_min, start_min + max_minutes)

    # Dimensión de paradas (si aplica)
    if max_stops_per_vehicle is not None and vehs > 1 and max_stops_per_vehicle < (n - 1):
        demand_cb = rout.RegisterUnaryTransitCallback(
            lambda i: 1 if man.IndexToNode(i) != depot else 0
        )
        rout.AddDimension(demand_cb, 0, max_stops_per_vehicle, True, "Stops")

    # Dimensión de coste para balanceo
    rout.AddDimension(dist_cb, 0, 10**9, True, "Cost")
    rout.GetDimensionOrDie("Cost").SetGlobalSpanCostCoefficient(1)

    # Balanceo basado en distancia (si aplica)
    if balance and vehs > 1:
        rout.AddDimension(dist_cb, 0, 1_000_000_000, True, "Distance")
        rout.GetDimensionOrDie("Distance").SetGlobalSpanCostCoefficient(1000)

    return man, rout, time_dim

def solve_vrp_simple(
    dist_m,
    time_m,
    vehs,
    depot,
    balance,
    start_min,
    service_time,
    max_stops_per_vehicle,
    fuel_price,
    fuel_consumption,
    price_per_hour,
    max_minutes
):
    cost_m = build_cost_matrix(
        dist_m, time_m,
        fuel_price=fuel_price,
        fuel_consumption=fuel_consumption,
        driver_price_h=price_per_hour
    )
    man, rout, time_dim = build_routing_model(
        arc_cost_m=cost_m,
        time_m=time_m,
        vehs=vehs,
        depot=depot,
        service_time=service_time,
        start_min=start_min,
        balance=balance,
        max_stops_per_vehicle=max_stops_per_vehicle,
        price_per_hour=price_per_hour,
        max_minutes=max_minutes
    )

    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    prm.time_limit.seconds = 60

    sol = rout.SolveWithParameters(prm)
    if sol is None:
        st.warning(f"No se encontró solución en solve_vrp_simple con {vehs} vehículos. Relaxing constraints.")
        return [], [None] * len(dist_m), 0

    routes: List[List[int]] = []
    eta = [None] * len(dist_m)
    for v in range(vehs):
        idx = rout.Start(v)
        r = []
        while not rout.IsEnd(idx):
            node = man.IndexToNode(idx)
            r.append(node)
            if node != depot:
                eta[node] = sol.Value(time_dim.CumulVar(idx))
            idx = sol.Value(rout.NextVar(idx))
        r.append(man.IndexToNode(idx))
        routes.append(r)

    used = sum(1 for r in routes if len(r) > 2)
    return routes, eta, used

def reassign_nearby_stops(
    routes: List[List[int]],
    dist_m: List[List[float]],
    time_m: List[List[int]],
    depot: int,
    balance: bool,
    start_min: int,
    service_time: int,
    max_stops_per_vehicle: int,
    max_distance_m: float,
    fuel_price: float,
    fuel_consumption: float,
    price_per_hour: float,
    max_minutes: int
):
    n = len(dist_m)
    vehs = len(routes)
    st.info(f"reassign_nearby_stops: Entrada con {vehs} vehículos, {n} nodos, rutas: {routes}")

    cost_m = build_cost_matrix(
        dist_m, time_m,
        fuel_price=fuel_price,
        fuel_consumption=fuel_consumption,
        driver_price_h=price_per_hour
    )
    man, rout, time_dim = build_routing_model(
        arc_cost_m=cost_m,
        time_m=time_m,
        vehs=vehs,
        depot=depot,
        service_time=service_time,
        start_min=start_min,
        balance=balance,
        max_stops_per_vehicle=max_stops_per_vehicle,
        price_per_hour=price_per_hour,
        max_minutes=max_minutes
    )

    for v in range(vehs):
        time_dim.CumulVar(rout.Start(v)).SetRange(start_min, start_min)

    for node in range(1, n):
        idx = man.NodeToIndex(node)
        if idx < 0 or idx >= man.GetNumberOfIndices():
            st.warning(f"Índice inválido para nodo {node}: idx={idx}. Saltando disyunción.")
            continue
        rout.AddDisjunction([idx], 1_000_000_000)

    for v1, r1 in enumerate(routes):
        if len(r1) <= 2:
            st.warning(f"Ruta {v1} vacía o solo depósito: {r1}. Saltando.")
            continue
        for node1 in r1[1:-1]:
            for v2, r2 in enumerate(routes):
                if v1 == v2 or len(r2) <= 2:
                    continue
                for node2 in r2[1:-1]:
                    if dist_m[node1][node2] <= max_distance_m:
                        try:
                            idx1 = man.NodeToIndex(node1)
                            idx2 = man.NodeToIndex(node2)
                            if not (0 <= idx1 < man.GetNumberOfIndices() and 0 <= idx2 < man.GetNumberOfIndices()):
                                st.warning(f"Índice inválido: idx1={idx1}, idx2={idx2} para nodos {node1}, {node2}. Saltando.")
                                continue
                            if not (0 <= v1 < vehs and 0 <= v2 < vehs):
                                st.warning(f"Índice de vehículo inválido: v1={v1}, v2={v2}. Saltando.")
                                continue
                            st.debug(f"Asignando vehículos {[v1, v2]} a nodos {node1} (idx={idx1}), {node2} (idx={idx2})")
                            rout.SetAllowedVehiclesForIndex([v1, v2], idx1)
                            rout.SetAllowedVehiclesForIndex([v1, v2], idx2)
                        except Exception as e:
                            continue

    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    prm.time_limit.seconds = 60
    prm.log_search = True

    sol = rout.SolveWithParameters(prm)
    if sol is None:
        st.warning("No se encontró solución al reasignar paradas cercanas. Devolviendo rutas originales.")
        return routes, [None] * n, sum(1 for r in routes if len(r) > 2)

    new_routes, eta = [], [None] * n
    for v in range(vehs):
        idx = rout.Start(v)
        r = []
        while not rout.IsEnd(idx):
            node = man.IndexToNode(idx)
            r.append(node)
            if node != depot:
                eta[node] = sol.Value(time_dim.CumulVar(idx))
            idx = sol.Value(rout.NextVar(idx))
        r.append(man.IndexToNode(idx))
        new_routes.append(r)

    used = sum(1 for r in new_routes if len(r) > 2)
    st.info(f"reassign_nearby_stops: Nuevas rutas: {new_routes}, vehículos usados: {used}")
    return new_routes, eta, used

def recompute_etas(routes, time_m, start_min, service_time, n):
    eta = [None] * n
    for r in routes:
        if len(r) <= 2:
            continue
        t = start_min
        eta[r[0]] = t
        for i in range(1, len(r) - 1):
            prev, node = r[i - 1], r[i]
            t += time_m[prev][node] + service_time
            eta[node] = t
        t += time_m[r[-2]][r[-1]]
        eta[r[-1]] = t
    return eta

# =============================
# Additional helpers
# =============================

def test_ors_matrix(api_key: str):
    """Prueba una solicitud simple a ORS Matrix API."""
    url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    body = {
        "locations": [[8.681495, 49.41461], [8.686507, 49.41943]],
        "metrics": ["distance", "duration"]
    }
    try:
        response = requests.post(url, json=body, headers=headers)
        if response.status_code == 200:
            st.success("La clave ORS es válida para Matrix API.")
        else:
            st.error(f"Error en la solicitud de prueba: Código {response.status_code}, Mensaje: {response.text}")
    except requests.RequestException as e:
        st.error(f"Error en la solicitud de prueba: {str(e)}")

def detect_redundant_routes_by_distance(routes: List[List[int]], dist_m: List[List[float]], jaccard_threshold: float = 0.8, dist_threshold: float = 0.1) -> int:
    """Detecta rutas redundantes usando Jaccard y distancia relativa."""
    clean_routes = [r[1:-1] for r in routes if len(r) > 2]
    if len(clean_routes) < 2:
        return 0
    route_dists = []
    for route in clean_routes:
        d = sum(dist_m[route[i]][route[i + 1]] for i in range(len(route) - 1))
        route_dists.append((set(route), d))
    redundant = 0
    for (ra, da), (rb, db) in combinations(route_dists, 2):
        inter = len(ra.intersection(rb))
        union = len(ra.union(rb))
        if union == 0:
            continue
        jaccard = inter / union
        if jaccard > jaccard_threshold and abs(da - db) / max(da, db) < dist_threshold:
            redundant += 1
    return redundant

def get_polyline_ors(orig: Tuple[float, float], dest: Tuple[float, float], api_key: str, mode: str = "driving-car") -> List[Tuple[float, float]]:
    """Obtiene polilínea de ruta desde ORS Directions API."""
    url = f"https://api.openrouteservice.org/v2/directions/{mode}/geojson"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    body = {"coordinates": [[orig[1], orig[0]], [dest[1], dest[0]]], "geometry": True}
    try:
        response = requests.post(url, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "features" in data and data["features"]:
            return [(lat, lon) for lon, lat in data["features"][0]["geometry"]["coordinates"]]
        return []
    except requests.RequestException:
        st.error("Error al conectar con ORS Directions API")
        return []

def recomendar_num_vehiculos(
    dist_m: List[List[float]], 
    time_m: List[List[int]], 
    vehs: int, 
    depot: int, 
    balance: bool, 
    start_time_minutes: int, 
    service_time: int,
    balance_threshold: float
) -> int:
    """Recomienda el número óptimo de vehículos basado en el número de paradas y restricciones."""
    n_stops = len(dist_m) - 1
    if n_stops <= 0:
        return 1
    max_stops_per_vehicle = math.ceil(n_stops / vehs) + 2
    recommended_vehs = math.ceil(n_stops / max_stops_per_vehicle)
    if balance:
        recommended_vehs = max(recommended_vehs, math.ceil(n_stops * balance_threshold / max_stops_per_vehicle))
    return min(vehs, max(1, recommended_vehs))

def fetch_fuel_price(fuel_type: str) -> float:
    """Devuelve el precio del combustible por defecto."""
    default_prices = {'diesel': 1.376, 'gasolina': 1.467}
    return default_prices.get(fuel_type, 1.376)

def calculate_kpis(
    plan: List[List[int]],
    dist_m: List[List[float]],
    time_m: List[List[int]],
    df_today: pd.DataFrame,
    price_per_hour: Optional[float] = None,
    fuel_price: Optional[float] = None,
    fuel_consumption: Optional[float] = None,
    extra_max: Optional[float] = None,
    extra_price: Optional[float] = None
) -> Tuple[pd.DataFrame, float, float]:
    """Calcula KPIs: km/ruta, km/pedido, euro/ruta, euro/pedido."""
    kpi_data = []
    total_distance = 0
    total_cost = 0
    total_stops = 0
    
    if not plan:
        st.error("El plan de rutas está vacío. Verifica los datos de entrada.")
        return pd.DataFrame(), 0, None
    
    for v, route in enumerate(plan):
        # Valores por defecto
        route_distance = 0.0
        route_time = 0.0
        route_cost = 0.0
        extra_h = 0.0

        if len(route) <= 2:  # Sólo depósito → saltar
            continue

        try:
            # Distancia (km) y tiempo (h)
            route_distance = sum(
                dist_m[route[i]][route[i + 1]] / 1000
                for i in range(len(route) - 1)
            )
            route_time = sum(
                time_m[route[i]][route[i + 1]]
                for i in range(len(route) - 1)
            ) / 60.0

            stops = len(route) - 2
            if stops == 0:
                continue

            # Coste de combustible
            if fuel_price and fuel_consumption:
                fuel_cost = (route_distance / 100) * fuel_consumption * fuel_price
                route_cost += fuel_cost

            # Coste de conductor (horas regulares + extra)
            if price_per_hour and price_per_hour > 0:
                extra_h = max(route_time - MAX_H_REG, 0)
                if extra_max is not None:
                    extra_h = min(extra_h, extra_max)
                driver_cost = (min(route_time, MAX_H_REG) * price_per_hour +
                               extra_h * (extra_price or 0))
                route_cost += driver_cost

            # Aviso si se supera el máximo permitido
            if extra_max is not None and route_time > MAX_H_REG + extra_max + 1e-6:
                st.warning(
                    f"Ruta Vehículo {v}: {route_time:.1f} h supera el límite "
                    f"({MAX_H_REG + extra_max} h) – considera añadir más vehículos."
                )

            # KPI por vehículo
            kpi_data.append({
                "Vehículo": f"Vehículo {v}",
                "km_ruta": round(route_distance, 2),
                "euro_ruta": round(route_cost, 2) if route_cost else None,
                "paradas": stops,
                "horas_totales": round(route_time, 2),
                "horas_extra": round(extra_h, 2)
            })

            total_distance += route_distance
            total_cost += route_cost
            total_stops += stops

        except IndexError as e:
            st.error(f"Error al procesar ruta Vehículo {v}: {e}")
            continue
    
    kpi_df = pd.DataFrame(kpi_data)
    km_per_order = total_distance / total_stops if total_stops > 0 else 0
    euro_per_order = total_cost / total_stops if total_stops > 0 and total_cost > 0 else None
    
    if kpi_df.empty:
        st.error("No se generaron KPIs: las rutas pueden estar vacías o no contener paradas válidas.")
    else:
        st.info(f"Procesadas {len(kpi_data)} rutas válidas con {total_stops} paradas totales.")
    
    return kpi_df, km_per_order, euro_per_order

def plot_kpi_bars(df: pd.DataFrame, column: str, title: str, y_label: str) -> alt.Chart:
    """Genera un gráfico de barras para un KPI específico."""
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Vehículo:N', title='Vehículo'),
        y=alt.Y(f'{column}:Q', title=y_label),
        color=alt.Color('Vehículo:N', legend=None)
    ).properties(
        title=title,
        width=400,
        height=300
    )
    return chart

def validate_routes(routes: List[List[int]], depot: int, df_today: pd.DataFrame) -> bool:
    """Valida que cada parada (excepto el depósito) se visite exactamente una vez."""
    non_depot_stops = set()
    for v, route in enumerate(routes):
        for node in route[1:-1]:
            if node in non_depot_stops:
                st.error(f"Error: La parada {df_today.at[node, 'DIRECCION']} (índice {node}) está asignada a múltiples rutas (Vehículo {v} y otro).")
                return False
            non_depot_stops.add(node)
    expected_stops = set(range(1, len(df_today)))
    missing_stops = expected_stops - non_depot_stops
    assigned_stops = non_depot_stops
    if missing_stops:
        st.error(f"Error: Las siguientes paradas no están asignadas: {[df_today.at[i, 'DIRECCION'] for i in missing_stops]}")
        st.info(f"Paradas asignadas: {[df_today.at[i, 'DIRECCION'] for i in assigned_stops]}")
        st.info(f"Rutas generadas: {routes}")
        return False
    return True

# =============================
# Streamlit application
# =============================

def main():
    st.set_page_config(page_title="Route Planner", page_icon=":truck:", layout="wide")
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "plan": None,
            "eta": None,
            "links": None,
            "map": None,
            "api": "",
            "dist_m_global": None,
            "time_m_global": None,
            "coords_hash": None,
            "fuel_price": None,
            "fuel_price_source": "default"
        }

    st.title("🛣️ Optimizador de Rutas Logísticas")
    st.markdown("Planifica rutas eficientes para tus vehículos con OpenRouteService.")

    with st.sidebar:
        st.header("⚙️ Configuración")
        
        with st.expander("📂 Carga de Datos", expanded=True):
            mode = st.selectbox("Fuente de datos", ["Automática", "Subir archivos"], key="mode")
            
            def load_any(fh) -> pd.DataFrame:
                ext = Path(fh.name).suffix.lower()
                try:
                    if ext == ".xlsx":
                        return pd.read_excel(fh)
                    sample = fh.read(1024).decode("utf-8", errors="ignore")
                    fh.seek(0)
                    sep = ";" if sample.count(";") > sample.count(",") else ","
                    return pd.read_csv(fh, sep=sep, encoding="utf-8")
                except Exception as e:
                    st.error(f"Error al cargar {fh.name}: {e}")
                    return pd.DataFrame()

            if mode == "Automática":
                cli_p, rta_p = Path("clientes.xlsx"), Path("ruta.xlsx")
                if not cli_p.exists() or not rta_p.exists():
                    st.warning("Coloca clientes.xlsx y ruta.xlsx en la carpeta o usa 'Subir archivos'.")
                    return
                df_cli, df_rta = pd.read_excel(cli_p), pd.read_excel(rta_p)
            else:
                up_cli = st.file_uploader("Maestro clientes (xlsx/csv)", key="upload_cli")
                up_rta = st.file_uploader("Rutas del día (xlsx/csv)", key="upload_rta")
                if not up_cli or not up_rta:
                    st.info("Sube ambos ficheros para continuar.")
                    return
                df_cli, df_rta = load_any(up_cli), load_any(up_rta)
                if df_cli.empty or df_rta.empty:
                    return

        with st.expander("🛠️ Configuración de Rutas"):
            dir_candidates = [c for c in df_cli.columns if "DIRE" in c or "ADDRESS" in c]
            default_idx = dir_candidates.index("DIRECCION") if "DIRECCION" in dir_candidates else 0
            addr_col_cli = st.selectbox("Columna dirección en CLIENTES", dir_candidates, index=default_idx, key="addr_col")
            df_cli["DIRECCION_CLEAN"] = df_cli[addr_col_cli].apply(clean_addr)
            df_rta["DIRECCION_CLEAN"] = df_rta["DIRECCION"].apply(clean_addr)

            df_today = df_rta.merge(df_cli, on="DIRECCION_CLEAN", how="left", suffixes=("_r", ""))

            required_cols = ["LATITUD", "LONGITUD", "DIRECCION"]
            missing_cols = [col for col in required_cols if col not in df_today.columns]
            if missing_cols:
                st.error(f"Faltan columnas en los datos: {', '.join(missing_cols)}")
                return

            for col in required_cols:
                if col in ["LATITUD", "LONGITUD"]:
                    df_today[col] = pd.to_numeric(df_today[col].astype(str).str.replace(",", "."), errors="coerce")
            if df_today[["LATITUD", "LONGITUD"]].isna().any().any():
                st.error("Hay paradas sin coordenadas válidas en el maestro.")
                st.dataframe(df_today[df_today["LATITUD"].isna() | df_today["LONGITUD"].isna()][["DIRECCION", "LATITUD", "LONGITUD"]])
                return

            invalid_coords = df_today[~df_today.apply(lambda row: validate_coords(row["LATITUD"], row["LONGITUD"]), axis=1)]
            if not invalid_coords.empty:
                st.error("Coordenadas inválidas encontradas:")
                st.dataframe(invalid_coords[["DIRECCION", "LATITUD", "LONGITUD"]])
                return

            duplicates = df_today[df_today["DIRECCION_CLEAN"].duplicated(keep=False)]
            if not duplicates.empty:
                st.warning("Direcciones duplicadas encontradas en los datos:")
                st.dataframe(duplicates[["DIRECCION", "DIRECCION_CLEAN", "LATITUD", "LONGITUD"]])
                df_today = df_today.drop_duplicates(subset=["DIRECCION_CLEAN"], keep="first").reset_index(drop=True)
                st.info("Se eliminaron duplicados, manteniendo la primera aparición.")

            depot_address = st.selectbox("Selecciona la dirección del almacén", df_today["DIRECCION"].unique(), key="depot")
            depot_idx = df_today[df_today["DIRECCION"] == depot_address].index[0]
            df_today = pd.concat([df_today.loc[[depot_idx]], df_today.drop(depot_idx)]).reset_index(drop=True)

            if "RUTA" in df_today.columns:
                df_today.loc[0, "RUTA"] = pd.NA
                rutas_distintas = (
                    df_today.loc[df_today.index > 0, "RUTA"]
                    .dropna()
                    .unique()
                )
                rutas_predef = len(rutas_distintas)
            else:
                rutas_predef = 0

            usar_rutas_existentes = False
            if "RUTA" in df_today.columns:
                usar_rutas_existentes = st.checkbox("Respetar rutas predefinidas del Excel (columna 'RUTA')", key="usar_rutas")
                reassign_stops = st.checkbox("Permitir reasignar paradas cercanas entre rutas", key="reassign_stops")
                reassign_distance = 5000.0
                if reassign_stops:
                    reassign_distance = st.number_input(
                        "Distancia máxima para reasignar paradas (km)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, key="reassign_distance"
                    ) * 1000
                if usar_rutas_existentes:
                    non_depot_rows = df_today.iloc[1:]
                    if non_depot_rows["RUTA"].isna().any():
                        st.error("La columna 'RUTA' contiene valores nulos en paradas que no son el depósito.")
                        return

            if usar_rutas_existentes:
                vehs_default = rutas_predef or 1
                st.info(f"Se detectaron **{vehs_default}** rutas predefinidas; se usará ese número de furgonetas.")
                vehs = vehs_default
                min_vehs = vehs
            else:
                min_vehs = 1
                vehs = st.slider("Furgonetas máximas disponibles", min_value=1, max_value=10, value=1, key="Vmax")

            balance = st.checkbox("Balancear rutas (minimiza la ruta más larga)", value=True, key="balance")
            start_time = st.time_input("Hora de salida", value=datetime.time(8, 0), key="hora")
            start_time_minutes = start_time.hour * 60 + start_time.minute
            service_time = st.number_input("Tiempo de servicio por parada (min)", min_value=0, value=10, key="service_time")
            balance_threshold = st.slider("Umbral de balanceo (%)", 50, 100, 90, key="balance_threshold") / 100
            recomendar = st.checkbox("Recomendar número óptimo de vehículos", value=True, key="recomendar")

        with st.expander("💶 Costes"):
            price_per_hour = st.number_input(
                "Sueldo por hora del repartidor (€/hora)", min_value=0.0, value=0.0, step=0.1, key="price_per_hour", 
                help="Dejar en 0 si no deseas incluir."
            )
            extra_max = st.number_input(
                "Máx. horas extra por conductor", min_value=0.0, max_value=12.0, step=0.5, value=1.0, key="extra_max_h"
            )
            extra_price = st.number_input(
                "Precio de la hora extra (€/h)", min_value=0.0, step=0.1, value=price_per_hour * 1.5, key="extra_price_h"
            )
            max_minutes = int((MAX_H_REG + extra_max) * 60)  # Minutos

            fuel_type = st.selectbox("Tipo de combustible", ["Diésel", "Gasolina"], key="fuel_type")
            fuel_type_key = 'diesel' if fuel_type == "Diésel" else 'gasolina'
            default_prices = {'diesel': 1.376, 'gasolina': 1.467}
            
            fuel_price = st.number_input(
                "Precio del combustible (€/litro)", min_value=0.0, 
                value=st.session_state["state"]["fuel_price"] or default_prices[fuel_type_key], 
                step=0.01, key="fuel_price", help="Ingresa un valor para sobrescribir el por defecto."
            )
            st.session_state["state"]["fuel_price"] = fuel_price
            st.session_state["state"]["fuel_price_source"] = "user" if fuel_price > 0 else "default"
            
            fuel_consumption = st.number_input(
                "Consumo de combustible (litros/100 km)", min_value=0.0, value=0.0, step=0.1, key="fuel_consumption", 
                help="Dejar en 0 si no deseas incluir."
            )
            st.info(f"Precio del {fuel_type.lower()} por defecto: {default_prices[fuel_type_key]:.3f} €/litro")

        with st.expander("🔑 API"):
            api_key = st.text_input("ORS API-Key", value=st.session_state["state"]["api"], type="password", key="api")
            st.session_state["state"]["api"] = api_key
            if st.button("Probar clave ORS", key="test_ors_key"):
                test_ors_matrix(api_key)

    st.header("📊 Datos Cargados")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Maestro de Clientes")
        st.dataframe(df_cli, use_container_width=True)
    with col2:
        st.subheader("Rutas del Día")
        st.dataframe(df_rta, use_container_width=True)

    if st.sidebar.button("🚚 Calcular Rutas", key="calcular"):
        if not api_key:
            st.error("Introduce una ORS API-Key válida.")
            return
        if not validate_ors_key(api_key):
            st.error("Clave de ORS API inválida para Directions API.")
            return

        with st.spinner("Calculando rutas óptimas..."):
            def resolver(df_seccion, assigned_vehs: int, start_time_minutes: int, service_time: int, fuel_price: float, fuel_consumption: float, price_per_hour: float, max_minutes: int):
                if len(df_seccion) < 2:
                    st.warning(f"Grupo con menos de 2 paradas, omitiendo: {df_seccion['DIRECCION'].tolist()}")
                    return None, None, None, None, None
                coords = list(zip(df_seccion["LATITUD"], df_seccion["LONGITUD"]))
                dist_m, time_m = ors_matrix_chunk(coords, api_key, block=10)
                if not dist_m:
                    st.error("Fallo al obtener la matriz de distancias para el grupo.")
                    return None, None, None, None, None
                n_stops = len(coords) - 1
                v = min(assigned_vehs, n_stops)
                if recomendar:
                    v = max(1, recomendar_num_vehiculos(
                        dist_m, time_m, v, 0, balance, start_time_minutes, service_time, balance_threshold
                    ))
                    st.info(f"Número recomendado de vehículos para grupo: {v}")
                max_stops_per_vehicle = math.ceil(n_stops / v) + 2
                routes, eta, assigned_vehicles = solve_vrp_simple(
                    dist_m=dist_m,
                    time_m=time_m,
                    vehs=v,
                    depot=0,
                    balance=balance,
                    start_min=start_time_minutes,
                    service_time=service_time,
                    max_stops_per_vehicle=max_stops_per_vehicle,
                    fuel_price=fuel_price,
                    fuel_consumption=fuel_consumption,
                    price_per_hour=price_per_hour,
                    max_minutes=max_minutes
                )
                if not routes:
                    st.warning(f"No se encontraron rutas válidas para el grupo: {df_seccion['DIRECCION'].tolist()}")
                return routes, eta, assigned_vehicles, dist_m, time_m

            coords_global = list(zip(df_today["LATITUD"], df_today["LONGITUD"]))
            coords_hash = hash(str(coords_global))
            dist_m_global = None
            time_m_global = None
            
            if usar_rutas_existentes:
                if st.session_state["state"]["coords_hash"] == coords_hash:
                    dist_m_global = st.session_state["state"]["dist_m_global"]
                    time_m_global = st.session_state["state"]["time_m_global"]
                else:
                    dist_m_global, time_m_global = ors_matrix_chunk(coords_global, api_key, block=10)
                    if not dist_m_global:
                        st.error("Fallo al obtener la matriz de distancias global.")
                        return
                    st.session_state["state"]["dist_m_global"] = dist_m_global
                    st.session_state["state"]["time_m_global"] = time_m_global
                    st.session_state["state"]["coords_hash"] = coords_hash

                total_stops = len(df_today) - 1
                groups = df_today.iloc[1:].groupby("RUTA", dropna=False)
                group_sizes = {name: len(group) for name, group in groups}
                total_groups = len(group_sizes)
                
                if total_groups > vehs:
                    st.warning(f"El número de rutas predefinidas ({total_groups}) excede las furgonetas disponibles ({vehs}). Combinando rutas.")
                    sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1])
                    groups_to_combine = total_groups - vehs
                    combined_groups = {}
                    group_data = dict(groups)
                    seen_indices = set()
                    for i in range(groups_to_combine):
                        if len(sorted_groups) >= 2:
                            name1, size1 = sorted_groups.pop(0)
                            name2, size2 = sorted_groups.pop(0)
                            new_name = f"{name1}+{name2}"
                            group1 = group_data[name1]
                            group2 = group_data[name2]
                            combined = pd.concat([group1, group2])
                            combined = combined.drop_duplicates(subset=["DIRECCION_CLEAN"], keep="first")
                            combined_groups[new_name] = combined
                            sorted_groups.append((new_name, len(combined)))
                            sorted_groups.sort(key=lambda x: x[1])
                    group_sizes = {name: len(group) for name, group in combined_groups.items()}
                    group_sizes.update({name: size for name, size in sorted_groups if name not in combined_groups})
                    groups = list(combined_groups.items()) + [(name, group_data[name]) for name, _ in sorted_groups if name not in combined_groups]

                if len(group_sizes) < min_vehs:
                    st.warning(f"El número de rutas ({len(group_sizes)}) es menor que el mínimo de furgonetas ({min_vehs}). Procediendo con {len(group_sizes)} rutas.")

                total_available_vehs = min(vehs, len(group_sizes))
                min_vehicles_per_group = max(1, min_vehs // total_available_vehs)
                remaining_vehs = total_available_vehs - min_vehicles_per_group * total_available_vehs
                vehicle_assignments = {name: min_vehicles_per_group for name, _ in group_sizes.items()}
                for name, _ in sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)[:remaining_vehs]:
                    vehicle_assignments[name] += 1

                rutas_grupo = []
                def process_group(nombre_ruta, grupo):
                    assigned_vehs = vehicle_assignments.get(nombre_ruta, 1)
                    global_indices = grupo.index.tolist()
                    depot_row = df_today.iloc[[0]]
                    grupo_reset = pd.concat([depot_row, grupo]).reset_index(drop=True)
                    original_indices = [0] + global_indices
                    if len(grupo_reset) < 2:
                        return []
                    routes, eta, assigned_vehicles, dist_m, time_m = resolver(
                        grupo_reset, assigned_vehs, start_time_minutes, service_time,
                        fuel_price, fuel_consumption, price_per_hour, max_minutes
                    )
                    if not routes:
                        return []
                    eta_mapped = [None] * len(df_today)
                    for loc_idx, et in enumerate(eta):
                        if et is None:
                            continue
                        gidx = original_indices[loc_idx]
                        eta_mapped[gidx] = et
                    return [[nombre_ruta, v, r, eta_mapped, original_indices] for v, r in enumerate(routes)]

                with ThreadPoolExecutor() as executor:
                    results = executor.map(lambda x: process_group(*x), groups)
                    for result in results:
                        rutas_grupo.extend(result)

                if not rutas_grupo:
                    st.error("No se pudieron resolver rutas para ningún grupo.")
                    return
                plan = []
                eta_global = [None] * len(df_today)
                total_assigned_vehicles = 0
                dist_m_global = st.session_state["state"]["dist_m_global"]
                time_m_global = st.session_state["state"]["time_m_global"]
                seen_stops = set()
                for nombre_ruta, v, r, eta_mapped, original_indices in rutas_grupo:
                    global_route = []
                    for node in r:
                        global_idx = original_indices[node]
                        if global_idx != 0 and global_idx in seen_stops:
                            continue
                        global_route.append(global_idx)
                        if global_idx != 0:
                            seen_stops.add(global_idx)
                    if len(global_route) > 2:
                        plan.append(global_route)
                        total_assigned_vehicles += 1
                        for idx, eta_val in enumerate(eta_mapped):
                            if eta_val is not None and idx < len(eta_global) and idx not in seen_stops:
                                eta_global[idx] = eta_val
                if total_assigned_vehicles > vehs:
                    st.error(f"Se asignaron {total_assigned_vehicles} vehículos, excede el límite de {vehs}.")
                    return
                if total_assigned_vehicles < min_vehs:
                    st.warning(f"Se asignaron {total_assigned_vehicles} vehículos, menos que el mínimo ({min_vehs}). Procediendo con la solución.")
                if reassign_stops and dist_m_global:
                    st.info(f"Reasignando paradas cercanas con distancia máxima {reassign_distance/1000} km")
                    plan, eta_global, assigned_vehicles = reassign_nearby_stops(
                        plan, dist_m_global, time_m_global, 0, balance, start_time_minutes, service_time,
                        max_stops_per_vehicle=math.ceil(total_stops / vehs) + 2,
                        max_distance_m=reassign_distance,
                        fuel_price=fuel_price,
                        fuel_consumption=fuel_consumption,
                        price_per_hour=price_per_hour,
                        max_minutes=max_minutes
                    )
                    if assigned_vehicles > vehs:
                        st.error(f"La reasignación resultó en {assigned_vehicles} vehículos, excede el límite de {vehs}.")
                        return
                    if assigned_vehicles < min_vehs:
                        st.warning(f"La reasignación usó {assigned_vehicles} vehículos, menos que el mínimo ({min_vehs}). Procediendo con la solución.")
                eta_global = recompute_etas(plan, time_m_global, start_time_minutes, service_time, len(df_today))
                kpi_df, km_per_order, euro_per_order = calculate_kpis(
                    plan, dist_m_global, time_m_global, df_today,
                    price_per_hour if price_per_hour > 0 else None,
                    fuel_price if fuel_price > 0 else st.session_state["state"]["fuel_price"],
                    fuel_consumption if fuel_consumption > 0 else None,
                    extra_max=extra_max,
                    extra_price=extra_price
                )
                st.session_state["state"].update(
                    plan=plan, 
                    eta=eta_global,
                    dist_m_global=dist_m_global, 
                    time_m_global=time_m_global, 
                    coords_hash=coords_hash
                )
            else:
                coords = list(zip(df_today["LATITUD"], df_today["LONGITUD"]))
                dist_m_global, time_m_global = ors_matrix_chunk(coords, api_key, block=10)
                if not dist_m_global:
                    st.error("Fallo al obtener la matriz de distancias.")
                    return
                n_stops = len(coords) - 1
                fixed_hours = 2
                fixed_cents = int(round(fixed_hours * price_per_hour * 100))

                resultados = []
                for v in range(min_vehs, vehs + 1):
                    if recomendar:
                        recommended_v = max(min_vehs, recomendar_num_vehiculos(
                            dist_m_global, time_m_global, v, 0, balance, start_time_minutes, service_time, balance_threshold
                        ))
                        if v < recommended_v:
                            continue
                    plan, eta, used = solve_vrp_simple(
                        dist_m=dist_m_global,
                        time_m=time_m_global,
                        vehs=v,
                        depot=0,
                        balance=balance,
                        start_min=start_time_minutes,
                        service_time=service_time,
                        max_stops_per_vehicle=math.ceil(n_stops / v) + 2,
                        fuel_price=fuel_price,
                        fuel_consumption=fuel_consumption,
                        price_per_hour=price_per_hour,
                        max_minutes=max_minutes
                    )
                    if not plan:
                        continue
                    kpi_df, km_per_order, euro_per_order = calculate_kpis(
                        plan, dist_m_global, time_m_global, df_today,
                        price_per_hour if price_per_hour > 0 else None,
                        fuel_price if fuel_price > 0 else st.session_state["state"]["fuel_price"],
                        fuel_consumption if fuel_consumption > 0 else None,
                        extra_max=extra_max,
                        extra_price=extra_price
                    )
                    coste_variable = kpi_df["euro_ruta"].sum() if kpi_df["euro_ruta"].notna().any() else 0
                    coste_fijo = v * fixed_cents / 100
                    coste_total = coste_fijo + coste_variable
                    resultados.append((v, coste_total, plan, kpi_df, eta, km_per_order, euro_per_order))

                if not resultados:
                    st.error("No se encontraron rutas válidas para ningún número de vehículos.")
                    return

                opt_v, opt_coste, opt_plan, opt_kpi, eta_global, km_per_order, euro_per_order = min(resultados, key=lambda t: t[1])
                assigned_vehicles = len([rt for rt in opt_plan if len(rt) > 2])
                st.success(f"Nº óptimo de furgonetas: {opt_v}  —  Coste total: {opt_coste:,.2f} €")

                df_curve = pd.DataFrame(
                    [(v, c) for v, c, *_ in resultados],
                    columns=["Vehículos", "Coste_total €"]
                )
                cost_chart = alt.Chart(df_curve).mark_line(point=True).encode(
                    x="Vehículos:O",
                    y="Coste_total €:Q"
                ).properties(
                    width=400, 
                    height=250, 
                    title="Coste total vs número de furgonetas"
                )
                st.altair_chart(cost_chart, use_container_width=True)

                st.session_state["state"].update(
                    plan=opt_plan, 
                    eta=eta_global,
                    dist_m_global=dist_m_global, 
                    time_m_global=time_m_global, 
                    coords_hash=coords_hash
                )
                if assigned_vehicles == 1 and opt_v > 1:
                    st.warning(
                        "⚠️ Solo se asignó 1 vehículo a pesar de tener más disponibles. "
                        "Considera aumentar el 'Umbral de balanceo', activar 'Balancear rutas', o reducir el 'Tiempo de servicio por parada'."
                    )

            if not st.session_state["state"]["plan"]:
                st.error("No se generaron rutas válidas.")
                return

            if not validate_routes(st.session_state["state"]["plan"], 0, df_today):
                st.error("Se detectaron paradas duplicadas o faltantes en las rutas. Revisa los datos de entrada y ajusta las restricciones.")
                return

            fmap = folium.Map(location=df_today[["LATITUD", "LONGITUD"]].mean().tolist(), zoom_start=10, tiles="OpenStreetMap", width=1200, height=800)
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"] * 5
            links = []
            for v, rt in enumerate(st.session_state["state"]["plan"]):
                color = palette[v % len(palette)]
                for i in range(len(rt) - 1):
                    orig = (df_today.at[rt[i], "LATITUD"], df_today.at[rt[i], "LONGITUD"])
                    dest = (df_today.at[rt[i + 1], "LATITUD"], df_today.at[rt[i + 1], "LONGITUD"])
                    pts = get_polyline_ors(orig, dest, api_key)
                    if pts:
                        folium.PolyLine(pts, color=color, weight=4).add_to(fmap)
                for seq, node in enumerate(rt):
                    eta_val = st.session_state["state"]["eta"][node] if node < len(st.session_state["state"]["eta"]) and st.session_state["state"]["eta"][node] is not None else None
                    eta_str = f"{eta_val // 60:02d}:{eta_val % 60:02d}" if eta_val is not None else "N/A"
                    folium.CircleMarker(
                        location=(df_today.at[node, "LATITUD"], df_today.at[node, "LONGITUD"]),
                        radius=6 if seq == 0 else 4,
                        color=color,
                        fill=True,
                        popup=f"V{v}·{seq} {df_today.at[node, 'DIRECCION']} ETA {eta_str}",
                    ).add_to(fmap)
                wps = [df_today.at[n, "DIRECCION"] for n in rt[1:-1]]
                for chunk in range(0, len(wps), 23):
                    seq = wps[chunk:chunk + 23]
                    url = "https://www.google.com/maps/dir/" + "/".join(
                        [df_today.at[rt[0], "DIRECCION"]] + seq + [df_today.at[rt[0], "DIRECCION"]]
                    ).replace(" ", "+")
                    links.append({"Vehículo": v, "Link": url})
            st.session_state["state"]["map"] = fmap
            st.session_state["state"]["links"] = pd.DataFrame(links)

            paradas_por_vehiculo = [len(r) - 2 for r in st.session_state["state"]["plan"] if len(r) > 2]
            if paradas_por_vehiculo:
                max_paradas = max(paradas_por_vehiculo)
                infractores = [p for p in paradas_por_vehiculo if p < max_paradas / 3]
                if infractores:
                    usados = len(paradas_por_vehiculo)
                    sugerido = max(1, usados - len(infractores))
                    st.warning(
                        f"⚠️ {len(infractores)} furgoneta(s) tienen menos de 1/3 de las paradas de la furgoneta más cargada."
                        f"💡 Prueba con **{sugerido} furgoneta(s)** en lugar de {usados} y vuelve a ejecutar para mejorar el equilibrio."
                    )

    state = st.session_state["state"]
    if state["plan"]:
        st.header("📈 Resultados de la Planificación")
        
        st.subheader("Indicadores Clave (KPIs)")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        total_stops = sum(len(rt) - 2 for rt in state["plan"])
        col_kpi1.metric("Paradas Asignadas", total_stops)
        col_kpi2.metric("Furgonetas Utilizadas", len([rt for rt in state["plan"] if len(rt) > 2]))
        col_kpi3.metric("Kilómetros por Pedido", f"{km_per_order:.2f} km" if km_per_order > 0 else "N/A")
        if euro_per_order is not None:
            col_kpi4.metric("Coste por Pedido", f"{euro_per_order:.2f} €")
        else:
            col_kpi4.warning("Coste por pedido no calculado: introduce sueldo por hora y/o consumo de combustible.")

        if not opt_kpi.empty:
            st.subheader("Análisis por Vehículo")
            st.dataframe(opt_kpi, use_container_width=True)
            
            col_chart1, col_chart2, col_chart3 = st.columns(3)
            with col_chart1:
                km_chart = plot_kpi_bars(opt_kpi, "km_ruta", "Kilómetros por Ruta", "Kilómetros")
                st.altair_chart(km_chart, use_container_width=True)
            
            with col_chart2:
                if opt_kpi["euro_ruta"].notna().any():
                    euro_chart = plot_kpi_bars(opt_kpi, "euro_ruta", "Coste por Ruta", "Euros")
                    st.altair_chart(euro_chart, use_container_width=True)
                else:
                    st.warning("Coste por ruta no calculado: introduce sueldo por hora y/o consumo de combustible.")
            
            with col_chart3:
                if opt_kpi["horas_extra"].notna().any():
                    extra_hours_chart = plot_kpi_bars(opt_kpi, "horas_extra", "Horas Extra por Ruta", "Horas Extra")
                    st.altair_chart(extra_hours_chart, use_container_width=True)
                else:
                    st.warning("Horas extra no calculadas: introduce sueldo por hora y máximo de horas extra.")

        st.subheader("🚛 Rutas Asignadas")
        rows = []
        for v, rt in enumerate(state["plan"]):
            for seq, node in enumerate(rt[1:-1], 1):
                eta_val = state["eta"][node] if node < len(state["eta"]) and state["eta"][node] is not None else None
                eta_str = f"{eta_val // 60:02d}:{eta_val % 60:02d}" if eta_val is not None else "N/A"
                rows.append({
                    "Vehículo": v,
                    "Secuencia": seq,
                    "Dirección": df_today.at[node, "DIRECCION"],
                    "ETA": eta_str
                })
        df_routes = pd.DataFrame(rows)
        st.dataframe(df_routes, use_container_width=True)

        excel_buffer = io.BytesIO()
        df_routes.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Descargar Planificación (Excel)",
            data=excel_buffer,
            file_name=f"planificacion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if state["links"] is not None and not state["links"].empty:
            st.subheader("🔗 Links para Conductores")
            st.dataframe(state["links"], use_container_width=True)

        if state["map"]:
            st.subheader("🗺️ Mapa de Rutas")
            st_folium(state["map"], width=1200, height=800, returned_objects=[])

            html_buffer = io.BytesIO()
            state["map"].save(html_buffer, close_file=False)
            html_buffer.seek(0)
            st.download_button(
                label="📥 Descargar Mapa (HTML)",
                data=html_buffer,
                file_name=f"mapa_rutas_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )

if __name__ == "__main__":
    main()