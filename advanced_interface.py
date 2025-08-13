import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import simplekml
import time
import math
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import io
import warnings
import zipfile
import tempfile
import os
from openpyxl import Workbook
from io import BytesIO
import random
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
import re
import openrouteservice
from geopy.distance import geodesic
import webbrowser

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="Bus Route Optimizer", layout="wide")

# Title and description
st.title("🚌 Bus Route Optimization System")
st.markdown("This system optimizes bus routes using fixed bus allocation with detailed timing calculations.")

# Initialize session state
if 'routes' not in st.session_state:
    st.session_state.routes = None
    st.session_state.optimized = False
    st.session_state.depot_lat = 35.72509
    st.session_state.depot_lon = 10.75339
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'removed_rows' not in st.session_state:
    st.session_state.removed_rows = None
if 'manual_override' not in st.session_state:
    st.session_state.manual_override = False
if 'bus_capacities' not in st.session_state:
    st.session_state.bus_capacities = [28]
if 'initial_optimization_done' not in st.session_state:
    st.session_state.initial_optimization_done = False
if 'ors_key' not in st.session_state:
    st.session_state.ors_key = ""
if 'ors_visualization' not in st.session_state:
    st.session_state.ors_visualization = None
if 'segment_speeds' not in st.session_state:
    st.session_state.segment_speeds = {}  # To store speeds between segments
if 'station_service_times' not in st.session_state:
    st.session_state.station_service_times = {}  # To store service times per station

# ADD THESE LINES TO FIX THE ERROR:
if 'default_speed' not in st.session_state:
    st.session_state.default_speed = 40  # Default speed in km/h
if 'default_service_time' not in st.session_state:
    st.session_state.default_service_time = 2  # Default service time in minutes
if 'school_arrival_time' not in st.session_state:
    st.session_state.school_arrival_time = "08:00"


# Color palette
route_colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige',
    'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue',
    'lightgreen', 'gray', 'black', 'lightgray'
]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = (sin(dLat/2) * sin(dLat/2) +
         cos(radians(lat1)) * cos(radians(lat2)) *
         sin(dLon/2) * sin(dLon/2))
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def calculate_route_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += haversine_distance(
            route.iloc[i]['Latitude'], route.iloc[i]['Longitude'],
            route.iloc[i+1]['Latitude'], route.iloc[i+1]['Longitude'])
    if len(route) > 0:
        last = route.iloc[-1]
        total_distance += haversine_distance(last['Latitude'], last['Longitude'],
                                           st.session_state.depot_lat, st.session_state.depot_lon)
    return total_distance

# Improved nearest neighbor algorithm with depot proximity consideration
def find_nearest_station(current_station, remaining_stations):
    current_coords = (current_station['Latitude'], current_station['Longitude'])
    current_to_depot = haversine_distance(current_coords[0], current_coords[1],
                                        st.session_state.depot_lat, st.session_state.depot_lon)

    min_score = float('inf')
    nearest = None

    for _, station in remaining_stations.iterrows():
        # Distance to next station
        dist_to_station = haversine_distance(current_coords[0], current_coords[1],
                                           station['Latitude'], station['Longitude'])
        # Distance from next station to depot
        dist_to_depot = haversine_distance(station['Latitude'], station['Longitude'],
                                         st.session_state.depot_lat, st.session_state.depot_lon)

        # Weighted score (60% distance to station, 40% reduction in depot distance)
        score = 0.6 * dist_to_station + 0.4 * max(0, current_to_depot - dist_to_depot)

        if score < min_score:
            min_score = score
            nearest = station

    return nearest

def two_opt_swap(route_df, neighborhood_size=3):
    if len(route_df) < 4:
        return route_df

    best_route = route_df.copy()
    best_distance = calculate_route_distance(best_route)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, min(i+neighborhood_size+1, len(best_route)-1)):
                if j - i == 1:
                    continue  # No point swapping adjacent points

                # Calculate the distance change without full recomputation
                old_dist = (haversine_distance(best_route.iloc[i-1]['Latitude'], best_route.iloc[i-1]['Longitude'],
                                             best_route.iloc[i]['Latitude'], best_route.iloc[i]['Longitude']) +
                          haversine_distance(best_route.iloc[j]['Latitude'], best_route.iloc[j]['Longitude'],
                                           best_route.iloc[j+1]['Latitude'], best_route.iloc[j+1]['Longitude']))

                new_dist = (haversine_distance(best_route.iloc[i-1]['Latitude'], best_route.iloc[i-1]['Longitude'],
                                             best_route.iloc[j]['Latitude'], best_route.iloc[j]['Longitude']) +
                          haversine_distance(best_route.iloc[i]['Latitude'], best_route.iloc[i]['Longitude'],
                                           best_route.iloc[j+1]['Latitude'], best_route.iloc[j+1]['Longitude']))

                if new_dist < old_dist:
                    # Create new route by reversing the segment between i and j
                    new_route = pd.concat([
                        best_route.iloc[:i],
                        best_route.iloc[i:j+1].iloc[::-1],
                        best_route.iloc[j+1:]
                    ])

                    best_route = new_route
                    best_distance = best_distance - old_dist + new_dist
                    improved = True
                    break  # Restart the search after improvement
            if improved:
                break

    return best_route

def optimize_cluster_order(cluster_df):
    if cluster_df.empty:
        return cluster_df

    # Start with farthest point from depot
    farthest_idx = cluster_df['distance_from_depot'].idxmax()
    route = [cluster_df.loc[farthest_idx]]
    remaining = cluster_df.drop(farthest_idx)

    while not remaining.empty:
        nearest = find_nearest_station(route[-1], remaining)
        route.append(nearest)
        remaining = remaining.drop(nearest.name)

    # apply 2-opt optimization
    optimized_route = pd.DataFrame(route)
    optimized_route = two_opt_swap(optimized_route, neighborhood_size=3)

    return optimized_route

def compute_required_buses(demands, capacities):
    if not capacities:
        return []

    capacities = sorted([int(c) for c in capacities], reverse=True)
    total_demand = sum(int(round(float(d))) for d in demands)

    buses = []
    remaining_demand = total_demand

    while remaining_demand > 0:
        assigned = False
        for cap in capacities:
            if remaining_demand >= cap * 0.7:  # Minimum 70% filling rate
                buses.append(cap)
                remaining_demand -= cap
                assigned = True
                break

        if not assigned:  # Add smallest bus if no good fit
            buses.append(min(capacities))
            remaining_demand = max(0, remaining_demand - min(capacities))

    return buses

def sweep_fixed_buses(data, available_buses):
    routes = []
    data_copy = data.copy()
    idx = 0

    # First pass - original optimization (completely isolated)
    for bus_cap in available_buses:
        current_route = []
        current_capacity = 0

        while idx < len(data_copy):
            demand = data_copy.loc[idx, 'Description']
            if current_capacity + demand <= bus_cap:
                current_route.append(data_copy.loc[idx])
                current_capacity += demand
                idx += 1
            else:
                break

        if current_route:
            cluster_df = pd.DataFrame(current_route)
            optimized = optimize_cluster_order(cluster_df)
            routes.append({
                'route': optimized,
                'capacity_used': current_capacity,
                'bus_capacity': bus_cap,
                'distance': calculate_route_distance(optimized)
            })

    # Second pass - completely independent re-optimization
    if idx < len(data_copy):
        remaining_stations = data_copy.iloc[idx:].copy()


        # 1. Prepare fresh data for isolated optimization
        remaining_stations[['theta', 'distance']] = remaining_stations.apply(
            lambda row: polar_coordinates(row), axis=1)
        remaining_stations = remaining_stations.sort_values('theta')
        remaining_stations['distance_from_depot'] = remaining_stations.apply(
            lambda row: haversine_distance(row['Latitude'], row['Longitude'],
                                        st.session_state.depot_lat, st.session_state.depot_lon),
            axis=1
        )

        # 2. Calculate fresh bus requirements
        remaining_demand = remaining_stations['Description'].sum()
        new_buses = []

        # Calculate using same logic as initial optimization
        capacities = sorted(st.session_state.bus_capacities, reverse=True)
        for cap in capacities:
            while remaining_demand >= cap * 0.7:  # Minimum 70% utilization
                new_buses.append(cap)
                remaining_demand -= cap
        if remaining_demand > 0:
            new_buses.append(min(st.session_state.bus_capacities))

        # 3. Run completely independent optimization
        if new_buses:
            additional_routes = []

            # Fresh sweep algorithm on remaining stations only
            reopt_idx = 0
            for bus_cap in new_buses:
                current_route = []
                current_capacity = 0

                while reopt_idx < len(remaining_stations):
                    demand = remaining_stations.iloc[reopt_idx]['Description']
                    if current_capacity + demand <= bus_cap:
                        current_route.append(remaining_stations.iloc[reopt_idx])
                        current_capacity += demand
                        reopt_idx += 1
                    else:
                        break

                if current_route:
                    cluster_df = pd.DataFrame(current_route)
                    optimized = optimize_cluster_order(cluster_df)
                    additional_routes.append({
                        'route': optimized,
                        'capacity_used': current_capacity,
                        'bus_capacity': bus_cap,
                        'distance': calculate_route_distance(optimized)
                    })

            routes.extend(additional_routes)


    return routes

def clean_data(data):

    original_count = len(data)

    # Make a copy of the original data
    cleaned = data.copy()

    # Remove rows with missing coordinates
    cleaned = cleaned.dropna(subset=['Latitude', 'Longitude'])

    # Remove rows where Description is 0 or non-numeric
    cleaned = cleaned[pd.to_numeric(cleaned['Description'], errors='coerce').notnull()]
    cleaned['Description'] = cleaned['Description'].astype(float)
    cleaned = cleaned[cleaned['Description'] > 0]

    # Remove duplicate stations (same coordinates)
    cleaned = cleaned.drop_duplicates(subset=['Latitude', 'Longitude'])

    removed_count = original_count - len(cleaned)
    removed_data = data[~data.index.isin(cleaned.index)]

    return cleaned, removed_data, removed_count

def generate_override_template(routes):
    try:
        template_data = []
        for route_idx, route in enumerate(routes, 1):
            route_df = route['route']
            for i, (_, station) in enumerate(route_df.iterrows(), 1):
                template_data.append({
                    'station_name': station['Nom'],
                    'station_designation': station['designation station'],
                    'required_route': route_idx,
                    'latitude': station['Latitude'],
                    'longitude': station['Longitude'],
                    'passengers': station['Description'],
                    'order_in_route': i
                })
        return pd.DataFrame(template_data)
    except Exception as e:
        st.error(f"Error generating template: {str(e)}")
        return None

def apply_manual_overrides(override_df, original_data, bus_capacities):
    try:
        required_columns = ['station_name', 'required_route', 'order_in_route']
        if not all(col in override_df.columns for col in required_columns):
            st.error("Override file missing required columns")
            return None

        route_groups = override_df.groupby('required_route')
        new_routes = []

        for route_num, group in route_groups:
            group_sorted = group.sort_values('order_in_route')
            route_stations = []
            capacity_used = 0

            for _, row in group_sorted.iterrows():
                station_match = original_data[original_data['Nom'] == row['station_name']]
                if len(station_match) == 0:
                    st.error(f"Station {row['station_name']} not found in original data")
                    continue

                station_data = station_match.iloc[0]
                route_stations.append(station_data)
                capacity_used += station_data['Description']

            if not route_stations:
                continue

            suitable_capacities = [cap for cap in bus_capacities if cap >= capacity_used]
            bus_capacity = min(suitable_capacities) if suitable_capacities else max(bus_capacities)

            route_df = pd.DataFrame(route_stations)
            distance = calculate_route_distance(route_df)

            new_routes.append({
                'route': route_df,
                'capacity_used': capacity_used,
                'bus_capacity': bus_capacity,
                'distance': distance
            })

        return new_routes if new_routes else None
    except Exception as e:
        st.error(f"Error applying overrides: {str(e)}")
        return None

def clean_coordinates(coords):

    unique_coords = []
    seen = set()
    for coord in coords:
        coord_tuple = tuple(coord)
        if coord_tuple not in seen:
            seen.add(coord_tuple)
            unique_coords.append(coord)
    return unique_coords

def is_detour_unnecessary(prev_point, current_point, next_point, threshold_km=0.2):

    prev = (prev_point[1], prev_point[0])
    curr = (current_point[1], current_point[0])
    nxt = (next_point[1], next_point[0])

    dist_direct = geodesic(prev, nxt).km
    dist_via = geodesic(prev, curr).km + geodesic(curr, nxt).km
    return (dist_via - dist_direct) > threshold_km

def generate_ors_visualization(routes, ors_key):

    try:
        client = openrouteservice.Client(key=ors_key)

        cleaned_clusters = []
        for route in routes:
            coords = []
            # Add depot as starting point
            coords.append([st.session_state.depot_lon, st.session_state.depot_lat])
            # Add route points
            for _, row in route['route'].iterrows():
                coords.append([row['Longitude'], row['Latitude']])
            # Add depot as ending point
            coords.append([st.session_state.depot_lon, st.session_state.depot_lat])

            if len(coords) < 3:
                cleaned_clusters.append(clean_coordinates(coords))
                continue

            cleaned = [coords[0]]
            for i in range(1, len(coords) - 1):
                if not is_detour_unnecessary(coords[i - 1], coords[i], coords[i + 1]):
                    cleaned.append(coords[i])
            cleaned.append(coords[-1])
            cleaned_clusters.append(clean_coordinates(cleaned))

        # Create map centered on depot
        m = folium.Map(location=[st.session_state.depot_lat, st.session_state.depot_lon], zoom_start=12)
        total_distance = 0
        successful_routes = 0
        route_details = []

        # Colors for routes
        colors = ['blue', 'purple', 'orange', 'darkred', 'cadetblue', 'green', 'pink', 'gray']

        # Process each route
        for i, coords in enumerate(cleaned_clusters):
            if not coords or len(coords) < 2:
                continue

            try:
                route = client.directions(
                    coordinates=coords,
                    profile='driving-car',
                    format='geojson',
                    optimize_waypoints=True,
                    instructions=True
                )

                geometry = route['features'][0]['geometry']
                distance = route['features'][0]['properties']['summary']['distance'] / 1000
                total_distance += distance
                duration = route['features'][0]['properties']['summary']['duration'] / 60
                successful_routes += 1

                route_details.append({
                    'route_number': i+1,
                    'distance_km': distance,
                    'duration_min': duration
                })

                folium.GeoJson(
                    geometry,
                    name=f'Route {i+1}',
                    style_function=lambda x, color=colors[i % len(colors)]: {
                        'color': color,
                        'weight': 5,
                        'opacity': 0.8
                    }
                ).add_to(m)

                # Add depot marker
                if i == 0:
                    folium.Marker(
                        [st.session_state.depot_lat, st.session_state.depot_lon],
                        popup='Depot',
                        icon=folium.Icon(color='black', icon='warehouse')
                    ).add_to(m)

                # Add station markers
                for j, point in enumerate(coords[1:-1]):
                    folium.CircleMarker(
                        location=[point[1], point[0]],
                        radius=4,
                        color=colors[i % len(colors)],
                        fill=True,
                        fill_color=colors[i % len(colors)],
                        fill_opacity=0.8,
                        popup=f'Stop {j+1} (Route {i+1})'
                    ).add_to(m)

                time.sleep(1)  # Respect API rate limits

            except Exception as e:
                st.error(f"Error processing Route {i+1}: {str(e)}")
                continue

        if successful_routes > 0:
            folium.LayerControl().add_to(m)
            st.session_state.ors_visualization = {
                'map': m,
                'route_details': route_details,
                'total_distance': total_distance
            }
            return True
        else:
            st.error("No routes could be calculated. Please check your data.")
            return False

    except Exception as e:
        st.error(f"Error initializing ORS client: {str(e)}")
        return False

def create_kml(route_df, route_num):
    kml = simplekml.Kml()
    depot = kml.newpoint(name="Depot",
                        coords=[(st.session_state.depot_lon, st.session_state.depot_lat)])
    depot.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/pal4/icon57.png'

    for _, row in route_df.iterrows():
        pnt = kml.newpoint(name=row['Nom'],
                          description=f"Designation: {row['designation station']}\\nPassengers: {row['Description']}",
                          coords=[(row['Longitude'], row['Latitude'])])
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/pal4/icon49.png'

    if len(route_df) > 0:
        first_station = route_df.iloc[0]
        coords = [(first_station['Longitude'], first_station['Latitude'])]
        coords.extend([(row['Longitude'], row['Latitude']) for _, row in route_df.iterrows()])
        coords.append((st.session_state.depot_lon, st.session_state.depot_lat))

        linestring = kml.newlinestring(name=f"Route {route_num}")
        linestring.coords = coords
        linestring.style.linestyle.color = simplekml.Color.red
        linestring.style.linestyle.width = 4

    return kml

def create_excel_file(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    processed_data = output.getvalue()
    return processed_data

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, 0

        # Clean the data
        cleaned_data, removed_data, removed_count = clean_data(data)

        return cleaned_data, removed_data, removed_count

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, 0

def calculate_route_timing_with_speeds(route_df, route_idx):

    timing_details = []

    # Calculate departure time based on school arrival time
    total_route_time = calculate_total_route_time(route_df)
    try:
        arrival_dt = datetime.strptime("08:00", "%H:%M")
        departure_dt = arrival_dt - timedelta(minutes=total_route_time)
        current_time = departure_dt
    except:
        current_time = datetime.now().replace(hour=6, minute=0, second=0)

    # Add depot departure
    timing_details.append({
        'station_name': "Depot",
        'station_designation': "Depot",
        'latitude': st.session_state.depot_lat,
        'longitude': st.session_state.depot_lon,
        'passengers': 0,
        'arrival_time': current_time.strftime("%H:%M"),
        'departure_time': current_time.strftime("%H:%M")
    })

    prev_point = None
    for i, (_, point) in enumerate(route_df.iterrows()):
        station_name = point['Nom']
        station_designation = point['designation station']

        # Get service time for this station
        service_time = st.session_state.station_service_times.get(station_designation, st.session_state.default_service_time)

        # Calculate distance and speed for this segment
        if i == 0:
            # From depot to first station
            distance = haversine_distance(
                st.session_state.depot_lat, st.session_state.depot_lon,
                point['Latitude'], point['Longitude']
            )
            segment_key = f"depot_to_{station_designation}"
        else:
            # From previous station to current station
            distance = haversine_distance(
                prev_point['Latitude'], prev_point['Longitude'],
                point['Latitude'], point['Longitude']
            )
            segment_key = f"{prev_point['designation station']}_to_{station_designation}"

        # Get speed for this segment
        speed = st.session_state.segment_speeds.get(segment_key, st.session_state.default_speed)

        # Calculate travel time in minutes
        travel_time = (distance / speed) * 60 if speed > 0 else 0

        # Arrival time is departure from previous + travel time
        arrival_time = current_time + timedelta(minutes=travel_time)

        # Departure time is arrival time + service time
        departure_time = arrival_time + timedelta(minutes=service_time)

        timing_details.append({
            'station_name': station_name,
            'station_designation': station_designation,
            'latitude': point['Latitude'],
            'longitude': point['Longitude'],
            'passengers': int(point['Description']),
            'arrival_time': arrival_time.strftime("%H:%M"),
            'departure_time': departure_time.strftime("%H:%M")
        })

        current_time = departure_time
        prev_point = point

    # Add return to depot if there are stations
    if len(route_df) > 0:
        last_point = route_df.iloc[-1]
        distance = haversine_distance(
            last_point['Latitude'], last_point['Longitude'],
            st.session_state.depot_lat, st.session_state.depot_lon
        )
        segment_key = f"{last_point['designation station']}_to_depot"
        speed = st.session_state.segment_speeds.get(segment_key, st.session_state.default_speed)
        travel_time = (distance / speed) * 60 if speed > 0 else 0
        arrival_time = current_time + timedelta(minutes=travel_time)

        timing_details.append({
            'station_name': "Depot",
            'station_designation': "Depot",
            'latitude': st.session_state.depot_lat,
            'longitude': st.session_state.depot_lon,
            'passengers': 0,
            'arrival_time': arrival_time.strftime("%H:%M"),
            'departure_time': arrival_time.strftime("%H:%M")
        })

    return pd.DataFrame(timing_details)

def calculate_total_route_time(route_df):

    total_time = 0
    prev_point = None

    for i, (_, point) in enumerate(route_df.iterrows()):
        station_designation = point['designation station']

        # Add service time
        service_time = st.session_state.station_service_times.get(station_designation, st.session_state.default_service_time)
        total_time += service_time

        # Add travel time
        if i == 0:
            # From depot to first station
            distance = haversine_distance(
                st.session_state.depot_lat, st.session_state.depot_lon,
                point['Latitude'], point['Longitude']
            )
            segment_key = f"depot_to_{station_designation}"
        else:
            # From previous station to current station
            distance = haversine_distance(
                prev_point['Latitude'], prev_point['Longitude'],
                point['Latitude'], point['Longitude']
            )
            segment_key = f"{prev_point['designation station']}_to_{station_designation}"

        speed = st.session_state.segment_speeds.get(segment_key, st.session_state.default_speed)
        travel_time = (distance / speed) * 60 if speed > 0 else 0
        total_time += travel_time

        prev_point = point

    # Add return to depot time
    if len(route_df) > 0:
        last_point = route_df.iloc[-1]
        distance = haversine_distance(
            last_point['Latitude'], last_point['Longitude'],
            st.session_state.depot_lat, st.session_state.depot_lon
        )
        segment_key = f"{last_point['designation station']}_to_depot"
        speed = st.session_state.segment_speeds.get(segment_key, st.session_state.default_speed)
        travel_time = (distance / speed) * 60 if speed > 0 else 0
        total_time += travel_time

    return total_time
def create_individual_route_map(route_df, route_num):
    if route_df.empty:
        return None

    # Create map centered on first station
    first_stop = route_df.iloc[0]
    m = folium.Map(
        location=[first_stop['Latitude'], first_stop['Longitude']],
        zoom_start=14,
        tiles='OpenStreetMap'
    )

    # Add depot marker (RED)
    folium.Marker(
        [st.session_state.depot_lat, st.session_state.depot_lon],
        popup="DEPOT (END)",
        icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
    ).add_to(m)

    # Create route path (first station -> ... -> depot)
    route_coords = []

    # 1. Start at first station
    route_coords.append([first_stop['Latitude'], first_stop['Longitude']])

    # 2. Add remaining stations
    for idx, row in route_df.iloc[1:].iterrows():
        route_coords.append([row['Latitude'], row['Longitude']])

        # Add station marker (BLUE)
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=f"{row['Nom']} (Stop {idx+1})",
            icon=folium.Icon(color='blue', icon='bus', prefix='fa')
        ).add_to(m)

    # 3. End at depot
    route_coords.append([st.session_state.depot_lat, st.session_state.depot_lon])

    # Draw the route path (GREEN line)
    folium.PolyLine(
        locations=route_coords,
        color='green',
        weight=6,
        opacity=0.8,
        tooltip=f"Route {route_num}"
    ).add_to(m)

    # Highlight first station differently (GREEN)
    folium.Marker(
        [first_stop['Latitude'], first_stop['Longitude']],
        popup=f"START: {first_stop['Nom']}",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)

    return m

def polar_coordinates(row):
    dx = (row['Longitude'] - st.session_state.depot_lon) * 111.320 * cos(radians(row['Latitude']))
    dy = (row['Latitude'] - st.session_state.depot_lat) * 110.574
    theta = np.arctan2(dy, dx)
    theta = theta if theta >= 0 else theta + 2 * np.pi
    return pd.Series({'theta': theta, 'distance': sqrt(dx**2 + dy**2)})

# UI Components
with st.sidebar:
    st.header("Configuration Parameters")
    depot_options = {
        "Jemmal": {"lat": 35.63154, "lon": 10.72993},
        "Sousse": {"lat": 35.78768, "lon": 10.66554},
        "Silana": {"lat": 36.09991, "lon": 9.35762},
        "El Jem": {"lat": 35.33615, "lon": 10.68279},
        "Zaouiet Sousse": {"lat": 35.79356, "lon": 10.62718},
        "Sahline": {"lat": 35.72509, "lon": 10.75339}
    }
    selected_depot = st.selectbox("Select Depot", list(depot_options.keys()))
    st.session_state.depot_lat = depot_options[selected_depot]["lat"]
    st.session_state.depot_lon = depot_options[selected_depot]["lon"]

    st.header("Data Upload")
    station_file = st.file_uploader("Upload Stations Data (CSV or Excel)", type=["csv", "xlsx"])

    st.header("Bus Parameters")
    bus_capacities_str = st.text_input("Bus Capacities (comma separated)", "28")
    st.session_state.bus_capacities = [int(cap.strip()) for cap in bus_capacities_str.split(",") if cap.strip().isdigit()]


    if st.button("Run Optimization") and station_file:
        st.session_state.optimized = True
        st.session_state.manual_override = False
        st.session_state.initial_optimization_done = True
        st.rerun()

# Main app logic
if station_file:
    try:
        # Load, validate and clean data
        cleaned_data, removed_data, removed_count = load_data(station_file)

        if cleaned_data is None:
            st.stop()

        st.session_state.original_data = cleaned_data.copy()
        st.session_state.cleaned_data = cleaned_data
        st.session_state.removed_rows = removed_data

        # Show data cleaning results
        if removed_count > 0:
            st.warning(f"Removed {removed_count} invalid rows (missing coordinates or zero passengers)")
            with st.expander("Show removed rows"):
                st.dataframe(removed_data)

        # Data preparation
        data_sorted = cleaned_data.copy()
        data_sorted[['theta', 'distance']] = data_sorted.apply(polar_coordinates, axis=1)
        data_sorted = data_sorted.sort_values('theta').reset_index(drop=True)
        data_sorted['distance_from_depot'] = data_sorted.apply(
            lambda row: haversine_distance(row['Latitude'], row['Longitude'],
                                          st.session_state.depot_lat, st.session_state.depot_lon),
            axis=1
        )

        # Perform optimization if needed
        if st.session_state.optimized and not st.session_state.manual_override:
           with st.spinner("Processing routes..."):
                available_buses = compute_required_buses(data_sorted['Description'], st.session_state.bus_capacities)
                routes = sweep_fixed_buses(data_sorted, available_buses)
                st.session_state.routes = routes

        # Results analysis
        if st.session_state.routes:
            total_distance = sum(route['distance'] for route in st.session_state.routes)
            total_buses = len(st.session_state.routes)
            total_passengers = sum(route['capacity_used'] for route in st.session_state.routes)
            total_capacity = sum(route['bus_capacity'] for route in st.session_state.routes)
            average_filling = (total_passengers / total_capacity) * 100 if total_capacity > 0 else 0
            used_buses = [route['bus_capacity'] for route in st.session_state.routes]

            # Display summary
            st.subheader("Route Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Routes", total_buses)
            col2.metric("Total Distance (km)", f"{total_distance:.2f}")
            col3.metric("Average Filling Rate", f"{average_filling:.1f}%")
            col4.metric("Used Buses", ", ".join(map(str, used_buses)))

            # Visualization of all routes
            st.subheader("All Routes Map")
            m = folium.Map(location=[st.session_state.depot_lat, st.session_state.depot_lon], zoom_start=12)

            # Add depot marker
            folium.Marker(
                [st.session_state.depot_lat, st.session_state.depot_lon],
                popup="Depot",
                icon=folium.Icon(color='black', icon='warehouse', prefix='fa')
            ).add_to(m)

            marker_cluster = MarkerCluster().add_to(m)
            for _, row in cleaned_data.iterrows():
                folium.Marker(
                    [row['Latitude'], row['Longitude']],
                    popup=f"<b>{row['Nom']}</b><br>Demand: {row['Description']}",
                    icon=folium.Icon(color='gray', icon='bus', prefix='fa')
                ).add_to(marker_cluster)

            for i, route in enumerate(st.session_state.routes):
                color = route_colors[i % len(route_colors)]
                route_df = route['route']
                route_coords = [[row['Latitude'], row['Longitude']] for _, row in route_df.iterrows()]
                route_coords.append([st.session_state.depot_lat, st.session_state.depot_lon])

                folium.PolyLine(
                    locations=route_coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Route {i+1}",
                    tooltip=f"Route {i+1}"
                ).add_to(m)

                for j, (_, point) in enumerate(route_df.iterrows(), 1):
                    folium.Marker(
                        [point['Latitude'], point['Longitude']],
                        popup=f"<b>Route {i+1}, Stop {j}</b><br>{point['Nom']}<br>Demand: {point['Description']}",
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(m)

            folium.LayerControl().add_to(m)
            title_html = '''
                <h3 align="center" style="font-size:16px"><b>Bus Route Optimization - Fixed Bus Plan</b></h3>
                <p align="center">Depot: black icon | Routes: colored lines | Stops: gray markers</p>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            folium_static(m, width=1200)

            # Detailed route information
            st.subheader("Detailed Route Information")
            if st.session_state.manual_override:
                st.info("⚠️ Manual overrides are currently applied to these routes")
            else:
                st.info("✅ These are the optimized routes")

            for i, route in enumerate(st.session_state.routes, 1):
                with st.expander(f"Route {i} Details"):
                    # Route statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        filling_rate = (route['capacity_used']/route['bus_capacity'])*100
                        html_content = f'''
                        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">
                            <h3 style="color:#2b5876;margin-top:0;">Route {i} Statistics</h3>
                            <p style="font-size:18px;margin-bottom:5px;"><strong>Bus Capacity:</strong> {route['bus_capacity']} passengers</p>
                            <p style="font-size:18px;margin-bottom:5px;"><strong>Passengers Assigned:</strong> {route['capacity_used']}</p>
                            <p style="font-size:18px;margin-bottom:5px;"><strong>Filling Rate:</strong> {filling_rate:.1f}%</p>
                        </div>
                        '''
                        st.markdown(html_content, unsafe_allow_html=True)

                    with col2:
                        total_route_time = calculate_total_route_time(route['route'])
                        html_content = f'''
                        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">
                            <h3 style="color:#2b5876;margin-top:0;">Distance & Time</h3>
                            <p style="font-size:18px;margin-bottom:5px;"><strong>Total Distance:</strong> {route['distance']:.2f} km</p>
                            <p style="font-size:18px;margin-bottom:5px;"><strong>Number of Stops:</strong> {len(route['route'])}</p>
                            <p style="font-size:18px;margin-bottom:0;"><strong>Total Route Time:</strong> {total_route_time:.1f} min</p>
                        </div>
                        '''
                        st.markdown(html_content, unsafe_allow_html=True)

                    st.markdown("### Route Map Visualization")

                    # Get the route dataframe
                    route_df = route['route'].copy()

                    # Ensure coordinates are float type
                    route_df['Latitude'] = route_df['Latitude'].astype(float)
                    route_df['Longitude'] = route_df['Longitude'].astype(float)

                    # Display the map
                    single_route_map = create_individual_route_map(route_df, i)
                    if single_route_map:
                        folium_static(single_route_map, width=700, height=400)
                    else:
                        st.warning("Could not generate map for this route")
                    st.markdown("**Segment Speeds and Service Times**")

                    # Default speed and service time configuration
                    col1, col2 = st.columns(2)
                    with col1:
                        new_default_speed = st.number_input(
                            f"Default Speed for Route {i} (km/h)",
                            min_value=10, max_value=100,
                            value=st.session_state.default_speed,
                            key=f"default_speed_{i}"
                        )

                    with col2:
                        new_default_service = st.number_input(
                            f"Default Service Time for Route {i} (min)",
                            min_value=1, max_value=10,
                            value=st.session_state.default_service_time,
                            key=f"default_service_{i}"
                        )

                    # Segment speed inputs
                    st.markdown("**Segment Speeds (km/h):**")
                    route_df = route['route']

                    # Depot to first station
                    if len(route_df) > 0:
                        first_station_name = route_df.iloc[0]['Nom']
                        first_station_designation = route_df.iloc[0]['designation station']
                        segment_key = f"depot_to_{first_station_designation}"
                        current_speed = st.session_state.segment_speeds.get(segment_key, new_default_speed)
                        new_speed = st.number_input(
                            f"Depot to {first_station_designation}",  # Show name in UI
                            min_value=10, max_value=100, value=current_speed,
                            key=f"speed_{segment_key}"  # Use designation in key
                        )
                        st.session_state.segment_speeds[segment_key] = new_speed

                    # Between stations
                    for j in range(len(route_df)-1):
                        from_station_name = route_df.iloc[j]['Nom']
                        to_station_name = route_df.iloc[j+1]['Nom']
                        from_designation = route_df.iloc[j]['designation station']
                        to_designation = route_df.iloc[j+1]['designation station']
                        segment_key = f"{from_designation}_to_{to_designation}"
                        current_speed = st.session_state.segment_speeds.get(segment_key, new_default_speed)
                        new_speed = st.number_input(
                            f"{from_designation} to {to_designation}",  # Show names in UI
                            min_value=10, max_value=100, value=current_speed,
                            key=f"speed_{segment_key}"  # Use designations in key
                        )
                        st.session_state.segment_speeds[segment_key] = new_speed


                    # Last station to depot
                    if len(route_df) > 0:
                        last_station_name = route_df.iloc[-1]['Nom']
                        last_station_designation = route_df.iloc[-1]['designation station']
                        segment_key = f"{last_station_designation}_to_depot"
                        current_speed = st.session_state.segment_speeds.get(segment_key, new_default_speed)
                        new_speed = st.number_input(
                            f"{last_station_designation} to Depot",  # Show name in UI
                            min_value=10, max_value=100, value=current_speed,
                            key=f"speed_{segment_key}"  # Use designation in key
                        )
                        st.session_state.segment_speeds[segment_key] = new_speed
                    # Station service time inputs
                    st.markdown("**Station Service Times (minutes):**")
                    cols = st.columns(3)
                    for j, (_, station) in enumerate(route_df.iterrows()):
                        station_name = station['Nom']
                        station_designation = station['designation station']
                        current_service = st.session_state.station_service_times.get(station_designation, new_default_service)
                        new_service = cols[j%3].number_input(
                            f"{station_designation}",  # Show name in UI
                            min_value=1, max_value=10, value=current_service,
                            key=f"service_{station_designation}"  # Use designation in key
                        )
                        st.session_state.station_service_times[station_designation] = new_service

                    # Calculate and display timing
                    timing_df = calculate_route_timing_with_speeds(route['route'], i)
                    st.markdown("**Detailed Timing Schedule:**")
                    st.dataframe(timing_df.style.format({
                        'latitude': '{:.6f}',
                        'longitude': '{:.6f}',
                        'passengers': '{:.0f}'  # Format as integer with 0 decimal places
                    }))


                    # Export options
                    st.markdown("**Export Station Data:**")
                    export_format = st.selectbox(f"Select export format for Route {i}",
                                              ["CSV", "XLSX", "KML", "KMZ"],
                                              key=f"export_format_{i}")

                    if export_format == "CSV":
                        csv = timing_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download Route {i} Schedule as CSV",
                            data=csv,
                            file_name=f'route_{i}_schedule.csv',
                            mime='text/csv',
                            key=f"csv_{i}"
                        )
                    elif export_format == "XLSX":
                        excel_data = create_excel_file(timing_df)
                        st.download_button(
                            label=f"Download Route {i} Schedule as Excel",
                            data=excel_data,
                            file_name=f'route_{i}_schedule.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key=f"excel_{i}"
                        )
                    elif export_format == "KML":
                        kml = create_kml(route['route'], i)
                        kml_data = kml.kml().encode('utf-8')
                        st.download_button(
                            label=f"Download Route {i} as KML",
                            data=kml_data,
                            file_name=f'route_{i}_stations.kml',
                            mime='application/vnd.google-earth.kml+xml',
                            key=f"kml_{i}"
                        )
                    elif export_format == "KMZ":
                        kml = create_kml(route['route'], i)
                        with tempfile.NamedTemporaryFile(suffix='.kmz', delete=False) as tmpfile:
                            kmz_path = tmpfile.name
                            kml.savekmz(kmz_path)
                            with open(kmz_path, 'rb') as f:
                                kmz_data = f.read()
                            st.download_button(
                                label=f"Download Route {i} as KMZ",
                                data=kmz_data,
                                file_name=f'route_{i}_stations.kmz',
                                mime='application/vnd.google-earth.kmz',
                                key=f"kmz_{i}"
                            )
                        os.unlink(kmz_path)

            # Manual Override Section
            st.subheader("Manual Route Adjustments")
            template_df = generate_override_template(st.session_state.routes)

            if template_df is not None:
                # Create Excel template download button
                excel_template = create_excel_file(template_df)
                st.download_button(
                    label="Download Override Template (Excel)",
                    data=excel_template,
                    file_name='route_override_template.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

                # Upload modified Excel file
                uploaded_override = st.file_uploader("Upload Modified Routes (Excel)", type=['xlsx'])

                if uploaded_override:
                    # Read the Excel file
                    override_df = pd.read_excel(uploaded_override)
                    st.write("Preview of uploaded overrides:")
                    st.dataframe(override_df, hide_index=True)

                    if st.button("Apply Manual Overrides"):
                        new_routes = apply_manual_overrides(override_df, st.session_state.original_data, st.session_state.bus_capacities)
                        if new_routes is not None:
                            st.session_state.routes = new_routes
                            st.session_state.manual_override = True
                            st.session_state.optimized = False
                            st.success("Manual overrides applied successfully!")
                            st.rerun()

                if st.session_state.manual_override:
                    st.warning("Manual overrides are currently applied.")
                    if st.button("Reset to Optimized Routes"):
                        st.session_state.manual_override = False
                        st.session_state.optimized = True
                        st.rerun()

            # OpenRouteService Visualization Section
            st.subheader("OpenRouteService Visualization")
            st.info("Generate detailed route visualization using OpenRouteService API")

            with st.expander("OpenRouteService Configuration"):
                st.session_state.ors_key = st.text_input("Enter OpenRouteService API Key",
                                                       value=st.session_state.ors_key,
                                                       type="password")
                st.markdown("[Get an API key from OpenRouteService](https://openrouteservice.org/)")

                if st.button("Generate ORS Visualization") and st.session_state.ors_key:
                    with st.spinner("Generating detailed route visualization..."):
                        if generate_ors_visualization(st.session_state.routes, st.session_state.ors_key):
                            st.success("ORS visualization generated successfully!")
                        else:
                            st.error("Failed to generate ORS visualization")

            if st.session_state.ors_visualization:
                st.subheader("ORS Route Visualization")
                folium_static(st.session_state.ors_visualization['map'], width=1200)

                st.subheader("Route Details")
                route_details = pd.DataFrame(st.session_state.ors_visualization['route_details'])
                st.dataframe(route_details.style.format({
                    'distance_km': '{:.2f} km',
                    'duration_min': '{:.1f} min'
                }))

                st.metric("Total Distance for All Routes",
                         f"{st.session_state.ors_visualization['total_distance']:.2f} km")


    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
elif not station_file:
    st.info("Please upload the station data file to begin optimization")
else:
    st.info("Configure parameters and click 'Run Optimization'")


# Authenticate ngrok
from pyngrok import ngrok
ngrok.set_auth_token("2l23Qv46hemr0QDyTDqp302OMCd_6a4vAdh3X7fsUmdYkrWr2")


# Start ngrok tunnel
import time
time.sleep(8)

try:
    public_url = ngrok.connect(8501)
    print("\n✅ Your Streamlit app is live at:", public_url.public_url)
    print("App may take 15-30 seconds to become fully available")
except Exception as e:
    print("\n❌ Failed to create Ngrok tunnel. Error:", str(e))
    print("\nTroubleshooting steps:")
    print("1. Check Streamlit logs:")
    print("\n2. Try these commands manually:")
    print("!streamlit run bus_optimizer.py --server.port 8501")
    print("!ngrok http 8501")

