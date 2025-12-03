"""Streamlit dashboard for pedestrian counting and analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Pedestrian Counting & Service Level Analysis",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .los-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        color: white;
    }
    .los-a { background-color: #2e8b57; }
    .los-b { background-color: #32cd32; }
    .los-c { background-color: #ffd700; color: black; }
    .los-d { background-color: #ff8c00; }
    .los-e { background-color: #ff4500; }
    .los-f { background-color: #dc143c; }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=30)
def fetch_data(endpoint: str) -> Optional[Dict]:
    """Fetch data from API with caching."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from {endpoint}: {e}")
        return None

def get_los_color(los_level: str) -> str:
    """Get color class for LOS level."""
    return f"los-{los_level.lower()}"

def format_los_display(los_level: str) -> str:
    """Format LOS level for display."""
    return f'<span class="los-indicator {get_los_color(los_level)}">{los_level}</span>'

def create_density_gauge(density: float, los_level: str):
    """Create a gauge chart for density."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = density,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Density (peds/m²)"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 2.0]},
            'bar': {'color': get_los_color(los_level).replace('los-', '#')},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1.1], 'color': "orange"},
                {'range': [1.1, 1.5], 'color': "red"},
                {'range': [1.5, 2.0], 'color': "darkred"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.5
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_time_series_chart(data: List[Dict], title: str):
    """Create time series chart."""
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.line(df, x='timestamp', y='density', 
                  title=title, labels={'density': 'Density (peds/m²)'})
    fig.update_layout(height=400)
    return fig

# Main dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🚶 Pedestrian Counting & Service Level Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Camera selection
    cameras_data = fetch_data("/cameras")
    if cameras_data:
        camera_options = {f"Camera {cam['id']:02d}": cam['id'] for cam in cameras_data}
        selected_camera = st.sidebar.selectbox("Select Camera", list(camera_options.keys()))
        camera_id = camera_options[selected_camera]
    else:
        camera_id = 1
        st.sidebar.warning("Using default camera (API unavailable)")
    
    # Time range selection
    time_range = st.sidebar.selectbox("Time Range", 
                                     ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Custom"])
    
    if time_range == "Custom":
        start_time = st.sidebar.datetime_input("Start Time", 
                                             value=datetime.now() - timedelta(hours=6))
        end_time = st.sidebar.datetime_input("End Time", value=datetime.now())
    else:
        hours = {"Last Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24}[time_range]
        start_time = datetime.now() - timedelta(hours=hours)
        end_time = datetime.now()
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-time Monitoring", "📈 Historical Analysis", 
                                     "🚨 Alerts & Status", "⚙️ Configuration"])
    
    with tab1:
        st.header("Real-time Monitoring")
        
        # Get current counts and LOS
        counts_data = fetch_data(f"/counts?camera_id={camera_id}")
        los_data = fetch_data(f"/los?camera_id={camera_id}")
        
        if counts_data and los_data:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_peds = counts_data[str(camera_id)]['current_pedestrians']
                st.metric("Current Pedestrians", current_peds)
            
            with col2:
                ingress = counts_data[str(camera_id)]['ingress_count']
                st.metric("Ingress (5min)", ingress)
            
            with col3:
                egress = counts_data[str(camera_id)]['egress_count']
                st.metric("Egress (5min)", egress)
            
            with col4:
                net_count = counts_data[str(camera_id)]['net_count']
                st.metric("Net Count", net_count, delta=net_count)
            
            # Level of Service
            st.subheader("Level of Service Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                los_level = los_data[str(camera_id)]['avg_los']
                density = los_data[str(camera_id)]['avg_density']
                
                st.markdown(f"**Current LOS:** {format_los_display(los_level)}", 
                           unsafe_allow_html=True)
                st.markdown(f"**Density:** {density:.2f} pedestrians/m²")
                
                # LOS description
                los_descriptions = {
                    "A": "Free flow - pedestrians can move freely without conflicts",
                    "B": "Reasonably free flow - minor conflicts possible",
                    "C": "Stable flow - some conflicts and queuing",
                    "D": "Approaching unstable flow - frequent conflicts",
                    "E": "Unstable flow - frequent conflicts and queuing",
                    "F": "Forced flow - breakdown of flow, queuing"
                }
                st.info(f"**Description:** {los_descriptions.get(los_level, 'Unknown')}")
            
            with col2:
                fig_gauge = create_density_gauge(density, los_level)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        else:
            st.error("Unable to fetch real-time data")
    
    with tab2:
        st.header("Historical Analysis")
        
        # Get historical data
        historical_data = fetch_data(f"/analysis/historical/{camera_id}?start_time={start_time.isoformat()}&end_time={end_time.isoformat()}")
        
        if historical_data:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Density", f"{historical_data['avg_density']:.2f} peds/m²")
            
            with col2:
                st.metric("Peak Density", f"{historical_data['peak_density']:.2f} peds/m²")
            
            with col3:
                st.metric("Total Ingress", historical_data['total_ingress'])
            
            with col4:
                st.metric("Total Egress", historical_data['total_egress'])
            
            # LOS comparison
            st.subheader("Level of Service Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Average LOS:** {format_los_display(historical_data['avg_los'])}", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Peak LOS:** {format_los_display(historical_data['peak_los'])}", 
                           unsafe_allow_html=True)
            
            # Peak hour factor
            st.metric("Peak Hour Factor", f"{historical_data['peak_hour_factor']:.2f}")
            
            # Sample time series (would be real data in production)
            st.subheader("Density Over Time")
            
            # Generate sample data for demonstration
            sample_data = []
            current_time = start_time
            while current_time <= end_time:
                # Simulate realistic density variation
                base_density = 0.3 + 0.2 * np.sin(2 * np.pi * current_time.hour / 24)
                noise = np.random.normal(0, 0.1)
                density = max(0, base_density + noise)
                
                sample_data.append({
                    'timestamp': current_time.isoformat(),
                    'density': density
                })
                current_time += timedelta(minutes=5)
            
            fig_time = create_time_series_chart(sample_data, "Pedestrian Density Over Time")
            if fig_time:
                st.plotly_chart(fig_time, use_container_width=True)
        
        else:
            st.error("Unable to fetch historical data")
    
    with tab3:
        st.header("Alerts & System Status")
        
        # System status
        status_data = fetch_data("/status")
        
        if status_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_color = "🟢" if status_data['is_running'] else "🔴"
                st.metric("Processing Status", f"{status_color} {'Running' if status_data['is_running'] else 'Stopped'}")
            
            with col2:
                st.metric("Online Cameras", f"{status_data['online_cameras']}/{status_data['total_cameras']}")
            
            with col3:
                st.metric("System Uptime", "99.9%")  # Would be calculated from real data
            
            # Camera status table
            st.subheader("Camera Status")
            
            camera_status_df = pd.DataFrame([
                {"Camera": f"Camera {i:02d}", "Status": status, "Last Update": datetime.now().strftime("%H:%M:%S")}
                for i, status in status_data['camera_status'].items()
            ])
            
            st.dataframe(camera_status_df, use_container_width=True)
            
            # Recent alerts (sample data)
            st.subheader("Recent Alerts")
            
            sample_alerts = [
                {"Time": "14:32:15", "Camera": "Camera 03", "Type": "High Density", "Severity": "Medium", "Message": "LOS D detected"},
                {"Time": "14:28:42", "Camera": "Camera 07", "Type": "Camera Offline", "Severity": "High", "Message": "Connection lost"},
                {"Time": "14:15:33", "Camera": "Camera 01", "Type": "High Density", "Severity": "Low", "Message": "LOS C detected"},
            ]
            
            alerts_df = pd.DataFrame(sample_alerts)
            st.dataframe(alerts_df, use_container_width=True)
        
        else:
            st.error("Unable to fetch system status")
    
    with tab4:
        st.header("Configuration")
        
        # LOS thresholds
        st.subheader("Fruin's Level of Service Thresholds")
        
        los_summary = fetch_data("/los/summary")
        if los_summary:
            los_df = pd.DataFrame([
                {
                    "Level": level,
                    "Density Range": data["density_range"],
                    "Description": data["description"],
                    "Comfort": data["comfort"],
                    "Flow": data["flow"]
                }
                for level, data in los_summary.items()
            ])
            
            st.dataframe(los_df, use_container_width=True)
        
        # Zone configuration
        st.subheader("Detection Zones")
        
        if st.button("Configure Zones"):
            st.info("Zone configuration interface would be implemented here")
        
        # Camera settings
        st.subheader("Camera Settings")
        
        if st.button("Camera Management"):
            st.info("Camera management interface would be implemented here")
        
        # System settings
        st.subheader("System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Analysis Window (minutes)", value=5, min_value=1, max_value=60)
            st.number_input("Confidence Threshold", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
        
        with col2:
            st.number_input("Max Tracking Distance", value=100, min_value=50, max_value=200)
            st.number_input("Max Disappeared Frames", value=30, min_value=10, max_value=100)

if __name__ == "__main__":
    main()