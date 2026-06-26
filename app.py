import os
import sys
import requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from geopy.geocoders import Nominatim
from src.predictor import run_live_inference

# 1. Page Configuration
st.set_page_config(
    page_title="EcoGuard Platform",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Light Professional Theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

        /* ── Base ── */
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif !important;
        }
        .stApp {
            background-color: #F0F2F5;
            color: #2D3748;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #718096 !important;
            font-size: 0.78rem !important;
            line-height: 1.7 !important;
        }
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stSubheader {
            color: #1A2332 !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }
        .sidebar-forest-item {
            display: block;
            font-size: 0.77rem;
            color: #4A5568;
            padding: 3px 0;
            border-left: 2px solid #CBD5E0;
            padding-left: 8px;
            margin-bottom: 4px;
            line-height: 1.5;
        }
        .status-line {
            font-size: 0.75rem;
            font-weight: 600;
            color: #2D6A4F;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            display: flex;
            align-items: center;
            gap: 7px;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #2D6A4F;
            display: inline-block;
            flex-shrink: 0;
        }

        /* ── Page header ── */
        .eco-header {
            background: #FFFFFF;
            border-bottom: 3px solid #2D6A4F;
            border-radius: 8px 8px 0 0;
            padding: 32px 36px 28px;
            margin-bottom: 2px;
        }
        .eco-eyebrow {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #2D6A4F;
            margin-bottom: 10px;
        }
        .eco-header h1 {
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            color: #1A2332 !important;
            letter-spacing: -0.025em;
            line-height: 1.2;
            margin: 0 0 10px !important;
        }
        .eco-header .subtitle {
            font-size: 0.875rem;
            color: #718096;
            line-height: 1.65;
            max-width: 780px;
            font-weight: 400;
        }
        .eco-header .subtitle strong {
            color: #2D6A4F;
            font-weight: 600;
        }

        /* ── Section labels / headings ── */
        h2 { color: #1A2332 !important; font-weight: 700 !important; font-size: 1.1rem !important; }
        h3 { color: #1A2332 !important; font-weight: 600 !important; font-size: 0.95rem !important; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: #FFFFFF;
            border-radius: 6px;
            padding: 4px;
            gap: 2px;
            border: 1px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: #718096 !important;
            border-radius: 4px !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            padding: 8px 20px !important;
            border: none !important;
            letter-spacing: 0.01em;
        }
        .stTabs [aria-selected="true"] {
            background: #2D6A4F !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 22px;
        }

        /* ── Selectbox & input ── */
        .stSelectbox > div > div,
        .stTextInput > div > div > input {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E0 !important;
            border-radius: 6px !important;
            color: #2D3748 !important;
            font-size: 0.875rem !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        .stSelectbox > div > div:focus-within,
        .stTextInput > div > div > input:focus {
            border-color: #2D6A4F !important;
            box-shadow: 0 0 0 3px rgba(45,106,79,0.12) !important;
        }
        .stSelectbox label, .stTextInput label {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            color: #4A5568 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        /* ── Button ── */
        .stButton > button {
            background: #2D6A4F !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
            padding: 10px 24px !important;
            letter-spacing: 0.02em;
            transition: background 0.15s;
        }
        .stButton > button:hover {
            background: #245A42 !important;
        }

        /* ── Card with left accent border (the signature) ── */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 3px solid #2D6A4F;
            border-radius: 6px;
            padding: 22px 24px;
            margin-bottom: 16px;
        }
        .card-title {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #2D6A4F;
            margin-bottom: 16px;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 9px 0;
            border-bottom: 1px solid #F0F2F5;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-label {
            font-size: 0.83rem;
            color: #718096;
            font-weight: 400;
        }
        .metric-value {
            font-family: 'DM Mono', monospace;
            font-size: 0.84rem;
            color: #1A2332;
            font-weight: 500;
        }
        .metric-value.accent {
            color: #2D6A4F;
        }

        /* ── Report header ── */
        .report-header {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 3px solid #2D6A4F;
            border-radius: 6px;
            padding: 20px 26px;
            margin-bottom: 20px;
        }
        .report-header h2 {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: #1A2332 !important;
            margin: 0 0 5px !important;
        }
        .report-coords {
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem;
            color: #718096;
            letter-spacing: 0.04em;
        }

        /* ── Alert banners ── */
        .alert-danger {
            background: #FFF5F5;
            border: 1px solid #FED7D7;
            border-left: 3px solid #C53030;
            border-radius: 6px;
            padding: 16px 22px;
            color: #C53030;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .alert-danger .alert-label {
            font-size: 0.67rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 4px;
            color: #9B2C2C;
        }
        .alert-safe {
            background: #F0FBF6;
            border: 1px solid #B7E4CC;
            border-left: 3px solid #2D6A4F;
            border-radius: 6px;
            padding: 16px 22px;
            color: #2D6A4F;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .alert-safe .alert-label {
            font-size: 0.67rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 4px;
            color: #245A42;
        }

        /* ── Section heading style ── */
        .section-heading {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #A0AEC0;
            margin: 28px 0 14px;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 8px;
        }

        /* ── Tab inner heading ── */
        .tab-heading {
            font-size: 0.9rem;
            font-weight: 600;
            color: #2D3748;
            margin-bottom: 16px;
        }

        /* ── Caption ── */
        .stCaption { color: #A0AEC0 !important; font-size: 0.78rem !important; }
        .sector-desc {
            font-size: 0.8rem;
            color: #718096;
            background: #F7FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 5px;
            padding: 10px 14px;
            margin-top: 6px;
            line-height: 1.6;
        }

        /* ── Progress bar ── */
        .stProgress > div > div {
            background-color: #2D6A4F !important;
            border-radius: 3px !important;
            height: 6px !important;
        }
        .stProgress > div {
            background-color: #E2E8F0 !important;
            border-radius: 3px !important;
            height: 6px !important;
        }

        /* ── Divider ── */
        hr { border-color: #E2E8F0 !important; margin: 24px 0 !important; }

        /* ── Spinner ── */
        .stSpinner > div { border-top-color: #2D6A4F !important; }
    </style>
""", unsafe_allow_html=True)

# 3. Forest Directory Database (unchanged)
FOREST_DATABASE = {
    "Algeria (Model Native Region)": {
        "Sidi Bel Abbés National Park": {"lat": 35.2012, "lon": -0.6317, "desc": "Dense Mediterranean pine and shrub ecosystems. High historical fire frequency."},
        "El Kala National Park": {"lat": 36.8872, "lon": 8.3542, "desc": "UNESCO biosphere reserve containing vital wetland and forest borders."},
        "Chréa Atlas Forest": {"lat": 36.4258, "lon": 2.8742, "desc": "High altitude mountainous terrain populated by ancient Atlas Cedars."}
    },
    "India": {
        "Western Ghats Reserved Forests": {"lat": 12.1565, "lon": 75.6948, "desc": "Dense tropical evergreen and moist deciduous canopy zones."},
        "Gir National Forest": {"lat": 21.1243, "lon": 70.8242, "desc": "Dry deciduous teak forests. Highly prone to surface fires during dry summers."},
        "Sundarbans Mangrove Border": {"lat": 21.9497, "lon": 89.1833, "desc": "Unique coastal ecosystem monitoring peripheral sal-forest interfaces."}
    },
    "United States": {
        "Sierra National Forest (California)": {"lat": 37.2842, "lon": -119.2789, "desc": "Heavy timber canopy. High seasonal drought vulnerability index."},
        "Tongass National Forest (Alaska)": {"lat": 56.5167, "lon": -132.3333, "desc": "Temperate rainforest monitoring localized low-risk baselines."}
    }
}

# 4. Identity Header
st.markdown("""
<div class="eco-header">
    <div class="eco-eyebrow">Forest Fire Risk Analytics</div>
    <h1>EcoGuard Platform</h1>
    <div class="subtitle">
        Enterprise monitoring powered by a <strong>Random Forest Pipeline (97.96% accuracy)</strong>,
        fused with live <strong>Open-Meteo REST streams</strong> and real-time
        <strong>Canadian Fire Weather Index (FWI)</strong> computation.
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Monitor Layout
st.sidebar.subheader("Automated Watchlist")
for country, forests in FOREST_DATABASE.items():
    st.sidebar.markdown(
        f"<span style='display:block;font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:#A0AEC0;margin:12px 0 6px;'>{country}</span>",
        unsafe_allow_html=True
    )
    for f_name in forests.keys():
        st.sidebar.markdown(
            f"<span class='sidebar-forest-item'>{f_name}</span>",
            unsafe_allow_html=True
        )

st.sidebar.write("---")
st.sidebar.markdown(
    "<div class='status-line'><span class='status-dot'></span> System Operational</div>",
    unsafe_allow_html=True
)

# 6. Navigation Tabs
tab1, tab2 = st.tabs(["Protected Forest Registry", "Global Coordinate Search"])
lat, lon, chosen_name = None, None, ""

with tab1:
    st.markdown("<div class='tab-heading'>Select a monitored sector to run analysis</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        country_select = st.selectbox("Country Jurisdiction:", options=list(FOREST_DATABASE.keys()))
    with col_b:
        forest_select = st.selectbox("Monitored Forest Sector:", options=list(FOREST_DATABASE[country_select].keys()))

    selected_meta = FOREST_DATABASE[country_select][forest_select]
    st.markdown(f"<div class='sector-desc'>{selected_meta['desc']}</div>", unsafe_allow_html=True)

    st.write("")
    if st.button("Run Sector Analysis", type="primary"):
        lat = selected_meta["lat"]
        lon = selected_meta["lon"]
        chosen_name = forest_select

with tab2:
    st.markdown("<div class='tab-heading'>Resolve any geographic location by name</div>", unsafe_allow_html=True)
    location_input = st.text_input("Location name:", placeholder="e.g., Sidi Bel Abbes, Algiers, Pune")

    st.write("")
    if st.button("Geocode and Analyse"):
        if not location_input.strip():
            st.warning("Please enter a valid location name.")
        else:
            try:
                geolocator = Nominatim(user_agent="ecoguard_fire_predictor")
                location_data = geolocator.geocode(location_input, timeout=10)
                if not location_data:
                    st.error(f"Could not resolve coordinates for '{location_input}'. Try a more specific name.")
                else:
                    lat = location_data.latitude
                    lon = location_data.longitude
                    chosen_name = location_data.address
            except Exception as e:
                st.error(f"Geocoding error: {str(e)}")

# 7. Execution and Prediction Engine
if lat is not None and lon is not None:
    st.write("---")
    with st.spinner("Fetching live weather data and computing fire indices..."):
        try:
            weather, indices, prediction, probabilities = run_live_inference(lat, lon)

            st.markdown(f"""
            <div class="report-header">
                <h2>Analysis Report — {chosen_name}</h2>
                <div class="report-coords">LAT {lat:.4f} &nbsp;&nbsp; LON {lon:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class='metric-card'>
                    <div class='card-title'>Meteorological Data</div>
                    <div class='metric-row'>
                        <span class='metric-label'>Temperature</span>
                        <span class='metric-value'>{temp} °C</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Relative Humidity (RH)</span>
                        <span class='metric-value'>{rh}%</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Wind Speed (Ws)</span>
                        <span class='metric-value'>{ws} km/h</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Precipitation (Rain)</span>
                        <span class='metric-value'>{rain} mm</span>
                    </div>
                </div>
                """.format(
                    temp=weather['temp'], rh=weather['rh'],
                    ws=weather['ws'], rain=weather['rain']
                ), unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class='metric-card'>
                    <div class='card-title'>Fire Weather Indices</div>
                    <div class='metric-row'>
                        <span class='metric-label'>Fine Fuel Moisture Code (FFMC)</span>
                        <span class='metric-value accent'>{ffmc}</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Initial Spread Index (ISI)</span>
                        <span class='metric-value accent'>{isi}</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Duff Moisture Code (DMC)</span>
                        <span class='metric-value accent'>{dmc}</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Drought Code (DC)</span>
                        <span class='metric-value accent'>{dc}</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Fire Weather Index (FWI)</span>
                        <span class='metric-value accent'>{fwi}</span>
                    </div>
                </div>
                """.format(
                    ffmc=f"{indices['FFMC']:.2f}", isi=f"{indices['ISI']:.2f}",
                    dmc=f"{indices['DMC']:.2f}", dc=f"{indices['DC']:.2f}",
                    fwi=f"{indices['FWI']:.2f}"
                ), unsafe_allow_html=True)

            st.markdown("<div class='section-heading'>ML Pipeline Evaluation</div>", unsafe_allow_html=True)

            if prediction == 1:
                confidence = probabilities[1] * 100
                st.markdown(f"""
                <div class='alert-danger'>
                    <div class='alert-label'>Critical Warning</div>
                    Fire risk detected &mdash; {confidence:.2f}% model confidence
                </div>
                """, unsafe_allow_html=True)
                st.progress(probabilities[1])
            else:
                confidence = probabilities[0] * 100
                st.markdown(f"""
                <div class='alert-safe'>
                    <div class='alert-label'>Status Nominal</div>
                    No fire risk detected &mdash; {confidence:.2f}% model confidence
                </div>
                """, unsafe_allow_html=True)
                st.progress(probabilities[1])

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")