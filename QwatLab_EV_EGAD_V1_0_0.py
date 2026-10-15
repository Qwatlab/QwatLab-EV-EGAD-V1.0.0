# QwatLab EV E.G.A.D. — validated repair build
# UI matnlari ingliz tilida; hisoblash algoritmlari, ichki izohlar va backend xabarlari o‘zbek tilida saqlanadi.
# =====================================================================
# GLOBAL KUTUBXONALAR VA MODULLAR (DEPENDENCIES)
# =====================================================================

# 1. UI LAYER (Foydalanuvchi interfeysi)
import streamlit as st

# 2. CORE MATH & KINEMATIC ENGINE (Hisoblash va filtrlar)
import numpy as np
import numba as nb
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# 3. DATA ENGINE & PARSING (Ma'lumotlarni o'qish va tahlil)
import pandas as pd
import xml.etree.ElementTree as ET
import atexit

# 4. VISUALIZATION LAYER (Grafika va xaritalar)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# 5. SYSTEM UTILITIES & EXPORT (Tizim, xotira va arxivlash)
import os
import io
import tempfile
import zipfile
import json
import gc
import re

# =====================================================================
# SAHIFA KENG EKRAN (DESKTOP) REJIMI
# =====================================================================
st.set_page_config(page_title="QwatLab: EV E.G.A.D. V1.0.2", layout="wide")

# Traektoriya davomiyligi uchun qat’iy xavfsizlik chegarasi (soniya).
MAX_TRAJECTORY_DURATION_S = 100000.0
MAX_OUTPUT_POINTS = 2_000_000
# =====================================================================
# 0-BLOK: KINEMATIK model VA MATEMATIKA
# =====================================================================
def render_block_0():
    """
    0-Blok: QwatLab: EV EGAD V1.0.2. Matematika
    va universal moddiy nuqta kinematikasi tenglamalarini o'zida jamlagan UI komponent.
    """
    with st.expander("📘 BLOCK 0: QwatLab Kinematic Parameters & Math Engine V1.0.2", expanded=True):
        
        tab_phil, tab_proj, tab_kin, tab_topo, tab_filt, tab_limits, tab_post, tab_donate = st.tabs([
            "🌐 1. Platform Overview", 
            "📐 2. Geospatial Projection", 
            "🏎️ 3. Universal Kinematics",
            "⛰️ 4. Topography", 
            "🔍 5. Filtering & ZUPT",
            "🚧 6. Validation & UI Limits",
            "📬 Contact & Support",
            "☕ Support the Project"
        ])
        
        # --- TAB 1: PLATFORMA FALSAFASI ---
        with tab_phil:
            st.markdown("### 🔬 Universal Kinematic Analysis (V1.0.2)")
            st.markdown(
                "This platform analyzes raw measurements and estimates a 3D trajectory and its kinematics. "
                "It uses a point-mass kinematic model; it does not infer vehicle dynamics, electrical, chemical, "
                "or aerodynamic forces from GPS data alone."
            )
            st.info(
                "**Purpose:** Create a consistent kinematic trajectory record from supported GPX and KML inputs. "
                "Valid coordinates are required. GPX and timestamped KML tracks can provide timestamps; standard KML coordinate lists "
                "without per-point timestamps use the configured default time step. Missing elevations are interpolated from valid neighboring values. "
                "The maximum accepted trajectory duration is hard-limited to 100,000 seconds (27 h 46 min 40 s); longer tracks are rejected. The output grid is capped at 2,000,000 samples. Input timestamps are never rewritten to force speed limits; large time gaps and physically implausible segment speeds are rejected with a diagnostic."
            )
            st.info(
                "**Configuration principle:** Analysis parameters and limits are configurable in Block 1. "
                "Assumptions and fixed physical constants are documented explicitly."
            )

        # --- TAB 2: GEOFAZOVIY PROYEKSIYA ---
        with tab_proj:
            st.markdown("#### WGS84-Based ECEF and ENU Transformation")
            st.markdown(
                "To avoid flat-Earth approximations, geographic coordinates are converted using the WGS84 ellipsoid. "
                "ECEF is used for global Cartesian calculations and ENU for local trajectory visualization."
            )
            st.markdown("**1. WGS84 Constants and ECEF ($X, Y, Z$) Calculation:**")
            st.latex(r"a = 6378137.0 \text{ (semi-major axis)}, \quad e^2 = 0.00669437999 \text{ (eccentricity squared)}")
            st.latex(r"N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2 \phi}}")
            st.latex(r"X = (N(\phi) + h) \cos \phi \cos \lambda")
            st.latex(r"Y = (N(\phi) + h) \cos \phi \sin \lambda")
            st.latex(r"Z = \left(N(\phi)(1 - e^2) + h\right) \sin \phi")
            st.caption(r"Here: $\lambda$ = longitude (Lon), $\phi$ = latitude (Lat), $h$ = elevation (Ele).")
            st.markdown(r"**2. ECEF to ENU:** A rotation matrix converts ECEF offsets to the local east-north-up frame relative to $(\phi_0, \lambda_0)$.")

        # --- TAB 3: UNIVERSAL KINEMATIKA ---
        with tab_kin:
            st.markdown("#### 3D Kinematics and Derived Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1. Kinematic State Vector:**")
                st.markdown(
                    "PCHIP resamples GPS coordinates to a uniform time grid. The Savitzky–Golay filter smooths selected derivatives. "
                    "The 9-state forward Kalman filter estimates position, velocity, and acceleration."
                )
                
            with col2:
                st.markdown(r"**2. Trajectory Heading (Standard Heading - $\psi$):**")
                st.latex(r"\psi = \operatorname{atan2}(v_{East}, v_{North})")

                st.markdown(r"**3. Trajectory Curvature ($\kappa$):**")
                st.latex(r"\kappa = \frac{v_x a_y - v_y a_x}{(v_x^2 + v_y^2)^{3/2}}")

        # --- TAB 4: TOPOGRAFIK LAR ---
        with tab_topo:
            st.markdown("#### ⛰️ Fail-Fast Topographic Validation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**1. Z-Axis Validation (Fail-Fast):**")
                st.error(
                    "Missing elevation values are interpolated between available valid samples; leading/trailing gaps use the nearest valid elevation. If all elevations are missing, processing stops. Atmospheric estimates are controlled independently from terrain-grade analysis."
                )
                
            with col2:
                st.markdown("**2. Outlier Removal:**")
                st.markdown(
                    "Abrupt GPS jumps are checked against elapsed time and the configured speed limit. Suspected isolated spikes "
                    "are removed only when the neighboring segments support that interpretation; other violations stop processing."
                )

        # --- TAB 5: FILTRLASH VA ZUPT ---
        with tab_filt:
            st.markdown("#### 🔍 9D Kalman Filter and Dynamic Matrices")
            st.markdown(
                "- **9-state vector:** The filter jointly estimates 3D position, velocity, and acceleration:"
            )
            st.latex(r"X_k = [x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]^T")
            
            st.markdown(r"- **State transition matrix ($\Phi$):**")
            st.latex(r"\Phi = \begin{bmatrix} I_3 & \Delta t I_3 & \frac{\Delta t^2}{2} I_3 \\ 0_3 & I_3 & \Delta t I_3 \\ 0_3 & 0_3 & I_3 \end{bmatrix}")
            
            st.markdown(
                "- **Process noise ($Q$) and ZUPT:** $Q$ uses a continuous white-jerk . When the configured low-speed "
                "condition is met, a zero-velocity measurement update is applied; position is not artificially frozen."
            )

        # --- TAB: ALOQA VA QO'LLAB-QUVVATLASH ---
        with tab_post:
            st.markdown("### QwatLab Team")
            st.markdown("**Communication and Support:**")
            st.markdown("* 🌐 Website: https://t.me/QwatLab")
            st.markdown("* **Made in Uzbekistan 🇺🇿**")
            st.markdown("* **Program registration certificate: No. DGU 69148**")

        # --- TAB: HOMIYLIK ---
        with tab_donate:
            st.markdown("### Support the Project")
            st.markdown("**Help improve the platform and support its development:**")
            st.markdown("* ☕ Donate: https://t.me/QwatLab")

        # --- TAB 6: VALIDATSIYA VA UI CHEGARALAR ---
        with tab_limits:
            st.markdown("#### 🚧 Input Validation and UI Limits")
            st.success(
                "**1. Time step and limits:** The default time step is configurable. The processing engine rejects tracks longer than "
                "100,000 seconds (about 27.8 hours). Plot and map point limits are controlled through the interface."
            )
            st.success(
                "**2. Kinematics:** The engine derives trajectory metrics from the available coordinates and timestamps. "
                "Derived values are estimates and should be validated against reference data before scientific use."
            )
# =====================================================================
# 0-BLOK OXIRI
# =====================================================================
# =====================================================================
# 1-BLOK: KINEMATIK LABORATORIYA SOZLAMALARI (UI & INPUTS)
# =====================================================================
# =====================================================================
# BACKEND: XAVFSIZLIK, KODLASH VA TOZALASH MEXANIZMLARI
# =====================================================================

def cleanup_temp_files():
    """Joriy sessiyaga tegishli vaqtinchalik faylni imkon qadar xavfsiz tozalash."""
    try:
        path = st.session_state.get('raw_file_path')
    except Exception:
        return
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass
    try:
        st.session_state.pop('raw_file_path', None)
    except Exception:
        pass

# Server to'xtaganda axlat fayllarni avtomatik o'chirishni ro'yxatdan o'tkazish
atexit.register(cleanup_temp_files)

# =====================================================================
# 1-BLOK: KINEMATIK LABORATORIYA SOZLAMALARI (UI & INPUTS)
# =====================================================================

def render_block_1():
    # Master Lock holati xotirada saqlanadi
    if 'locked' not in st.session_state:
        st.session_state.locked = False

    # Qulflangan bo'lsa, avtomatik ravishda yopiladi
    with st.expander("🎛️ BLOCK 1: Kinematic Parameters & Input", expanded=not st.session_state.locked):
        
        tab_file, tab_math, tab_topo, tab_limits = st.tabs([
            "📂 1. File & Mapping", 
            "🧮 2. Mathematics & Engine", 
            "⛰️ 3. Topography & Environment", 
            "🚧 4. Physical & UI Limits"
        ])

        file_settings = {'default_dt': 1.0} 
        math_settings = {}
        topo_settings = {}
        limits_settings = {}

        # --- 1-TAB: FAYL YUKLASH VA XARITALASH ---
        with tab_file:
            st.markdown("### 📂 Upload GPX/KML Data & Data-Quality Review")
            
            uploaded_file = st.file_uploader(
                "Upload a trajectory file (see Block 0):", 
                type=["gpx", "kml"],
                key="b1_file_upload",
                disabled=st.session_state.locked,
                help="Upload a GPX or KML trajectory. Other formats are intentionally disabled. The file is checked for missing/invalid coordinates, missing elevations, and duplicate records before processing."
            )
            
            if uploaded_file:
                ext = uploaded_file.name.split('.')[-1].lower()
                file_settings['ext'] = ext
                
                st.success(f"✅ File selected: {uploaded_file.name} ({ext.upper()} format)")
                st.markdown("#### 🗺️ XML Tag Mapping")
                
                c_map1, c_map2, c_map3, c_map4 = st.columns(4)
                
                if ext == "gpx":
                    with c_map1: file_settings['tag_point'] = st.text_input("Point tag:", value="trkpt", disabled=st.session_state.locked, key="b1_point_tag", help="GPX point element used for track samples, normally trkpt. The parser compares the local XML tag name.")
                    with c_map2: file_settings['tag_lat'] = st.text_input("Latitude attribute:", value="lat", disabled=st.session_state.locked, key="b1_latitude_attribute", help="XML attribute containing latitude in decimal degrees, from -90 to +90.")
                    with c_map3: file_settings['tag_lon'] = st.text_input("Longitude attribute:", value="lon", disabled=st.session_state.locked, key="b1_longitude_attribute", help="XML attribute containing longitude in decimal degrees, from -180 to +180.")
                    with c_map4: file_settings['tag_ele'] = st.text_input("Elevation tag:", value="ele", disabled=st.session_state.locked, key="b1_elevation_tag", help="Child tag containing elevation in metres. Missing values are interpolated if at least one valid elevation exists.")
                    file_settings['tag_time'] = st.text_input("Time tag:", value="time", disabled=st.session_state.locked, key="b1_gpx_time_tag", help="Child tag containing each point's timestamp, usually time. Missing timestamps are not silently fabricated when other timestamps exist.")

                elif ext == "kml":
                    with c_map1: file_settings['tag_coords'] = st.text_input("Coordinates tag:", value="coordinates", disabled=st.session_state.locked, key="b1_coordinates_tag", help="KML coordinate tag. Standard KML coordinates use longitude,latitude,altitude tuples.")
                    with c_map2: file_settings['tag_time'] = st.text_input("Time tag:", value="when", disabled=st.session_state.locked, key="b1_kml_time_tag", help="Timestamp tag used by timestamped KML tracks, commonly when. Untimestamped coordinate lists use the default time step.")

                # Fayl tanlanganda ma'lumot sifati bo'yicha dastlabki auditni shu tabda ko'rsatish.
                try:
                    import hashlib
                    file_bytes = uploaded_file.getvalue()
                    digest = hashlib.sha256(file_bytes + repr(sorted((k, v) for k, v in file_settings.items() if k.startswith("tag_") or k in ("tag_point",))).encode("utf-8")).hexdigest()
                    if st.session_state.get("diagnostic_file_hash") != digest:
                        st.session_state["diagnostic_file_hash"] = digest
                        st.session_state.pop("file_quality_report", None)
                    if "file_quality_report" not in st.session_state:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as preview_file:
                            preview_file.write(file_bytes)
                            preview_path = preview_file.name
                        try:
                            preview_df = parse_large_xml(preview_path, ext, file_settings)
                        finally:
                            try:
                                os.remove(preview_path)
                            except OSError:
                                pass
                        if preview_df.empty:
                            st.session_state["file_quality_report"] = {
                                "summary": {"Rows read": 0, "Invalid/missing coordinates": 0, "Missing elevations": 0,
                                            "Duplicate coordinate-time rows": 0, "Rows with duplicate timestamps": 0, "Missing timestamps": 0},
                                "rows": pd.DataFrame()
                            }
                        else:
                            raw = preview_df.copy()
                            for c in ("lat", "lon", "ele"):
                                raw[c] = pd.to_numeric(raw[c], errors="coerce")
                            coord_valid = (
                                np.isfinite(raw["lat"]) & np.isfinite(raw["lon"]) &
                                raw["lat"].between(-90, 90) & raw["lon"].between(-180, 180)
                            )
                            missing_ele = ~np.isfinite(raw["ele"])
                            time_values = pd.to_datetime(raw["time"], format='mixed', errors="coerce", utc=True)
                            duplicate_mask = raw.duplicated(subset=["lat", "lon", "time"], keep="first")
                            issue_rows = raw.loc[(~coord_valid) | missing_ele | duplicate_mask | time_values.isna()].copy()
                            issue_rows["Issue"] = ""
                            issue_rows.loc[~coord_valid, "Issue"] += "Missing/invalid coordinates; "
                            issue_rows.loc[missing_ele, "Issue"] += "Missing elevation (will interpolate); "
                            issue_rows.loc[duplicate_mask, "Issue"] += "Duplicate coordinate-time row; "
                            issue_rows.loc[duplicate_time_mask, "Issue"] += "Duplicate timestamp; check whether coordinates conflict; "
                            issue_rows.loc[time_values.isna(), "Issue"] += "Missing timestamp; "
                            st.session_state["file_quality_report"] = {
                                "summary": {
                                    "Rows read": int(len(raw)),
                                    "Invalid/missing coordinates": int((~coord_valid).sum()),
                                    "Valid elevations": int((~missing_ele).sum()),
                                    "Missing elevations": int(missing_ele.sum()),
                                    "Duplicate coordinate-time rows": int(duplicate_mask.sum()),
                                    "Rows with duplicate timestamps": int(duplicate_time_mask.sum()),
                                    "Missing timestamps": int(time_values.isna().sum())
                                },
                                "rows": issue_rows.head(500)
                            }
                    report = st.session_state["file_quality_report"]
                    st.markdown("#### 🔎 File quality check")
                    st.caption("Preview of rows with missing/invalid values or duplicates. The table is capped at 500 issue rows for interface performance.")
                    st.json(report["summary"])
                    if report["summary"].get("Rows read", 0) > 0 and report["summary"].get("Valid elevations", 0) == 0:
                        st.error("No valid elevation value was found. Elevation interpolation requires at least one valid elevation sample.")
                    elif report["summary"].get("Missing elevations", 0) > 0:
                        st.info("Missing elevations will be interpolated by elapsed time from available valid elevation samples; leading/trailing gaps use the nearest valid sample.")
                    if not report["rows"].empty:
                        st.dataframe(report["rows"], use_container_width=True, hide_index=True)
                    else:
                        st.success("No rows with missing/invalid coordinates, missing elevations, duplicate coordinate-time keys, or missing timestamps were found.")
                except Exception as diagnostic_error:
                    st.warning(f"Pre-check could not fully parse this file: {diagnostic_error}. The processing engine will validate it again when RUN is pressed.")

            st.divider()
            st.markdown("#### ⏳ Time & Safety Settings")
            
            c_time, c_dt, c_gap = st.columns(3)
            with c_time:
                utc_options = [f"UTC{'+' if i > 0 else ''}{i}" for i in range(-12, 15)]
                file_settings['utc_shift'] = st.selectbox("UTC time zone:", options=utc_options, index=12, disabled=st.session_state.locked, key="b1_utc_time_zone", help="Timezone offset used for timestamp display. It does not change elapsed time between samples.")
            with c_dt:
                file_settings['default_dt'] = st.number_input("Default time step (seconds):", value=1.0, min_value=0.01, disabled=st.session_state.locked, key="b1_default_time_step_seconds", help="Time interval assigned to points when the GPX/KML file has no usable per-point timestamps.")
            with c_gap:
                file_settings['gap_tolerance'] = st.number_input("Gap tolerance (s):", value=10.0, min_value=1.0, disabled=st.session_state.locked, key="b1_gap_tolerance_s", help="A timestamp interval above this threshold is flagged as a time gap.")

        # --- 2-TAB: MATEMATIKA VA DVIGATEL ---
        with tab_math:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("**WGS84 Ellipsoid (Projection)**")
                math_settings['wgs84_a'] = st.number_input("Semi-major axis (a, m):", value=6378137.0, disabled=st.session_state.locked, key="b1_semi_major_axis_a_m", help="WGS84 ellipsoid semi-major axis in metres; normally keep the standard value.")
                math_settings['wgs84_e2'] = st.number_input("Eccentricity squared (e²):", value=0.00669437999, format="%.9f", disabled=st.session_state.locked, key="b1_eccentricity_squared_e", help="Dimensionless WGS84 ellipsoid eccentricity squared; must be between 0 and 1.")
                math_settings['target_hz'] = st.number_input("Target frequency (Hz):", value=10.0, min_value=1.0, max_value=50.0, disabled=st.session_state.locked, key="b1_target_frequency_hz", help="Sampling frequency for interpolation. Higher values create more output points; this build caps the output at 2,000,000 samples.")
                
                st.markdown("**Savitzky–Golay (SG) Smoothing Filter**")
                math_settings['sg_window'] = st.number_input("Window length:", value=15, min_value=3, step=2, disabled=st.session_state.locked, key="b1_window_length", help="Odd number of samples used by the Savitzky–Golay smoothing filter.")
                math_settings['sg_polyorder'] = st.number_input("Polynomial order:", value=3, min_value=1, disabled=st.session_state.locked, key="b1_polynomial_order", help="Polynomial degree for Savitzky–Golay smoothing; it must be lower than the window length.")
            
            with col_m2:
                st.markdown("**Kalman Filter (Covariance & ZUPT)**")
                math_settings['kalman_r'] = st.number_input("Measurement noise (Kalman R):", value=10.0, format="%.2f", disabled=st.session_state.locked, key="b1_measurement_noise_kalman_r", help="Kalman measurement-noise covariance scale. Larger values trust GPS position measurements less.")
                math_settings['kalman_q'] = st.number_input("Process noise (Kalman Q):", value=0.1, format="%.4f", disabled=st.session_state.locked, key="b1_process_noise_kalman_q", help="Kalman process-noise scale. Larger values allow the estimated motion state to change more freely.")
                math_settings['zupt_damping'] = st.number_input("ZUPT damping coefficient:", value=0.8, min_value=0.01, max_value=1.0, disabled=st.session_state.locked, key="b1_zupt_damping_coefficient", help="Controls the strength of the zero-velocity update; allowed range is 0.01–1.0.")

        # --- 3-TAB: TOPOGRAFIYA VA ATROF-MUHIT ---
        with tab_topo:
            c_env1, c_env2, c_env3 = st.columns(3)
            with c_env1:
                st.markdown("**⛰️ Terrain & Elevation Validation**")
                topo_settings['enabled'] = st.toggle("Enable terrain/slope analysis", value=True, disabled=st.session_state.locked, key="b1_enable_terrain_slope_analysis", help="Calculate terrain grade and slope-limit flags from the interpolated elevation profile.")
                topo_settings['atmosphere_enabled'] = st.toggle("Enable ISA atmosphere model", value=True, disabled=st.session_state.locked, key="b1_enable_isa_atmosphere_model", help="Independently calculate approximate standard-atmosphere temperature, pressure and density from elevation and the baseline settings. These are model estimates, not live weather measurements.")
                topo_settings['max_topo_grad'] = st.number_input("Maximum allowed slope (degrees):", value=45.0, min_value=0.1, max_value=89.0, disabled=st.session_state.locked, key="b1_maximum_allowed_slope_degrees", help="Slope threshold in degrees used to flag unusually steep terrain; it is a warning threshold, not a physical guarantee.")
                topo_settings['spatial_step'] = st.number_input("Spatial averaging step (m):", value=50.0, min_value=1.0, max_value=10000.0, disabled=st.session_state.locked, key="b1_spatial_averaging_step_m", help="Distance interval used when calculating terrain grade. Larger values smooth local elevation noise.")
            with c_env2:
                st.markdown("**🌬️ Global Wind Settings**")
                topo_settings['wind_speed'] = st.number_input("Wind speed (m/s):", value=0.0, min_value=0.0, max_value=100.0, disabled=st.session_state.locked, key="b1_wind_speed_m_s", help="Assumed ambient wind speed in metres per second, used only for relative wind metrics.")
                wind_directions = {"0° (N)": 0, "45° (NE)": 45, "90° (E)": 90, "135° (SE)": 135, "180° (S)": 180, "225° (SW)": 225, "270° (W)": 270, "315° (NW)": 315}
                wind_choice = st.selectbox("Wind direction:", options=list(wind_directions.keys()), disabled=st.session_state.locked, key="b1_wind_direction", help="Meteorological wind-FROM direction in compass degrees: 0° means wind comes from north. Headwind is positive when the wind comes from the direction of travel; crosswind is signed.")
                topo_settings['wind_direction_deg'] = wind_directions[wind_choice]
            with c_env3:
                st.markdown("**🌡️ Atmospheric Baseline (ISA)**")
                topo_settings['base_alt'] = st.number_input("Baseline elevation (h0, m):", value=0.0, min_value=-500.0, max_value=11000.0, disabled=st.session_state.locked, key="b1_baseline_elevation_h0_m", help="Reference elevation in metres for the standard-atmosphere pressure/temperature approximation.")
                topo_settings['base_press'] = st.number_input("Baseline air pressure (P0, Pa):", value=101325.0, min_value=1000.0, max_value=120000.0, disabled=st.session_state.locked, key="b1_baseline_air_pressure_p0_pa", help="Reference atmospheric pressure in pascals at the baseline elevation.")
                topo_settings['base_temp'] = st.number_input("Baseline air temperature (T0, °C):", value=15.0, min_value=-80.0, max_value=60.0, disabled=st.session_state.locked, key="b1_baseline_air_temperature_t0_c", help="Reference air temperature in degrees Celsius; must be above absolute zero.")

        # --- 4-TAB: FIZIK VA UI CHEGARALAR ---
        with tab_limits:
            c_lim1, c_lim2, c_lim3 = st.columns(3)
            with c_lim1:
                st.markdown("**Longitudinal Limits**")
                limits_settings['max_v'] = st.number_input("Maximum speed (km/h):", value=250.0, min_value=0.1, max_value=2000.0, disabled=st.session_state.locked, key="b1_maximum_speed_km_h", help="Physical speed threshold used to detect implausible GPS segments.")
                limits_settings['max_a'] = st.number_input("Maximum acceleration (m/s²):", value=10.0, min_value=0.1, max_value=500.0, disabled=st.session_state.locked, key="b1_maximum_acceleration_m_s", help="Acceleration threshold used to flag results beyond the configured limit.")
                limits_settings['max_b'] = st.number_input("Maximum braking deceleration (m/s²):", value=12.0, min_value=0.1, max_value=500.0, disabled=st.session_state.locked, key="b1_maximum_braking_deceleration_m_s", help="Magnitude threshold for braking deceleration flags.")
            with c_lim2:
                st.markdown("**Lateral & Geometric Limits**")
                limits_settings['max_alat'] = st.number_input("Maximum lateral acceleration (m/s²):", value=10.0, min_value=0.1, max_value=500.0, disabled=st.session_state.locked, key="b1_maximum_lateral_acceleration_m_s", help="Lateral acceleration threshold used to flag sharp turns.")
                limits_settings['max_jerk'] = st.number_input("Maximum jerk (m/s³):", value=15.0, min_value=0.1, max_value=10000.0, disabled=st.session_state.locked, key="b1_maximum_jerk_m_s", help="Jerk threshold in metres per second cubed; high values may indicate noise or abrupt motion.")
                limits_settings['zupt_lim'] = st.number_input("ZUPT zero-speed threshold (km/h):", value=1.5, min_value=0.0, max_value=50.0, disabled=st.session_state.locked, key="b1_zupt_zero_speed_threshold_km_h", help="Speeds below this threshold can trigger the zero-velocity Kalman update.")
                limits_settings['safe_omega'] = st.number_input("Safe angular velocity (rad/s):", value=0.0001, min_value=1e-8, format="%.7f", disabled=st.session_state.locked, key="b1_safe_angular_velocity_rad_s", help="Small denominator safeguard for turn-radius calculations; it is a numerical threshold.")
            with c_lim3:
                st.markdown("**System & Memory Limits**")
                limits_settings['max_dist_km'] = st.number_input("Maximum distance (km):", value=500.0, min_value=0.001, max_value=100000.0, disabled=st.session_state.locked, key="b1_maximum_distance_km", help="Maximum cumulative trajectory distance retained in the output.")
                limits_settings['max_plot_points'] = st.number_input("Maximum plot points:", value=3000, min_value=100, step=500, disabled=st.session_state.locked, key="b1_maximum_plot_points", help="Maximum number of points used for standard plots to reduce browser load.")
                limits_settings['max_radius'] = st.number_input("Maximum turning radius (m):", value=10000.0, min_value=1.0, step=100.0, disabled=st.session_state.locked, key="b1_maximum_turning_radius_m", help="Upper display/processing cap for estimated turning radius.")

        # --- MASTER LOCK BUTTON ---
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.locked:
            st.success("🔒 All parameters confirmed. The Block 2 engine is ready!")
            if st.button("🔓 Unlock & Edit Parameters", width='stretch', key="b1_button_unlock_edit_parameters", help="Unlock Block 1 so you can change the file or parameters. You will need to confirm them again before running Block 2."):
                st.session_state.locked = False
                for stale_key in ('processed_data', 'zip_ready', 'calc_mode', 'processing_quality_report'):
                    st.session_state.pop(stale_key, None)
                cleanup_temp_files() # Eskisini qo'lda tozalash
                st.rerun()
        else:
            if st.button("✅ Master Lock: Confirm Parameters", type="primary", width='stretch', key="b1_button_master_lock_confirm_parameters", help="Validate the uploaded GPX/KML file and all parameter values, then freeze the configuration for Block 2."):
                errors = []
                if not uploaded_file:
                    errors.append("Please upload a GPX or KML input file before starting.")
                if 'default_dt' not in file_settings or not np.isfinite(file_settings['default_dt']) or file_settings['default_dt'] <= 0:
                    errors.append("The default time step must be finite and greater than zero.")
                if math_settings['sg_window'] % 2 == 0:
                    errors.append("The Savitzky–Golay window length must be an odd number (e.g., 7 or 15).")
                if math_settings['sg_polyorder'] >= math_settings['sg_window']:
                    errors.append("The SG polynomial order must be smaller than the window length.")
                if math_settings['sg_polyorder'] < 1 or math_settings['sg_window'] < 3:
                    errors.append("SG polynomial order must be at least 1 and window length at least 3.")
                if math_settings.get('target_hz', 0) > 50:
                    errors.append("Target frequency cannot exceed 50 Hz in this build.")
                if uploaded_file and file_settings.get('ext') not in ('gpx', 'kml'):
                    errors.append("Only GPX and KML formats are currently supported.")
                # Barcha fizik va matematik sozlamalar chekli hamda fizik jihatdan maqbul bo'lishi shart.
                for key in ('wgs84_a', 'target_hz', 'kalman_r', 'kalman_q'):
                    value = math_settings.get(key, np.nan)
                    if not np.isfinite(value) or value <= 0:
                        errors.append(f"{key} must be a finite positive number.")
                for key in ('max_v', 'max_a', 'max_b', 'max_alat', 'max_jerk', 'safe_omega', 'max_dist_km', 'max_plot_points', 'max_radius'):
                    value = limits_settings.get(key, np.nan)
                    if not np.isfinite(value) or value <= 0:
                        errors.append(f"{key} must be a finite positive number.")
                if not (0 < math_settings.get('wgs84_e2', 0) < 1):
                    errors.append("WGS84 eccentricity squared must be between 0 and 1.")
                for key in ('wind_speed', 'max_topo_grad', 'spatial_step', 'base_alt', 'base_press', 'base_temp'):
                    value = topo_settings.get(key, np.nan)
                    if not np.isfinite(value):
                        errors.append(f"{key} must be finite.")
                if topo_settings.get('base_press', 0) <= 0:
                    errors.append("Baseline air pressure must be positive.")
                if topo_settings.get('base_temp', -273.15) <= -273.15:
                    errors.append("Baseline air temperature must be above absolute zero.")
                if errors:
                    for err in errors:
                        st.error(f"❌ {err}")
                else:
                    cleanup_temp_files() # Yangi fayl yuklashdan oldin xotirani tozalash
                    
                    # RAM HIMOYA: Faylni xotiraga nusxalamay, xavfsiz vaqtinchalik disk manziliga yozamiz
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_settings['ext']}") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_file_path = tmp_file.name
                        
                    st.session_state.locked = True
                    st.session_state.raw_file_path = temp_file_path 
                    st.session_state.file_settings = dict(file_settings)
                    st.session_state.math_settings = dict(math_settings)
                    st.session_state.topo_settings = dict(topo_settings)
                    st.session_state.limits_settings = dict(limits_settings)
                    st.rerun()
# =====================================================================
# 1-BLOK OXIRI
# =====================================================================
# =====================================================================
# 2-BLOK BOSHI: KINEMATIK VA DINAMIK HISOB-KITOB DVIGATELI (REWRITTEN ENGINE)
# =====================================================================
@nb.njit
def ecef_to_enu_vel(vx, vy, vz, lat_deg, lon_deg):
    """
    ECEF tezlik vektorlarini (Vx, Vy, Vz) Mahalliy Tangensal Tekislikka (ENU - East, North, Up) o'tkazish.
    Bu geografik jihatdan to'g'ri Azimut (Heading) ni topish uchun zarur.
    """
    n = len(vx)
    v_east = np.zeros(n)
    v_north = np.zeros(n)
    v_up = np.zeros(n)
    
    for i in range(n):
        lat_rad = np.radians(lat_deg[i])
        lon_rad = np.radians(lon_deg[i])
        clat = np.cos(lat_rad)
        slat = np.sin(lat_rad)
        clon = np.cos(lon_rad)
        slon = np.sin(lon_rad)
        
        # ECEF -> ENU Transformatsiya matritsasi
        v_east[i] = -slon * vx[i] + clon * vy[i]
        v_north[i] = -slat * clon * vx[i] - slat * slon * vy[i] + clat * vz[i]
        v_up[i] = clat * clon * vx[i] + clat * slon * vy[i] + slat * vz[i]
        
    return v_east, v_north, v_up


def parse_large_xml(filepath, ext, settings):
    """GPX/KML parser: nuqta konteksti saqlanadi va noto'g'ri qatorlar sifat auditi uchun qaytariladi."""
    data = []

    def local_name(tag):
        return tag.rsplit('}', 1)[-1]

    def number_or_nan(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else np.nan
        except (TypeError, ValueError, OverflowError):
            return np.nan

    def parse_kml_token(token, coord_style='comma'):
        try:
            if coord_style == 'space':
                vals = [number_or_nan(x) for x in token.strip().split()]
            else:
                vals = [number_or_nan(x) for x in token.strip().split(',')]
            if len(vals) < 2:
                return {'lat': np.nan, 'lon': np.nan, 'ele': np.nan, 'time': None}
            lon, lat = vals[0], vals[1]
            ele = vals[2] if len(vals) > 2 else np.nan
            valid = np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            return {'lat': lat if valid else np.nan, 'lon': lon if valid else np.nan,
                    'ele': ele, 'time': None}
        except Exception:
            return {'lat': np.nan, 'lon': np.nan, 'ele': np.nan, 'time': None}

    ext = str(ext).lower().lstrip('.')
    if ext == 'gpx':
        point_tag = settings.get('tag_point', 'trkpt')
        lat_tag = settings.get('tag_lat', 'lat')
        lon_tag = settings.get('tag_lon', 'lon')
        ele_tag = settings.get('tag_ele', 'ele')
        time_tag = settings.get('tag_time', 'time')
        for _, elem in ET.iterparse(filepath, events=('end',)):
            if local_name(elem.tag) != point_tag:
                continue
            lat = number_or_nan(elem.attrib.get(lat_tag))
            lon = number_or_nan(elem.attrib.get(lon_tag))
            valid = np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            rec = {'lat': lat if valid else np.nan, 'lon': lon if valid else np.nan,
                   'ele': np.nan, 'time': None}
            for child in elem:
                tag = local_name(child.tag)
                if tag == ele_tag:
                    rec['ele'] = number_or_nan(child.text)
                elif tag == time_tag and child.text and child.text.strip():
                    rec['time'] = child.text.strip()
            data.append(rec)
            elem.clear()
    elif ext == 'kml':
        coords_tag = settings.get('tag_coords', 'coordinates')
        time_tag = settings.get('tag_time', 'when')
        for _, placemark in ET.iterparse(filepath, events=('end',)):
            if local_name(placemark.tag) != 'Placemark':
                continue
            # gx:Track nuqtalari: when va coord faqat bir Track doirasida juftlanadi.
            track_nodes = [node for node in placemark.iter() if local_name(node.tag) == 'Track']
            handled_coord_ids = set()
            for track in track_nodes:
                times = [n.text.strip() for n in track.iter() if local_name(n.tag) == time_tag and n.text and n.text.strip()]
                coords = [n for n in track.iter() if local_name(n.tag) == 'coord']
                for n in coords:
                    handled_coord_ids.add(id(n))
                for i, node in enumerate(coords):
                    rec = parse_kml_token(node.text or '', coord_style='space')
                    if len(times) == len(coords):
                        rec['time'] = times[i]
                    data.append(rec)
            # Standard KML coordinates: a single timestamp is only valid for a single coordinate.
            for node in placemark.iter():
                if local_name(node.tag) != coords_tag or not node.text:
                    continue
                tokens = node.text.strip().split()
                recs = [parse_kml_token(token, coord_style='comma') for token in tokens]
                times = [n.text.strip() for n in placemark.iter()
                         if local_name(n.tag) == time_tag and n.text and n.text.strip()]
                if not track_nodes and len(times) == len(recs):
                    for rec, ts in zip(recs, times): rec['time'] = ts
                elif not track_nodes and len(recs) == 1 and len(times) == 1:
                    recs[0]['time'] = times[0]
                data.extend(recs)
            placemark.clear()
    else:
        raise ValueError(f"Unsupported input format '{ext}'. Only GPX and KML are supported.")
    return pd.DataFrame(data, columns=['lat', 'lon', 'ele', 'time'])


def wgs84_to_ecef(lat, lon, alt, a, e2):
    if not (np.isfinite(a) and a > 0 and np.isfinite(e2) and 0 <= e2 < 1):
        raise ValueError("WGS84 parameters must satisfy a > 0 and 0 <= e² < 1.")
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    return x, y, z


def ecef_to_wgs84(x, y, z, a, e2):
    if not (np.isfinite(a) and a > 0 and np.isfinite(e2) and 0 <= e2 < 1):
        raise ValueError("WGS84 parameters must satisfy a > 0 and 0 <= e² < 1.")
    if np.any(np.sqrt(np.asarray(x, dtype=float)**2 + np.asarray(y, dtype=float)**2 + np.asarray(z, dtype=float)**2) < 1.0):
        raise ValueError("ECEF origin is not a valid Earth-surface coordinate.")
    b = np.sqrt(a**2 * (1 - e2))
    ep2 = (a**2 - b**2) / b**2
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th)**3, p - e2 * a * np.cos(th)**3)
    
    # Matematik singulyarliklarni oldini olish (safe_cos)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    safe_cos = np.maximum(np.abs(np.cos(lat)), 1e-12)
    
    
    near_pole = np.abs(lat) > np.radians(89.0)
    sin_abs = np.maximum(np.abs(np.sin(lat)), 1e-12)
    alt_pole = np.abs(z) / sin_abs - N * (1 - e2)
    alt_equator = p / safe_cos - N
    alt = np.where(near_pole, alt_pole, alt_equator)
                   
    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu_position(x, y, z, lat0_deg, lon0_deg, x0=None, y0=None, z0=None):
    """ECEF koordinatalarini boshlang'ich nuqtaga nisbatan lokal ENU metrlariga o'tkazadi."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); z = np.asarray(z, dtype=float)
    if x0 is None: x0, y0, z0 = float(x[0]), float(y[0]), float(z[0])
    lat0 = np.radians(float(lat0_deg)); lon0 = np.radians(float(lon0_deg))
    dx, dy, dz = x - x0, y - y0, z - z0
    east = -np.sin(lon0)*dx + np.cos(lon0)*dy
    north = -np.sin(lat0)*np.cos(lon0)*dx - np.sin(lat0)*np.sin(lon0)*dy + np.cos(lat0)*dz
    up = np.cos(lat0)*np.cos(lon0)*dx + np.cos(lat0)*np.sin(lon0)*dy + np.sin(lat0)*dz
    return east, north, up


@nb.njit
def run_kalman_filter_9d(x_raw, y_raw, z_raw, dt, r_val, q_val, zupt_lim, zupt_damp):
    """9D Kalman filtri: white-jerk jarayon shovqini va Joseph kovariatsiya yangilanishi."""
    x_raw = np.asarray(x_raw, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)
    z_raw = np.asarray(z_raw, dtype=np.float64)
    n = len(x_raw)
    if not (len(y_raw) == n and len(z_raw) == n):
        raise ValueError("Kalman filtri: koordinata massivlari uzunligi bir xil bo'lishi kerak.")
    if n == 0: return np.empty((9, 0)), np.empty(0), np.empty(0, dtype=np.bool_)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(r_val) or r_val <= 0 or not np.isfinite(q_val) or q_val <= 0:
        raise ValueError("Kalman filtri parametrlari dt, R va Q musbat va chekli bo'lishi kerak.")
    if not (np.all(np.isfinite(x_raw)) and np.all(np.isfinite(y_raw)) and np.all(np.isfinite(z_raw))):
        raise ValueError("Kalman filtri koordinatalarida NaN yoki cheksiz qiymatlar mavjud.")
    state = np.zeros((9, n)); raw_v_arr = np.zeros(n); zupt_applied = np.zeros(n, dtype=np.bool_)
    state[:3, 0] = [x_raw[0], y_raw[0], z_raw[0]]
    if n > 1: state[3:6, 0] = [(x_raw[1]-x_raw[0])/dt, (y_raw[1]-y_raw[0])/dt, (z_raw[1]-z_raw[0])/dt]
    P = np.eye(9) * max(float(r_val), 1e-6)
    R = np.eye(3) * float(r_val)
    H = np.zeros((3, 9)); H[:, :3] = np.eye(3)
    F = np.eye(9)
    F[:3, 3:6] = np.eye(3)*dt
    F[:3, 6:9] = np.eye(3)*(0.5*dt*dt)
    F[3:6, 6:9] = np.eye(3)*dt
    # Uzluksiz oq rangli jerk uchun har bir o'qning 3x3 kovariatsiyasi.
    Q = np.zeros((9, 9))
    q_block = float(q_val) * np.array([[dt**5/20, dt**4/8, dt**3/6], [dt**4/8, dt**3/3, dt**2/2], [dt**3/6, dt**2/2, dt]])
    for base in (0, 1, 2):
        ix0, ix1, ix2 = base, base + 3, base + 6
        Q[ix0, ix0], Q[ix0, ix1], Q[ix0, ix2] = q_block[0, 0], q_block[0, 1], q_block[0, 2]
        Q[ix1, ix0], Q[ix1, ix1], Q[ix1, ix2] = q_block[1, 0], q_block[1, 1], q_block[1, 2]
        Q[ix2, ix0], Q[ix2, ix1], Q[ix2, ix2] = q_block[2, 0], q_block[2, 1], q_block[2, 2]
    I = np.eye(9)
    for i in range(1, n):
        delta = np.array([x_raw[i]-x_raw[i-1], y_raw[i]-y_raw[i-1], z_raw[i]-z_raw[i-1]])
        raw_v_arr[i] = np.linalg.norm(delta)/dt
        xp = F @ state[:, i-1].copy()
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = np.linalg.solve(S, H @ Pp).T
        innovation = np.array([x_raw[i], y_raw[i], z_raw[i]]) - H @ xp
        xi = xp + K @ innovation
        A = I - K @ H
        P = A @ Pp @ A.T + K @ R @ K.T  # Joseph form
        # ZUPT: nol tezlikni pseudo-measurement sifatida yangilash; holatni qo'pol nolga tenglashtirmaslik.
        if raw_v_arr[i] < float(zupt_lim)/3.6:
            zupt_applied[i] = True
            Hv = np.zeros((3, 9)); Hv[:, 3:6] = np.eye(3)
            Rv = np.eye(3) * max(float(r_val) * max(float(zupt_damp), 1e-3), 1e-6)
            Sv = Hv @ P @ Hv.T + Rv
            Kv = np.linalg.solve(Sv, Hv @ P).T
            xi = xi + Kv @ (-Hv @ xi)
            Av = I - Kv @ Hv
            P = Av @ P @ Av.T + Kv @ Rv @ Kv.T
        state[:, i] = xi
    return state, raw_v_arr, zupt_applied


def render_block_2():
    st.divider()
    st.markdown("## ⚙️ BLOCK 2: Computation Engine (RUN)")
    
    if not st.session_state.get('locked', False):
        st.warning("⚠️ Please confirm the parameters in Block 1 (Master Lock) first.")
        return

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        run_time = st.button("⏱️ RUN (Time [s])", type="primary", key="ui_run_time_s", help="Run the trajectory analysis using time as the independent variable. Requires a confirmed Block 1 configuration.")
    with col2:
        run_dist = st.button("📏 RUN (Distance [m])", type="primary", key="ui_run_distance_m", help="Run the trajectory analysis and resample the results to a 1-m distance grid. Requires enough valid trajectory points.")
    with col3:
        if st.button("🔄 Restart", key="ui_restart", help="Clear the current configuration and results, then return to the initial setup."):
            cleanup_temp_files()
            st.session_state.clear()
            st.rerun()

    if run_time or run_dist:
        if 'zip_ready' in st.session_state:
            del st.session_state['zip_ready']
        st.session_state.pop('processed_data', None)
        st.session_state.pop('processing_quality_report', None)
            
        raw_path = st.session_state.raw_file_path
        f_set = st.session_state.file_settings
        m_set = st.session_state.math_settings
        t_set = st.session_state.topo_settings
        l_set = st.session_state.limits_settings

        with st.status("Computation started...", expanded=True) as status:
            try:
                st.write("⏳ Reading and cleaning input data from disk...")
                ext = f_set['ext']
                if ext not in ('gpx', 'kml'):
                    raise ValueError("Only GPX and KML input files are supported in this version.")
                df = parse_large_xml(raw_path, ext, f_set)

                if df.empty:
                    raise ValueError("No coordinates were found in the file. Check the file format and tags.")
                quality_summary = {"parsed_rows": int(len(df)), "invalid_coordinate_rows_dropped": 0,
                                   "invalid_timestamp_rows_dropped": 0, "exact_duplicate_rows_removed": 0,
                                   "timestamps_generated_from_default_step": 0, "elevation_values_interpolated": 0,
                                   "isolated_gps_spikes_removed": 0}

                col_map = {
                    f_set.get('col_lat', 'lat'): 'lat', f_set.get('col_lon', 'lon'): 'lon',
                    f_set.get('col_ele', 'ele'): 'ele', f_set.get('col_time', 'time'): 'time'
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                # Backend tekshiruvi: koordinatalarni sonli turga o'tkazish va geografik chegaralarni tekshirish.
                for c in ('lat', 'lon'):
                    if c not in df.columns:
                        raise ValueError(f"Required coordinate column '{c}' was not found. Check column mapping.")
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                if 'ele' not in df.columns: df['ele'] = np.nan
                df['ele'] = pd.to_numeric(df['ele'], errors='coerce')
                coord_ok = np.isfinite(df['lat']) & np.isfinite(df['lon']) & df['lat'].between(-90, 90) & df['lon'].between(-180, 180)
                bad_coord_count = int((~coord_ok).sum())
                quality_summary['invalid_coordinate_rows_dropped'] = bad_coord_count
                df = df.loc[coord_ok].copy()
                if bad_coord_count: st.warning(f"Dropped {bad_coord_count} rows with missing or invalid latitude/longitude.")
                if df.empty: raise ValueError("No valid latitude/longitude coordinates remain after validation.")
                parsed_time = pd.to_datetime(df['time'], format='mixed', errors='coerce', utc=True) if 'time' in df.columns else pd.Series(pd.NaT, index=df.index)
                if parsed_time.notna().any():
                    invalid_time = parsed_time.isna()
                    if invalid_time.any():
                        quality_summary['invalid_timestamp_rows_dropped'] = int(invalid_time.sum())
                        st.warning(f"Dropped {int(invalid_time.sum())} rows with missing/invalid timestamps. No timestamps are forward-filled.")
                        df = df.loc[~invalid_time].copy(); parsed_time = parsed_time.loc[~invalid_time]
                    df['time'] = parsed_time.dt.tz_convert(None)
                    # Bir xil vaqt va aynan bir xil nuqta bo'lgan takrorlar xavfsiz olib tashlanadi.
                    exact_dup = df.duplicated(subset=['lat', 'lon', 'ele', 'time'], keep='first')
                    if exact_dup.any():
                        quality_summary['exact_duplicate_rows_removed'] = int(exact_dup.sum())
                        st.warning(f"Removed {int(exact_dup.sum())} exact duplicate coordinate/time rows.")
                        df = df.loc[~exact_dup].copy()
                    # Bir vaqt belgisida turli joylar bo'lsa, sort/drop_duplicates bilan jim yo'qotmaymiz.
                    time_conflicts = df.groupby('time')[['lat', 'lon']].nunique().max(axis=1) > 1
                    if time_conflicts.any():
                        raise ValueError(f"Found {int(time_conflicts.sum())} timestamp(s) assigned to different coordinates. Correct duplicate/conflicting timestamps in the source file.")
                    df = df.drop_duplicates(subset=['time'], keep='first')
                else:
                    st.info("No usable per-point timestamps were found. The configured default time step will be used.")
                    quality_summary['timestamps_generated_from_default_step'] = int(len(df))
                    time_steps = np.arange(len(df), dtype=float) * float(f_set['default_dt'])
                    df['time'] = pd.Timestamp('1970-01-01') + pd.to_timedelta(time_steps, unit='s')
                df = df.sort_values('time').reset_index(drop=True)
                source_time_origin = pd.Timestamp(df['time'].iloc[0])

                # Balandlik yetishmasa, vaqt bo'yicha mavjud haqiqiy qiymatlar orasida interpolyatsiya qilinadi.
                # Chekka nuqtalarda eng yaqin mavjud balandlik ushlab turiladi; barcha balandliklar yo'q bo'lsa, taxmin uydirilmaydi.
                if 'ele' not in df.columns:
                    df['ele'] = np.nan
                df['ele'] = pd.to_numeric(df['ele'], errors='coerce')
                valid_ele = np.isfinite(df['ele'].to_numpy(dtype=float))
                missing_ele_count = int((~valid_ele).sum())
                quality_summary['elevation_values_interpolated'] = missing_ele_count
                df['elevation_interpolated_input'] = ~valid_ele
                if not valid_ele.any():
                    raise ValueError("All elevation values are missing. Provide at least one valid elevation so missing values can be interpolated.")
                if missing_ele_count:
                    ele_values = df['ele'].to_numpy(dtype=float)
                    elapsed_for_ele = (df['time'] - df['time'].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
                    valid_times = elapsed_for_ele[valid_ele]
                    ele_values[~valid_ele] = np.interp(elapsed_for_ele[~valid_ele], valid_times, ele_values[valid_ele])
                    df['ele'] = ele_values
                    st.info(f"Interpolated {missing_ele_count} missing/invalid elevation value(s) using neighboring valid elevations. Leading/trailing gaps use the nearest valid elevation.")

                if len(df) < 3:
                    raise ValueError("Too few points remain after filtering (at least 3 are required).")

                # =====================================================================
                # GPS ANOMALIYALARI VA VAQT UZILISHLARI: asl timestamplar o'zgartirilmaydi
                # =====================================================================
                st.write("⏳ Checking GPS anomalies and time gaps...")
                max_v_ms = float(l_set['max_v']) / 3.6
                lats = df['lat'].to_numpy(dtype=float)
                lons = df['lon'].to_numpy(dtype=float)
                eles = df['ele'].to_numpy(dtype=float)
                if not (np.all(np.isfinite(lats)) and np.all(np.isfinite(lons))):
                    raise ValueError("Latitude/longitude contain non-finite values after cleaning.")
                x_pts, y_pts, z_pts = wgs84_to_ecef(lats, lons, eles, m_set['wgs84_a'], m_set['wgs84_e2'])
                ds_pts = np.sqrt(np.diff(x_pts, prepend=x_pts[0])**2 +
                                 np.diff(y_pts, prepend=y_pts[0])**2 +
                                 np.diff(z_pts, prepend=z_pts[0])**2)
                t_sec_raw = (df['time'] - df['time'].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
                if len(t_sec_raw) and (not np.all(np.isfinite(t_sec_raw)) or np.any(np.diff(t_sec_raw) <= 0)):
                    raise ValueError("Timestamps must be finite and strictly increasing after duplicate handling.")
                duration_s = float(t_sec_raw[-1] - t_sec_raw[0]) if len(t_sec_raw) else 0.0
                if duration_s > MAX_TRAJECTORY_DURATION_S:
                    raise ValueError(f"Trajectory duration {duration_s:.1f} s exceeds the hard limit of {MAX_TRAJECTORY_DURATION_S:.0f} seconds.")
                dt_pts = np.diff(t_sec_raw, prepend=t_sec_raw[0])
                gap_tolerance_sec = float(f_set.get('gap_tolerance', 10.0))
                gap_mask = dt_pts > gap_tolerance_sec
                if np.any(gap_mask):
                    gap_indices = np.flatnonzero(gap_mask)[:10].tolist()
                    raise ValueError(f"Detected {int(gap_mask.sum())} time gap(s) greater than {gap_tolerance_sec:g} s at sample indices {gap_indices}. Processing stops rather than interpolating across unobserved intervals; split the file into continuous segments.")
                # Yolg'iz GPS spike faqat ikkala yon segment limitdan oshib, spike'ni chetlab o'tgan segment fizik jihatdan maqbul bo'lsa olib tashlanadi.
                # Qolgan barcha nuqtalar va asl timestamp'lar o'zgartirilmaydi.
                removed_spikes = 0
                for _ in range(3):
                    if len(df) < 4:
                        break
                    dt_seg = np.diff(t_sec_raw)
                    seg_dist = np.sqrt(np.diff(x_pts)**2 + np.diff(y_pts)**2 + np.diff(z_pts)**2)
                    seg_speed = np.divide(seg_dist, dt_seg, out=np.full_like(seg_dist, np.inf), where=dt_seg > 0)
                    candidates = []
                    for i in range(1, len(df) - 1):
                        if seg_speed[i - 1] <= max_v_ms or seg_speed[i] <= max_v_ms:
                            continue
                        bridge_dt = t_sec_raw[i + 1] - t_sec_raw[i - 1]
                        bridge_dist = np.sqrt((x_pts[i + 1] - x_pts[i - 1])**2 + (y_pts[i + 1] - y_pts[i - 1])**2 + (z_pts[i + 1] - z_pts[i - 1])**2)
                        if bridge_dt > 0 and bridge_dist / bridge_dt <= max_v_ms:
                            candidates.append(i)
                    if not candidates:
                        break
                    # Bir iteratsiyada aniqlangan spike nuqtalari olib tashlanadi; keyingi iteratsiyada barcha hosila massivlar qayta hisoblanadi.
                    df = df.drop(df.index[candidates]).reset_index(drop=True)
                    removed_spikes += len(candidates)
                    if len(df) < 3:
                        raise ValueError("Too few points remain after isolated GPS-spike removal.")
                    lats = df['lat'].to_numpy(dtype=float); lons = df['lon'].to_numpy(dtype=float); eles = df['ele'].to_numpy(dtype=float)
                    x_pts, y_pts, z_pts = wgs84_to_ecef(lats, lons, eles, m_set['wgs84_a'], m_set['wgs84_e2'])
                    t_sec_raw = (df['time'] - df['time'].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
                    dt_pts = np.diff(t_sec_raw, prepend=t_sec_raw[0])
                    if np.any(dt_pts[1:] > gap_tolerance_sec):
                        raise ValueError(f"A time gap greater than {gap_tolerance_sec:g} s remains after GPS-spike filtering. Split the source into continuous segments.")
                quality_summary['isolated_gps_spikes_removed'] = removed_spikes
                if removed_spikes:
                    st.warning(f"Removed {removed_spikes} isolated GPS spike point(s); all retained source timestamps were preserved.")

                dt_pts = np.diff(t_sec_raw, prepend=t_sec_raw[0])
                ds_pts = np.sqrt(np.diff(x_pts, prepend=x_pts[0])**2 + np.diff(y_pts, prepend=y_pts[0])**2 + np.diff(z_pts, prepend=z_pts[0])**2)
                safe_dt = np.where(dt_pts > 0, dt_pts, np.inf)
                segment_speed = np.divide(ds_pts, safe_dt, out=np.zeros_like(ds_pts), where=np.isfinite(safe_dt) & (safe_dt > 0))
                speed_bad = segment_speed > max_v_ms
                if np.any(speed_bad):
                    bad_idx = np.flatnonzero(speed_bad)[:10].tolist()
                    observed_max = float(np.max(segment_speed[speed_bad]) * 3.6)
                    raise ValueError(f"Detected {int(speed_bad.sum())} segment(s) above the configured maximum speed. Highest observed segment speed is {observed_max:.1f} km/h; indices {bad_idx}. Source timestamps were not modified. Review GPS spikes or the configured physical limit.")

                # =====================================================================
                # INTERPOLYATSIYA VA KALMAN FILTRI
                # =====================================================================
                hz = m_set['target_hz']
                st.write(f"⏳ Running PCHIP interpolation and ECEF conversion ({hz} Hz)...")
                
                dt_target = 1.0 / hz
                
                if len(t_sec_raw) < 3 or not np.all(np.isfinite(t_sec_raw)) or np.any(np.diff(t_sec_raw) <= 0):
                    raise ValueError('Timestamps must be finite and strictly increasing after cleaning.')
                duration_s = float(t_sec_raw[-1] - t_sec_raw[0])
                n_intervals = int(np.floor(duration_s * hz + 1e-10))
                estimated_points = n_intervals + 1
                if estimated_points > MAX_OUTPUT_POINTS:
                    raise ValueError(f"Requested sampling would create {estimated_points:,} points, above the safety cap of {MAX_OUTPUT_POINTS:,}. Reduce target frequency or process a shorter track.")
                if n_intervals < 2:
                    raise ValueError("The time interval is too short for at least three output samples at the selected target frequency.")
                # Uniform target gridning oxirgi kuzatuv vaqtini ham qamrab olamiz; effective rate tanlangan limitdan oshmaydi.
                t_target = np.linspace(0.0, duration_s, n_intervals + 1, dtype=float)
                dt_target = float(t_target[1] - t_target[0])
                
                lat_pchip = PchipInterpolator(t_sec_raw, lats)(t_target)
                # Uzunlikni unwrap qilish antimeridian (-180/180°) orqali noto'g'ri interpolatsiyani oldini oladi.
                lon_unwrapped = np.degrees(np.unwrap(np.radians(lons)))
                lon_pchip = (PchipInterpolator(t_sec_raw, lon_unwrapped)(t_target) + 180.0) % 360.0 - 180.0
                ele_pchip = PchipInterpolator(t_sec_raw, eles)(t_target)

                x_raw, y_raw, z_raw = wgs84_to_ecef(lat_pchip, lon_pchip, ele_pchip, m_set['wgs84_a'], m_set['wgs84_e2'])

                st.write("⏳ Running the 9D Kalman filter (Numba JIT engine)...")
                state, raw_v_arr, zupt_applied = run_kalman_filter_9d(
                    x_raw, y_raw, z_raw, dt_target, 
                    m_set['kalman_r'], m_set['kalman_q'], 
                    l_set['zupt_lim'], m_set['zupt_damping']
                )
                if not np.all(np.isfinite(state)) or not np.all(np.isfinite(raw_v_arr)):
                    raise ValueError("Kalman filter returned non-finite values. Check GPS quality and noise settings.")

                df_res = pd.DataFrame({'time_sec': t_target})
                # Har bir qayta namunalangan nuqtaga vaqt bo'yicha eng yaqin manba balandligi sifat flagini biriktiramiz.
                src_elev_flag = df['elevation_interpolated_input'].to_numpy(dtype=bool)
                right_idx = np.clip(np.searchsorted(t_sec_raw, t_target, side='left'), 0, len(t_sec_raw) - 1)
                left_idx = np.clip(right_idx - 1, 0, len(t_sec_raw) - 1)
                nearest_idx = np.where(np.abs(t_sec_raw[left_idx] - t_target) <= np.abs(t_sec_raw[right_idx] - t_target), left_idx, right_idx)
                df_res['input_elevation_interpolated'] = src_elev_flag[nearest_idx].astype(bool)
                df_res['zupt_update_applied'] = zupt_applied.astype(bool)
                df_res['velocity_raw_kmh'] = raw_v_arr * 3.6
                df_res['x_m'], df_res['y_m'], df_res['z_m'] = state[0, :], state[1, :], state[2, :]
                vx, vy, vz = state[3, :], state[4, :], state[5, :]
                ax_ecef, ay_ecef, az_ecef = state[6, :], state[7, :], state[8, :]
                df_res['vx_ecef_ms'], df_res['vy_ecef_ms'], df_res['vz_ecef_ms'] = vx, vy, vz
                df_res['ax_ecef_ms2'], df_res['ay_ecef_ms2'], df_res['az_ecef_ms2'] = ax_ecef, ay_ecef, az_ecef
                
                raw_vel = np.sqrt(vx**2 + vy**2 + vz**2)
                df_res['velocity_ms'] = np.maximum(raw_vel, 0.0)
                df_res['speed_limit_exceeded'] = df_res['velocity_ms'] > max_v_ms
                df_res['velocity_kmh'] = df_res['velocity_ms'] * 3.6
                
                lat_smooth, lon_smooth, ele_smooth = ecef_to_wgs84(df_res['x_m'].values, df_res['y_m'].values, df_res['z_m'].values, m_set['wgs84_a'], m_set['wgs84_e2'])
                df_res['lat_smooth'], df_res['lon_smooth'], df_res['elevation_m'] = lat_smooth, lon_smooth, ele_smooth
                # Masofa filtered ECEF pozitsiyasidan qayta hisoblanadi; raw input bilan output tezligi aralashmaydi.
                lat_r = np.radians(lat_smooth); lon_r = np.radians(lon_smooth)
                dlat = np.diff(lat_r, prepend=lat_r[0]); dlon = np.diff(lon_r, prepend=lon_r[0])
                dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi
                lat_prev = np.roll(lat_r, 1)
                hav = np.sin(dlat/2.0)**2 + np.cos(lat_r) * np.cos(lat_prev) * np.sin(dlon/2.0)**2
                hav[0] = 0.0
                horizontal = 2.0 * 6371008.8 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
                vertical = np.diff(ele_smooth, prepend=ele_smooth[0])
                dist_diff = np.sqrt(horizontal**2 + vertical**2)
                df_res['dist_m'] = np.cumsum(np.maximum(dist_diff, 0.0))
                x_enu, y_enu, z_enu = ecef_to_enu_position(df_res['x_m'].values, df_res['y_m'].values, df_res['z_m'].values, lat_smooth[0], lon_smooth[0])
                df_res['east_m'], df_res['north_m'], df_res['up_m'] = x_enu, y_enu, z_enu

                v_east, v_north, v_up = ecef_to_enu_vel(vx, vy, vz, df_res['lat_smooth'].values, df_res['lon_smooth'].values)
                df_res['v_east_ms'], df_res['v_north_ms'], df_res['v_up_ms'] = v_east, v_north, v_up
                a_east, a_north, a_up = ecef_to_enu_vel(ax_ecef, ay_ecef, az_ecef, df_res['lat_smooth'].values, df_res['lon_smooth'].values)
                df_res['accel_east_ms2'], df_res['accel_north_ms2'], df_res['accel_up_ms2'] = a_east, a_north, a_up
                df_res['accel_vector_ms2'] = np.sqrt(a_east**2 + a_north**2 + a_up**2)

                vel_sq = v_east**2 + v_north**2
                valid_heading = vel_sq >= 1e-4
                heading_estimated = ~valid_heading
                raw_head = np.arctan2(v_east[valid_heading], v_north[valid_heading])
                if np.any(valid_heading):
                    valid_idx = np.flatnonzero(valid_heading)
                    raw_head_unwrapped = np.unwrap(raw_head)
                    heading_filled = np.interp(np.arange(len(df_res), dtype=float), valid_idx, raw_head_unwrapped)
                    heading_filled = (heading_filled + np.pi) % (2 * np.pi) - np.pi
                else:
                    heading_filled = np.zeros(len(df_res), dtype=float)
                df_res['heading_estimated'] = heading_estimated.astype(bool)
                df_res['heading_rad'] = heading_filled
                df_res['heading_deg'] = (np.degrees(df_res['heading_rad']) + 360) % 360
                
                # Savitzky-Golay filtri artefaktlarini tozalash (mode='nearest')
                sg_win, sg_poly = m_set['sg_window'], m_set['sg_polyorder']
                acc_raw = np.gradient(df_res['velocity_ms'], dt_target)
                jerk_raw = np.gradient(acc_raw, dt_target)
                
                if len(df_res) > sg_win:
                    df_res['accel_long_ms2'] = savgol_filter(acc_raw, sg_win, sg_poly, mode='nearest')
                    df_res['jerk_ms3'] = savgol_filter(jerk_raw, sg_win, sg_poly, mode='nearest')
                else:
                    df_res['accel_long_ms2'], df_res['jerk_ms3'] = acc_raw, jerk_raw

                df_res['accel_limit_exceeded'] = (df_res['accel_long_ms2'] > l_set['max_a']) | (df_res['accel_long_ms2'] < -l_set['max_b'])
                df_res['jerk_limit_exceeded'] = np.abs(df_res['jerk_ms3']) > l_set['max_jerk']
                
                unwrapped_heading = np.unwrap(df_res['heading_rad'])
                omega_raw = np.gradient(unwrapped_heading, dt_target)
                omega_estimate = savgol_filter(omega_raw, sg_win, sg_poly, mode='nearest') if len(df_res) > sg_win else omega_raw
                omega_smooth = omega_estimate.copy()
                omega_smooth[heading_estimated] = np.nan
                    
                df_res['angular_vel_rad_s'], df_res['angular_vel_degs'] = omega_smooth, np.degrees(omega_smooth)
                
                abs_omega = np.abs(omega_smooth)
                df_res['turn_radius_m'] = np.divide(np.abs(df_res['velocity_ms']), abs_omega, out=np.full(len(df_res), np.nan), where=abs_omega >= float(l_set['safe_omega']))
                df_res['turn_radius_limit_exceeded'] = np.isfinite(df_res['turn_radius_m']) & (df_res['turn_radius_m'] > float(l_set['max_radius']))
                df_res['turn_radius_display_m'] = np.where(np.isfinite(df_res['turn_radius_m']), np.minimum(df_res['turn_radius_m'], float(l_set['max_radius'])), float(l_set['max_radius']))
                
                alat_raw = np.abs(df_res['velocity_ms'] * omega_estimate)
                alat_filtered = savgol_filter(alat_raw, sg_win, sg_poly, mode='nearest') if len(df_res) > sg_win else alat_raw
                alat_filtered[heading_estimated] = np.nan
                df_res['accel_lat_ms2'] = alat_filtered
                df_res['lateral_accel_limit_exceeded'] = df_res['accel_lat_ms2'] > l_set['max_alat']

                # =====================================================================
                # TOPOGRAFIYA VA ATMOSFERA (ISA) HISOB-KITOBI
                # =====================================================================
                st.write("⏳ Calculating terrain gradients and ISA atmospheric values...")
                
                # Dinamik step_idx ning sakrashini oldini olish
                safe_avg_v = max(0.5, df_res['velocity_ms'].mean()) 
                step_idx = int(max(1, min(len(df_res)//10, t_set['spatial_step'] / (dt_target * safe_avg_v))))
                
                dz_step = df_res['elevation_m'] - df_res['elevation_m'].shift(step_idx)
                ds_step = df_res['dist_m'] - df_res['dist_m'].shift(step_idx)
                safe_ds = np.where((ds_step.isna()) | (ds_step == 0), 1e-9, ds_step)

                R_spec = 287.058
                T0_k = t_set['base_temp'] + 273.15

                if t_set.get('enabled', True):
                    df_res['grade_percent'] = (dz_step / safe_ds).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
                    df_res['grade_deg'] = np.degrees(np.arctan(df_res['grade_percent'] / 100.0))
                    df_res['grade_limit_exceeded'] = np.abs(df_res['grade_deg']) > float(t_set['max_topo_grad'])
                else:
                    df_res['grade_percent'] = 0.0
                    df_res['grade_deg'] = 0.0
                    df_res['grade_limit_exceeded'] = False

                if t_set.get('atmosphere_enabled', True):
                    h0, P0 = float(t_set['base_alt']), float(t_set['base_press'])
                    L, g, R, M = 0.0065, 9.80665, 8.31446, 0.0289644
                    T_k = T0_k - L * (df_res['elevation_m'].to_numpy(dtype=float) - h0)
                    elevation_arr = df_res['elevation_m'].to_numpy(dtype=float)
                    isa_valid = np.isfinite(T_k) & (T_k > 0) & (elevation_arr >= -500) & (elevation_arr <= 11000)
                    air_temp = np.full(len(df_res), np.nan); air_press = np.full(len(df_res), np.nan); air_density = np.full(len(df_res), np.nan)
                    air_temp[isa_valid] = T_k[isa_valid] - 273.15
                    air_press[isa_valid] = P0 * ((T_k[isa_valid] / T0_k) ** ((g * M) / (R * L)))
                    air_density[isa_valid] = air_press[isa_valid] / (R_spec * T_k[isa_valid])
                    df_res['air_temp_C'] = air_temp
                    df_res['air_press_Pa'] = air_press
                    df_res['air_density_kgm3'] = air_density
                    if not np.all(isa_valid):
                        st.warning("ISA values are shown as missing outside the approximate model altitude range (-500 to 11,000 m) or where absolute temperature is invalid.")
                else:
                    df_res['air_temp_C'] = np.nan
                    df_res['air_press_Pa'] = np.nan
                    df_res['air_density_kgm3'] = np.nan

                wind_spd, wind_dir = t_set['wind_speed'], np.radians(t_set['wind_direction_deg'])
                rel_angle = df_res['heading_rad'] - wind_dir
                df_res['wind_head_ms'] = wind_spd * np.cos(rel_angle)
                df_res['wind_cross_ms'] = wind_spd * np.sin(rel_angle)

                max_dist_meters = float(l_set.get('max_dist_km', 500.0)) * 1000.0
                if df_res['dist_m'].iloc[-1] > max_dist_meters:
                    st.info(f"📏 The trajectory was truncated at the configured maximum distance ({l_set['max_dist_km']} km).")
                    df_res = df_res[df_res['dist_m'] <= max_dist_meters].reset_index(drop=True)
                    if len(df_res) < 3:
                        raise ValueError("The configured maximum distance leaves fewer than three output points. Increase the distance limit.")

                # =====================================================================
                # MASOFA DOMENIGA O'TISH (PROJECTION & CRASH FIX)
                # =====================================================================
                calc_type = "Time domain"
                st.session_state.calc_mode = 'time'
                
                if run_dist:
                    st.write("⏳ Re-projecting results onto the distance domain...")
                    
                    # Azimutni Aylanma (Circular) O'rtacha bilan hisoblash
                    df_res['heading_sin'] = np.sin(df_res['heading_rad'])
                    df_res['heading_cos'] = np.cos(df_res['heading_rad'])
                    boolean_cols = [c for c in df_res.columns if pd.api.types.is_bool_dtype(df_res[c]) or c.endswith('_exceeded') or c in ('input_elevation_interpolated', 'heading_estimated', 'zupt_update_applied')]
                    
                    df_unique = df_res.groupby('dist_m', as_index=False).mean(numeric_only=True)
                    
                    df_unique['heading_rad'] = np.arctan2(df_unique['heading_sin'], df_unique['heading_cos'])
                    df_unique['heading_deg'] = (np.degrees(df_unique['heading_rad']) + 360) % 360
                    
                    # Groupby dist_m allaqachon takroriy masofalarni birlashtiradi; fizik masofaga epsilon qo'shilmaydi.
                    dist_vals = df_unique['dist_m'].to_numpy(dtype=float)
                    if len(dist_vals) > 1 and np.any(np.diff(dist_vals) <= 0):
                        raise ValueError("Distance values must be strictly increasing after duplicate-distance aggregation.")

                    if len(df_unique) > 3:
                        max_d = np.floor(df_unique['dist_m'].max())
                        dist_new = np.arange(0, max_d + 1, 1.0)
                        df_spatial = pd.DataFrame({'dist_m': dist_new})
                        
                        for col in df_unique.columns:
                            if col != 'dist_m' and col not in ('heading_rad', 'heading_deg', 'heading_sin', 'heading_cos', 'lat_smooth', 'lon_smooth', 'elevation_m'):
                                values = df_unique[col].to_numpy()
                                if col in boolean_cols or pd.api.types.is_bool_dtype(values.dtype):
                                    df_spatial[col] = np.interp(dist_new, df_unique['dist_m'].values, values.astype(float)) >= 0.5
                                elif np.issubdtype(values.dtype, np.number) and np.all(np.isfinite(values)):
                                    df_spatial[col] = PchipInterpolator(df_unique['dist_m'].values, values)(dist_new)
                                else:
                                    df_spatial[col] = np.interp(dist_new, df_unique['dist_m'].values, values.astype(float))
                        sin_i = PchipInterpolator(df_unique['dist_m'].values, df_unique['heading_sin'].values)(dist_new)
                        cos_i = PchipInterpolator(df_unique['dist_m'].values, df_unique['heading_cos'].values)(dist_new)
                        df_spatial['heading_rad'] = np.arctan2(sin_i, cos_i)
                        df_spatial['heading_deg'] = (np.degrees(df_spatial['heading_rad']) + 360.0) % 360.0
                        lat_d, lon_d, ele_d = ecef_to_wgs84(df_spatial['x_m'].to_numpy(), df_spatial['y_m'].to_numpy(), df_spatial['z_m'].to_numpy(), m_set['wgs84_a'], m_set['wgs84_e2'])
                        df_spatial['lat_smooth'], df_spatial['lon_smooth'], df_spatial['elevation_m'] = lat_d, lon_d, ele_d
                        # Distance-domain natijalari asosiy holat vektorlaridan qayta hosil qilinadi.
                        ve, vn, vu = ecef_to_enu_vel(df_spatial['vx_ecef_ms'].to_numpy(), df_spatial['vy_ecef_ms'].to_numpy(), df_spatial['vz_ecef_ms'].to_numpy(), lat_d, lon_d)
                        ae, an, au = ecef_to_enu_vel(df_spatial['ax_ecef_ms2'].to_numpy(), df_spatial['ay_ecef_ms2'].to_numpy(), df_spatial['az_ecef_ms2'].to_numpy(), lat_d, lon_d)
                        df_spatial['v_east_ms'], df_spatial['v_north_ms'], df_spatial['v_up_ms'] = ve, vn, vu
                        df_spatial['accel_east_ms2'], df_spatial['accel_north_ms2'], df_spatial['accel_up_ms2'] = ae, an, au
                        df_spatial['accel_vector_ms2'] = np.sqrt(ae**2 + an**2 + au**2)
                        df_spatial['velocity_ms'] = np.sqrt(ve**2 + vn**2 + vu**2)
                        df_spatial['velocity_kmh'] = df_spatial['velocity_ms'] * 3.6
                        speed_horiz_sq = ve**2 + vn**2
                        low_speed_heading = speed_horiz_sq < 1e-4
                        heading_dist = np.arctan2(ve, vn)
                        heading_dist[low_speed_heading] = df_spatial['heading_rad'].to_numpy()[low_speed_heading]
                        df_spatial['heading_estimated'] = low_speed_heading.astype(bool)
                        df_spatial['heading_rad'] = heading_dist
                        df_spatial['heading_deg'] = (np.degrees(heading_dist) + 360.0) % 360.0
                        if len(df_spatial) >= 3 and np.all(np.isfinite(df_spatial['time_sec'])) and np.all(np.diff(df_spatial['time_sec']) > 0):
                            dt_dist = df_spatial['time_sec'].to_numpy(dtype=float)
                            acc_dist = np.gradient(df_spatial['velocity_ms'].to_numpy(dtype=float), dt_dist)
                            jerk_dist = np.gradient(acc_dist, dt_dist)
                            df_spatial['accel_long_ms2'] = savgol_filter(acc_dist, sg_win, sg_poly, mode='nearest') if len(df_spatial) > sg_win else acc_dist
                            df_spatial['jerk_ms3'] = savgol_filter(jerk_dist, sg_win, sg_poly, mode='nearest') if len(df_spatial) > sg_win else jerk_dist
                            omega_dist = np.gradient(np.unwrap(df_spatial['heading_rad'].to_numpy(dtype=float)), dt_dist)
                            omega_dist = savgol_filter(omega_dist, sg_win, sg_poly, mode='nearest') if len(df_spatial) > sg_win else omega_dist
                            omega_dist[low_speed_heading] = np.nan
                            df_spatial['angular_vel_rad_s'], df_spatial['angular_vel_degs'] = omega_dist, np.degrees(omega_dist)
                            df_spatial['accel_lat_ms2'] = np.abs(df_spatial['velocity_ms'].to_numpy(dtype=float) * np.nan_to_num(omega_dist, nan=0.0))
                            df_spatial.loc[low_speed_heading, 'accel_lat_ms2'] = np.nan
                            abs_omega_dist = np.abs(omega_dist)
                            df_spatial['turn_radius_m'] = np.divide(df_spatial['velocity_ms'].to_numpy(dtype=float), abs_omega_dist, out=np.full(len(df_spatial), np.nan), where=abs_omega_dist >= float(l_set['safe_omega']))
                            df_spatial['turn_radius_display_m'] = np.where(np.isfinite(df_spatial['turn_radius_m']), np.minimum(df_spatial['turn_radius_m'], l_set['max_radius']), l_set['max_radius'])
                            df_spatial['turn_radius_limit_exceeded'] = np.isfinite(df_spatial['turn_radius_m']) & (df_spatial['turn_radius_m'] > l_set['max_radius'])
                        df_spatial['speed_limit_exceeded'] = df_spatial['velocity_ms'] > max_v_ms
                        df_spatial['accel_limit_exceeded'] = (df_spatial['accel_long_ms2'] > l_set['max_a']) | (df_spatial['accel_long_ms2'] < -l_set['max_b'])
                        df_spatial['jerk_limit_exceeded'] = np.abs(df_spatial['jerk_ms3']) > l_set['max_jerk']
                        df_spatial['lateral_accel_limit_exceeded'] = df_spatial['accel_lat_ms2'] > l_set['max_alat']
                        df_spatial['turn_radius_limit_exceeded'] = np.isfinite(df_spatial['turn_radius_m']) & (df_spatial['turn_radius_m'] > l_set['max_radius'])
                        df_spatial['turn_radius_display_m'] = np.where(np.isfinite(df_spatial['turn_radius_m']), np.minimum(df_spatial['turn_radius_m'], l_set['max_radius']), l_set['max_radius'])
                        # Qiyalik 1 m masofa qatoridan spatial_step oralig'ida qayta hisoblanadi.
                        if t_set.get('enabled', True):
                            step_m = max(1, int(round(float(t_set['spatial_step']))))
                            dz = df_spatial['elevation_m'] - df_spatial['elevation_m'].shift(step_m)
                            ds = df_spatial['dist_m'] - df_spatial['dist_m'].shift(step_m)
                            grade_pct = (dz / ds.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            df_spatial['grade_percent'] = grade_pct
                            df_spatial['grade_deg'] = np.degrees(np.arctan(grade_pct / 100.0))
                        else:
                            df_spatial['grade_percent'], df_spatial['grade_deg'] = 0.0, 0.0
                        df_spatial['grade_limit_exceeded'] = np.abs(df_spatial['grade_deg']) > t_set['max_topo_grad'] if t_set.get('enabled', True) else False
                        # Shamol va ISA qiymatlari ham distance-domain koordinatalaridan qayta olinadi.
                        rel_dist = df_spatial['heading_rad'] - np.radians(t_set['wind_direction_deg'])
                        df_spatial['wind_head_ms'] = float(t_set['wind_speed']) * np.cos(rel_dist)
                        df_spatial['wind_cross_ms'] = float(t_set['wind_speed']) * np.sin(rel_dist)
                        if t_set.get('atmosphere_enabled', True):
                            T0_dist = float(t_set['base_temp']) + 273.15
                            T_dist = T0_dist - 0.0065 * (df_spatial['elevation_m'].to_numpy(dtype=float) - float(t_set['base_alt']))
                            h_dist = df_spatial['elevation_m'].to_numpy(dtype=float)
                            ok_isa = np.isfinite(T_dist) & (T_dist > 0) & (h_dist >= -500) & (h_dist <= 11000)
                            press_dist = np.full(len(df_spatial), np.nan)
                            temp_dist = np.full(len(df_spatial), np.nan)
                            density_dist = np.full(len(df_spatial), np.nan)
                            temp_dist[ok_isa] = T_dist[ok_isa] - 273.15
                            press_dist[ok_isa] = float(t_set['base_press']) * ((T_dist[ok_isa] / T0_dist) ** ((9.80665 * 0.0289644) / (8.31446 * 0.0065)))
                            density_dist[ok_isa] = press_dist[ok_isa] / (287.058 * T_dist[ok_isa])
                            df_spatial['air_temp_C'], df_spatial['air_press_Pa'], df_spatial['air_density_kgm3'] = temp_dist, press_dist, density_dist
                        df_spatial = df_spatial.drop(columns=['heading_sin', 'heading_cos'], errors='ignore')
                        lat_d, lon_d, ele_d = ecef_to_wgs84(df_spatial['x_m'].to_numpy(), df_spatial['y_m'].to_numpy(), df_spatial['z_m'].to_numpy(), m_set['wgs84_a'], m_set['wgs84_e2'])
                        df_spatial['lat_smooth'], df_spatial['lon_smooth'], df_spatial['elevation_m'] = lat_d, lon_d, ele_d
                        east_d, north_d, up_d = ecef_to_enu_position(df_spatial['x_m'].to_numpy(), df_spatial['y_m'].to_numpy(), df_spatial['z_m'].to_numpy(), lat_d[0], lon_d[0])
                        df_spatial['east_m'], df_spatial['north_m'], df_spatial['up_m'] = east_d, north_d, up_d
                        if 'grade_percent' in df_spatial:
                            df_spatial['grade_deg'] = np.degrees(np.arctan(df_spatial['grade_percent'] / 100.0))
                        df_res = df_spatial
                        calc_type = "Distance domain (1 m step)"
                        st.session_state.calc_mode = 'dist'
                    else:
                        st.warning("Insufficient data for distance-domain processing. Results remain in the time domain.")

                offset_text = str(f_set.get('utc_shift', 'UTC+0'))
                try:
                    offset_hours = int(offset_text.replace('UTC', ''))
                except ValueError:
                    offset_hours = 0
                df_res['timestamp_utc'] = source_time_origin + pd.to_timedelta(df_res['time_sec'].to_numpy(dtype=float), unit='s')
                df_res['timestamp_display'] = df_res['timestamp_utc'] + pd.to_timedelta(offset_hours, unit='h')
                quality_summary['zupt_update_samples'] = int(df_res['zupt_update_applied'].sum()) if 'zupt_update_applied' in df_res.columns else 0
                quality_summary['output_points'] = int(len(df_res))
                quality_summary['duration_seconds'] = float(df_res['time_sec'].max()) if 'time_sec' in df_res.columns else 0.0
                quality_summary['calculation_mode'] = calc_type
                quality_summary['time_grid_effective_rate_hz'] = float(1.0 / dt_target)
                st.session_state.processing_quality_report = quality_summary
                st.session_state.processed_data = df_res
                status.update(label=f"✅ Calculation completed successfully: {len(df_res):,} points.", state="complete", expanded=False)
                st.success(f"The engine completed processing in {calc_type} mode. Review the charts in Block 3.")
                
            except Exception as e:
                status.update(label="❌ An error occurred!", state="error", expanded=True)
                st.error(f"The engine stopped safely: {str(e)}")
# =====================================================================
# 2-BLOK OXIRI
# =====================================================================
# =====================================================================
# 3-BLOK BOSHI: VIZUALIZATSIYA VA EKSPORT (Optimallashtirilgan)
# =====================================================================
def generate_static_plot(df, x_col, y_col, title, y_label, dpi=300, fmt='png'):
    """Statik grafiklarni Thread-Safe usulda shakllantirish (plt.close('all') o'chirildi)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(y_col, list):
        for col, label in y_col:
            ax.plot(df[x_col], df[col], label=label, linewidth=1.5)
        ax.legend()
    else:
        ax.plot(df[x_col], df[y_col], color='navy', linewidth=1.5)
    
    x_label_text = 'Distance (m)' if x_col == 'dist_m' else 'Time (s)'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_label_text, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close(fig) # Faqat joriy thread'dagi figura yopiladi
    gc.collect()   
    return buf.getvalue()

def generate_transparent_trajectory(df, x_col='x_m', y_col='y_m', color_col='velocity_kmh', dpi=300):
    """
    Paint va boshqa tahrirlovchilar uchun foni shaffof, faqat chiziq va tezlik rangi ko'rinadigan grafik.
    Geometriya to'g'ri bo'lishi uchun east_m va north_m lokal ENU koordinatalari ishlatiladi.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap='jet', s=8, alpha=1.0, edgecolors='none')
    
    ax.axis('off') # O'qlarni yashirish
    fig.patch.set_alpha(0.0) # Figura foni shaffof
    ax.patch.set_alpha(0.0)  # O'q foni shaffof
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    gc.collect()
    return buf.getvalue()

def downsample_preserve_extrema(df, max_points, x_col, important_cols=None):
    """Grafiklar uchun namunani kamaytiradi, lekin tanlangan muhim qatorlarning lokal min/max nuqtalarini saqlaydi."""
    n = len(df)
    max_points = max(3, int(max_points))
    if n <= max_points:
        return df.copy()
    important_cols = important_cols or []
    cols = [c for c in important_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        positions = np.linspace(0, n - 1, max_points, dtype=int)
        return df.iloc[np.unique(positions)].copy()
    bins = max(1, (max_points - 2) // (2 * len(cols) + 2))
    keep = {0, n - 1}
    for chunk in np.array_split(np.arange(n), bins):
        if len(chunk) == 0:
            continue
        keep.add(int(chunk[0])); keep.add(int(chunk[-1]))
        for col in cols:
            vals = pd.to_numeric(df.iloc[chunk][col], errors='coerce').to_numpy(dtype=float)
            valid = np.flatnonzero(np.isfinite(vals))
            if len(valid):
                keep.add(int(chunk[valid[np.argmin(vals[valid])]]))
                keep.add(int(chunk[valid[np.argmax(vals[valid])]]))
    positions = np.array(sorted(keep), dtype=int)
    if len(positions) > max_points:
        # Himoya: ekstremumlarni tanlash odatdagi holatda limit ichida qoladi; patologik ustunlar ko'p bo'lsa, teng masofali yakuniy namuna.
        positions = positions[np.linspace(0, len(positions) - 1, max_points, dtype=int)]
    return df.iloc[positions].copy()


def render_block_3():
    st.divider()
    
    # Xotirani tozalash paneli (Memory Leak himoyasi)
    col_title, col_ram = st.columns([3, 1])
    with col_title:
        st.markdown("## 📊 BLOCK 3: Analysis, Visualization & Export")
    with col_ram:
        if st.button("🧹 Clear Memory (RAM)", key="ui_clear_memory_ram", help="Clear the processed results and export archive from this session. The uploaded source file and confirmed settings are retained."):
            keys_to_clear = ['processed_data', 'zip_ready', 'processing_quality_report']
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            gc.collect()
            st.success("Memory cleared. Use Block 2 to run the calculation again.")
            st.rerun()

    if 'processed_data' not in st.session_state:
        st.info("Run a calculation in Block 2 to get started.")
        return
        
    df_full = st.session_state.processed_data
    quality_report = st.session_state.get('processing_quality_report')
    if quality_report:
        with st.expander("Data Quality & Processing Report", expanded=False):
            st.json(quality_report)
    calc_mode = st.session_state.get('calc_mode', 'time')
    x_col = 'dist_m' if calc_mode == 'dist' else 'time_sec'
    
    st.markdown("### 🎛️ Key Kinematic Indicators (KPIs)")
    max_v_ms = df_full['velocity_ms'].max()
    avg_v_ms = df_full['velocity_ms'].mean()
    tot_dist_m = df_full['dist_m'].max()
    tot_time_s = df_full['time_sec'].max() if 'time_sec' in df_full.columns else 0.0
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Maximum Speed", f"{max_v_ms*3.6:.1f} km/h", f"{max_v_ms:.1f} m/s")
    if 'speed_limit_exceeded' in df_full.columns and df_full['speed_limit_exceeded'].any():
        st.warning(f"Speed limit exceeded at {int(df_full['speed_limit_exceeded'].sum())} calculated samples. Values are shown without clipping.")
    if calc_mode == 'time':
        kpi2.metric("Mean Speed", f"{avg_v_ms*3.6:.1f} km/h", f"{avg_v_ms:.1f} m/s")
    else:
        st.caption("In distance-domain mode, this mean is a spatial sample mean, not trip-average speed.")
        kpi2.metric("Spatial Mean Speed", f"{avg_v_ms*3.6:.1f} km/h", f"{avg_v_ms:.1f} m/s")
    kpi3.metric("Total Distance", f"{tot_dist_m/1000:.2f} km", f"{tot_dist_m:.0f} m")
    
    if calc_mode == 'time':
        kpi4.metric("Total Time", f"{tot_time_s/60:.1f} daq", f"{tot_time_s:.0f} s")
    else:
        kpi4.metric("Calculation Mode", "Distance Domain (1 m)", "Time axis hidden")

    limits = st.session_state.get('limits_settings', {})
    max_pts = limits.get('max_plot_points', 3000)
    total_pts = len(df_full)
    
    important_plot_cols = ['velocity_kmh', 'velocity_raw_kmh', 'accel_long_ms2', 'dist_m', 'grade_deg',
                           'turn_radius_display_m', 'heading_deg', 'angular_vel_degs', 'jerk_ms3',
                           'accel_lat_ms2', 'elevation_m', 'air_temp_C', 'wind_head_ms', 'wind_cross_ms',
                           'air_density_kgm3', 'air_press_Pa']
    df = downsample_preserve_extrema(df_full, max_pts, x_col, important_plot_cols)
    step = max(1, int(np.ceil(total_pts / max(1, len(df)))))
    if len(df) < total_pts:
        st.caption(f"⚡ RAM optimization: displaying {len(df):,} extrema-preserving samples from {total_pts:,} output points.")

    st.markdown("### 📈 Interactive Charts")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Kinematics", "🧭 Dynamics", "🌡️ Environment", "🔬 Filtering & 3D", "🌍 Interactive Map"
    ])

    def create_line_fig(y_col, title, color_seq=None):
        fig = px.line(
            df, x=x_col, y=y_col, 
            title=title, 
            color_discrete_sequence=color_seq,
            render_mode='svg' 
        )
        fig.update_traces(connectgaps=False)
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        return fig

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(create_line_fig('velocity_kmh', "1. Speed (km/h)", ['#00CC96']))
        with c2: st.plotly_chart(create_line_fig('accel_long_ms2', "2. Longitudinal Acceleration (m/s²)", ['#EF553B']))
        with c3: 
            y_param = 'dist_m' if calc_mode == 'time' else 'velocity_ms'
            title_param = "3. Distance Travelled (m)" if calc_mode == 'time' else "3. Speed (m/s)"
            st.plotly_chart(create_line_fig(y_param, title_param, ['#636EFA']))

    with tab2:
        c4, c5, c6 = st.columns(3)
        with c4:
            st.plotly_chart(create_line_fig('grade_deg', "4. Road Grade (degrees)"))
            st.plotly_chart(create_line_fig('turn_radius_display_m', "8. Turning Radius (m; capped for display)"))
        with c5:
            st.plotly_chart(create_line_fig('heading_deg', "7. Heading (degrees)"))
            st.plotly_chart(create_line_fig('angular_vel_degs', "9. Angular Velocity (deg/s)"))
        with c6:
            st.plotly_chart(create_line_fig('jerk_ms3', "16. Jerk (m/s³)"))
            st.plotly_chart(create_line_fig('accel_lat_ms2', "10. Lateral Acceleration (m/s²)"))

    with tab3:
        c7, c8, c9 = st.columns(3)
        with c7:
            st.plotly_chart(create_line_fig('elevation_m', "11. Elevation Profile (m)"))
            st.plotly_chart(create_line_fig('air_temp_C', "15. Air Temperature (°C)"))
        with c8:
            st.plotly_chart(create_line_fig('wind_head_ms', "5. Headwind (m/s)"))
            st.plotly_chart(create_line_fig('air_density_kgm3', "14. Air Density (kg/m³)"))
        with c9:
            st.plotly_chart(create_line_fig('wind_cross_ms', "6. Crosswind (m/s)"))
            st.plotly_chart(create_line_fig('air_press_Pa', "17. Atmospheric Pressure (Pa)"))

    with tab4:
        c10, c11 = st.columns(2)
        with c10:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df[x_col], y=df['velocity_raw_kmh'], mode='lines', name='Raw speed', connectgaps=False, line=dict(color='gray', width=1, dash='dot')))
            fig_comp.add_trace(go.Scatter(x=df[x_col], y=df['velocity_kmh'], mode='lines', name='Filtered', connectgaps=False, line=dict(color='blue', width=2)))
            fig_comp.update_layout(title="12. Raw vs. Filtered Speed", hovermode="x unified")
            st.plotly_chart(fig_comp)
        with c11:
            fig_3d = go.Figure(data=go.Scatter3d(
                x=df['east_m'], y=df['north_m'], z=df['up_m'],
                mode='lines', line=dict(color='#00CC96', width=3)
            ))
            fig_3d.update_layout(title="13. 3D Trajectory (local ENU)", scene=dict(xaxis_title='East (m)', yaxis_title='North (m)', zaxis_title='Up (m)'), margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_3d)
            
    with tab5:
        st.markdown("**18. Geospatial Map (Crash-Protected)**")
        map_color_opt = st.selectbox("Color the map by:",
                                     key="ui_map_color_by",
                                     options=['velocity_kmh', 'grade_deg'],
                                     format_func=lambda x: {"velocity_kmh": "Speed (km/h)", "grade_deg": "Slope (degrees)"}[x],
                                     help="Choose whether map points are colored by calculated speed or terrain slope. Other variables remain available in the charts.")

        max_map_points = 1000
        map_step = max(1, int(np.ceil(len(df_full) / max_map_points)))
        df_map = df_full.iloc[::map_step].copy()
        
        # Plotly muvofiqligi (Version Compatibility Fallback)
        if hasattr(px, 'scatter_map'):
            fig_map = px.scatter_map(
                df_map, lat="lat_smooth", lon="lon_smooth", color=map_color_opt,
                color_continuous_scale=px.colors.sequential.Viridis,
                hover_data=[x_col, 'velocity_kmh', 'grade_deg', 'elevation_m'],
                title=f"18. Interactive Map — {'Speed (km/h)' if map_color_opt == 'velocity_kmh' else 'Slope (degrees)'}"
            )
            fig_map.update_layout(map_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
        else:
            fig_map = px.scatter_mapbox(
                df_map, lat="lat_smooth", lon="lon_smooth", color=map_color_opt,
                color_continuous_scale=px.colors.sequential.Viridis,
                hover_data=[x_col, 'velocity_kmh', 'grade_deg', 'elevation_m'],
                title=f"18. Interactive Map — {'Speed (km/h)' if map_color_opt == 'velocity_kmh' else 'Slope (degrees)'}"
            )
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
            
        st.plotly_chart(fig_map)

    # -----------------------------------------------------------------
    # EKSPORT VA ZIP SHAKLLANTIRISH
    # -----------------------------------------------------------------
    st.divider()
    st.markdown("### 📦 Export Center")
    
    col_name, col_fmt, col_dpi = st.columns([2, 1, 1])
    with col_name:
        zip_filename = st.text_input("Archive filename:", value="QwatLab_EV_EGAD", key="ui_archive_filename", help="Name for the downloadable ZIP file. The .zip extension is added automatically.")
    with col_fmt:
        export_fmt = st.selectbox("Table format:", ["CSV", "XLSX"], key="ui_table_format", help="Choose CSV for a lightweight table or XLSX for an Excel workbook.")
    with col_dpi:
        img_dpi = st.selectbox("Chart resolution (DPI):", [300, 600], key="ui_chart_resolution_dpi", help="Higher DPI produces sharper static charts but increases archive size and export time.")
        img_fmt = st.selectbox("Chart format:", ["png", "eps", "pdf"], key="ui_chart_format", help="Choose the static image/vector format for chart files in the ZIP archive.")
        
    if st.button("🗜️ Generate ZIP Archive", type="primary", width='stretch', key="ui_generate_zip_archive", help="Package processed data, settings, charts and the interactive map into a ZIP archive."):
        with st.spinner("Starting export... Please wait."):
            
            prog_bar = st.progress(0, text="Preparing export...")
            try:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    
                    prog_bar.progress(5, text="📂 Adding raw input file to the archive...")
                    raw_settings = st.session_state.get('file_settings', {})
                    raw_ext = raw_settings.get('ext', 'csv')
                    
                    if 'raw_file_path' in st.session_state and os.path.exists(st.session_state.raw_file_path):
                        with open(st.session_state.raw_file_path, 'rb') as rf:
                            zip_file.writestr(f"1_Raw_Input_Data.{raw_ext}", rf.read())
                    
                    prog_bar.progress(10, text="⚙️ Writing settings log...")
                    log_data = {
                        "Calculation Mode": "Distance (1m step)" if calc_mode == 'dist' else "Time (Chronological)",
                        "File Settings": st.session_state.get('file_settings', {}),
                        "Math & Filter Settings": st.session_state.get('math_settings', {}),
                        "Topo & Env Settings": st.session_state.get('topo_settings', {}),
                        "Physical Limits": st.session_state.get('limits_settings', {}),
                        "Safety Caps": {"max_trajectory_duration_s": MAX_TRAJECTORY_DURATION_S, "max_output_points": MAX_OUTPUT_POINTS},
                        "Processing Quality Report": st.session_state.get('processing_quality_report', {})
                    }
                    zip_file.writestr("2_Settings_Log.txt", json.dumps(log_data, indent=4))
                    
                    prog_bar.progress(20, text="📊 Generating results table...")
                    if export_fmt == "CSV":
                        csv_data = df_full.to_csv(index=False).encode('utf-8')
                        zip_file.writestr("3_Processed_Calculations.csv", csv_data)
                    else:
                        xlsx_buf = io.BytesIO()
                        max_excel_data_rows = 1_048_575
                        with pd.ExcelWriter(xlsx_buf) as writer:
                            if len(df_full) <= max_excel_data_rows:
                                df_full.to_excel(writer, index=False, sheet_name='Results_1')
                            else:
                                for sheet_i, start_row in enumerate(range(0, len(df_full), max_excel_data_rows), start=1):
                                    df_full.iloc[start_row:start_row + max_excel_data_rows].to_excel(writer, index=False, sheet_name=f'Results_{sheet_i}')
                        zip_file.writestr("3_Processed_Calculations.xlsx", xlsx_buf.getvalue())
                        del xlsx_buf
                        
                    df_plot_export = downsample_preserve_extrema(df_full, max_pts, x_col, important_plot_cols)
                    plots_def = [
                        ('velocity_kmh', '1_Speed', 'Speed (km/h)'),
                        ('accel_long_ms2', '2_Acceleration', 'Acceleration (m/s²)'),
                        ('dist_m' if calc_mode=='time' else 'velocity_ms', '3_Distance_or_Speed', 'Distance (m)' if calc_mode=='time' else 'Speed (m/s)'),
                        ('grade_deg', '4_Grade', 'Grade (degrees)'),
                        ('wind_head_ms', '5_Headwind', 'Speed (m/s)'),
                        ('wind_cross_ms', '6_Crosswind', 'Speed (m/s)'),
                        ('heading_deg', '7_Heading', 'Angle (degrees)'),
                        ('turn_radius_display_m', '8_Radius', 'Radius (m; display cap)'),
                        ('angular_vel_degs', '9_Angular_Velocity', 'Angular velocity (deg/s)'),
                        ('accel_lat_ms2', '10_Lateral_Acceleration', 'Acceleration (m/s²)'),
                        ('elevation_m', '11_Elevation', 'Elevation (m)'),
                        ('air_density_kgm3', '14_Air_Density', 'Density (kg/m³)'),
                        ('air_temp_C', '15_Temperature', 'Temperature (°C)'),
                        ('jerk_ms3', '16_Jerk', 'Jerk (m/s³)'),
                        ('air_press_Pa', '17_Pressure', 'Pressure (Pa)')
                    ]
                    
                    total_plots = len(plots_def) + 1
                    base_prog = 30
                    prog_step = 50 / total_plots
                    
                    for i, (col, name, ylabel) in enumerate(plots_def):
                        prog_bar.progress(int(base_prog + (i * prog_step)), text=f"📈 Generating static chart: {name} ({img_dpi} DPI)...")
                        img_bytes = generate_static_plot(df_plot_export, x_col, col, name.replace('_', ' '), ylabel, dpi=img_dpi, fmt=img_fmt)
                        zip_file.writestr(f"4_Time_Series_Charts/{name}.{img_fmt}", img_bytes)
                        
                    prog_bar.progress(85, text="📉 Generating comparison chart...")
                    img_bytes = generate_static_plot(df_plot_export, x_col, [('velocity_raw_kmh', 'Raw V'), ('velocity_kmh', 'Filtered V')], "12 Raw vs. Filtered Speed", "Speed", dpi=img_dpi, fmt=img_fmt)
                    zip_file.writestr(f"4_Time_Series_Charts/12_Raw_vs_Filtered.{img_fmt}", img_bytes)
                    
                    prog_bar.progress(90, text="🖼️ Generating trajectory image with a transparent background...")
                    transparent_img_bytes = generate_transparent_trajectory(df_plot_export, 'east_m', 'north_m', 'velocity_kmh', dpi=img_dpi)
                    zip_file.writestr("5_Transparent_Trajectory_Map.png", transparent_img_bytes)
                    
                    prog_bar.progress(95, text="🌍 Exporting map...")
                    zip_file.writestr("6_Geospatial_Map.html", fig_map.to_html(full_html=True, include_plotlyjs=True))

                prog_bar.progress(100, text="✅ All files have been archived and are ready to download.")
                st.session_state.zip_ready = zip_buffer.getvalue()
                st.success("Analysis complete! Download the archive using the button below.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                gc.collect()

    if 'zip_ready' in st.session_state:
        # Fayl nomiga maxsus kengaytma (zip) qo'shilganiga ishonch hosil qilish
        safe_filename = re.sub(r'[^A-Za-z0-9._-]+', '_', zip_filename.strip()).strip('._-') or "QwatLab_EV_EGAD"
        if not safe_filename.lower().endswith(".zip"):
            safe_filename += ".zip"
            
        st.download_button(
            label=f"⬇️ Download {safe_filename}",
            data=st.session_state.zip_ready,
            file_name=safe_filename,
            mime="application/zip",
            key="ui_download_archive",
            help="Download the generated archive containing results, charts, settings, and the map.",
            type="primary",
            width='stretch'
        )

# =====================================================================
# 3-BLOK OXIRI
# =====================================================================
# =====================================================================
# ASOSIY ISHGA TUSHIRISH (MAIN)
# =====================================================================
if __name__ == "__main__":
    st.title("🔬 QwatLab: EV E.G.A.D. V1.0.2")

    render_block_0()
    render_block_1()
    render_block_2()
    render_block_3()
