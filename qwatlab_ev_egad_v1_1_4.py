# =====================================================================
# QwatLab: EV E.G.A.D. V1.1.4 — Experimental GNSS/GPS Acquisition,
# 9D Kalman-ZUPT & 3-Mode Hypotenuse Topography Engine
# =====================================================================
# Version: 1.1.4 (Audited Production & Open-Science Release — True Anisotropic 9D KF)
# Program Registration Certificate: No. DGU 69148 (Made in Uzbekistan)
#
# End-to-End 4-Program Pipeline:
#   Program 1 (EGAD V1.1.4) -> Program 2 (DCMG V1.1.4) -> Program 3 (MADE V1.1.1) -> Program 4 (Plotter V1.1.1)
#
# Key Engineering Capabilities:
# - 3-Mode Road Gradient & Hypotenuse Chord Reconstruction:
#     1) Mode 1: Spatial Segment Arithmetic Mean & Hypotenuse Chord (Recommended);
#     2) Mode 2: User-Prescribed Gradient (Constant or Segment Sequence in deg / %);
#     3) Mode 3: Stabilized Spatial Lookback Derivative;
# - Single & Batch Multi-File Processing (1 to 100 GPX/KML Trajectory Files);
# - 100% Closed Kinematic Chain (s <-> v <-> a <-> j) with Horizontal ENU ZUPT Lock;
# - Direct 8-Column (MADE V1.1.1) and 55/29-Column (DCMG V1.1.4) Standardized Exports;
# - Two-Tier Memory Management (Reset Results Only & Deep Clear All).
# =====================================================================

# 1. UI LAYER (Foydalanuvchi interfeysi va loglarni tozalash)
import logging
import streamlit as st

for _st_log_name in (
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.scriptrunner.script_runner",
    "streamlit.runtime.state.session_state_proxy",
):
    logging.getLogger(_st_log_name).setLevel(logging.ERROR)

# 2. CORE MATH & KINEMATIC ENGINE (Hisoblash va filtrlar)
import os as _os_init
import tempfile as _tmp_init
try:
    _nb_cache_dir = _os_init.path.join(_tmp_init.gettempdir(), "QwatLab_EGAD_Numba_Cache")
    _os_init.makedirs(_nb_cache_dir, exist_ok=True)
    _os_init.environ.setdefault("NUMBA_CACHE_DIR", _nb_cache_dir)
except Exception:
    pass

def _st_width_kw():
    """Cross-version Streamlit full-width helper (compatible with 1.30 through 1.50+ and PyInstaller EXE)."""
    try:
        _parts = [int(p) for p in str(getattr(st, "__version__", "1.40")).split(".")[:2]]
        return {"width": "stretch"} if tuple(_parts) >= (1, 45) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

try:
    import numba as _nb

    class _SafeNumba:
        @staticmethod
        def njit(*args, **kwargs):
            def _decorator(func):
                try:
                    return _nb.njit(func)
                except Exception:
                    return func
            if args and callable(args[0]):
                return _decorator(args[0])
            return _decorator

    nb = _SafeNumba()
except Exception:
    class _NumbaFallback:
        @staticmethod
        def njit(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

    nb = _NumbaFallback()

# 3. DATA ENGINE & PARSING (Ma'lumotlarni o'qish va tahlil)
import pandas as pd
import xml.etree.ElementTree as ET
import atexit

# 4. VISUALIZATION LAYER (Grafika va xaritalar)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
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
# SAHIFA KENG EKRAN (DESKTOP) REJIMI VA KONSTANTALAR
# =====================================================================
st.set_page_config(page_title="QwatLab: EV E.G.A.D. V1.1.4", layout="wide")

MAX_TRAJECTORY_DURATION_S = 100000.0
MAX_OUTPUT_POINTS = 2_000_000
MAX_BATCH_GPX_FILES = 100
STANDARD_GRAVITY_MS2 = 9.80665

# 3-dastur (QwatLab EV MADE V1.1.1) va 2-dastur (DCMG V1.1.4) qabul qiladigan Yagona 8-Ustunli Standart Shartnoma
MADE_V111_8_COLUMNS = [
    "Time [s]",
    "Interval [s]",
    "GPS_Speed [km/h]",
    "OBD_Speed [km/h]",
    "Wind_Speed [m/s]",
    "Distance [km]",
    "Acceleration [m/s2]",
    "Road_Gradient [deg]",
]

# 2-dastur (QwatLab EV DCMG V1.1.4) bilan 100% bir xil 55 ustunli rang ierarxiyasi (16 Qizil + 13 Sariq + 26 Ko'k)
RED_MARKOV_CORE_COLUMNS = [
    "trip_id",
    "microtrip_id",
    "trip_microtrip_id",
    "recorded_datetime_local",
    "recorded_time_hms",
    "time_sec",
    "trip_time_sec",
    "dt_sec",
    "dist_m",
    "trip_dist_m",
    "velocity_ms",
    "velocity_kmh",
    "accel_long_ms2",
    "grade_deg",
    "grade_percent",
    "is_stopped",
]

YELLOW_ARTICLE_COLUMNS = [
    "driving_mode",
    "recorded_hour_float",
    "velocity_raw_kmh",
    "elevation_m",
    "jerk_ms3",
    "accel_lat_ms2",
    "angular_vel_degs",
    "turn_radius_m",
    "heading_deg",
    "vsp_w_per_kg",
    "zupt_update_applied",
    "lat_smooth",
    "lon_smooth",
]

BLUE_DIAGNOSTIC_COLUMNS = [
    "recorded_datetime_utc",
    "x_m",
    "y_m",
    "z_m",
    "east_m",
    "north_m",
    "up_m",
    "vx_ecef_ms",
    "vy_ecef_ms",
    "vz_ecef_ms",
    "ax_ecef_ms2",
    "ay_ecef_ms2",
    "az_ecef_ms2",
    "v_east_ms",
    "v_north_ms",
    "v_up_ms",
    "accel_east_ms2",
    "accel_north_ms2",
    "accel_up_ms2",
    "accel_vector_ms2",
    "heading_rad",
    "angular_vel_rad_s",
    "turn_radius_display_m",
    "wind_head_ms",
    "wind_cross_ms",
    "air_density_kgm3",
]

MARKOV_29_COLUMNS = RED_MARKOV_CORE_COLUMNS + YELLOW_ARTICLE_COLUMNS
ALL_55_ORDERED_COLUMNS = RED_MARKOV_CORE_COLUMNS + YELLOW_ARTICLE_COLUMNS + BLUE_DIAGNOSTIC_COLUMNS
ALL_54_ORDERED_COLUMNS = ALL_55_ORDERED_COLUMNS  # Backward-compatibility alias


def central_diff_1d(y: np.ndarray, dt: float) -> np.ndarray:
    """2-tartibli markaziy ayirma hosilasi: (y_{k+1} - y_{k-1}) / (2*dt), chekkalarda 2-tartibli."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    dt_f = max(float(dt), 1e-6)
    if n == 2:
        out[:] = (y[1] - y[0]) / dt_f
        return out
    out[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt_f)
    out[0] = (-1.5 * y[0] + 2.0 * y[1] - 0.5 * y[2]) / dt_f
    out[-1] = (0.5 * y[-3] - 2.0 * y[-2] + 1.5 * y[-1]) / dt_f
    return out


# =====================================================================
# 0-BLOK: KINEMATIK MODEL, GIPOTENUZA QIYALIK VA MATEMATIKA
# =====================================================================
def render_block_0():
    """
    0-Blok: QwatLab: EV EGAD V1.1.1. Matematika, 3-variantli Gipotenuza yo'l qiyaligi
    va 9D Kalman-ZUPT tenglamalarini o'zida jamlagan UI komponent.
    """
    with st.expander("📘 BLOCK 0: QwatLab Kinematic Parameters, 3-Mode Hypotenuse Road Grade & Math Engine V1.1.4", expanded=True):
        tab_phil, tab_proj, tab_kin, tab_topo, tab_filt, tab_contract, tab_post, tab_donate = st.tabs([
            "🌐 1. Platform Overview",
            "📐 2. Geospatial Projection",
            "🏎️ 3. Closed-Chain Kinematics",
            "⛰️ 4. 3-Mode Hypotenuse Topography",
            "🔍 5. 9D Kalman & Horizontal ZUPT",
            "🔗 6. DCMG & MADE V1.1.1 Contracts",
            "📬 Contact & Support",
            "☕ Support the Project"
        ])

        with tab_phil:
            st.markdown("### 🔬 Universal Kinematic & Topographic Trajectory Engine (V1.1.4)")
            st.markdown(
                "This platform (**Program 1: QwatLab EV E.G.A.D. V1.1.4**) processes raw GNSS/GPS telemetry (`.gpx` and `.kml`, "
                "from 1 up to 100 trip files in batch mode) into physically consistent 3D trajectories, road gradients, and longitudinal kinematics. "
                "Its outputs feed directly into **Program 2 (`QwatLab EV DCMG V1.1.4`)** and **Program 3 (`QwatLab EV MADE V1.1.1`)**."
            )
            st.info(
                "**Key Engineering Architecture in V1.1.2:**\n"
                "1. **100% Closed Kinematic Chain ($s \\leftrightarrow v \\leftrightarrow a \\leftrightarrow j$):** Velocity is filtered and rate-bounded first, with exact causal step acceleration $a_k = (v_k - v_{k-1})/\\Delta t$ and trapezoidal distance integration $s_k = s_{k-1} + \\frac{1}{2}(v_{k-1}+v_k)\\Delta t$ ($\\Delta s_k \\equiv 0$ during ZUPT stops).\n"
                "2. **3-Mode Road Gradient Engine:** Solves GPS vertical dilution of precision (VDOP) noise and zero-speed division explosions via the **Spatial Segment Mean & Hypotenuse Chord Method (Mode 1)**, **User-Prescribed Gradient (Mode 2)**, or **Stabilized Spatial Lookback (Mode 3)**.\n"
                "3. **Single & Multi-File Batch Processing (1–100 Files):** Process 1 trip or up to 100 GPX/KML trips simultaneously into a unified multi-trip master table plus individual per-trip datasets.\n"
                "4. **Direct 8-Column (`MADE V1.1.1`) and 55/29-Column (`DCMG V1.1.4`) Exports:** Zero manual column reformatting required."
            )

        with tab_proj:
            st.markdown("#### WGS84-Based ECEF and ENU Transformation")
            st.markdown(
                "To avoid flat-Earth approximations, geographic coordinates are converted using the WGS84 ellipsoid. "
                "ECEF is used for global Cartesian calculations and ENU for local trajectory visualization."
            )
            st.latex(r"a = 6378137.0 \text{ (semi-major axis)}, \quad e^2 = 0.00669437999 \text{ (eccentricity squared)}")
            st.latex(r"N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2 \phi}}, \qquad X = (N(\phi) + h) \cos \phi \cos \lambda, \qquad Y = (N(\phi) + h) \cos \phi \sin \lambda, \qquad Z = \left(N(\phi)(1 - e^2) + h\right) \sin \phi")

        with tab_kin:
            st.markdown("#### Closed-Chain Longitudinal Kinematics, Trapezoidal Distance and Curvature")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**1. Exact Uniform Grid ($\\Delta t = 1/f_s$) & Closed Kinematic Chain:**")
                st.latex(r"t_k = k\,\Delta t, \qquad a_{\text{long},k} = \frac{v_k - v_{k-1}}{\Delta t}, \qquad s_k = s_{k-1} + \frac{1}{2}(v_{k-1} + v_k)\,\Delta t")
                st.caption("Guarantees $v_k - v_{k-1} \\equiv a_k\\,\\Delta t$ and $\\Delta s_k \\equiv 0$ whenever $v_{k-1}=v_k=0$, with boundary conditions $v_0 = 0,\\; v_N = 0$.")
            with col2:
                st.markdown(r"**2. Trajectory Heading ($\psi$) & Curvature ($\kappa$):**")
                st.latex(r"\psi = \operatorname{atan2}(v_{\text{East}}, v_{\text{North}}), \qquad \kappa = \frac{|\dot{\psi}|}{\max(v, v_{\min})}")
                st.caption("During standstill intervals ($v = 0$), vehicle heading $\\psi$ is held constant (`ffill`) at its pre-stop orientation.")

        with tab_topo:
            st.markdown("#### ⛰️ 3-Mode Road Gradient & Hypotenuse Elevation Reconstruction")
            c_t1, c_t2, c_t3 = st.columns(3)
            with c_t1:
                st.success(
                    "**Mode 1: Spatial Segment Averaging & Hypotenuse Chord (Default)**\n"
                    "Divides total distance into segments of length $\\Delta L_{\\text{seg}}$ (e.g. $100\\text{ m}$), computes the outlier-trimmed arithmetic mean elevation "
                    "$\\bar{z}_m = \\frac{1}{N_m}\\sum_{i \\in \\text{Seg}_m} z_i$ across moving points in each segment, and connects consecutive segment means with a $C^1$ PCHIP hypotenuse chord:"
                )
                st.latex(r"\tan\theta_m = \frac{\bar{z}_{m+1} - \bar{z}_m}{\Delta L_{\text{seg}}}, \quad L_{\text{hyp},m} = \sqrt{\Delta L_{\text{seg}}^2 + (\Delta\bar{z}_m)^2}")
            with c_t2:
                st.info(
                    "**Mode 2: User-Prescribed Gradient (Constant or Segment Table)**\n"
                    "Bypasses noisy GPX $z$-coordinates and applies user-specified slope values (in degrees or %) across each $\\Delta L_{\\text{seg}}$ interval, reconstructing elevation via exact spatial integration:"
                )
                st.latex(r"z_{\text{hyp}}(s) = z_0 + \int_0^s \tan\theta(s')\,ds'")
            with c_t3:
                st.warning(
                    "**Mode 3: Stabilized Legacy Spatial Lookback**\n"
                    "Evaluates elevation change over a true spatial lookback window $[s - \\Delta L_{\\text{seg}},\\, s]$ rather than a fixed time-index shift, preventing zero-division at traffic stops while retaining true stop inclination $\\theta(s_{\\text{stop}})$."
                )

        with tab_filt:
            st.markdown("#### 🔍 Anisotropic 9D Kalman Filter and Horizontal Zero-Velocity Update (ZUPT)")
            st.latex(r"X_k = [x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]^T, \qquad \Phi = \begin{bmatrix} I_3 & \Delta t I_3 & \frac{\Delta t^2}{2} I_3 \\ 0_3 & I_3 & \Delta t I_3 \\ 0_3 & 0_3 & I_3 \end{bmatrix}")
            st.caption("ZUPT is triggered strictly by horizontal ground displacement $\\|\\Delta\\mathbf{r}_{\\text{horiz}}\\|/\\Delta t \\le v_{\\text{ZUPT}}$, making standstill detection 100% immune to vertical GPS altitude jitter.")

        with tab_contract:
            st.markdown("#### 🔗 Direct Handshake with Program 2 (`DCMG V1.1.4`) & Program 3 (`MADE V1.1.1`)")
            st.markdown(
                "- **`1_MADE_8Col_Ready (.csv / .xlsx)`:** `Time [s]`, `Interval [s]`, `GPS_Speed [km/h]`, `OBD_Speed [km/h]`, `Wind_Speed [m/s]`, `Distance [km]`, `Acceleration [m/s2]`, `Road_Gradient [deg]`.\n"
                "- **`2_DCMG_55Col_ColorCoded (.csv / .xlsx)`:** Full 55-column color-coded table (`16 Red + 13 Yellow + 26 Blue`) with automatic `trip_id` and `microtrip_id` numbering.\n"
                "- **`3_DCMG_29Col_Markov_Core (.csv / .xlsx)`:** Compact 29-column table (`16 Red + 13 Yellow`).\n"
                "- **`4_Road_Grade_Hypotenuse_Segments (.csv / .xlsx)`:** Segment-by-segment arithmetic mean elevation and hypotenuse chord audit table."
            )

        with tab_post:
            st.markdown("### QwatLab Team")
            st.markdown("* 🌐 Website: https://t.me/QwatLab")
            st.markdown("* **Made in Uzbekistan 🇺🇿** | **Program registration certificate: No. DGU 69148**")

        with tab_donate:
            st.markdown("### Support the Project")
            st.markdown("* ☕ Donate: https://t.me/QwatLab")


# =====================================================================
# BACKEND: XAVFSIZLIK, TOZALASH VA DEMO GPX GENERATORI
# =====================================================================
def cleanup_temp_files():
    """Joriy sessiyaga tegishli vaqtinchalik fayllarni xavfsiz tozalash."""
    try:
        paths = st.session_state.get("raw_file_paths", [])
        for p_info in paths:
            p = p_info.get("path") if isinstance(p_info, dict) else p_info
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    except Exception:
        pass
    try:
        st.session_state.pop("raw_file_paths", None)
    except Exception:
        pass


atexit.register(cleanup_temp_files)


def generate_synthetic_tashkent_gpx_bytes(duration_s: int = 900, trip_idx: int = 1) -> bytes:
    """Foydalanuvchida GPX fayl bo'lmaganda sinov uchun haqiqiy Toshkent GPX traektoriyasini yaratadi."""
    rng = np.random.default_rng(100 + trip_idx)
    t_arr = np.arange(0, duration_s + 1, 1, dtype=float)
    tau = t_arr / max(float(duration_s), 1.0)
    v_ms = np.clip(11.5 * (np.sin(5.0 * np.pi * tau) ** 2) + rng.normal(0.0, 0.3, len(t_arr)), 0.0, 18.0)
    v_ms[v_ms < 0.6] = 0.0
    v_ms[0] = 0.0
    v_ms[-1] = 0.0
    s_m = np.cumsum(v_ms)
    heading = 0.35 * np.sin(2.0 * np.pi * tau) + 0.15 * trip_idx
    deast = v_ms * np.sin(heading)
    dnorth = v_ms * np.cos(heading)
    lat = 41.2995 + np.cumsum(dnorth) / 111132.0
    lon = 69.2401 + np.cumsum(deast) / (111320.0 * np.cos(np.radians(41.3)))
    ele_true = 445.0 + 14.0 * np.sin(2.0 * np.pi * s_m / max(s_m[-1], 500.0)) + 0.008 * s_m
    ele_noisy = ele_true + rng.normal(0.0, 1.4, len(t_arr))
    t0 = pd.Timestamp("2026-05-12 03:30:00Z") + pd.to_timedelta(trip_idx * 45, unit="m")
    times_iso = (t0 + pd.to_timedelta(t_arr, unit="s")).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="QwatLab EV EGAD V1.1.4">',
        f'  <trk><name>Tashkent_Demo_Trip_{trip_idx:02d}</name><trkseg>'
    ]
    for la, lo, el, ts in zip(lat, lon, ele_noisy, times_iso):
        lines.append(f'    <trkpt lat="{la:.7f}" lon="{lo:.7f}"><ele>{el:.2f}</ele><time>{ts}</time></trkpt>')
    lines.append('  </trkseg></trk></gpx>')
    return "\n".join(lines).encode("utf-8")


# =====================================================================
# 3-VARIANTLI YO'L QIYALIGI VA GIPOTENUZA BALANDLIK ALGORITMI (V1.1.1 CORE)
# =====================================================================
def compute_road_grade_and_hypotenuse_profile(
    dist_m: np.ndarray,
    ele_raw_m: np.ndarray,
    velocity_ms: np.ndarray,
    zupt_lim_kmh: float,
    topo_settings: dict
):
    """
    Yo'l qiyaligi va balandlik profilini hisoblashning 3-variantli muhandislik yadrosi:
    - Mode 1 (Spatial Segment Averaging & Hypotenuse Chord):
      Har bir Delta_L metr kesmadagi barcha harakat nuqtalarining o'rta arifmetigini (outlier-trimmed mean) oladi,
      va ikki kesma o'rtacha balandligi orasidan gipotenuza chizib, silliq qiyalik chiqaradi.
    - Mode 2 (User-Prescribed Constant or Segment-Table Gradient):
      GPX dagi z o'qini ishlatmasdan, foydalanuvchi kiritgan gradus yoki % qiyalik bo'yicha
      balandlik va qiyalikni gipotenuza integrali orqali quradi.
    - Mode 3 (Stabilized Legacy Spatial Derivative):
      Vaqt siljishi emas, haqiqiy masofa o'qi bo'yicha [s - Delta_L, s] oraliqda qiyalik oladi.
    Mashina to'xtab turgan nuqtalarda (v == 0, dist_m == const) yo'l qiyaligi sun'iy 0° ga tushmaydi,
    balki o'sha to'xtagan joyning haqiqiy qiyaligini (theta(s_stop)) tabiiy saqlaydi!
    """
    dist_m = np.asarray(dist_m, dtype=np.float64)
    ele_raw_m = np.asarray(ele_raw_m, dtype=np.float64)
    velocity_ms = np.asarray(velocity_ms, dtype=np.float64)
    n = len(dist_m)

    if n == 0:
        return ele_raw_m.copy(), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), pd.DataFrame()

    if not topo_settings.get("enabled", True):
        empty_seg = pd.DataFrame(columns=[
            "Segment_ID", "Start_Dist_m", "End_Dist_m", "Span_m", "Sample_Count",
            "Raw_Mean_Elevation_m", "Next_Mean_Elevation_m", "Delta_Z_m",
            "Hypotenuse_Length_m", "Grade_Percent", "Grade_Deg"
        ])
        return ele_raw_m.copy(), np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool), empty_seg

    grade_mode = str(topo_settings.get("grade_mode", "Mode 1: Spatial Segment Mean & Hypotenuse Chord (Recommended)"))
    seg_len_m = max(5.0, float(topo_settings.get("spatial_step", 100.0)))
    max_grad_deg = max(1.0, float(topo_settings.get("max_topo_grad", 25.0)))
    max_grad_pct = np.tan(np.radians(max_grad_deg)) * 100.0
    smooth_knots = bool(topo_settings.get("smooth_hypotenuse_knots", True))
    is_moving = (velocity_ms * 3.6) > max(0.2, float(zupt_lim_kmh) * 0.5)

    total_dist_m = float(dist_m[-1]) if n > 0 else 0.0

    # -----------------------------------------------------------------
    # MODE 2: FOYDALANUVCHI KIRITGAN QIYALIK (CONSTANT YOKI KESMALAR JADVALI)
    # -----------------------------------------------------------------
    if grade_mode.startswith("Mode 2"):
        user_unit = str(topo_settings.get("user_grade_unit", "Degrees (deg)"))
        user_submode = str(topo_settings.get("user_grade_submode", "Constant Gradient Along Entire Route"))
        base_z0 = float(ele_raw_m[0]) if np.isfinite(ele_raw_m[0]) else float(topo_settings.get("base_alt", 450.0))

        n_segs = max(1, int(np.ceil(max(total_dist_m, seg_len_m) / seg_len_m)))
        if "Constant" in user_submode:
            val_const = float(topo_settings.get("user_const_grade_val", 0.0))
            seg_vals = np.full(n_segs, val_const, dtype=np.float64)
        else:
            raw_seq_str = str(topo_settings.get("user_segment_grades_str", "0.0, 0.5, 1.0, 0.0, -0.5"))
            parsed_tokens = []
            for tok in re.split(r"[,;\s]+", raw_seq_str.strip()):
                if tok:
                    try:
                        parsed_tokens.append(float(tok))
                    except ValueError:
                        pass
            if not parsed_tokens:
                parsed_tokens = [0.0]
            seg_vals = np.resize(np.array(parsed_tokens, dtype=np.float64), n_segs)

        if "Percent" in user_unit or "%" in user_unit:
            seg_grade_pct = seg_vals.copy()
            seg_grade_deg = np.degrees(np.arctan(seg_grade_pct / 100.0))
        else:
            seg_grade_deg = np.clip(seg_vals.copy(), -80.0, 80.0)
            seg_grade_pct = np.tan(np.radians(seg_grade_deg)) * 100.0

        edges_s = np.arange(n_segs + 1, dtype=np.float64) * seg_len_m
        edges_s[-1] = max(edges_s[-1], total_dist_m)
        knots_z = np.zeros(n_segs + 1, dtype=np.float64)
        knots_z[0] = base_z0
        seg_records = []
        for m in range(n_segs):
            s0, s1 = edges_s[m], edges_s[m + 1]
            span = max(1e-6, s1 - s0)
            dz = span * (seg_grade_pct[m] / 100.0)
            knots_z[m + 1] = knots_z[m] + dz
            hyp_len = float(np.sqrt(span ** 2 + dz ** 2))
            cnt = int(np.sum((dist_m >= s0) & (dist_m <= s1)))
            seg_records.append({
                "Segment_ID": m + 1,
                "Start_Dist_m": round(s0, 2),
                "End_Dist_m": round(s1, 2),
                "Span_m": round(span, 2),
                "Sample_Count": cnt,
                "Raw_Mean_Elevation_m": round(knots_z[m], 3),
                "Next_Mean_Elevation_m": round(knots_z[m + 1], 3),
                "Delta_Z_m": round(dz, 3),
                "Hypotenuse_Length_m": round(hyp_len, 3),
                "Grade_Percent": round(float(seg_grade_pct[m]), 4),
                "Grade_Deg": round(float(seg_grade_deg[m]), 4),
            })

        if smooth_knots and len(edges_s) >= 4:
            pchip_z = PchipInterpolator(edges_s, knots_z)
            ele_hyp_m = pchip_z(dist_m)
            dz_ds = pchip_z.derivative()(dist_m)
            grade_pct_out = np.clip(dz_ds * 100.0, -max_grad_pct, max_grad_pct)
            grade_deg_out = np.degrees(np.arctan(grade_pct_out / 100.0))
        else:
            ele_hyp_m = np.interp(dist_m, edges_s, knots_z)
            seg_idx = np.clip(np.searchsorted(edges_s[1:], dist_m, side="right"), 0, n_segs - 1)
            grade_pct_out = seg_grade_pct[seg_idx]
            grade_deg_out = seg_grade_deg[seg_idx]

        exceeded = np.abs(grade_deg_out) > max_grad_deg
        return ele_hyp_m, grade_pct_out, grade_deg_out, exceeded, pd.DataFrame(seg_records)

    # -----------------------------------------------------------------
    # MODE 1: MASOFAVIY KESMALAR O'RTA ARIFMETIGI VA GIPOTENUZA USULI (RECOMMENDED)
    # -----------------------------------------------------------------
    if grade_mode.startswith("Mode 1"):
        if total_dist_m <= seg_len_m * 1.1:
            z_start = float(np.mean(ele_raw_m[:max(1, n // 3)]))
            z_end = float(np.mean(ele_raw_m[max(0, n - max(1, n // 3)):]))
            span = max(total_dist_m, 1.0)
            dz = z_end - z_start
            g_pct = float(np.clip((dz / span) * 100.0, -max_grad_pct, max_grad_pct)) if total_dist_m > 2.0 else 0.0
            g_deg = float(np.degrees(np.arctan(g_pct / 100.0)))
            ele_hyp_m = z_start + (dist_m / span) * (span * g_pct / 100.0)
            grade_pct_out = np.full(n, g_pct, dtype=np.float64)
            grade_deg_out = np.full(n, g_deg, dtype=np.float64)
            seg_df = pd.DataFrame([{
                "Segment_ID": 1,
                "Start_Dist_m": 0.0,
                "End_Dist_m": round(total_dist_m, 2),
                "Span_m": round(span, 2),
                "Sample_Count": n,
                "Raw_Mean_Elevation_m": round(z_start, 3),
                "Next_Mean_Elevation_m": round(z_end, 3),
                "Delta_Z_m": round(dz, 3),
                "Hypotenuse_Length_m": round(float(np.sqrt(span ** 2 + dz ** 2)), 3),
                "Grade_Percent": round(g_pct, 4),
                "Grade_Deg": round(g_deg, 4),
            }])
            return ele_hyp_m, grade_pct_out, grade_deg_out, np.abs(grade_deg_out) > max_grad_deg, seg_df

        n_segs = max(2, int(np.ceil(total_dist_m / seg_len_m)))
        edges_s = np.linspace(0.0, total_dist_m, n_segs + 1)

        seg_means_z = np.zeros(n_segs, dtype=np.float64)
        seg_counts = np.zeros(n_segs, dtype=int)
        for m in range(n_segs):
            s_lo, s_hi = edges_s[m], edges_s[m + 1]
            if m == n_segs - 1:
                mask_m = (dist_m >= s_lo) & (dist_m <= s_hi)
            else:
                mask_m = (dist_m >= s_lo) & (dist_m < s_hi)
            cnt_m = int(np.sum(mask_m))
            seg_counts[m] = cnt_m
            if cnt_m > 0:
                moving_mask_m = mask_m & is_moving
                z_vals_m = ele_raw_m[moving_mask_m] if np.any(moving_mask_m) else ele_raw_m[mask_m]
                if len(z_vals_m) >= 5:
                    p10, p90 = np.percentile(z_vals_m, [10.0, 90.0])
                    trimmed = z_vals_m[(z_vals_m >= p10) & (z_vals_m <= p90)]
                    seg_means_z[m] = float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(z_vals_m))
                else:
                    seg_means_z[m] = float(np.mean(z_vals_m))
            else:
                seg_means_z[m] = np.nan

        valid_seg = np.isfinite(seg_means_z)
        if np.any(valid_seg):
            idx_seg = np.arange(n_segs, dtype=float)
            seg_means_z = np.interp(idx_seg, idx_seg[valid_seg], seg_means_z[valid_seg])
        else:
            seg_means_z[:] = float(ele_raw_m[0])

        knot_z = np.zeros(n_segs + 1, dtype=np.float64)
        knot_z[0] = seg_means_z[0]
        knot_z[-1] = seg_means_z[-1]
        for m in range(1, n_segs):
            knot_z[m] = 0.5 * (seg_means_z[m - 1] + seg_means_z[m])

        for m in range(n_segs):
            span_m = max(1e-6, edges_s[m + 1] - edges_s[m])
            max_dz = span_m * (max_grad_pct / 100.0)
            dz_now = knot_z[m + 1] - knot_z[m]
            if abs(dz_now) > max_dz:
                knot_z[m + 1] = knot_z[m] + np.sign(dz_now) * max_dz

        seg_records = []
        seg_grade_pct = np.zeros(n_segs, dtype=np.float64)
        seg_grade_deg = np.zeros(n_segs, dtype=np.float64)
        for m in range(n_segs):
            s0, s1 = edges_s[m], edges_s[m + 1]
            span = max(1e-6, s1 - s0)
            z0_k, z1_k = knot_z[m], knot_z[m + 1]
            dz = z1_k - z0_k
            hyp_len = float(np.sqrt(span ** 2 + dz ** 2))
            g_pct = (dz / span) * 100.0
            g_deg = float(np.degrees(np.arctan(g_pct / 100.0)))
            seg_grade_pct[m] = g_pct
            seg_grade_deg[m] = g_deg
            seg_records.append({
                "Segment_ID": m + 1,
                "Start_Dist_m": round(s0, 2),
                "End_Dist_m": round(s1, 2),
                "Span_m": round(span, 2),
                "Sample_Count": int(seg_counts[m]),
                "Raw_Mean_Elevation_m": round(float(seg_means_z[m]), 3),
                "Next_Mean_Elevation_m": round(float(seg_means_z[min(m + 1, n_segs - 1)]), 3),
                "Hypotenuse_Start_Z_m": round(z0_k, 3),
                "Hypotenuse_End_Z_m": round(z1_k, 3),
                "Delta_Z_m": round(dz, 3),
                "Hypotenuse_Length_m": round(hyp_len, 3),
                "Grade_Percent": round(g_pct, 4),
                "Grade_Deg": round(g_deg, 4),
            })

        if smooth_knots and len(edges_s) >= 4:
            pchip_z = PchipInterpolator(edges_s, knot_z)
            ele_hyp_m = pchip_z(dist_m)
            dz_ds = pchip_z.derivative()(dist_m)
            grade_pct_out = np.clip(dz_ds * 100.0, -max_grad_pct, max_grad_pct)
            grade_deg_out = np.degrees(np.arctan(grade_pct_out / 100.0))
        else:
            ele_hyp_m = np.interp(dist_m, edges_s, knot_z)
            seg_idx = np.clip(np.searchsorted(edges_s[1:], dist_m, side="right"), 0, n_segs - 1)
            grade_pct_out = seg_grade_pct[seg_idx]
            grade_deg_out = seg_grade_deg[seg_idx]

        exceeded = np.abs(grade_deg_out) > max_grad_deg
        return ele_hyp_m, grade_pct_out, grade_deg_out, exceeded, pd.DataFrame(seg_records)

    # -----------------------------------------------------------------
    # MODE 3: HIMOYALANGAN KLASSIK MASOFAVIY HOSILA (STABILIZED LEGACY)
    # -----------------------------------------------------------------
    half_w = 0.5 * seg_len_m
    z_ahead = np.interp(np.minimum(dist_m + half_w, total_dist_m), dist_m, ele_raw_m)
    z_behind = np.interp(np.maximum(dist_m - half_w, 0.0), dist_m, ele_raw_m)
    ds_window = np.minimum(dist_m + half_w, total_dist_m) - np.maximum(dist_m - half_w, 0.0)
    grade_pct_out = np.where(ds_window >= 2.0, ((z_ahead - z_behind) / np.maximum(ds_window, 2.0)) * 100.0, 0.0)
    exceeded = np.abs(np.degrees(np.arctan(grade_pct_out / 100.0))) > max_grad_deg
    grade_pct_out = np.clip(grade_pct_out, -max_grad_pct, max_grad_pct)
    grade_deg_out = np.degrees(np.arctan(grade_pct_out / 100.0))
    seg_df = pd.DataFrame([{
        "Segment_ID": 1,
        "Start_Dist_m": 0.0,
        "End_Dist_m": round(total_dist_m, 2),
        "Span_m": round(seg_len_m, 2),
        "Sample_Count": n,
        "Raw_Mean_Elevation_m": round(float(np.mean(ele_raw_m)), 3),
        "Next_Mean_Elevation_m": round(float(np.mean(ele_raw_m)), 3),
        "Delta_Z_m": round(float(ele_raw_m[-1] - ele_raw_m[0]), 3),
        "Hypotenuse_Length_m": round(total_dist_m, 3),
        "Grade_Percent": round(float(np.mean(grade_pct_out)), 4),
        "Grade_Deg": round(float(np.mean(grade_deg_out)), 4),
    }])
    return ele_raw_m.copy(), grade_pct_out, grade_deg_out, exceeded, seg_df


# =====================================================================
# 1-BLOK: KINEMATIK LABORATORIYA SOZLAMALARI (100% "?" TOOLTIP COVERAGE)
# =====================================================================
def render_block_1():
    if "locked" not in st.session_state:
        st.session_state.locked = False

    with st.expander("🎛️ BLOCK 1: GPX/KML Upload (1–100 Files), 3-Mode Hypotenuse Road Grade & Kinematic Parameters", expanded=not st.session_state.locked):
        st.caption("ℹ️ Hover over the **`?`** icon next to any parameter or button below to view recommended engineering bounds and mathematical explanations.")
        tab_file, tab_math, tab_topo, tab_limits = st.tabs([
            "📂 1. File Upload (1–100 GPX/KML) & Quality Audit",
            "🧮 2. 9D Kalman & SG Filter Engine",
            "⛰️ 3. 3-Mode Hypotenuse Road Grade & Environment",
            "🚧 4. Physical, EV VSP & UI Limits"
        ])

        file_settings = {"default_dt": 1.0}
        math_settings = {}
        topo_settings = {}
        limits_settings = {}

        # --- 1-TAB: FAYL YUKLASH VA SIFAT AUDITI ---
        with tab_file:
            st.markdown("### 📂 Upload GPX/KML Trajectory Files (Single Trip or Batch of Up to 100 Trips)")
            c_up1, c_up2 = st.columns([2.2, 1.0])
            with c_up1:
                uploaded_files = st.file_uploader(
                    "Upload 1 to 100 GPX or KML trajectory files (each file is assigned `trip_id = 1..N`):",
                    type=["gpx", "kml"],
                    accept_multiple_files=True,
                    key="b1_file_upload",
                    disabled=st.session_state.locked,
                    help="Upload one GPX/KML file for single-trip analysis, or up to 100 GPX/KML files to batch-process an entire fleet for Program 2 (DCMG) and Program 3 (MADE)."
                )
            with c_up2:
                use_demo_gpx = st.toggle(
                    "🧪 Use Built-In Tashkent Demo GPX Track(s) (if no file uploaded)",
                    value=False,
                    disabled=st.session_state.locked,
                    key="b1_use_demo_gpx",
                    help="Generates realistic synthetic Tashkent urban GPX tracks with elevation and traffic stops for testing when no external GPX file is uploaded."
                )
                demo_gpx_count = st.slider(
                    "Number of Demo GPX Trips:",
                    min_value=1,
                    max_value=20,
                    value=3,
                    disabled=(st.session_state.locked or not use_demo_gpx),
                    key="b1_demo_gpx_count",
                    help="Select how many synthetic urban GPX trajectories (1 to 20 trips) to generate in demo mode."
                )

            if uploaded_files and len(uploaded_files) > MAX_BATCH_GPX_FILES:
                st.warning(f"⚠️ Maximum allowed batch size is {MAX_BATCH_GPX_FILES} files. Using the first {MAX_BATCH_GPX_FILES} files.")
                uploaded_files = uploaded_files[:MAX_BATCH_GPX_FILES]

            st.markdown("#### 🗺️ XML Tag Mapping")
            c_map1, c_map2, c_map3, c_map4, c_map5 = st.columns(5)
            with c_map1:
                file_settings["tag_point"] = st.text_input(
                    "GPX Point tag:", value="trkpt", disabled=st.session_state.locked, key="b1_point_tag",
                    help="XML element name representing each track point in GPX files (standard GPX 1.1 default: 'trkpt')."
                )
            with c_map2:
                file_settings["tag_lat"] = st.text_input(
                    "Latitude attr:", value="lat", disabled=st.session_state.locked, key="b1_latitude_attribute",
                    help="XML attribute name inside `<trkpt>` storing WGS84 latitude in decimal degrees (standard default: 'lat')."
                )
            with c_map3:
                file_settings["tag_lon"] = st.text_input(
                    "Longitude attr:", value="lon", disabled=st.session_state.locked, key="b1_longitude_attribute",
                    help="XML attribute name inside `<trkpt>` storing WGS84 longitude in decimal degrees (standard default: 'lon')."
                )
            with c_map4:
                file_settings["tag_ele"] = st.text_input(
                    "Elevation tag:", value="ele", disabled=st.session_state.locked, key="b1_elevation_tag",
                    help="Child XML tag storing altitude/elevation in meters above sea level (standard GPX default: 'ele')."
                )
            with c_map5:
                file_settings["tag_time"] = st.text_input(
                    "Time tag (GPX/KML):", value="time", disabled=st.session_state.locked, key="b1_gpx_time_tag",
                    help="Child XML tag storing ISO-8601 UTC timestamp for each track point (standard GPX default: 'time', KML: 'when')."
                )
            file_settings["tag_coords"] = "coordinates"

            if uploaded_files:
                first_uf = uploaded_files[0]
                ext = first_uf.name.split(".")[-1].lower()
                file_settings["ext"] = ext
                st.success(f"✅ Loaded {len(uploaded_files)} file(s). Primary file: `{first_uf.name}` ({ext.upper()} format)")
                try:
                    file_bytes = first_uf.getvalue()
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

                    if not preview_df.empty:
                        raw = preview_df.copy()
                        for c in ("lat", "lon", "ele"):
                            raw[c] = pd.to_numeric(raw[c], errors="coerce")
                        coord_valid = (
                            np.isfinite(raw["lat"]) & np.isfinite(raw["lon"]) &
                            raw["lat"].between(-90, 90) & raw["lon"].between(-180, 180)
                        )
                        missing_ele = ~np.isfinite(raw["ele"])
                        time_values = pd.to_datetime(raw["time"], format="mixed", errors="coerce", utc=True)
                        duplicate_mask = raw.duplicated(subset=["lat", "lon", "time"], keep="first")
                        duplicate_time_mask = time_values.notna() & time_values.duplicated(keep="first")

                        q_summary = {
                            "Files queued": int(len(uploaded_files)),
                            "Rows in File #1": int(len(raw)),
                            "Invalid/missing coordinates": int((~coord_valid).sum()),
                            "Valid elevations": int((~missing_ele).sum()),
                            "Missing elevations": int(missing_ele.sum()),
                            "Duplicate coordinate-time rows": int(duplicate_mask.sum()),
                            "Rows with duplicate timestamps": int(duplicate_time_mask.sum()),
                            "Missing timestamps": int(time_values.isna().sum()),
                        }
                        st.markdown("#### 🔎 File #1 Quality Pre-Check")
                        st.json(q_summary)
                except Exception as diagnostic_error:
                    st.warning(f"Pre-check notice: {diagnostic_error}")

            st.divider()
            st.markdown("#### ⏳ Time, Gap-Healing & Spike Handling Settings")
            c_time, c_dt, c_gap, c_heal = st.columns([1.0, 1.0, 1.0, 1.8])
            with c_time:
                utc_options = [f"UTC{'+' if i > 0 else ''}{i}" for i in range(-12, 15)]
                file_settings["utc_shift"] = st.selectbox(
                    "UTC time zone:", options=utc_options, index=17, disabled=st.session_state.locked, key="b1_utc_time_zone",
                    help="Local time zone offset from UTC (default: UTC+5 for Tashkent, Uzbekistan). Used to populate local timestamp columns."
                )
            with c_dt:
                file_settings["default_dt"] = st.number_input(
                    "Default time step (s):", value=1.0, min_value=0.01, disabled=st.session_state.locked, key="b1_default_time_step_seconds",
                    help="Fallback sampling step in seconds used only when an uploaded KML/GPX file lacks timestamps (recommended: 1.0 s)."
                )
            with c_gap:
                file_settings["gap_tolerance"] = st.number_input(
                    "Gap threshold (s):", value=15.0, min_value=1.0, disabled=st.session_state.locked, key="b1_gap_tolerance_s",
                    help="Maximum allowable time gap between consecutive GPS points before gap-healing or fail-fast is triggered (recommended: 15.0 s)."
                )
            with c_heal:
                file_settings["anomaly_policy"] = st.selectbox(
                    "Time-Gap & Multi-Point GPS Spike Policy:",
                    options=[
                        "Auto-Heal Gaps & Hampel Multi-Spike Filter (Recommended)",
                        "Strict Fail-Fast (Stop on Gap or Speed Spike)"
                    ],
                    index=0,
                    disabled=st.session_state.locked,
                    key="b1_anomaly_policy",
                    help="Recommended mode automatically cleans multi-point GPS jumps and safely bridges signal losses so urban GPX files process cleanly."
                )

        # --- 2-TAB: MATEMATIKA VA DVIGATEL ---
        with tab_math:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("**WGS84 Ellipsoid & Target Sampling Rate**")
                math_settings["wgs84_a"] = st.number_input(
                    "Semi-major axis (a, m):", value=6378137.0, disabled=st.session_state.locked, key="b1_semi_major_axis_a_m",
                    help="WGS84 reference ellipsoid equatorial radius in meters (standard: 6,378,137.0 m)."
                )
                math_settings["wgs84_e2"] = st.number_input(
                    "Eccentricity squared (e²):", value=0.00669437999, format="%.9f", disabled=st.session_state.locked, key="b1_eccentricity_squared_e",
                    help="WGS84 first eccentricity squared e² = 2f - f² (standard: 0.00669437999)."
                )
                math_settings["target_hz"] = st.number_input(
                    "Target frequency (Hz) [1 Hz recommended for DCMG & MADE]:",
                    value=1.0, min_value=0.2, max_value=50.0, step=1.0,
                    disabled=st.session_state.locked, key="b1_target_frequency_hz",
                    help="1.0 Hz (exact Δt = 1.0000 s) is the international certification standard (WLTC, NEDC, WMTC) and matches Program 2 (DCMG) and Program 3 (MADE)."
                )
                st.markdown("**Savitzky–Golay (SG) Zero-Phase Smoothing Filter**")
                math_settings["sg_window"] = st.number_input(
                    "Window length (odd):", value=9, min_value=3, step=2, disabled=st.session_state.locked, key="b1_window_length",
                    help="Odd window length for zero-phase Savitzky–Golay velocity smoothing before causal differentiation (recommended: 7 to 11)."
                )
                math_settings["sg_polyorder"] = st.number_input(
                    "Polynomial order:", value=2, min_value=1, disabled=st.session_state.locked, key="b1_polynomial_order",
                    help="Polynomial degree for Savitzky–Golay filter (must be strictly less than window length; recommended: 2)."
                )

            with col_m2:
                st.markdown("**9D Kalman Filter (Anisotropic Covariance & Horizontal ZUPT)**")
                math_settings["kalman_r"] = st.number_input(
                    "Horizontal measurement noise (Kalman R_xy):", value=6.0, format="%.2f", disabled=st.session_state.locked, key="b1_measurement_noise_kalman_r",
                    help="GPS position measurement noise variance R (m²). Higher values increase trajectory smoothing (recommended: 4.0 to 10.0)."
                )
                math_settings["kalman_vz_factor"] = st.number_input(
                    "Vertical GPS noise multiplier (R_z / R_xy):", value=4.0, min_value=1.0, max_value=25.0, step=0.5, disabled=st.session_state.locked, key="b1_kalman_vz_factor",
                    help="GPS vertical elevation noise (VDOP) is typically 3–4x larger than horizontal HDOP noise."
                )
                math_settings["kalman_q"] = st.number_input(
                    "Process noise (Kalman Q):", value=0.15, format="%.4f", disabled=st.session_state.locked, key="b1_process_noise_kalman_q",
                    help="Continuous white-jerk process noise spectral density Q (recommended: 0.05 to 0.50)."
                )
                math_settings["zupt_damping"] = st.number_input(
                    "ZUPT damping coefficient:", value=0.85, min_value=0.01, max_value=1.0, disabled=st.session_state.locked, key="b1_zupt_damping_coefficient",
                    help="Zero-Velocity Update (ZUPT) covariance damping factor applied during detected stops (recommended: 0.80 to 0.95)."
                )

        # --- 3-TAB: 3-VARIANTLI YO'L QIYALIGI VA ATROF-MUHIT ---
        with tab_topo:
            st.markdown("### ⛰️ 3-Mode Road Gradient & Hypotenuse Elevation Architecture")
            topo_settings["enabled"] = st.toggle(
                "Enable terrain & road gradient calculation", value=True, disabled=st.session_state.locked, key="b1_enable_terrain_slope_analysis",
                help="When enabled, reconstructs elevation and road gradient θ(s). When disabled, assumes a flat road (0.0°)."
            )

            grade_mode_sel = st.radio(
                "Select Road Gradient Calculation Methodology:",
                options=[
                    "Mode 1: Spatial Segment Mean & Hypotenuse Chord (Recommended — Arithmetic Mean Z per ΔL Segment)",
                    "Mode 2: User-Prescribed Road Gradient (Ignore GPX Z-Axis; Use Constant or Segment Table Grade)",
                    "Mode 3: Stabilized Legacy Spatial Lookback Derivative"
                ],
                index=0,
                disabled=st.session_state.locked,
                key="b1_grade_mode_radio",
                help="Mode 1 averages noisy GPS altitude over ΔL_seg meter segments and connects them with C¹ PCHIP hypotenuse chords. Mode 2 uses user-entered slopes. Mode 3 uses spatial lookback."
            )
            topo_settings["grade_mode"] = grade_mode_sel

            c_env1, c_env2, c_env3 = st.columns(3)
            with c_env1:
                st.markdown("**📐 Segment & Slope Parameters**")
                topo_settings["spatial_step"] = st.number_input(
                    "Segment length ΔL_seg (m) [e.g. 100 m]:",
                    value=100.0, min_value=5.0, max_value=10000.0, step=10.0,
                    disabled=st.session_state.locked, key="b1_spatial_averaging_step_m",
                    help="In Mode 1, the engine averages moving elevation points inside each ΔL_seg meter segment (e.g. 100 m) and constructs a hypotenuse chord between consecutive segment means."
                )
                topo_settings["max_topo_grad"] = st.number_input(
                    "Max physical road slope cap (degrees):",
                    value=18.0, min_value=1.0, max_value=45.0, step=1.0,
                    disabled=st.session_state.locked, key="b1_maximum_allowed_slope_degrees",
                    help="Physical ceiling on road inclination angle (degrees) to reject unphysical GPS altitude glitches (urban/highway standard: 12° to 18°)."
                )
                topo_settings["smooth_hypotenuse_knots"] = st.toggle(
                    "Smooth C¹ transitions between hypotenuse chords",
                    value=True, disabled=st.session_state.locked, key="b1_smooth_hyp_knots",
                    help="When enabled, applies monotonic PCHIP spline interpolation across segment anchor nodes so road grade θ(s) is C¹-continuous without step jumps."
                )

                if grade_mode_sel.startswith("Mode 2"):
                    st.markdown("**✏️ Mode 2 User-Prescribed Grade Settings**")
                    topo_settings["user_grade_unit"] = st.selectbox(
                        "User grade input unit:",
                        options=["Degrees (deg)", "Percent (%)"],
                        index=0, disabled=st.session_state.locked, key="b1_user_grade_unit",
                        help="Choose whether your prescribed road gradient values are entered in Degrees (°) or Percent Slope (%)."
                    )
                    topo_settings["user_grade_submode"] = st.selectbox(
                        "User grade specification:",
                        options=[
                            "Constant Gradient Along Entire Route",
                            "Segment-by-Segment Gradient Sequence (every ΔL_seg m)"
                        ],
                        index=0, disabled=st.session_state.locked, key="b1_user_grade_submode",
                        help="Apply either a single constant slope across the whole route or a comma-separated sequence of slopes for each ΔL_seg segment."
                    )
                    if "Constant" in topo_settings["user_grade_submode"]:
                        topo_settings["user_const_grade_val"] = st.number_input(
                            f"Constant road gradient ({topo_settings['user_grade_unit']}):",
                            value=0.0, min_value=-35.0, max_value=35.0, step=0.2,
                            disabled=st.session_state.locked, key="b1_user_const_grade",
                            help="Constant road inclination applied along the entire trajectory in Mode 2."
                        )
                    else:
                        topo_settings["user_segment_grades_str"] = st.text_input(
                            f"Comma-separated grades for each {topo_settings['spatial_step']:.0f} m segment:",
                            value="0.0, 0.6, 1.2, 0.4, -0.8, 0.0",
                            disabled=st.session_state.locked, key="b1_user_seg_grades_str",
                            help="Enter grade values separated by commas for Segment 1, Segment 2, etc. (repeats cyclically if route is longer)."
                        )

            with c_env2:
                st.markdown("**🌬️ Global Wind & OBD-Equivalent Settings**")
                topo_settings["wind_speed"] = st.number_input(
                    "Mean ambient wind speed (m/s) [0.0 = WLTC/NEDC Standard, >0 = Stochastic Gusts]:",
                    value=0.0, min_value=0.0, max_value=100.0, step=0.2,
                    disabled=st.session_state.locked, key="b1_wind_speed_m_s",
                    help="Set to 0.0 m/s for standard certification cycles, or > 0 m/s to simulate realistic atmospheric boundary-layer wind with Ornstein-Uhlenbeck turbulent gusts."
                )
                wind_directions = {"0° (N)": 0, "45° (NE)": 45, "90° (E)": 90, "135° (SE)": 135, "180° (S)": 180, "225° (SW)": 225, "270° (W)": 270, "315° (NW)": 315}
                wind_choice = st.selectbox(
                    "Wind direction (meteorological FROM):", options=list(wind_directions.keys()),
                    disabled=st.session_state.locked, key="b1_wind_direction",
                    help="Meteorological compass direction from which the ambient wind blows (0° = North, 90° = East)."
                )
                topo_settings["wind_direction_deg"] = wind_directions[wind_choice]
                topo_settings["obd_slip_pct"] = st.number_input(
                    "Nominal tire slip for MADE OBD_Speed (%):",
                    value=0.8, min_value=0.0, max_value=5.0, step=0.2,
                    disabled=st.session_state.locked, key="b1_obd_slip_pct",
                    help="Used when generating the `OBD_Speed [km/h]` column via longitudinal tractive/braking slip κ(a) for the 8-column MADE V1.1.1 export."
                )

            with c_env3:
                st.markdown("**🌡️ Atmospheric Baseline (ISA)**")
                topo_settings["atmosphere_enabled"] = st.toggle(
                    "Enable ISA atmosphere model", value=True, disabled=st.session_state.locked, key="b1_enable_isa_atmosphere_model",
                    help="Computes altitude-dependent air temperature, barometric pressure, and air density ρ(z) according to the International Standard Atmosphere (ISA)."
                )
                topo_settings["base_alt"] = st.number_input(
                    "Baseline elevation (h0, m):", value=450.0, min_value=-500.0, max_value=11000.0, disabled=st.session_state.locked, key="b1_baseline_elevation_h0_m",
                    help="Reference altitude h0 above sea level in meters (Tashkent average: ~450 m)."
                )
                topo_settings["base_press"] = st.number_input(
                    "Baseline air pressure (P0, Pa):", value=101325.0, min_value=1000.0, max_value=120000.0, disabled=st.session_state.locked, key="b1_baseline_air_pressure_p0_pa",
                    help="Reference atmospheric pressure at baseline elevation in Pascals (standard: 101,325 Pa)."
                )
                topo_settings["base_temp"] = st.number_input(
                    "Baseline air temperature (T0, °C):", value=20.0, min_value=-80.0, max_value=60.0, disabled=st.session_state.locked, key="b1_baseline_air_temperature_t0_c",
                    help="Reference ambient air temperature at baseline elevation in °C."
                )

        # --- 4-TAB: FIZIK VA UI CHEGARALAR ---
        with tab_limits:
            c_lim1, c_lim2, c_lim3 = st.columns(3)
            with c_lim1:
                st.markdown("**Longitudinal Limits**")
                limits_settings["max_v"] = st.number_input(
                    "Maximum speed (km/h):", value=140.0, min_value=10.0, max_value=2000.0, disabled=st.session_state.locked, key="b1_maximum_speed_km_h",
                    help="Maximum physically valid vehicle ground speed (km/h). Segments exceeding this are rejected as GPS multipath spikes."
                )
                limits_settings["max_a"] = st.number_input(
                    "Maximum acceleration (m/s²):", value=3.8, min_value=0.5, max_value=500.0, disabled=st.session_state.locked, key="b1_maximum_acceleration_m_s",
                    help="Forward longitudinal acceleration ceiling (m/s²) enforced by the kinematic slew-rate filter (urban EV typical: 3.0 to 4.0 m/s²)."
                )
                limits_settings["max_b"] = st.number_input(
                    "Maximum braking deceleration (m/s²):", value=4.2, min_value=0.5, max_value=500.0, disabled=st.session_state.locked, key="b1_maximum_braking_deceleration_m_s",
                    help="Braking deceleration ceiling (m/s²) enforced by the backward kinematic slew-rate filter to prevent artificial braking cliffs."
                )
            with c_lim2:
                st.markdown("**Lateral & ZUPT Limits**")
                limits_settings["max_alat"] = st.number_input(
                    "Maximum lateral acceleration (m/s²):", value=8.0, min_value=0.1, max_value=500.0, disabled=st.session_state.locked, key="b1_maximum_lateral_acceleration_m_s",
                    help="Threshold for flagging high cornering lateral acceleration a_lat = v * ω (m/s²)."
                )
                limits_settings["max_jerk"] = st.number_input(
                    "Maximum jerk (m/s³):", value=15.0, min_value=0.1, max_value=10000.0, disabled=st.session_state.locked, key="b1_maximum_jerk_m_s",
                    help="Threshold for flagging longitudinal jerk j = da/dt (m/s³)."
                )
                limits_settings["zupt_lim"] = st.number_input(
                    "ZUPT zero-speed threshold (km/h):", value=1.5, min_value=0.0, max_value=50.0, disabled=st.session_state.locked, key="b1_zupt_zero_speed_threshold_km_h",
                    help="Speeds at or below this threshold (default 1.5 km/h) are clamped to exact 0.0 km/h with ΔDistance = 0.000000 km and a = 0.0 m/s²."
                )
                limits_settings["safe_omega"] = st.number_input(
                    "Safe angular velocity (rad/s):", value=0.0001, min_value=1e-8, format="%.7f", disabled=st.session_state.locked, key="b1_safe_angular_velocity_rad_s",
                    help="Minimum yaw rate |ω| (rad/s) required before computing finite turning radius R = v / |ω|."
                )
            with c_lim3:
                st.markdown("**Reference Vehicle VSP & Memory Limits**")
                limits_settings["ref_mass_kg"] = st.number_input(
                    "Reference EV gross mass for VSP (kg):", value=1800.0, min_value=20.0, max_value=5000.0, step=50.0, disabled=st.session_state.locked, key="b1_ref_mass_kg",
                    help="Reference vehicle gross mass (default: 1800 kg Urban Taxi EV) used to compute the `vsp_w_per_kg` column in the 55-column and 29-column DCMG tables."
                )
                limits_settings["max_dist_km"] = st.number_input(
                    "Maximum distance per trip (km):", value=500.0, min_value=0.001, max_value=100000.0, disabled=st.session_state.locked, key="b1_maximum_distance_km",
                    help="Safety upper bound on single-trip distance (km)."
                )
                limits_settings["max_plot_points"] = st.number_input(
                    "Maximum plot points:", value=3000, min_value=100, step=500, disabled=st.session_state.locked, key="b1_maximum_plot_points",
                    help="Maximum visual points rendered in interactive browser charts (extrema-preserving downsampling; exported CSV/XLSX files always retain 100% of samples)."
                )
                limits_settings["max_radius"] = st.number_input(
                    "Maximum turning radius (m):", value=10000.0, min_value=1.0, step=100.0, disabled=st.session_state.locked, key="b1_maximum_turning_radius_m",
                    help="Cap applied to turning radius R (m) on straight road sections where yaw rate ω ≈ 0."
                )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.locked:
            n_q = len(st.session_state.get("raw_file_paths", []))
            st.success(f"🔒 All parameters confirmed ({n_q} trajectory file(s) locked). Proceed to **Block 2** below!")
            if st.button(
                "🔓 Unlock & Edit Parameters", **_st_width_kw(), key="b1_button_unlock_edit_parameters",
                help="Unlocks Block 1 so you can upload different GPX/KML files or modify filter and road-gradient settings."
            ):
                st.session_state.locked = False
                for stale_key in ("processed_data", "per_trip_dfs", "segments_df", "trip_audit_df", "zip_ready", "calc_mode", "processing_quality_report", "_egad_xlsx_sig"):
                    st.session_state.pop(stale_key, None)
                cleanup_temp_files()
                st.rerun()
        else:
            if st.button(
                "✅ Master Lock: Confirm Files & Parameters", type="primary", **_st_width_kw(), key="b1_button_master_lock_confirm_parameters",
                help="Validates all Block 1 parameters, stages the uploaded GPX/KML files, and unlocks the Block 2 execution engine."
            ):
                errors = []
                if not uploaded_files and not use_demo_gpx:
                    errors.append("Please upload at least one GPX/KML file or enable the Built-In Tashkent Demo GPX toggle.")
                if math_settings["sg_window"] % 2 == 0:
                    errors.append("The Savitzky–Golay window length must be an odd number.")
                if math_settings["sg_polyorder"] >= math_settings["sg_window"]:
                    errors.append("The SG polynomial order must be smaller than the window length.")
                if errors:
                    for err in errors:
                        st.error(f"❌ {err}")
                else:
                    cleanup_temp_files()
                    saved_paths = []
                    if uploaded_files:
                        _sorted_ufs = sorted(uploaded_files, key=lambda _f: tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(_f.name))))
                        for idx_f, uf in enumerate(_sorted_ufs, start=1):
                            ext_f = uf.name.split(".")[-1].lower()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext_f}") as tmp_f:
                                tmp_f.write(uf.getbuffer())
                                saved_paths.append({"trip_id": idx_f, "name": uf.name, "ext": ext_f, "path": tmp_f.name})
                    else:
                        for idx_d in range(1, int(demo_gpx_count) + 1):
                            gpx_b = generate_synthetic_tashkent_gpx_bytes(duration_s=900, trip_idx=idx_d)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as tmp_f:
                                tmp_f.write(gpx_b)
                                saved_paths.append({"trip_id": idx_d, "name": f"Tashkent_Demo_Trip_{idx_d:02d}.gpx", "ext": "gpx", "path": tmp_f.name})

                    st.session_state.locked = True
                    st.session_state.raw_file_paths = saved_paths
                    st.session_state.file_settings = dict(file_settings)
                    st.session_state.math_settings = dict(math_settings)
                    st.session_state.topo_settings = dict(topo_settings)
                    st.session_state.limits_settings = dict(limits_settings)
                    st.rerun()


# =====================================================================
# 2-BLOK BOSHI: KINEMATIK VA DINAMIK HISOB-KITOB DVIGATELI (V1.1.1 ENGINE)
# =====================================================================
@nb.njit
def ecef_to_enu_vel(vx, vy, vz, lat_deg, lon_deg):
    """ECEF tezlik vektorlarini (Vx, Vy, Vz) Mahalliy Tangensal Tekislikka (ENU) o'tkazish."""
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

        v_east[i] = -slon * vx[i] + clon * vy[i]
        v_north[i] = -slat * clon * vx[i] - slat * slon * vy[i] + clat * vz[i]
        v_up[i] = clat * clon * vx[i] + clat * slon * vy[i] + slat * vz[i]

    return v_east, v_north, v_up


def parse_large_xml(filepath, ext, settings):
    """GPX/KML parser: nuqta konteksti saqlanadi va noto'g'ri qatorlar sifat auditi uchun qaytariladi."""
    data = []

    def local_name(tag):
        return tag.rsplit("}", 1)[-1]

    def number_or_nan(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else np.nan
        except (TypeError, ValueError, OverflowError):
            return np.nan

    def parse_kml_token(token, coord_style="comma"):
        try:
            if coord_style == "space":
                vals = [number_or_nan(x) for x in token.strip().split()]
            else:
                vals = [number_or_nan(x) for x in token.strip().split(",")]
            if len(vals) < 2:
                return {"lat": np.nan, "lon": np.nan, "ele": np.nan, "time": None}
            lon, lat = vals[0], vals[1]
            ele = vals[2] if len(vals) > 2 else np.nan
            valid = np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            return {"lat": lat if valid else np.nan, "lon": lon if valid else np.nan, "ele": ele, "time": None}
        except Exception:
            return {"lat": np.nan, "lon": np.nan, "ele": np.nan, "time": None}

    ext = str(ext).lower().lstrip(".")
    if ext == "gpx":
        point_tag = settings.get("tag_point", "trkpt")
        lat_tag = settings.get("tag_lat", "lat")
        lon_tag = settings.get("tag_lon", "lon")
        ele_tag = settings.get("tag_ele", "ele")
        time_tag = settings.get("tag_time", "time")
        for _, elem in ET.iterparse(filepath, events=("end",)):
            if local_name(elem.tag) != point_tag:
                continue
            lat = number_or_nan(elem.attrib.get(lat_tag))
            lon = number_or_nan(elem.attrib.get(lon_tag))
            valid = np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            rec = {"lat": lat if valid else np.nan, "lon": lon if valid else np.nan, "ele": np.nan, "time": None}
            for child in elem:
                tag = local_name(child.tag)
                if tag == ele_tag:
                    rec["ele"] = number_or_nan(child.text)
                elif tag == time_tag and child.text and child.text.strip():
                    rec["time"] = child.text.strip()
            data.append(rec)
            elem.clear()
    elif ext == "kml":
        coords_tag = settings.get("tag_coords", "coordinates")
        time_tag = settings.get("tag_time", "when")
        for _, placemark in ET.iterparse(filepath, events=("end",)):
            if local_name(placemark.tag) != "Placemark":
                continue
            track_nodes = [node for node in placemark.iter() if local_name(node.tag) == "Track"]
            for track in track_nodes:
                times = [n.text.strip() for n in track.iter() if local_name(n.tag) in (time_tag, "when", "time") and n.text and n.text.strip()]
                coords = [n for n in track.iter() if local_name(n.tag) == "coord"]
                for i, node in enumerate(coords):
                    rec = parse_kml_token(node.text or "", coord_style="space")
                    if len(times) == len(coords):
                        rec["time"] = times[i]
                    data.append(rec)
            for node in placemark.iter():
                if local_name(node.tag) != coords_tag or not node.text:
                    continue
                tokens = node.text.strip().split()
                recs = [parse_kml_token(token, coord_style="comma") for token in tokens]
                times = [n.text.strip() for n in placemark.iter() if local_name(n.tag) in (time_tag, "when", "time") and n.text and n.text.strip()]
                if not track_nodes and len(times) == len(recs):
                    for rec, ts in zip(recs, times):
                        rec["time"] = ts
                elif not track_nodes and len(recs) == 1 and len(times) == 1:
                    recs[0]["time"] = times[0]
                data.extend(recs)
            placemark.clear()
    else:
        raise ValueError(f"Unsupported input format '{ext}'. Only GPX and KML are supported.")
    return pd.DataFrame(data, columns=["lat", "lon", "ele", "time"])


def wgs84_to_ecef(lat, lon, alt, a, e2):
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    return x, y, z


def ecef_to_wgs84(x, y, z, a, e2):
    b = np.sqrt(a ** 2 * (1 - e2))
    ep2 = (a ** 2 - b ** 2) / b ** 2
    p = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    safe_cos = np.maximum(np.abs(np.cos(lat)), 1e-12)
    near_pole = np.abs(lat) > np.radians(89.0)
    sin_abs = np.maximum(np.abs(np.sin(lat)), 1e-12)
    alt_pole = np.abs(z) / sin_abs - N * (1 - e2)
    alt_equator = p / safe_cos - N
    alt = np.where(near_pole, alt_pole, alt_equator)
    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu_position(x, y, z, lat0_deg, lon0_deg, x0=None, y0=None, z0=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if x0 is None:
        x0, y0, z0 = float(x[0]), float(y[0]), float(z[0])
    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    dx, dy, dz = x - x0, y - y0, z - z0
    east = -np.sin(lon0) * dx + np.cos(lon0) * dy
    north = -np.sin(lat0) * np.cos(lon0) * dx - np.sin(lat0) * np.sin(lon0) * dy + np.cos(lat0) * dz
    up = np.cos(lat0) * np.cos(lon0) * dx + np.cos(lat0) * np.sin(lon0) * dy + np.sin(lat0) * dz
    return east, north, up


@nb.njit
def run_kalman_filter_9d(x_raw, y_raw, z_raw, dt, r_val, q_val, zupt_lim, zupt_damp, vz_factor=4.0):
    """
    Haqiqiy Anizotropik 9D Kalman filtri (V1.1.4):
    - Har bir nuqtada mahalliy vertikal birlik vektor u_i = r_i / ||r_i|| yangilanadi (Yer egriligiga 100% mos);
    - Anizotropik o'lchov kovariatsiyasi: R_ecef = R_xy * I_3 + (R_z - R_xy) * (u_i u_i^T), bu yerda R_z = R_xy * vz_factor;
    - Gorizontal ZUPT detektori (vertikal GPS VDOP sakrashlariga 100% chidamli) va Joseph-forma kovariatsiya yangilanishi.
    """
    x_raw = np.asarray(x_raw, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)
    z_raw = np.asarray(z_raw, dtype=np.float64)
    n = len(x_raw)
    if n == 0:
        return np.empty((9, 0)), np.empty(0), np.empty(0, dtype=np.bool_)
    state = np.zeros((9, n))
    raw_v_arr = np.zeros(n)
    zupt_applied = np.zeros(n, dtype=np.bool_)
    zupt_applied[0] = True
    state[:3, 0] = [x_raw[0], y_raw[0], z_raw[0]]

    r_xy = max(float(r_val), 1e-6)
    vz_mult = max(1.0, float(vz_factor))
    r_delta = r_xy * (vz_mult - 1.0)

    r0_norm = max(1.0, np.sqrt(x_raw[0] ** 2 + y_raw[0] ** 2 + z_raw[0] ** 2))
    up_x, up_y, up_z = x_raw[0] / r0_norm, y_raw[0] / r0_norm, z_raw[0] / r0_norm

    R_init = np.eye(3) * r_xy
    R_init[0, 0] += r_delta * up_x * up_x
    R_init[0, 1] += r_delta * up_x * up_y
    R_init[0, 2] += r_delta * up_x * up_z
    R_init[1, 0] += r_delta * up_y * up_x
    R_init[1, 1] += r_delta * up_y * up_y
    R_init[1, 2] += r_delta * up_y * up_z
    R_init[2, 0] += r_delta * up_z * up_x
    R_init[2, 1] += r_delta * up_z * up_y
    R_init[2, 2] += r_delta * up_z * up_z

    P = np.eye(9) * r_xy
    P[:3, :3] = R_init.copy()
    H = np.zeros((3, 9))
    H[:, :3] = np.eye(3)
    F = np.eye(9)
    F[:3, 3:6] = np.eye(3) * dt
    F[:3, 6:9] = np.eye(3) * (0.5 * dt * dt)
    F[3:6, 6:9] = np.eye(3) * dt

    Q = np.zeros((9, 9))
    q_block = float(q_val) * np.array([
        [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
        [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
        [dt ** 3 / 6.0, dt ** 2 / 2.0, dt]
    ])
    for base in (0, 1, 2):
        ix0, ix1, ix2 = base, base + 3, base + 6
        Q[ix0, ix0], Q[ix0, ix1], Q[ix0, ix2] = q_block[0, 0], q_block[0, 1], q_block[0, 2]
        Q[ix1, ix0], Q[ix1, ix1], Q[ix1, ix2] = q_block[1, 0], q_block[1, 1], q_block[1, 2]
        Q[ix2, ix0], Q[ix2, ix1], Q[ix2, ix2] = q_block[2, 0], q_block[2, 1], q_block[2, 2]

    I = np.eye(9)
    Hv = np.zeros((3, 9))
    Hv[:, 3:6] = np.eye(3)
    Rv = np.eye(3) * max(r_xy * max(float(zupt_damp), 1e-3), 1e-6)
    zupt_thr_ms = float(zupt_lim) / 3.6

    for i in range(1, n):
        # Har bir nuqtada lokal vertikal birlik vektor (Yer egriligiga aniq mos)
        ri_norm = max(1.0, np.sqrt(x_raw[i] ** 2 + y_raw[i] ** 2 + z_raw[i] ** 2))
        up_x = x_raw[i] / ri_norm
        up_y = y_raw[i] / ri_norm
        up_z = z_raw[i] / ri_norm

        dx = x_raw[i] - x_raw[i - 1]
        dy = y_raw[i] - y_raw[i - 1]
        dz = z_raw[i] - z_raw[i - 1]
        d_up = dx * up_x + dy * up_y + dz * up_z
        hx, hy, hz = dx - d_up * up_x, dy - d_up * up_y, dz - d_up * up_z
        raw_v_arr[i] = np.sqrt(hx * hx + hy * hy + hz * hz) / dt

        # Anizotropik ECEF o'lchov kovariatsiya matritsasi R_i = R_xy*I + (R_z - R_xy)*(u u^T)
        R = np.eye(3) * r_xy
        R[0, 0] += r_delta * up_x * up_x
        R[0, 1] += r_delta * up_x * up_y
        R[0, 2] += r_delta * up_x * up_z
        R[1, 0] += r_delta * up_y * up_x
        R[1, 1] += r_delta * up_y * up_y
        R[1, 2] += r_delta * up_y * up_z
        R[2, 0] += r_delta * up_z * up_x
        R[2, 1] += r_delta * up_z * up_y
        R[2, 2] += r_delta * up_z * up_z

        xp = F @ state[:, i - 1].copy()
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = np.linalg.solve(S, H @ Pp).T
        innovation = np.array([x_raw[i], y_raw[i], z_raw[i]]) - H @ xp
        xi = xp + K @ innovation
        A = I - K @ H
        P = A @ Pp @ A.T + K @ R @ K.T
        if raw_v_arr[i] <= zupt_thr_ms:
            zupt_applied[i] = True
            xi[0] = state[0, i - 1]
            xi[1] = state[1, i - 1]
            xi[2] = state[2, i - 1]
            xi[3:9] = 0.0
            Sv = Hv @ P @ Hv.T + Rv
            Kv = np.linalg.solve(Sv, Hv @ P).T
            Av = I - Kv @ Hv
            P = Av @ P @ Av.T + Kv @ Rv @ Kv.T
        state[:, i] = xi
    return state, raw_v_arr, zupt_applied


def process_single_gpx_kml_trajectory(
    trip_id: int,
    trip_name: str,
    raw_path: str,
    ext: str,
    f_set: dict,
    m_set: dict,
    t_set: dict,
    l_set: dict,
    run_dist: bool = False
):
    """
    Bitta GPX/KML faylni o'qish, tozalash, 9D Kalman-ZUPT filtridan o'tkazish,
    3-variantli Gipotenuza yo'l qiyaligini hisoblash va 100% yopiq kinematik zanjir
    (s <-> v <-> a <-> j) asosida barcha 54 + 8 ustunlarni shakllantirish.
    """
    df = parse_large_xml(raw_path, ext, f_set)
    if df.empty:
        raise ValueError(f"[{trip_name}] No coordinates were found in the file.")

    quality_summary = {
        "trip_id": int(trip_id),
        "trip_name": str(trip_name),
        "parsed_rows": int(len(df)),
        "invalid_coordinate_rows_dropped": 0,
        "invalid_timestamp_rows_dropped": 0,
        "exact_duplicate_rows_removed": 0,
        "timestamps_generated_from_default_step": 0,
        "elevation_values_interpolated": 0,
        "gps_spikes_removed_or_smoothed": 0,
        "time_gaps_compressed": 0,
    }

    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ele" not in df.columns:
        df["ele"] = np.nan
    df["ele"] = pd.to_numeric(df["ele"], errors="coerce")

    coord_ok = np.isfinite(df["lat"]) & np.isfinite(df["lon"]) & df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    quality_summary["invalid_coordinate_rows_dropped"] = int((~coord_ok).sum())
    df = df.loc[coord_ok].copy()
    if df.empty:
        raise ValueError(f"[{trip_name}] No valid latitude/longitude coordinates remain.")

    parsed_time = pd.to_datetime(df["time"], format="mixed", errors="coerce", utc=True) if "time" in df.columns else pd.Series(pd.NaT, index=df.index)
    auto_heal = str(f_set.get("anomaly_policy", "Auto-Heal")).startswith("Auto-Heal")

    if parsed_time.notna().sum() >= 3:
        invalid_time = parsed_time.isna()
        if invalid_time.any():
            quality_summary["invalid_timestamp_rows_dropped"] = int(invalid_time.sum())
            df = df.loc[~invalid_time].copy()
            parsed_time = parsed_time.loc[~invalid_time]
        df["time"] = parsed_time.dt.tz_convert(None)
        exact_dup = df.duplicated(subset=["lat", "lon", "ele", "time"], keep="first")
        if exact_dup.any():
            quality_summary["exact_duplicate_rows_removed"] = int(exact_dup.sum())
            df = df.loc[~exact_dup].copy()
        if not auto_heal:
            time_conflicts = df.groupby("time")[["lat", "lon"]].nunique().max(axis=1) > 1
            if time_conflicts.any():
                raise ValueError(f"[{trip_name}] Found {int(time_conflicts.sum())} duplicate conflicting timestamps.")
        df = df.drop_duplicates(subset=["time"], keep="first")
    else:
        quality_summary["timestamps_generated_from_default_step"] = int(len(df))
        time_steps = np.arange(len(df), dtype=float) * float(f_set["default_dt"])
        df["time"] = pd.Timestamp("2026-05-12 03:00:00") + pd.to_timedelta(time_steps, unit="s")

    df = df.sort_values("time").reset_index(drop=True)
    source_time_origin = pd.Timestamp(df["time"].iloc[0])

    valid_ele = np.isfinite(df["ele"].to_numpy(dtype=float))
    missing_ele_count = int((~valid_ele).sum())
    quality_summary["elevation_values_interpolated"] = missing_ele_count
    df["elevation_interpolated_input"] = ~valid_ele
    if not valid_ele.any():
        if auto_heal or str(t_set.get("grade_mode", "")).startswith("Mode 2"):
            df["ele"] = float(t_set.get("base_alt", 450.0))
        else:
            raise ValueError(f"[{trip_name}] All elevation values are missing.")
    elif missing_ele_count > 0:
        ele_values = df["ele"].to_numpy(dtype=float)
        elapsed_for_ele = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        ele_values[~valid_ele] = np.interp(elapsed_for_ele[~valid_ele], elapsed_for_ele[valid_ele], ele_values[valid_ele])
        df["ele"] = ele_values

    if len(df) < 4:
        raise ValueError(f"[{trip_name}] Too few points remain ({len(df)} < 4).")

    max_v_ms = float(l_set["max_v"]) / 3.6
    lats = df["lat"].to_numpy(dtype=float)
    lons = df["lon"].to_numpy(dtype=float)
    eles = df["ele"].to_numpy(dtype=float)
    x_pts, y_pts, z_pts = wgs84_to_ecef(lats, lons, eles, m_set["wgs84_a"], m_set["wgs84_e2"])
    t_sec_raw = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    gap_tolerance_sec = float(f_set.get("gap_tolerance", 15.0))
    dt_pts = np.diff(t_sec_raw, prepend=t_sec_raw[0])
    gap_mask = dt_pts > gap_tolerance_sec
    if np.any(gap_mask):
        if not auto_heal:
            gap_indices = np.flatnonzero(gap_mask)[:10].tolist()
            raise ValueError(f"[{trip_name}] Detected {int(gap_mask.sum())} time gap(s) > {gap_tolerance_sec:g} s at indices {gap_indices}.")
        else:
            ds_gap = np.sqrt(np.diff(x_pts, prepend=x_pts[0]) ** 2 + np.diff(y_pts, prepend=y_pts[0]) ** 2)
            healed_dt = dt_pts.copy()
            for g_i in np.flatnonzero(gap_mask):
                est_dt = max(float(f_set["default_dt"]), min(gap_tolerance_sec, ds_gap[g_i] / max(2.0, 0.35 * max_v_ms)))
                healed_dt[g_i] = est_dt
            t_sec_raw = np.cumsum(healed_dt)
            t_sec_raw = t_sec_raw - t_sec_raw[0]
            quality_summary["time_gaps_compressed"] = int(np.sum(gap_mask))

    removed_spikes = 0
    for _ in range(5):
        if len(df) < 5:
            break
        dt_seg = np.diff(t_sec_raw)
        seg_dist = np.sqrt(np.diff(x_pts) ** 2 + np.diff(y_pts) ** 2 + np.diff(z_pts) ** 2)
        seg_speed = np.divide(seg_dist, dt_seg, out=np.full_like(seg_dist, np.inf), where=dt_seg > 0)
        bad_segs = np.flatnonzero(seg_speed > max_v_ms)
        if len(bad_segs) == 0:
            break
        candidates = [int(idx_b) for idx_b in bad_segs if 0 < idx_b < len(df) - 1]
        if not candidates:
            break
        candidates = sorted(set(candidates))
        if len(df) - len(candidates) < 4:
            break
        df = df.drop(df.index[candidates]).reset_index(drop=True)
        removed_spikes += len(candidates)
        lats = df["lat"].to_numpy(dtype=float)
        lons = df["lon"].to_numpy(dtype=float)
        eles = df["ele"].to_numpy(dtype=float)
        x_pts, y_pts, z_pts = wgs84_to_ecef(lats, lons, eles, m_set["wgs84_a"], m_set["wgs84_e2"])
        t_sec_raw = t_sec_raw[np.setdiff1d(np.arange(len(t_sec_raw)), candidates)]
        t_sec_raw = t_sec_raw - t_sec_raw[0]

    quality_summary["gps_spikes_removed_or_smoothed"] = removed_spikes

    dt_pts = np.diff(t_sec_raw, prepend=t_sec_raw[0])
    ds_pts = np.sqrt(np.diff(x_pts, prepend=x_pts[0]) ** 2 + np.diff(y_pts, prepend=y_pts[0]) ** 2 + np.diff(z_pts, prepend=z_pts[0]) ** 2)
    safe_dt = np.where(dt_pts > 0, dt_pts, np.inf)
    segment_speed = np.divide(ds_pts, safe_dt, out=np.zeros_like(ds_pts), where=np.isfinite(safe_dt) & (safe_dt > 0))
    speed_bad = segment_speed > max_v_ms
    if np.any(speed_bad):
        if not auto_heal:
            bad_idx = np.flatnonzero(speed_bad)[:10].tolist()
            observed_max = float(np.max(segment_speed[speed_bad]) * 3.6)
            raise ValueError(f"[{trip_name}] Segment speed {observed_max:.1f} km/h exceeds max limit at {bad_idx}.")
        else:
            req_dt = np.where(speed_bad, ds_pts / (0.92 * max_v_ms), dt_pts)
            req_dt[0] = 0.0
            t_sec_raw = np.cumsum(req_dt)

    # =====================================================================
    # INTERPOLYATSIYA VA 9D KALMAN FILTRI (QAT'IY Δt = 1/Hz VA YOPIQ KINEMATIK ZANJIR)
    # =====================================================================
    hz = float(m_set["target_hz"])
    dt_target = round(1.0 / max(hz, 1e-6), 6)
    duration_s = float(t_sec_raw[-1] - t_sec_raw[0])
    if duration_s > MAX_TRAJECTORY_DURATION_S:
        raise ValueError(f"[{trip_name}] Duration {duration_s:.1f} s exceeds limit.")
    n_intervals = max(3, int(np.floor(duration_s / dt_target + 1e-9)))
    if n_intervals + 1 > MAX_OUTPUT_POINTS:
        raise ValueError(f"[{trip_name}] Output points exceed cap {MAX_OUTPUT_POINTS:,}.")

    t_target = np.arange(n_intervals + 1, dtype=float) * dt_target

    lat_pchip = PchipInterpolator(t_sec_raw, lats)(t_target)
    lon_unwrapped = np.degrees(np.unwrap(np.radians(lons)))
    lon_pchip = (PchipInterpolator(t_sec_raw, lon_unwrapped)(t_target) + 180.0) % 360.0 - 180.0

    if len(eles) >= 5:
        ele_med = pd.Series(eles).rolling(window=min(7, len(eles)), center=True, min_periods=1).median().to_numpy(dtype=float)
    else:
        ele_med = eles
    ele_pchip = PchipInterpolator(t_sec_raw, ele_med)(t_target)

    x_raw, y_raw, z_raw = wgs84_to_ecef(lat_pchip, lon_pchip, ele_pchip, m_set["wgs84_a"], m_set["wgs84_e2"])
    vz_factor_val = max(1.0, float(m_set.get("kalman_vz_factor", 4.0)))
    state, raw_v_arr, zupt_applied = run_kalman_filter_9d(
        x_raw, y_raw, z_raw, dt_target,
        m_set["kalman_r"], m_set["kalman_q"],
        l_set["zupt_lim"], m_set["zupt_damping"],
        vz_factor_val
    )

    df_res = pd.DataFrame({"time_sec": t_target})
    src_elev_flag = df["elevation_interpolated_input"].to_numpy(dtype=bool)
    right_idx = np.clip(np.searchsorted(t_sec_raw, t_target, side="left"), 0, len(t_sec_raw) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(t_sec_raw) - 1)
    nearest_idx = np.where(np.abs(t_sec_raw[left_idx] - t_target) <= np.abs(t_sec_raw[right_idx] - t_target), left_idx, right_idx)
    df_res["input_elevation_interpolated"] = src_elev_flag[nearest_idx].astype(bool)

    df_res["x_m"], df_res["y_m"], df_res["z_m"] = state[0, :], state[1, :], state[2, :]
    vx, vy, vz = state[3, :], state[4, :], state[5, :]
    ax_ecef, ay_ecef, az_ecef = state[6, :], state[7, :], state[8, :]
    df_res["vx_ecef_ms"], df_res["vy_ecef_ms"], df_res["vz_ecef_ms"] = vx, vy, vz
    df_res["ax_ecef_ms2"], df_res["ay_ecef_ms2"], df_res["az_ecef_ms2"] = ax_ecef, ay_ecef, az_ecef

    lat_smooth, lon_smooth, ele_kalman = ecef_to_wgs84(
        df_res["x_m"].values, df_res["y_m"].values, df_res["z_m"].values, m_set["wgs84_a"], m_set["wgs84_e2"]
    )
    df_res["lat_smooth"], df_res["lon_smooth"] = lat_smooth, lon_smooth
    df_res["elevation_raw_m"] = ele_kalman

    v_east, v_north, v_up = ecef_to_enu_vel(vx, vy, vz, lat_smooth, lon_smooth)
    a_east, a_north, a_up = ecef_to_enu_vel(ax_ecef, ay_ecef, az_ecef, lat_smooth, lon_smooth)

    # 1. Gorizontal ENU tezligi (vertikal GPS sakrashi vz tezlikni sun'iy oshirmasligi uchun)
    zupt_thr_ms = float(l_set["zupt_lim"]) / 3.6
    raw_vel_horiz = np.sqrt(v_east ** 2 + v_north ** 2)
    df_res["velocity_raw_kmh"] = raw_v_arr * 3.6
    df_res["speed_limit_exceeded"] = raw_vel_horiz > max_v_ms

    v_ms_clean = np.clip(raw_vel_horiz, 0.0, max_v_ms)
    v_ms_clean = np.where(zupt_applied | (v_ms_clean <= zupt_thr_ms), 0.0, v_ms_clean)

    # 2. Savitzky-Golay silliqlashni TEZLANISHGA EMAS, AVVAL TEZLIKGA qo'llaymiz (kinematik zanjir uzilmasligi uchun!)
    sg_win, sg_poly = int(m_set["sg_window"]), int(m_set["sg_polyorder"])
    if len(v_ms_clean) > sg_win:
        was_stopped = v_ms_clean <= 0.0
        v_ms_smooth = savgol_filter(v_ms_clean, sg_win, sg_poly, mode="nearest")
        v_ms_clean = np.where(was_stopped | (v_ms_smooth <= zupt_thr_ms), 0.0, np.clip(v_ms_smooth, 0.0, max_v_ms))

    # 3. Standart harakat sikli chegaraviy sharti: t = 0 va t = T da v = 0 km/h
    v_ms_clean[0] = 0.0
    v_ms_clean[-1] = 0.0

    # 4. Ikki tomonlama (Oldinga va Orqaga) fizik tezlanish/tormozlanish cheklagichi (foydalanuvchi limitlari 100% hurmat qilinadi)
    max_a_lim = max(0.5, float(l_set["max_a"]))
    max_b_lim = max(0.5, float(l_set["max_b"]))
    max_dv_up = max_a_lim * dt_target
    max_dv_dn = max_b_lim * dt_target
    for i_f in range(1, len(v_ms_clean)):
        dv_f = v_ms_clean[i_f] - v_ms_clean[i_f - 1]
        if dv_f > max_dv_up:
            v_ms_clean[i_f] = v_ms_clean[i_f - 1] + max_dv_up
        elif dv_f < -max_dv_dn:
            v_ms_clean[i_f] = max(0.0, v_ms_clean[i_f - 1] - max_dv_dn)
    for i_b in range(len(v_ms_clean) - 2, -1, -1):
        if v_ms_clean[i_b] - v_ms_clean[i_b + 1] > max_dv_dn:
            v_ms_clean[i_b] = v_ms_clean[i_b + 1] + max_dv_dn
        elif v_ms_clean[i_b + 1] - v_ms_clean[i_b] > max_dv_up:
            v_ms_clean[i_b] = max(0.0, v_ms_clean[i_b + 1] - max_dv_up)

    v_ms_clean = np.where(v_ms_clean <= zupt_thr_ms, 0.0, v_ms_clean)
    v_ms_clean[0] = 0.0
    v_ms_clean[-1] = 0.0
    v_kmh_exact = np.round(v_ms_clean * 3.6, 4)
    v_ms_exact = v_kmh_exact / 3.6
    zupt_final = v_kmh_exact <= 0.0

    df_res["zupt_update_applied"] = zupt_final.astype(bool)
    df_res["velocity_ms"] = v_ms_exact
    df_res["velocity_kmh"] = v_kmh_exact

    # 5. Masofa (dist_m va Distance [km]): QAT'IY TRAPETSIYA TEZLIK INTEGRALI (to'xtab turganda Δs === 0.000000!)
    ds_step_m = np.zeros(len(v_ms_exact), dtype=float)
    both_stopped = zupt_final[1:] & zupt_final[:-1]
    ds_step_m[1:] = np.where(both_stopped, 0.0, 0.5 * (v_ms_exact[:-1] + v_ms_exact[1:]) * dt_target)
    df_res["dist_m"] = np.cumsum(ds_step_m)

    max_dist_km_lim = float(l_set.get("max_dist_km", 500.0))
    actual_dist_km = float(df_res["dist_m"].iloc[-1]) / 1000.0
    if actual_dist_km > max_dist_km_lim:
        raise ValueError(
            f"[{trip_name}] Computed trip distance ({actual_dist_km:.3f} km) exceeds configured limit ({max_dist_km_lim:.3f} km). "
            "Increase 'Maximum distance per trip (km)' in Block 1 -> Tab 4 if this is a long highway trip."
        )

    # =====================================================================
    # 3-VARIANTLI YO'L QIYALIGI VA GIPOTENUZA BALANDLIK HISOB-KITOBI
    # =====================================================================
    ele_hyp_m, grade_pct, grade_deg, grade_exc, df_segments = compute_road_grade_and_hypotenuse_profile(
        df_res["dist_m"].to_numpy(dtype=float),
        ele_kalman,
        df_res["velocity_ms"].to_numpy(dtype=float),
        float(l_set["zupt_lim"]),
        t_set
    )
    df_res["elevation_m"] = ele_hyp_m
    df_res["grade_percent"] = grade_pct
    df_res["grade_deg"] = grade_deg
    df_res["grade_limit_exceeded"] = grade_exc
    if not df_segments.empty:
        df_segments.insert(0, "trip_id", int(trip_id))
        df_segments.insert(1, "trip_name", str(trip_name))

    x_enu, y_enu, z_enu = ecef_to_enu_position(
        df_res["x_m"].values, df_res["y_m"].values, df_res["z_m"].values, lat_smooth[0], lon_smooth[0]
    )
    df_res["east_m"], df_res["north_m"], df_res["up_m"] = x_enu, y_enu, ele_hyp_m - ele_hyp_m[0]

    vel_sq = v_east ** 2 + v_north ** 2
    valid_heading = (vel_sq >= 1e-4) & (~zupt_final)
    heading_estimated = ~valid_heading
    raw_head = np.arctan2(v_east[valid_heading], v_north[valid_heading])
    if np.any(valid_heading):
        valid_idx = np.flatnonzero(valid_heading)
        raw_head_unwrapped = np.unwrap(raw_head)
        s_head = pd.Series(np.nan, index=np.arange(len(df_res)))
        s_head.iloc[valid_idx] = raw_head_unwrapped
        heading_filled = s_head.ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        heading_filled = (heading_filled + np.pi) % (2.0 * np.pi) - np.pi
    else:
        heading_filled = np.zeros(len(df_res), dtype=float)
    df_res["heading_estimated"] = heading_estimated.astype(bool)
    df_res["heading_rad"] = heading_filled
    df_res["heading_deg"] = (np.degrees(df_res["heading_rad"]) + 360.0) % 360.0

    df_res["v_east_ms"] = v_ms_exact * np.sin(heading_filled)
    df_res["v_north_ms"] = v_ms_exact * np.cos(heading_filled)
    df_res["v_up_ms"] = v_ms_exact * np.tan(np.radians(df_res["grade_deg"].to_numpy(dtype=float)))
    df_res["accel_east_ms2"], df_res["accel_north_ms2"], df_res["accel_up_ms2"] = (
        np.where(zupt_final, 0.0, a_east),
        np.where(zupt_final, 0.0, a_north),
        np.where(zupt_final, 0.0, a_up),
    )
    df_res["accel_vector_ms2"] = np.sqrt(
        df_res["accel_east_ms2"] ** 2 + df_res["accel_north_ms2"] ** 2 + df_res["accel_up_ms2"] ** 2
    )

    # 6. Bo'ylama Tezlanish (a = Δv / Δt) va Jerk (j = Δa / Δt): 100% YOPIQ KINEMATIK ZANJIR!
    acc_exact = np.zeros(len(v_ms_exact), dtype=float)
    acc_exact[1:] = (v_ms_exact[1:] - v_ms_exact[:-1]) / dt_target
    both_zero = zupt_final[1:] & zupt_final[:-1]
    acc_exact[1:] = np.where(both_zero, 0.0, acc_exact[1:])
    acc_exact = np.round(acc_exact, 5)

    jerk_exact = np.zeros(len(acc_exact), dtype=float)
    jerk_exact[1:] = (acc_exact[1:] - acc_exact[:-1]) / dt_target
    jerk_exact[1:] = np.where(both_zero, 0.0, jerk_exact[1:])
    jerk_exact = np.round(jerk_exact, 5)

    df_res["accel_long_ms2"] = acc_exact
    df_res["jerk_ms3"] = jerk_exact
    df_res["accel_limit_exceeded"] = (df_res["accel_long_ms2"] > l_set["max_a"]) | (df_res["accel_long_ms2"] < -l_set["max_b"])
    df_res["jerk_limit_exceeded"] = np.abs(df_res["jerk_ms3"]) > l_set["max_jerk"]

    unwrapped_heading = np.unwrap(df_res["heading_rad"])
    omega_raw = central_diff_1d(unwrapped_heading, dt_target)
    omega_estimate = savgol_filter(omega_raw, sg_win, sg_poly, mode="nearest") if len(df_res) > sg_win else omega_raw
    omega_smooth = omega_estimate.copy()
    omega_smooth[heading_estimated | zupt_final] = 0.0
    df_res["angular_vel_rad_s"], df_res["angular_vel_degs"] = omega_smooth, np.degrees(omega_smooth)

    abs_omega = np.abs(omega_smooth)
    df_res["turn_radius_m"] = np.divide(
        np.abs(df_res["velocity_ms"]), abs_omega,
        out=np.full(len(df_res), float(l_set["max_radius"])),
        where=abs_omega >= float(l_set["safe_omega"])
    )
    df_res["turn_radius_limit_exceeded"] = np.isfinite(df_res["turn_radius_m"]) & (df_res["turn_radius_m"] > float(l_set["max_radius"]))
    df_res["turn_radius_display_m"] = np.minimum(df_res["turn_radius_m"], float(l_set["max_radius"]))

    alat_raw = np.abs(df_res["velocity_ms"] * omega_smooth)
    alat_filtered = savgol_filter(alat_raw, sg_win, sg_poly, mode="nearest") if len(df_res) > sg_win else alat_raw
    alat_filtered[heading_estimated | zupt_final] = 0.0
    df_res["accel_lat_ms2"] = alat_filtered
    df_res["lateral_accel_limit_exceeded"] = df_res["accel_lat_ms2"] > l_set["max_alat"]

    # Atmosfera (ISA) va Tabiiy Turbulent Shamol (Ornstein-Uhlenbeck Gust Model)
    R_spec = 287.058
    T0_k = float(t_set["base_temp"]) + 273.15
    if t_set.get("atmosphere_enabled", True):
        h0, P0 = float(t_set["base_alt"]), float(t_set["base_press"])
        L, g_c, R_u, M_a = 0.0065, 9.80665, 8.31446, 0.0289644
        elevation_arr = df_res["elevation_m"].to_numpy(dtype=float)
        T_k = T0_k - L * (elevation_arr - h0)
        isa_valid = np.isfinite(T_k) & (T_k > 0) & (elevation_arr >= -500.0) & (elevation_arr <= 11000.0)
        air_temp = np.full(len(df_res), float(t_set["base_temp"]))
        air_press = np.full(len(df_res), P0)
        air_density = np.full(len(df_res), 1.204)
        air_temp[isa_valid] = T_k[isa_valid] - 273.15
        air_press[isa_valid] = P0 * ((T_k[isa_valid] / T0_k) ** ((g_c * M_a) / (R_u * L)))
        air_density[isa_valid] = air_press[isa_valid] / (R_spec * T_k[isa_valid])
        df_res["air_temp_C"] = air_temp
        df_res["air_press_Pa"] = air_press
        df_res["air_density_kgm3"] = air_density
    else:
        df_res["air_temp_C"] = float(t_set["base_temp"])
        df_res["air_press_Pa"] = float(t_set["base_press"])
        df_res["air_density_kgm3"] = 1.204

    wind_spd = float(t_set["wind_speed"])
    wind_dir = np.radians(float(t_set["wind_direction_deg"]))
    if wind_spd <= 1e-6:
        df_res["wind_head_ms"] = np.zeros(len(df_res), dtype=float)
        df_res["wind_cross_ms"] = np.zeros(len(df_res), dtype=float)
    else:
        rng_wind = np.random.default_rng(1000 + int(trip_id) * 37)
        tau_gust = 12.0
        sigma_gust = 0.22 * wind_spd
        sigma_dir = np.radians(14.0)
        alpha_ou = np.exp(-dt_target / tau_gust)
        s_scale = np.sqrt(max(1e-6, 1.0 - alpha_ou ** 2))
        gust_u = np.zeros(len(df_res), dtype=float)
        gust_ang = np.zeros(len(df_res), dtype=float)
        for k_w in range(1, len(df_res)):
            gust_u[k_w] = alpha_ou * gust_u[k_w - 1] + sigma_gust * s_scale * rng_wind.standard_normal()
            gust_ang[k_w] = alpha_ou * gust_ang[k_w - 1] + sigma_dir * s_scale * rng_wind.standard_normal()
        inst_wind_spd = np.maximum(0.0, wind_spd + gust_u)
        rel_angle = (df_res["heading_rad"].to_numpy(dtype=float) - wind_dir) + gust_ang
        df_res["wind_head_ms"] = inst_wind_spd * np.cos(rel_angle)
        df_res["wind_cross_ms"] = inst_wind_spd * np.sin(rel_angle)

    # =====================================================================
    # 2-DASTUR (DCMG V1.1.4) VA 3-DASTUR (MADE V1.1.1) MAXSUS USTUNLARI
    # =====================================================================
    offset_text = str(f_set.get("utc_shift", "UTC+5"))
    try:
        offset_hours = int(offset_text.replace("UTC", ""))
    except ValueError:
        offset_hours = 5

    ts_utc = source_time_origin + pd.to_timedelta(df_res["time_sec"].to_numpy(dtype=float), unit="s")
    ts_local = ts_utc + pd.to_timedelta(offset_hours, unit="h")
    df_res["timestamp_utc"] = ts_utc
    df_res["timestamp_display"] = ts_local
    df_res["recorded_datetime_utc"] = ts_utc.strftime("%Y-%m-%d %H:%M:%S")
    df_res["recorded_datetime_local"] = ts_local.strftime("%Y-%m-%d %H:%M:%S")
    df_res["recorded_time_hms"] = ts_local.strftime("%H:%M:%S")
    df_res["recorded_hour_float"] = np.round(ts_local.hour + ts_local.minute / 60.0 + ts_local.second / 3600.0, 4)

    df_res["trip_id"] = int(trip_id)
    df_res["trip_name"] = str(trip_name)
    df_res["trip_time_sec"] = df_res["time_sec"]
    df_res["dt_sec"] = float(dt_target)
    df_res["trip_dist_m"] = df_res["dist_m"]

    v_kmh_arr = df_res["velocity_kmh"].to_numpy(dtype=float)
    v_ms_arr = df_res["velocity_ms"].to_numpy(dtype=float)
    a_ms2_arr = df_res["accel_long_ms2"].to_numpy(dtype=float)
    g_rad_arr = np.radians(df_res["grade_deg"].to_numpy(dtype=float))

    is_stopped_arr = v_kmh_arr <= 0.0
    df_res["is_stopped"] = is_stopped_arr
    mt_ids = np.maximum(1, np.cumsum(np.diff((~is_stopped_arr).astype(int), prepend=0) == 1))
    df_res["microtrip_id"] = mt_ids
    df_res["trip_microtrip_id"] = mt_ids

    modes = np.full(len(df_res), "Cruising", dtype=object)
    modes[is_stopped_arr] = "Idle"
    modes[(~is_stopped_arr) & (a_ms2_arr > 0.10)] = "Acceleration"
    modes[(~is_stopped_arr) & (a_ms2_arr < -0.10)] = "Deceleration"
    df_res["driving_mode"] = modes

    # Fizik VSP (W/kg) — Shamol vektori (v_air = v + v_wind) va ISA rho_air bilan 100% DCMG/MADE sinxron
    ref_m_kg = max(20.0, float(l_set.get("ref_mass_kg", 1800.0)))
    rho_arr = df_res["air_density_kgm3"].to_numpy(dtype=float)
    v_air_arr = np.where(is_stopped_arr, 0.0, v_ms_arr + df_res["wind_head_ms"].to_numpy(dtype=float))
    vsp_arr = (
        v_ms_arr * (1.06 * a_ms2_arr + STANDARD_GRAVITY_MS2 * np.sin(g_rad_arr) + STANDARD_GRAVITY_MS2 * 0.011 * np.cos(g_rad_arr))
        + (0.5 * rho_arr * 0.28 * 2.32 / ref_m_kg) * v_ms_arr * v_air_arr * np.abs(v_air_arr)
    )
    df_res["vsp_w_per_kg"] = np.where(is_stopped_arr, 0.0, vsp_arr)

    slip_pct = float(t_set.get("obd_slip_pct", 0.8))
    slip_frac = (slip_pct / 100.0) * np.tanh(a_ms2_arr / 2.5)
    obd_kmh_arr = np.where(is_stopped_arr, 0.0, np.maximum(0.0, v_kmh_arr * (1.0 + slip_frac)))

    df_res["Time [s]"] = np.round(df_res["time_sec"], 4)
    df_res["Interval [s]"] = np.round(df_res["dt_sec"], 4)
    df_res["GPS_Speed [km/h]"] = np.round(v_kmh_arr, 4)
    df_res["OBD_Speed [km/h]"] = np.round(obd_kmh_arr, 4)
    df_res["Wind_Speed [m/s]"] = np.round(df_res["wind_head_ms"], 4)
    df_res["Distance [km]"] = np.round(df_res["dist_m"] / 1000.0, 6)
    df_res["Acceleration [m/s2]"] = np.round(a_ms2_arr, 5)
    df_res["Road_Gradient [deg]"] = np.round(df_res["grade_deg"], 4)

    quality_summary["zupt_update_samples"] = int(df_res["zupt_update_applied"].sum())
    quality_summary["output_points"] = int(len(df_res))
    quality_summary["duration_seconds"] = round(float(df_res["time_sec"].max()), 2)
    quality_summary["distance_km"] = round(float(df_res["dist_m"].max()) / 1000.0, 3)
    quality_summary["mean_speed_kmh"] = round(float(df_res["velocity_kmh"].mean()), 2)
    quality_summary["max_speed_kmh"] = round(float(df_res["velocity_kmh"].max()), 2)
    quality_summary["mean_grade_deg"] = round(float(df_res["grade_deg"].mean()), 3)
    quality_summary["max_abs_grade_deg"] = round(float(np.max(np.abs(df_res["grade_deg"]))), 3)
    quality_summary["hypotenuse_segments_count"] = int(len(df_segments))

    return df_res, df_segments, quality_summary


def render_block_2():
    st.divider()
    st.markdown("## ⚙️ BLOCK 2: 9D Kalman-ZUPT & 3-Mode Hypotenuse Road Grade Engine (RUN)")

    if not st.session_state.get("locked", False):
        st.warning("⚠️ Please confirm the files and parameters in **Block 1 (Master Lock)** first.")
        return

    col1, col2, col3, col4 = st.columns([2.0, 1.8, 1.3, 1.3])
    with col1:
        run_time = st.button(
            "⏱️ RUN ENGINE (Time Domain — Recommended for DCMG & MADE)", type="primary", **_st_width_kw(), key="ui_run_time_s",
            help="Executes the 9D Kalman-ZUPT filter, closed kinematic chain (s <-> v <-> a), and 3-mode hypotenuse road gradient engine on a uniform time grid (Δt = 1/Hz)."
        )
    with col2:
        run_dist = st.button(
            "📏 RUN ENGINE (Distance Domain Plot X-Axis)", **_st_width_kw(), key="ui_run_distance_m",
            help="Executes the trajectory engine and sets the Block 3 diagnostic plots X-axis to Cumulative Distance s [m]."
        )
    with col3:
        if st.button(
            "🔄 Reset Results Only (Keep Files)", **_st_width_kw(), key="ui_reset_soft",
            help="Clears computed results and frees RAM while keeping your uploaded GPX/KML files and Block 1 configuration intact."
        ):
            for k in ("processed_data", "per_trip_dfs", "segments_df", "trip_audit_df", "zip_ready", "processing_quality_report", "_egad_xlsx_sig"):
                st.session_state.pop(k, None)
            gc.collect()
            st.rerun()
    with col4:
        if st.button(
            "🧹 Deep Clear All & Full Restart", **_st_width_kw(), key="ui_restart",
            help="Clears all session state, temporary files, and caches for a complete fresh restart."
        ):
            cleanup_temp_files()
            st.session_state.clear()
            gc.collect()
            st.rerun()

    if run_time or run_dist:
        for k in ("zip_ready", "processed_data", "per_trip_dfs", "segments_df", "trip_audit_df", "processing_quality_report", "_egad_xlsx_sig"):
            st.session_state.pop(k, None)

        file_list = st.session_state.get("raw_file_paths", [])
        f_set = st.session_state.file_settings
        m_set = st.session_state.math_settings
        t_set = st.session_state.topo_settings
        l_set = st.session_state.limits_settings

        with st.status(f"Processing {len(file_list)} trajectory file(s) through 9D Kalman-ZUPT & Hypotenuse Grade Engine...", expanded=True) as status:
            try:
                all_dfs = []
                per_trip_standalone_dfs = []
                all_segs = []
                all_audits = []
                cum_time_offset = 0.0
                cum_dist_offset = 0.0
                global_mt_offset = 0

                for item in file_list:
                    t_id = int(item["trip_id"])
                    t_name = str(item["name"])
                    t_path = str(item["path"])
                    t_ext = str(item["ext"])

                    st.write(f"⏳ Processing Trip #{t_id}/{len(file_list)}: `{t_name}`...")
                    df_t, seg_t, q_rep = process_single_gpx_kml_trajectory(
                        t_id, t_name, t_path, t_ext, f_set, m_set, t_set, l_set, run_dist=run_dist
                    )

                    # Har bir faylning alohida mustaqil (Time [s] = 0..T, Distance [km] = 0..D) nusxasini saqlaymiz
                    per_trip_standalone_dfs.append(df_t.copy().reset_index(drop=True))

                    # Ko'p faylli batch rejimida global time_sec, dist_m va microtrip_id uzluksiz yuritiladi
                    df_master_item = df_t.copy().reset_index(drop=True)
                    df_master_item["time_sec"] = df_master_item["trip_time_sec"] + cum_time_offset
                    df_master_item["Time [s]"] = np.round(df_master_item["time_sec"], 4)
                    df_master_item["dist_m"] = df_master_item["trip_dist_m"] + cum_dist_offset
                    df_master_item["Distance [km]"] = np.round(df_master_item["dist_m"] / 1000.0, 6)
                    df_master_item["microtrip_id"] = df_master_item["trip_microtrip_id"] + global_mt_offset

                    cum_time_offset = float(df_master_item["time_sec"].iloc[-1]) + float(df_master_item["dt_sec"].iloc[-1])
                    cum_dist_offset = float(df_master_item["dist_m"].iloc[-1])
                    global_mt_offset = int(df_master_item["microtrip_id"].max())

                    all_dfs.append(df_master_item)
                    if not seg_t.empty:
                        all_segs.append(seg_t)
                    all_audits.append(q_rep)

                df_master = pd.concat(all_dfs, ignore_index=True)
                df_segs_master = pd.concat(all_segs, ignore_index=True) if all_segs else pd.DataFrame()
                df_audit_master = pd.DataFrame(all_audits)

                st.session_state.calc_mode = "dist" if run_dist else "time"
                st.session_state.processed_data = df_master
                st.session_state.per_trip_dfs = per_trip_standalone_dfs
                st.session_state.segments_df = df_segs_master
                st.session_state.trip_audit_df = df_audit_master
                st.session_state.processing_quality_report = {
                    "Total_Trips_Processed": len(all_dfs),
                    "Total_Samples": int(len(df_master)),
                    "Total_Duration_s": round(float(df_master["time_sec"].max()), 1),
                    "Total_Distance_km": round(float(df_master["dist_m"].max()) / 1000.0, 3),
                    "Road_Grade_Methodology": t_set.get("grade_mode"),
                    "Segment_Length_m": t_set.get("spatial_step"),
                    "Total_Hypotenuse_Segments": int(len(df_segs_master)),
                }
                status.update(
                    label=f"✅ Completed {len(all_dfs)} trip(s): {len(df_master):,} samples, {len(df_segs_master)} hypotenuse grade segments.",
                    state="complete",
                    expanded=False
                )
            except Exception as e:
                status.update(label="❌ Computation Error", state="error", expanded=True)
                st.error(f"The engine stopped safely: {str(e)}")


# =====================================================================
# 3-BLOK: VIZUALIZATSIYA, GIPOTENUZA QIYALIK DIAGNOSTIKASI VA EKSPORT MARKAZI
# =====================================================================
def write_styled_egad_excel_bytes(sheets_dict: dict) -> bytes:
    """
    MADE 8-ustunli (Yashil), DCMG 54-ustunli (16 Qizil + 13 Sariq + 26 Ko'k) va
    Gipotenuza kesmalar sahifalarini ranglab Excel (.xlsx) faylga yozadi.
    """
    from openpyxl.styles import PatternFill, Font, Alignment

    xlsx_buf = io.BytesIO()
    red_fill = PatternFill(start_color="FFD9534F", end_color="FFD9534F", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFF0AD4E", end_color="FFF0AD4E", fill_type="solid")
    blue_fill = PatternFill(start_color="FF5BC0DE", end_color="FF5BC0DE", fill_type="solid")
    green_made_fill = PatternFill(start_color="FF1E8449", end_color="FF1E8449", fill_type="solid")
    header_fill = PatternFill(start_color="FF1F4E78", end_color="FF1F4E78", fill_type="solid")

    font_white = Font(bold=True, color="FFFFFFFF")
    font_dark = Font(bold=True, color="FF000000")
    align_center = Alignment(horizontal="center", vertical="center")

    red_set = set(RED_MARKOV_CORE_COLUMNS)
    yellow_set = set(YELLOW_ARTICLE_COLUMNS)
    blue_set = set(BLUE_DIAGNOSTIC_COLUMNS)
    made_set = set(MADE_V111_8_COLUMNS)

    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        for s_name, df_s in sheets_dict.items():
            safe_sheet = str(s_name)[:31]
            df_s.to_excel(writer, index=False, sheet_name=safe_sheet)
            ws = writer.sheets[safe_sheet]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                c_name = str(cell.value)
                cell.alignment = align_center
                if c_name in made_set:
                    cell.fill = green_made_fill
                    cell.font = font_white
                elif c_name in red_set:
                    cell.fill = red_fill
                    cell.font = font_white
                elif c_name in yellow_set:
                    cell.fill = yellow_fill
                    cell.font = font_dark
                elif c_name in blue_set:
                    cell.fill = blue_fill
                    cell.font = font_dark
                else:
                    cell.fill = header_fill
                    cell.font = font_white

    return xlsx_buf.getvalue()


def downsample_preserve_extrema(df, max_points, x_col, important_cols=None):
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
        keep.add(int(chunk[0]))
        keep.add(int(chunk[-1]))
        for col in cols:
            vals = pd.to_numeric(df.iloc[chunk][col], errors="coerce").to_numpy(dtype=float)
            valid = np.flatnonzero(np.isfinite(vals))
            if len(valid):
                keep.add(int(chunk[valid[np.argmin(vals[valid])]]))
                keep.add(int(chunk[valid[np.argmax(vals[valid])]]))
    positions = np.array(sorted(keep), dtype=int)
    if len(positions) > max_points:
        positions = positions[np.linspace(0, len(positions) - 1, max_points, dtype=int)]
    return df.iloc[positions].copy()


def render_block_3():
    st.divider()
    st.markdown("## 📊 BLOCK 3: Hypotenuse Topography Diagnostics, Visualization & Universal Export Center")

    if "processed_data" not in st.session_state:
        st.info("Run the engine in **Block 2** to view trajectory analysis, hypotenuse road grade profiles, and export tables.")
        return

    df_full = st.session_state.processed_data
    df_segs = st.session_state.get("segments_df", pd.DataFrame())
    df_audit = st.session_state.get("trip_audit_df", pd.DataFrame())
    quality_report = st.session_state.get("processing_quality_report", {})
    calc_mode = st.session_state.get("calc_mode", "time")
    x_col = "dist_m" if calc_mode == "dist" else "time_sec"

    st.markdown("### 🎛️ Key Kinematic & Topographic Indicators (KPIs)")
    max_v_ms = float(df_full["velocity_ms"].max())
    avg_v_ms = float(df_full["velocity_ms"].mean())
    tot_dist_m = float(df_full["dist_m"].max())
    tot_time_s = float(df_full["time_sec"].max())
    max_grade_deg = float(np.max(np.abs(df_full["grade_deg"])))
    n_trips = int(df_full["trip_id"].nunique())

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric(
        "Processed Trips", f"{n_trips} trip(s)", f"{len(df_full):,} samples",
        help="Total number of GPX/KML trips processed and total 1 Hz time-series rows generated."
    )
    kpi2.metric(
        "Maximum Speed", f"{max_v_ms * 3.6:.1f} km/h", f"{max_v_ms:.1f} m/s",
        help="Peak horizontal ground speed observed after 9D Kalman-ZUPT filtering and rate limiting."
    )
    kpi3.metric(
        "Mean Speed", f"{avg_v_ms * 3.6:.1f} km/h", f"{avg_v_ms:.1f} m/s",
        help="Overall average ground speed across all processed samples (including ZUPT standstill periods)."
    )
    kpi4.metric(
        "Total Distance", f"{tot_dist_m / 1000.0:.3f} km", f"{tot_dist_m:,.0f} m",
        help="Cumulative distance traveled computed via exact trapezoidal integration of ground velocity (Δs = 0 during stops)."
    )
    kpi5.metric(
        "Total Duration", f"{tot_time_s / 60.0:.1f} min", f"{tot_time_s:,.0f} s",
        help="Total duration across all processed trips on the uniform Δt = 1/Hz time grid."
    )
    kpi6.metric(
        "Max |Road Grade|", f"{max_grade_deg:.2f}°", f"{len(df_segs)} Hypotenuse Segs",
        help="Maximum absolute road inclination angle reconstructed by the 3-Mode Hypotenuse Topography Engine."
    )

    # Built-in Automated Kinematic & Physical Integrity Verification Card (Live Computed Metrics in V1.1.2)
    _v_chk = df_full["GPS_Speed [km/h]"].to_numpy(dtype=float)
    _a_chk = df_full["Acceleration [m/s2]"].to_numpy(dtype=float)
    _d_chk = df_full["Distance [km]"].to_numpy(dtype=float)
    _t_ids = df_full["trip_id"].to_numpy(dtype=int)
    _dt_chk = float(df_full["Interval [s]"].iloc[0])
    _same_trip = (_t_ids[1:] == _t_ids[:-1])
    _both_stop = _same_trip & (_v_chk[1:] <= 0.0) & (_v_chk[:-1] <= 0.0)
    _standstill_drift_km = float(np.max(np.abs(np.diff(_d_chk)[_both_stop]))) if np.any(_both_stop) else 0.0
    _a_expected = np.where((_v_chk[1:] <= 0.0) & (_v_chk[:-1] <= 0.0), 0.0, ((_v_chk[1:] - _v_chk[:-1]) / 3.6) / max(_dt_chk, 1e-6))
    _max_a_err = float(np.max(np.abs(_a_chk[1:][_same_trip] - _a_expected[_same_trip]))) if np.any(_same_trip) else 0.0
    _vz_used = float(st.session_state.get("math_settings", {}).get("kalman_vz_factor", 4.0))
    _bound_ok = bool(all(
        (grp["GPS_Speed [km/h]"].iloc[0] == 0.0 and grp["GPS_Speed [km/h]"].iloc[-1] == 0.0)
        for _, grp in df_full.groupby("trip_id")
    ))
    st.success(
        "🛡️ **Automated Kinematic & Physical Integrity Verification (V1.1.4 Live Audit):** "
        f"Uniform Step $\\Delta t = {_dt_chk:.4f}\\text{{ s}}$ | "
        f"Anisotropic KF $R_z/R_{{xy}} = {_vz_used:.1f}$ | "
        f"Standstill Drift ($v_{{k-1}}=v_k=0$): **`{_standstill_drift_km:.6f} km`** | "
        f"Velocity–Accel Identity $\\max|a_k - \\Delta v_k/\\Delta t|$: **`{_max_a_err:.5f} m/s²`** | "
        f"Terminal Boundaries $v(0)=v(T)=0$: **`{'Verified' if _bound_ok else 'Check'}`**"
    )

    limits = st.session_state.get("limits_settings", {})
    max_pts = limits.get("max_plot_points", 3000)
    important_plot_cols = [
        "velocity_kmh", "velocity_raw_kmh", "accel_long_ms2", "dist_m", "grade_deg", "grade_percent",
        "turn_radius_display_m", "heading_deg", "angular_vel_degs", "jerk_ms3",
        "accel_lat_ms2", "elevation_m", "elevation_raw_m", "air_temp_C", "wind_head_ms", "wind_cross_ms"
    ]
    df = downsample_preserve_extrema(df_full, max_pts, x_col, important_plot_cols)

    st.markdown("### 📈 Interactive Multi-Physics & Topography Visualization Suite")
    tab_g, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⛰️ 1. Hypotenuse Road Grade & Segment Audit",
        "🚀 2. Longitudinal Kinematics",
        "🧭 3. Lateral Dynamics & Curvature",
        "🌡️ 4. Wind & ISA Atmosphere",
        "🔬 5. 9D Kalman Filtering & 3D ENU",
        "🌍 6. Interactive Geospatial Map",
        "🔗 7. MADE (8-Col) & DCMG (55-Col) Previews"
    ])

    def create_line_fig(y_col, title, color_seq=None):
        fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=color_seq, render_mode="svg")
        fig.update_traces(connectgaps=False)
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        return fig

    with tab_g:
        st.markdown("#### ⛰️ Raw GPS Elevation vs. Segment-Mean Hypotenuse Elevation & Road Gradient Profile")
        fig_hyp = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=(
                "a) Raw GPS Elevation vs. Segment-Mean Hypotenuse Chord Profile z(s) [m]",
                "b) Resulting Smooth Road Gradient Profile θ(s) [deg] and Slope [%]"
            )
        )
        fig_hyp.add_trace(
            go.Scatter(x=df["dist_m"], y=df["elevation_raw_m"], mode="lines", name="Raw GPS Elevation (Noisy)", line=dict(color="rgba(150,150,150,0.65)", width=1.1, dash="dot")),
            row=1, col=1
        )
        fig_hyp.add_trace(
            go.Scatter(x=df["dist_m"], y=df["elevation_m"], mode="lines", name="Hypotenuse Elevation Profile (Clean)", line=dict(color="#D9534F", width=2.4)),
            row=1, col=1
        )
        if not df_segs.empty and "Start_Dist_m" in df_segs.columns and "Raw_Mean_Elevation_m" in df_segs.columns:
            seg_mid_s = 0.5 * (df_segs["Start_Dist_m"] + df_segs["End_Dist_m"])
            fig_hyp.add_trace(
                go.Scatter(x=seg_mid_s, y=df_segs["Raw_Mean_Elevation_m"], mode="markers", name="Segment Arithmetic Mean Z", marker=dict(color="#1F77B4", size=7, symbol="diamond")),
                row=1, col=1
            )
        fig_hyp.add_trace(
            go.Scatter(x=df["dist_m"], y=df["grade_deg"], mode="lines", name="Road Gradient [deg]", line=dict(color="#2E8B57", width=2.0)),
            row=2, col=1
        )
        fig_hyp.add_trace(
            go.Scatter(x=df["dist_m"], y=df["grade_percent"], mode="lines", name="Road Slope [%]", line=dict(color="#E67E22", width=1.4, dash="dash")),
            row=2, col=1
        )
        fig_hyp.update_xaxes(title_text="Cumulative Distance s [m]", row=2, col=1)
        fig_hyp.update_yaxes(title_text="Elevation [m]", row=1, col=1)
        fig_hyp.update_yaxes(title_text="Grade [deg / %]", row=2, col=1)
        fig_hyp.update_layout(height=620, hovermode="x unified")
        st.plotly_chart(fig_hyp, **_st_width_kw())

        if not df_segs.empty:
            st.markdown("##### 📋 Spatial Segment Arithmetic Mean & Hypotenuse Chord Table (`4_Road_Grade_Hypotenuse_Segments`)")
            st.dataframe(df_segs, **_st_width_kw(), hide_index=True)

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(create_line_fig("velocity_kmh", "1. GPS Filtered Speed (km/h)", ["#00CC96"]), **_st_width_kw())
        with c2:
            st.plotly_chart(create_line_fig("accel_long_ms2", "2. Longitudinal Acceleration (m/s²)", ["#EF553B"]), **_st_width_kw())
        with c3:
            st.plotly_chart(create_line_fig("dist_m", "3. Cumulative Distance (m)", ["#636EFA"]), **_st_width_kw())

    with tab2:
        c4, c5, c6 = st.columns(3)
        with c4:
            st.plotly_chart(create_line_fig("grade_deg", "4. Road Grade (degrees)"), **_st_width_kw())
            st.plotly_chart(create_line_fig("turn_radius_display_m", "8. Turning Radius (m)"), **_st_width_kw())
        with c5:
            st.plotly_chart(create_line_fig("heading_deg", "7. Heading (degrees)"), **_st_width_kw())
            st.plotly_chart(create_line_fig("angular_vel_degs", "9. Angular Velocity (deg/s)"), **_st_width_kw())
        with c6:
            st.plotly_chart(create_line_fig("jerk_ms3", "16. Longitudinal Jerk (m/s³)"), **_st_width_kw())
            st.plotly_chart(create_line_fig("accel_lat_ms2", "10. Lateral Acceleration (m/s²)"), **_st_width_kw())

    with tab3:
        c7, c8, c9 = st.columns(3)
        with c7:
            st.plotly_chart(create_line_fig("elevation_m", "11. Hypotenuse Elevation Profile (m)"), **_st_width_kw())
            st.plotly_chart(create_line_fig("air_temp_C", "15. Air Temperature (°C)"), **_st_width_kw())
        with c8:
            st.plotly_chart(create_line_fig("wind_head_ms", "5. Headwind (m/s)"), **_st_width_kw())
            st.plotly_chart(create_line_fig("air_density_kgm3", "14. Air Density (kg/m³)"), **_st_width_kw())
        with c9:
            st.plotly_chart(create_line_fig("wind_cross_ms", "6. Crosswind (m/s)"), **_st_width_kw())
            st.plotly_chart(create_line_fig("air_press_Pa", "17. Atmospheric Pressure (Pa)"), **_st_width_kw())

    with tab4:
        c10, c11 = st.columns(2)
        with c10:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df[x_col], y=df["velocity_raw_kmh"], mode="lines", name="Raw GPS Speed", line=dict(color="gray", width=1, dash="dot")))
            fig_comp.add_trace(go.Scatter(x=df[x_col], y=df["velocity_kmh"], mode="lines", name="9D Kalman Filtered Speed", line=dict(color="blue", width=2)))
            fig_comp.update_layout(title="12. Raw vs. 9D Kalman Filtered Speed", hovermode="x unified")
            st.plotly_chart(fig_comp, **_st_width_kw())
        with c11:
            fig_3d = go.Figure(data=go.Scatter3d(
                x=df["east_m"], y=df["north_m"], z=df["up_m"],
                mode="lines", line=dict(color="#00CC96", width=3)
            ))
            fig_3d.update_layout(title="13. 3D Trajectory (Local ENU with Hypotenuse Up)", scene=dict(xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Up (m)"), margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_3d, **_st_width_kw())

    with tab5:
        st.markdown("**18. Geospatial Map**")
        map_color_opt = st.selectbox(
            "Color the map by:",
            key="ui_map_color_by",
            options=["velocity_kmh", "grade_deg", "elevation_m"],
            format_func=lambda x: {"velocity_kmh": "Speed (km/h)", "grade_deg": "Road Grade (degrees)", "elevation_m": "Elevation (m)"}[x],
            help="Select which physical variable colors the trajectory points on the interactive OpenStreetMap view."
        )
        max_map_points = 1000
        map_step = max(1, int(np.ceil(len(df_full) / max_map_points)))
        df_map = df_full.iloc[::map_step].copy()
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(
                df_map, lat="lat_smooth", lon="lon_smooth", color=map_color_opt,
                color_continuous_scale=px.colors.sequential.Viridis,
                hover_data=["trip_id", x_col, "velocity_kmh", "grade_deg", "elevation_m"],
                title=f"18. Interactive Map — {map_color_opt}"
            )
            fig_map.update_layout(map_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
        else:
            fig_map = px.scatter_mapbox(
                df_map, lat="lat_smooth", lon="lon_smooth", color=map_color_opt,
                color_continuous_scale=px.colors.sequential.Viridis,
                hover_data=["trip_id", x_col, "velocity_kmh", "grade_deg", "elevation_m"],
                title=f"18. Interactive Map — {map_color_opt}"
            )
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig_map, **_st_width_kw())

    with tab6:
        st.markdown("#### 🔗 Direct Handshake Previews for Program 3 (`MADE V1.1.1`) & Program 2 (`DCMG V1.1.4`)")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown("##### 🟢 `1_MADE_8Col_Ready` (8 Universal Columns for Program 3 & Program 2)")
            st.dataframe(df_full[MADE_V111_8_COLUMNS].head(15), **_st_width_kw(), hide_index=True)
        with p_c2:
            st.markdown("##### 🔴🟡🔵 `2_DCMG_55Col_ColorCoded` (55 Columns for Program 2)")
            st.dataframe(df_full[ALL_54_ORDERED_COLUMNS].head(15), **_st_width_kw(), hide_index=True)
        if not df_audit.empty:
            st.markdown("##### 📋 Per-Trip Processing & Quality Audit Table")
            st.dataframe(df_audit, **_st_width_kw(), hide_index=True)

    # -----------------------------------------------------------------
    # EKSPORT MARKAZI (MADE 8-COL, DCMG 54-COL, DCMG 29-COL, HYPOTENUSE SEGMENTS & ZIP)
    # -----------------------------------------------------------------
    st.divider()
    st.markdown("### 💾 Export Center: Direct MADE V1.1.1 (8-Col), DCMG V1.1.4 (55-Col & 29-Col), Hypotenuse Segments & Master Workbook")

    df_made_8 = df_full[MADE_V111_8_COLUMNS].copy()
    df_dcmg_54 = df_full[ALL_54_ORDERED_COLUMNS].copy()
    df_dcmg_29 = df_full[MARKOV_29_COLUMNS].copy()

    df_pub_cfg = pd.DataFrame([
        {"Parameter": "Font_Family", "Value": "Times New Roman", "Target_Program": "Program 4 (QwatLab Scientific Visualizer)"},
        {"Parameter": "Base_Font_Size_pt", "Value": "11", "Target_Program": "Program 4 (QwatLab Scientific Visualizer)"},
        {"Parameter": "Target_DPI", "Value": "600", "Target_Program": "Program 4 (QwatLab Scientific Visualizer)"},
        {"Parameter": "Color_Palette", "Value": "Okabe-Ito Standard", "Target_Program": "Program 4 (QwatLab Scientific Visualizer)"},
    ])

    master_sheets = {
        "1_MADE_8Col_Ready": df_made_8,
        "2_DCMG_55Col_ColorCoded": df_dcmg_54,
        "3_DCMG_29Col_Markov_Core": df_dcmg_29,
        "4_Road_Grade_Hypotenuse": df_segs if not df_segs.empty else pd.DataFrame({"Info": ["No segments"]}),
        "5_Per_Trip_Audit": df_audit if not df_audit.empty else pd.DataFrame({"Info": ["Single trip"]}),
        "6_Publication_Style_Config": df_pub_cfg,
    }
    _cache_sig = (len(df_full), float(df_full["dist_m"].iloc[-1]), len(df_segs))
    if st.session_state.get("_egad_xlsx_sig") != _cache_sig:
        st.session_state["_egad_master_xlsx"] = write_styled_egad_excel_bytes(master_sheets)
        st.session_state["_egad_made8_xlsx"] = write_styled_egad_excel_bytes({"1_MADE_8Col_Ready": df_made_8})
        st.session_state["_egad_xlsx_sig"] = _cache_sig
    master_xlsx_bytes = st.session_state["_egad_master_xlsx"]
    made_8_xlsx_bytes = st.session_state["_egad_made8_xlsx"]

    q_b1, q_b2, q_b3, q_b4 = st.columns(4)
    with q_b1:
        st.download_button(
            label=f"⬇️ 1_MADE_8Col_Ready (.CSV, {len(df_made_8):,} rows)",
            data=df_made_8.to_csv(index=False).encode("utf-8"),
            file_name="1_EGAD_to_MADE_8Col_Ready.csv",
            mime="text/csv",
            **_st_width_kw(),
            key="egad_dl_made8_csv",
            help="Downloads the standardized 8-column driving cycle CSV ready for direct upload into Program 3 (MADE V1.1.1) or Program 2 (DCMG V1.1.4)."
        )
    with q_b2:
        st.download_button(
            label=f"⬇️ 1_MADE_8Col_Ready (.XLSX, {len(df_made_8):,} rows)",
            data=made_8_xlsx_bytes,
            file_name="1_EGAD_to_MADE_8Col_Ready.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            **_st_width_kw(),
            key="egad_dl_made8_xlsx",
            help="Downloads the color-coded 8-column Excel workbook ready for Program 3 (MADE V1.1.1)."
        )
    with q_b3:
        st.download_button(
            label=f"⬇️ 2_DCMG_55Col_Master (.CSV, {n_trips} trip[s])",
            data=df_dcmg_54.to_csv(index=False).encode("utf-8"),
            file_name="2_EGAD_to_DCMG_55Col_Master.csv",
            mime="text/csv",
            **_st_width_kw(),
            key="egad_dl_dcmg54_csv",
            help="Downloads the complete 55-column multi-trip CSV table (with trip_id and microtrip_id) ready for direct upload into Program 2 (DCMG V1.1.4)."
        )
    with q_b4:
        st.download_button(
            label="⬇️ Master 6-Sheet Color-Coded Workbook (.XLSX)",
            data=master_xlsx_bytes,
            file_name="QwatLab_EGAD_V1_1_4_Master_Workbook.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            **_st_width_kw(),
            key="egad_dl_master_xlsx",
            help="Downloads the 6-sheet Master Excel Workbook containing 8-col MADE, 54-col DCMG, 29-col Markov Core, Hypotenuse Segments, Per-Trip Audit, and Plotter Config."
        )

    per_trip_list = st.session_state.get("per_trip_dfs", [])
    if len(per_trip_list) > 1:
        st.markdown(f"#### 📂 Individual Trip Datasets ({len(per_trip_list)} Separate Trip Files)")
        st.caption("Each uploaded trip below has its own independent `Time [s] = 0..T`, `Distance [km] = 0..D`, and `v(0) = v(T) = 0 km/h` ready for individual upload into `DCMG` or `MADE`:")
        for idx_pt, df_pt in enumerate(per_trip_list, start=1):
            pt_name = str(df_pt["trip_name"].iloc[0]) if "trip_name" in df_pt.columns else f"Trip_{idx_pt:02d}"
            safe_pt = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(pt_name)[0]).strip("_")
            pt_dur = float(df_pt["trip_time_sec"].iloc[-1])
            pt_dist = float(df_pt["trip_dist_m"].iloc[-1]) / 1000.0
            c_pt1, c_pt2, c_pt3 = st.columns([2.2, 1.4, 1.4])
            with c_pt1:
                st.markdown(f"**Trip #{idx_pt}: `{pt_name}`** — Duration: `{pt_dur:.0f} s` (`{len(df_pt):,}` rows) | Distance: `{pt_dist:.3f} km`")
            with c_pt2:
                st.download_button(
                    label=f"⬇️ Trip #{idx_pt} MADE 8-Col (.CSV)",
                    data=df_pt[MADE_V111_8_COLUMNS].to_csv(index=False).encode("utf-8"),
                    file_name=f"Trip_{idx_pt:02d}_{safe_pt}_8Col_MADE.csv",
                    mime="text/csv",
                    **_st_width_kw(),
                    key=f"egad_dl_pt_made_{idx_pt}",
                    help=f"Downloads standalone 8-column MADE CSV for Trip #{idx_pt} ({pt_name}) with Time [s] starting at 0.0 s."
                )
            with c_pt3:
                st.download_button(
                    label=f"⬇️ Trip #{idx_pt} DCMG 55-Col (.CSV)",
                    data=df_pt[ALL_54_ORDERED_COLUMNS].to_csv(index=False).encode("utf-8"),
                    file_name=f"Trip_{idx_pt:02d}_{safe_pt}_55Col_DCMG.csv",
                    mime="text/csv",
                    **_st_width_kw(),
                    key=f"egad_dl_pt_dcmg_{idx_pt}",
                    help=f"Downloads standalone 55-column DCMG CSV for Trip #{idx_pt} ({pt_name}) with Time [s] starting at 0.0 s."
                )

    st.markdown("#### 🗜️ Full Archive Package (.ZIP)")
    col_name, col_fmt = st.columns([2.5, 1.5])
    with col_name:
        zip_filename = st.text_input(
            "Archive filename:", value="QwatLab_EV_EGAD_V1_1_4_Package", key="ui_archive_filename",
            help="Base filename for the downloadable ZIP bundle containing all exported tables, audit logs, and interactive map."
        )
    with col_fmt:
        export_fmt = st.selectbox(
            "Archive table format:", ["Both CSV & XLSX", "CSV Only", "XLSX Only"], key="ui_table_format",
            help="Select whether the ZIP archive should include both CSV and Excel (.xlsx) tables or only one format."
        )

    if st.button(
        "🗜️ Generate Complete ZIP Archive", type="primary", **_st_width_kw(), key="ui_generate_zip_archive",
        help="Packages the 8-column MADE table, 55/29-column DCMG tables, individual trip files, Hypotenuse segment audit, JSON parameter log, and interactive HTML map into a single ZIP file."
    ):
        with st.spinner("Packaging 8-column MADE tables, 55/29-column DCMG tables, Hypotenuse segment audit, and interactive map..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                log_data = {
                    "Platform": "QwatLab: EV E.G.A.D. V1.1.4",
                    "File_Settings": st.session_state.get("file_settings", {}),
                    "Math_Filter_Settings": st.session_state.get("math_settings", {}),
                    "Topography_Hypotenuse_Settings": st.session_state.get("topo_settings", {}),
                    "Physical_Limits": st.session_state.get("limits_settings", {}),
                    "Processing_Quality_Report": quality_report,
                }
                zip_file.writestr("0_Settings_and_Quality_Log.json", json.dumps(log_data, indent=4))

                if export_fmt in ("Both CSV & XLSX", "CSV Only"):
                    zip_file.writestr("1_EGAD_to_MADE_8Col_Ready.csv", df_made_8.to_csv(index=False).encode("utf-8"))
                    zip_file.writestr("2_EGAD_to_DCMG_55Col_ColorCoded.csv", df_dcmg_54.to_csv(index=False).encode("utf-8"))
                    zip_file.writestr("3_EGAD_to_DCMG_29Col_Markov_Core.csv", df_dcmg_29.to_csv(index=False).encode("utf-8"))
                    for idx_pt, df_pt in enumerate(st.session_state.get("per_trip_dfs", []), start=1):
                        pt_name = str(df_pt["trip_name"].iloc[0]) if "trip_name" in df_pt.columns else f"Trip_{idx_pt:02d}"
                        safe_pt = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(pt_name)[0]).strip("_")
                        zip_file.writestr(f"Individual_Trips/Trip_{idx_pt:02d}_{safe_pt}_8Col_MADE.csv", df_pt[MADE_V111_8_COLUMNS].to_csv(index=False).encode("utf-8"))
                        zip_file.writestr(f"Individual_Trips/Trip_{idx_pt:02d}_{safe_pt}_55Col_DCMG.csv", df_pt[ALL_54_ORDERED_COLUMNS].to_csv(index=False).encode("utf-8"))
                    if not df_segs.empty:
                        zip_file.writestr("4_Road_Grade_Hypotenuse_Segments.csv", df_segs.to_csv(index=False).encode("utf-8"))
                    if not df_audit.empty:
                        zip_file.writestr("5_Per_Trip_Quality_Audit.csv", df_audit.to_csv(index=False).encode("utf-8"))

                if export_fmt in ("Both CSV & XLSX", "XLSX Only"):
                    zip_file.writestr("QwatLab_EGAD_V1_1_4_Master_Workbook.xlsx", master_xlsx_bytes)

                zip_file.writestr("6_Interactive_Geospatial_Map.html", fig_map.to_html(full_html=True, include_plotlyjs=True))

            st.session_state.zip_ready = zip_buffer.getvalue()
            st.success("✅ Archive complete! Download using the button below.")

    if "zip_ready" in st.session_state:
        safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", zip_filename.strip()).strip("._-") or "QwatLab_EV_EGAD_V1_1_4"
        if not safe_filename.lower().endswith(".zip"):
            safe_filename += ".zip"
        st.download_button(
            label=f"⬇️ Download {safe_filename}",
            data=st.session_state.zip_ready,
            file_name=safe_filename,
            mime="application/zip",
            key="ui_download_archive",
            type="primary",
            **_st_width_kw(),
            help="Downloads the generated ZIP archive containing all datasets, logs, and maps to your computer."
        )


# =====================================================================
# ASOSIY ISHGA TUSHIRISH (MAIN)
# =====================================================================
if __name__ == "__main__":
    st.title("🔬 QwatLab: EV E.G.A.D. V1.1.4 (Anisotropic 9D Kalman-ZUPT & 3-Mode Hypotenuse Road Grade Engine)")
    st.caption(
        "Experimental GNSS/GPS Trajectory Processing (1–100 Batch GPX/KML Files), Segment-Mean Hypotenuse Road Grade Reconstruction & Direct 8-Col (`MADE`) / 55-Col (`DCMG`) Exporter"
    )

    render_block_0()
    render_block_1()
    render_block_2()
    render_block_3()