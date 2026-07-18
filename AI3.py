# ═══════════════════════════════════════════════════════════════════
# AI.py  —  MLB Statcast Pro Dashboard  (Part 1 of 3)
# Lines 1-800: imports, config, CSS, data loading, helpers
# ═══════════════════════════════════════════════════════════════════

# ── 1. IMPORTS ────────────────────────────────────────────────────────
import os
import re
import warnings
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
st.set_page_config(page_title="MLB Statcast Pro", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

# Szybkie ładowanie
st.cache_data.clear()  # wyczyść cache przy każdym redeploy

# Ogranicz preload
@st.cache_data(ttl=3600, show_spinner=False)
def load_minimal_data():
    # Załaduj tylko to co absolutnie potrzebne na start
    return load_pitching_stats([2025])  # tylko jeden sezon na początek

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_OK = True
except ImportError:
    PYARROW_OK = False

warnings.filterwarnings("ignore")
pd.set_option("styler.render.max_elements", 20_000_000)

# ── 2. PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Statcast Pro",
    page_icon="⚾",
    layout="wide",
)

# ── 3. DATA PATHS ─────────────────────────────────────────────────────
# ── 3. DATA PATHS ─────────────────────────────────────────────────────
DATA_DIR = Path(".")

# Automatyczna lista plików miesięcznych (najlepsze rozwiązanie)
STATCAST_FILES = {
    year: [
        DATA_DIR / f"statcast_{year}_{month:02d}.parquet"
        for month in range(3, 11)
    ]
    for year in [2024, 2025, 2026]
}

# Opcjonalnie: usuń nieistniejące pliki (bezpiecznik)
for year in list(STATCAST_FILES.keys()):
    STATCAST_FILES[year] = [f for f in STATCAST_FILES[year] if f.exists()]
    if not STATCAST_FILES[year]:
        del STATCAST_FILES[year]

PITCHING_FILES = {
    2024: DATA_DIR / "pitching_stats_2024.parquet",
    2025: DATA_DIR / "pitching_stats_2025.parquet",
    2026: DATA_DIR / "pitching_stats_2026.parquet",
}

BR_CITATION = (
    "Data from Baseball Reference and Statcast. "
    "When using Baseball-Reference data, please cite us: "
    "https://www.baseball-reference.com/"
)

# Columns we need when loading statcast per-entity
ENTITY_COLS = [
    "pitch_type", "pitch_name", "game_date", "game_year",
    "release_speed", "effective_speed", "release_spin_rate",
    "release_extension", "arm_angle",
    "pfx_x", "pfx_z", "plate_x", "plate_z",
    "release_pos_x", "release_pos_z",
    "zone", "stand", "p_throws",
    "batter", "pitcher", "player_name",
    "events", "description",
    "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "balls", "strikes", "bb_type",
    "home_team", "away_team", "inning_topbot",
]

# ── 4. CONSTANTS ──────────────────────────────────────────────────────

# Statcast zone centres derived from data means
ZONE_CENTERS = {
    1: (-0.530, 3.152), 2: (-0.004, 3.148), 3: (0.524, 3.143),
    4: (-0.530, 2.515), 5: (0.000,  2.508), 6: (0.530, 2.503),
    7: (-0.524, 1.873), 8: (0.003,  1.869), 9: (0.535, 1.862),
    11: (-0.40, 3.60),  12: (1.10, 2.51),
    13: (-0.40, 1.30),  14: (-1.10, 2.51),
}
SUB_LABELS = ["TL", "TR", "BL", "BR"]

PITCH_COLORS = {
    "FF": "#ef4444", "SI": "#f97316", "FC": "#f59e0b",
    "SL": "#eab308", "ST": "#84cc16", "CH": "#22c55e",
    "CU": "#06b6d4", "KC": "#3b82f6", "SV": "#8b5cf6",
    "FS": "#ec4899", "FO": "#10b981", "KN": "#6366f1",
    "FA": "#fb923c",
}
PITCH_LONG = {
    "FF": "Four-Seam Fastball", "SI": "Sinker",      "FC": "Cutter",
    "SL": "Slider",            "ST": "Sweeper",      "CH": "Changeup",
    "CU": "Curveball",         "KC": "Knuckle Curve","SV": "Slurve",
    "FS": "Splitter",          "FO": "Forkball",     "KN": "Knuckleball",
    "FA": "Fastball",
}

MLB_AVG = {
    "FF": {"velo": 94.0, "spin": 2270, "whiff": 23.5, "hbrk":  6.8, "vbrk": 13.2, "ext": 6.3},
    "SI": {"velo": 93.2, "spin": 2090, "whiff": 16.1, "hbrk": 12.5, "vbrk":  5.4, "ext": 6.2},
    "FC": {"velo": 88.6, "spin": 2460, "whiff": 26.1, "hbrk": -4.2, "vbrk":  9.3, "ext": 6.1},
    "SL": {"velo": 85.0, "spin": 2380, "whiff": 33.2, "hbrk": -6.8, "vbrk":  0.8, "ext": 6.0},
    "ST": {"velo": 82.1, "spin": 2220, "whiff": 37.4, "hbrk":-14.2, "vbrk":  3.1, "ext": 6.1},
    "CH": {"velo": 84.2, "spin": 1780, "whiff": 31.5, "hbrk":  9.8, "vbrk":  2.8, "ext": 6.1},
    "CU": {"velo": 77.4, "spin": 2530, "whiff": 29.8, "hbrk": -1.8, "vbrk": -7.8, "ext": 5.9},
    "KC": {"velo": 76.2, "spin": 2640, "whiff": 30.9, "hbrk": -1.2, "vbrk": -9.5, "ext": 5.9},
    "SV": {"velo": 78.3, "spin": 2310, "whiff": 28.4, "hbrk": -4.8, "vbrk": -5.8, "ext": 5.9},
    "FS": {"velo": 85.3, "spin": 1480, "whiff": 32.1, "hbrk":  5.2, "vbrk": -5.8, "ext": 6.0},
}

STAT_CONFIG = {
    "Whiff %":      {"col": "whiff_pct",     "rng": (0,   70), "fmt": "{:.1f}%"},
    "Swing %":      {"col": "swing_pct",     "rng": (10,  90), "fmt": "{:.1f}%"},
    "Contact %":    {"col": "contact_pct",   "rng": (30, 100), "fmt": "{:.1f}%"},
    "xwOBA":        {"col": "avg_xwoba",     "rng": (0.10, 0.55), "fmt": "{:.3f}"},
    "Exit Velo":    {"col": "avg_ev",        "rng": (70,  98), "fmt": "{:.1f}"},
    "Launch Angle": {"col": "avg_la",        "rng": (-15, 45), "fmt": "{:.1f}°"},
    "Barrel %":     {"col": "barrel_pct",    "rng": (0,   25), "fmt": "{:.1f}%"},
    "Hard Hit %":   {"col": "hard_hit_pct",  "rng": (0,   60), "fmt": "{:.1f}%"},
    "GB %":         {"col": "gb_pct",        "rng": (0,   80), "fmt": "{:.1f}%"},
}
STAT_LABELS = list(STAT_CONFIG.keys())

ARM_SLOTS = [
    (75, 95,  "Over the Top",       "#ef4444"),
    (60, 75,  "High Three-Quarter", "#f97316"),
    (45, 60,  "Three-Quarter",      "#eab308"),
    (30, 45,  "Low Three-Quarter",  "#22c55e"),
    (15, 30,  "Sidearm",            "#06b6d4"),
    ( 0, 15,  "Low Sidearm",        "#8b5cf6"),
    (-90,  0, "Submarine",          "#ec4899"),
]

SWING_EV   = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
               "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
WHIFF_EV   = {"swinging_strike", "swinging_strike_blocked"}
CONTACT_EV = {"foul", "foul_tip", "hit_into_play",
               "hit_into_play_no_out", "hit_into_play_score"}

ALL_COUNTS = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]

REBUILD_PROFILES = {
    "low_k": {
        "title": "🎯 Low Strikeout Rate — Missing Out-Pitch",
        "desc":  "K/9 well below league average (~9.0). Pitcher relies on contact management without a reliable miss-bat weapon.",
        "fix":   "Add sweeper or high-spin curveball; increase 2-strike usage of best whiff pitch. Consider grip changes for more horizontal or vertical break.",
        "timeline": "1–2 seasons",
    },
    "high_bb": {
        "title": "⚠️ Command Issue — High Walk Rate",
        "desc":  "BB/9 elevated above 3.5. Struggles in hitter counts (3-1, 3-2) — lacks reliable strike-throwing secondary.",
        "fix":   "Develop backdoor breaking ball or cutter for zone-attack in hitter counts. Focus on first-pitch strike rate.",
        "timeline": "1 spring training",
    },
    "poor_fip": {
        "title": "🔧 Underlying Stuff Needs Work",
        "desc":  "FIP above 4.50 — actual pitch quality is below average. ERA reflects genuine stuff deficiencies.",
        "fix":   "Movement optimisation (grip/seam adjustment for +3–5\" break), extension mechanics work (+0.5 ft = free velocity), arsenal diversification.",
        "timeline": "Off-season + spring training",
    },
    "hr_suppression": {
        "title": "💣 Home Run Suppression Needed",
        "desc":  "HR/9 above 1.8. Likely pitching up in zone or lacking heavy sink/cut to suppress fly balls.",
        "fix":   "Add sinker or split-finger for ground-ball induction. Work on extension (more perceived velocity, less loft).",
        "timeline": "Off-season mechanical work",
    },
    "era_fip_gap": {
        "title": "🍀 Luck / BABIP — Not Primarily an Arsenal Issue",
        "desc":  "ERA significantly above FIP despite solid underlying metrics (K/9, BB/9 acceptable). Performance likely driven by bad luck on batted balls.",
        "fix":   "Review pitch sequencing and first-pitch approach. Arsenal is not the primary problem — focus on execution and defence.",
        "timeline": "Immediate (coaching / sequencing review)",
    },
}

# ── 5. CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ─────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f17 !important;
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #111621 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

p, div, label, span, li, td, th, h1, h2, h3, h4,
.stMarkdown, [data-testid="stMarkdownContainer"] * { color: #e2e8f0; }

/* ── Widget labels — always white ─────────────────────────────────── */
.stSelectbox label,
.stMultiSelect label,
.stTextInput label,
.stNumberInput label,
.stSlider label,
.stRadio label,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] span {
    color: #e2e8f0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Selectbox / Multiselect boxes ────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #111621 !important;
    border-color: #2a3545 !important;
    color: #e2e8f0 !important;
    font-size: 0.85rem !important;
}
/* Value text inside the select box */
[data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #e2e8f0 !important;
}
/* Dropdown menu items */
[data-baseweb="menu"],
[data-baseweb="menu"] * {
    background: #111621 !important;
    color: #e2e8f0 !important;
}
[data-baseweb="option"]:hover,
[data-baseweb="option"][aria-selected="true"] {
    background: #1a3a66 !important;
}
/* Placeholder & typed text */
[data-baseweb="select"] input,
[data-baseweb="input"] input {
    color: #e2e8f0 !important;
    caret-color: #e2e8f0 !important;
}
/* Multiselect tags */
[data-baseweb="tag"] {
    background: #1a3a66 !important;
    border: 1px solid #2f7cf6 !important;
}
[data-baseweb="tag"] span { color: #79b8ff !important; }
[data-baseweb="tag"] button svg { fill: #79b8ff !important; }

/* ── Text / Number inputs ─────────────────────────────────────────── */
.stTextInput input,
.stNumberInput input {
    background: #111621 !important;
    border-color: #2a3545 !important;
    color: #e2e8f0 !important;
}
.stTextInput input::placeholder,
.stNumberInput input::placeholder { color: #5a6478 !important; }

/* ── Radio pills ──────────────────────────────────────────────────── */
div[data-testid="stRadio"] > div { gap: 8px !important; flex-wrap: wrap; }
div[data-testid="stRadio"] label {
    display: inline-flex !important; align-items: center !important;
    background: #181f2e !important; border: 1px solid #2a3545 !important;
    border-radius: 20px !important; padding: 5px 15px !important;
    cursor: pointer !important; margin: 0 !important;
}
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #e2e8f0 !important; font-size: 0.82rem !important;
    font-weight: 500 !important; line-height: 1 !important;
}
div[data-testid="stRadio"] input[type="radio"] {
    width: 0 !important; height: 0 !important;
    opacity: 0 !important; position: absolute !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #1a3a66 !important; border-color: #2f7cf6 !important;
}
div[data-testid="stRadio"] label:has(input:checked) p,
div[data-testid="stRadio"] label:has(input:checked) span {
    color: #79b8ff !important; font-weight: 600 !important;
}

/* ── Sliders ──────────────────────────────────────────────────────── */
.stSlider [data-baseweb="thumb"] { background: #2f7cf6 !important; border-color: #2f7cf6 !important; }
.stSlider [data-baseweb="track-fill"] { background: #2f7cf6 !important; }

/* ── Expander ─────────────────────────────────────────────────────── */
details > summary {
    background: #111621 !important; border: 1px solid #1e2535 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-size: 0.84rem !important; font-weight: 500 !important;
    padding: 10px 16px !important;
}
details[open] > summary { border-radius: 8px 8px 0 0 !important; }
details > div {
    background: #0b0f17 !important; border: 1px solid #1e2535 !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
    padding: 14px !important;
}
.streamlit-expanderHeader p,
.streamlit-expanderHeader span { color: #e2e8f0 !important; }

/* ── Dataframe ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #1e2535 !important; border-radius: 8px !important; }
.dataframe th { background: #111621 !important; color: #a0aec0 !important; font-size: 0.76rem !important; font-weight: 600 !important; }
.dataframe td { font-size: 0.81rem !important; color: #e2e8f0 !important; }

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {
    background: #181f2e; border: 1px solid #2a3545;
    color: #e2e8f0 !important; border-radius: 7px;
    font-size: 0.83rem; font-weight: 500; transition: all .15s;
}
.stButton > button:hover { background: #1e2535; border-color: #2f7cf6; color: #79b8ff !important; }

/* ── Tabs ─────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #111621 !important; border-bottom: 2px solid #1e2535; gap: 4px; padding: 0 8px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: #8892a4 !important; font-size: 0.88rem !important;
    font-weight: 500 !important; padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    background: #1a3a6655 !important; color: #79b8ff !important;
    border-bottom: 2px solid #2f7cf6 !important; font-weight: 600 !important;
}
[data-testid="stTabPanel"] { padding-top: 20px !important; }

/* ── Cards & layout helpers ───────────────────────────────────────── */
.dash-hdr { padding: 18px 0 4px; border-bottom: 1px solid #1e2535; margin-bottom: 20px; }
.dash-ttl { font-size: 1.75rem; font-weight: 700; color: #f0f6ff !important; letter-spacing: -.4px; }
.dash-sub { font-size: .82rem; color: #7a8494 !important; margin-top: 4px; }

.sec-hdr {
    display: flex; align-items: center; gap: 10px;
    background: linear-gradient(90deg, #1a3a6622, transparent);
    border-left: 3px solid #2f7cf6; padding: 10px 18px;
    border-radius: 0 8px 8px 0; margin: 24px 0 16px;
    color: #79b8ff !important; font-weight: 600; font-size: 1rem;
}

.ref-card  { background: #111621; border: 1px solid #1e2535; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
.ref-title { color: #79b8ff; font-weight: 600; font-size: .83rem; margin-bottom: 8px; }
.ref-body  { color: #a0aec0; font-size: .78rem; line-height: 1.55; }
.ref-badge {
    display: inline-block; background: #181f2e; border: 1px solid #2a3545;
    border-radius: 5px; padding: 3px 9px; margin: 2px 2px 4px 0;
    font-size: .73rem; color: #63b3ff; font-family: 'JetBrains Mono', monospace;
}

.metric-grid { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
.metric-card {
    background: #111621; border: 1px solid #1e2535; border-radius: 9px;
    padding: 11px 14px; flex: 1; min-width: 110px; text-align: center;
}
.metric-label { color: #6e7a8a; font-size: .68rem; text-transform: uppercase; letter-spacing: .8px; margin-bottom: 5px; }
.metric-val   { color: #e2e8f0; font-size: 1.35rem; font-weight: 700; line-height: 1.1; }
.metric-sub   { color: #5a6478; font-size: .7rem; margin-top: 3px; }

.report-wrap { background: #0d1220; border: 1px solid #2a3545; border-radius: 12px; padding: 22px; margin: 10px 0; }
.rpt-h2  { color: #79b8ff; font-size: .98rem; font-weight: 700; margin: 18px 0 7px; padding-bottom: 5px; border-bottom: 1px solid #1e2535; }
.rpt-h3  { color: #63b3ff; font-size: .88rem; font-weight: 600; margin: 12px 0 5px; }
.rpt-p   { color: #c9d1d9; font-size: .85rem; line-height: 1.75; margin: 5px 0; }
.rpt-li  { color: #c9d1d9; font-size: .85rem; line-height: 1.7; margin: 3px 0 3px 16px; }

.pill { display: inline-block; border-radius: 5px; padding: 2px 9px; margin: 1px 2px; font-size: .73rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.pill-elite { background: #14532d; color: #4ade80; border: 1px solid #16a34a; }
.pill-above { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
.pill-avg   { background: #292524; color: #a8a29e; border: 1px solid #57534e; }
.pill-below { background: #4a3200; color: #fbbf24; border: 1px solid #d97706; }
.pill-poor  { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }

.insight-box { background: #0f1e35; border: 1px solid #2f7cf6; border-radius: 8px; padding: 11px 15px; margin: 9px 0; }
.insight-box .rpt-p { color: #93c5fd; }
.warn-box   { background: #1f1000; border: 1px solid #d97706; border-radius: 8px; padding: 11px 15px; margin: 9px 0; }
.warn-box .rpt-p   { color: #fde68a; }
.danger-box { background: #1f0000; border: 1px solid #dc2626; border-radius: 8px; padding: 11px 15px; margin: 9px 0; }
.danger-box .rpt-p { color: #fca5a5; }
.ok-box     { background: #001f0f; border: 1px solid #16a34a; border-radius: 8px; padding: 11px 15px; margin: 9px 0; }
.ok-box .rpt-p     { color: #86efac; }

.score-wrap { background: #1e2535; border-radius: 6px; height: 9px; width: 100%; margin: 7px 0; }
.score-bar  { height: 9px; border-radius: 6px; }
.dash-divider { height: 1px; background: #1e2535; margin: 28px 0; border: none; }
.citation { color: #5a6478 !important; font-size: .73rem !important; line-height: 1.45; }
.stCaption, [data-testid="stCaptionContainer"] p { color: #718096 !important; font-size: .76rem !important; }

@media (max-width: 768px) {
    [data-testid="column"] { min-width: 100% !important; flex: 100% !important; }
    .dash-ttl { font-size: 1.3rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# 6. DATA LOADING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

# Verb pattern used to extract batter names from 'des' field
_VERB_PAT = re.compile(
    r'\b(lines|flies|grounds|strikes|walks|singles|doubles|triples|homers|'
    r'pops|grounded|flied|lined|singled|doubled|tripled|homered|popped|'
    r'struck|walked|reaches|scores|bunts|fouled|batter|intentionally|'
    r'hit by|called)\b',
    re.IGNORECASE,
)


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary outcome flags — bezpieczna wersja (obsługuje brakujące kolumny)."""
    if df.empty:
        return df

    df = df.copy()

    # Description-based flags
    desc = df["description"].fillna("")
    df["is_swing"]   = desc.isin(SWING_EV).astype("int8")
    df["is_whiff"]   = desc.isin(WHIFF_EV).astype("int8")
    df["is_contact"] = desc.isin(CONTACT_EV).astype("int8")

    # Launch metrics
    ls = pd.to_numeric(df.get("launch_speed", pd.Series(dtype=float)), errors="coerce")
    la = pd.to_numeric(df.get("launch_angle",  pd.Series(dtype=float)), errors="coerce")

    df["is_barrel"] = ((ls >= 98) & la.between(26, 30)).fillna(False).astype("int8")
    df["is_hh"]     = (ls >= 95).fillna(False).astype("int8")
    df["is_gb"]     = (la < 10).fillna(False).astype("int8")

    # Movement
    df["hbrk"] = pd.to_numeric(df.get("pfx_x", pd.Series(dtype=float)), errors="coerce") * 12
    df["vbrk"] = pd.to_numeric(df.get("pfx_z", pd.Series(dtype=float)), errors="coerce") * 12

    # Count state
    df["count_state"] = (
        df["balls"].astype(str).str.strip() + "-" +
        df["strikes"].astype(str).str.strip()
    )

    return df


@st.cache_data(ttl=7200, show_spinner=False)
def build_meta_maps(years: tuple = (2024, 2025, 2026)) -> tuple:
    batter_meta:  dict = {}
    pitcher_meta: dict = {}

    for yr in years:
        if yr not in STATCAST_FILES:
            continue

        # ← ZMIANA: czytaj tylko JEDEN plik na rok (wystarczy do budowy indeksu)
        paths_to_scan = STATCAST_FILES[yr][:1]   # tylko pierwszy miesiąc
        # Jeśli chcesz pełny indeks teamów, weź max 2 pliki:
        # paths_to_scan = STATCAST_FILES[yr][:2]

        for path in paths_to_scan:
            if not path.exists():
                continue
            meta_cols = ["batter", "pitcher", "player_name", "des",
                         "stand", "p_throws", "home_team", "away_team", "inning_topbot"]
            try:
                df = pd.read_parquet(path, engine="pyarrow", columns=meta_cols)
            except Exception:
                continue

            # ── Pitcher meta ─────────────────────────────────────
            pm = (df[["pitcher", "player_name", "p_throws"]]
                  .dropna(subset=["player_name"])
                  .drop_duplicates("pitcher"))
            for _, row in pm.iterrows():
                pid = int(row["pitcher"])
                if pid not in pitcher_meta:
                    pitcher_meta[pid] = {
                        "name": str(row["player_name"]),
                        "hand": str(row["p_throws"]),
                    }

            # ── Batter team + name ───────────────────────────────
            df["batter_team"] = np.where(
                df["inning_topbot"] == "Top",
                df["away_team"],
                df["home_team"],
            )
            bm = (df[["batter", "stand", "batter_team"]]
                  .dropna(subset=["batter"])
                  .groupby("batter")
                  .first()
                  .reset_index())
            for _, row in bm.iterrows():
                bid = int(row["batter"])
                if bid not in batter_meta:
                    batter_meta[bid] = {
                        "name":  None,
                        "team":  str(row.get("batter_team", "?")),
                        "stand": str(row.get("stand", "?")),
                    }

            # ── Batter name from 'des' field ─────────────────────
            des_df = df[df["des"].notna() & (df["des"].str.len() > 5)]
            first_des = des_df.groupby("batter")["des"].first()
            for bid_raw, des in first_des.items():
                bid = int(bid_raw)
                if batter_meta.get(bid, {}).get("name"):
                    continue
                m = _VERB_PAT.search(str(des))
                if m:
                    name_part = des[: m.start()].strip()
                    words = name_part.split()
                    if 2 <= len(words) <= 4:
                        if bid not in batter_meta:
                            batter_meta[bid] = {"name": name_part, "team": "?", "stand": "?"}
                        else:
                            batter_meta[bid]["name"] = name_part

    return batter_meta, pitcher_meta


# ─────────────────────────────────────────────────────────────────────
# NOWE WERSJE FUNKCJI ŁADOWANIA (zastąp stare)
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_pitcher_data(pitcher_id: int, year: int) -> pd.DataFrame:
    """Ładuje dane dla jednego pitchera — teraz po miesiącach."""
    if year not in STATCAST_FILES:
        return pd.DataFrame()

    dfs = []
    for path in STATCAST_FILES[year]:
        if not path.exists():
            continue
        try:
            if PYARROW_OK:
                table = pq.read_table(
                    str(path),
                    columns=[c for c in ENTITY_COLS if c in pq.read_schema(str(path)).names],
                    filters=[("pitcher", "=", pitcher_id)],
                )
                df = table.to_pandas()
            else:
                df = pd.read_parquet(path, columns=ENTITY_COLS)
                df = df[df["pitcher"] == pitcher_id]
            
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()
    return _add_flags(pd.concat(dfs, ignore_index=True))


@st.cache_data(ttl=3600, show_spinner=False)
def load_batter_data(batter_id: int, year: int) -> pd.DataFrame:
    """Analogicznie dla battera."""
    if year not in STATCAST_FILES:
        return pd.DataFrame()

    dfs = []
    for path in STATCAST_FILES[year]:
        if not path.exists():
            continue
        try:
            if PYARROW_OK:
                table = pq.read_table(
                    str(path),
                    columns=[c for c in ENTITY_COLS if c in pq.read_schema(str(path)).names],
                    filters=[("batter", "=", batter_id)],
                )
                df = table.to_pandas()
            else:
                df = pd.read_parquet(path, columns=ENTITY_COLS)
                df = df[df["batter"] == batter_id]
            
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()
    return _add_flags(pd.concat(dfs, ignore_index=True))

@st.cache_data(ttl=7200, show_spinner=False)
def load_pitching_stats(years: tuple = (2024, 2025, 2026)) -> pd.DataFrame:
    """Load Baseball Reference pitching stats — nie zmieniło się dużo."""
    parts = []
    for yr in years:
        path = PITCHING_FILES.get(yr)
        if path is None or not path.exists():
            continue
        try:
            df = pd.read_parquet(path, engine="pyarrow")
            df["season"] = yr

            # Normalise player name column
            for col in ["Player", "Player▲", "Name"]:
                if col in df.columns:
                    df["Name"] = (df[col]
                                  .astype(str)
                                  .str.replace(r"[▲\*#]", "", regex=True)
                                  .str.strip())
                    break

            # Normalise BR ID
            for col in ["Player-additional", "Player▲-additional", "bbref_id"]:
                if col in df.columns:
                    df["bbref_id"] = df[col]
                    break

            # Cast numeric columns
            numeric_cols = ["ERA", "FIP", "WHIP", "SO9", "BB9", "HR9", "ERA+", 
                           "IP", "Age", "WAR", "SO", "BB", "H9", "H"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Derive SO9 / BB9 if missing
            if "SO9" not in df.columns and "SO" in df.columns and "IP" in df.columns:
                df["SO9"] = (df["SO"] / df["IP"].replace(0, np.nan)) * 9
            if "BB9" not in df.columns and "BB" in df.columns and "IP" in df.columns:
                df["BB9"] = (df["BB"] / df["IP"].replace(0, np.nan)) * 9

            parts.append(df)
        except Exception as e:
            st.warning(f"Błąd przy wczytywaniu pitching_stats_{yr}: {e}")
            continue

    if not parts:
        return pd.DataFrame()
    
    return pd.concat(parts, ignore_index=True)


@st.cache_data(ttl=7200, show_spinner=False)
def precompute_zone_stats(years: tuple = (2024, 2025, 2026)) -> pd.DataFrame:
    """Precompute — teraz po miesiącach."""
    parts = []
    for yr in years:
        if yr not in STATCAST_FILES:
            continue
        for path in STATCAST_FILES[yr]:
            if not path.exists():
                continue
            agg_cols = [
                "pitch_type", "zone", "stand", "p_throws", "balls", "strikes",
                "description", "launch_speed", "launch_angle",
                "estimated_woba_using_speedangle",
            ]
            try:
                if PYARROW_OK:
                    use_cols = [c for c in agg_cols if c in pq.read_schema(str(path)).names]
                    df = pd.read_parquet(path, columns=use_cols)
                else:
                    df = pd.read_parquet(path, columns=agg_cols)
                df["game_year"] = yr
                df = _add_flags(df)
                parts.append(df)
            except Exception:
                continue

    if not parts:
        return pd.DataFrame()

    full = pd.concat(parts, ignore_index=True)

    grp = full.groupby(
        ["game_year", "zone", "pitch_type", "p_throws", "stand", "count_state"],
        dropna=False,
        as_index=False,
    ).agg(
        total       = ("is_swing",   "count"),
        swings      = ("is_swing",   "sum"),
        whiffs      = ("is_whiff",   "sum"),
        contacts    = ("is_contact", "sum"),
        barrels     = ("is_barrel",  "sum"),
        hard_hits   = ("is_hh",      "sum"),
        gbs         = ("is_gb",      "sum"),
        batted      = ("launch_speed","count"),
        avg_ev      = ("launch_speed", "mean"),
        avg_la      = ("launch_angle", "mean"),
        avg_xwoba   = ("estimated_woba_using_speedangle", "mean"),
    )

    # Compute rate stats
    n  = grp["total"].replace(0, np.nan)
    sw = grp["swings"].replace(0, np.nan)
    bt = grp["batted"].replace(0, np.nan)

    grp["swing_pct"]    = (grp["swings"]   / n  * 100).round(1)
    grp["whiff_pct"]    = (grp["whiffs"]   / sw * 100).round(1)
    grp["contact_pct"]  = (grp["contacts"] / sw * 100).round(1)
    grp["barrel_pct"]   = (grp["barrels"]  / bt * 100).round(1)
    grp["hard_hit_pct"] = (grp["hard_hits"]/ bt * 100).round(1)
    grp["gb_pct"]       = (grp["gbs"]      / bt * 100).round(1)
    grp["avg_ev"]       = grp["avg_ev"].round(1)
    grp["avg_la"]       = grp["avg_la"].round(1)
    grp["avg_xwoba"]    = grp["avg_xwoba"].round(3)
    return grp


def compute_zone_stats_from_raw(
    df: pd.DataFrame,
    extra_filters: dict | None = None,
) -> pd.DataFrame:
    """
    Compute zone-level stats from a raw (per-entity) statcast DataFrame.
    Optional extra_filters: {"count_state": "0-2", "stand": "R", "pitch_type": "FF"}
    Returns a DataFrame with one row per zone and all STAT_CONFIG columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    dff = df.copy()

    if extra_filters:
        if extra_filters.get("count_state"):
            dff = dff[dff["count_state"] == extra_filters["count_state"]]
        if extra_filters.get("stand") and extra_filters["stand"] != "All":
            dff = dff[dff["stand"] == extra_filters["stand"]]
        if extra_filters.get("p_throws") and extra_filters["p_throws"] != "All":
            dff = dff[dff["p_throws"] == extra_filters["p_throws"]]
        if extra_filters.get("pitch_type") and extra_filters["pitch_type"] != "All":
            dff = dff[dff["pitch_type"] == extra_filters["pitch_type"]]

    dff = dff[dff["zone"].between(1, 14)]
    if dff.empty:
        return pd.DataFrame()

    grp = dff.groupby("zone", as_index=False).agg(
        total     = ("is_swing",  "count"),
        swings    = ("is_swing",  "sum"),
        whiffs    = ("is_whiff",  "sum"),
        contacts  = ("is_contact","sum"),
        barrels   = ("is_barrel", "sum"),
        hard_hits = ("is_hh",     "sum"),
        gbs       = ("is_gb",     "sum"),
        batted    = ("launch_speed","count"),
        avg_ev    = ("launch_speed","mean"),
        avg_la    = ("launch_angle","mean"),
        avg_xwoba = ("estimated_woba_using_speedangle","mean"),
    )

    n  = grp["total"].replace(0, np.nan)
    sw = grp["swings"].replace(0, np.nan)
    bt = grp["batted"].replace(0, np.nan)

    grp["swing_pct"]    = (grp["swings"]   / n  * 100).round(1)
    grp["whiff_pct"]    = (grp["whiffs"]   / sw * 100).round(1)
    grp["contact_pct"]  = (grp["contacts"] / sw * 100).round(1)
    grp["barrel_pct"]   = (grp["barrels"]  / bt * 100).round(1)
    grp["hard_hit_pct"] = (grp["hard_hits"]/ bt * 100).round(1)
    grp["gb_pct"]       = (grp["gbs"]      / bt * 100).round(1)
    grp["avg_ev"]       = grp["avg_ev"].round(1)
    grp["avg_la"]       = grp["avg_la"].round(1)
    grp["avg_xwoba"]    = grp["avg_xwoba"].round(3)
    return grp


def apply_tab1_filters(
    zone_df: pd.DataFrame,
    years: list,
    pitch_type: str,
    p_throws: str,
    stand: str,
    count_state: str,
) -> pd.DataFrame:
    """
    Filter the pre-aggregated zone_stats DataFrame and re-aggregate.
    Returns a zone-level DataFrame suitable for draw_heatmap().
    """
    dff = zone_df[zone_df["game_year"].isin(years)].copy()
    if pitch_type  != "All": dff = dff[dff["pitch_type"]  == pitch_type]
    if p_throws    != "All": dff = dff[dff["p_throws"]    == p_throws]
    if stand       != "All": dff = dff[dff["stand"]       == stand]
    if count_state != "All": dff = dff[dff["count_state"] == count_state]

    if dff.empty:
        return pd.DataFrame()

    # Re-aggregate sums across filters, then recompute rates
    grp = dff.groupby("zone", as_index=False).agg(
        total     = ("total",    "sum"),
        swings    = ("swings",   "sum"),
        whiffs    = ("whiffs",   "sum"),
        contacts  = ("contacts", "sum"),
        barrels   = ("barrels",  "sum"),
        hard_hits = ("hard_hits","sum"),
        gbs       = ("gbs",      "sum"),
        batted    = ("batted",   "sum"),
        avg_ev    = ("avg_ev",   "mean"),
        avg_la    = ("avg_la",   "mean"),
        avg_xwoba = ("avg_xwoba","mean"),
    )
    n  = grp["total"].replace(0, np.nan)
    sw = grp["swings"].replace(0, np.nan)
    bt = grp["batted"].replace(0, np.nan)

    grp["swing_pct"]    = (grp["swings"]   / n  * 100).round(1)
    grp["whiff_pct"]    = (grp["whiffs"]   / sw * 100).round(1)
    grp["contact_pct"]  = (grp["contacts"] / sw * 100).round(1)
    grp["barrel_pct"]   = (grp["barrels"]  / bt * 100).round(1)
    grp["hard_hit_pct"] = (grp["hard_hits"]/ bt * 100).round(1)
    grp["gb_pct"]       = (grp["gbs"]      / bt * 100).round(1)
    grp["avg_ev"]       = grp["avg_ev"].round(1)
    grp["avg_la"]       = grp["avg_la"].round(1)
    grp["avg_xwoba"]    = grp["avg_xwoba"].round(3)
    return grp


# ══════════════════════════════════════════════════════════════════════
# 7. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def safe_num(v, default: float = np.nan) -> float:
    """Safely cast any value to float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def arm_info(angle: float) -> tuple:
    """Return (slot_name, hex_color) for an arm angle in degrees."""
    try:
        angle = float(angle)
    except (TypeError, ValueError):
        return "Unknown", "#94a3b8"
    if np.isnan(angle):
        return "Unknown", "#94a3b8"
    for lo, hi, name, color in ARM_SLOTS:
        if lo <= angle < hi:
            return name, color
    return "Unknown", "#94a3b8"


def rate(val: float, avg: float,
         hi_is_good: bool = True,
         thr: tuple = (5, 10)) -> str:
    """Rate a metric vs its MLB average. Returns elite/above/avg/below/poor."""
    try:
        d = (val - avg) if hi_is_good else (avg - val)
    except TypeError:
        return "avg"
    if d >= thr[1]:   return "elite"
    if d >= thr[0]:   return "above"
    if d >= -thr[0]:  return "avg"
    if d >= -thr[1]:  return "below"
    return "poor"


_RATE_LABEL = {
    "elite": "🟢 Elite",  "above": "🔵 Above Avg",
    "avg":   "⚪ Average", "below": "🟡 Below Avg",
    "poor":  "🔴 Needs Work",
}
_RATE_CSS = {
    "elite": "pill-elite", "above": "pill-above",
    "avg":   "pill-avg",   "below": "pill-below",
    "poor":  "pill-poor",
}


def pill(text: str, css_class: str) -> str:
    return f'<span class="pill {css_class}">{text}</span>'


def rate_pill(val: float, avg: float,
              hi_is_good: bool = True,
              thr: tuple = (5, 10)) -> str:
    """Return a ready-to-render HTML pill for a value vs its average."""
    r = rate(val, avg, hi_is_good, thr)
    return pill(_RATE_LABEL[r], _RATE_CSS[r])


def classify_subzone(plate_x: float, plate_z: float, zone: int) -> str:
    """
    Split each Statcast zone into 4 sub-quadrants (TL, TR, BL, BR)
    using the zone's known centre from ZONE_CENTERS.
    """
    cx, cz = ZONE_CENTERS.get(int(zone), (0.0, 2.5))
    try:
        top  = "T" if float(plate_z) >= cz else "B"
        side = "L" if float(plate_x) <  cx else "R"
    except (TypeError, ValueError):
        return "TL"
    return top + side


def pitcher_display_name(raw_name: str) -> str:
    """Convert 'Last, First' → 'First Last'."""
    if not raw_name or raw_name == "nan":
        return raw_name or "Unknown"
    parts = raw_name.split(",")
    if len(parts) == 2:
        return f"{parts[1].strip()} {parts[0].strip()}"
    return raw_name.strip()


def build_pitcher_selectbox(pitcher_meta: dict) -> tuple:
    """
    Build a sorted list of display strings and a reverse lookup dict.
    Returns (display_list, {display_str: pitcher_id}).
    """
    dm = {}
    for pid, meta in pitcher_meta.items():
        dn = pitcher_display_name(meta["name"])
        hand = meta.get("hand", "?")
        key  = f"{dn}  ({'RHP' if hand=='R' else 'LHP'})"
        dm[key] = pid
    return sorted(dm.keys()), dm


def build_batter_selectbox(batter_meta: dict) -> tuple:
    """
    Build a sorted list of display strings and a reverse lookup dict.
    Returns (display_list, {display_str: batter_id}).
    """
    dm = {}
    for bid, meta in batter_meta.items():
        name = meta.get("name") or f"Batter #{bid}"
        team = meta.get("team", "?")
        stand= meta.get("stand", "?")
        key  = f"{name}  ({team} · {'RHB' if stand=='R' else 'LHB' if stand=='L' else stand})"
        dm[key] = bid
    return sorted(dm.keys()), dm



# ═══════════════════════════════════════════════════════════════════
# AI.py  —  MLB Statcast Pro Dashboard  (Part 2 of 3)
# Lines 801-1600: drawing, analysis, report generators
# Paste BELOW Part 1 in the final AI.py
# ═══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# 8. HEATMAP DRAWING
# ══════════════════════════════════════════════════════════════════════

def draw_heatmap(
    df_f: pd.DataFrame,
    stat_label: str,
    title: str,
    subzones: bool = False,
    batter_mode: bool = False,
) -> plt.Figure:
    """
    Draw a standard Statcast 14-zone heatmap.

    Parameters
    ----------
    df_f        : DataFrame with columns 'zone','total' + stat column
    stat_label  : key into STAT_CONFIG
    title       : figure title string
    subzones    : if True each zone 1-9 is drawn as 2×2 sub-grid
    batter_mode : invert colour direction (hot = bad for pitcher = good for batter analysis)
    """
    cfg  = STAT_CONFIG.get(stat_label, STAT_CONFIG["Whiff %"])
    col  = cfg["col"]
    vmin, vmax = cfg["rng"]
    fmt  = cfg["fmt"]

    empty_fig = lambda msg="No data": _empty_heatmap(msg)

    if df_f is None or df_f.empty:
        return empty_fig()
    if col not in df_f.columns:
        return empty_fig(f"Column '{col}' not found")

    pv = df_f.groupby("zone")[col].mean().to_dict()
    pp = df_f.groupby("zone")["total"].sum().to_dict()

    # Colour map direction
    if batter_mode:
        cmap = sns.color_palette("RdYlGn_r", as_cmap=True)   # red = danger for batter
    else:
        cmap = sns.color_palette("YlOrRd",   as_cmap=True)   # red = high stat

    # ── Geometry ───────────────────────────────────────────────────
    brd = 0.85; ms = 3.3; cell = ms / 3
    mx  = brd;  my = brd; top  = my + ms
    rx  = mx + ms; hlf = ms / 2; sy = 2.5

    fig, ax = plt.subplots(figsize=(7, 7.2))
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#0b0f17")

    # ── Helpers ────────────────────────────────────────────────────
    def _fill(z):
        v, t = pv.get(z, np.nan), pp.get(z, 0)
        if pd.isna(v) or t == 0:
            return "#1c2230"
        return cmap(np.clip((v - vmin) / (vmax - vmin), 0, 1))

    def _lbl(z):
        v, t = pv.get(z, np.nan), pp.get(z, 0)
        if t == 0 or pd.isna(v):
            return str(z)
        return f"{z}\n{fmt.format(v)}"

    def _tc(z):
        v, t = pv.get(z, np.nan), pp.get(z, 0)
        return "#4a5568" if (pd.isna(v) or t == 0) else "#111111"

    # ── Zones 1-9 ──────────────────────────────────────────────────
    for i in range(3):
        for j in range(3):
            z  = i * 3 + j + 1
            x0 = mx + j * cell
            y0 = my + (2 - i) * cell

            if subzones:
                sc = cell / 2
                for qi, (quad, (dx, dy)) in enumerate({
                    "TL": (0,   sc),
                    "TR": (sc,  sc),
                    "BL": (0,   0),
                    "BR": (sc,  0),
                }.items()):
                    ax.add_patch(plt.Rectangle(
                        (x0 + dx, y0 + dy), sc, sc,
                        facecolor=_fill(z),
                        edgecolor="#1e2535", linewidth=1.2,
                    ))
                    ax.text(
                        x0 + dx + sc / 2, y0 + dy + sc / 2,
                        quad, ha="center", va="center",
                        fontsize=6, color="#555", fontweight="bold",
                    )
                # zone number top-right corner
                ax.text(x0 + cell - 0.04, y0 + cell - 0.04,
                        str(z), ha="right", va="top",
                        fontsize=7, color="#4a5568")
            else:
                ax.add_patch(plt.Rectangle(
                    (x0, y0), cell, cell,
                    facecolor=_fill(z), edgecolor="#1e2535", linewidth=2.0,
                ))
                ax.text(
                    x0 + cell / 2, y0 + cell / 2, _lbl(z),
                    ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=_tc(z),
                )

    # ── Shadow zones 11-14 ─────────────────────────────────────────
    shadow_paths = [
        (11, [(0, sy), (brd, sy), (brd, top), (mx, top),
              (mx + hlf, top), (mx + hlf, 5), (0, 5), (0, sy)]),
        (12, [(rx, sy), (rx, top), (mx + hlf, top),
              (mx + hlf, 5), (5, 5), (5, sy), (rx, sy)]),
        (13, [(0, sy), (brd, sy), (brd, my), (mx, my),
              (mx + hlf, my), (mx + hlf, 0), (0, 0), (0, sy)]),
        (14, [(rx, sy), (rx, my), (mx + hlf, my),
              (mx + hlf, 0), (5, 0), (5, sy), (rx, sy)]),
    ]
    shadow_centres = {11: (brd/2, 5 - brd/2), 12: (5 - brd/2, 5 - brd/2),
                      13: (brd/2, brd/2),      14: (5 - brd/2, brd/2)}
    for z, verts in shadow_paths:
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 1)
        ax.add_patch(PathPatch(
            MPath(verts, codes),
            facecolor=_fill(z), edgecolor="#1e2535", linewidth=2.0,
        ))
        xt, yt = shadow_centres[z]
        ax.text(xt, yt, _lbl(z), ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=_tc(z))

    # Strike-zone border
    ax.add_patch(plt.Rectangle(
        (mx, my), ms, ms, fill=False,
        edgecolor="#f85149", linewidth=3.0, zorder=10,
    ))

    ax.set_xlim(0, 5); ax.set_ylim(-0.85, 5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11.5, pad=14, color="#c9d1d9", fontweight="600")

    # Colour bar
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.68, pad=0.03)
    cbar.set_label(stat_label, fontsize=9, color="#8892a4")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8892a4", fontsize=8)
    cbar.outline.set_edgecolor("#1e2535")

    n_total = int(sum(pp.values()))
    ax.text(2.5, -0.38, f"n = {n_total:,} pitches",
            ha="center", fontsize=8.5, color="#718096", style="italic")

    plt.tight_layout()
    return fig


def _empty_heatmap(msg: str = "No data") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7.2))
    fig.patch.set_facecolor("#0b0f17"); ax.set_facecolor("#0b0f17")
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            color="#8892a4", fontsize=14, transform=ax.transAxes)
    ax.axis("off")
    return fig


def draw_subzone_detail(df_zone: pd.DataFrame, zone: int) -> plt.Figure:
    """Bar chart showing whiff%, swing%, xwOBA, barrel% per sub-zone."""
    metrics = [
        ("whiff_pct",  "Whiff %",    "#ef4444"),
        ("swing_pct",  "Swing %",    "#f97316"),
        ("avg_xwoba",  "xwOBA",      "#3b82f6"),
        ("barrel_pct", "Barrel %",   "#eab308"),
    ]
    metrics = [(c, l, col) for c, l, col in metrics if c in df_zone.columns]
    if not metrics:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No sub-zone data", ha="center", va="center", color="#8892a4")
        ax.axis("off"); return fig

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.5))
    if len(metrics) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#111621")

    for ax, (sc, lbl, base_col) in zip(axes, metrics):
        ax.set_facecolor("#0b0f17")
        vals = df_zone.groupby("sub_zone")[sc].mean().reindex(SUB_LABELS).fillna(0)
        bar_colors = [base_col if v == vals.max() else "#2a3545" for v in vals]
        bars = ax.bar(SUB_LABELS, vals, color=bar_colors, edgecolor="#1e2535", width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + vals.max() * 0.03,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#e2e8f0")
        ax.set_title(lbl, fontsize=9, color="#c9d1d9", fontweight="600")
        ax.tick_params(colors="#8892a4", labelsize=7.5)
        ax.set_ylim(0, max(vals.max() * 1.3, 0.01))
        for sp in ax.spines.values(): sp.set_edgecolor("#2a3545")

    fig.suptitle(f"Zone {zone} — Sub-Zone Breakdown",
                 color="#f0f6ff", fontsize=10, fontweight="700", y=1.02)
    plt.tight_layout()
    return fig


def plot_arsenal(ars: pd.DataFrame, name: str, hand: str) -> plt.Figure:
    """Movement scatter (solid=pitcher, hollow=MLB avg) + usage bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0b0f17")

    # Left: movement scatter
    ax = axes[0]; ax.set_facecolor("#111621")
    ax.axhline(0, color="#4a5568", lw=0.8, ls="--")
    ax.axvline(0, color="#4a5568", lw=0.8, ls="--")

    for _, r in ars.iterrows():
        pt  = r["pitch_type"]
        c   = PITCH_COLORS.get(pt, "#94a3b8")
        sz  = max(80, r["usage"] * 18)
        ax.scatter(r["avg_h"], r["avg_v"], s=sz, color=c,
                   zorder=5, edgecolors="white", lw=1.2, alpha=0.9)
        ax.annotate(pt, (r["avg_h"], r["avg_v"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="white")
        av = MLB_AVG.get(pt)
        if av:
            ax.scatter(av["hbrk"], av["vbrk"], s=55,
                       color="none", edgecolors=c, lw=1.5, zorder=4, alpha=0.4)

    ax.set_xlabel("H-Break (in)  →  Arm-side", color="#a0aec0", fontsize=9)
    ax.set_ylabel("V-Break (in)  ↑  Rise",     color="#a0aec0", fontsize=9)
    ax.set_title("Movement Profile\n(solid = pitcher · hollow = MLB avg)",
                 color="#e8edf5", fontsize=9, fontweight="600")
    ax.tick_params(colors="#8892a4", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3545")

    # Right: usage bar chart
    ax2 = axes[1]; ax2.set_facecolor("#111621")
    colors = [PITCH_COLORS.get(pt, "#94a3b8") for pt in ars["pitch_type"]]
    pitch_labels = ars.get("pitch_name", ars["pitch_type"])
    bars = ax2.barh(pitch_labels, ars["usage"], color=colors,
                    edgecolor="#1e2535", height=0.6)
    for bar, val in zip(bars, ars["usage"]):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=8.5,
                 fontweight="bold", color="#e2e8f0")
    ax2.set_xlabel("Usage %", color="#a0aec0", fontsize=9)
    ax2.set_title("Pitch Usage", color="#e8edf5", fontsize=9, fontweight="600")
    ax2.invert_yaxis()
    ax2.tick_params(colors="#8892a4", labelsize=8.5)
    ax2.set_xlim(0, ars["usage"].max() + 14)
    for sp in ax2.spines.values(): sp.set_edgecolor("#2a3545")

    hand_str = "RHP" if hand == "R" else "LHP"
    fig.suptitle(f"{name}  ({hand_str})", color="#f0f6ff",
                 fontsize=11, fontweight="700", y=1.02)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 9. PITCHER ARSENAL BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_pitcher_arsenal(df: pd.DataFrame) -> tuple:
    """
    From raw statcast rows for one pitcher, compute arsenal summary.

    Returns
    -------
    ars      : DataFrame with one row per pitch type
    total    : int total pitches
    arm_avg  : float mean arm angle (degrees)
    ext_avg  : float mean extension (ft)
    hand     : "R" | "L"
    """
    if df is None or df.empty:
        return pd.DataFrame(), 0, np.nan, np.nan, "R"

    df = df.copy()
    for c in ["release_speed", "release_spin_rate", "hbrk", "vbrk",
               "release_extension", "arm_angle", "effective_speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    total = len(df)

    # Use pitch_name for display if available
    pt_col    = "pitch_type"
    name_col  = "pitch_name" if "pitch_name" in df.columns else None

    g = df.groupby(pt_col).agg(
        count     = (pt_col,           "count"),
        avg_velo  = ("release_speed",  "mean"),
        max_velo  = ("release_speed",  "max"),
        avg_spin  = ("release_spin_rate","mean"),
        avg_h     = ("hbrk",           "mean"),
        avg_v     = ("vbrk",           "mean"),
        avg_ext   = ("release_extension","mean"),
        avg_arm   = ("arm_angle",      "mean"),
        swings    = ("is_swing",       "sum"),
        whiffs    = ("is_whiff",       "sum"),
    ).reset_index()

    g["usage"] = (g["count"] / total * 100).round(1)
    g["whiff"] = (g["whiffs"] / g["swings"].replace(0, np.nan) * 100).round(1)

    for c in ["avg_velo", "max_velo", "avg_h", "avg_v", "avg_ext", "avg_arm"]:
        g[c] = g[c].round(1)
    g["avg_spin"] = g["avg_spin"].round(0)

    # Add display pitch name
    if name_col:
        pn_map = df.groupby(pt_col)[name_col].first().to_dict()
        g["pitch_name"] = g[pt_col].map(pn_map).fillna(g[pt_col].map(PITCH_LONG)).fillna(g[pt_col])
    else:
        g["pitch_name"] = g[pt_col].map(PITCH_LONG).fillna(g[pt_col])

    arm_avg = float(df["arm_angle"].mean())      if "arm_angle"         in df.columns else np.nan
    ext_avg = float(df["release_extension"].mean()) if "release_extension" in df.columns else np.nan
    hand    = df["p_throws"].mode()[0]           if "p_throws"          in df.columns and not df.empty else "R"

    return (g.sort_values("usage", ascending=False).reset_index(drop=True),
            total, arm_avg, ext_avg, hand)


# ══════════════════════════════════════════════════════════════════════
# 10. REBUILD SCORE  (fixed — ERA-FIP gap correctly weighted)
# ══════════════════════════════════════════════════════════════════════

def rebuild_score(row: pd.Series) -> tuple:
    """
    Score a pitcher's improvement potential from Baseball Reference stats.

    Key fix vs previous version:
      • ERA-FIP gap ONLY earns big points when FIP itself is also bad (≥ 4.0).
        A pitcher with ERA 5.5, FIP 3.0 is LUCKY not broken — minimal score.
      • K/9 and BB/9 are primary drivers (actual arsenal/command signals).
      • Extreme outliers (FIP > 15, IP < 5) should be filtered before calling.

    Returns (score_0_100, reasons_list, profile_key, is_luck_not_arsenal)
    """
    score   = 0.0
    reasons = []
    profiles = []

    def g(col):
        return safe_num(row.get(col, np.nan))

    era      = g("ERA");  fip = g("FIP");  so9 = g("SO9")
    bb9      = g("BB9");  era_plus = g("ERA+"); hr9 = g("HR9")
    age      = g("Age");  whip = g("WHIP")

    # ── 1. K/9  — primary arsenal signal ─────────────────────────────
    if not np.isnan(so9):
        if so9 < 6.0:
            score += 28; profiles.append("low_k")
            reasons.append(f"K/9 {so9:.1f} — very low vs avg 9.0; urgently needs out-pitch")
        elif so9 < 7.5:
            score += 18; profiles.append("low_k")
            reasons.append(f"K/9 {so9:.1f} — below average; lacks consistent miss-bat weapon")
        elif so9 < 8.5:
            score += 8;  profiles.append("low_k")
            reasons.append(f"K/9 {so9:.1f} — slightly below average")
        elif so9 > 11.0:
            score -= 6   # elite K rate = good arsenal

    # ── 2. BB/9  — command / arsenal signal ──────────────────────────
    if not np.isnan(bb9):
        if bb9 > 4.5:
            score += 20; profiles.append("high_bb")
            reasons.append(f"BB/9 {bb9:.1f} — serious command crisis (avg 3.0)")
        elif bb9 > 3.5:
            score += 10; profiles.append("high_bb")
            reasons.append(f"BB/9 {bb9:.1f} — elevated walk rate")
        elif bb9 > 3.0:
            score += 4

    # ── 3. FIP (absolute level) ───────────────────────────────────────
    if not np.isnan(fip):
        if fip > 5.0:
            score += 20; profiles.append("poor_fip")
            reasons.append(f"FIP {fip:.2f} — well above average 4.00; underlying stuff deficient")
        elif fip > 4.5:
            score += 12; profiles.append("poor_fip")
            reasons.append(f"FIP {fip:.2f} — above average; pitch quality needs improvement")
        elif fip > 4.0:
            score += 5
        elif fip < 3.5:
            score -= 8   # good FIP = strong arsenal

    # ── 4. ERA-FIP gap  — ONLY meaningful if FIP is also bad ─────────
    is_luck = False
    if not (np.isnan(era) or np.isnan(fip)):
        gap = era - fip
        if gap > 1.0 and fip >= 4.0:
            # Real performance problem — ERA and FIP both bad
            score += min(gap * 8, 14); profiles.append("era_fip_gap")
            reasons.append(f"ERA {era:.2f} > FIP {fip:.2f} (+{gap:.2f}); both metrics elevated")
        elif gap > 1.5 and fip < 4.0:
            # Good underlying stuff but bad ERA — luck/BABIP issue
            is_luck = True
            score += 3   # tiny contribution — not an arsenal problem
            reasons.append(
                f"ERA {era:.2f} >> FIP {fip:.2f}: likely BABIP/sequencing — "
                f"NOT primarily an arsenal issue"
            )
            profiles.append("era_fip_gap")

    # ── 5. ERA+  ─────────────────────────────────────────────────────
    if not np.isnan(era_plus) and era_plus > 0:
        if era_plus < 80:
            score += 8
            reasons.append(f"ERA+ {int(era_plus)} — well below league average (100)")
        elif era_plus < 90:
            score += 4
        elif era_plus > 120:
            score -= 4   # above-average = reward

    # ── 6. HR/9 ──────────────────────────────────────────────────────
    if not np.isnan(hr9):
        if hr9 > 2.0:
            score += 10; profiles.append("hr_suppression")
            reasons.append(f"HR/9 {hr9:.1f} — elevated (avg 1.3); fly-ball/HR suppression needed")
        elif hr9 > 1.6:
            score += 5;  profiles.append("hr_suppression")

    # ── 7. Age bonus ─────────────────────────────────────────────────
    if not np.isnan(age):
        if age <= 25:   score += 6    # high ceiling / trainability
        elif age >= 35: score -= 8    # lower ceiling

    # Determine primary profile
    if profiles:
        primary = Counter(profiles).most_common(1)[0][0]
    else:
        primary = "low_k"

    # If luck-not-arsenal is the ONLY signal, clearly flag it
    only_luck = (is_luck and set(profiles) == {"era_fip_gap"})

    return (
        min(max(round(score, 1), 0), 100),
        reasons[:4],
        primary,
        only_luck,
    )


# ══════════════════════════════════════════════════════════════════════
# 11. BATTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyze_batter(df: pd.DataFrame, filters: dict | None = None) -> dict:
    """
    Comprehensive batter analysis from raw statcast data.
    """
    if df is None or df.empty:
        return {}

    dff = df.copy()
    if filters:
        if filters.get("p_throws")    not in (None, "All"):
            dff = dff[dff["p_throws"]    == filters["p_throws"]]
        if filters.get("count_state") not in (None, "All"):
            dff = dff[dff["count_state"] == filters["count_state"]]
        if filters.get("pitch_type")  not in (None, "All"):
            dff = dff[dff["pitch_type"]  == filters["pitch_type"]]

    if dff.empty:
        return {}

    stand = dff["stand"].mode()[0] if "stand" in dff.columns else "R"

    # ── Zone stats ───────────────────────────────────────────────────
    zone_df = compute_zone_stats_from_raw(dff)

    # ── Sub-zone stats ───────────────────────────────────────────────
    bz = dff[dff["zone"].between(1, 9)].copy()
    sub_stats = pd.DataFrame()
    if "plate_x" in bz.columns and "plate_z" in bz.columns and not bz.empty:
        bz["sub_zone"] = bz.apply(
            lambda r: classify_subzone(
                safe_num(r["plate_x"]), safe_num(r["plate_z"]), int(r["zone"])
            ) if not pd.isna(r["zone"]) else "??",
            axis=1,
        )
        sub_stats = bz.groupby(["zone", "sub_zone"], as_index=False).agg(
            total     = ("is_swing",  "count"),
            swings    = ("is_swing",  "sum"),
            whiffs    = ("is_whiff",  "sum"),
            contacts  = ("is_contact","sum"),
            avg_xwoba = ("estimated_woba_using_speedangle","mean"),
            avg_ev    = ("launch_speed","mean"),
            avg_la    = ("launch_angle","mean"),
        )
        sw_s = sub_stats["swings"].replace(0, np.nan)
        n_s  = sub_stats["total"].replace(0, np.nan)
        sub_stats["whiff_pct"] = (sub_stats["whiffs"] / sw_s * 100).round(1)
        sub_stats["swing_pct"] = (sub_stats["swings"] / n_s  * 100).round(1)
        sub_stats["avg_xwoba"] = sub_stats["avg_xwoba"].round(3)
        sub_stats["avg_ev"]    = sub_stats["avg_ev"].round(1)

    # ── Pitch-type stats ─────────────────────────────────────────────
    pt_agg = dff.groupby("pitch_type").agg(
        total      = ("is_swing",     "count"),
        swings     = ("is_swing",     "sum"),
        whiffs     = ("is_whiff",     "sum"),
        avg_xwoba  = ("estimated_woba_using_speedangle","mean"),
        avg_ev     = ("launch_speed", "mean"),
        avg_velo   = ("release_speed",    "mean"),
        avg_spin   = ("release_spin_rate","mean"),
        avg_hbrk   = ("hbrk",            "mean"),
        avg_vbrk   = ("vbrk",            "mean"),
        avg_ext    = ("release_extension","mean"),
        avg_arm    = ("arm_angle",        "mean"),
    ).reset_index()
    sw2 = pt_agg["swings"].replace(0, np.nan)
    n2  = pt_agg["total"].replace(0, np.nan)
    pt_agg["whiff_pct"] = (pt_agg["whiffs"] / sw2 * 100).round(1)
    pt_agg["swing_pct"] = (pt_agg["swings"] / n2  * 100).round(1)
    pt_agg["usage"]     = (pt_agg["total"]  / len(dff) * 100).round(1)
    for c in ["avg_xwoba","avg_ev","avg_velo","avg_hbrk","avg_vbrk","avg_ext","avg_arm"]:
        pt_agg[c] = pt_agg[c].round(2)
    pt_agg["avg_spin"] = pt_agg["avg_spin"].round(0)
    pt_agg["pitch_name"] = pt_agg["pitch_type"].map(PITCH_LONG).fillna(pt_agg["pitch_type"])

    # ── Count stats ──────────────────────────────────────────────────
    cnt_agg = dff.groupby("count_state").agg(
        pitches   = ("is_swing",  "count"),
        swing_pct = ("is_swing",  "mean"),
        whiff_pct = ("is_whiff",  "mean"),
        avg_xwoba = ("estimated_woba_using_speedangle","mean"),
    ).reset_index()
    cnt_agg["swing_pct"] = (cnt_agg["swing_pct"] * 100).round(1)
    cnt_agg["whiff_pct"] = (cnt_agg["whiff_pct"] * 100).round(1)
    cnt_agg["avg_xwoba"] = cnt_agg["avg_xwoba"].round(3)

    # ── Platoon splits ───────────────────────────────────────────────
    platoon = {}
    for hand, grp in dff.groupby("p_throws"):
        sw3 = grp["is_swing"].sum()
        platoon[hand] = {
            "pitches":   len(grp),
            "swing_pct": round(sw3 / len(grp) * 100, 1) if len(grp) else 0,
            "whiff_pct": round(grp["is_whiff"].sum() / max(sw3, 1) * 100, 1),
            "avg_xwoba": round(grp["estimated_woba_using_speedangle"].mean(), 3)
                         if "estimated_woba_using_speedangle" in grp.columns else np.nan,
        }

    # ── Key weaknesses ───────────────────────────────────────────────
    weak_zones   = []
    avoid_zones  = []
    if not zone_df.empty and "whiff_pct" in zone_df.columns:
        valid = zone_df[zone_df["zone"].between(1, 14)].dropna(subset=["whiff_pct"])
        weak_zones  = valid.sort_values("whiff_pct", ascending=False)["zone"].head(3).tolist()
    if not zone_df.empty and "avg_xwoba" in zone_df.columns:
        valid = zone_df[zone_df["zone"].between(1, 14)].dropna(subset=["avg_xwoba"])
        avoid_zones = valid.sort_values("avg_xwoba", ascending=False)["zone"].head(3).tolist()

    return {
        "zone_stats":   zone_df,
        "sub_stats":    sub_stats,
        "pt_stats":     pt_agg,
        "count_stats":  cnt_agg,
        "platoon":      platoon,
        "stand":        stand,
        "total_pitches":len(dff),
        "weak_zones":   weak_zones,
        "avoid_zones":  avoid_zones,
    }


# ══════════════════════════════════════════════════════════════════════
# 12. PITCHING PLAN GENERATOR  (batter scout — general plan)
# ══════════════════════════════════════════════════════════════════════

def generate_pitching_plan(batter_stats: dict, batter_name: str) -> str:
    """
    Generate a rule-based pitching plan for attacking a specific batter.
    Returns an HTML string.
    """
    if not batter_stats:
        return '<div class="rpt-p">No data available.</div>'

    stand       = batter_stats.get("stand", "R")
    weak_zones  = batter_stats.get("weak_zones",  [])
    avoid_zones = batter_stats.get("avoid_zones", [])
    pt          = batter_stats.get("pt_stats",  pd.DataFrame())
    zs          = batter_stats.get("zone_stats",pd.DataFrame())
    platoon     = batter_stats.get("platoon",   {})

    opp  = "LHP" if stand == "R" else "RHP"
    same = "RHP" if stand == "R" else "LHP"

    ZONE_DESC = {
        1:"top-inside",  2:"top-middle",  3:"top-outside",
        4:"mid-inside",  5:"center",      6:"mid-outside",
        7:"bot-inside",  8:"bot-middle",  9:"bot-outside",
        11:"shadow-top", 12:"shadow-right",
        13:"shadow-bottom", 14:"shadow-left",
    }

    html = [f'<div class="report-wrap">']
    html.append(
        f'<div style="font-size:1.1rem;font-weight:800;color:#f0f6ff;margin-bottom:12px;">'
        f'🎯 Pitching Plan — How to Attack {batter_name}</div>'
    )

    # 1. Profile
    html.append('<div class="rpt-h2">1. BATTER PROFILE & PLATOON</div>')
    html.append(
        f'<div class="rpt-p"><b>Handedness:</b> '
        f'{"Right-handed (RHB)" if stand=="R" else "Left-handed (LHB)"}.</div>'
    )

    # Platoon advantage
    if platoon:
        rh = platoon.get("R", {}); lh = platoon.get("L", {})
        if rh and lh:
            rh_whiff = rh.get("whiff_pct", 0); lh_whiff = lh.get("whiff_pct", 0)
            better_hand = "R" if rh_whiff > lh_whiff else "L"
            better_label = "RHP" if better_hand == "R" else "LHP"
            html.append(
                f'<div class="insight-box"><div class="rpt-p">'
                f'<b>Platoon split:</b> {rh_whiff:.1f}% whiff vs RHP, '
                f'{lh_whiff:.1f}% whiff vs LHP. '
                f'{better_label} has the platoon advantage with more misses.</div></div>'
            )

    # 2. Zone attack
    html.append('<div class="rpt-h2">2. ZONE ATTACK STRATEGY</div>')
    if weak_zones:
        z_str = ", ".join(f"Zone {z} ({ZONE_DESC.get(z,'?')})" for z in weak_zones)
        html.append(f'<div class="rpt-p"><b>Highest whiff zones (attack here):</b> {z_str}</div>')
        html.append(
            '<div class="ok-box"><div class="rpt-p">🎯 Deploy your best out-pitch '
            'in these zones in 2-strike counts. Establish them early in the at-bat '
            'with a fastball so the batter expects location before you expand.</div></div>'
        )
    if avoid_zones:
        a_str = ", ".join(f"Zone {z}" for z in avoid_zones)
        html.append(
            f'<div class="rpt-p"><b>High damage zones (avoid as primary target):</b> {a_str}</div>'
        )
        html.append(
            '<div class="danger-box"><div class="rpt-p">⚠️ Do NOT elevate into these zones '
            'unless ahead in the count (0-2, 1-2) and using a borderline pitch.</div></div>'
        )

    # 3. Pitch selection
    html.append('<div class="rpt-h2">3. BEST PITCHES TO USE</div>')
    if not pt.empty and "whiff_pct" in pt.columns:
        pt_sorted = (pt[pt["total"] >= 10]
                     .dropna(subset=["whiff_pct"])
                     .sort_values("whiff_pct", ascending=False))
        if not pt_sorted.empty:
            best = pt_sorted.iloc[0]
            bc   = PITCH_COLORS.get(best["pitch_type"], "#94a3b8")
            bn   = best.get("pitch_name", PITCH_LONG.get(best["pitch_type"], best["pitch_type"]))
            html.append(
                f'<div class="rpt-p"><b>Best out-pitch against this batter:</b> '
                f'<span style="color:{bc};font-weight:700;">{bn}</span> — '
                f'{best["whiff_pct"]:.1f}% whiff '
                f'({int(best["total"])} pitches, '
                f'xwOBA {best["avg_xwoba"]:.3f}). '
                f'Avg velo faced: <b>{best["avg_velo"]:.1f} mph</b>, '
                f'spin: <b>{int(best["avg_spin"])} rpm</b>.</div>'
            )
            html.append(
                f'<div class="ok-box"><div class="rpt-p">Increase usage of {bn} in '
                f'2-strike counts (0-2, 1-2, 2-2). Tunnel with primary fastball '
                f'to maximise late-break surprise.</div></div>'
            )
            if len(pt_sorted) > 1:
                worst = pt_sorted.iloc[-1]
                wn    = worst.get("pitch_name", PITCH_LONG.get(worst["pitch_type"], worst["pitch_type"]))
                html.append(
                    f'<div class="rpt-p"><b>Pitch to avoid:</b> '
                    f'{wn} — only {worst["whiff_pct"]:.1f}% whiff, '
                    f'xwOBA {worst["avg_xwoba"]:.3f}. Use sparingly.</div>'
                )

    # 4. Count-specific approach
    html.append('<div class="rpt-h2">4. COUNT-SPECIFIC APPROACH</div>')
    cnt = batter_stats.get("count_stats", pd.DataFrame())
    if not cnt.empty:
        cnt_map = cnt.set_index("count_state")
        for count, advice in [
            ("0-0",  "Attack with best fastball/secondary for a called strike. "
                     "First-pitch strikes dramatically improve at-bat outcomes."),
            ("0-2",  "Expand to shadow zones 13/14 with highest-whiff pitch. "
                     "Never groove a hittable fastball."),
            ("3-2",  "Best fastball in zone — must compete, no room for mistake."),
            ("2-2",  "Perfect setup count — elevated fastball then breaking ball to low shadow."),
        ]:
            row = cnt_map.get(count) if hasattr(cnt_map, "get") else cnt_map.loc[count] if count in cnt_map.index else None
            extra = ""
            if row is not None:
                sw_pct  = safe_num(row.get("swing_pct", row["swing_pct"]) if isinstance(row, dict) else row["swing_pct"])
                wh_pct  = safe_num(row.get("whiff_pct", row["whiff_pct"]) if isinstance(row, dict) else row["whiff_pct"])
                extra   = f" [Batter swings {sw_pct:.0f}%, whiffs {wh_pct:.0f}%]"
            html.append(f'<div class="rpt-li">• <b>{count}:</b> {advice}{extra}</div>')

    html.append('</div>')
    return "\n".join(html)


# ══════════════════════════════════════════════════════════════════════
# 13. PITCHER SCOUT REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════

def generate_pitcher_report(
    name: str,
    ars: pd.DataFrame,
    total: int,
    arm_avg: float,
    ext_avg: float,
    hand: str,
    season_lbl: str,
    age: float | None = None,
) -> str:
    """
    Full HTML scouting report for a pitcher.
    All f-strings use safe concatenation to avoid NameError on 'sign'.
    """
    sn, sc = arm_info(arm_avg)

    # sign = +1 for RHP (positive pfx_x = arm-side), -1 for LHP
    sign = 1 if hand == "R" else -1

    age_badge = ""
    if age and not pd.isna(age):
        age_badge = f'<span class="ref-badge">Age {int(age)}</span>'

    html = ['<div class="report-wrap">']

    # Header strip
    arm_badge = (
        f'<span class="ref-badge" style="border-color:{sc};color:{sc};">'
        f'🎯 {arm_avg:.1f}° — {sn}</span>'
        if not np.isnan(arm_avg) else ""
    )
    ext_badge = (
        f'<span class="ref-badge">📏 Ext {ext_avg:.2f} ft</span>'
        if not np.isnan(ext_avg) else ""
    )
    html.append(
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'flex-wrap:wrap;margin-bottom:14px;">'
        f'<div style="font-size:1.3rem;font-weight:800;color:#f0f6ff;">{name}</div>'
        f'<span class="ref-badge">{"RHP" if hand=="R" else "LHP"}</span>'
        f'<span class="ref-badge">{season_lbl}</span>'
        f'<span class="ref-badge">{total:,} pitches</span>'
        f'{age_badge}{arm_badge}{ext_badge}'
        f'</div>'
    )

    # Section 1: Arsenal breakdown
    html.append('<div class="rpt-h2">1. ARSENAL BREAKDOWN vs. MLB AVERAGES</div>')
    pts = ars["pitch_type"].tolist()

    for _, r in ars.iterrows():
        pt   = r["pitch_type"]
        av   = MLB_AVG.get(pt)
        nm   = r.get("pitch_name", PITCH_LONG.get(pt, pt))
        col  = PITCH_COLORS.get(pt, "#94a3b8")

        html.append(
            f'<div style="margin:10px 0;padding:11px 14px;background:#151c2e;'
            f'border-left:3px solid {col};border-radius:0 8px 8px 0;">'
        )
        html.append(
            f'<div style="color:{col};font-weight:700;font-size:.88rem;margin-bottom:5px;">'
            f'{nm} '
            f'<span style="color:#8892a4;font-size:.77rem;font-weight:400;">'
            f'{r["usage"]:.1f}% usage · {int(r["count"]):,} pitches</span></div>'
        )

        if av:
            # Velocity
            vr  = rate(r["avg_velo"], av["velo"], True, (2, 4))
            html.append(
                f'<div class="rpt-p"><b>Velocity:</b> {pill(RATE_LABEL[vr], RATE_CSS[vr])} '
                f'{r["avg_velo"]:.1f} mph (avg {av["velo"]:.1f}) · '
                f'max {r["max_velo"]:.1f}</div>'
            )
            # Spin
            if not pd.isna(r["avg_spin"]):
                sr = rate(r["avg_spin"], av["spin"], True, (150, 300))
                html.append(
                    f'<div class="rpt-p"><b>Spin rate:</b> {pill(RATE_LABEL[sr], RATE_CSS[sr])} '
                    f'{int(r["avg_spin"]):,} rpm (avg {av["spin"]:,})</div>'
                )
            # H-Break
            hd = r["avg_h"] - av["hbrk"]
            # "more arm-side" means hd and sign have same polarity
            hd_dir = "more" if (hd * sign) > 0 else "less"
            html.append(
                f'<div class="rpt-p"><b>H-Break:</b> '
                f'{r["avg_h"]:+.1f}" (avg {av["hbrk"]:+.1f}") — '
                f'{abs(hd):.1f}" {hd_dir} arm-side movement than MLB avg</div>'
            )
            # V-Break
            vd     = r["avg_v"] - av["vbrk"]
            vd_dir = "more" if vd > 0 else "less"
            html.append(
                f'<div class="rpt-p"><b>V-Break:</b> '
                f'{r["avg_v"]:+.1f}" (avg {av["vbrk"]:+.1f}") — '
                f'{abs(vd):.1f}" {vd_dir} vertical break</div>'
            )
            # Whiff
            if not pd.isna(r["whiff"]):
                wr       = rate(r["whiff"], av["whiff"], True, (5, 10))
                under    = ""
                if r["whiff"] > av["whiff"] + 5 and r["usage"] < 15:
                    under = " — ⚡ UNDERUSED for its whiff rate!"
                html.append(
                    f'<div class="rpt-p"><b>Whiff rate:</b> {pill(RATE_LABEL[wr], RATE_CSS[wr])} '
                    f'{r["whiff"]:.1f}% (avg {av["whiff"]:.1f}%){under}</div>'
                )
            # Extension (per-pitch avg stored in arsenal)
            if not pd.isna(r.get("avg_ext", np.nan)):
                er = rate(r["avg_ext"], av["ext"], True, (0.2, 0.5))
                html.append(
                    f'<div class="rpt-p"><b>Extension:</b> '
                    f'{pill(RATE_LABEL[er], RATE_CSS[er])} '
                    f'{r["avg_ext"]:.2f} ft (avg {av["ext"]:.2f} ft)</div>'
                )
        else:
            html.append('<div class="rpt-p"><i>No MLB avg benchmark for this pitch type.</i></div>')

        html.append('</div>')   # end pitch block

    # Arsenal completeness
    has_off = any(p in ["CH","FS","SV"] for p in pts)
    has_brk = any(p in ["SL","ST","CU","KC"] for p in pts)
    if has_off and has_brk:
        html.append('<div class="ok-box"><div class="rpt-p">✅ Complete three-category arsenal (fastball + offspeed + breaking). Focus on sequencing quality and movement optimisation.</div></div>')
    if not has_off:
        html.append('<div class="danger-box"><div class="rpt-p">🔴 No offspeed pitch (CH/FS/SV) — critical vulnerability vs. opposite-handed batters. Adding a changeup is the highest-priority arsenal addition.</div></div>')
    if not has_brk:
        html.append('<div class="danger-box"><div class="rpt-p">🔴 No breaking ball — 2-strike arsenal severely limited. Add slider or curveball immediately.</div></div>')

    # Section 2: Arm slot & extension
    html.append('<div class="rpt-h2">2. ARM SLOT & EXTENSION</div>')
    if not np.isnan(arm_avg):
        html.append(f'<div class="rpt-p">Arm slot: <b style="color:{sc};">{arm_avg:.1f}° — {sn}</b></div>')
        if   arm_avg >= 60:
            html.append('<div class="insight-box"><div class="rpt-p"><b>High slot:</b> strong vertical plane, fastball rides, breaking balls break sharply top-to-bottom. Ideal for curveball/KC but sweepers sacrifice horizontal movement for plane.</div></div>')
        elif arm_avg >= 45:
            html.append('<div class="insight-box"><div class="rpt-p"><b>Three-quarter (MLB sweet spot):</b> balanced H/V movement. Natural fastball-changeup tunnel. All pitch types effective.</div></div>')
        elif arm_avg >= 25:
            html.append('<div class="insight-box"><div class="rpt-p"><b>Low three-quarter:</b> increased horizontal movement, large platoon advantage for same-hand batters. Consider cutter/sweeper to amplify the angle.</div></div>')
        else:
            html.append('<div class="insight-box"><div class="rpt-p"><b>Sidearm/submarine:</b> largest platoon advantage in baseball. Minimise opposite-hand matchups or use velocity variation to neutralise.</div></div>')

    if not np.isnan(ext_avg):
        er = rate(ext_avg, 6.2, True, (0.2, 0.5))
        ext_note = (
            'Elite — every pitch plays ~1 mph faster than radar reading.'
            if ext_avg >= 6.5 else
            'Below average — stride/release-point work could add 0.3–0.5 ft ≈ +1 mph perceived velocity. Highest-leverage free improvement.'
            if ext_avg < 6.0 else
            'Solid extension.'
        )
        html.append(
            f'<div class="rpt-p"><b>Extension:</b> {pill(RATE_LABEL[er], RATE_CSS[er])} '
            f'{ext_avg:.2f} ft (avg ~6.2 ft). {ext_note}</div>'
        )

    # Section 3: Pitch mix
    html.append('<div class="rpt-h2">3. PITCH MIX & SEQUENCING</div>')
    ranked = ars.dropna(subset=["whiff"]).sort_values("whiff", ascending=False)
    if not ranked.empty:
        best   = ranked.iloc[0]
        bc     = PITCH_COLORS.get(best["pitch_type"], "#94a3b8")
        bn     = best.get("pitch_name", PITCH_LONG.get(best["pitch_type"], best["pitch_type"]))
        usage  = best["usage"]
        usage_note = (
            "Underused — increase to 20–25% in 2-strike counts."
            if usage < 15 else
            "Heavy usage — monitor batter adjustment rates vs. this pitch."
            if usage > 45 else
            "Prioritise in 2-strike counts and against same-handed batters."
        )
        html.append(
            f'<div class="rpt-p"><b>Best out-pitch:</b> '
            f'<span style="color:{bc};font-weight:700;">{bn}</span> '
            f'({best["whiff"]:.1f}% whiff · {usage:.1f}% usage). {usage_note}</div>'
        )

    fb_usage = ars[ars["pitch_type"].isin(["FF","SI","FC"])]["usage"].sum()
    fb_note  = (
        "Heavy fastball reliance — batters can sit on the heater. Mix secondaries in 0-0 and 1-0 counts."
        if fb_usage > 65 else
        "Low fastball usage — ensure zone-attack foundation is not compromised."
        if fb_usage < 35 else
        "Healthy fastball/secondary balance."
    )
    html.append(f'<div class="rpt-p"><b>Fastball usage:</b> {fb_usage:.1f}%. {fb_note}</div>')

    # Tunneling pairs
    tunnels = []
    for i, r1 in ars.iterrows():
        for j, r2 in ars.iterrows():
            if j <= i: continue
            hd = abs(r1["avg_h"] - r2["avg_h"])
            vd = abs(r1["avg_v"] - r2["avg_v"])
            c1 = PITCH_COLORS.get(r1["pitch_type"], "#94a3b8")
            c2 = PITCH_COLORS.get(r2["pitch_type"], "#94a3b8")
            n1 = r1.get("pitch_name", r1["pitch_type"])
            n2 = r2.get("pitch_name", r2["pitch_type"])
            if hd < 4 and vd > 8:
                tunnels.append(
                    f'<span style="color:{c1};font-weight:700;">{n1}</span> + '
                    f'<span style="color:{c2};font-weight:700;">{n2}</span>: '
                    f'H-plane within {hd:.1f}", V diverges {vd:.1f}" late — '
                    f'<b>elite tunnel pair</b>'
                )
            elif hd > 10 and vd < 5:
                tunnels.append(
                    f'<span style="color:{c1};font-weight:700;">{n1}</span> + '
                    f'<span style="color:{c2};font-weight:700;">{n2}</span>: '
                    f'Horizontal attack pair ({hd:.1f}" H-break difference)'
                )
    if tunnels:
        html.append('<div class="rpt-h3">Best tunneling pairs:</div>')
        for t in tunnels[:3]:
            html.append(f'<div class="insight-box"><div class="rpt-p">• {t}</div></div>')
    else:
        html.append('<div class="warn-box"><div class="rpt-p">No strong tunneling pairs found. Consider adding a pitch that shares the fastball release trajectory for late-break deception.</div></div>')

    # Section 4: Priority actions
    html.append('<div class="rpt-h2">4. PRIORITY ACTION ITEMS</div>')
    actions = []
    for _, r in ars.iterrows():
        av = MLB_AVG.get(r["pitch_type"])
        if not av: continue
        if not pd.isna(r["whiff"]) and r["whiff"] < av["whiff"] - 8 and r["usage"] > 15:
            nm_a = r.get("pitch_name", r["pitch_type"])
            actions.append(("🔴","Long-term",
                f'{nm_a} whiff {r["whiff"]:.1f}% vs avg {av["whiff"]:.1f}% — '
                f'review grip and movement profile; restrict use to favourable counts'))
        hd = abs(r["avg_h"] - av["hbrk"])
        if hd > 5 and r["usage"] > 10:
            nm_a = r.get("pitch_name", r["pitch_type"])
            pot  = round(min(hd * 0.6, 5), 1)
            actions.append(("🟡","Medium-term",
                f'{nm_a}: {hd:.1f}" below-average movement — '
                f'grip/seam adjustment could add {pot}" break'))
    if not np.isnan(ext_avg) and ext_avg < 6.0:
        actions.append(("🟢","Easy win",
            f'Extension {ext_avg:.2f} ft — stride/release work = '
            f'+{round(6.2-ext_avg,2):.2f} ft potential (+1 mph perceived)'))
    if not has_off:
        actions.append(("🔴","Long-term","Add changeup — highest-priority arsenal gap"))
    if not actions:
        actions.append(("🟢","Maintain",
            "Competitive arsenal — focus on count-based execution and sequence variety"))

    for em, tl, txt in actions:
        tc  = "#4ade80" if em == "🟢" else "#fbbf24" if em == "🟡" else "#f87171"
        box = "ok-box" if em == "🟢" else "warn-box" if em == "🟡" else "danger-box"
        html.append(
            f'<div class="{box}" style="margin:6px 0;"><div class="rpt-p">'
            f'{em} <span style="color:{tc};font-size:.76rem;font-weight:600;">[{tl}]</span>'
            f' <b>{txt}</b></div></div>'
        )

    html.append('</div>')
    return "\n".join(html)


# Constants used in the report (defined here so Part 2 is self-contained)
RATE_LABEL = {
    "elite": "🟢 Elite",  "above": "🔵 Above Avg",
    "avg":   "⚪ Average", "below": "🟡 Below Avg",
    "poor":  "🔴 Needs Work",
}
RATE_CSS = {
    "elite": "pill-elite", "above": "pill-above",
    "avg":   "pill-avg",   "below": "pill-below",
    "poor":  "pill-poor",
}


# ══════════════════════════════════════════════════════════════════════
# 14. MATCHUP PLAN  (pitcher vs specific batter — Tab 2 section)
# ══════════════════════════════════════════════════════════════════════

def generate_matchup_plan(
    pitcher_ars: pd.DataFrame,
    batter_stats: dict,
    pitcher_name: str,
    batter_name: str,
    pitcher_hand: str,
) -> str:
    """
    Generate a specific pitcher-vs-batter matchup plan.
    Combines pitcher's best pitches with batter's zone weaknesses.
    """
    if pitcher_ars.empty or not batter_stats:
        return '<div class="rpt-p">Insufficient data for matchup analysis.</div>'

    stand      = batter_stats.get("stand", "R")
    weak_zones = batter_stats.get("weak_zones", [])
    pt         = batter_stats.get("pt_stats",   pd.DataFrame())

    # Pitcher's best pitches by whiff rate
    p_ranked = (pitcher_ars.dropna(subset=["whiff"])
                .sort_values("whiff", ascending=False))

    ZONE_DESC = {
        1:"top-inside", 2:"top-mid", 3:"top-out",
        4:"mid-in",     5:"center",  6:"mid-out",
        7:"bot-in",     8:"bot-mid", 9:"bot-out",
        11:"shadow-T", 12:"shadow-R", 13:"shadow-B", 14:"shadow-L",
    }

    html = [
        f'<div class="ref-card" style="margin-top:8px;">'
        f'<div class="ref-title">⚔️ {pitcher_name} vs {batter_name}</div>'
        f'<div class="rpt-p"><b>Matchup:</b> '
        f'{"RHP" if pitcher_hand=="R" else "LHP"} vs '
        f'{"RHB" if stand=="R" else "LHB"}. '
    ]

    # Platoon note
    same_hand = (pitcher_hand == "R" and stand == "R") or (pitcher_hand == "L" and stand == "L")
    html.append(
        '<b style="color:#4ade80;">Pitcher platoon advantage</b> (same hand).'
        if same_hand else
        '<b style="color:#fbbf24;">Batter platoon advantage</b> (opposite hand) — increase offspeed usage.'
    )
    html.append('</div>')

    # Primary recommendation
    if not p_ranked.empty:
        best_p = p_ranked.iloc[0]
        bp_col = PITCH_COLORS.get(best_p["pitch_type"], "#94a3b8")
        bp_nm  = best_p.get("pitch_name", PITCH_LONG.get(best_p["pitch_type"], best_p["pitch_type"]))

        # Does batter struggle vs this pitch type?
        batter_vs = pd.DataFrame()
        if not pt.empty and "pitch_type" in pt.columns:
            batter_vs = pt[pt["pitch_type"] == best_p["pitch_type"]]

        batter_whiff_vs = ""
        if not batter_vs.empty and "whiff_pct" in batter_vs.columns:
            bw = safe_num(batter_vs.iloc[0].get("whiff_pct"))
            if not np.isnan(bw):
                batter_whiff_vs = f" (batter whiffs {bw:.1f}% vs this pitch type)"

        html.append(
            f'<div class="rpt-p" style="margin-top:6px;"><b>Primary weapon:</b> '
            f'<span style="color:{bp_col};font-weight:700;">{bp_nm}</span> — '
            f'{best_p["whiff"]:.1f}% whiff for pitcher{batter_whiff_vs}.</div>'
        )

    # Zone attack
    if weak_zones:
        z_str = ", ".join(f"Z{z}({ZONE_DESC.get(z,'?')})" for z in weak_zones[:2])
        html.append(
            f'<div class="rpt-p"><b>Target zones:</b> {z_str} — '
            f'batter\'s highest whiff areas.</div>'
        )

    # Count plan
    html.append('<div class="rpt-p"><b>Attack sequence:</b>')
    html.append('0-0 → fastball/best secondary for called strike; ')
    if not p_ranked.empty:
        out_nm = p_ranked.iloc[0].get("pitch_name", p_ranked.iloc[0]["pitch_type"])
        html.append(f'0-2/1-2 → {out_nm} below zone (shadow 13/14); ')
    html.append('3-2 → best fastball in zone.</div>')

    html.append('</div>')  # end ref-card
    return "\n".join(html)



# ═══════════════════════════════════════════════════════════════════
# AI.py  —  MLB Statcast Pro Dashboard  (Part 3 of 3)
# Lines 1601-end: sidebar, Tab 1-4 UI, main()
# Paste BELOW Part 1 + Part 2 in the final AI.py
# ═══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# 15. STARTUP — load data shared across all tabs
# ══════════════════════════════════════════════════════════════════════

ALL_YEARS = (2024, 2025, 2026)

# Tylko mały plik BR — szybkie (<1s)
with st.spinner("📋  Loading pitching stats…"):
    _pitching_df = load_pitching_stats(ALL_YEARS)

# Meta mapy — lazy, tylko jeśli jeszcze nie załadowane
if "meta_loaded" not in st.session_state:
    with st.spinner("🔍  Building player index (first load ~15s)…"):
        _batter_meta, _pitcher_meta = build_meta_maps(ALL_YEARS)
        st.session_state["_batter_meta"]   = _batter_meta
        st.session_state["_pitcher_meta"]  = _pitcher_meta
        st.session_state["meta_loaded"]    = True
else:
    _batter_meta  = st.session_state["_batter_meta"]
    _pitcher_meta = st.session_state["_pitcher_meta"]

_pitcher_disp_list, _pitcher_disp_map = build_pitcher_selectbox(_pitcher_meta)
_batter_disp_list,  _batter_disp_map  = build_batter_selectbox(_batter_meta)

_all_teams = sorted({
    m.get("team", "?")
    for m in _batter_meta.values()
    if m.get("team") and m.get("team") != "?"
})
_avail_seasons = (
    sorted(_pitching_df["season"].dropna().unique().tolist())
    if not _pitching_df.empty and "season" in _pitching_df.columns
    else list(ALL_YEARS)
)

# ══════════════════════════════════════════════════════════════════════
# 16. SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:12px 0 8px;'
        'border-bottom:1px solid #1e2535;margin-bottom:14px;">'
        '<span style="font-size:2rem;display:block;margin-bottom:4px;">⚾</span>'
        '<div style="color:#79b8ff;font-weight:700;font-size:.95rem;">MLB Statcast Pro</div>'
        '<div style="color:#718096;font-size:.7rem;margin-top:2px;">Advanced Pitch Analytics</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Arm slot reference
    st.markdown(
        '<div style="color:#79b8ff;font-weight:600;font-size:.86rem;margin-bottom:5px;">💪 Arm Slot Guide</div>',
        unsafe_allow_html=True,
    )
    rows_html = ""
    for lo, hi, nm, c in ARM_SLOTS:
        w = max(10, int((hi - lo) * 1.1))
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;'
            f'border-bottom:1px solid #1e253570;">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:.71rem;'
            f'color:#63b3ff;min-width:58px;">{lo}–{hi}°</div>'
            f'<div style="font-size:.75rem;color:#c4cdd8;flex:1;">{nm}</div>'
            f'<div style="width:{w}px;height:5px;border-radius:3px;background:{c};'
            f'flex-shrink:0;"></div></div>'
        )
    st.markdown(f'<div class="ref-card">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🔄  Clear All Cache", width="stretch"):
        st.cache_data.clear()
        st.success("✅  Cache cleared — reload the page.")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# 17. PAGE HEADER
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="dash-hdr">'
    '<div class="dash-ttl">⚾ MLB Statcast Pro Dashboard</div>'
    '<div class="dash-sub">'
    'Pitch Analytics 2024–2026 · Statcast + Baseball Reference · '
    'Zone heatmaps · Arsenal analysis · Batter scouting'
    '</div></div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# 18. TABS
# ══════════════════════════════════════════════════════════════════════

tab_main, tab_pitcher, tab_rebuild, tab_batter = st.tabs([
    "📊  Dashboard",
    "🤖  Pitcher Scout",
    "🏆  Top 10 Rebuild",
    "🎯  Batter Scout",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — MAIN DASHBOARD  (complete replacement)
# Paste this block to replace your entire "with tab_main:" section.
#
# Changes vs original:
#  • Velocity, spin rate, H-break, V-break FILTERS
#  • Sub-zone 2×2 panel shown INLINE below heatmap (no popups)
#  • Zone summary table includes avg spin, H-break, V-break
#  • Comparison column gets the same sub-zone treatment
# ═══════════════════════════════════════════════════════════════════

# ── Helper: load raw columns for sub-zone + movement aggregation ─────
# Add near precompute_zone_stats in your helpers section.

@st.cache_data(ttl=3600, show_spinner=False)
def load_raw_for_subzones(
    years: tuple,
    pitch_type: str,
    p_throws: str,
    stand: str,
    count_state: str,
    velo_min: float,
    velo_max: float,
    spin_min: float,
    spin_max: float,
    hbrk_min: float,
    hbrk_max: float,
    vbrk_min: float,
    vbrk_max: float,
) -> pd.DataFrame:
    """
    Load only the columns needed for sub-zone analysis and movement stats.
    Uses pyarrow predicate pushdown per year — fast (~1-2 s total, cached).
    """
    RAW_COLS = [
        "pitch_type","zone","stand","p_throws","balls","strikes",
        "plate_x","plate_z","description",
        "release_speed","release_spin_rate","pfx_x","pfx_z",
        "launch_speed","launch_angle","estimated_woba_using_speedangle",
    ]
    parts = []
    for yr in years:
        path = STATCAST_FILES.get(yr)
        if path is None or not path.exists():
            continue
        if PYARROW_OK:
            sc  = set(pq.read_schema(str(path)).names)
            use = [c for c in RAW_COLS if c in sc]
            # Predicate pushdown on pitch_type and p_throws if specific
            filters = []
            if pitch_type != "All":  filters.append(("pitch_type","=",pitch_type))
            if p_throws   != "All":  filters.append(("p_throws","=",p_throws))
            if stand      != "All":  filters.append(("stand","=",stand))
            table = pq.read_table(str(path), columns=use, filters=filters or None)
            df = table.to_pandas()
        else:
            df = pd.read_parquet(path, engine="pyarrow", columns=RAW_COLS)
            if pitch_type != "All": df = df[df["pitch_type"] == pitch_type]
            if p_throws   != "All": df = df[df["p_throws"]   == p_throws]
            if stand      != "All": df = df[df["stand"]       == stand]
        df["game_year"] = yr
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    full = pd.concat(parts, ignore_index=True)
    full = full[full["zone"].between(1, 14)]

    # Count filter
    if count_state != "All":
        full["count_state"] = full["balls"].astype(str) + "-" + full["strikes"].astype(str)
        full = full[full["count_state"] == count_state]

    # Movement columns (inches)
    full["hbrk"] = pd.to_numeric(full["pfx_x"], errors="coerce") * 12
    full["vbrk"] = pd.to_numeric(full["pfx_z"], errors="coerce") * 12

    # Numeric casts
    for c in ["release_speed","release_spin_rate","launch_speed","launch_angle",
              "estimated_woba_using_speedangle"]:
        full[c] = pd.to_numeric(full[c], errors="coerce")

    # Velocity filter
    full = full[full["release_speed"].between(velo_min, velo_max).fillna(True)]
    # Spin filter
    full = full[full["release_spin_rate"].between(spin_min, spin_max).fillna(True)]
    # H-break filter
    full = full[full["hbrk"].between(hbrk_min, hbrk_max).fillna(True)]
    # V-break filter
    full = full[full["vbrk"].between(vbrk_min, vbrk_max).fillna(True)]

    # Event flags
    desc = full["description"].fillna("")
    full["is_swing"]   = desc.isin(SWING_EV).astype("int8")
    full["is_whiff"]   = desc.isin(WHIFF_EV).astype("int8")
    full["is_contact"] = desc.isin(CONTACT_EV).astype("int8")
    ls = full["launch_speed"]; la = full["launch_angle"]
    full["is_barrel"]  = ((ls >= 98) & la.between(26,30)).fillna(False).astype("int8")
    full["is_hh"]      = (ls >= 95).fillna(False).astype("int8")
    full["is_gb"]      = (la < 10).fillna(False).astype("int8")



    return full


def compute_zone_stats_with_movement(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Oblicza statystyki zone z surowych danych (z ruchem)."""
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    
    # <<< NAJWAŻNIEJSZE: dodaj flagi >>>
    df = _add_flags(df)

    df = df[df["zone"].between(1, 14)].copy()

    if df.empty:
        return pd.DataFrame()

    grp = df.groupby("zone", as_index=False).agg(
        total       = ("is_swing",   "count"),
        swings      = ("is_swing",   "sum"),
        whiffs      = ("is_whiff",   "sum"),
        contacts    = ("is_contact", "sum"),
        barrels     = ("is_barrel",  "sum"),
        hard_hits   = ("is_hh",      "sum"),
        gbs         = ("is_gb",      "sum"),
        batted      = ("launch_speed","count"),
        avg_ev      = ("launch_speed", "mean"),
        avg_la      = ("launch_angle", "mean"),
        avg_xwoba   = ("estimated_woba_using_speedangle", "mean"),
        avg_velo    = ("release_speed", "mean"),
        avg_spin    = ("release_spin_rate", "mean"),
        avg_hbrk    = ("hbrk", "mean"),
        avg_vbrk    = ("vbrk", "mean"),
    )

    # Oblicz procenty
    n  = grp["total"].replace(0, np.nan)
    sw = grp["swings"].replace(0, np.nan)
    bt = grp["batted"].replace(0, np.nan)

    grp["swing_pct"]    = (grp["swings"]   / n  * 100).round(1)
    grp["whiff_pct"]    = (grp["whiffs"]   / sw * 100).round(1)
    grp["contact_pct"]  = (grp["contacts"] / sw * 100).round(1)
    grp["barrel_pct"]   = (grp["barrels"]  / bt * 100).round(1)
    grp["hard_hit_pct"] = (grp["hard_hits"]/ bt * 100).round(1)
    grp["gb_pct"]       = (grp["gbs"]      / bt * 100).round(1)

    for col in ["avg_ev", "avg_la", "avg_xwoba", "avg_velo", "avg_spin", "avg_hbrk", "avg_vbrk"]:
        if col in grp.columns:
            grp[col] = grp[col].round(2 if "xwoba" in col else 1)

    return grp


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — MAIN DASHBOARD  (complete replacement)
# Paste this block to replace your entire "with tab_main:" section.
#
# Changes vs original:
#  • Velocity, spin rate, H-break, V-break FILTERS
#  • Sub-zone 2×2 panel shown INLINE below heatmap (no popups)
#  • Zone summary table includes avg spin, H-break, V-break
#  • Comparison column gets the same sub-zone treatment
# ═══════════════════════════════════════════════════════════════════

# ── Helper: load raw columns for sub-zone + movement aggregation ─────
# Add near precompute_zone_stats in your helpers section.

@st.cache_data(ttl=3600, show_spinner=False)
def load_raw_for_subzones(
    years: tuple,
    pitch_type: str,
    p_throws: str,
    stand: str,
    count_state: str,
    velo_min: float,
    velo_max: float,
    spin_min: float,
    spin_max: float,
    hbrk_min: float,
    hbrk_max: float,
    vbrk_min: float,
    vbrk_max: float,
) -> pd.DataFrame:
    """
    Load only the columns needed for sub-zone analysis and movement stats.
    Uses pyarrow predicate pushdown per year — fast (~1-2 s total, cached).
    """
    RAW_COLS = [
        "pitch_type","zone","stand","p_throws","balls","strikes",
        "plate_x","plate_z","description",
        "release_speed","release_spin_rate","pfx_x","pfx_z",
        "launch_speed","launch_angle","estimated_woba_using_speedangle",
    ]
    parts = []
    for yr in years:
        path = STATCAST_FILES.get(yr)
        if path is None or not path.exists():
            continue
        if PYARROW_OK:
            sc  = set(pq.read_schema(str(path)).names)
            use = [c for c in RAW_COLS if c in sc]
            # Predicate pushdown on pitch_type and p_throws if specific
            filters = []
            if pitch_type != "All":  filters.append(("pitch_type","=",pitch_type))
            if p_throws   != "All":  filters.append(("p_throws","=",p_throws))
            if stand      != "All":  filters.append(("stand","=",stand))
            table = pq.read_table(str(path), columns=use, filters=filters or None)
            df = table.to_pandas()
        else:
            df = pd.read_parquet(path, engine="pyarrow", columns=RAW_COLS)
            if pitch_type != "All": df = df[df["pitch_type"] == pitch_type]
            if p_throws   != "All": df = df[df["p_throws"]   == p_throws]
            if stand      != "All": df = df[df["stand"]       == stand]
        df["game_year"] = yr
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    full = pd.concat(parts, ignore_index=True)
    full = full[full["zone"].between(1, 14)]

    # Count filter
    if count_state != "All":
        full["count_state"] = full["balls"].astype(str) + "-" + full["strikes"].astype(str)
        full = full[full["count_state"] == count_state]

    # Movement columns (inches)
    full["hbrk"] = pd.to_numeric(full["pfx_x"], errors="coerce") * 12
    full["vbrk"] = pd.to_numeric(full["pfx_z"], errors="coerce") * 12

    # Numeric casts
    for c in ["release_speed","release_spin_rate","launch_speed","launch_angle",
              "estimated_woba_using_speedangle"]:
        full[c] = pd.to_numeric(full[c], errors="coerce")

    # Velocity filter
    full = full[full["release_speed"].between(velo_min, velo_max).fillna(True)]
    # Spin filter
    full = full[full["release_spin_rate"].between(spin_min, spin_max).fillna(True)]
    # H-break filter
    full = full[full["hbrk"].between(hbrk_min, hbrk_max).fillna(True)]
    # V-break filter
    full = full[full["vbrk"].between(vbrk_min, vbrk_max).fillna(True)]

    # Event flags
    desc = full["description"].fillna("")
    full["is_swing"]   = desc.isin(SWING_EV).astype("int8")
    full["is_whiff"]   = desc.isin(WHIFF_EV).astype("int8")
    full["is_contact"] = desc.isin(CONTACT_EV).astype("int8")
    ls = full["launch_speed"]; la = full["launch_angle"]
    full["is_barrel"]  = ((ls >= 98) & la.between(26,30)).fillna(False).astype("int8")
    full["is_hh"]      = (ls >= 95).fillna(False).astype("int8")
    full["is_gb"]      = (la < 10).fillna(False).astype("int8")

    return full


def compute_zone_stats_with_movement(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Like compute_zone_stats_from_raw but also returns avg_spin, avg_hbrk, avg_vbrk.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    valid = raw[raw["zone"].between(1, 14)]
    if valid.empty:
        return pd.DataFrame()

    grp = valid.groupby("zone", as_index=False).agg(
        total       = ("is_swing",   "count"),
        swings      = ("is_swing",   "sum"),
        whiffs      = ("is_whiff",   "sum"),
        contacts    = ("is_contact", "sum"),
        barrels     = ("is_barrel",  "sum"),
        hard_hits   = ("is_hh",      "sum"),
        gbs         = ("is_gb",      "sum"),
        batted      = ("launch_speed","count"),
        avg_ev      = ("launch_speed","mean"),
        avg_la      = ("launch_angle","mean"),
        avg_xwoba   = ("estimated_woba_using_speedangle","mean"),
        avg_spin    = ("release_spin_rate","mean"),
        avg_hbrk    = ("hbrk","mean"),
        avg_vbrk    = ("vbrk","mean"),
        avg_velo    = ("release_speed","mean"),
    )
    n  = grp["total"].replace(0, np.nan)
    sw = grp["swings"].replace(0, np.nan)
    bt = grp["batted"].replace(0, np.nan)
    grp["swing_pct"]    = (grp["swings"]   / n  * 100).round(1)
    grp["whiff_pct"]    = (grp["whiffs"]   / sw * 100).round(1)
    grp["contact_pct"]  = (grp["contacts"] / sw * 100).round(1)
    grp["barrel_pct"]   = (grp["barrels"]  / bt * 100).round(1)
    grp["hard_hit_pct"] = (grp["hard_hits"]/ bt * 100).round(1)
    grp["gb_pct"]       = (grp["gbs"]      / bt * 100).round(1)
    for c in ["avg_ev","avg_la","avg_spin","avg_hbrk","avg_vbrk","avg_velo"]:
        grp[c] = grp[c].round(1)
    grp["avg_xwoba"] = grp["avg_xwoba"].round(3)
    return grp


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — MAIN DASHBOARD  (complete replacement)
#
# Layout:
#   ROW 1 │ LEFT: main 14-zone heatmap │ RIGHT: sub-zone grid (zones 1-9)
#   ROW 2 │ Side-by-side comparison (A vs B)
#   ROW 3 │ Zone summary table with spin · H-break · V-break
#
# FIX: draw_subzone_panel → draw_subzone_detail (function defined in Part 2)
# ═══════════════════════════════════════════════════════════════════

# ── New helper: load raw columns for subzone + movement ──────────────
# Add this function near precompute_zone_stats in your helpers section.

@st.cache_data(ttl=3600, show_spinner=False)
def load_raw_for_subzones(
    years: tuple,
    pitch_type: str = "All",
    p_throws: str = "All",
    stand: str = "All",
    count_state: str = "All",
    velo_min: float = 60.0,
    velo_max: float = 105.0,
    spin_min: float = 800.0,
    spin_max: float = 3600.0,
    hbrk_min: float = -22.0,
    hbrk_max: float = 22.0,
    vbrk_min: float = -18.0,
    vbrk_max: float = 22.0,
) -> pd.DataFrame:
    """
    Ładuje surowe dane potrzebne do sub-zone analysis — teraz po miesiącach.
    """
    RAW_COLS = [
        "pitch_type","zone","stand","p_throws","balls","strikes",
        "plate_x","plate_z","description",
        "release_speed","release_spin_rate","pfx_x","pfx_z",
        "launch_speed","launch_angle","estimated_woba_using_speedangle",
    ]
    parts = []

    for yr in years:
        if yr not in STATCAST_FILES:
            continue
        for path in STATCAST_FILES[yr]:
            if not path.exists():
                continue
            try:
                if PYARROW_OK:
                    schema_cols = set(pq.read_schema(str(path)).names)
                    use_cols = [c for c in RAW_COLS if c in schema_cols]
                    
                    filters = []
                    if pitch_type != "All":
                        filters.append(("pitch_type", "=", pitch_type))
                    if p_throws != "All":
                        filters.append(("p_throws", "=", p_throws))
                    if stand != "All":
                        filters.append(("stand", "=", stand))
                    
                    table = pq.read_table(
                        str(path),
                        columns=use_cols,
                        filters=filters if filters else None
                    )
                    df = table.to_pandas()
                else:
                    df = pd.read_parquet(path, columns=RAW_COLS)
                    if pitch_type != "All":
                        df = df[df["pitch_type"] == pitch_type]
                    if p_throws != "All":
                        df = df[df["p_throws"] == p_throws]
                    if stand != "All":
                        df = df[df["stand"] == stand]

                df["game_year"] = yr
                parts.append(df)
            except Exception:
                continue

    if not parts:
        return pd.DataFrame()

    full = pd.concat(parts, ignore_index=True)
    full = full[full["zone"].between(1, 14)]

    # Dodatkowe filtry po wczytaniu
    if count_state != "All":
        full["count_state"] = (
            full["balls"].astype(str).str.strip() + "-" +
            full["strikes"].astype(str).str.strip()
        )
        full = full[full["count_state"] == count_state]

    # Filtry numeryczne
    if "release_speed" in full.columns:
        full = full[
            pd.to_numeric(full["release_speed"], errors="coerce")
            .between(velo_min, velo_max).fillna(True)
        ]
    if "release_spin_rate" in full.columns:
        full = full[
            pd.to_numeric(full["release_spin_rate"], errors="coerce")
            .between(spin_min, spin_max).fillna(True)
        ]
    if "hbrk" in full.columns or "pfx_x" in full.columns:
        hbrk_col = "hbrk" if "hbrk" in full.columns else "pfx_x"
        full[hbrk_col] = pd.to_numeric(full[hbrk_col], errors="coerce") * 12
        full = full[full[hbrk_col].between(hbrk_min, hbrk_max).fillna(True)]
    if "vbrk" in full.columns or "pfx_z" in full.columns:
        vbrk_col = "vbrk" if "vbrk" in full.columns else "pfx_z"
        full[vbrk_col] = pd.to_numeric(full[vbrk_col], errors="coerce") * 12
        full = full[full[vbrk_col].between(vbrk_min, vbrk_max).fillna(True)]

    return full


def compute_zone_stats_with_movement(raw: pd.DataFrame) -> pd.DataFrame:
    """compute_zone_stats_from_raw + avg spin, hbrk, vbrk, velo per zone."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    valid = raw[raw["zone"].between(1, 14)]
    if valid.empty:
        return pd.DataFrame()

    grp = valid.groupby("zone", as_index=False).agg(
        total     = ("is_swing",   "count"),
        swings    = ("is_swing",   "sum"),
        whiffs    = ("is_whiff",   "sum"),
        contacts  = ("is_contact", "sum"),
        barrels   = ("is_barrel",  "sum"),
        hard_hits = ("is_hh",      "sum"),
        gbs       = ("is_gb",      "sum"),
        batted    = ("launch_speed","count"),
        avg_ev    = ("launch_speed","mean"),
        avg_la    = ("launch_angle","mean"),
        avg_xwoba = ("estimated_woba_using_speedangle","mean"),
        avg_spin  = ("release_spin_rate","mean"),
        avg_hbrk  = ("hbrk","mean"),
        avg_vbrk  = ("vbrk","mean"),
        avg_velo  = ("release_speed","mean"),
    )
    n  = grp["total"].replace(0, np.nan)
    sw = grp["swings"].replace(0, np.nan)
    bt = grp["batted"].replace(0, np.nan)
    grp["swing_pct"]    = (grp["swings"]   / n  * 100).round(1)
    grp["whiff_pct"]    = (grp["whiffs"]   / sw * 100).round(1)
    grp["contact_pct"]  = (grp["contacts"] / sw * 100).round(1)
    grp["barrel_pct"]   = (grp["barrels"]  / bt * 100).round(1)
    grp["hard_hit_pct"] = (grp["hard_hits"]/ bt * 100).round(1)
    grp["gb_pct"]       = (grp["gbs"]      / bt * 100).round(1)
    for c in ["avg_ev","avg_la","avg_spin","avg_hbrk","avg_vbrk","avg_velo"]:
        grp[c] = grp[c].round(1)
    grp["avg_xwoba"] = grp["avg_xwoba"].round(3)
    return grp


# ── New: draw ALL 9 zones as sub-zone grid in one figure ─────────────
def draw_all_subzones_grid(
    raw: pd.DataFrame,
    stat_label: str = "Whiff %",
) -> plt.Figure:
    """
    Draw a 3×3 strike zone where each of the 9 zones is divided into
    4 sub-quadrants (TL, TR, BL, BR), coloured by stat_label.
    Shown RIGHT of the main heatmap — no popup, fully inline.
    """
    STAT_COL_MAP = {
        "Whiff %":     "whiff_pct",
        "Swing %":     "swing_pct",
        "Contact %":   "contact_pct",
        "xwOBA":       "avg_xwoba",
        "Exit Velo":   "avg_ev",
        "Launch Angle":"avg_la",
        "Barrel %":    "barrel_pct",
        "Hard Hit %":  "hard_hit_pct",
        "GB %":        "gb_pct",
    }
    FMT_MAP = {
        "Whiff %":"{:.1f}%", "Swing %":"{:.1f}%", "Contact %":"{:.1f}%",
        "xwOBA":"{:.3f}",    "Exit Velo":"{:.1f}", "Launch Angle":"{:.1f}°",
        "Barrel %":"{:.1f}%","Hard Hit %":"{:.1f}%","GB %":"{:.1f}%",
    }
    RANGE_MAP = {
        "Whiff %":(0,70), "Swing %":(10,90), "Contact %":(30,100),
        "xwOBA":(0.10,0.55), "Exit Velo":(70,98), "Launch Angle":(-15,45),
        "Barrel %":(0,25), "Hard Hit %":(0,60), "GB %":(0,80),
    }

    stat_col    = STAT_COL_MAP.get(stat_label, "whiff_pct")
    fmt         = FMT_MAP.get(stat_label, "{:.1f}")
    vmin, vmax  = RANGE_MAP.get(stat_label, (0, 100))
    cmap        = sns.color_palette("YlOrRd", as_cmap=True)

    # Pre-classify every pitch into its sub-zone
    if raw.empty or "plate_x" not in raw.columns:
        fig, ax = plt.subplots(figsize=(6, 6.5))
        fig.patch.set_facecolor("#0b0f17"); ax.axis("off")
        ax.text(0.5, 0.5, "No data\n(apply filters first)",
                ha="center", va="center", color="#8892a4", fontsize=11)
        return fig

    bz = raw[raw["zone"].between(1, 9)].copy()
    bz["sub"] = bz.apply(
        lambda r: classify_subzone(
            float(r["plate_x"]) if not pd.isna(r["plate_x"]) else 0.0,
            float(r["plate_z"]) if not pd.isna(r["plate_z"]) else 2.5,
            int(r["zone"]),
        ), axis=1,
    )

    # Aggregate per (zone, sub_zone)
    sg = bz.groupby(["zone","sub"]).agg(
        total     = ("is_swing",   "count"),
        swings    = ("is_swing",   "sum"),
        whiffs    = ("is_whiff",   "sum"),
        contacts  = ("is_contact", "sum"),
        barrels   = ("is_barrel",  "sum"),
        hard_hits = ("is_hh",      "sum"),
        gbs       = ("is_gb",      "sum"),
        batted    = ("launch_speed","count"),
        avg_ev    = ("launch_speed","mean"),
        avg_la    = ("launch_angle","mean"),
        avg_xwoba = ("estimated_woba_using_speedangle","mean"),
    ).reset_index()
    sw2 = sg["swings"].replace(0, np.nan)
    n2  = sg["total"].replace(0, np.nan)
    bt2 = sg["batted"].replace(0, np.nan)
    sg["whiff_pct"]    = (sg["whiffs"]   / sw2 * 100).round(1)
    sg["swing_pct"]    = (sg["swings"]   / n2  * 100).round(1)
    sg["contact_pct"]  = (sg["contacts"] / sw2 * 100).round(1)
    sg["barrel_pct"]   = (sg["barrels"]  / bt2 * 100).round(1)
    sg["hard_hit_pct"] = (sg["hard_hits"]/ bt2 * 100).round(1)
    sg["gb_pct"]       = (sg["gbs"]      / bt2 * 100).round(1)
    sg["avg_xwoba"]    = sg["avg_xwoba"].round(3)
    sg["avg_ev"]       = sg["avg_ev"].round(1)
    sg["avg_la"]       = sg["avg_la"].round(1)
    sg_idx = sg.set_index(["zone","sub"])

    # Zone layout: row 0 = top (zones 1,2,3), row 2 = bottom (7,8,9)
    # Sub-zone positions within each zone cell:
    #   TL=(0,1)  TR=(1,1)
    #   BL=(0,0)  BR=(1,0)
    sub_pos = {"TL":(0,1), "TR":(1,1), "BL":(0,0), "BR":(1,0)}

    fig, ax = plt.subplots(figsize=(6.5, 7.2))
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#0b0f17")
    ax.set_xlim(0, 3); ax.set_ylim(0, 3.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Sub-Zone Grid — {stat_label}", color="#c9d1d9",
                 fontsize=10, fontweight="600", pad=8)

    CELL = 1.0     # each zone = 1×1
    SUB  = 0.5     # each sub = 0.5×0.5
    PAD  = 0.015   # gap between sub-cells

    for i in range(3):          # row: 0=top(zones1-3), 1=mid(4-6), 2=bot(7-9)
        for j in range(3):      # col: 0=left, 1=center, 2=right
            zone    = i * 3 + j + 1
            x0_zone = j * CELL
            y0_zone = (2 - i) * CELL   # flip so top row is drawn at top

            # Draw zone border
            ax.add_patch(plt.Rectangle(
                (x0_zone, y0_zone), CELL, CELL,
                fill=False, edgecolor="#f85149", linewidth=1.8, zorder=5,
            ))

            # Zone label (top-right corner)
            ax.text(x0_zone + CELL - 0.04, y0_zone + CELL - 0.04,
                    str(zone), ha="right", va="top",
                    fontsize=7.5, color="#e2e8f0", fontweight="700", zorder=6)

            # Draw 4 sub-cells
            for quad, (qi, qj) in sub_pos.items():
                x0_sub = x0_zone + qi * SUB + PAD
                y0_sub = y0_zone + qj * SUB + PAD
                w_sub  = SUB - 2 * PAD

                # Get value
                try:
                    row_sg = sg_idx.loc[(zone, quad)]
                    val    = float(row_sg.get(stat_col, np.nan))
                    n_p    = int(row_sg.get("total", 0))
                except (KeyError, TypeError):
                    val  = np.nan
                    n_p  = 0

                if not pd.isna(val) and n_p > 0:
                    colour = cmap(np.clip((val - vmin) / (vmax - vmin), 0, 1))
                    fc     = colour
                    txt_v  = fmt.format(val)
                    txt_c  = "#111111"
                else:
                    fc     = "#1c2230"
                    txt_v  = "—"
                    txt_c  = "#4a5568"

                ax.add_patch(plt.Rectangle(
                    (x0_sub, y0_sub), w_sub, w_sub,
                    facecolor=fc, edgecolor="#1e2535", linewidth=0.8,
                ))
                # Value text
                ax.text(x0_sub + w_sub/2, y0_sub + w_sub*0.60,
                        txt_v, ha="center", va="center",
                        fontsize=6.5, fontweight="700", color=txt_c)
                # Quad label
                ax.text(x0_sub + w_sub/2, y0_sub + w_sub*0.24,
                        quad, ha="center", va="center",
                        fontsize=5.5, color=txt_c, alpha=0.7)

    # Colour bar
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=18)
    cbar.set_label(stat_label, fontsize=8, color="#8892a4")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8892a4", fontsize=7)
    cbar.outline.set_edgecolor("#1e2535")

    ax.text(1.5, -0.18, "← Inside (catcher's view)  /  Outside →",
            ha="center", va="top", fontsize=7, color="#718096", style="italic")

    plt.tight_layout(pad=0.3)
    return fig


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — MAIN DASHBOARD (clean + collapsible sub-zones)
# ─────────────────────────────────────────────────────────────────────
# ── Dodaj tu przed with tab_main: ─────────────────────────────────
def draw_subzone_panel(raw_df: pd.DataFrame, zone: int,
                       stat_label: str = "Whiff %") -> plt.Figure:
    """2×2 sub-zone panel from raw data — inline, no popup."""
    STAT_COL_MAP = {
        "Whiff %":"whiff_pct","Swing %":"swing_pct","Contact %":"contact_pct",
        "xwOBA":"avg_xwoba","Exit Velo":"avg_ev","Launch Angle":"avg_la",
        "Barrel %":"barrel_pct","Hard Hit %":"hard_hit_pct","GB %":"gb_pct",
    }
    FMT_MAP = {
        "Whiff %":"{:.1f}%","Swing %":"{:.1f}%","Contact %":"{:.1f}%",
        "xwOBA":"{:.3f}","Exit Velo":"{:.1f}","Launch Angle":"{:.1f}°",
        "Barrel %":"{:.1f}%","Hard Hit %":"{:.1f}%","GB %":"{:.1f}%",
    }
    RANGE_MAP = {
        "Whiff %":(0,70),"Swing %":(10,90),"Contact %":(30,100),
        "xwOBA":(0.10,0.55),"Exit Velo":(70,98),"Launch Angle":(-15,45),
        "Barrel %":(0,25),"Hard Hit %":(0,60),"GB %":(0,80),
    }
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    vmin, vmax = RANGE_MAP.get(stat_label, (0, 100))
    fmt = FMT_MAP.get(stat_label, "{:.1f}")
    stat_col = STAT_COL_MAP.get(stat_label, "whiff_pct")

    bz = raw_df[raw_df["zone"] == zone].copy()
    if bz.empty:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#111621"); ax.axis("off")
        ax.text(0.5, 0.5, f"No data\nfor Zone {zone}",
                ha="center", va="center", color="#8892a4", fontsize=12)
        return fig

    bz["sub"] = bz.apply(
        lambda r: classify_subzone(
            float(r["plate_x"]) if not pd.isna(r.get("plate_x")) else 0.0,
            float(r["plate_z"]) if not pd.isna(r.get("plate_z")) else 2.5,
            zone,
        ), axis=1,
    )
    desc = bz["description"].fillna("")
    bz["is_swing"]   = desc.isin(SWING_EV).astype(int)
    bz["is_whiff"]   = desc.isin(WHIFF_EV).astype(int)
    bz["is_contact"] = desc.isin(CONTACT_EV).astype(int)
    ls = pd.to_numeric(bz["launch_speed"], errors="coerce")
    la = pd.to_numeric(bz["launch_angle"], errors="coerce")
    bz["is_barrel"] = ((ls >= 98) & la.between(26,30)).fillna(False).astype(int)
    bz["is_hh"]     = (ls >= 95).fillna(False).astype(int)
    bz["is_gb"]     = (la < 10).fillna(False).astype(int)
    if "hbrk" not in bz.columns and "pfx_x" in bz.columns:
        bz["hbrk"] = pd.to_numeric(bz["pfx_x"], errors="coerce") * 12
    if "vbrk" not in bz.columns and "pfx_z" in bz.columns:
        bz["vbrk"] = pd.to_numeric(bz["pfx_z"], errors="coerce") * 12

    sg = bz.groupby("sub", as_index=False).agg(
        total     = ("is_swing",   "count"),
        swings    = ("is_swing",   "sum"),
        whiffs    = ("is_whiff",   "sum"),
        contacts  = ("is_contact", "sum"),
        barrels   = ("is_barrel",  "sum"),
        hhs       = ("is_hh",      "sum"),
        gbs       = ("is_gb",      "sum"),
        batted    = ("launch_speed","count"),
        avg_ev    = ("launch_speed","mean"),
        avg_la    = ("launch_angle","mean"),
        avg_xwoba = ("estimated_woba_using_speedangle","mean"),
    )
    sw = sg["swings"].replace(0, np.nan)
    n  = sg["total"].replace(0, np.nan)
    bt = sg["batted"].replace(0, np.nan)
    sg["whiff_pct"]   = (sg["whiffs"]  / sw * 100).round(1)
    sg["swing_pct"]   = (sg["swings"]  / n  * 100).round(1)
    sg["contact_pct"] = (sg["contacts"]/ sw * 100).round(1)
    sg["barrel_pct"]  = (sg["barrels"] / bt * 100).round(1)
    sg["hard_hit_pct"]= (sg["hhs"]     / bt * 100).round(1)
    sg["gb_pct"]      = (sg["gbs"]     / bt * 100).round(1)
    sg["avg_ev"]      = sg["avg_ev"].round(1)
    sg["avg_la"]      = sg["avg_la"].round(1)
    sg["avg_xwoba"]   = sg["avg_xwoba"].round(3)
    sg = sg.set_index("sub")

    positions = {"TL":(0,1),"TR":(1,1),"BL":(0,0),"BR":(1,0)}
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    fig.patch.set_facecolor("#0b0f17")
    fig.suptitle(f"Zone {zone} — {stat_label}",
                 color="#f0f6ff", fontsize=10, fontweight="700", y=1.01)
    for quad, (ci, ri) in positions.items():
        ax = axes[1 - ri][ci]
        ax.set_facecolor("#111621"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        if quad in sg.index:
            row = sg.loc[quad]
            val = row.get(stat_col, np.nan)
            n_p = int(row.get("total", 0))
            if not pd.isna(val) and n_p > 0:
                colour = cmap(np.clip((val - vmin)/(vmax - vmin), 0, 1))
                ax.add_patch(plt.Rectangle((0.05,0.05),0.9,0.9,
                    facecolor=colour, edgecolor="#1e2535", linewidth=2.0))
                ax.text(0.5, 0.60, fmt.format(val), ha="center", va="center",
                        fontsize=15, fontweight="800", color="#111111")
                ax.text(0.5, 0.32, quad, ha="center", va="center",
                        fontsize=9, fontweight="600", color="#333333")
                ax.text(0.5, 0.14, f"n={n_p}", ha="center", va="center",
                        fontsize=7.5, color="#555555")
            else:
                ax.add_patch(plt.Rectangle((0.05,0.05),0.9,0.9,
                    facecolor="#1c2230", edgecolor="#1e2535", linewidth=1.5))
                ax.text(0.5, 0.5, f"{quad}\nno data", ha="center", va="center",
                        fontsize=8, color="#4a5568")
        else:
            ax.add_patch(plt.Rectangle((0.05,0.05),0.9,0.9,
                facecolor="#1c2230", edgecolor="#1e2535", linewidth=1.5))
            ax.text(0.5, 0.5, f"{quad}\nno data", ha="center", va="center",
                    fontsize=8, color="#4a5568")
    plt.tight_layout(pad=0.5)
    return fig

@st.cache_data(ttl=86400, show_spinner="⏳ Loading zone stats…")
def _get_zone_stats():
    # Najpierw sprawdź czy mamy gotowy plik (z precompute.py)
    precomp = DATA_DIR / "zone_stats_agg.parquet"
    if precomp.exists():
        return pd.read_parquet(precomp, engine="pyarrow")  # ~0.2s
    # Fallback: oblicz od zera (~60s)
    return precompute_zone_stats(ALL_YEARS)

with tab_main:
    st.markdown('<div class="sec-hdr">📊 Zone Heatmap — All Pitches</div>',
                unsafe_allow_html=True)

    # Pre-aggregated stats
    @st.cache_data(ttl=3600, show_spinner="⏳ Loading zone stats...")
    def _get_zone_stats():
        precomp = DATA_DIR / "zone_stats_agg.parquet"
        if precomp.exists():
            return pd.read_parquet(precomp, engine="pyarrow")
        return precompute_zone_stats(ALL_YEARS)

    _zone_df = _get_zone_stats()

    if _zone_df.empty:
        st.error("No Statcast parquet files found.")
        st.stop()

    _pt_opts = ["All"] + sorted(_zone_df["pitch_type"].dropna().unique().tolist())
    _yr_opts = sorted(_zone_df["game_year"].dropna().unique().astype(int).tolist())

    # Filters
    r1 = st.columns([1,1,1,1,1,2])
    with r1[0]: yr_m = st.multiselect("Season", _yr_opts, default=_yr_opts, key="m_yr")
    with r1[1]: pt_m = st.selectbox("Pitch Type", _pt_opts, key="m_pt")
    with r1[2]: ph_m = st.selectbox("Pitcher Hand", ["All","R","L"], key="m_ph")
    with r1[3]: bh_m = st.selectbox("Batter Hand", ["All","R","L"], key="m_bh")
    with r1[4]: cnt_m = st.selectbox("Count", ["All"]+ALL_COUNTS, key="m_cnt")
    with r1[5]: stat_m = st.selectbox("Statistic", STAT_LABELS, key="m_stat")

    # Movement filters
    with st.expander("🔬 Velocity · Spin · Movement filters", expanded=False):
        ef1, ef2 = st.columns(2)
        with ef1:
            velo_m = st.slider("Velocity (mph)", 60, 105, (70, 102), key="m_velo")
            spin_m = st.slider("Spin rate (rpm)", 800, 3600, (1200, 3400), key="m_spin")
        with ef2:
            hbrk_m = st.slider("H-Break (in)", -22.0, 22.0, (-18.0, 18.0), key="m_hbrk")
            vbrk_m = st.slider("V-Break (in)", -18.0, 22.0, (-8.0, 18.0), key="m_vbrk")

    _has_mvmt = (velo_m != (70,102) or spin_m != (1200,3400) or
                 hbrk_m != (-18.0,18.0) or vbrk_m != (-8.0,18.0))

    if _has_mvmt:
        with st.spinner("Applying movement filters..."):
            _raw_m = load_raw_for_subzones(
                years=tuple(yr_m), pitch_type=pt_m, p_throws=ph_m,
                stand=bh_m, count_state=cnt_m,
                velo_min=float(velo_m[0]), velo_max=float(velo_m[1]),
                spin_min=float(spin_m[0]), spin_max=float(spin_m[1]),
                hbrk_min=float(hbrk_m[0]), hbrk_max=float(hbrk_m[1]),
                vbrk_min=float(vbrk_m[0]), vbrk_max=float(vbrk_m[1])
            )
        _df_main = compute_zone_stats_with_movement(_raw_m) if '_raw_m' in locals() else pd.DataFrame()
    else:
        _df_main = apply_tab1_filters(_zone_df, yr_m, pt_m, ph_m, bh_m, cnt_m)

    # Main Heatmap + Sub-zone
    col_hm, col_sz = st.columns([5, 4], gap="medium")

    with col_hm:
        _main_title = " · ".join(filter(None, [
            stat_m,
            pt_m if pt_m != "All" else None,
            f"P:{ph_m}HP" if ph_m != "All" else None,
            f"B:{bh_m}HB" if bh_m != "All" else None,
            f"Count {cnt_m}" if cnt_m != "All" else None
        ]))
        fig_m = draw_heatmap(_df_main, stat_m, _main_title)
        st.pyplot(fig_m, use_container_width=True)
        plt.close(fig_m)

    with col_sz:
        st.markdown(
            f'<div style="color:#79b8ff;font-weight:600;font-size:.9rem;margin-bottom:6px;">'
            f'Sub-Zone Breakdown — {stat_m}</div>',
            unsafe_allow_html=True,
        )

        sz_zone_m = st.selectbox(
            "Select zone for detailed sub-zone view:",
            list(range(1,10)),
            format_func=lambda z: f"Zone {z}",
            key="sz_zone_m_main"
        )
        sz_stat_m = st.selectbox(
            "Sub-zone statistic:",
            STAT_LABELS,
            key="sz_stat_m_main",
        )

        # Sub-zone data
        if _has_mvmt and '_raw_m' in locals() and not _raw_m.empty:
            _raw_for_sz = _raw_m
        else:
            with st.spinner("Loading sub-zone data..."):
                _raw_for_sz = load_raw_for_subzones(
                    years=tuple(yr_m), pitch_type=pt_m, p_throws=ph_m,
                    stand=bh_m, count_state=cnt_m,
                    velo_min=60, velo_max=105, spin_min=800, spin_max=3600,
                    hbrk_min=-22, hbrk_max=22, vbrk_min=-18, vbrk_max=22
                )

        if not _raw_for_sz.empty:
            fig_sz = draw_subzone_panel(_raw_for_sz, sz_zone_m, sz_stat_m)
            st.pyplot(fig_sz, use_container_width=True)
            plt.close(fig_sz)

           # Quick stats table
            _sz_data = _raw_for_sz[_raw_for_sz["zone"] == sz_zone_m].copy()
            if not _sz_data.empty and "plate_x" in _sz_data.columns:
                # BRAKUJĄCY KROK: utwórz kolumnę "sub" przed groupby
                _sz_data["sub"] = _sz_data.apply(
                    lambda r: classify_subzone(
                        float(r["plate_x"]) if not pd.isna(r["plate_x"]) else 0.0,
                        float(r["plate_z"]) if not pd.isna(r["plate_z"]) else 2.5,
                        sz_zone_m,
                    ), axis=1,
                )

                # Upewnij się że mamy flagi
                if "is_swing" not in _sz_data.columns:
                    _sz_data = _add_flags(_sz_data)
                if "hbrk" not in _sz_data.columns and "pfx_x" in _sz_data.columns:
                    _sz_data["hbrk"] = pd.to_numeric(_sz_data["pfx_x"], errors="coerce") * 12
                if "vbrk" not in _sz_data.columns and "pfx_z" in _sz_data.columns:
                    _sz_data["vbrk"] = pd.to_numeric(_sz_data["pfx_z"], errors="coerce") * 12

                _sz_tbl = _sz_data.groupby("sub", as_index=False).agg(
                    Pitches = ("is_swing",  "count"),
                    Swing_p = ("is_swing",  "mean"),
                    Whiff_p = ("is_whiff",  "mean"),
                    xwOBA   = ("estimated_woba_using_speedangle", "mean"),
                    EV      = ("launch_speed", "mean"),
                    Spin    = ("release_spin_rate", "mean"),
                    HBreak  = ("hbrk", "mean"),
                    VBreak  = ("vbrk", "mean"),
                )
                _sz_tbl["Swing%"]  = (_sz_tbl["Swing_p"] * 100).round(1)
                _sz_tbl["Whiff%"]  = (_sz_tbl["Whiff_p"] * 100).round(1)
                _sz_tbl["xwOBA"]   = _sz_tbl["xwOBA"].round(3)
                _sz_tbl["EV"]      = _sz_tbl["EV"].round(1)
                _sz_tbl["Spin"]    = _sz_tbl["Spin"].round(0)
                _sz_tbl["HBreak"]  = _sz_tbl["HBreak"].round(1)
                _sz_tbl["VBreak"]  = _sz_tbl["VBreak"].round(1)

                final_cols = ["sub", "Pitches", "Swing%", "Whiff%",
                              "xwOBA", "EV", "Spin", "HBreak", "VBreak"]
                _sz_tbl = _sz_tbl[[c for c in final_cols if c in _sz_tbl.columns]]
                _sz_tbl = _sz_tbl.rename(columns={
                    "sub": "Quadrant", "HBreak": "H-Brk\"", "VBreak": "V-Brk\""
                })
                st.dataframe(
                    _sz_tbl.set_index("Quadrant"),
                    width="stretch",
                    height=200,
                )
           

    # Zone summary table + Comparison (możesz zostawić resztę jak była)

    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)

    # ── Zone Summary Table with Spin + Movement ────────────────────────
    st.markdown(
        '<div class="sec-hdr">Zone-by-Zone Summary — Spin · H-Break · V-Break</div>',
        unsafe_allow_html=True,
    )
    if not _df_main.empty:
        _sum_base = [c for c in [
            "zone","total","swing_pct","whiff_pct","contact_pct",
            "barrel_pct","hard_hit_pct","gb_pct","avg_xwoba","avg_ev","avg_la",
        ] if c in _df_main.columns]
        _sum_mvmt = [c for c in [
            "avg_velo","avg_spin","avg_hbrk","avg_vbrk",
        ] if c in _df_main.columns]

            # If movement cols not in _df_main (pre-agg path), compute from raw
    if not _sum_mvmt and not _raw_for_sz.empty:
        with st.spinner("Computing movement stats..."):
            _raw_for_mvmt = _add_flags(_raw_for_sz.copy())   # <<< TO JEST KLUCZ
            _df_mvmt = compute_zone_stats_with_movement(_raw_for_mvmt)
            _mvmt_cols = [c for c in ["avg_spin","avg_hbrk","avg_vbrk","avg_velo"]
                          if c in _df_mvmt.columns]
            if _mvmt_cols:
                _df_main = _df_main.merge(
                    _df_mvmt[["zone"] + _mvmt_cols], on="zone", how="left"
                )
                _sum_mvmt = [c for c in _mvmt_cols if c in _df_main.columns]

        _rn = {
            "total":"Pitches","swing_pct":"Swing%","whiff_pct":"Whiff%",
            "contact_pct":"Contact%","barrel_pct":"Barrel%",
            "hard_hit_pct":"Hard Hit%","gb_pct":"GB%",
            "avg_xwoba":"xwOBA","avg_ev":"Exit V","avg_la":"Launch°",
            "avg_velo":"Avg Velo","avg_spin":"Spin rpm",
            "avg_hbrk":"H-Brk\"","avg_vbrk":"V-Brk\"",
        }
        st.dataframe(
            _df_main[_sum_base + _sum_mvmt].rename(columns=_rn)
            .sort_values("zone").reset_index(drop=True),
            width="stretch", 
            height=430,
        )
    else:
        st.info("No data for the current filter combination.")

    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # ROW 2 — SIDE-BY-SIDE COMPARISON
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-hdr">🔄 Side-by-Side Comparison</div>',
                unsafe_allow_html=True)

    col_cA, col_cB = st.columns(2, gap="large")

    for col_w, pfx, default_stat, lbl in [
        (col_cA, "ca", "Whiff %", "⬅️ Config A"),
        (col_cB, "cb", "xwOBA",  "➡️ Config B"),
    ]:
        with col_w:
            st.markdown(
                f'<div style="color:#79b8ff;font-weight:600;font-size:.88rem;'
                f'margin-bottom:8px;">{lbl}</div>',
                unsafe_allow_html=True,
            )
            xc1, xc2, xc3 = st.columns(3)
            with xc1: pt_x  = st.selectbox("Pitch Type",   _pt_opts,       key=f"{pfx}_pt")
            with xc2: ph_x  = st.selectbox("Pitcher Hand", ["All","R","L"], key=f"{pfx}_ph")
            with xc3: bh_x  = st.selectbox("Batter Hand",  ["All","R","L"], key=f"{pfx}_bh")
            xc4, xc5 = st.columns(2)
            with xc4: cnt_x = st.selectbox("Count",     ["All"]+ALL_COUNTS, key=f"{pfx}_cnt")
            with xc5:
                _di = STAT_LABELS.index(default_stat) if default_stat in STAT_LABELS else 0
                st_x = st.selectbox("Statistic", STAT_LABELS,
                                     index=_di, key=f"{pfx}_stat")

            _df_x = apply_tab1_filters(
                _zone_df,
                years=yr_m, pitch_type=pt_x,
                p_throws=ph_x, stand=bh_x, count_state=cnt_x,
            )
            _ttl_x = st_x
            if pt_x  != "All": _ttl_x += f"  ·  {pt_x}"
            if cnt_x != "All": _ttl_x += f"  ·  {cnt_x}"
            if ph_x  != "All": _ttl_x += f"  P:{ph_x}HP"
            if bh_x  != "All": _ttl_x += f"  B:{bh_x}HB"

            fig_x = draw_heatmap(_df_x, st_x, _ttl_x)
            st.pyplot(fig_x, use_container_width=True)
            plt.close(fig_x)

            # Sub-zone grid for comparison columns — inline, no popup
            sz_stat_cmp = st.selectbox(
                f"Sub-zone stat ({lbl}):",
                STAT_LABELS,
                key=f"{pfx}_sz_stat",
            )
            with st.spinner(f"Sub-zone grid {lbl}…"):
                _raw_cmp = load_raw_for_subzones(
                    years=tuple(yr_m), pitch_type=pt_x, p_throws=ph_x,
                    stand=bh_x, count_state=cnt_x,
                    velo_min=60.0, velo_max=105.0,
                    spin_min=800.0, spin_max=3600.0,
                    hbrk_min=-22.0, hbrk_max=22.0,
                    vbrk_min=-18.0, vbrk_max=22.0,
                )
            if not _raw_cmp.empty:
                fig_sz_cmp = draw_all_subzones_grid(_raw_cmp, sz_stat_cmp)
                st.pyplot(fig_sz_cmp, use_container_width=True)
                plt.close(fig_sz_cmp)
            else:
                st.info("No sub-zone data for this config.")

    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — PITCHER SCOUT  (complete replacement)
# Paste this block to replace your entire "with tab_pitcher:" section
# ═══════════════════════════════════════════════════════════════════

# ── Helper: build pitcher roster per season ──────────────────────────
# Put this near build_meta_maps(), before the tabs.
# It reads from pitching_stats_*.parquet (already loaded in _pitching_df)
# and cross-references with _pitcher_meta by name match.

@st.cache_data(ttl=7200, show_spinner=False)
def build_pitcher_team_roster(pitching_df: pd.DataFrame,
                               pitcher_meta: dict) -> dict:
    """
    Returns {season: {team: [display_str, ...]}}
    where display_str matches keys in _pitcher_disp_map.

    Strategy: pitching_stats parquets have Name + Team + season.
    We match to pitcher_meta by display name (first+last).
    """
    # Build reverse map: display_name_lower → display_str (key in _pitcher_disp_map)
    # display_str format: "First Last  (RHP)"
    rev_lower = {}
    for pid, meta in pitcher_meta.items():
        raw = meta.get("name", "")
        parts = raw.split(",")
        if len(parts) == 2:
            fn = parts[1].strip(); ln = parts[0].strip()
            disp_name = f"{fn} {ln}"
        else:
            disp_name = raw.strip()
        hand = meta.get("hand", "?")
        disp_key = f"{disp_name}  ({'RHP' if hand=='R' else 'LHP'})"
        rev_lower[disp_name.lower()] = disp_key

    roster = {}  # {season: {team: [disp_key, ...]}}

    if pitching_df.empty or "Name" not in pitching_df.columns:
        return roster

    for season, grp in pitching_df.groupby("season"):
        roster[int(season)] = {}
        for _, row in grp.iterrows():
            name_raw = str(row.get("Name", "")).strip()
            team     = str(row.get("Team", "?")).strip()
            # Try direct match
            disp_key = rev_lower.get(name_raw.lower())
            if disp_key is None:
                # Try partial last-name match
                last = name_raw.split()[-1].lower() if name_raw else ""
                first= name_raw.split()[0].lower() if len(name_raw.split()) > 1 else ""
                for k, v in rev_lower.items():
                    if last and last in k and first and first in k:
                        disp_key = v
                        break
            if disp_key:
                roster[int(season)].setdefault(team, [])
                roster[int(season)][team].append(disp_key)

    # Sort each team's list
    for season in roster:
        for team in roster[season]:
            roster[season][team] = sorted(set(roster[season][team]))

    return roster


# ── Sub-zone 2×2 matplotlib chart (inline, no popup) ─────────────────
def draw_subzone_panel(
    raw_df: pd.DataFrame,
    zone: int,
    stat_label: str = "Whiff %",
) -> plt.Figure:
    """
    Draw a 2×2 grid showing the 4 sub-quadrants of one zone.
    Each cell shows the stat value + sample size.
    Used inline — no popups.
    """
    from matplotlib.patches import FancyBboxPatch

    STAT_COL_MAP = {
        "Whiff %":     "whiff_pct",
        "Swing %":     "swing_pct",
        "xwOBA":       "avg_xwoba",
        "Exit Velo":   "avg_ev",
        "Barrel %":    "barrel_pct",
        "Hard Hit %":  "hard_hit_pct",
        "GB %":        "gb_pct",
        "Contact %":   "contact_pct",
        "Launch Angle":"avg_la",
    }
    FMT_MAP = {
        "Whiff %":"{:.1f}%","Swing %":"{:.1f}%","Contact %":"{:.1f}%",
        "xwOBA":"{:.3f}","Exit Velo":"{:.1f}","Launch Angle":"{:.1f}°",
        "Barrel %":"{:.1f}%","Hard Hit %":"{:.1f}%","GB %":"{:.1f}%",
    }
    RANGE_MAP = {
        "Whiff %":(0,70),"Swing %":(10,90),"Contact %":(30,100),
        "xwOBA":(0.10,0.55),"Exit Velo":(70,98),"Launch Angle":(-15,45),
        "Barrel %":(0,25),"Hard Hit %":(0,60),"GB %":(0,80),
    }

    cmap     = sns.color_palette("YlOrRd", as_cmap=True)
    vmin, vmax = RANGE_MAP.get(stat_label, (0, 100))
    fmt      = FMT_MAP.get(stat_label, "{:.1f}")

    bz = raw_df[raw_df["zone"] == zone].copy()
    if bz.empty:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#111621"); ax.axis("off")
        ax.text(0.5, 0.5, f"No data\nfor Zone {zone}",
                ha="center", va="center", color="#8892a4", fontsize=12)
        return fig

    # Classify sub-zones
    bz["sub"] = bz.apply(
        lambda r: classify_subzone(
            float(r["plate_x"]) if not pd.isna(r["plate_x"]) else 0.0,
            float(r["plate_z"]) if not pd.isna(r["plate_z"]) else 2.5,
            zone,
        ),
        axis=1,
    )
    bz["is_swing"]   = bz["description"].isin(SWING_EV).astype(int)
    bz["is_whiff"]   = bz["description"].isin(WHIFF_EV).astype(int)
    bz["is_contact"] = bz["description"].isin(CONTACT_EV).astype(int)
    ls = pd.to_numeric(bz["launch_speed"], errors="coerce")
    la = pd.to_numeric(bz["launch_angle"], errors="coerce")
    bz["is_barrel"]  = ((ls >= 98) & la.between(26, 30)).fillna(False).astype(int)
    bz["is_hh"]      = (ls >= 95).fillna(False).astype(int)
    bz["is_gb"]      = (la < 10).fillna(False).astype(int)

    # Aggregate per sub-zone
       # Aggregate per sub-zone — bezpieczna wersja
    agg_dict = {
        "is_swing":   "count",
        "is_whiff":   "sum",
        "is_contact": "sum",
        "is_barrel":  "sum",
        "is_hh":      "sum",
        "is_gb":      "sum",
    }
    
    # Dodajemy tylko kolumny, które istnieją
    if "launch_speed" in bz.columns:
        agg_dict["launch_speed"] = ["count", "mean"]
    if "launch_angle" in bz.columns:
        agg_dict["launch_angle"] = "mean"
    if "estimated_woba_using_speedangle" in bz.columns:
        agg_dict["estimated_woba_using_speedangle"] = "mean"
    if "hbrk" in bz.columns:
        agg_dict["hbrk"] = "mean"
    elif "pfx_x" in bz.columns:
        bz["hbrk"] = pd.to_numeric(bz["pfx_x"], errors="coerce") * 12
        agg_dict["hbrk"] = "mean"
    if "vbrk" in bz.columns:
        agg_dict["vbrk"] = "mean"
    elif "pfx_z" in bz.columns:
        bz["vbrk"] = pd.to_numeric(bz["pfx_z"], errors="coerce") * 12
        agg_dict["vbrk"] = "mean"

    sg = bz.groupby("sub").agg(agg_dict).reset_index()

    # Spłaszcz kolumny po agregacji (jeśli były tuple)
    sg.columns = [col[0] if isinstance(col, tuple) else col for col in sg.columns]

    # Oblicz rate stats
    sw = sg["is_swing"].replace(0, np.nan)   # swings
    n  = sg["is_swing"].replace(0, np.nan)   # total (is_swing count)
    bt = sg.get("launch_speed_count", sg["is_swing"]).replace(0, np.nan)  # batted balls

    sg["whiff_pct"]   = (sg.get("is_whiff", 0) / sw * 100).round(1)
    sg["swing_pct"]   = (sg["is_swing"] / n  * 100).round(1)
    sg["contact_pct"] = (sg.get("is_contact", 0) / sw * 100).round(1)
    sg["barrel_pct"]  = (sg.get("is_barrel", 0) / bt * 100).round(1)
    sg["hard_hit_pct"]= (sg.get("is_hh", 0) / bt * 100).round(1)
    sg["gb_pct"]      = (sg.get("is_gb", 0) / bt * 100).round(1)

    if "launch_speed_mean" in sg.columns:
        sg["avg_ev"] = sg["launch_speed_mean"].round(1)
    if "launch_angle_mean" in sg.columns:
        sg["avg_la"] = sg["launch_angle_mean"].round(1)
    if "estimated_woba_using_speedangle_mean" in sg.columns:
        sg["avg_xwoba"] = sg["estimated_woba_using_speedangle_mean"].round(3)

    sg = sg.set_index("sub")

    stat_col = STAT_COL_MAP.get(stat_label, "whiff_pct")

    # Layout: TL(0,1) TR(1,1) BL(0,0) BR(1,0)
    positions = {"TL": (0, 1), "TR": (1, 1), "BL": (0, 0), "BR": (1, 0)}

    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    fig.patch.set_facecolor("#0b0f17")
    fig.suptitle(f"Zone {zone} — {stat_label} by Sub-Zone",
                 color="#f0f6ff", fontsize=10, fontweight="700", y=1.01)

    for quad, (col_idx, row_idx) in positions.items():
        ax = axes[1 - row_idx][col_idx]   # flip so TL is top-left
        ax.set_facecolor("#111621")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        if quad in sg.index:
            row = sg.loc[quad]
            val = row.get(stat_col, np.nan)
            n_p = int(row.get("total", 0))
            if not pd.isna(val):
                colour = cmap(np.clip((val - vmin) / (vmax - vmin), 0, 1))
                ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                                           facecolor=colour, edgecolor="#1e2535",
                                           linewidth=2.0))
                ax.text(0.5, 0.60, fmt.format(val),
                        ha="center", va="center", fontsize=15,
                        fontweight="800", color="#111111")
                ax.text(0.5, 0.32, quad,
                        ha="center", va="center", fontsize=9,
                        fontweight="600", color="#333333")
                ax.text(0.5, 0.14, f"n={n_p}",
                        ha="center", va="center", fontsize=7.5,
                        color="#555555")
            else:
                ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                                           facecolor="#1c2230", edgecolor="#1e2535",
                                           linewidth=1.5))
                ax.text(0.5, 0.5, f"{quad}\nno data",
                        ha="center", va="center", fontsize=8, color="#4a5568")
        else:
            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                                       facecolor="#1c2230", edgecolor="#1e2535",
                                       linewidth=1.5))
            ax.text(0.5, 0.5, f"{quad}\nno data",
                    ha="center", va="center", fontsize=8, color="#4a5568")

    plt.tight_layout(pad=0.5)
    return fig


# ── Detailed matchup plan ─────────────────────────────────────────────
def generate_detailed_matchup_plan(
    pitcher_name: str,
    pitcher_hand: str,
    arsenal_df: pd.DataFrame,
    batter_name: str,
    batter_stand: str,
    batter_zone_stats: pd.DataFrame,
    batter_pt_stats: pd.DataFrame,
    batter_count_stats: pd.DataFrame,
) -> str:
    """
    Generate a very detailed, data-driven pitcher-vs-batter attack plan.
    """
    sign   = 1 if pitcher_hand == "R" else -1
    opp    = batter_stand != pitcher_hand   # True = opposite-hand matchup
    b_hand_str = "RHB" if batter_stand == "R" else "LHB"
    p_hand_str = "RHP" if pitcher_hand == "R" else "LHP"
    platoon_lbl = ("✅ Pitcher platoon advantage (same hand)"
                   if not opp else
                   "⚠️ Batter platoon advantage (opposite hand)")

    ZONE_NAMES = {
        1:"Top-Inside",2:"Top-Middle",3:"Top-Outside",
        4:"Mid-Inside", 5:"Center",   6:"Mid-Outside",
        7:"Bot-Inside", 8:"Bot-Middle",9:"Bot-Outside",
        11:"Shadow-Top",12:"Shadow-Right",
        13:"Shadow-Bottom",14:"Shadow-Left",
    }

    html = ['<div class="report-wrap">']
    html.append(
        f'<div style="font-size:1.1rem;font-weight:800;color:#f0f6ff;margin-bottom:4px;">'
        f'⚔️ {pitcher_name} vs {batter_name}</div>'
        f'<div style="color:#8892a4;font-size:.8rem;margin-bottom:12px;">'
        f'{p_hand_str} vs {b_hand_str} · {platoon_lbl}</div>'
    )

    # ── 1. Platoon context ──────────────────────────────────────────
    html.append('<div class="rpt-h2">1. PLATOON CONTEXT & APPROACH</div>')
    if not opp:
        html.append(
            '<div class="ok-box"><div class="rpt-p">'
            f'<b>Same-hand matchup (pitcher advantage).</b> '
            f'{"Slider/sweeper" if pitcher_hand=="R" else "Slider/sweeper"} runs away from the batter — '
            f'primary put-away pitch. Fastball inside establishes plate coverage. '
            f'Changeup/splitter less effective here (runs toward batter — harder to execute). '
            f'Keep the breaking ball away and expand to shadow Zone 12 with it.</div></div>'
        )
    else:
        html.append(
            '<div class="warn-box"><div class="rpt-p">'
            f'<b>Opposite-hand matchup (batter platoon advantage).</b> '
            f'The batter picks up release point more easily. '
            f'Offspeed (changeup/splitter) is critical here — it runs arm-side INTO the batter, '
            f'creating natural deception. Lead with elevated fastball then drop changeup '
            f'to same location. Avoid over-throwing the breaking ball; batter reads it early.</div></div>'
        )

    # ── 2. Pitcher's arsenal vs this batter ────────────────────────
    html.append('<div class="rpt-h2">2. PITCH ARSENAL — WHICH PITCHES TO USE</div>')

    # Cross-reference pitcher's pitches with batter's pitch-type vulnerability
    pt_vuln = {}
    if not batter_pt_stats.empty and "pitch_type" in batter_pt_stats.columns:
        for _, row in batter_pt_stats.iterrows():
            pt_vuln[row["pitch_type"]] = {
                "whiff":  float(row.get("whiff_pct",0) or 0),
                "xwoba":  float(row.get("avg_xwoba",0.320) or 0.320),
                "total":  int(row.get("total",0) or 0),
                "velo":   float(row.get("avg_velo",0) or 0),
            }

    ranked_pitches = []
    for _, r in arsenal_df.iterrows():
        pt    = r["pitch_type"]
        p_wh  = float(r.get("whiff", 0) or 0)
        bvuln = pt_vuln.get(pt, {})
        b_wh  = bvuln.get("whiff", None)
        b_xw  = bvuln.get("xwoba", None)
        n_seen= bvuln.get("total", 0)
        combined = p_wh * 0.6 + (b_wh or 0) * 0.4
        ranked_pitches.append((pt, r, p_wh, b_wh, b_xw, n_seen, combined))

    ranked_pitches.sort(key=lambda x: x[6], reverse=True)

    for rank, (pt, r, p_wh, b_wh, b_xw, n_seen, _) in enumerate(ranked_pitches, 1):
        nm   = r.get("pitch_name", PITCH_LONG.get(pt, pt))
        col  = PITCH_COLORS.get(pt, "#94a3b8")
        av   = MLB_AVG.get(pt, {})
        hd   = r.get("avg_h", 0) - av.get("hbrk", 0)
        use  = float(r.get("usage", 0) or 0)
        box  = "ok-box" if rank == 1 else "insight-box" if rank <= 3 else "ref-card"

        html.append(f'<div class="{box}" style="margin:8px 0;">')
        html.append(
            f'<div style="color:{col};font-weight:700;font-size:.9rem;margin-bottom:5px;">'
            f'#{rank} {nm}  '
            f'<span style="color:#8892a4;font-size:.76rem;font-weight:400;">'
            f'({use:.1f}% usage · {r["avg_velo"]:.1f} mph · {r["avg_h"]:+.1f}" H-break)</span>'
            f'</div>'
        )
        html.append(f'<div class="rpt-p"><b>Pitcher whiff rate:</b> {p_wh:.1f}%</div>')

        if b_wh is not None and n_seen >= 5:
            colour_txt = (
                '<span style="color:#4ade80;font-weight:700;">VULNERABLE</span>'
                if b_wh > p_wh + 5 else
                '<span style="color:#f87171;font-weight:700;">DANGEROUS</span>'
                if b_wh < p_wh - 8 else
                '<span style="color:#fbbf24;">AVERAGE</span>'
            )
            html.append(
                f'<div class="rpt-p"><b>Batter whiff vs this pitch:</b> {b_wh:.1f}% '
                f'({n_seen} seen) — {colour_txt}'
                + (f', xwOBA {b_xw:.3f}' if b_xw else '')
                + '</div>'
            )
        elif n_seen < 5:
            html.append(f'<div class="rpt-p"><b>Batter sample:</b> only {n_seen} pitches seen — use cautiously, data unreliable.</div>')

        # Specific usage recommendation
        if rank == 1:
            html.append(
                f'<div class="rpt-p"><b>✅ Primary out-pitch for this matchup.</b> '
                f'Deploy in 2-strike counts (0-2, 1-2, 2-2) and as the put-away pitch '
                f'vs this specific batter. Aim for shadow Zone 13/14.</div>'
            )
        elif opp and pt in ["CH","FS","SV"]:
            html.append(
                f'<div class="rpt-p"><b>✅ Best pitch for this platoon.</b> '
                f'Arm-side movement runs toward opposite-hand batter — '
                f'natural deception. Use at every opportunity.</div>'
            )
        elif not opp and pt in ["SL","ST","CU","KC"]:
            html.append(
                f'<div class="rpt-p"><b>✅ Best pitch for same-hand matchup.</b> '
                f'Glove-side movement runs away — use liberally in 2-strike counts.</div>'
            )
        elif b_xw is not None and b_xw > 0.380:
            html.append(
                f'<div class="rpt-p"><b>⚠️ Avoid or use sparingly.</b> '
                f'Batter hits this pitch hard (xwOBA {b_xw:.3f}). '
                f'If you use it, execute perfectly (arm-side or shadow only).</div>'
            )

        # Movement advice
        if hd * sign < -3:
            html.append(
                f'<div class="rpt-p"><b>Movement note:</b> '
                f'H-break is {abs(hd):.1f}" below MLB average for this pitch type. '
                f'Consider grip adjustment to add break before relying on it heavily.</div>'
            )
        html.append('</div>')

    # ── 3. Zone attack map ──────────────────────────────────────────
    html.append('<div class="rpt-h2">3. ZONE-BY-ZONE ATTACK STRATEGY</div>')

    if not batter_zone_stats.empty:
        zs = batter_zone_stats.copy()
        # Identify best attack zones (batter whiffs a lot)
        if "whiff_pct" in zs.columns:
            zs_v = zs[zs["zone"].between(1,9)].dropna(subset=["whiff_pct"])
            if not zs_v.empty:
                attack = zs_v.sort_values("whiff_pct", ascending=False).head(3)
                avoid  = zs_v.sort_values("avg_xwoba" if "avg_xwoba" in zs_v.columns else "whiff_pct",
                                           ascending=False).head(2)

                html.append('<div class="rpt-h3">🎯 Attack Zones (batter whiffs most):</div>')
                for _, z in attack.iterrows():
                    zn   = int(z["zone"])
                    wh   = z.get("whiff_pct", 0)
                    xw   = z.get("avg_xwoba", np.nan)
                    xw_s = f", xwOBA {xw:.3f}" if not pd.isna(xw) else ""
                    html.append(
                        f'<div class="rpt-li">• <b>Zone {zn} ({ZONE_NAMES.get(zn,"?")}):</b> '
                        f'{wh:.1f}% whiff{xw_s}. '
                        f'{"Tunnel fastball here then break to shadow." if wh > 30 else "Solid target — work this quadrant consistently."}'
                        f'</div>'
                    )

                html.append('<div class="rpt-h3">🚫 Avoid Zones (batter does damage):</div>')
                for _, z in avoid.iterrows():
                    zn  = int(z["zone"])
                    xw  = z.get("avg_xwoba", np.nan)
                    ev  = z.get("avg_ev", np.nan)
                    html.append(
                        f'<div class="rpt-li">• <b>Zone {zn} ({ZONE_NAMES.get(zn,"?")}):</b> '
                        + (f'xwOBA {xw:.3f}' if not pd.isna(xw) else '')
                        + (f', Exit Velo {ev:.1f} mph' if not pd.isna(ev) else '')
                        + '. Do NOT groove pitches here.</div>'
                    )
    else:
        html.append('<div class="rpt-p"><i>No zone data available for this batter — proceed with general approach.</i></div>')

    # ── 4. Count-by-count plan ──────────────────────────────────────
    html.append('<div class="rpt-h2">4. COUNT-SPECIFIC ATTACK SEQUENCE</div>')

    best_op = ranked_pitches[0][0] if ranked_pitches else "FF"
    best_nm = PITCH_LONG.get(best_op, best_op)
    fb_pt   = next((pt for pt,*_ in ranked_pitches if pt in ["FF","SI","FC"]), "FF")
    fb_nm   = PITCH_LONG.get(fb_pt, fb_pt)
    off_pt  = next((pt for pt,*_ in ranked_pitches if pt in ["CH","FS","SV"]), None)
    off_nm  = PITCH_LONG.get(off_pt, off_pt) if off_pt else "changeup"
    brk_pt  = next((pt for pt,*_ in ranked_pitches if pt in ["SL","ST","CU","KC","SV"]), None)
    brk_nm  = PITCH_LONG.get(brk_pt, brk_pt) if brk_pt else "breaking ball"

    # Get count tendencies
    cnt_map = {}
    if not batter_count_stats.empty and "count_state" in batter_count_stats.columns:
        cnt_map = batter_count_stats.set_index("count_state").to_dict("index")

    def count_note(cnt):
        r = cnt_map.get(cnt, {})
        sw  = r.get("swing_pct", None)
        wh  = r.get("whiff_pct", None)
        xw  = r.get("avg_xwoba", None)
        out = []
        if sw  is not None: out.append(f"batter swings {sw:.0f}%")
        if wh  is not None: out.append(f"whiffs {wh:.0f}%")
        if xw  is not None: out.append(f"xwOBA {xw:.3f}")
        return f" [{', '.join(out)}]" if out else ""

    count_plan = [
        ("0-0",
         f"First-pitch strike is everything. Throw {fb_nm} for called strike (top of zone) "
         f"or {best_nm} for swing-and-miss. Get ahead — it changes the entire at-bat.",
        ),
        ("0-1",
         f"Batter is defensive. Offer {best_nm if not opp else off_nm} just off the zone "
         f"to chase or expand. If they take, you're 0-2 with room to expand further.",
        ),
        ("0-2",
         f"Batter in full defense mode. 3-step sequence: (1) {fb_nm} elevated to freeze, "
         f"(2) {best_nm} same release point breaking to Shadow Zone 13/14. "
         f"Never throw a hittable fastball — batter is guessing fastball.",
        ),
        ("1-0",
         f"Batter may be sitting on fastball. Throw {best_nm if not opp else off_nm} "
         f"to reset expectations. If batter is aggressive, offer {fb_nm} inside then "
         f"{'break away' if not opp else 'fade away'} with the secondary.",
        ),
        ("1-1",
         f"Even count — full arsenal available. Use this to set up 1-2: {fb_nm} in, "
         f"then {best_nm} out. Or reverse to confuse: off-speed first, fastball second.",
        ),
        ("1-2",
         f"Prime put-away count. Lead with elevated {fb_nm} (batter instinctively protects), "
         f"then {best_nm} breaking to Shadow Zone 13 or 14. "
         f"Alternatively: {off_nm if off_pt else brk_nm} at exact same release as the fastball.",
        ),
        ("2-0",
         f"Hitter's count — batter sitting fastball. Surprise with {best_nm if ranked_pitches and ranked_pitches[0][2]>25 else fb_nm} "
         f"for strike. Never walk here — {fb_nm} in zone even if batter is ready.",
        ),
        ("2-1",
         f"Still hitter's count. Must throw a strike. {fb_nm} with purpose — pick a spot "
         f"(away or inside) rather than 'just throw it over'. Avoid giving up power pitch in center.",
        ),
        ("2-2",
         f"Two-strike leverage. Best count to bury batter: {fb_nm} elevated to raise eyes, "
         f"then {best_nm} down and away (same arm-side tunnel). "
         f"Or: {brk_nm} backdoor for called strike if batter is sitting offspeed.",
        ),
        ("3-0",
         f"Take strike. {fb_nm} for called strike — never give free pass. "
         f"Batter knows you must throw a strike; use that expectation by locating perfectly.",
        ),
        ("3-1",
         f"Batter sitting dead-red fastball. This is the only count to consider off-speed: "
         f"{off_nm if off_pt else brk_nm} for strike surprises batter who is timing fastball. "
         f"Otherwise: {fb_nm} located precisely (away vs RHB, inside vs LHB).",
        ),
        ("3-2",
         f"Full count — win the at-bat. {fb_nm} with max velocity and best location "
         f"(away vs. batter's stance). You must throw a strike but make it a GOOD strike. "
         f"Avoid middle-middle {fb_nm} — that becomes a home run.",
        ),
    ]

    for cnt, advice in count_plan:
        cn = count_note(cnt)
        html.append(
            f'<div class="rpt-li">• <b>{cnt}:</b> {advice}{cn}</div>'
        )

    # ── 5. In-game adjustments ───────────────────────────────────────
    html.append('<div class="rpt-h2">5. IN-GAME ADJUSTMENTS</div>')
    html.append(
        f'<div class="insight-box"><div class="rpt-p">'
        f'<b>1st time through:</b> Establish {fb_nm} in/out. Show {best_nm} early '
        f'even if not getting swing-and-miss — the look matters for later PAs.'
        f'</div></div>'
    )
    html.append(
        f'<div class="insight-box"><div class="rpt-p">'
        f'<b>2nd time through:</b> Batter has a read on your {fb_nm} location and {best_nm} break. '
        f'Vary speeds by 3–5 mph. Use {off_nm if off_pt else brk_nm} more — '
        f'especially if batter sat on {fb_nm} in 1st PA.'
        f'</div></div>'
    )
    html.append(
        f'<div class="insight-box"><div class="rpt-p">'
        f'<b>3rd time through:</b> Batter has your patterns. '
        f'Reverse sequences from earlier (lead with the pitch you typically threw 2nd). '
        f'If {best_nm} was your 0-2 pitch all game, use it 0-0 to break the pattern. '
        f'Arm-side location becomes critical.'
        f'</div></div>'
    )

    html.append('</div>')   # close report-wrap
    return "\n".join(html)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — PITCHER SCOUT  (main UI block)
# Replace the entire "with tab_pitcher:" block with this.
# Requires:
#   _pitching_df, _pitcher_meta, _pitcher_disp_list, _pitcher_disp_map
#   _batter_meta, _batter_disp_list, _batter_disp_map, _all_teams
#   ALL_YEARS, ALL_COUNTS, STAT_LABELS, BR_CITATION
# ─────────────────────────────────────────────────────────────────────

# ── Build pitcher team roster (call once near startup) ────────────────
# Add this line near where you build _pitcher_disp_list etc:
#   _pitcher_team_roster = build_pitcher_team_roster(_pitching_df, _pitcher_meta)
# If you already have it, skip the line above.

with tab_pitcher:
    st.markdown(
        '<div class="dash-hdr">'
        '<div class="dash-ttl">🤖 Pitcher Scout</div>'
        '<div class="dash-sub">Team → Season → Pitcher  ·  Full arsenal + zone analysis  ·  Pitcher vs Batter matchup plan</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Step 1: Team + Season + Pitcher selectors ─────────────────────
    st.markdown('<div class="sec-hdr">🔍 Select Pitcher</div>', unsafe_allow_html=True)

    sel_c1, sel_c2, sel_c3 = st.columns([2, 1, 3])
    with sel_c1:
        p_team_sel = st.selectbox(
            "Team",
            ["All Teams"] + sorted(_all_teams),
            key="p_team_sel",
        )
    with sel_c2:
        p_season_sel = st.selectbox(
            "Season",
            list(ALL_YEARS),
            index=1,
            key="p_season_sel",
        )
    with sel_c3:
        # ── FIXED filtering: use pitching_stats to get correct team roster ──
        # Build pitcher team roster if not already in session_state
        if "_pitcher_team_roster" not in st.session_state:
            st.session_state["_pitcher_team_roster"] = build_pitcher_team_roster(
                _pitching_df, _pitcher_meta
            )
        _ptr = st.session_state["_pitcher_team_roster"]

        # Get pitchers for selected team + season
        if p_team_sel == "All Teams":
            # All pitchers who appear in pitching_stats for this season
            season_pitchers_set = set()
            for team_pitchers in _ptr.get(p_season_sel, {}).values():
                season_pitchers_set.update(team_pitchers)
            # Fall back to full list if empty (e.g. year not in pitching_stats)
            available_p = sorted(season_pitchers_set) if season_pitchers_set else _pitcher_disp_list
        else:
            available_p = sorted(
                _ptr.get(p_season_sel, {}).get(p_team_sel, [])
            )
            if not available_p:
                st.caption(f"No pitchers found for {p_team_sel} in {p_season_sel}. Showing all.")
                available_p = _pitcher_disp_list

        ps_choice = st.selectbox(
            "Pitcher",
            available_p if available_p else _pitcher_disp_list,
            key="p_pitcher_sel",
        )

    if st.button("▶️  Load Pitcher Data", key="ps_go2",):
        _pid = _pitcher_disp_map.get(ps_choice)
        if _pid:
            with st.spinner(f"Loading statcast data for {ps_choice}…"):
                _praw = load_pitcher_data(_pid, p_season_sel)
            st.session_state["ps_raw"]  = _praw
            st.session_state["ps_pid"]  = _pid
            st.session_state["ps_year"] = p_season_sel
            st.session_state["ps_name"] = pitcher_display_name(
                _pitcher_meta.get(_pid, {}).get("name", ps_choice)
            )
        else:
            st.warning("Pitcher not found in Statcast data.")

    # ─────────────────────────────────────────────────────────────────
    _ps_raw  = st.session_state.get("ps_raw",  pd.DataFrame())
    _ps_pid  = st.session_state.get("ps_pid",  None)
    _ps_name = st.session_state.get("ps_name", "")
    _ps_year = st.session_state.get("ps_year", p_season_sel)

    if not _ps_raw.empty and _ps_pid is not None:
        _ars, _total, _arm_avg, _ext_avg, _hand = build_pitcher_arsenal(_ps_raw)
        _hand_str = "RHP" if _hand == "R" else "LHP"

        st.markdown(
            f'<div class="sec-hdr">{_ps_name}  ({_hand_str}) — {_ps_year} · {_total:,} pitches</div>',
            unsafe_allow_html=True,
        )

        # ── Arsenal chart + table ─────────────────────────────────────
        fig_ars = plot_arsenal(_ars, _ps_name, _hand)
        st.pyplot(fig_ars, use_container_width=True)
        plt.close(fig_ars)

        # Arsenal table
        _show_cols = [c for c in [
            "pitch_name","pitch_type","usage","avg_velo","max_velo",
            "avg_spin","avg_h","avg_v","avg_ext","avg_arm","whiff",
        ] if c in _ars.columns]
        _tbl = _ars[_show_cols].copy()
        _tbl.columns = [
            c.replace("pitch_name","Pitch").replace("pitch_type","Code")
             .replace("usage","Usage%").replace("avg_velo","AvgV")
             .replace("max_velo","MaxV").replace("avg_spin","Spin")
             .replace("avg_h","HBrk\"").replace("avg_v","VBrk\"")
             .replace("avg_ext","Ext ft").replace("avg_arm","Arm°")
             .replace("whiff","Whiff%")
            for c in _show_cols
        ]
        def _mfmt(f):
            def _fn(x):
                if pd.isna(x): return "—"
                try:    return f.format(x)
                except: return str(x)
            return _fn
        _fmt_d = {"Usage%":_mfmt("{:.1f}%"),"AvgV":_mfmt("{:.1f}"),
                  "MaxV":_mfmt("{:.1f}"),"HBrk\"":_mfmt("{:+.1f}"),
                  "VBrk\"":_mfmt("{:+.1f}"),"Ext ft":_mfmt("{:.2f}"),
                  "Arm°":_mfmt("{:.1f}"),"Whiff%":_mfmt("{:.1f}%")}
        st.dataframe(
            _tbl.style.format({k:v for k,v in _fmt_d.items() if k in _tbl.columns}),
            width="stretch", 
            height=min(380, 60 + len(_ars)*40),
        )

        # ── Zone Heatmap with full filters ────────────────────────────
        st.markdown('<div class="sec-hdr">🗺️ Strike Zone Heatmap</div>', unsafe_allow_html=True)

        fcols = st.columns([2, 1, 1, 1, 1])
        with fcols[0]: hz_stat  = st.selectbox("Statistic",    STAT_LABELS,          key="ps_hstat2")
        with fcols[1]: hz_bh    = st.selectbox("vs Hand",      ["All","R","L"],       key="ps_hbh2")
        with fcols[2]: hz_cnt   = st.selectbox("Count",        ["All"]+ALL_COUNTS,    key="ps_hcnt2")
        with fcols[3]:
            _pt_opts_p = sorted(_ps_raw["pitch_type"].dropna().unique().tolist())
            hz_pt = st.selectbox("Pitch Type", ["All"]+_pt_opts_p, key="ps_hpt2")
        with fcols[4]:
            hz_spin_min, hz_spin_max = st.select_slider(
                "Spin rpm", options=list(range(1000,3600,50)),
                value=(1500, 3200), key="ps_spin2",
            )

        # Movement sliders
        mf1, mf2 = st.columns(2)
        with mf1:
            hz_hbrk = st.slider("H-Break (in)", -22.0, 22.0, (-15.0, 15.0), step=0.5, key="ps_hbrk2")
        with mf2:
            hz_vbrk = st.slider("V-Break (in)", -18.0, 22.0, (-8.0, 18.0), step=0.5, key="ps_vbrk2")

        # Apply filters
        _dff = _ps_raw.copy()
        if hz_bh  != "All": _dff = _dff[_dff["stand"]       == hz_bh]
        if hz_cnt != "All": _dff = _dff[_dff["count_state"] == hz_cnt]
        if hz_pt  != "All": _dff = _dff[_dff["pitch_type"]  == hz_pt]
        if "release_spin_rate" in _dff.columns:
            _spin = pd.to_numeric(_dff["release_spin_rate"], errors="coerce")
            _dff  = _dff[_spin.between(hz_spin_min, hz_spin_max).fillna(True)]
        if "hbrk" in _dff.columns:
            _dff = _dff[_dff["hbrk"].between(hz_hbrk[0], hz_hbrk[1]).fillna(True)]
        if "vbrk" in _dff.columns:
            _dff = _dff[_dff["vbrk"].between(hz_vbrk[0], hz_vbrk[1]).fillna(True)]

        _hz_zdf = compute_zone_stats_from_raw(_dff) if not _dff.empty else pd.DataFrame()

        # Title
        _hz_title = f"{_ps_name} — {hz_stat}"
        if hz_bh  != "All": _hz_title += f" vs {hz_bh}HB"
        if hz_cnt != "All": _hz_title += f" · {hz_cnt}"
        if hz_pt  != "All": _hz_title += f" · {hz_pt}"

        # ── Heatmap + inline sub-zone (no popup) ─────────────────────
        hz_col1, hz_col2 = st.columns([1, 1])

        with hz_col1:
            if not _hz_zdf.empty:
                fig_hz = draw_heatmap(_hz_zdf, hz_stat, _hz_title)
                st.pyplot(fig_hz, use_container_width=True)
                plt.close(fig_hz)
            else:
                st.info("No data for selected filters.")

        with hz_col2:
            # Zone selector for sub-zone panel
            _sz_zone_p = st.selectbox(
                "🔬 Select zone for sub-zone detail:",
                list(range(1, 10)),
                format_func=lambda z: {
                    1:"Zone 1 (Top-In)",2:"Zone 2 (Top-Mid)",3:"Zone 3 (Top-Out)",
                    4:"Zone 4 (Mid-In)",5:"Zone 5 (Center)",6:"Zone 6 (Mid-Out)",
                    7:"Zone 7 (Bot-In)",8:"Zone 8 (Bot-Mid)",9:"Zone 9 (Bot-Out)",
                }.get(z, str(z)),
                key="sz_zone_p2",
            )
            _sz_stat_p = st.selectbox("Sub-zone stat:", STAT_LABELS, key="sz_stat_p2")

            # INLINE sub-zone panel — no button, no popup
            if not _dff.empty:
                fig_sz_p = draw_subzone_panel(_dff, _sz_zone_p, _sz_stat_p)
                st.pyplot(fig_sz_p, use_container_width=True)
                plt.close(fig_sz_p)
            else:
                st.info("No filtered data for sub-zone analysis.")

        # Zone summary table
        if not _hz_zdf.empty:
            with st.expander("📋 Zone Summary Table", expanded=False):
                _hz_sum_cols = [c for c in [
                    "zone","total","whiff_pct","swing_pct","contact_pct",
                    "avg_xwoba","avg_ev","avg_la","barrel_pct","hard_hit_pct","gb_pct",
                ] if c in _hz_zdf.columns]
                st.dataframe(
                    _hz_zdf[_hz_sum_cols].sort_values("zone").reset_index(drop=True),
                    width="stretch", 
                    height=340,
                )

        # ── Full scouting report ──────────────────────────────────────
        st.markdown('<div class="sec-hdr">📋 Full Scouting Report</div>', unsafe_allow_html=True)
        st.markdown(
            generate_pitcher_report(
                _ps_name, _ars, _total, _arm_avg, _ext_avg, _hand,
                f"{_ps_year} Season",
            ),
            unsafe_allow_html=True,
        )

        # ── Download ──────────────────────────────────────────────────
        _rep_txt = re.sub(r"<[^>]+>","",
            generate_pitcher_report(_ps_name,_ars,_total,_arm_avg,_ext_avg,_hand,f"{_ps_year}")
        ).replace("&amp;","&")
        st.download_button(
            "📥  Download Scout Report (.txt)",
            data=_rep_txt,
            file_name=f"{_ps_name.replace(' ','_')}_{_ps_year}_scout.txt",
            mime="text/plain", key="ps_dl2",
        )

        # ════════════════════════════════════════════════════════════════
        # PITCHER vs BATTER MATCHUP SECTION
        # ════════════════════════════════════════════════════════════════
        st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-hdr">⚔️  Pitcher vs. Batter — Detailed Matchup Plans</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Select opposing team and up to 5 batters. "
            "AI generates a detailed attack plan using pitcher's arsenal "
            "AND batter's zone/pitch-type weaknesses."
        )

        vb_c1, vb_c2, vb_c3 = st.columns([1, 1, 2])
        with vb_c1:
            vb_team = st.selectbox("Opposing team", ["—"] + sorted(_all_teams), key="vb_team2")
        with vb_c2:
            vb_season = st.selectbox("Batter data season", list(ALL_YEARS), index=1, key="vb_season2")
        with vb_c3:
            # Filter batters by team + season from batter_meta
            if vb_team == "—":
                _vb_bat_list = _batter_disp_list[:300]
            else:
                _vb_bat_list = [
                    disp for disp, bid in _batter_disp_map.items()
                    if _batter_meta.get(bid, {}).get("team") == vb_team
                ]
                _vb_bat_list = sorted(_vb_bat_list)
                if not _vb_bat_list:
                    st.caption(f"No batters found for {vb_team}. Showing all.")
                    _vb_bat_list = _batter_disp_list[:300]

            vb_batters = st.multiselect(
                "Select batters (max 5)",
                _vb_bat_list,
                max_selections=5,
                key="vb_batters2",
            )

        if st.button("🎯  Generate Matchup Plans", key="vb_go2") and vb_batters:
            _vb_res = {}
            prog_vb = st.progress(0, text="Loading batter data…")
            for i, bdisp in enumerate(vb_batters):
                bid  = _batter_disp_map.get(bdisp)
                bname= _batter_meta.get(bid, {}).get("name", f"Batter #{bid}") if bid else bdisp
                prog_vb.progress((i+1)/len(vb_batters), text=f"Loading {bname}…")
                if bid:
                    with st.spinner(f"Loading {bname}…"):
                        _braw_vb = load_batter_data(bid, vb_season)
                    _bstats_vb = analyze_batter(_braw_vb) if not _braw_vb.empty else {}
                    bstand     = _batter_meta.get(bid, {}).get("stand", "R")
                    _vb_res[bdisp] = (bid, bname, bstand, _bstats_vb)
            prog_vb.empty()
            st.session_state["vb_results2"]  = _vb_res
            st.session_state["vb_ars_cache"] = (_ps_name, _ars, _hand)

        _vb_results2 = st.session_state.get("vb_results2", {})
        _vb_ars_c    = st.session_state.get("vb_ars_cache", None)

        if _vb_results2 and _vb_ars_c:
            vb_pname, vb_ars, vb_hand = _vb_ars_c

            for bdisp, (bid, bname, bstand, bstats) in _vb_results2.items():
                with st.expander(f"⚔️  {vb_pname} vs {bname}", expanded=True):
                    # Quick overview columns
                    ov1, ov2, ov3 = st.columns([1, 1, 2])
                    with ov1:
                        # Batter's zone heatmap (whiff%)
                        _bz_vb = bstats.get("zone_stats", pd.DataFrame())
                        if not _bz_vb.empty:
                            fig_bz_vb = draw_heatmap(
                                _bz_vb, "Whiff %",
                                f"{bname} — Whiff% by Zone",
                                batter_mode=True,
                            )
                            st.pyplot(fig_bz_vb, use_container_width=True)
                            plt.close(fig_bz_vb)
                        else:
                            st.info("No zone data for this batter.")
                    with ov2:
                        # Pitch vulnerability table
                        _bpt_vb = bstats.get("pt_stats", pd.DataFrame())
                        if not _bpt_vb.empty:
                            _bpt_cols = [c for c in [
                                "pitch_name","total","whiff_pct","avg_xwoba","avg_ev","avg_velo",
                            ] if c in _bpt_vb.columns]
                            st.caption(f"**{bname}** — pitch vulnerability")
                            st.dataframe(
                                _bpt_vb[_bpt_cols]
                                .sort_values("whiff_pct", ascending=False, na_position="last")
                                .reset_index(drop=True),
                                width="stretch",
                                height=220,
                            )
                    with ov3:
                        # Quick stat pills
                        _bz_vb2 = bstats.get("zone_stats", pd.DataFrame())
                        if not _bz_vb2.empty:
                            sw_t  = _bz_vb2.get("swings",  pd.Series([0])).sum()
                            wh_t  = _bz_vb2.get("whiffs",  pd.Series([0])).sum()
                            tot_t = _bz_vb2.get("total",   pd.Series([1])).sum()
                            xw_t  = _bz_vb2.get("avg_xwoba",pd.Series([np.nan])).mean()
                            sw_pct = sw_t/max(tot_t,1)*100
                            wh_pct = wh_t/max(sw_t,1)*100
                            plat   = bstats.get("platoon",{})
                            def _pc(lbl,val,sub=""):
                                return (f'<div class="metric-card">'
                                        f'<div class="metric-label">{lbl}</div>'
                                        f'<div class="metric-val">{val}</div>'
                                        f'<div class="metric-sub">{sub}</div></div>')
                            _m = (
                                _pc("Pitches Seen", f"{int(tot_t):,}")
                                + _pc("Swing%", f"{sw_pct:.1f}%")
                                + _pc("Whiff%", f"{wh_pct:.1f}%")
                                + _pc("xwOBA",  f"{xw_t:.3f}" if not np.isnan(xw_t) else "—")
                            )
                            if plat:
                                rh = plat.get("R",{}); lh = plat.get("L",{})
                                _m += _pc(f"vs RHP Whiff", f'{rh.get("whiff_pct",0):.1f}%')
                                _m += _pc(f"vs LHP Whiff", f'{lh.get("whiff_pct",0):.1f}%')
                            st.markdown(f'<div class="metric-grid">{_m}</div>',
                                        unsafe_allow_html=True)

                    # ── DETAILED MATCHUP PLAN ─────────────────────────
                    st.markdown("---")
                    matchup_html = generate_detailed_matchup_plan(
                        pitcher_name      = vb_pname,
                        pitcher_hand      = vb_hand,
                        arsenal_df        = vb_ars,
                        batter_name       = bname,
                        batter_stand      = bstand,
                        batter_zone_stats = bstats.get("zone_stats",  pd.DataFrame()),
                        batter_pt_stats   = bstats.get("pt_stats",    pd.DataFrame()),
                        batter_count_stats= bstats.get("count_stats", pd.DataFrame()),
                    )
                    st.markdown(matchup_html, unsafe_allow_html=True)

                    # Download per-batter plan
                    _plan_txt = re.sub(r"<[^>]+>","",matchup_html).replace("&amp;","&")
                    st.download_button(
                        f"📥 Download {bname} plan",
                        data=_plan_txt,
                        file_name=f"{vb_pname.replace(' ','_')}_vs_{bname.replace(' ','_')}_{vb_season}.txt",
                        mime="text/plain",
                        key=f"vb_dl_{bid}",
                    )

    else:
        st.info("👆  Select a team and pitcher above, then click **Load Pitcher Data**.")

    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 3 — TOP 10 REBUILD
# ─────────────────────────────────────────────────────────────────────
with tab_rebuild:
    st.markdown(
        '<div class="dash-hdr">'
        '<div class="dash-ttl">🏆 Top 10 Arsenal Rebuild Candidates</div>'
        '<div class="dash-sub">All MLB · Baseball Reference stats 2024–2026 · Fixed scoring (ERA-FIP gap correctly weighted)</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="citation" style="margin-bottom:12px;">{BR_CITATION}</div>',
                unsafe_allow_html=True)

    with st.expander("ℹ️  How the rebuild score works", expanded=False):
        st.markdown("""
**Rebuild Score (0–100)** — higher = more improvement potential through arsenal changes.

| Signal | Weight | Notes |
|--------|--------|-------|
| K/9 below 8.5 | up to **28 pts** | Primary arsenal signal |
| BB/9 above 3.0 | up to **20 pts** | Command/arsenal signal |
| FIP above 4.0  | up to **20 pts** | Underlying stuff |
| ERA-FIP gap > 1 **AND** FIP ≥ 4.0 | up to **14 pts** | Both metrics must be bad |
| ERA-FIP gap > 1 **BUT** FIP < 4.0 | only **3 pts** | → Luck/BABIP, not arsenal |
| ERA+ below 90 | up to **8 pts** | |
| HR/9 above 1.6 | up to **10 pts** | |
| Age ≤ 25 | +6 pts | Higher rebuild upside |
| Age ≥ 35 | −8 pts | Lower ceiling |

**Eric Fedde note:** ERA 5.49, FIP 5.20, SO9 5.3, BB9 4.3 → all genuine arsenal issues.
His score is high because K/9 and BB/9 are legitimately bad, **not** just ERA-FIP luck.
        """)

    if _pitching_df.empty:
        st.error("⚠️  No pitching_stats parquet files found.")
    else:
        rb_c = st.columns([1, 1, 2, 1])
        with rb_c[0]: rb_season = st.selectbox("Season", _avail_seasons, index=min(1, len(_avail_seasons)-1), key="rb_yr")
        with rb_c[1]: rb_minip  = st.number_input("Min IP", 5, 150, value=20, step=5, key="rb_ip")
        with rb_c[2]: rb_team   = st.text_input("Team filter (e.g. NYY,LAD — blank=all MLB)", key="rb_team")
        with rb_c[3]: rb_topn   = st.number_input("Show top N", 5, 50, value=10, key="rb_n")

        if st.button("🔄  Rank Pitchers", key="rb_go"):
            _p_df = _pitching_df[_pitching_df["season"] == rb_season].copy()
            if "IP" in _p_df.columns:
                _p_df = _p_df[pd.to_numeric(_p_df["IP"], errors="coerce") >= rb_minip]
            # Filter extreme outliers before scoring
            if "FIP" in _p_df.columns:
                _p_df = _p_df[pd.to_numeric(_p_df["FIP"], errors="coerce") < 15]
            if rb_team.strip():
                _teams = [t.strip().upper() for t in rb_team.split(",")]
                _p_df  = _p_df[_p_df["Team"].str.upper().isin(_teams)]

            if _p_df.empty:
                st.warning("No pitchers match the criteria.")
            else:
                _rows = []
                for _, row in _p_df.iterrows():
                    sc, rsns, prof, only_luck = rebuild_score(row)
                    _rows.append({
                        "name":      row.get("Name", "?"),
                        "team":      row.get("Team", "?"),
                        "ip":        safe_num(row.get("IP", 0)),
                        "era":       safe_num(row.get("ERA")),
                        "fip":       safe_num(row.get("FIP")),
                        "era_p":     safe_num(row.get("ERA+")),
                        "so9":       safe_num(row.get("SO9")),
                        "bb9":       safe_num(row.get("BB9")),
                        "hr9":       safe_num(row.get("HR9")),
                        "whip":      safe_num(row.get("WHIP")),
                        "age":       safe_num(row.get("Age")),
                        "score":     sc,
                        "reasons":   rsns,
                        "profile":   prof,
                        "only_luck": only_luck,
                    })
                _rows.sort(key=lambda x: x["score"], reverse=True)
                st.session_state[f"rb_{rb_season}"] = _rows
                st.success(f"✅  Ranked {len(_rows)} pitchers")

        _rb_results = st.session_state.get(f"rb_{rb_season}")

        if _rb_results:
            _top = _rb_results[:int(rb_topn)]

            def _fv(v, fmt="{:.2f}"):
                return fmt.format(v) if not (isinstance(v, float) and (np.isnan(v) or v <= 0)) else "—"

            # Summary table
            _summ = [{
                "#": i+1, "Name": r["name"], "Team": r["team"],
                "IP": f'{r["ip"]:.0f}',
                "ERA": _fv(r["era"]), "FIP": _fv(r["fip"]),
                "ERA+": _fv(r["era_p"],"{:.0f}"),
                "K/9":  _fv(r["so9"]), "BB/9": _fv(r["bb9"]),
                "WHIP": _fv(r["whip"]), "Age": _fv(r["age"],"{:.0f}"),
                "Score": r["score"],
            } for i, r in enumerate(_top)]
            st.dataframe(pd.DataFrame(_summ).set_index("#"),
                         width="stretch", height=400)

            st.markdown('<div class="sec-hdr">Detailed Breakdown</div>', unsafe_allow_html=True)

            for rank, r in enumerate(_top, 1):
                sc_pct = min(r["score"], 100)
                bc = "#ef4444" if sc_pct>65 else "#f97316" if sc_pct>45 else "#eab308" if sc_pct>30 else "#22c55e"

                with st.expander(
                    f"#{rank}  {r['name']}  ({r['team']})  —  Score: {r['score']:.1f}  |  "
                    f"ERA {_fv(r['era'])}  FIP {_fv(r['fip'])}  "
                    f"ERA+ {_fv(r['era_p'],'{:.0f}')}  "
                    f"K/9 {_fv(r['so9'])}  BB/9 {_fv(r['bb9'])}  IP {r['ip']:.0f}",
                    expanded=(rank <= 3),
                ):
                    # Luck warning
                    if r.get("only_luck"):
                        st.warning(
                            "⚠️  **This pitcher's score is driven mainly by ERA-FIP gap "
                            "despite a solid FIP.** "
                            "This is likely a luck/BABIP issue, NOT a primary arsenal problem. "
                            "Sequencing coaching may help more than arsenal changes."
                        )

                    cl, cr = st.columns([2, 3])
                    with cl:
                        _prof_info = REBUILD_PROFILES.get(r["profile"], {})
                        st.markdown(f"""
                        <div class="ref-card">
                            <div style="font-size:1.8rem;font-weight:800;color:{bc};line-height:1;">
                                {r['score']:.1f}
                                <span style="font-size:.85rem;color:#8892a4;">/100</span>
                            </div>
                            <div class="score-wrap">
                                <div class="score-bar" style="width:{sc_pct}%;background:{bc};"></div>
                            </div>
                            <div style="color:#8892a4;font-size:.76rem;font-weight:600;margin-top:10px;">
                                KEY ISSUES:
                            </div>
                            {"".join(f'<div style="color:#c9d1d9;font-size:.79rem;margin:3px 0;line-height:1.35;">• {rs}</div>' for rs in r["reasons"])}
                            <div style="color:#8892a4;font-size:.73rem;margin-top:8px;">
                                ERA {_fv(r["era"])} · FIP {_fv(r["fip"])} ·
                                ERA+ {_fv(r["era_p"],"{:.0f}")} · K/9 {_fv(r["so9"])} ·
                                BB/9 {_fv(r["bb9"])} · WHIP {_fv(r["whip"])} ·
                                HR/9 {_fv(r["hr9"])} · Age {_fv(r["age"],"{:.0f}")}
                            </div>
                        </div>""", unsafe_allow_html=True)

                        if _prof_info:
                            st.markdown(f"""
                            <div class="ref-card" style="margin-top:0;">
                                <div class="ref-title">🔧 Rebuild Profile</div>
                                <div style="color:#c9d1d9;font-size:.84rem;font-weight:700;margin-bottom:4px;">
                                    {_prof_info.get('title','')}
                                </div>
                                <div style="color:#a0aec0;font-size:.78rem;line-height:1.45;margin-bottom:5px;">
                                    {_prof_info.get('desc','')}
                                </div>
                                <div style="color:#63b3ff;font-size:.78rem;">
                                    <b>Fix:</b> {_prof_info.get('fix','')}
                                </div>
                                <div style="color:#718096;font-size:.73rem;margin-top:3px;">
                                    ⏱ {_prof_info.get('timeline','')}
                                </div>
                            </div>""", unsafe_allow_html=True)

                    with cr:
                        # Contextual stat interpretation
                        st.markdown('<div class="report-wrap">', unsafe_allow_html=True)

                        era  = r["era"]; fip = r["fip"]; so9 = r["so9"]; bb9 = r["bb9"]

                        if not (np.isnan(era) or np.isnan(fip)):
                            gap = era - fip
                            if gap > 1.5 and fip < 4.0:
                                st.markdown(
                                    f'<div class="warn-box"><div class="rpt-p">'
                                    f'🍀 <b>ERA-FIP gap {gap:.2f} with good FIP {fip:.2f}</b> — '
                                    f'This is primarily a luck/BABIP issue. '
                                    f'Sequencing coaching and defensive alignment '
                                    f'may yield faster results than arsenal overhaul.</div></div>',
                                    unsafe_allow_html=True)
                            elif gap > 0.5 and fip >= 4.0:
                                st.markdown(
                                    f'<div class="danger-box"><div class="rpt-p">'
                                    f'🔴 <b>ERA {era:.2f} > FIP {fip:.2f} (+{gap:.2f})</b> — '
                                    f'Both metrics are elevated — genuine stuff and results problem.</div></div>',
                                    unsafe_allow_html=True)
                            elif gap < -0.5:
                                st.markdown(
                                    f'<div class="ok-box"><div class="rpt-p">'
                                    f'✅ <b>ERA below FIP</b> — outperforming underlying metrics. '
                                    f'Build in more swing-and-miss options for sustainable results.</div></div>',
                                    unsafe_allow_html=True)

                        if not np.isnan(so9):
                            if so9 < 7.0:
                                st.markdown(
                                    f'<div class="danger-box"><div class="rpt-p">'
                                    f'🔴 <b>K/9 {so9:.1f} — critically low.</b> '
                                    f'Urgently needs a reliable miss-bat pitch (sweeper, splitter, '
                                    f'high-spin curveball). This is the single most impactful arsenal fix.</div></div>',
                                    unsafe_allow_html=True)
                            elif so9 < 8.5:
                                st.markdown(
                                    f'<div class="warn-box"><div class="rpt-p">'
                                    f'🟡 <b>K/9 {so9:.1f}</b> — below average. '
                                    f'Identify which pitch generates most whiffs and increase its '
                                    f'2-strike usage. Check if best whiff pitch is tunnelled with fastball.</div></div>',
                                    unsafe_allow_html=True)

                        if not np.isnan(bb9) and bb9 > 3.5:
                            st.markdown(
                                f'<div class="warn-box"><div class="rpt-p">'
                                f'🟡 <b>BB/9 {bb9:.1f}</b> — elevated. '
                                f'Focus on 3-1 and 3-2 counts. Develop a reliable zone-attack '
                                f'secondary pitch (backdoor cutter, two-seam fastball arm-side).</div></div>',
                                unsafe_allow_html=True)

                        if not np.isnan(r["age"]):
                            if r["age"] <= 25:
                                st.markdown(
                                    '<div class="insight-box"><div class="rpt-p">'
                                    '🌱 <b>Development phase (≤25):</b> Highest rebuild upside. '
                                    'Arsenal additions and mechanical changes have the best long-term probability of success.</div></div>',
                                    unsafe_allow_html=True)
                            elif r["age"] >= 33:
                                st.markdown(
                                    '<div class="insight-box"><div class="rpt-p">'
                                    '🎓 <b>Veteran phase (≥33):</b> Command and sequencing improvements '
                                    'yield better ROI than velocity-chasing. Grip changes and extension '
                                    'work are most accessible mechanical levers.</div></div>',
                                    unsafe_allow_html=True)

                        st.markdown('</div>', unsafe_allow_html=True)

                    # On-demand Statcast deep-dive
                    st.markdown("---")
                    _rb_parts = r["name"].split()
                    _rb_fn = _rb_parts[0] if _rb_parts else ""
                    _rb_ln = " ".join(_rb_parts[1:]) if len(_rb_parts) > 1 else ""

                    if st.button(f"📡  Load Live Statcast for {r['name']}",
                                 key=f"rb_sc_{rank}_{rb_season}",):
                        # Find pitcher ID from meta map
                        _found_pid = None
                        for _p2id, _p2meta in _pitcher_meta.items():
                            _p2parts = _p2meta["name"].split(",")
                            if len(_p2parts) == 2:
                                _p2fn = _p2parts[1].strip().lower()
                                _p2ln = _p2parts[0].strip().lower()
                                if _rb_ln.lower() in _p2ln and _rb_fn.lower() in _p2fn:
                                    _found_pid = _p2id
                                    break
                        if _found_pid:
                            with st.spinner(f"Loading Statcast data for {r['name']}…"):
                                _rb_raw = load_pitcher_data(_found_pid, rb_season)
                            if not _rb_raw.empty:
                                _rb_ars, _rb_tot, _rb_arm, _rb_ext, _rb_hand = \
                                    build_pitcher_arsenal(_rb_raw)
                                fig_rb = plot_arsenal(_rb_ars, r["name"], _rb_hand)
                                st.pyplot(fig_rb, use_container_width=True)
                                plt.close(fig_rb)
                                st.markdown(
                                    generate_pitcher_report(
                                        r["name"], _rb_ars, _rb_tot,
                                        _rb_arm, _rb_ext, _rb_hand,
                                        f"{rb_season}", age=r["age"],
                                    ),
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.warning("No Statcast data found for this pitcher/season.")
                        else:
                            st.warning(
                                "Could not match pitcher name to Statcast ID. "
                                "Try the Pitcher Scout tab and search manually."
                            )


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — BATTER SCOUT  (complete replacement)
# Paste this block to replace your entire "with tab_batter:" section.
#
# Also add build_batter_team_roster() near build_pitcher_team_roster()
# in your helpers section, and call it once near startup:
#   _batter_team_roster = build_batter_team_roster(...)
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# HELPER — add near build_pitcher_team_roster in your helpers section
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def build_batter_team_roster(
    batter_meta: dict,
    all_years: tuple = (2024, 2025, 2026),
) -> dict:
    """
    Build {season: {team: [display_str, ...]}} for batters.

    Fast path: if meta_batters.parquet exists (from precompute.py), use it.
    Fallback:  scan statcast meta cols (home_team/away_team/inning_topbot)
               for each year — loads only 5 columns, ~0.5 s per year.

    A batter is included for a team+season if they appeared in at least
    one game for that team in that season.
    """
    roster: dict = {}   # {season: {team: [disp_key, ...]}}

    # Build reverse lookup: batter_id → display_str
    rev_map: dict = {}   # {batter_id: disp_key}
    for disp, bid in _batter_disp_map.items():
        rev_map[bid] = disp

    # ── Fast path: precomputed parquet ────────────────────────────────
    meta_path = DATA_DIR / "meta_batters.parquet"
    if meta_path.exists():
        mb = pd.read_parquet(meta_path, engine="pyarrow")
        for _, row in mb.iterrows():
            bid    = int(row["batter_id"])
            season = int(row["season"])
            team   = str(row["team"])
            disp   = rev_map.get(bid)
            if disp:
                roster.setdefault(season, {}).setdefault(team, [])
                if disp not in roster[season][team]:
                    roster[season][team].append(disp)
        for season in roster:
            for team in roster[season]:
                roster[season][team] = sorted(set(roster[season][team]))
        return roster

    # ── Fallback: scan statcast meta columns (monthly files) ──────────
    for yr in all_years:
        if yr not in STATCAST_FILES:
            continue
        parts_bt = []
        for path in STATCAST_FILES[yr]:
            if not path.exists():
                continue
            try:
                meta_cols = ["batter","home_team","away_team","inning_topbot"]
                if PYARROW_OK:
                    sc = set(pq.read_schema(str(path)).names)
                    use_cols = [c for c in meta_cols if c in sc]
                else:
                    use_cols = meta_cols
                df_tmp = pd.read_parquet(path, engine="pyarrow", columns=use_cols)
                parts_bt.append(df_tmp)
            except Exception:
                continue

        if not parts_bt:
            continue
        df = pd.concat(parts_bt, ignore_index=True)
        df["batter_team"] = np.where(
            df["inning_topbot"] == "Top",
            df["away_team"],
            df["home_team"],
        )
        bt = (df[["batter","batter_team"]]
              .dropna(subset=["batter_team"])
              .drop_duplicates())
        for _, row in bt.iterrows():
            bid  = int(row["batter"])
            team = str(row["batter_team"])
            disp = rev_map.get(bid)
            if disp:
                roster.setdefault(yr, {}).setdefault(team, [])
                if disp not in roster[yr][team]:
                    roster[yr][team].append(disp)

    for season in roster:
        for team in roster[season]:
            roster[season][team] = sorted(set(roster[season][team]))
    return roster


# ─────────────────────────────────────────────────────────────────────
# STARTUP CALL — add these lines near where you call
# build_pitcher_team_roster (before the tabs):
#
#   if "_batter_team_roster" not in st.session_state:
#       st.session_state["_batter_team_roster"] = build_batter_team_roster(
#           _batter_meta, ALL_YEARS
#       )
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# TAB 4 — BATTER SCOUT
# ─────────────────────────────────────────────────────────────────────
with tab_batter:
    st.markdown(
        '<div class="dash-hdr">'
        '<div class="dash-ttl">🎯 Batter Scout</div>'
        '<div class="dash-sub">'
        'Team → Season → Batter  ·  Zone heatmap + inline sub-zone detail  ·  '
        'Pitch-type vulnerability (velo · spin · movement)  ·  '
        'Count analysis · Platoon splits · AI pitching plan'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Step 1: Team + Season + Batter selectors ──────────────────────
    st.markdown('<div class="sec-hdr">🔍 Select Batter</div>', unsafe_allow_html=True)

    bs_c1, bs_c2, bs_c3 = st.columns([2, 1, 3])
    with bs_c1:
        bs_team_sel = st.selectbox(
            "Team",
            ["All Teams"] + sorted(_all_teams),
            key="bs_team_sel",
        )
    with bs_c2:
        bs_season_sel = st.selectbox(
            "Season",
            list(ALL_YEARS),
            index=1,
            key="bs_season_sel",
        )
    with bs_c3:
        # Ensure batter team roster is built
        if "_batter_team_roster" not in st.session_state:
            with st.spinner("Building batter team roster…"):
                st.session_state["_batter_team_roster"] = build_batter_team_roster(
                    _batter_meta, ALL_YEARS
                )
        _btr = st.session_state["_batter_team_roster"]

        # ── FIXED filtering: season-specific team roster ──────────────
        if bs_team_sel == "All Teams":
            season_batters_set = set()
            for team_batters in _btr.get(bs_season_sel, {}).values():
                season_batters_set.update(team_batters)
            available_b = sorted(season_batters_set) if season_batters_set else _batter_disp_list
        else:
            available_b = sorted(
                _btr.get(bs_season_sel, {}).get(bs_team_sel, [])
            )
            if not available_b:
                st.caption(f"No batters found for {bs_team_sel} in {bs_season_sel}. Showing all.")
                available_b = _batter_disp_list

        bs_choice = st.selectbox(
            "Batter",
            available_b if available_b else _batter_disp_list,
            key="bs_batter_sel",
        )

    if st.button("▶️  Load Batter Data", key="bs_go2",):
        _bid = _batter_disp_map.get(bs_choice)
        if _bid:
            with st.spinner(f"Loading statcast data for {bs_choice}…"):
                _braw = load_batter_data(_bid, bs_season_sel)
            st.session_state["bs_raw2"]  = _braw
            st.session_state["bs_bid2"]  = _bid
            st.session_state["bs_year2"] = bs_season_sel
            st.session_state["bs_name2"] = _batter_meta.get(_bid, {}).get("name", bs_choice)
        else:
            st.warning("Batter not found.")

    # ─────────────────────────────────────────────────────────────────
    _bs_raw  = st.session_state.get("bs_raw2",  pd.DataFrame())
    _bs_bid  = st.session_state.get("bs_bid2",  None)
    _bs_name = st.session_state.get("bs_name2", "")
    _bs_year = st.session_state.get("bs_year2", bs_season_sel)

    if not _bs_raw.empty and _bs_bid is not None:

        # ── Filters (same depth as Tab 1) ─────────────────────────────
        st.markdown(
            f'<div class="sec-hdr">🎯 {_bs_name} — {_bs_year}</div>',
            unsafe_allow_html=True,
        )

        bf1, bf2, bf3, bf4 = st.columns([1, 1, 2, 2])
        with bf1:
            bs_ph  = st.selectbox("vs Pitcher Hand", ["All","R","L"], key="bs_ph2")
        with bf2:
            bs_cnt = st.selectbox("Count", ["All"] + ALL_COUNTS, key="bs_cnt2")
        with bf3:
            _pt_opts_b = sorted(_bs_raw["pitch_type"].dropna().unique().tolist())
            bs_pt = st.selectbox(
                "Pitch Type",
                ["All"] + _pt_opts_b,
                key="bs_pt2",
            )
        with bf4:
            bs_stat = st.selectbox("Heatmap Statistic", STAT_LABELS, key="bs_stat2")

        # Movement sliders
        mbc1, mbc2 = st.columns(2)
        with mbc1:
            bs_hbrk = st.slider(
                "H-Break of pitches faced (in)",
                -22.0, 22.0, (-18.0, 18.0), step=0.5, key="bs_hbrk2",
            )
        with mbc2:
            bs_velo = st.slider(
                "Velocity faced (mph)",
                60, 105, (70, 102), step=1, key="bs_velo2",
            )

        # ── Apply filters ─────────────────────────────────────────────
        _bff = _bs_raw.copy()
        if bs_ph  != "All": _bff = _bff[_bff["p_throws"]    == bs_ph]
        if bs_cnt != "All": _bff = _bff[_bff["count_state"] == bs_cnt]
        if bs_pt  != "All": _bff = _bff[_bff["pitch_type"]  == bs_pt]
        if "hbrk" in _bff.columns:
            _bff = _bff[_bff["hbrk"].between(bs_hbrk[0], bs_hbrk[1]).fillna(True)]
        if "release_speed" in _bff.columns:
            _bff = _bff[
                pd.to_numeric(_bff["release_speed"], errors="coerce")
                .between(bs_velo[0], bs_velo[1]).fillna(True)
            ]

        # Run analysis
        _bstats = analyze_batter(_bff) if not _bff.empty else {}

        if not _bstats:
            st.warning("No data for the current filter combination.")
        else:
            _bzdf = _bstats.get("zone_stats",  pd.DataFrame())
            _bsub = _bstats.get("sub_stats",   pd.DataFrame())
            _bpt  = _bstats.get("pt_stats",    pd.DataFrame())
            _bcnt = _bstats.get("count_stats", pd.DataFrame())
            _bplt = _bstats.get("platoon",     {})
            total_p = _bstats.get("total_pitches", 0)

            # ── Top-level metrics ─────────────────────────────────────
            sw_all = (
                _bzdf["swings"].sum() / max(_bzdf["total"].sum(), 1) * 100
                if not _bzdf.empty and "swings" in _bzdf.columns else np.nan
            )
            wh_all = (
                _bzdf["whiffs"].sum() / max(_bzdf["swings"].sum(), 1) * 100
                if not _bzdf.empty and "whiffs" in _bzdf.columns else np.nan
            )
            xw_all = _bzdf["avg_xwoba"].mean() if not _bzdf.empty and "avg_xwoba" in _bzdf.columns else np.nan
            ev_all = _bzdf["avg_ev"].mean()    if not _bzdf.empty and "avg_ev"    in _bzdf.columns else np.nan

            def _mc(lbl, val, sub=""):
                return (
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{lbl}</div>'
                    f'<div class="metric-val">{val}</div>'
                    f'<div class="metric-sub">{sub}</div></div>'
                )

            _m_html = (
                _mc("Pitches Seen", f"{total_p:,}", f"{_bs_year}")
                + _mc("Swing %",   f"{sw_all:.1f}%" if not np.isnan(sw_all) else "—")
                + _mc("Whiff %",   f"{wh_all:.1f}%" if not np.isnan(wh_all) else "—", "on swings")
                + _mc("xwOBA",     f"{xw_all:.3f}"  if not np.isnan(xw_all) else "—", "expected")
                + _mc("Exit Velo", f"{ev_all:.1f}"  if not np.isnan(ev_all) else "—", "mph")
            )
            # Platoon context
            plat = _bstats.get("platoon", {})
            if plat.get("R") and plat.get("L"):
                rh_w = plat["R"].get("whiff_pct", 0)
                lh_w = plat["L"].get("whiff_pct", 0)
                _m_html += _mc("vs RHP Whiff", f"{rh_w:.1f}%")
                _m_html += _mc("vs LHP Whiff", f"{lh_w:.1f}%")

            st.markdown(f'<div class="metric-grid">{_m_html}</div>',
                        unsafe_allow_html=True)

            # ── Zone Heatmap + Sub-zone panel (side by side, inline) ──
            st.markdown('<div class="sec-hdr">🗺️ Strike Zone Heatmap + Sub-Zone Detail</div>',
                        unsafe_allow_html=True)

            # Zone selector for sub-zone detail
            sz_zone_b = st.selectbox(
                "🔬 Select zone for sub-zone breakdown (1–9):",
                list(range(1, 10)),
                format_func=lambda z: {
                    1: "Zone 1 — Top-Inside",  2: "Zone 2 — Top-Middle",
                    3: "Zone 3 — Top-Outside", 4: "Zone 4 — Mid-Inside",
                    5: "Zone 5 — Center",      6: "Zone 6 — Mid-Outside",
                    7: "Zone 7 — Bot-Inside",  8: "Zone 8 — Bot-Middle",
                    9: "Zone 9 — Bot-Outside",
                }.get(z, str(z)),
                key="bs_sz_zone2",
            )
            sz_stat_b = st.selectbox(
                "Sub-zone stat:",
                STAT_LABELS,
                key="bs_sz_stat2",
            )

            col_hm_b, col_sz_b = st.columns([1, 1], gap="medium")

            with col_hm_b:
                _bs_title = f"{_bs_name} — {_bs_year} · {bs_stat}"
                if bs_ph  != "All": _bs_title += f" vs {bs_ph}HP"
                if bs_cnt != "All": _bs_title += f" · {bs_cnt}"
                if bs_pt  != "All": _bs_title += f" · {bs_pt}"
                fig_bs_hm = draw_heatmap(
                    _bzdf, bs_stat, _bs_title,
                    batter_mode=True,
                )
                st.pyplot(fig_bs_hm, use_container_width=True)
                plt.close(fig_bs_hm)

            with col_sz_b:
                # ── INLINE sub-zone panel — no button, no popup ───────
                st.markdown(
                    f'<div style="color:#79b8ff;font-weight:600;font-size:.9rem;'
                    f'margin-bottom:8px;">Zone {sz_zone_b} — {sz_stat_b} by Quadrant</div>',
                    unsafe_allow_html=True,
                )
                if not _bff.empty:
                    fig_sz_b = draw_subzone_panel(_bff, sz_zone_b, sz_stat_b)
                    st.pyplot(fig_sz_b, use_container_width=True)
                    plt.close(fig_sz_b)

                    # Sub-zone quick stats below the chart
                    _bff_zone = _bff[_bff["zone"] == sz_zone_b].copy()
                    if not _bff_zone.empty and "plate_x" in _bff_zone.columns:
                        _bff_zone["sub"] = _bff_zone.apply(
                            lambda r: classify_subzone(
                                float(r["plate_x"]) if not pd.isna(r["plate_x"]) else 0.0,
                                float(r["plate_z"]) if not pd.isna(r["plate_z"]) else 2.5,
                                sz_zone_b,
                            ), axis=1,
                        )
                        _sz_tbl = _bff_zone.groupby("sub").agg(
                            Pitches = ("is_swing",  "count"),
                            Swing_p = ("is_swing",  "mean"),
                            Whiff_p = ("is_whiff",  "mean"),
                            xwOBA   = ("estimated_woba_using_speedangle","mean"),
                            EV      = ("launch_speed","mean"),
                        ).reset_index()
                        _sz_tbl["Swing%"] = (_sz_tbl["Swing_p"]*100).round(1)
                        _sz_tbl["Whiff%"] = (_sz_tbl["Whiff_p"]*100).round(1)
                        _sz_tbl["xwOBA"]  = _sz_tbl["xwOBA"].round(3)
                        _sz_tbl["EV"]     = _sz_tbl["EV"].round(1)
                        _sz_tbl = _sz_tbl[["sub","Pitches","Swing%","Whiff%","xwOBA","EV"]]
                        _sz_tbl.columns = ["Quadrant","Pitches","Swing%","Whiff%","xwOBA","Exit V"]
                        st.dataframe(
                            _sz_tbl.set_index("Quadrant"),
                            width="stretch",
                            height=190,
                        )
                else:
                    st.info("No data for selected filters.")

            # ── Pitch-Type Vulnerability (detailed) ───────────────────
            st.markdown(
                '<div class="sec-hdr">🎯 Pitch-Type Vulnerability — Velo · Spin · Movement · Whiff</div>',
                unsafe_allow_html=True,
            )
            if not _bpt.empty:
                _bpt_cols = [c for c in [
                    "pitch_name","pitch_type","usage","total",
                    "whiff_pct","swing_pct","avg_xwoba","avg_ev",
                    "avg_velo","avg_spin","avg_hbrk","avg_vbrk",
                    "avg_ext","avg_arm",
                ] if c in _bpt.columns]
                _bpt_disp = _bpt[_bpt_cols].copy()
                _col_rn = {
                    "pitch_name":"Pitch",    "pitch_type":"Code",
                    "usage":"Seen%",         "total":"Pitches",
                    "whiff_pct":"Whiff%",    "swing_pct":"Swing%",
                    "avg_xwoba":"xwOBA",     "avg_ev":"Exit V",
                    "avg_velo":"Avg Velo",   "avg_spin":"Spin rpm",
                    "avg_hbrk":"H-Brk\"",    "avg_vbrk":"V-Brk\"",
                    "avg_ext":"Ext ft",      "avg_arm":"Arm°",
                }
                _bpt_disp.rename(columns=_col_rn, inplace=True)

                def _sf(f):
                    def _fn(x):
                        if pd.isna(x): return "—"
                        try:    return f.format(x)
                        except: return str(x)
                    return _fn

                _bpt_fmt = {
                    "Seen%":  _sf("{:.1f}%"), "Whiff%": _sf("{:.1f}%"),
                    "Swing%": _sf("{:.1f}%"), "xwOBA":  _sf("{:.3f}"),
                    "Exit V": _sf("{:.1f}"),  "Avg Velo":_sf("{:.1f}"),
                    "Spin rpm":_sf("{:.0f}"), "H-Brk\"": _sf("{:+.1f}"),
                    "V-Brk\"": _sf("{:+.1f}"),"Ext ft":  _sf("{:.2f}"),
                    "Arm°":   _sf("{:.1f}"),
                }
                _sort_col = "Whiff%" if "Whiff%" in _bpt_disp.columns else _bpt_disp.columns[0]
                st.dataframe(
                    _bpt_disp
                    .sort_values(_sort_col, ascending=False, na_position="last")
                    .reset_index(drop=True)
                    .style.format({k:v for k,v in _bpt_fmt.items() if k in _bpt_disp.columns}),
                    width="stretch",
                    height=min(420, 60 + len(_bpt_disp)*40),
                )

                     # ── Visual pitch breakdown bars ───────────────────────────────
        if not _bpt.empty and "whiff_pct" in _bpt.columns:
            _bpt_sorted = _bpt.dropna(subset=["whiff_pct"]).sort_values(
                "whiff_pct", ascending=False)
            if not _bpt_sorted.empty:
                fig_pt, ax_pt = plt.subplots(figsize=(10, 3.5))
                fig_pt.patch.set_facecolor("#0b0f17")
                ax_pt.set_facecolor("#111621")
                _colors_pt = [PITCH_COLORS.get(pt,"#94a3b8") for pt in _bpt_sorted["pitch_type"]]
                _labels_pt = _bpt_sorted.get("pitch_name", _bpt_sorted["pitch_type"])
                bars_pt = ax_pt.barh(_labels_pt, _bpt_sorted["whiff_pct"],
                                     color=_colors_pt, edgecolor="#1e2535", height=0.65)
                for bar, val in zip(bars_pt, _bpt_sorted["whiff_pct"]):
                    ax_pt.text(val+0.8, bar.get_y()+bar.get_height()/2,
                               f"{val:.1f}%", va="center", fontsize=8.5,
                               fontweight="bold", color="#e2e8f0")
                ax_pt.set_xlabel("Whiff % (higher = more vulnerable)",
                                 color="#a0aec0", fontsize=9)
                ax_pt.invert_yaxis()
                ax_pt.tick_params(colors="#8892a4", labelsize=8.5)
                ax_pt.set_xlim(0, _bpt_sorted["whiff_pct"].max()+18)
                for sp in ax_pt.spines.values(): sp.set_edgecolor("#2a3545")
                st.pyplot(fig_pt, use_container_width=True)
                plt.close(fig_pt)
        else:
            st.info("No pitch-type data for current filters.")  

            # ── Count Breakdown ────────────────────────────────────────
            st.markdown('<div class="sec-hdr">📊 Count-Based Tendencies</div>',
                        unsafe_allow_html=True)
            if not _bcnt.empty:
                # Rename for display
                _bcnt_disp = _bcnt.rename(columns={
                    "count_state":"Count","pitches":"Pitches",
                    "swing_pct":"Swing%","whiff_pct":"Whiff%","avg_xwoba":"xwOBA",
                })
                # Visual: swing% and whiff% by count
                cnt_sorted = ALL_COUNTS
                _bcnt_indexed = _bcnt_disp.set_index("Count") if "Count" in _bcnt_disp.columns else _bcnt_disp
                fig_cnt, axes_cnt = plt.subplots(1, 2, figsize=(12, 3.5))
                fig_cnt.patch.set_facecolor("#0b0f17")
                for ax_c, metric, color in zip(
                    axes_cnt,
                    ["Swing%", "Whiff%"],
                    ["#2f7cf6", "#ef4444"],
                ):
                    ax_c.set_facecolor("#111621")
                    cnts_avail = [c for c in cnt_sorted if c in _bcnt_indexed.index]
                    vals_c = [
                        safe_num(_bcnt_indexed.loc[c, metric])
                        if c in _bcnt_indexed.index and metric in _bcnt_indexed.columns
                        else 0
                        for c in cnts_avail
                    ]
                    ax_c.bar(cnts_avail, vals_c, color=color, alpha=0.85, edgecolor="#1e2535")
                    for x, v in zip(cnts_avail, vals_c):
                        ax_c.text(x, v+0.8, f"{v:.0f}%", ha="center", fontsize=7.5,
                                  color="#e2e8f0", fontweight="600")
                    ax_c.set_title(metric, color="#c9d1d9", fontsize=9, fontweight="600")
                    ax_c.tick_params(colors="#8892a4", labelsize=7.5, rotation=45)
                    ax_c.set_ylim(0, max(vals_c)*1.2+5 if vals_c else 100)
                    for sp in ax_c.spines.values(): sp.set_edgecolor("#2a3545")
                fig_cnt.suptitle(f"{_bs_name} — Tendencies by Count",
                                  color="#f0f6ff", fontsize=9, fontweight="700")
                plt.tight_layout()
                st.pyplot(fig_cnt, use_container_width=True)
                plt.close(fig_cnt)

                # Table as well
                with st.expander("📋 Count table", expanded=False):
                    st.dataframe(
                        _bcnt_disp.sort_values("Count").reset_index(drop=True),
                        width="stretch", 
                        height=350,
                    )

            # ── Platoon Splits ─────────────────────────────────────────
            st.markdown('<div class="sec-hdr">🔄 Platoon Splits</div>',
                        unsafe_allow_html=True)
            if _bplt:
                _plt_rows = []
                for h, d in sorted(_bplt.items()):
                    _plt_rows.append({
                        "vs":     "RHP" if h=="R" else "LHP",
                        "Pitches":int(d.get("pitches", 0)),
                        "Swing%": round(d.get("swing_pct", 0), 1),
                        "Whiff%": round(d.get("whiff_pct", 0), 1),
                        "xwOBA":  round(d.get("avg_xwoba", np.nan) or np.nan, 3),
                    })
                _plt_df = pd.DataFrame(_plt_rows).set_index("vs")
                st.dataframe(
                    _plt_df.style.format({
                        "Swing%":"{:.1f}%","Whiff%":"{:.1f}%","xwOBA":"{:.3f}",
                    }),
                    width="stretch", 
                    height=120,
                )
                # Interpretation
                if "R" in _bplt and "L" in _bplt:
                    rh_xw = _bplt["R"].get("avg_xwoba", 0.320) or 0.320
                    lh_xw = _bplt["L"].get("avg_xwoba", 0.320) or 0.320
                    rh_wh = _bplt["R"].get("whiff_pct", 0)
                    lh_wh = _bplt["L"].get("whiff_pct", 0)
                    if abs(rh_xw - lh_xw) > 0.030 or abs(rh_wh - lh_wh) > 5:
                        better  = "RHP" if rh_xw < lh_xw else "LHP"
                        st.markdown(
                            f'<div class="insight-box"><div class="rpt-p">'
                            f'<b>Platoon edge:</b> {better} has an advantage. '
                            f'xwOBA {rh_xw:.3f} vs RHP, {lh_xw:.3f} vs LHP — '
                            f'{"use a RHP" if better=="RHP" else "use a LHP"} '
                            f'in high-leverage spots against this batter.</div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="ok-box"><div class="rpt-p">'
                            'No significant platoon split — batter performs similarly vs RHP and LHP.</div></div>',
                            unsafe_allow_html=True,
                        )

            # ── Zone Summary Table ─────────────────────────────────────
            if not _bzdf.empty:
                with st.expander("📋 Full Zone-by-Zone Summary", expanded=False):
                    _bz_cols = [c for c in [
                        "zone","total","swing_pct","whiff_pct","contact_pct",
                        "avg_xwoba","avg_ev","avg_la","barrel_pct","hard_hit_pct","gb_pct",
                    ] if c in _bzdf.columns]
                    _bz_rn = {
                        "total":"Pitches","swing_pct":"Swing%","whiff_pct":"Whiff%",
                        "contact_pct":"Contact%","avg_xwoba":"xwOBA",
                        "avg_ev":"Exit V","avg_la":"Launch°",
                        "barrel_pct":"Barrel%","hard_hit_pct":"Hard Hit%","gb_pct":"GB%",
                    }
                    st.dataframe(
                        _bzdf[_bz_cols].rename(columns=_bz_rn)
                        .sort_values("zone").reset_index(drop=True),
                        width="stretch", 
                        height=400,
                    )

            # ── AI Pitching Plan ───────────────────────────────────────
            st.markdown('<div class="sec-hdr">🤖 AI Pitching Plan — How to Attack {}</div>'.format(_bs_name),
                        unsafe_allow_html=True)
            _plan_html = generate_pitching_plan(_bstats, _bs_name)
            st.markdown(_plan_html, unsafe_allow_html=True)

            # Download
            _plan_txt = re.sub(r"<[^>]+>","",_plan_html).replace("&amp;","&")
            st.download_button(
                "📥  Download Pitching Plan (.txt)",
                data=_plan_txt,
                file_name=f"{_bs_name.replace(' ','_')}_{_bs_year}_pitching_plan.txt",
                mime="text/plain",
                key="bs_dl2",
            )

    else:
        st.info(
            "👆  Select a team, season, and batter above, "
            "then click **Load Batter Data** to see the full analysis."
        )

    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# 19. GLOBAL FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:14px;">
    <div class="citation">
        MLB Statcast Pro · Built with Streamlit + pyarrow<br>
        {BR_CITATION}
    </div>
    <div class="citation" style="text-align:right;">
        <b>Emergent improvement tips (10 credits):</b><br>
        1. (2cr) Replace matplotlib with Plotly — true click-to-select zones<br>
        2. (2cr) Year-over-year trend sparklines per pitcher/batter<br>
        3. (2cr) Side-by-side player comparison mode<br>
        4. (2cr) Export: heatmap PNG + zone stats CSV button<br>
        5. (2cr) Mobile-first responsive layout (accordion sidebar)
    </div>
</div>
""", unsafe_allow_html=True)

