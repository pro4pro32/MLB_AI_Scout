# ═══════════════════════════════════════════════════════════════════
# AI3.py  —  MLB Statcast Pro Dashboard  (pybaseball edition, v2)
#
# Zero local parquet files. All data pulled on-demand from
# Baseball Savant / FanGraphs (via `pybaseball`), cached with
# @st.cache_data.
#
# v2 additions:
#   • Team + Season roster filtering (via FanGraphs season leaderboards)
#     instead of free-text name search only
#   • Velocity / Spin rate / H-Break / V-Break sliders on EVERY heatmap
#     (Dashboard, Pitcher Scout, Batter Scout)
#   • Deeper AI scout: tunneling-pair detection, priority action items,
#     richer pitcher-vs-batter matchup plan (count-by-count sequencing)
#   • Fixed ERA+/ERA- handling (FanGraphs exposes "ERA-", not "ERA+")
#
# Tabs:
#   1. 📊 Dashboard      — league/team zone heatmap for a date range
#   2. 🤖 Pitcher Scout  — Team→Season→Pitcher, full arsenal + zone
#                          report, matchup vs a specific batter
#   3. 🏆 Top 10 Rebuild — FanGraphs season pitching stats, rebuild score
#   4. 🎯 Batter Scout   — Team→Season→Batter, zone/pitch-type
#                          vulnerability, AI pitching plan
# ═══════════════════════════════════════════════════════════════════

import re
import warnings
from collections import Counter
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MLB Statcast Pro", page_icon="⚾",
                    layout="wide", initial_sidebar_state="collapsed")

try:
    import pybaseball as pyb
    pyb.cache.enable()
    PYB_OK = True
except ImportError:
    PYB_OK = False

if not PYB_OK:
    st.error(
        "⚠️  `pybaseball` is not installed. Add it to requirements.txt "
        "(`pip install pybaseball`) and redeploy."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ══════════════════════════════════════════════════════════════════════

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

ALL_COUNTS = ["0-0","0-1","0-2","1-0","1-1","1-2","2-0","2-1","2-2","3-0","3-1","3-2"]

REBUILD_PROFILES = {
    "low_k": {
        "title": "🎯 Low Strikeout Rate — Missing Out-Pitch",
        "desc":  "K/9 well below league average (~9.0). Relies on contact management without a reliable miss-bat weapon.",
        "fix":   "Add sweeper or high-spin curveball; increase 2-strike usage of best whiff pitch.",
        "timeline": "1–2 seasons",
    },
    "high_bb": {
        "title": "⚠️ Command Issue — High Walk Rate",
        "desc":  "BB/9 elevated above 3.5. Struggles in hitter counts — lacks a reliable strike-throwing secondary.",
        "fix":   "Develop a backdoor breaking ball or cutter for zone-attack in hitter counts.",
        "timeline": "1 spring training",
    },
    "poor_fip": {
        "title": "🔧 Underlying Stuff Needs Work",
        "desc":  "FIP above 4.50 — pitch quality is genuinely below average.",
        "fix":   "Movement optimisation (grip/seam), extension mechanics work, arsenal diversification.",
        "timeline": "Off-season + spring training",
    },
    "hr_suppression": {
        "title": "💣 Home Run Suppression Needed",
        "desc":  "HR/9 above 1.8. Likely pitching up in the zone or lacking sink/cut.",
        "fix":   "Add sinker or splitter for ground-ball induction; work on extension.",
        "timeline": "Off-season mechanical work",
    },
    "era_fip_gap": {
        "title": "🍀 Luck / BABIP — Not Primarily an Arsenal Issue",
        "desc":  "ERA significantly above FIP despite acceptable K/9, BB/9. Likely bad luck on batted balls.",
        "fix":   "Review pitch sequencing and first-pitch approach — arsenal isn't the primary problem.",
        "timeline": "Immediate (coaching / sequencing review)",
    },
}

BR_CITATION = (
    "Statcast data via Baseball Savant (pulled live with `pybaseball`). "
    "Season stats & rosters via FanGraphs (`pybaseball.pitching_stats` / `batting_stats`)."
)


# ══════════════════════════════════════════════════════════════════════
# 2. CSS  (dark theme)
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [data-testid="stAppViewContainer"] { background:#0b0f17 !important; color:#e2e8f0; font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#111621 !important; border-right:1px solid #1e2535; }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
p,div,label,span,li,td,th,h1,h2,h3,h4,.stMarkdown,[data-testid="stMarkdownContainer"] * { color:#e2e8f0; }
.stSelectbox label,.stMultiSelect label,.stTextInput label,.stNumberInput label,.stSlider label,.stRadio label,
div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] label, div[data-testid="stWidgetLabel"] span {
    color:#e2e8f0 !important; font-size:.82rem !important; font-weight:500 !important; }
.stSelectbox > div > div, .stMultiSelect > div > div { background:#111621 !important; border-color:#2a3545 !important; color:#e2e8f0 !important; }
[data-baseweb="select"] [data-testid="stMarkdownContainer"], [data-baseweb="select"] span, [data-baseweb="select"] div { color:#e2e8f0 !important; }
[data-baseweb="menu"], [data-baseweb="menu"] * { background:#111621 !important; color:#e2e8f0 !important; }
[data-baseweb="option"]:hover, [data-baseweb="option"][aria-selected="true"] { background:#1a3a66 !important; }
[data-baseweb="select"] input, [data-baseweb="input"] input { color:#e2e8f0 !important; caret-color:#e2e8f0 !important; }
[data-baseweb="tag"] { background:#1a3a66 !important; border:1px solid #2f7cf6 !important; }
[data-baseweb="tag"] span { color:#79b8ff !important; }
.stTextInput input, .stNumberInput input { background:#111621 !important; border-color:#2a3545 !important; color:#e2e8f0 !important; }
.stSlider [data-baseweb="thumb"] { background:#2f7cf6 !important; border-color:#2f7cf6 !important; }
.stSlider [data-baseweb="track-fill"] { background:#2f7cf6 !important; }
details > summary { background:#111621 !important; border:1px solid #1e2535 !important; border-radius:8px !important; color:#e2e8f0 !important; font-weight:500 !important; padding:10px 16px !important; }
details[open] > summary { border-radius:8px 8px 0 0 !important; }
details > div { background:#0b0f17 !important; border:1px solid #1e2535 !important; border-top:none !important; border-radius:0 0 8px 8px !important; padding:14px !important; }
[data-testid="stDataFrame"] { border:1px solid #1e2535 !important; border-radius:8px !important; }
.stButton > button { background:#181f2e; border:1px solid #2a3545; color:#e2e8f0 !important; border-radius:7px; font-weight:500; }
.stButton > button:hover { background:#1e2535; border-color:#2f7cf6; color:#79b8ff !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { background:#111621 !important; border-bottom:2px solid #1e2535; gap:4px; padding:0 8px; }
[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent !important; color:#8892a4 !important; font-weight:500 !important; padding:10px 20px !important; border-radius:8px 8px 0 0 !important; }
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] { background:#1a3a6655 !important; color:#79b8ff !important; border-bottom:2px solid #2f7cf6 !important; font-weight:600 !important; }
.dash-hdr { padding:18px 0 4px; border-bottom:1px solid #1e2535; margin-bottom:20px; }
.dash-ttl { font-size:1.75rem; font-weight:700; color:#f0f6ff !important; letter-spacing:-.4px; }
.dash-sub { font-size:.82rem; color:#7a8494 !important; margin-top:4px; }
.sec-hdr { display:flex; align-items:center; gap:10px; background:linear-gradient(90deg,#1a3a6622,transparent); border-left:3px solid #2f7cf6; padding:10px 18px; border-radius:0 8px 8px 0; margin:24px 0 16px; color:#79b8ff !important; font-weight:600; }
.ref-card { background:#111621; border:1px solid #1e2535; border-radius:10px; padding:14px 16px; margin-bottom:10px; }
.ref-title { color:#79b8ff; font-weight:600; font-size:.83rem; margin-bottom:8px; }
.ref-badge { display:inline-block; background:#181f2e; border:1px solid #2a3545; border-radius:5px; padding:3px 9px; margin:2px 2px 4px 0; font-size:.73rem; color:#63b3ff; font-family:'JetBrains Mono',monospace; }
.metric-grid { display:flex; gap:10px; flex-wrap:wrap; margin:12px 0; }
.metric-card { background:#111621; border:1px solid #1e2535; border-radius:9px; padding:11px 14px; flex:1; min-width:110px; text-align:center; }
.metric-label { color:#6e7a8a; font-size:.68rem; text-transform:uppercase; letter-spacing:.8px; margin-bottom:5px; }
.metric-val { color:#e2e8f0; font-size:1.35rem; font-weight:700; }
.metric-sub { color:#5a6478; font-size:.7rem; margin-top:3px; }
.report-wrap { background:#0d1220; border:1px solid #2a3545; border-radius:12px; padding:22px; margin:10px 0; }
.rpt-h2 { color:#79b8ff; font-size:.98rem; font-weight:700; margin:18px 0 7px; padding-bottom:5px; border-bottom:1px solid #1e2535; }
.rpt-h3 { color:#63b3ff; font-size:.88rem; font-weight:600; margin:12px 0 5px; }
.rpt-p { color:#c9d1d9; font-size:.85rem; line-height:1.75; margin:5px 0; }
.rpt-li { color:#c9d1d9; font-size:.85rem; line-height:1.7; margin:3px 0 3px 16px; }
.pill { display:inline-block; border-radius:5px; padding:2px 9px; margin:1px 2px; font-size:.73rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.pill-elite { background:#14532d; color:#4ade80; border:1px solid #16a34a; }
.pill-above { background:#1e3a5f; color:#60a5fa; border:1px solid #2563eb; }
.pill-avg   { background:#292524; color:#a8a29e; border:1px solid #57534e; }
.pill-below { background:#4a3200; color:#fbbf24; border:1px solid #d97706; }
.pill-poor  { background:#450a0a; color:#f87171; border:1px solid #dc2626; }
.insight-box { background:#0f1e35; border:1px solid #2f7cf6; border-radius:8px; padding:11px 15px; margin:9px 0; }
.insight-box .rpt-p { color:#93c5fd; }
.warn-box   { background:#1f1000; border:1px solid #d97706; border-radius:8px; padding:11px 15px; margin:9px 0; }
.warn-box .rpt-p   { color:#fde68a; }
.danger-box { background:#1f0000; border:1px solid #dc2626; border-radius:8px; padding:11px 15px; margin:9px 0; }
.danger-box .rpt-p { color:#fca5a5; }
.ok-box     { background:#001f0f; border:1px solid #16a34a; border-radius:8px; padding:11px 15px; margin:9px 0; }
.ok-box .rpt-p     { color:#86efac; }
.score-wrap { background:#1e2535; border-radius:6px; height:9px; width:100%; margin:7px 0; }
.score-bar  { height:9px; border-radius:6px; }
.dash-divider { height:1px; background:#1e2535; margin:28px 0; border:none; }
.citation { color:#5a6478 !important; font-size:.73rem !important; line-height:1.45; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# 3. SEASON DATE RANGES
# ══════════════════════════════════════════════════════════════════════

AVAILABLE_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
SEASON_RANGES = {
    2021: ("2021-04-01", "2021-11-02"),
    2022: ("2022-04-07", "2022-11-05"),
    2023: ("2023-03-30", "2023-11-01"),
    2024: ("2024-03-20", "2024-10-30"),
    2025: ("2025-03-27", "2025-10-31"),
    2026: ("2026-03-26", "2026-10-31"),
}


def season_dates(year: int) -> tuple:
    if year in SEASON_RANGES:
        return SEASON_RANGES[year]
    return (f"{year}-03-20", f"{year}-11-05")


def clamp_end_date(end_str: str) -> str:
    today = date.today().isoformat()
    return min(end_str, today)


# ══════════════════════════════════════════════════════════════════════
# 4. CACHED DATA LOADERS  (all network I/O goes through here)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner="🔍  Building player search index (first run only)…")
def load_registry() -> pd.DataFrame:
    """Chadwick register: maps names <-> MLBAM ids. Cached 24h."""
    try:
        df = pyb.chadwick_register()
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["key_mlbam", "name_last", "name_first"]).copy()
    df["key_mlbam"] = df["key_mlbam"].astype(int)
    df["full_name"] = df["name_first"].astype(str) + " " + df["name_last"].astype(str)
    keep = [c for c in ["key_mlbam", "name_first", "name_last", "full_name",
                         "mlb_played_first", "mlb_played_last"] if c in df.columns]
    return df[keep].drop_duplicates("key_mlbam").reset_index(drop=True)


def search_people(query: str, registry: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    if registry is None or registry.empty or not query or len(query.strip()) < 2:
        return pd.DataFrame()
    q = query.lower().strip()
    hit = registry[registry["full_name"].str.lower().str.contains(q, na=False, regex=False)]
    if "mlb_played_last" in hit.columns:
        hit = hit.sort_values("mlb_played_last", ascending=False, na_position="last")
    return hit.head(limit)


def resolve_mlbam_id(name: str, registry: pd.DataFrame) -> tuple:
    """
    Best-effort match of a plain 'First Last' name (e.g. from a FanGraphs
    roster) to a Chadwick MLBAM id. Returns (mlbam_id_or_None, note_str).
    """
    if registry is None or registry.empty or not name:
        return None, "empty registry"
    name_clean = re.sub(r"[.\*#▲]", "", str(name)).strip()
    exact = registry[registry["full_name"].str.lower() == name_clean.lower()]
    if len(exact) == 1:
        return int(exact.iloc[0]["key_mlbam"]), ""
    if len(exact) > 1:
        best = exact.sort_values("mlb_played_last", ascending=False, na_position="last").iloc[0]
        return int(best["key_mlbam"]), "multiple exact matches — picked most recent"

    parts = name_clean.split()
    if len(parts) < 2:
        return None, "name too short to match"
    first, last = parts[0].lower(), parts[-1].lower()
    cand = registry[
        registry["name_last"].str.lower().str.contains(re.escape(last), na=False) &
        registry["name_first"].str.lower().str.contains(re.escape(first), na=False)
    ]
    if cand.empty:
        return None, "no match found in registry"
    best = cand.sort_values("mlb_played_last", ascending=False, na_position="last").iloc[0]
    return int(best["key_mlbam"]), ""


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary outcome flags + movement columns to a raw Statcast dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    df["hbrk"] = pd.to_numeric(df["pfx_x"], errors="coerce") * 12 if "pfx_x" in df.columns else np.nan
    df["vbrk"] = pd.to_numeric(df["pfx_z"], errors="coerce") * 12 if "pfx_z" in df.columns else np.nan

    desc = df.get("description", pd.Series([""] * len(df), index=df.index)).fillna("")
    df["is_swing"]   = desc.isin(SWING_EV).astype("int8")
    df["is_whiff"]   = desc.isin(WHIFF_EV).astype("int8")
    df["is_contact"] = desc.isin(CONTACT_EV).astype("int8")

    ls = pd.to_numeric(df.get("launch_speed", pd.Series(dtype=float, index=df.index)), errors="coerce")
    la = pd.to_numeric(df.get("launch_angle", pd.Series(dtype=float, index=df.index)), errors="coerce")
    df["is_barrel"] = ((ls >= 98) & la.between(26, 30)).fillna(False).astype("int8")
    df["is_hh"]     = (ls >= 95).fillna(False).astype("int8")
    df["is_gb"]     = (la < 10).fillna(False).astype("int8")

    if "balls" in df.columns and "strikes" in df.columns:
        df["count_state"] = (
            df["balls"].astype("Int64").astype(str).str.strip() + "-" +
            df["strikes"].astype("Int64").astype(str).str.strip()
        )
    if "zone" in df.columns:
        df["zone"] = pd.to_numeric(df["zone"], errors="coerce")

    for c in ["release_speed", "release_spin_rate", "release_extension",
              "arm_angle", "effective_speed", "plate_x", "plate_z"]:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "pitch_type" not in df.columns:
        df["pitch_type"] = np.nan
    if "pitch_name" not in df.columns:
        df["pitch_name"] = df["pitch_type"].map(PITCH_LONG)

    return df


@st.cache_data(ttl=3600, show_spinner="⚾  Pulling pitcher Statcast data from Baseball Savant…")
def load_pitcher_statcast(pitcher_id: int, season: int) -> pd.DataFrame:
    start, end = season_dates(season)
    end = clamp_end_date(end)
    try:
        df = pyb.statcast_pitcher(start, end, pitcher_id)
    except Exception as e:
        st.warning(f"Statcast pull failed: {e}")
        return pd.DataFrame()
    return _add_flags(df)


@st.cache_data(ttl=3600, show_spinner="🏏  Pulling batter Statcast data from Baseball Savant…")
def load_batter_statcast(batter_id: int, season: int) -> pd.DataFrame:
    start, end = season_dates(season)
    end = clamp_end_date(end)
    try:
        df = pyb.statcast_batter(start, end, batter_id)
    except Exception as e:
        st.warning(f"Statcast pull failed: {e}")
        return pd.DataFrame()
    return _add_flags(df)


@st.cache_data(ttl=3600, show_spinner="📋  Pulling FanGraphs season pitching stats…")
def load_fg_pitching(season: int, qual: int = 0) -> pd.DataFrame:
    try:
        df = pyb.pitching_stats(season, qual=qual)
    except Exception as e:
        st.warning(f"FanGraphs pitching pull failed: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["season"] = season
    numeric_cols = ["ERA", "FIP", "WHIP", "K/9", "BB/9", "HR/9", "IP", "Age", "WAR", "SO", "BB", "ERA-", "FIP-"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "K/9" not in df.columns and "SO" in df.columns and "IP" in df.columns:
        df["K/9"] = df["SO"] / df["IP"].replace(0, np.nan) * 9
    if "BB/9" not in df.columns and "BB" in df.columns and "IP" in df.columns:
        df["BB/9"] = df["BB"] / df["IP"].replace(0, np.nan) * 9
    # FanGraphs exposes ERA- (lower=better, 100=avg), not ERA+. Derive ERA+ for
    # display/scoring purposes: ERA+ * ERA- ≈ 10,000 (both are % of league avg).
    if "ERA-" in df.columns:
        df["ERA+"] = (10000.0 / df["ERA-"].replace(0, np.nan)).round(0)
    return df


@st.cache_data(ttl=3600, show_spinner="📋  Pulling FanGraphs season batting stats…")
def load_fg_batting(season: int, qual: int = 0) -> pd.DataFrame:
    try:
        df = pyb.batting_stats(season, qual=qual)
    except Exception as e:
        st.warning(f"FanGraphs batting pull failed: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["season"] = season
    return df


def build_team_roster(fg_df: pd.DataFrame) -> dict:
    """{team: sorted [names]} from a FanGraphs season leaderboard (pitching or batting)."""
    if fg_df is None or fg_df.empty or "Team" not in fg_df.columns or "Name" not in fg_df.columns:
        return {}
    roster = {}
    for team, grp in fg_df.groupby("Team"):
        names = sorted(grp["Name"].dropna().astype(str).unique().tolist())
        if names:
            roster[str(team)] = names
    return roster


@st.cache_data(ttl=1800, show_spinner="📡  Pulling league Statcast for the selected date range…")
def load_league_statcast(start: str, end: str, team: str | None) -> pd.DataFrame:
    end = clamp_end_date(end)
    try:
        if team and team != "All":
            try:
                df = pyb.statcast(start_dt=start, end_dt=end, team=team)
            except TypeError:
                df = pyb.statcast(start_dt=start, end_dt=end)
                df = df[(df.get("home_team") == team) | (df.get("away_team") == team)]
        else:
            df = pyb.statcast(start_dt=start, end_dt=end)
    except Exception as e:
        st.warning(f"League Statcast pull failed: {e}")
        return pd.DataFrame()
    return _add_flags(df)


# ══════════════════════════════════════════════════════════════════════
# 5. HELPERS — zones, ratings, formatting, movement filters
# ══════════════════════════════════════════════════════════════════════

def safe_num(v, default: float = np.nan) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def classify_subzone(plate_x: float, plate_z: float, zone: int) -> str:
    cx, cz = ZONE_CENTERS.get(int(zone), (0.0, 2.5))
    try:
        top  = "T" if float(plate_z) >= cz else "B"
        side = "L" if float(plate_x) <  cx else "R"
    except (TypeError, ValueError):
        return "TL"
    return top + side


def arm_info(angle: float) -> tuple:
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


def rate(val, avg, hi_is_good: bool = True, thr: tuple = (5, 10)) -> str:
    try:
        d = (val - avg) if hi_is_good else (avg - val)
        if np.isnan(d):
            return "avg"
    except TypeError:
        return "avg"
    if d >= thr[1]:   return "elite"
    if d >= thr[0]:   return "above"
    if d >= -thr[0]:  return "avg"
    if d >= -thr[1]:  return "below"
    return "poor"


RATE_LABEL = {"elite": "🟢 Elite", "above": "🔵 Above Avg", "avg": "⚪ Average",
              "below": "🟡 Below Avg", "poor": "🔴 Needs Work"}
RATE_CSS   = {"elite": "pill-elite", "above": "pill-above", "avg": "pill-avg",
              "below": "pill-below", "poor": "pill-poor"}


def pill(text: str, css_class: str) -> str:
    return f'<span class="pill {css_class}">{text}</span>'


def apply_movement_filters(df: pd.DataFrame, velo_rng, spin_rng, hbrk_rng, vbrk_rng) -> pd.DataFrame:
    """Filter a raw pitch-level dataframe by velocity / spin / H-break / V-break ranges."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "release_speed" in out.columns:
        out = out[pd.to_numeric(out["release_speed"], errors="coerce").between(*velo_rng).fillna(True)]
    if "release_spin_rate" in out.columns:
        out = out[pd.to_numeric(out["release_spin_rate"], errors="coerce").between(*spin_rng).fillna(True)]
    if "hbrk" in out.columns:
        out = out[out["hbrk"].between(*hbrk_rng).fillna(True)]
    if "vbrk" in out.columns:
        out = out[out["vbrk"].between(*vbrk_rng).fillna(True)]
    return out


def movement_filter_widgets(key_prefix: str) -> tuple:
    """Renders the 4 sliders (velo/spin/H-break/V-break) inside an expander.
    Returns (velo_rng, spin_rng, hbrk_rng, vbrk_rng)."""
    with st.expander("🔬 Velocity · Spin · Movement filters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            velo_rng = st.slider("Velocity (mph)", 55, 105, (55, 105), key=f"{key_prefix}_velo")
            spin_rng = st.slider("Spin rate (rpm)", 500, 3600, (500, 3600), key=f"{key_prefix}_spin")
        with c2:
            hbrk_rng = st.slider("H-Break (in)", -25.0, 25.0, (-25.0, 25.0), step=0.5, key=f"{key_prefix}_hbrk")
            vbrk_rng = st.slider("V-Break (in)", -20.0, 25.0, (-20.0, 25.0), step=0.5, key=f"{key_prefix}_vbrk")
    return velo_rng, spin_rng, hbrk_rng, vbrk_rng


def compute_zone_stats_from_raw(df: pd.DataFrame, extra_filters: dict | None = None) -> pd.DataFrame:
    if df is None or df.empty or "zone" not in df.columns:
        return pd.DataFrame()
    dff = df.copy()
    if extra_filters:
        if extra_filters.get("count_state") and "count_state" in dff.columns:
            dff = dff[dff["count_state"] == extra_filters["count_state"]]
        if extra_filters.get("stand") not in (None, "All") and "stand" in dff.columns:
            dff = dff[dff["stand"] == extra_filters["stand"]]
        if extra_filters.get("p_throws") not in (None, "All") and "p_throws" in dff.columns:
            dff = dff[dff["p_throws"] == extra_filters["p_throws"]]
        if extra_filters.get("pitch_type") not in (None, "All") and "pitch_type" in dff.columns:
            dff = dff[dff["pitch_type"] == extra_filters["pitch_type"]]

    dff = dff[dff["zone"].between(1, 14)]
    if dff.empty:
        return pd.DataFrame()

    grp = dff.groupby("zone", as_index=False).agg(
        total=("is_swing", "count"), swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"), contacts=("is_contact", "sum"),
        barrels=("is_barrel", "sum"), hard_hits=("is_hh", "sum"), gbs=("is_gb", "sum"),
        batted=("launch_speed", "count"), avg_ev=("launch_speed", "mean"),
        avg_la=("launch_angle", "mean"),
        avg_xwoba=("estimated_woba_using_speedangle", "mean"),
        avg_velo=("release_speed", "mean"), avg_spin=("release_spin_rate", "mean"),
        avg_hbrk=("hbrk", "mean"), avg_vbrk=("vbrk", "mean"),
    )
    n = grp["total"].replace(0, np.nan); sw = grp["swings"].replace(0, np.nan); bt = grp["batted"].replace(0, np.nan)
    grp["swing_pct"]    = (grp["swings"]   / n  * 100).round(1)
    grp["whiff_pct"]    = (grp["whiffs"]   / sw * 100).round(1)
    grp["contact_pct"]  = (grp["contacts"] / sw * 100).round(1)
    grp["barrel_pct"]   = (grp["barrels"]  / bt * 100).round(1)
    grp["hard_hit_pct"] = (grp["hard_hits"]/ bt * 100).round(1)
    grp["gb_pct"]       = (grp["gbs"]      / bt * 100).round(1)
    for c in ["avg_ev", "avg_la", "avg_velo", "avg_spin", "avg_hbrk", "avg_vbrk"]:
        grp[c] = grp[c].round(1)
    grp["avg_xwoba"] = grp["avg_xwoba"].round(3)
    return grp


# ══════════════════════════════════════════════════════════════════════
# 6. DRAWING
# ══════════════════════════════════════════════════════════════════════

def _empty_heatmap(msg="No data") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 6.8))
    fig.patch.set_facecolor("#0b0f17"); ax.set_facecolor("#0b0f17")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color="#8892a4", fontsize=13, transform=ax.transAxes)
    ax.axis("off")
    return fig


def draw_heatmap(df_f: pd.DataFrame, stat_label: str, title: str, batter_mode: bool = False) -> plt.Figure:
    cfg = STAT_CONFIG.get(stat_label, STAT_CONFIG["Whiff %"])
    col, (vmin, vmax), fmt = cfg["col"], cfg["rng"], cfg["fmt"]

    if df_f is None or df_f.empty or col not in df_f.columns:
        return _empty_heatmap()

    pv = df_f.groupby("zone")[col].mean().to_dict()
    pp = df_f.groupby("zone")["total"].sum().to_dict()
    cmap = sns.color_palette("RdYlGn_r" if batter_mode else "YlOrRd", as_cmap=True)

    brd, ms = 0.85, 3.3
    cell = ms / 3; mx = my = brd; top = my + ms; rx = mx + ms; hlf = ms / 2; sy = 2.5

    fig, ax = plt.subplots(figsize=(6.6, 6.9))
    fig.patch.set_facecolor("#0b0f17"); ax.set_facecolor("#0b0f17")

    def _fill(z):
        v, t = pv.get(z, np.nan), pp.get(z, 0)
        return "#1c2230" if (pd.isna(v) or t == 0) else cmap(np.clip((v - vmin) / (vmax - vmin), 0, 1))

    def _lbl(z):
        v, t = pv.get(z, np.nan), pp.get(z, 0)
        return str(z) if (t == 0 or pd.isna(v)) else f"{z}\n{fmt.format(v)}"

    def _tc(z):
        v, t = pv.get(z, np.nan), pp.get(z, 0)
        return "#4a5568" if (pd.isna(v) or t == 0) else "#111111"

    for i in range(3):
        for j in range(3):
            z = i * 3 + j + 1
            x0, y0 = mx + j * cell, my + (2 - i) * cell
            ax.add_patch(plt.Rectangle((x0, y0), cell, cell, facecolor=_fill(z), edgecolor="#1e2535", linewidth=2.0))
            ax.text(x0 + cell / 2, y0 + cell / 2, _lbl(z), ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=_tc(z))

    shadow_paths = [
        (11, [(0, sy), (brd, sy), (brd, top), (mx, top), (mx + hlf, top), (mx + hlf, 5), (0, 5), (0, sy)]),
        (12, [(rx, sy), (rx, top), (mx + hlf, top), (mx + hlf, 5), (5, 5), (5, sy), (rx, sy)]),
        (13, [(0, sy), (brd, sy), (brd, my), (mx, my), (mx + hlf, my), (mx + hlf, 0), (0, 0), (0, sy)]),
        (14, [(rx, sy), (rx, my), (mx + hlf, my), (mx + hlf, 0), (5, 0), (5, sy), (rx, sy)]),
    ]
    shadow_centres = {11: (brd/2, 5-brd/2), 12: (5-brd/2, 5-brd/2), 13: (brd/2, brd/2), 14: (5-brd/2, brd/2)}
    for z, verts in shadow_paths:
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 1)
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=_fill(z), edgecolor="#1e2535", linewidth=2.0))
        xt, yt = shadow_centres[z]
        ax.text(xt, yt, _lbl(z), ha="center", va="center", fontsize=10.5, fontweight="bold", color=_tc(z))

    ax.add_patch(plt.Rectangle((mx, my), ms, ms, fill=False, edgecolor="#f85149", linewidth=3.0, zorder=10))
    ax.set_xlim(0, 5); ax.set_ylim(-0.85, 5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11, pad=14, color="#c9d1d9", fontweight="600")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.68, pad=0.03)
    cbar.set_label(stat_label, fontsize=9, color="#8892a4")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8892a4", fontsize=8)
    cbar.outline.set_edgecolor("#1e2535")

    n_total = int(sum(pp.values()))
    ax.text(2.5, -0.38, f"n = {n_total:,} pitches", ha="center", fontsize=8.5, color="#718096", style="italic")
    plt.tight_layout()
    return fig


def draw_subzone_panel(raw_df: pd.DataFrame, zone: int, stat_label: str = "Whiff %") -> plt.Figure:
    cfg = STAT_CONFIG.get(stat_label, STAT_CONFIG["Whiff %"])
    col, (vmin, vmax), fmt = cfg["col"], cfg["rng"], cfg["fmt"]
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    bz = raw_df[raw_df["zone"] == zone].copy() if raw_df is not None and not raw_df.empty else pd.DataFrame()
    if bz.empty or "plate_x" not in bz.columns:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#111621"); ax.axis("off")
        ax.text(0.5, 0.5, f"No data\nfor Zone {zone}", ha="center", va="center", color="#8892a4", fontsize=12)
        return fig

    bz["sub"] = bz.apply(
        lambda r: classify_subzone(
            float(r["plate_x"]) if not pd.isna(r.get("plate_x")) else 0.0,
            float(r["plate_z"]) if not pd.isna(r.get("plate_z")) else 2.5,
            zone,
        ), axis=1,
    )
    sg = bz.groupby("sub", as_index=False).agg(
        total=("is_swing", "count"), swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"),
        contacts=("is_contact", "sum"), barrels=("is_barrel", "sum"), hard_hits=("is_hh", "sum"),
        gbs=("is_gb", "sum"), batted=("launch_speed", "count"),
        avg_ev=("launch_speed", "mean"), avg_la=("launch_angle", "mean"),
        avg_xwoba=("estimated_woba_using_speedangle", "mean"),
    )
    sw = sg["swings"].replace(0, np.nan); n = sg["total"].replace(0, np.nan); bt = sg["batted"].replace(0, np.nan)
    sg["whiff_pct"]    = (sg["whiffs"]   / sw * 100).round(1)
    sg["swing_pct"]    = (sg["swings"]   / n  * 100).round(1)
    sg["contact_pct"]  = (sg["contacts"] / sw * 100).round(1)
    sg["barrel_pct"]   = (sg["barrels"]  / bt * 100).round(1)
    sg["hard_hit_pct"] = (sg["hard_hits"]/ bt * 100).round(1)
    sg["gb_pct"]       = (sg["gbs"]      / bt * 100).round(1)
    sg["avg_ev"] = sg["avg_ev"].round(1); sg["avg_la"] = sg["avg_la"].round(1)
    sg["avg_xwoba"] = sg["avg_xwoba"].round(3)
    sg = sg.set_index("sub")

    positions = {"TL": (0, 1), "TR": (1, 1), "BL": (0, 0), "BR": (1, 0)}
    fig, axes = plt.subplots(2, 2, figsize=(4.6, 4.6))
    fig.patch.set_facecolor("#0b0f17")
    fig.suptitle(f"Zone {zone} — {stat_label} by Sub-Zone", color="#f0f6ff", fontsize=9.5, fontweight="700", y=1.02)

    for quad, (ci, ri) in positions.items():
        ax = axes[1 - ri][ci]
        ax.set_facecolor("#111621"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        if quad in sg.index:
            row = sg.loc[quad]
            val, n_p = row.get(col, np.nan), int(row.get("total", 0))
            if not pd.isna(val) and n_p > 0:
                colour = cmap(np.clip((val - vmin) / (vmax - vmin), 0, 1))
                ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, facecolor=colour, edgecolor="#1e2535", linewidth=2.0))
                ax.text(0.5, 0.60, fmt.format(val), ha="center", va="center", fontsize=14, fontweight="800", color="#111111")
                ax.text(0.5, 0.32, quad, ha="center", va="center", fontsize=9, fontweight="600", color="#333333")
                ax.text(0.5, 0.14, f"n={n_p}", ha="center", va="center", fontsize=7.5, color="#555555")
                continue
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, facecolor="#1c2230", edgecolor="#1e2535", linewidth=1.5))
        ax.text(0.5, 0.5, f"{quad}\nno data", ha="center", va="center", fontsize=8, color="#4a5568")

    plt.tight_layout(pad=0.5)
    return fig


def plot_arsenal(ars: pd.DataFrame, name: str, hand: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor("#0b0f17")

    ax = axes[0]; ax.set_facecolor("#111621")
    ax.axhline(0, color="#4a5568", lw=0.8, ls="--"); ax.axvline(0, color="#4a5568", lw=0.8, ls="--")
    for _, r in ars.iterrows():
        pt = r["pitch_type"]; c = PITCH_COLORS.get(pt, "#94a3b8"); sz = max(80, r["usage"] * 18)
        ax.scatter(r["avg_h"], r["avg_v"], s=sz, color=c, zorder=5, edgecolors="white", lw=1.2, alpha=0.9)
        ax.annotate(pt, (r["avg_h"], r["avg_v"]), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="white")
        av = MLB_AVG.get(pt)
        if av:
            ax.scatter(av["hbrk"], av["vbrk"], s=55, color="none", edgecolors=c, lw=1.5, zorder=4, alpha=0.4)
    ax.set_xlabel("H-Break (in)  →  Arm-side", color="#a0aec0", fontsize=9)
    ax.set_ylabel("V-Break (in)  ↑  Rise", color="#a0aec0", fontsize=9)
    ax.set_title("Movement Profile\n(solid = pitcher · hollow = MLB avg)", color="#e8edf5", fontsize=9, fontweight="600")
    ax.tick_params(colors="#8892a4", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3545")

    ax2 = axes[1]; ax2.set_facecolor("#111621")
    colors = [PITCH_COLORS.get(pt, "#94a3b8") for pt in ars["pitch_type"]]
    labels = ars.get("pitch_name", ars["pitch_type"])
    bars = ax2.barh(labels, ars["usage"], color=colors, edgecolor="#1e2535", height=0.6)
    for bar, val in zip(bars, ars["usage"]):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center",
                  fontsize=8.5, fontweight="bold", color="#e2e8f0")
    ax2.set_xlabel("Usage %", color="#a0aec0", fontsize=9)
    ax2.set_title("Pitch Usage", color="#e8edf5", fontsize=9, fontweight="600")
    ax2.invert_yaxis(); ax2.tick_params(colors="#8892a4", labelsize=8.5)
    ax2.set_xlim(0, ars["usage"].max() + 14)
    for sp in ax2.spines.values(): sp.set_edgecolor("#2a3545")

    hand_str = "RHP" if hand == "R" else "LHP"
    fig.suptitle(f"{name}  ({hand_str})", color="#f0f6ff", fontsize=11, fontweight="700", y=1.03)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 7. ARSENAL / ANALYSIS / REPORT GENERATORS
# ══════════════════════════════════════════════════════════════════════

def build_pitcher_arsenal(df: pd.DataFrame) -> tuple:
    if df is None or df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame(), 0, np.nan, np.nan, "R"

    df = df.copy()
    for c in ["release_speed", "release_spin_rate", "hbrk", "vbrk", "release_extension", "arm_angle"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    total = len(df)
    g = df.groupby("pitch_type").agg(
        count=("pitch_type", "count"), avg_velo=("release_speed", "mean"),
        max_velo=("release_speed", "max"), avg_spin=("release_spin_rate", "mean"),
        avg_h=("hbrk", "mean"), avg_v=("vbrk", "mean"),
        avg_ext=("release_extension", "mean"), avg_arm=("arm_angle", "mean"),
        swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"),
    ).reset_index()

    g["usage"] = (g["count"] / total * 100).round(1)
    g["whiff"] = (g["whiffs"] / g["swings"].replace(0, np.nan) * 100).round(1)
    for c in ["avg_velo", "max_velo", "avg_h", "avg_v", "avg_ext", "avg_arm"]:
        g[c] = g[c].round(1)
    g["avg_spin"] = g["avg_spin"].round(0)

    if "pitch_name" in df.columns:
        pn_map = df.groupby("pitch_type")["pitch_name"].first().to_dict()
        g["pitch_name"] = g["pitch_type"].map(pn_map).fillna(g["pitch_type"].map(PITCH_LONG)).fillna(g["pitch_type"])
    else:
        g["pitch_name"] = g["pitch_type"].map(PITCH_LONG).fillna(g["pitch_type"])

    arm_avg = float(df["arm_angle"].mean())
    ext_avg = float(df["release_extension"].mean())
    hand = df["p_throws"].mode()[0] if "p_throws" in df.columns and not df.empty else "R"

    return g.sort_values("usage", ascending=False).reset_index(drop=True), total, arm_avg, ext_avg, hand


def analyze_batter(df: pd.DataFrame, filters: dict | None = None) -> dict:
    if df is None or df.empty:
        return {}
    dff = df.copy()
    if filters:
        if filters.get("p_throws") not in (None, "All") and "p_throws" in dff.columns:
            dff = dff[dff["p_throws"] == filters["p_throws"]]
        if filters.get("count_state") not in (None, "All") and "count_state" in dff.columns:
            dff = dff[dff["count_state"] == filters["count_state"]]
        if filters.get("pitch_type") not in (None, "All") and "pitch_type" in dff.columns:
            dff = dff[dff["pitch_type"] == filters["pitch_type"]]
    if dff.empty:
        return {}

    stand = dff["stand"].mode()[0] if "stand" in dff.columns and not dff.empty else "R"
    zone_df = compute_zone_stats_from_raw(dff)

    pt_agg = pd.DataFrame()
    if "pitch_type" in dff.columns:
        pt_agg = dff.groupby("pitch_type").agg(
            total=("is_swing", "count"), swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"),
            avg_xwoba=("estimated_woba_using_speedangle", "mean"), avg_ev=("launch_speed", "mean"),
            avg_velo=("release_speed", "mean"), avg_spin=("release_spin_rate", "mean"),
            avg_hbrk=("hbrk", "mean"), avg_vbrk=("vbrk", "mean"),
        ).reset_index()
        sw2 = pt_agg["swings"].replace(0, np.nan); n2 = pt_agg["total"].replace(0, np.nan)
        pt_agg["whiff_pct"] = (pt_agg["whiffs"] / sw2 * 100).round(1)
        pt_agg["swing_pct"] = (pt_agg["swings"] / n2 * 100).round(1)
        pt_agg["usage"] = (pt_agg["total"] / len(dff) * 100).round(1)
        for c in ["avg_xwoba", "avg_ev", "avg_velo", "avg_hbrk", "avg_vbrk"]:
            pt_agg[c] = pt_agg[c].round(2)
        pt_agg["avg_spin"] = pt_agg["avg_spin"].round(0)
        pt_agg["pitch_name"] = pt_agg["pitch_type"].map(PITCH_LONG).fillna(pt_agg["pitch_type"])

    cnt_agg = pd.DataFrame()
    if "count_state" in dff.columns:
        cnt_agg = dff.groupby("count_state").agg(
            pitches=("is_swing", "count"), swing_pct=("is_swing", "mean"),
            whiff_pct=("is_whiff", "mean"), avg_xwoba=("estimated_woba_using_speedangle", "mean"),
        ).reset_index()
        cnt_agg["swing_pct"] = (cnt_agg["swing_pct"] * 100).round(1)
        cnt_agg["whiff_pct"] = (cnt_agg["whiff_pct"] * 100).round(1)
        cnt_agg["avg_xwoba"] = cnt_agg["avg_xwoba"].round(3)

    platoon = {}
    if "p_throws" in dff.columns:
        for hand, grp in dff.groupby("p_throws"):
            sw3 = grp["is_swing"].sum()
            platoon[hand] = {
                "pitches": len(grp),
                "swing_pct": round(sw3 / len(grp) * 100, 1) if len(grp) else 0,
                "whiff_pct": round(grp["is_whiff"].sum() / max(sw3, 1) * 100, 1),
                "avg_xwoba": round(grp["estimated_woba_using_speedangle"].mean(), 3)
                             if "estimated_woba_using_speedangle" in grp.columns else np.nan,
            }

    weak_zones, avoid_zones = [], []
    if not zone_df.empty and "whiff_pct" in zone_df.columns:
        valid = zone_df.dropna(subset=["whiff_pct"])
        weak_zones = valid.sort_values("whiff_pct", ascending=False)["zone"].head(3).tolist()
    if not zone_df.empty and "avg_xwoba" in zone_df.columns:
        valid = zone_df.dropna(subset=["avg_xwoba"])
        avoid_zones = valid.sort_values("avg_xwoba", ascending=False)["zone"].head(3).tolist()

    return {
        "zone_stats": zone_df, "pt_stats": pt_agg, "count_stats": cnt_agg,
        "platoon": platoon, "stand": stand, "total_pitches": len(dff),
        "weak_zones": weak_zones, "avoid_zones": avoid_zones,
    }


def generate_pitching_plan(batter_stats: dict, batter_name: str) -> str:
    if not batter_stats:
        return '<div class="rpt-p">No data available.</div>'
    stand = batter_stats.get("stand", "R")
    weak_zones, avoid_zones = batter_stats.get("weak_zones", []), batter_stats.get("avoid_zones", [])
    pt, platoon = batter_stats.get("pt_stats", pd.DataFrame()), batter_stats.get("platoon", {})

    ZONE_DESC = {1:"top-inside",2:"top-middle",3:"top-outside",4:"mid-inside",5:"center",6:"mid-outside",
                 7:"bot-inside",8:"bot-middle",9:"bot-outside",11:"shadow-top",12:"shadow-right",
                 13:"shadow-bottom",14:"shadow-left"}

    html = ['<div class="report-wrap">']
    html.append(f'<div style="font-size:1.1rem;font-weight:800;color:#f0f6ff;margin-bottom:12px;">'
                f'🎯 Pitching Plan — How to Attack {batter_name}</div>')

    html.append('<div class="rpt-h2">1. BATTER PROFILE & PLATOON</div>')
    html.append(f'<div class="rpt-p"><b>Handedness:</b> {"Right-handed (RHB)" if stand=="R" else "Left-handed (LHB)"}.</div>')
    if platoon:
        rh, lh = platoon.get("R", {}), platoon.get("L", {})
        if rh and lh:
            better = "RHP" if rh.get("whiff_pct",0) > lh.get("whiff_pct",0) else "LHP"
            html.append(f'<div class="insight-box"><div class="rpt-p"><b>Platoon split:</b> '
                        f'{rh.get("whiff_pct",0):.1f}% whiff vs RHP, {lh.get("whiff_pct",0):.1f}% vs LHP. '
                        f'{better} has the platoon advantage.</div></div>')

    html.append('<div class="rpt-h2">2. ZONE ATTACK STRATEGY</div>')
    if weak_zones:
        z_str = ", ".join(f"Zone {z} ({ZONE_DESC.get(z,'?')})" for z in weak_zones)
        html.append(f'<div class="rpt-p"><b>Highest whiff zones (attack here):</b> {z_str}</div>')
        html.append('<div class="ok-box"><div class="rpt-p">🎯 Deploy your best out-pitch here in 2-strike counts.</div></div>')
    if avoid_zones:
        a_str = ", ".join(f"Zone {z}" for z in avoid_zones)
        html.append(f'<div class="rpt-p"><b>High damage zones (avoid):</b> {a_str}</div>')
        html.append('<div class="danger-box"><div class="rpt-p">⚠️ Do NOT groove pitches here unless ahead 0-2/1-2.</div></div>')

    html.append('<div class="rpt-h2">3. BEST PITCHES TO USE</div>')
    if not pt.empty and "whiff_pct" in pt.columns:
        pt_sorted = pt[pt["total"] >= 8].dropna(subset=["whiff_pct"]).sort_values("whiff_pct", ascending=False)
        if not pt_sorted.empty:
            best = pt_sorted.iloc[0]
            bc = PITCH_COLORS.get(best["pitch_type"], "#94a3b8")
            bn = best.get("pitch_name", best["pitch_type"])
            html.append(f'<div class="rpt-p"><b>Best out-pitch:</b> '
                        f'<span style="color:{bc};font-weight:700;">{bn}</span> — '
                        f'{best["whiff_pct"]:.1f}% whiff ({int(best["total"])} pitches).</div>')
            html.append(f'<div class="ok-box"><div class="rpt-p">Increase usage of {bn} in 2-strike counts.</div></div>')
            if len(pt_sorted) > 1:
                worst = pt_sorted.iloc[-1]
                wn = worst.get("pitch_name", worst["pitch_type"])
                html.append(f'<div class="rpt-p"><b>Pitch to avoid:</b> {wn} — only '
                            f'{worst["whiff_pct"]:.1f}% whiff. Use sparingly.</div>')

    html.append('<div class="rpt-h2">4. COUNT-SPECIFIC APPROACH</div>')
    cnt = batter_stats.get("count_stats", pd.DataFrame())
    if not cnt.empty:
        cnt_map = cnt.set_index("count_state")
        for count, advice in [
            ("0-0", "Attack for a called strike — first-pitch strikes dramatically improve outcomes."),
            ("0-1", "Batter is defensive — offer something just off the zone to expand."),
            ("0-2", "Expand to shadow zones with highest-whiff pitch. Never groove a fastball."),
            ("1-2", "Prime put-away count — same release point, break to the shadow zone."),
            ("2-2", "Elevated fastball then breaking ball to low shadow."),
            ("3-1", "Batter sitting dead-red — only count to consider a surprise off-speed strike."),
            ("3-2", "Best fastball in zone — must compete, no room for mistake."),
        ]:
            extra = ""
            if count in cnt_map.index:
                row = cnt_map.loc[count]
                extra = f" [Batter swings {safe_num(row.get('swing_pct')):.0f}%, whiffs {safe_num(row.get('whiff_pct')):.0f}%]"
            html.append(f'<div class="rpt-li">• <b>{count}:</b> {advice}{extra}</div>')

    html.append('</div>')
    return "\n".join(html)


def generate_pitcher_report(name: str, ars: pd.DataFrame, total: int, arm_avg: float,
                             ext_avg: float, hand: str, season_lbl: str) -> str:
    """Full scouting report — arsenal breakdown, arm slot, tunneling pairs,
    and priority action items (restored from the original feature set)."""
    sn, sc = arm_info(arm_avg)
    sign = 1 if hand == "R" else -1
    html = ['<div class="report-wrap">']
    arm_badge = (f'<span class="ref-badge" style="border-color:{sc};color:{sc};">🎯 {arm_avg:.1f}° — {sn}</span>'
                 if not np.isnan(arm_avg) else "")
    ext_badge = f'<span class="ref-badge">📏 Ext {ext_avg:.2f} ft</span>' if not np.isnan(ext_avg) else ""
    html.append(f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px;">'
                f'<div style="font-size:1.3rem;font-weight:800;color:#f0f6ff;">{name}</div>'
                f'<span class="ref-badge">{"RHP" if hand=="R" else "LHP"}</span>'
                f'<span class="ref-badge">{season_lbl}</span>'
                f'<span class="ref-badge">{total:,} pitches</span>{arm_badge}{ext_badge}</div>')

    # ── 1. Arsenal breakdown ─────────────────────────────────────────
    html.append('<div class="rpt-h2">1. ARSENAL BREAKDOWN vs. MLB AVERAGES</div>')
    pts = ars["pitch_type"].tolist()
    for _, r in ars.iterrows():
        pt, av = r["pitch_type"], MLB_AVG.get(r["pitch_type"])
        nm, col = r.get("pitch_name", pt), PITCH_COLORS.get(pt, "#94a3b8")
        html.append(f'<div style="margin:10px 0;padding:11px 14px;background:#151c2e;'
                    f'border-left:3px solid {col};border-radius:0 8px 8px 0;">')
        html.append(f'<div style="color:{col};font-weight:700;font-size:.88rem;margin-bottom:5px;">{nm} '
                    f'<span style="color:#8892a4;font-size:.77rem;font-weight:400;">'
                    f'{r["usage"]:.1f}% usage · {int(r["count"]):,} pitches</span></div>')
        if av:
            vr = rate(r["avg_velo"], av["velo"], True, (2, 4))
            html.append(f'<div class="rpt-p"><b>Velocity:</b> {pill(RATE_LABEL[vr], RATE_CSS[vr])} '
                        f'{r["avg_velo"]:.1f} mph (avg {av["velo"]:.1f}) · max {r["max_velo"]:.1f}</div>')
            if not pd.isna(r["avg_spin"]):
                sr = rate(r["avg_spin"], av["spin"], True, (150, 300))
                html.append(f'<div class="rpt-p"><b>Spin rate:</b> {pill(RATE_LABEL[sr], RATE_CSS[sr])} '
                            f'{int(r["avg_spin"]):,} rpm (avg {av["spin"]:,})</div>')
            hd = r["avg_h"] - av["hbrk"]
            hd_dir = "more" if (hd * sign) > 0 else "less"
            html.append(f'<div class="rpt-p"><b>H-Break:</b> {r["avg_h"]:+.1f}" (avg {av["hbrk"]:+.1f}") — '
                        f'{abs(hd):.1f}" {hd_dir} arm-side movement than MLB avg</div>')
            vd = r["avg_v"] - av["vbrk"]; vd_dir = "more" if vd > 0 else "less"
            html.append(f'<div class="rpt-p"><b>V-Break:</b> {r["avg_v"]:+.1f}" (avg {av["vbrk"]:+.1f}") — '
                        f'{abs(vd):.1f}" {vd_dir} vertical break</div>')
            if not pd.isna(r["whiff"]):
                wr = rate(r["whiff"], av["whiff"], True, (5, 10))
                under = " — ⚡ UNDERUSED for its whiff rate!" if (r["whiff"] > av["whiff"] + 5 and r["usage"] < 15) else ""
                html.append(f'<div class="rpt-p"><b>Whiff rate:</b> {pill(RATE_LABEL[wr], RATE_CSS[wr])} '
                            f'{r["whiff"]:.1f}% (avg {av["whiff"]:.1f}%){under}</div>')
            if not pd.isna(r.get("avg_ext", np.nan)):
                er = rate(r["avg_ext"], av["ext"], True, (0.2, 0.5))
                html.append(f'<div class="rpt-p"><b>Extension:</b> {pill(RATE_LABEL[er], RATE_CSS[er])} '
                            f'{r["avg_ext"]:.2f} ft (avg {av["ext"]:.2f} ft)</div>')
        else:
            html.append('<div class="rpt-p"><i>No MLB avg benchmark for this pitch type.</i></div>')
        html.append('</div>')

    has_off = any(p in ["CH", "FS", "SV"] for p in pts)
    has_brk = any(p in ["SL", "ST", "CU", "KC"] for p in pts)
    if has_off and has_brk:
        html.append('<div class="ok-box"><div class="rpt-p">✅ Complete three-category arsenal (fastball + offspeed + breaking).</div></div>')
    if not has_off:
        html.append('<div class="danger-box"><div class="rpt-p">🔴 No offspeed pitch (CH/FS/SV) — critical vulnerability vs. opposite-handed batters.</div></div>')
    if not has_brk:
        html.append('<div class="danger-box"><div class="rpt-p">🔴 No breaking ball — 2-strike arsenal severely limited.</div></div>')

    # ── 2. Arm slot & extension ──────────────────────────────────────
    html.append('<div class="rpt-h2">2. ARM SLOT & EXTENSION</div>')
    if not np.isnan(arm_avg):
        html.append(f'<div class="rpt-p">Arm slot: <b style="color:{sc};">{arm_avg:.1f}° — {sn}</b></div>')
        if arm_avg >= 60:
            html.append('<div class="insight-box"><div class="rpt-p"><b>High slot:</b> strong vertical plane, fastball rides. Ideal for curveball/KC.</div></div>')
        elif arm_avg >= 45:
            html.append('<div class="insight-box"><div class="rpt-p"><b>Three-quarter (MLB sweet spot):</b> balanced H/V movement, natural fastball-changeup tunnel.</div></div>')
        elif arm_avg >= 25:
            html.append('<div class="insight-box"><div class="rpt-p"><b>Low three-quarter:</b> increased horizontal movement, large same-hand platoon advantage.</div></div>')
        else:
            html.append('<div class="insight-box"><div class="rpt-p"><b>Sidearm/submarine:</b> largest platoon advantage in baseball. Minimise opposite-hand matchups.</div></div>')
    if not np.isnan(ext_avg):
        er = rate(ext_avg, 6.2, True, (0.2, 0.5))
        ext_note = ('Elite — every pitch plays ~1 mph faster than radar reading.' if ext_avg >= 6.5 else
                    'Below average — stride/release-point work could add +0.3–0.5 ft ≈ +1 mph perceived velocity.'
                    if ext_avg < 6.0 else 'Solid extension.')
        html.append(f'<div class="rpt-p"><b>Extension:</b> {pill(RATE_LABEL[er], RATE_CSS[er])} '
                    f'{ext_avg:.2f} ft (avg ~6.2 ft). {ext_note}</div>')

    # ── 3. Pitch mix & tunneling ─────────────────────────────────────
    html.append('<div class="rpt-h2">3. PITCH MIX & SEQUENCING</div>')
    ranked = ars.dropna(subset=["whiff"]).sort_values("whiff", ascending=False)
    if not ranked.empty:
        best = ranked.iloc[0]
        bc, bn, usage = PITCH_COLORS.get(best["pitch_type"], "#94a3b8"), best.get("pitch_name", best["pitch_type"]), best["usage"]
        usage_note = ("Underused — increase to 20–25% in 2-strike counts." if usage < 15 else
                      "Heavy usage — monitor batter adjustment rates." if usage > 45 else
                      "Prioritise in 2-strike counts and vs same-handed batters.")
        html.append(f'<div class="rpt-p"><b>Best out-pitch:</b> <span style="color:{bc};font-weight:700;">{bn}</span> '
                    f'({best["whiff"]:.1f}% whiff · {usage:.1f}% usage). {usage_note}</div>')

    fb_usage = ars[ars["pitch_type"].isin(["FF","SI","FC"])]["usage"].sum()
    fb_note = ("Heavy fastball reliance — mix secondaries in 0-0/1-0 counts." if fb_usage > 65 else
               "Low fastball usage — ensure zone-attack foundation is not compromised." if fb_usage < 35 else
               "Healthy fastball/secondary balance.")
    html.append(f'<div class="rpt-p"><b>Fastball usage:</b> {fb_usage:.1f}%. {fb_note}</div>')

    tunnels = []
    for i, r1 in ars.iterrows():
        for j, r2 in ars.iterrows():
            if j <= i: continue
            hd, vd = abs(r1["avg_h"] - r2["avg_h"]), abs(r1["avg_v"] - r2["avg_v"])
            c1, c2 = PITCH_COLORS.get(r1["pitch_type"], "#94a3b8"), PITCH_COLORS.get(r2["pitch_type"], "#94a3b8")
            n1, n2 = r1.get("pitch_name", r1["pitch_type"]), r2.get("pitch_name", r2["pitch_type"])
            if hd < 4 and vd > 8:
                tunnels.append(f'<span style="color:{c1};font-weight:700;">{n1}</span> + '
                               f'<span style="color:{c2};font-weight:700;">{n2}</span>: H-plane within {hd:.1f}", '
                               f'V diverges {vd:.1f}" late — <b>elite tunnel pair</b>')
            elif hd > 10 and vd < 5:
                tunnels.append(f'<span style="color:{c1};font-weight:700;">{n1}</span> + '
                               f'<span style="color:{c2};font-weight:700;">{n2}</span>: horizontal attack pair '
                               f'({hd:.1f}" H-break difference)')
    if tunnels:
        html.append('<div class="rpt-h3">Best tunneling pairs:</div>')
        for t in tunnels[:3]:
            html.append(f'<div class="insight-box"><div class="rpt-p">• {t}</div></div>')
    else:
        html.append('<div class="warn-box"><div class="rpt-p">No strong tunneling pairs found. Consider adding a pitch that shares the fastball release trajectory for late-break deception.</div></div>')

    # ── 4. Priority action items ─────────────────────────────────────
    html.append('<div class="rpt-h2">4. PRIORITY ACTION ITEMS</div>')
    actions = []
    for _, r in ars.iterrows():
        av = MLB_AVG.get(r["pitch_type"])
        if not av: continue
        nm_a = r.get("pitch_name", r["pitch_type"])
        if not pd.isna(r["whiff"]) and r["whiff"] < av["whiff"] - 8 and r["usage"] > 15:
            actions.append(("🔴", "Long-term", f'{nm_a} whiff {r["whiff"]:.1f}% vs avg {av["whiff"]:.1f}% — '
                                                 f'review grip/movement; restrict to favourable counts'))
        hd = abs(r["avg_h"] - av["hbrk"])
        if hd > 5 and r["usage"] > 10:
            pot = round(min(hd * 0.6, 5), 1)
            actions.append(("🟡", "Medium-term", f'{nm_a}: {hd:.1f}" below-average movement — '
                                                   f'grip/seam adjustment could add {pot}" break'))
    if not np.isnan(ext_avg) and ext_avg < 6.0:
        actions.append(("🟢", "Easy win", f'Extension {ext_avg:.2f} ft — stride/release work = '
                                           f'+{round(6.2-ext_avg,2):.2f} ft potential (+1 mph perceived)'))
    if not has_off:
        actions.append(("🔴", "Long-term", "Add changeup — highest-priority arsenal gap"))
    if not actions:
        actions.append(("🟢", "Maintain", "Competitive arsenal — focus on count-based execution and sequence variety"))

    for em, tl, txt in actions:
        tc = "#4ade80" if em == "🟢" else "#fbbf24" if em == "🟡" else "#f87171"
        box = "ok-box" if em == "🟢" else "warn-box" if em == "🟡" else "danger-box"
        html.append(f'<div class="{box}" style="margin:6px 0;"><div class="rpt-p">'
                    f'{em} <span style="color:{tc};font-size:.76rem;font-weight:600;">[{tl}]</span> '
                    f'<b>{txt}</b></div></div>')

    html.append('</div>')
    return "\n".join(html)


def generate_matchup_plan(pitcher_name: str, pitcher_hand: str, arsenal_df: pd.DataFrame,
                           batter_name: str, batter_stand: str, batter_stats: dict) -> str:
    """Richer pitcher-vs-batter plan: cross-references pitcher arsenal with
    batter pitch-type/zone vulnerability, plus a count-by-count sequence."""
    if arsenal_df.empty or not batter_stats:
        return '<div class="rpt-p">Insufficient data for matchup analysis.</div>'

    zone_stats = batter_stats.get("zone_stats", pd.DataFrame())
    pt_stats   = batter_stats.get("pt_stats", pd.DataFrame())
    weak_zones = batter_stats.get("weak_zones", [])
    avoid_zones = batter_stats.get("avoid_zones", [])

    sign = 1 if pitcher_hand == "R" else -1
    opp = batter_stand != pitcher_hand
    p_hand_str = "RHP" if pitcher_hand == "R" else "LHP"
    b_hand_str = "RHB" if batter_stand == "R" else "LHB"
    platoon_lbl = "✅ Pitcher platoon advantage (same hand)" if not opp else "⚠️ Batter platoon advantage (opposite hand)"

    ZONE_NAMES = {1:"Top-In",2:"Top-Mid",3:"Top-Out",4:"Mid-In",5:"Center",6:"Mid-Out",
                  7:"Bot-In",8:"Bot-Mid",9:"Bot-Out",11:"Shadow-Top",12:"Shadow-Right",
                  13:"Shadow-Bottom",14:"Shadow-Left"}

    html = ['<div class="report-wrap">']
    html.append(f'<div style="font-size:1.1rem;font-weight:800;color:#f0f6ff;margin-bottom:4px;">'
                f'⚔️ {pitcher_name} vs {batter_name}</div>'
                f'<div style="color:#8892a4;font-size:.8rem;margin-bottom:12px;">{p_hand_str} vs {b_hand_str} · {platoon_lbl}</div>')

    html.append('<div class="rpt-h2">1. PLATOON CONTEXT</div>')
    if not opp:
        html.append('<div class="ok-box"><div class="rpt-p"><b>Same-hand matchup (pitcher advantage).</b> '
                    'Slider/sweeper runs away from the batter — primary put-away pitch. '
                    'Changeup/splitter is less effective here (runs toward the batter).</div></div>')
    else:
        html.append('<div class="warn-box"><div class="rpt-p"><b>Opposite-hand matchup (batter advantage).</b> '
                    'The batter reads release point more easily. Offspeed (changeup/splitter) is critical here — '
                    'it runs arm-side into the batter for natural deception.</div></div>')

    html.append('<div class="rpt-h2">2. PITCH ARSENAL — CROSS-REFERENCED WITH BATTER DATA</div>')
    pt_vuln = {}
    if not pt_stats.empty and "pitch_type" in pt_stats.columns:
        for _, row in pt_stats.iterrows():
            pt_vuln[row["pitch_type"]] = {"whiff": float(row.get("whiff_pct", 0) or 0),
                                           "xwoba": float(row.get("avg_xwoba", 0.320) or 0.320),
                                           "total": int(row.get("total", 0) or 0)}
    ranked = []
    for _, r in arsenal_df.iterrows():
        pt = r["pitch_type"]; p_wh = float(r.get("whiff", 0) or 0)
        bvuln = pt_vuln.get(pt, {})
        combined = p_wh * 0.6 + bvuln.get("whiff", 0) * 0.4
        ranked.append((pt, r, p_wh, bvuln.get("whiff"), bvuln.get("xwoba"), bvuln.get("total", 0), combined))
    ranked.sort(key=lambda x: x[6], reverse=True)

    for rank, (pt, r, p_wh, b_wh, b_xw, n_seen, _) in enumerate(ranked, 1):
        nm, col = r.get("pitch_name", pt), PITCH_COLORS.get(pt, "#94a3b8")
        use = float(r.get("usage", 0) or 0)
        box = "ok-box" if rank == 1 else "insight-box" if rank <= 3 else "ref-card"
        html.append(f'<div class="{box}" style="margin:8px 0;">')
        html.append(f'<div style="color:{col};font-weight:700;font-size:.9rem;margin-bottom:5px;">#{rank} {nm} '
                    f'<span style="color:#8892a4;font-size:.76rem;font-weight:400;">'
                    f'({use:.1f}% usage · {r["avg_velo"]:.1f} mph)</span></div>')
        html.append(f'<div class="rpt-p"><b>Pitcher whiff rate:</b> {p_wh:.1f}%</div>')
        if b_wh is not None and n_seen >= 5:
            txt = ('<span style="color:#4ade80;font-weight:700;">VULNERABLE</span>' if b_wh > p_wh + 5 else
                   '<span style="color:#f87171;font-weight:700;">DANGEROUS</span>' if b_wh < p_wh - 8 else
                   '<span style="color:#fbbf24;">AVERAGE</span>')
            html.append(f'<div class="rpt-p"><b>Batter whiff vs this pitch:</b> {b_wh:.1f}% ({n_seen} seen) — {txt}'
                        + (f', xwOBA {b_xw:.3f}' if b_xw else '') + '</div>')
        elif n_seen and n_seen < 5:
            html.append(f'<div class="rpt-p"><b>Batter sample:</b> only {n_seen} pitches seen — use cautiously.</div>')
        if rank == 1:
            html.append('<div class="rpt-p"><b>✅ Primary out-pitch for this matchup.</b> '
                        'Deploy in 2-strike counts (0-2, 1-2, 2-2) as the put-away pitch.</div>')
        html.append('</div>')

    html.append('<div class="rpt-h2">3. ZONE ATTACK MAP</div>')
    if weak_zones:
        z_str = ", ".join(f"Z{z} ({ZONE_NAMES.get(z,'?')})" for z in weak_zones[:3])
        html.append(f'<div class="rpt-p"><b>🎯 Attack zones (batter whiffs most):</b> {z_str}</div>')
    if avoid_zones:
        a_str = ", ".join(f"Z{z} ({ZONE_NAMES.get(z,'?')})" for z in avoid_zones[:2])
        html.append(f'<div class="rpt-p"><b>🚫 Avoid zones (batter does damage):</b> {a_str}</div>')

    html.append('<div class="rpt-h2">4. COUNT-BY-COUNT SEQUENCE</div>')
    best_pt = ranked[0][0] if ranked else "FF"
    best_nm = PITCH_LONG.get(best_pt, best_pt)
    fb_pt = next((pt for pt, *_ in ranked if pt in ["FF","SI","FC"]), "FF")
    fb_nm = PITCH_LONG.get(fb_pt, fb_pt)
    off_pt = next((pt for pt, *_ in ranked if pt in ["CH","FS","SV"]), None)
    off_nm = PITCH_LONG.get(off_pt, off_pt) if off_pt else None

    sequence = [
        ("0-0", f"{fb_nm} for a called strike, or {best_nm} if you want an early swing-and-miss look."),
        ("0-2", f"Elevated {fb_nm} to freeze, then {best_nm} breaking to the shadow zone. Never groove a fastball."),
        ("1-2", f"Same release point: {fb_nm} in, {best_nm} out of the zone."
                 + (f" Or drop {off_nm} at the exact same arm speed." if off_nm else "")),
        ("2-2", f"{fb_nm} elevated to raise the batter's eyes, then {best_nm} down and away."),
        ("3-1", (f"{off_nm} for a surprise strike — batter is sitting fastball." if off_nm else
                  f"{fb_nm} located precisely — batter is sitting fastball, don't give in.")),
        ("3-2", f"Best {fb_nm} with max velocity and precise location — must throw a strike but make it a good one."),
    ]
    for cnt, advice in sequence:
        html.append(f'<div class="rpt-li">• <b>{cnt}:</b> {advice}</div>')

    html.append('</div>')
    return "\n".join(html)


def rebuild_score(row: pd.Series) -> tuple:
    """Score a pitcher's improvement potential from FanGraphs season stats."""
    score, reasons, profiles = 0.0, [], []

    def g(*cols):
        for c in cols:
            v = safe_num(row.get(c, np.nan))
            if not np.isnan(v):
                return v
        return np.nan

    era, fip = g("ERA"), g("FIP")
    so9, bb9 = g("K/9", "SO9"), g("BB/9", "BB9")
    hr9, age = g("HR/9", "HR9"), g("Age")
    era_plus = g("ERA+")   # derived from FanGraphs' ERA- in load_fg_pitching()

    if not np.isnan(so9):
        if so9 < 6.0:
            score += 28; profiles.append("low_k"); reasons.append(f"K/9 {so9:.1f} — very low; urgently needs out-pitch")
        elif so9 < 7.5:
            score += 18; profiles.append("low_k"); reasons.append(f"K/9 {so9:.1f} — below average")
        elif so9 < 8.5:
            score += 8; profiles.append("low_k"); reasons.append(f"K/9 {so9:.1f} — slightly below average")
        elif so9 > 11.0:
            score -= 6

    if not np.isnan(bb9):
        if bb9 > 4.5:
            score += 20; profiles.append("high_bb"); reasons.append(f"BB/9 {bb9:.1f} — serious command crisis")
        elif bb9 > 3.5:
            score += 10; profiles.append("high_bb"); reasons.append(f"BB/9 {bb9:.1f} — elevated walk rate")
        elif bb9 > 3.0:
            score += 4

    if not np.isnan(fip):
        if fip > 5.0:
            score += 20; profiles.append("poor_fip"); reasons.append(f"FIP {fip:.2f} — well above average")
        elif fip > 4.5:
            score += 12; profiles.append("poor_fip"); reasons.append(f"FIP {fip:.2f} — above average")
        elif fip > 4.0:
            score += 5
        elif fip < 3.5:
            score -= 8

    is_luck = False
    if not (np.isnan(era) or np.isnan(fip)):
        gap = era - fip
        if gap > 1.0 and fip >= 4.0:
            score += min(gap * 8, 14); profiles.append("era_fip_gap")
            reasons.append(f"ERA {era:.2f} > FIP {fip:.2f} (+{gap:.2f}); both elevated")
        elif gap > 1.5 and fip < 4.0:
            is_luck = True; score += 3; profiles.append("era_fip_gap")
            reasons.append(f"ERA {era:.2f} >> FIP {fip:.2f}: likely BABIP/sequencing, NOT arsenal")

    if not np.isnan(era_plus) and era_plus > 0:
        if era_plus < 80:
            score += 8; reasons.append(f"ERA+ {int(era_plus)} — well below league average")
        elif era_plus < 90:
            score += 4
        elif era_plus > 120:
            score -= 4

    if not np.isnan(hr9):
        if hr9 > 2.0:
            score += 10; profiles.append("hr_suppression"); reasons.append(f"HR/9 {hr9:.1f} — elevated")
        elif hr9 > 1.6:
            score += 5; profiles.append("hr_suppression")

    if not np.isnan(age):
        if age <= 25: score += 6
        elif age >= 35: score -= 8

    primary = Counter(profiles).most_common(1)[0][0] if profiles else "low_k"
    only_luck = is_luck and set(profiles) == {"era_fip_gap"}
    return min(max(round(score, 1), 0), 100), reasons[:4], primary, only_luck


# ══════════════════════════════════════════════════════════════════════
# 8. SIDEBAR + HEADER
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:12px 0 8px;border-bottom:1px solid #1e2535;margin-bottom:14px;">'
        '<span style="font-size:2rem;display:block;margin-bottom:4px;">⚾</span>'
        '<div style="color:#79b8ff;font-weight:700;font-size:.95rem;">MLB Statcast Pro</div>'
        '<div style="color:#718096;font-size:.7rem;margin-top:2px;">Live via pybaseball · no local data files</div>'
        '</div>', unsafe_allow_html=True,
    )
    st.markdown('<div style="color:#79b8ff;font-weight:600;font-size:.86rem;margin-bottom:5px;">💪 Arm Slot Guide</div>',
                unsafe_allow_html=True)
    rows_html = ""
    for lo, hi, nm, c in ARM_SLOTS:
        w = max(10, int((hi - lo) * 1.1))
        rows_html += (f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid #1e253570;">'
                      f'<div style="font-family:JetBrains Mono,monospace;font-size:.71rem;color:#63b3ff;min-width:58px;">{lo}–{hi}°</div>'
                      f'<div style="font-size:.75rem;color:#c4cdd8;flex:1;">{nm}</div>'
                      f'<div style="width:{w}px;height:5px;border-radius:3px;background:{c};flex-shrink:0;"></div></div>')
    st.markdown(f'<div class="ref-card">{rows_html}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄  Clear All Cache", width="stretch"):
        st.cache_data.clear()
        st.success("✅  Cache cleared.")
        st.rerun()

st.markdown(
    '<div class="dash-hdr"><div class="dash-ttl">⚾ MLB Statcast Pro Dashboard</div>'
    '<div class="dash-sub">Live Statcast + FanGraphs data via pybaseball · Team/Season rosters · '
    'Velocity/Spin/Movement filters · Zone heatmaps · Arsenal analysis · Batter scouting</div></div>',
    unsafe_allow_html=True,
)

tab_main, tab_pitcher, tab_rebuild, tab_batter = st.tabs(
    ["📊  Dashboard", "🤖  Pitcher Scout", "🏆  Top 10 Rebuild", "🎯  Batter Scout"]
)


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD  (league / team zone heatmap for a date range)
# ══════════════════════════════════════════════════════════════════════

with tab_main:
    st.markdown('<div class="sec-hdr">📊 Zone Heatmap — Live Statcast Pull</div>', unsafe_allow_html=True)
    st.caption(
        "Pulls pitch-level Statcast data directly from Baseball Savant for a date range. "
        "Keep ranges short (≤ 10–14 days) for a fast first load — results are cached after that."
    )

    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        start_d = st.date_input("Start date", value=date.today() - timedelta(days=7), key="dash_start")
    with d2:
        end_d = st.date_input("End date", value=date.today() - timedelta(days=1), key="dash_end")
    with d3:
        team_sel = st.text_input("Team filter (3-letter code, blank = all MLB)", value="", key="dash_team").strip().upper()

    f1, f2, f3, f4 = st.columns(4)
    with f1: dash_stat = st.selectbox("Statistic", STAT_LABELS, key="dash_stat")
    with f2: dash_ph   = st.selectbox("Pitcher Hand", ["All", "R", "L"], key="dash_ph")
    with f3: dash_bh   = st.selectbox("Batter Hand", ["All", "R", "L"], key="dash_bh")
    with f4: dash_cnt  = st.selectbox("Count", ["All"] + ALL_COUNTS, key="dash_cnt")

    dash_velo, dash_spin, dash_hbrk, dash_vbrk = movement_filter_widgets("dash")

    if st.button("▶️  Load League Statcast", key="dash_go"):
        _team_arg = team_sel if team_sel else None
        _raw = load_league_statcast(start_d.isoformat(), end_d.isoformat(), _team_arg)
        st.session_state["dash_raw"] = _raw

    _dash_raw = st.session_state.get("dash_raw", pd.DataFrame())

    if not _dash_raw.empty:
        _dff = _dash_raw.copy()
        if dash_ph != "All" and "p_throws" in _dff.columns: _dff = _dff[_dff["p_throws"] == dash_ph]
        if dash_bh != "All" and "stand" in _dff.columns:    _dff = _dff[_dff["stand"] == dash_bh]
        if dash_cnt != "All" and "count_state" in _dff.columns: _dff = _dff[_dff["count_state"] == dash_cnt]
        _dff = apply_movement_filters(_dff, dash_velo, dash_spin, dash_hbrk, dash_vbrk)

        _zdf = compute_zone_stats_from_raw(_dff)

        col_hm, col_sz = st.columns([5, 4], gap="medium")
        with col_hm:
            title = " · ".join(filter(None, [dash_stat, f"P:{dash_ph}HP" if dash_ph != "All" else None,
                                              f"B:{dash_bh}HB" if dash_bh != "All" else None,
                                              f"Count {dash_cnt}" if dash_cnt != "All" else None]))
            fig = draw_heatmap(_zdf, dash_stat, title)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with col_sz:
            st.markdown('<div style="color:#79b8ff;font-weight:600;font-size:.9rem;margin-bottom:6px;">Sub-Zone Breakdown</div>',
                        unsafe_allow_html=True)
            sz_zone = st.selectbox("Zone (1–9)", list(range(1, 10)), format_func=lambda z: f"Zone {z}", key="dash_sz_zone")
            sz_stat = st.selectbox("Sub-zone statistic", STAT_LABELS, key="dash_sz_stat")
            fig_sz = draw_subzone_panel(_dff, sz_zone, sz_stat)
            st.pyplot(fig_sz, use_container_width=True); plt.close(fig_sz)

        st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Zone-by-Zone Summary</div>', unsafe_allow_html=True)
        if not _zdf.empty:
            _rn = {"total":"Pitches","swing_pct":"Swing%","whiff_pct":"Whiff%","contact_pct":"Contact%",
                   "barrel_pct":"Barrel%","hard_hit_pct":"Hard Hit%","gb_pct":"GB%","avg_xwoba":"xwOBA",
                   "avg_ev":"Exit V","avg_la":"Launch°","avg_velo":"Avg Velo","avg_spin":"Spin rpm",
                   "avg_hbrk":"H-Brk\"","avg_vbrk":"V-Brk\""}
            st.dataframe(_zdf.rename(columns=_rn).sort_values("zone").reset_index(drop=True),
                         width="stretch", height=420)
        else:
            st.info("No data for the current filter combination.")
    else:
        st.info("👆  Pick a date range (and optional team) then click **Load League Statcast**.")

    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — PITCHER SCOUT
# ══════════════════════════════════════════════════════════════════════

with tab_pitcher:
    st.markdown(
        '<div class="dash-hdr"><div class="dash-ttl">🤖 Pitcher Scout</div>'
        '<div class="dash-sub">Team → Season → Pitcher  ·  Full arsenal + zone report  ·  '
        'Pitcher vs Batter matchup plan</div></div>',
        unsafe_allow_html=True,
    )

    _registry = load_registry()

    st.markdown('<div class="sec-hdr">🔍 Select Pitcher</div>', unsafe_allow_html=True)
    sel_c1, sel_c2 = st.columns([1, 1])
    with sel_c1:
        p_season = st.selectbox("Season", AVAILABLE_SEASONS, index=len(AVAILABLE_SEASONS) - 2, key="p_season")
    with sel_c2:
        p_qual = st.number_input("Min innings pitched (roster size filter)", 0, 200, value=10, step=5, key="p_qual")

    _fg_pitch_roster = load_fg_pitching(p_season, qual=0)
    if not _fg_pitch_roster.empty and "IP" in _fg_pitch_roster.columns:
        _fg_pitch_roster = _fg_pitch_roster[pd.to_numeric(_fg_pitch_roster["IP"], errors="coerce") >= p_qual]
    _p_roster = build_team_roster(_fg_pitch_roster)

    sel_c3, sel_c4 = st.columns([1, 3])
    with sel_c3:
        p_team_sel = st.selectbox("Team", ["All Teams"] + sorted(_p_roster.keys()), key="p_team_sel")
    with sel_c4:
        if p_team_sel == "All Teams":
            _p_names = sorted({n for names in _p_roster.values() for n in names})
        else:
            _p_names = _p_roster.get(p_team_sel, [])
        if not _p_names:
            st.caption("No pitchers found for this team/season/IP filter combination.")
        ps_choice = st.selectbox("Pitcher", _p_names if _p_names else ["—"], key="p_pitcher_sel")

    with st.expander("🔎  Can't find them? Search by name directly", expanded=False):
        p_query = st.text_input("Search any pitcher (all-time)", key="p_query", placeholder="e.g. Paul Skenes")
        _p_matches = search_people(p_query, _registry)
        if not _p_matches.empty:
            _p_opts = {f'{r["full_name"]}  (id {int(r["key_mlbam"])})': int(r["key_mlbam"]) for _, r in _p_matches.iterrows()}
            p_choice_search = st.selectbox("Matches", list(_p_opts.keys()), key="p_choice_search")
        else:
            p_choice_search = None

    if st.button("▶️  Load Pitcher Data", key="ps_go"):
        if p_choice_search:
            p_id = _p_opts[p_choice_search]
            p_name = p_choice_search.split("  (id")[0]
        elif ps_choice and ps_choice != "—":
            p_id, note = resolve_mlbam_id(ps_choice, _registry)
            p_name = ps_choice
            if p_id is None:
                st.warning(f"Couldn't resolve '{ps_choice}' to a Statcast ID ({note}). Try the name-search box above.")
        else:
            p_id, p_name = None, None

        if p_id:
            _praw = load_pitcher_statcast(p_id, p_season)
            st.session_state["ps_raw"] = _praw
            st.session_state["ps_pid"] = p_id
            st.session_state["ps_name"] = p_name
            st.session_state["ps_year"] = p_season

    _ps_raw  = st.session_state.get("ps_raw", pd.DataFrame())
    _ps_name = st.session_state.get("ps_name", "")
    _ps_year = st.session_state.get("ps_year", p_season)

    if not _ps_raw.empty:
        _ars, _total, _arm_avg, _ext_avg, _hand = build_pitcher_arsenal(_ps_raw)
        _hand_str = "RHP" if _hand == "R" else "LHP"
        st.markdown(f'<div class="sec-hdr">{_ps_name} ({_hand_str}) — {_ps_year} · {_total:,} pitches</div>',
                    unsafe_allow_html=True)

        if _ars.empty:
            st.warning("No pitch-level data found for this pitcher/season.")
        else:
            fig_ars = plot_arsenal(_ars, _ps_name, _hand)
            st.pyplot(fig_ars, use_container_width=True); plt.close(fig_ars)

            _show_cols = [c for c in ["pitch_name","pitch_type","usage","avg_velo","max_velo",
                                       "avg_spin","avg_h","avg_v","avg_ext","avg_arm","whiff"] if c in _ars.columns]
            _tbl = _ars[_show_cols].copy()
            _tbl.columns = [c.replace("pitch_name","Pitch").replace("pitch_type","Code").replace("usage","Usage%")
                             .replace("avg_velo","AvgV").replace("max_velo","MaxV").replace("avg_spin","Spin")
                             .replace("avg_h","HBrk\"").replace("avg_v","VBrk\"").replace("avg_ext","Ext ft")
                             .replace("avg_arm","Arm°").replace("whiff","Whiff%") for c in _show_cols]
            st.dataframe(_tbl, width="stretch", height=min(380, 60 + len(_ars) * 40))

            st.markdown('<div class="sec-hdr">🗺️ Strike Zone Heatmap</div>', unsafe_allow_html=True)
            fcols = st.columns(4)
            with fcols[0]: hz_stat = st.selectbox("Statistic", STAT_LABELS, key="ps_hstat")
            with fcols[1]: hz_bh   = st.selectbox("vs Hand", ["All","R","L"], key="ps_hbh")
            with fcols[2]: hz_cnt  = st.selectbox("Count", ["All"] + ALL_COUNTS, key="ps_hcnt")
            with fcols[3]:
                _pt_opts_p = sorted(_ps_raw["pitch_type"].dropna().unique().tolist())
                hz_pt = st.selectbox("Pitch Type", ["All"] + _pt_opts_p, key="ps_hpt")

            ps_velo, ps_spin, ps_hbrk, ps_vbrk = movement_filter_widgets("ps")

            _dff = _ps_raw.copy()
            if hz_bh  != "All": _dff = _dff[_dff["stand"] == hz_bh]
            if hz_cnt != "All": _dff = _dff[_dff["count_state"] == hz_cnt]
            if hz_pt  != "All": _dff = _dff[_dff["pitch_type"] == hz_pt]
            _dff = apply_movement_filters(_dff, ps_velo, ps_spin, ps_hbrk, ps_vbrk)
            _hz_zdf = compute_zone_stats_from_raw(_dff)

            hz_col1, hz_col2 = st.columns([1, 1])
            with hz_col1:
                _title = f"{_ps_name} — {hz_stat}"
                fig_hz = draw_heatmap(_hz_zdf, hz_stat, _title)
                st.pyplot(fig_hz, use_container_width=True); plt.close(fig_hz)
            with hz_col2:
                _sz_zone_p = st.selectbox("🔬 Sub-zone detail (1–9)", list(range(1, 10)),
                                           format_func=lambda z: f"Zone {z}", key="ps_sz_zone")
                _sz_stat_p = st.selectbox("Sub-zone stat", STAT_LABELS, key="ps_sz_stat")
                fig_sz = draw_subzone_panel(_dff, _sz_zone_p, _sz_stat_p)
                st.pyplot(fig_sz, use_container_width=True); plt.close(fig_sz)

            with st.expander("📋 Zone Summary Table", expanded=False):
                if not _hz_zdf.empty:
                    st.dataframe(_hz_zdf.sort_values("zone").reset_index(drop=True), width="stretch", height=340)

            st.markdown('<div class="sec-hdr">📋 Full Scouting Report</div>', unsafe_allow_html=True)
            _report_html = generate_pitcher_report(_ps_name, _ars, _total, _arm_avg, _ext_avg, _hand, f"{_ps_year} Season")
            st.markdown(_report_html, unsafe_allow_html=True)
            _report_txt = re.sub(r"<[^>]+>", "", _report_html).replace("&amp;", "&")
            st.download_button("📥  Download Scout Report (.txt)", data=_report_txt,
                                file_name=f"{_ps_name.replace(' ','_')}_{_ps_year}_scout.txt",
                                mime="text/plain", key="ps_dl")

            # ── Matchup vs a specific batter ────────────────────────
            st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-hdr">⚔️ Matchup vs. a Batter</div>', unsafe_allow_html=True)

            vb0a, vb0b = st.columns([1, 1])
            with vb0a:
                vb_season = st.selectbox("Batter season", AVAILABLE_SEASONS, index=len(AVAILABLE_SEASONS) - 2, key="vb_season")
            with vb0b:
                vb_qual = st.number_input("Min plate appearances (roster filter)", 0, 700, value=50, step=25, key="vb_qual")

            _fg_bat_roster = load_fg_batting(vb_season, qual=0)
            if not _fg_bat_roster.empty and "PA" in _fg_bat_roster.columns:
                _fg_bat_roster = _fg_bat_roster[pd.to_numeric(_fg_bat_roster["PA"], errors="coerce") >= vb_qual]
            _b_roster = build_team_roster(_fg_bat_roster)

            vb1, vb2 = st.columns([1, 3])
            with vb1:
                vb_team_sel = st.selectbox("Opposing team", ["All Teams"] + sorted(_b_roster.keys()), key="vb_team_sel")
            with vb2:
                if vb_team_sel == "All Teams":
                    _vb_names = sorted({n for names in _b_roster.values() for n in names})
                else:
                    _vb_names = _b_roster.get(vb_team_sel, [])
                vb_choice = st.selectbox("Batter", _vb_names if _vb_names else ["—"], key="vb_choice")

            with st.expander("🔎  Can't find them? Search by name directly", expanded=False):
                b_query = st.text_input("Search any batter (all-time)", key="vb_query", placeholder="e.g. Aaron Judge")
                _b_matches = search_people(b_query, _registry)
                if not _b_matches.empty:
                    _b_opts = {f'{r["full_name"]}  (id {int(r["key_mlbam"])})': int(r["key_mlbam"]) for _, r in _b_matches.iterrows()}
                    b_choice_search = st.selectbox("Matches", list(_b_opts.keys()), key="vb_choice_search")
                else:
                    b_choice_search = None

            if st.button("🎯  Generate Matchup Plan", key="vb_go"):
                if b_choice_search:
                    b_id, b_name = _b_opts[b_choice_search], b_choice_search.split("  (id")[0]
                elif vb_choice and vb_choice != "—":
                    b_id, note = resolve_mlbam_id(vb_choice, _registry)
                    b_name = vb_choice
                    if b_id is None:
                        st.warning(f"Couldn't resolve '{vb_choice}' to a Statcast ID ({note}).")
                else:
                    b_id, b_name = None, None

                if b_id:
                    _braw = load_batter_statcast(b_id, vb_season)
                    _bstats = analyze_batter(_braw) if not _braw.empty else {}
                    st.session_state["vb_bstats"] = _bstats
                    st.session_state["vb_bname"] = b_name
                    st.session_state["vb_bstand"] = _bstats.get("stand", "R")

            _vb_bstats = st.session_state.get("vb_bstats", {})
            _vb_bname  = st.session_state.get("vb_bname", "")
            _vb_bstand = st.session_state.get("vb_bstand", "R")

            if _vb_bstats:
                ov1, ov2 = st.columns([1, 1])
                with ov1:
                    _bz = _vb_bstats.get("zone_stats", pd.DataFrame())
                    fig_bz = draw_heatmap(_bz, "Whiff %", f"{_vb_bname} — Whiff% by Zone", batter_mode=True)
                    st.pyplot(fig_bz, use_container_width=True); plt.close(fig_bz)
                with ov2:
                    _bpt = _vb_bstats.get("pt_stats", pd.DataFrame())
                    if not _bpt.empty:
                        _cols = [c for c in ["pitch_name","total","whiff_pct","avg_xwoba","avg_ev","avg_velo"] if c in _bpt.columns]
                        st.caption(f"**{_vb_bname}** — pitch vulnerability")
                        st.dataframe(_bpt[_cols].sort_values("whiff_pct", ascending=False, na_position="last")
                                     .reset_index(drop=True), width="stretch", height=260)

                st.markdown("---")
                _matchup_html = generate_matchup_plan(_ps_name, _hand, _ars, _vb_bname, _vb_bstand, _vb_bstats)
                st.markdown(_matchup_html, unsafe_allow_html=True)
                _matchup_txt = re.sub(r"<[^>]+>", "", _matchup_html).replace("&amp;", "&")
                st.download_button("📥  Download Matchup Plan (.txt)", data=_matchup_txt,
                                    file_name=f"{_ps_name.replace(' ','_')}_vs_{_vb_bname.replace(' ','_')}.txt",
                                    mime="text/plain", key="vb_dl")
    else:
        st.info("👆  Pick a team/season/pitcher above (or search by name), then click **Load Pitcher Data**.")

    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — TOP 10 REBUILD  (FanGraphs season stats)
# ══════════════════════════════════════════════════════════════════════

with tab_rebuild:
    st.markdown(
        '<div class="dash-hdr"><div class="dash-ttl">🏆 Top 10 Arsenal Rebuild Candidates</div>'
        '<div class="dash-sub">FanGraphs season pitching stats (live pull) · '
        'ERA-FIP gap correctly weighted vs. genuine arsenal problems</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="citation" style="margin-bottom:12px;">{BR_CITATION}</div>', unsafe_allow_html=True)

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
| ERA+ below 90 | up to **8 pts** | Derived from FanGraphs' ERA- (ERA+ ≈ 10000 / ERA-) |
| HR/9 above 1.6 | up to **10 pts** | |
| Age ≤ 25 | +6 pts | Higher rebuild upside |
| Age ≥ 35 | −8 pts | Lower ceiling |
        """)

    rb_c = st.columns([1, 1, 2, 1])
    with rb_c[0]: rb_season = st.selectbox("Season", AVAILABLE_SEASONS, index=len(AVAILABLE_SEASONS) - 2, key="rb_yr")
    with rb_c[1]: rb_minip  = st.number_input("Min IP", 5, 250, value=20, step=5, key="rb_ip")
    with rb_c[2]: rb_team   = st.text_input("Team filter (e.g. NYY,LAD — blank=all MLB)", key="rb_team")
    with rb_c[3]: rb_topn   = st.number_input("Show top N", 5, 50, value=10, key="rb_n")

    if st.button("🔄  Rank Pitchers", key="rb_go"):
        _p_df = load_fg_pitching(rb_season, qual=0)
        if not _p_df.empty:
            if "IP" in _p_df.columns:
                _p_df = _p_df[pd.to_numeric(_p_df["IP"], errors="coerce") >= rb_minip]
            if "FIP" in _p_df.columns:
                _p_df = _p_df[pd.to_numeric(_p_df["FIP"], errors="coerce") < 15]
            if rb_team.strip() and "Team" in _p_df.columns:
                _teams = [t.strip().upper() for t in rb_team.split(",")]
                _p_df = _p_df[_p_df["Team"].astype(str).str.upper().isin(_teams)]

        if _p_df.empty:
            st.warning("No pitchers match the criteria (or FanGraphs pull failed).")
            st.session_state[f"rb_{rb_season}"] = []
        else:
            _rows = []
            for _, row in _p_df.iterrows():
                sc, rsns, prof, only_luck = rebuild_score(row)
                _rows.append({
                    "name": row.get("Name", "?"), "team": row.get("Team", "?"),
                    "ip": safe_num(row.get("IP", 0)), "era": safe_num(row.get("ERA")),
                    "fip": safe_num(row.get("FIP")), "era_p": safe_num(row.get("ERA+")),
                    "so9": safe_num(row.get("K/9")), "bb9": safe_num(row.get("BB/9")),
                    "hr9": safe_num(row.get("HR/9")), "whip": safe_num(row.get("WHIP")),
                    "age": safe_num(row.get("Age")), "score": sc, "reasons": rsns,
                    "profile": prof, "only_luck": only_luck,
                })
            _rows.sort(key=lambda x: x["score"], reverse=True)
            st.session_state[f"rb_{rb_season}"] = _rows
            st.success(f"✅  Ranked {len(_rows)} pitchers")

    _rb_results = st.session_state.get(f"rb_{rb_season}")

    if _rb_results:
        _top = _rb_results[: int(rb_topn)]

        def _fv(v, fmt="{:.2f}"):
            return fmt.format(v) if not (isinstance(v, float) and (np.isnan(v) or v <= 0)) else "—"

        _summ = [{"#": i + 1, "Name": r["name"], "Team": r["team"], "IP": f'{r["ip"]:.0f}',
                  "ERA": _fv(r["era"]), "FIP": _fv(r["fip"]), "ERA+": _fv(r["era_p"], "{:.0f}"),
                  "K/9": _fv(r["so9"]), "BB/9": _fv(r["bb9"]), "WHIP": _fv(r["whip"]),
                  "Age": _fv(r["age"], "{:.0f}"), "Score": r["score"]} for i, r in enumerate(_top)]
        st.dataframe(pd.DataFrame(_summ).set_index("#"), width="stretch", height=400)

        st.markdown('<div class="sec-hdr">Detailed Breakdown</div>', unsafe_allow_html=True)
        for rank, r in enumerate(_top, 1):
            sc_pct = min(r["score"], 100)
            bc = "#ef4444" if sc_pct > 65 else "#f97316" if sc_pct > 45 else "#eab308" if sc_pct > 30 else "#22c55e"
            with st.expander(
                f"#{rank}  {r['name']}  ({r['team']})  —  Score: {r['score']:.1f}  |  "
                f"ERA {_fv(r['era'])}  FIP {_fv(r['fip'])}  K/9 {_fv(r['so9'])}  BB/9 {_fv(r['bb9'])}  IP {r['ip']:.0f}",
                expanded=(rank <= 3),
            ):
                if r.get("only_luck"):
                    st.warning("⚠️  Score driven mainly by ERA-FIP gap despite a solid FIP — likely luck/BABIP, not an arsenal problem.")

                cl, cr = st.columns([2, 3])
                with cl:
                    _prof_info = REBUILD_PROFILES.get(r["profile"], {})
                    st.markdown(f"""
                    <div class="ref-card">
                        <div style="font-size:1.8rem;font-weight:800;color:{bc};line-height:1;">
                            {r['score']:.1f} <span style="font-size:.85rem;color:#8892a4;">/100</span>
                        </div>
                        <div class="score-wrap"><div class="score-bar" style="width:{sc_pct}%;background:{bc};"></div></div>
                        <div style="color:#8892a4;font-size:.76rem;font-weight:600;margin-top:10px;">KEY ISSUES:</div>
                        {"".join(f'<div style="color:#c9d1d9;font-size:.79rem;margin:3px 0;">• {rs}</div>' for rs in r["reasons"])}
                    </div>""", unsafe_allow_html=True)
                    if _prof_info:
                        st.markdown(f"""
                        <div class="ref-card" style="margin-top:0;">
                            <div class="ref-title">🔧 Rebuild Profile</div>
                            <div style="color:#c9d1d9;font-size:.84rem;font-weight:700;margin-bottom:4px;">{_prof_info.get('title','')}</div>
                            <div style="color:#a0aec0;font-size:.78rem;line-height:1.45;margin-bottom:5px;">{_prof_info.get('desc','')}</div>
                            <div style="color:#63b3ff;font-size:.78rem;"><b>Fix:</b> {_prof_info.get('fix','')}</div>
                            <div style="color:#718096;font-size:.73rem;margin-top:3px;">⏱ {_prof_info.get('timeline','')}</div>
                        </div>""", unsafe_allow_html=True)

                with cr:
                    st.markdown('<div class="report-wrap">', unsafe_allow_html=True)
                    era, fip, so9, bb9 = r["era"], r["fip"], r["so9"], r["bb9"]
                    if not (np.isnan(era) or np.isnan(fip)):
                        gap = era - fip
                        if gap > 1.5 and fip < 4.0:
                            st.markdown(f'<div class="warn-box"><div class="rpt-p">🍀 ERA-FIP gap {gap:.2f} with good FIP {fip:.2f} — '
                                        f'primarily a luck/BABIP issue.</div></div>', unsafe_allow_html=True)
                        elif gap > 0.5 and fip >= 4.0:
                            st.markdown(f'<div class="danger-box"><div class="rpt-p">🔴 ERA {era:.2f} > FIP {fip:.2f} — '
                                        f'both metrics elevated, genuine stuff problem.</div></div>', unsafe_allow_html=True)
                        elif gap < -0.5:
                            st.markdown('<div class="ok-box"><div class="rpt-p">✅ ERA below FIP — outperforming underlying metrics.</div></div>',
                                        unsafe_allow_html=True)
                    if not np.isnan(so9) and so9 < 7.0:
                        st.markdown(f'<div class="danger-box"><div class="rpt-p">🔴 K/9 {so9:.1f} — critically low; '
                                    f'urgently needs a reliable miss-bat pitch.</div></div>', unsafe_allow_html=True)
                    if not np.isnan(bb9) and bb9 > 3.5:
                        st.markdown(f'<div class="warn-box"><div class="rpt-p">🟡 BB/9 {bb9:.1f} — elevated; '
                                    f'focus on 3-1/3-2 counts and a reliable zone-attack secondary.</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                if st.button(f"📡  Load Live Statcast for {r['name']}", key=f"rb_sc_{rank}_{rb_season}"):
                    _match = search_people(r["name"], load_registry(), limit=5)
                    if not _match.empty:
                        _found_pid = int(_match.iloc[0]["key_mlbam"])
                        _rb_raw = load_pitcher_statcast(_found_pid, rb_season)
                        if not _rb_raw.empty:
                            _rb_ars, _rb_tot, _rb_arm, _rb_ext, _rb_hand = build_pitcher_arsenal(_rb_raw)
                            fig_rb = plot_arsenal(_rb_ars, r["name"], _rb_hand)
                            st.pyplot(fig_rb, use_container_width=True); plt.close(fig_rb)
                            st.markdown(generate_pitcher_report(r["name"], _rb_ars, _rb_tot, _rb_arm, _rb_ext,
                                                                 _rb_hand, f"{rb_season}"), unsafe_allow_html=True)
                        else:
                            st.warning("No Statcast data found for this pitcher/season.")
                    else:
                        st.warning("Could not match this name in the player registry.")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — BATTER SCOUT
# ══════════════════════════════════════════════════════════════════════

with tab_batter:
    st.markdown(
        '<div class="dash-hdr"><div class="dash-ttl">🎯 Batter Scout</div>'
        '<div class="dash-sub">Team → Season → Batter  ·  zone heatmap + sub-zone detail  ·  '
        'pitch-type vulnerability · count tendencies · platoon splits · AI pitching plan</div></div>',
        unsafe_allow_html=True,
    )

    _registry_b = load_registry()

    st.markdown('<div class="sec-hdr">🔍 Select Batter</div>', unsafe_allow_html=True)
    bs0a, bs0b = st.columns([1, 1])
    with bs0a:
        bs_season = st.selectbox("Season", AVAILABLE_SEASONS, index=len(AVAILABLE_SEASONS) - 2, key="bs_season")
    with bs0b:
        bs_qual = st.number_input("Min plate appearances (roster filter)", 0, 700, value=50, step=25, key="bs_qual")

    _fg_bat_roster2 = load_fg_batting(bs_season, qual=0)
    if not _fg_bat_roster2.empty and "PA" in _fg_bat_roster2.columns:
        _fg_bat_roster2 = _fg_bat_roster2[pd.to_numeric(_fg_bat_roster2["PA"], errors="coerce") >= bs_qual]
    _b_roster2 = build_team_roster(_fg_bat_roster2)

    bs1, bs2 = st.columns([1, 3])
    with bs1:
        bs_team_sel = st.selectbox("Team", ["All Teams"] + sorted(_b_roster2.keys()), key="bs_team_sel")
    with bs2:
        if bs_team_sel == "All Teams":
            _bs_names = sorted({n for names in _b_roster2.values() for n in names})
        else:
            _bs_names = _b_roster2.get(bs_team_sel, [])
        if not _bs_names:
            st.caption("No batters found for this team/season/PA filter combination.")
        bs_choice = st.selectbox("Batter", _bs_names if _bs_names else ["—"], key="bs_batter_sel")

    with st.expander("🔎  Can't find them? Search by name directly", expanded=False):
        b_query2 = st.text_input("Search any batter (all-time)", key="bs_query", placeholder="e.g. Shohei Ohtani")
        _b_matches2 = search_people(b_query2, _registry_b)
        if not _b_matches2.empty:
            _b_opts2 = {f'{r["full_name"]}  (id {int(r["key_mlbam"])})': int(r["key_mlbam"]) for _, r in _b_matches2.iterrows()}
            bs_choice_search = st.selectbox("Matches", list(_b_opts2.keys()), key="bs_choice_search")
        else:
            bs_choice_search = None

    if st.button("▶️  Load Batter Data", key="bs_go"):
        if bs_choice_search:
            bs_id, bs_name = _b_opts2[bs_choice_search], bs_choice_search.split("  (id")[0]
        elif bs_choice and bs_choice != "—":
            bs_id, note = resolve_mlbam_id(bs_choice, _registry_b)
            bs_name = bs_choice
            if bs_id is None:
                st.warning(f"Couldn't resolve '{bs_choice}' to a Statcast ID ({note}). Try the name-search box above.")
        else:
            bs_id, bs_name = None, None

        if bs_id:
            _braw = load_batter_statcast(bs_id, bs_season)
            st.session_state["bs_raw"] = _braw
            st.session_state["bs_name"] = bs_name
            st.session_state["bs_year"] = bs_season

    _bs_raw  = st.session_state.get("bs_raw", pd.DataFrame())
    _bs_name = st.session_state.get("bs_name", "")
    _bs_year = st.session_state.get("bs_year", bs_season)

    if not _bs_raw.empty:
        st.markdown(f'<div class="sec-hdr">🎯 {_bs_name} — {_bs_year}</div>', unsafe_allow_html=True)

        bf1, bf2, bf3, bf4 = st.columns([1, 1, 2, 2])
        with bf1: bs_ph  = st.selectbox("vs Pitcher Hand", ["All","R","L"], key="bs_ph")
        with bf2: bs_cnt = st.selectbox("Count", ["All"] + ALL_COUNTS, key="bs_cnt")
        with bf3:
            _pt_opts_b = sorted(_bs_raw["pitch_type"].dropna().unique().tolist())
            bs_pt = st.selectbox("Pitch Type", ["All"] + _pt_opts_b, key="bs_pt")
        with bf4:
            bs_stat = st.selectbox("Heatmap Statistic", STAT_LABELS, key="bs_stat")

        bs_velo, bs_spin, bs_hbrk, bs_vbrk = movement_filter_widgets("bs")

        _bff = _bs_raw.copy()
        if bs_ph  != "All": _bff = _bff[_bff["p_throws"] == bs_ph]
        if bs_cnt != "All": _bff = _bff[_bff["count_state"] == bs_cnt]
        if bs_pt  != "All": _bff = _bff[_bff["pitch_type"] == bs_pt]
        _bff = apply_movement_filters(_bff, bs_velo, bs_spin, bs_hbrk, bs_vbrk)

        _bstats = analyze_batter(_bff) if not _bff.empty else {}

        if not _bstats:
            st.warning("No data for the current filter combination.")
        else:
            _bzdf = _bstats.get("zone_stats", pd.DataFrame())
            _bpt  = _bstats.get("pt_stats", pd.DataFrame())
            _bcnt = _bstats.get("count_stats", pd.DataFrame())
            _bplt = _bstats.get("platoon", {})
            total_p = _bstats.get("total_pitches", 0)

            sw_all = (_bzdf["swings"].sum() / max(_bzdf["total"].sum(), 1) * 100) if not _bzdf.empty else np.nan
            wh_all = (_bzdf["whiffs"].sum() / max(_bzdf["swings"].sum(), 1) * 100) if not _bzdf.empty else np.nan
            xw_all = _bzdf["avg_xwoba"].mean() if not _bzdf.empty else np.nan
            ev_all = _bzdf["avg_ev"].mean() if not _bzdf.empty else np.nan

            def _mc(lbl, val, sub=""):
                return (f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                        f'<div class="metric-val">{val}</div><div class="metric-sub">{sub}</div></div>')

            _m_html = (_mc("Pitches Seen", f"{total_p:,}", f"{_bs_year}")
                       + _mc("Swing %", f"{sw_all:.1f}%" if not np.isnan(sw_all) else "—")
                       + _mc("Whiff %", f"{wh_all:.1f}%" if not np.isnan(wh_all) else "—", "on swings")
                       + _mc("xwOBA", f"{xw_all:.3f}" if not np.isnan(xw_all) else "—", "expected")
                       + _mc("Exit Velo", f"{ev_all:.1f}" if not np.isnan(ev_all) else "—", "mph"))
            if _bplt.get("R") and _bplt.get("L"):
                _m_html += _mc("vs RHP Whiff", f'{_bplt["R"].get("whiff_pct",0):.1f}%')
                _m_html += _mc("vs LHP Whiff", f'{_bplt["L"].get("whiff_pct",0):.1f}%')
            st.markdown(f'<div class="metric-grid">{_m_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sec-hdr">🗺️ Strike Zone Heatmap + Sub-Zone Detail</div>', unsafe_allow_html=True)
            sz_zone_b = st.selectbox("🔬 Sub-zone detail (1–9)", list(range(1, 10)),
                                      format_func=lambda z: f"Zone {z}", key="bs_sz_zone")
            sz_stat_b = st.selectbox("Sub-zone stat", STAT_LABELS, key="bs_sz_stat")

            col_hm_b, col_sz_b = st.columns([1, 1], gap="medium")
            with col_hm_b:
                _title_b = f"{_bs_name} — {_bs_year} · {bs_stat}"
                fig_bs_hm = draw_heatmap(_bzdf, bs_stat, _title_b, batter_mode=True)
                st.pyplot(fig_bs_hm, use_container_width=True); plt.close(fig_bs_hm)
            with col_sz_b:
                fig_sz_b = draw_subzone_panel(_bff, sz_zone_b, sz_stat_b)
                st.pyplot(fig_sz_b, use_container_width=True); plt.close(fig_sz_b)

            st.markdown('<div class="sec-hdr">🎯 Pitch-Type Vulnerability</div>', unsafe_allow_html=True)
            if not _bpt.empty:
                _bpt_cols = [c for c in ["pitch_name","usage","total","whiff_pct","swing_pct","avg_xwoba",
                                          "avg_ev","avg_velo","avg_spin","avg_hbrk","avg_vbrk"] if c in _bpt.columns]
                _rn2 = {"pitch_name":"Pitch","usage":"Seen%","total":"Pitches","whiff_pct":"Whiff%",
                        "swing_pct":"Swing%","avg_xwoba":"xwOBA","avg_ev":"Exit V","avg_velo":"Avg Velo",
                        "avg_spin":"Spin rpm","avg_hbrk":"H-Brk\"","avg_vbrk":"V-Brk\""}
                st.dataframe(_bpt[_bpt_cols].rename(columns=_rn2)
                             .sort_values("Whiff%" if "Whiff%" in _rn2.values() else _bpt_cols[0],
                                          ascending=False, na_position="last")
                             .reset_index(drop=True), width="stretch",
                             height=min(420, 60 + len(_bpt) * 40))

                _bpt_sorted = _bpt.dropna(subset=["whiff_pct"]).sort_values("whiff_pct", ascending=False)
                if not _bpt_sorted.empty:
                    fig_pt, ax_pt = plt.subplots(figsize=(9.5, 3.3))
                    fig_pt.patch.set_facecolor("#0b0f17"); ax_pt.set_facecolor("#111621")
                    colors_pt = [PITCH_COLORS.get(pt, "#94a3b8") for pt in _bpt_sorted["pitch_type"]]
                    labels_pt = _bpt_sorted.get("pitch_name", _bpt_sorted["pitch_type"])
                    bars_pt = ax_pt.barh(labels_pt, _bpt_sorted["whiff_pct"], color=colors_pt,
                                         edgecolor="#1e2535", height=0.65)
                    for bar, val in zip(bars_pt, _bpt_sorted["whiff_pct"]):
                        ax_pt.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                                   va="center", fontsize=8.5, fontweight="bold", color="#e2e8f0")
                    ax_pt.set_xlabel("Whiff % (higher = more vulnerable)", color="#a0aec0", fontsize=9)
                    ax_pt.invert_yaxis(); ax_pt.tick_params(colors="#8892a4", labelsize=8.5)
                    ax_pt.set_xlim(0, _bpt_sorted["whiff_pct"].max() + 18)
                    for sp in ax_pt.spines.values(): sp.set_edgecolor("#2a3545")
                    st.pyplot(fig_pt, use_container_width=True); plt.close(fig_pt)
            else:
                st.info("No pitch-type data for current filters.")

            st.markdown('<div class="sec-hdr">📊 Count-Based Tendencies</div>', unsafe_allow_html=True)
            if not _bcnt.empty:
                with st.expander("📋 Count table", expanded=True):
                    st.dataframe(_bcnt.rename(columns={"count_state":"Count","pitches":"Pitches",
                                                        "swing_pct":"Swing%","whiff_pct":"Whiff%","avg_xwoba":"xwOBA"})
                                 .sort_values("Count").reset_index(drop=True), width="stretch", height=350)

            st.markdown('<div class="sec-hdr">🔄 Platoon Splits</div>', unsafe_allow_html=True)
            if _bplt:
                _plt_rows = [{"vs": "RHP" if h == "R" else "LHP", "Pitches": int(d.get("pitches", 0)),
                              "Swing%": round(d.get("swing_pct", 0), 1), "Whiff%": round(d.get("whiff_pct", 0), 1),
                              "xwOBA": round(d.get("avg_xwoba", np.nan) or np.nan, 3)}
                             for h, d in sorted(_bplt.items())]
                st.dataframe(pd.DataFrame(_plt_rows).set_index("vs"), width="stretch", height=120)

            with st.expander("📋 Full Zone-by-Zone Summary", expanded=False):
                if not _bzdf.empty:
                    st.dataframe(_bzdf.sort_values("zone").reset_index(drop=True), width="stretch", height=400)

            st.markdown(f'<div class="sec-hdr">🤖 AI Pitching Plan — How to Attack {_bs_name}</div>', unsafe_allow_html=True)
            _plan_html = generate_pitching_plan(_bstats, _bs_name)
            st.markdown(_plan_html, unsafe_allow_html=True)
            _plan_txt = re.sub(r"<[^>]+>", "", _plan_html).replace("&amp;", "&")
            st.download_button("📥  Download Pitching Plan (.txt)", data=_plan_txt,
                                file_name=f"{_bs_name.replace(' ','_')}_{_bs_year}_pitching_plan.txt",
                                mime="text/plain", key="bs_dl")
    else:
        st.info("👆  Pick a team/season/batter above (or search by name), then click **Load Batter Data**.")

    st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="citation">{BR_CITATION}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# 9. FOOTER
# ══════════════════════════════════════════════════════════════════════

st.markdown('<div class="dash-divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="citation">
    MLB Statcast Pro · Built with Streamlit + pybaseball (no local data files)<br>
    {BR_CITATION}
</div>
""", unsafe_allow_html=True)
   

