"""
precompute.py  —  Run once before deploying to Streamlit Cloud.

Generates 3 small parquet files that make the dashboard start instantly:
  • meta_maps.parquet       : batter/pitcher names, teams, season rosters
  • zone_stats_agg.parquet  : pre-aggregated Tab 1 heatmap data

Usage:
    python precompute.py
    # Then commit all *.parquet files to your GitHub repo.

After running, the dashboard reads these cached files instead of
re-computing from raw statcast on every cold start.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR  = Path(".")          # Parquet files are in the same directory
ALL_YEARS = (2024, 2025, 2026)

STATCAST_FILES = {yr: DATA_DIR / f"statcast_{yr}.parquet" for yr in ALL_YEARS}

# Verb pattern for batter name extraction from 'des'
_VERB_PAT = re.compile(
    r'\b(lines|flies|grounds|strikes|walks|singles|doubles|triples|homers|'
    r'pops|grounded|flied|lined|singled|doubled|tripled|homered|popped|'
    r'struck|walked|reaches|scores|bunts|fouled|batter|intentionally|hit by|called)\b',
    re.IGNORECASE,
)

SWING_EV   = {"swinging_strike","swinging_strike_blocked","foul","foul_tip",
               "hit_into_play","hit_into_play_no_out","hit_into_play_score"}
WHIFF_EV   = {"swinging_strike","swinging_strike_blocked"}
CONTACT_EV = {"foul","foul_tip","hit_into_play","hit_into_play_no_out","hit_into_play_score"}

# ── STEP 1: Build meta maps + team rosters ──────────────────────────

print("Building player meta maps and team rosters…")

batter_rows  = []   # one row per (batter_id, season, team)
pitcher_rows = []   # one row per (pitcher_id, season, team)

for yr in ALL_YEARS:
    path = STATCAST_FILES.get(yr)
    if not path or not path.exists():
        print(f"  Skipping {yr} — file not found: {path}")
        continue
    print(f"  Loading {path.name}…")

    meta_cols = ["batter", "pitcher", "player_name", "des",
                 "stand", "p_throws", "home_team", "away_team", "inning_topbot"]
    df = pd.read_parquet(path, engine="pyarrow", columns=meta_cols)

    # Determine pitcher team
    df["pitcher_team"] = np.where(df["inning_topbot"] == "Top",
                                   df["home_team"], df["away_team"])
    # Determine batter team
    df["batter_team"]  = np.where(df["inning_topbot"] == "Top",
                                   df["away_team"], df["home_team"])

    # ── Pitcher rows ─────────────────────────────────────────────────
    pm = (df[["pitcher","player_name","p_throws","pitcher_team"]]
          .dropna(subset=["player_name","pitcher_team"])
          .drop_duplicates(["pitcher","pitcher_team"]))
    for _, row in pm.iterrows():
        pitcher_rows.append({
            "pitcher_id": int(row["pitcher"]),
            "raw_name":   str(row["player_name"]),
            "hand":       str(row["p_throws"]),
            "team":       str(row["pitcher_team"]),
            "season":     yr,
        })

    # ── Batter rows ──────────────────────────────────────────────────
    bm = (df[["batter","stand","batter_team"]]
          .dropna(subset=["batter_team"])
          .drop_duplicates(["batter","batter_team"]))
    # Batter names from des
    first_des = (df[df["des"].notna() & (df["des"].str.len() > 5)]
                 .groupby("batter")["des"].first())
    batter_name_map = {}
    for bid_raw, des in first_des.items():
        bid = int(bid_raw)
        m = _VERB_PAT.search(str(des))
        if m:
            name_part = des[:m.start()].strip()
            words = name_part.split()
            if 2 <= len(words) <= 4:
                batter_name_map[bid] = name_part

    for _, row in bm.iterrows():
        bid = int(row["batter"])
        batter_rows.append({
            "batter_id":  bid,
            "name":       batter_name_map.get(bid, f"Batter #{bid}"),
            "stand":      str(row.get("stand", "?")),
            "team":       str(row["batter_team"]),
            "season":     yr,
        })

# Deduplicate (keep first occurrence per id+season+team)
batter_df  = (pd.DataFrame(batter_rows)
              .drop_duplicates(["batter_id","season","team"])
              .reset_index(drop=True))
pitcher_df = (pd.DataFrame(pitcher_rows)
              .drop_duplicates(["pitcher_id","season","team"])
              .reset_index(drop=True))

# Convert player_name "Last, First" → "First Last" for pitchers
def _fmt(s):
    parts = str(s).split(",")
    if len(parts) == 2:
        return f"{parts[1].strip()} {parts[0].strip()}"
    return str(s).strip()

pitcher_df["display_name"] = pitcher_df["raw_name"].apply(_fmt)

print(f"  → {len(batter_df)} batter rows, {len(pitcher_df)} pitcher rows")

# Save
batter_df.to_parquet(DATA_DIR / "meta_batters.parquet", engine="pyarrow", index=False)
pitcher_df.to_parquet(DATA_DIR / "meta_pitchers.parquet", engine="pyarrow", index=False)
print("  Saved meta_batters.parquet and meta_pitchers.parquet")


# ── STEP 2: Pre-aggregate zone stats ────────────────────────────────

print("\nPre-aggregating zone statistics for Tab 1…")

agg_parts = []
for yr in ALL_YEARS:
    path = STATCAST_FILES.get(yr)
    if not path or not path.exists():
        continue
    print(f"  Aggregating {path.name}…")
    agg_cols = ["pitch_type","zone","stand","p_throws","balls","strikes",
                "description","launch_speed","launch_angle",
                "estimated_woba_using_speedangle"]
    df2 = pd.read_parquet(path, engine="pyarrow", columns=agg_cols)
    df2["game_year"]    = yr
    df2["count_state"]  = df2["balls"].astype(str) + "-" + df2["strikes"].astype(str)
    desc = df2["description"].fillna("")
    df2["is_swing"]  = desc.isin(SWING_EV).astype("int8")
    df2["is_whiff"]  = desc.isin(WHIFF_EV).astype("int8")
    df2["is_contact"]= desc.isin(CONTACT_EV).astype("int8")
    ls = pd.to_numeric(df2["launch_speed"], errors="coerce")
    la = pd.to_numeric(df2["launch_angle"], errors="coerce")
    df2["is_barrel"] = ((ls >= 98) & la.between(26, 30)).fillna(False).astype("int8")
    df2["is_hh"]     = (ls >= 95).fillna(False).astype("int8")
    df2["is_gb"]     = (la < 10).fillna(False).astype("int8")
    df2 = df2[df2["zone"].between(1, 14)]
    agg_parts.append(df2)

full = pd.concat(agg_parts, ignore_index=True)
grp  = full.groupby(
    ["game_year","zone","pitch_type","p_throws","stand","count_state"],
    dropna=False, as_index=False,
).agg(
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

grp.to_parquet(DATA_DIR / "zone_stats_agg.parquet", engine="pyarrow", index=False)
print(f"  Saved zone_stats_agg.parquet ({len(grp):,} rows)")

print("\n✅ Precompute complete. Commit these files to GitHub:")
print("   meta_batters.parquet")
print("   meta_pitchers.parquet")
print("   zone_stats_agg.parquet")
