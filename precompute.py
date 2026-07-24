"""
precompute.py  —  Run once locally BEFORE deploying to Streamlit Cloud.

Generates small parquet files that make the dashboard start instantly:
  • meta_batters.parquet    : batter names, teams, seasons
  • meta_pitchers.parquet   : pitcher names, hand, teams, seasons
  • zone_stats_agg.parquet  : pre-aggregated Tab 1 heatmap data

IMPORTANT: this version reads MONTHLY files
    statcast_{year}_{month:02d}.parquet   (month = 3..10)
to match the file layout used in AI.py. If your data is laid out
differently, adjust STATCAST_FILES below.

Usage:
    python precompute.py
    # Then commit the 3 generated *.parquet files to your GitHub repo,
    # next to the monthly statcast_*.parquet files.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

DATA_DIR  = Path(".")
ALL_YEARS = (2024, 2025, 2026)

# Monthly file layout — must match AI.py's STATCAST_FILES
STATCAST_FILES = {
    year: [
        DATA_DIR / f"statcast_{year}_{month:02d}.parquet"
        for month in range(3, 11)
    ]
    for year in ALL_YEARS
}
for year in list(STATCAST_FILES.keys()):
    STATCAST_FILES[year] = [f for f in STATCAST_FILES[year] if f.exists()]
    if not STATCAST_FILES[year]:
        del STATCAST_FILES[year]

_VERB_PAT = re.compile(
    r'\b(lines|flies|grounds|strikes|walks|singles|doubles|triples|homers|'
    r'pops|grounded|flied|lined|singled|doubled|tripled|homered|popped|'
    r'struck|walked|reaches|scores|bunts|fouled|batter|intentionally|hit by|called)\b',
    re.IGNORECASE,
)

SWING_EV   = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
              "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
WHIFF_EV   = {"swinging_strike", "swinging_strike_blocked"}
CONTACT_EV = {"foul", "foul_tip", "hit_into_play",
              "hit_into_play_no_out", "hit_into_play_score"}

if not STATCAST_FILES:
    raise SystemExit(
        "No statcast_{year}_{month}.parquet files found in this directory. "
        "Run this script from the folder that contains them."
    )

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Build meta maps + team rosters (batters & pitchers)
# ═══════════════════════════════════════════════════════════════════

print("Building player meta maps and team rosters…")

batter_rows  = []   # one row per (batter_id, season, team)
pitcher_rows = []   # one row per (pitcher_id, season, team)

for yr, paths in STATCAST_FILES.items():
    for path in paths:
        print(f"  Loading {path.name}…")
        meta_cols = ["batter", "pitcher", "player_name", "des",
                     "stand", "p_throws", "home_team", "away_team", "inning_topbot"]
        try:
            df = pd.read_parquet(path, engine="pyarrow", columns=meta_cols)
        except Exception as e:
            print(f"    Skipping {path.name}: {e}")
            continue

        df["pitcher_team"] = np.where(df["inning_topbot"] == "Top",
                                       df["home_team"], df["away_team"])
        df["batter_team"]  = np.where(df["inning_topbot"] == "Top",
                                       df["away_team"], df["home_team"])

        # Pitchers
        pm = (df[["pitcher", "player_name", "p_throws", "pitcher_team"]]
              .dropna(subset=["player_name", "pitcher_team"])
              .drop_duplicates(["pitcher", "pitcher_team"]))
        for _, row in pm.iterrows():
            pitcher_rows.append({
                "pitcher_id": int(row["pitcher"]),
                "raw_name":   str(row["player_name"]),
                "hand":       str(row["p_throws"]),
                "team":       str(row["pitcher_team"]),
                "season":     yr,
            })

        # Batters
        bm = (df[["batter", "stand", "batter_team"]]
              .dropna(subset=["batter_team"])
              .drop_duplicates(["batter", "batter_team"]))

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
                "batter_id": bid,
                "name":      batter_name_map.get(bid, f"Batter #{bid}"),
                "stand":     str(row.get("stand", "?")),
                "team":      str(row["batter_team"]),
                "season":    yr,
            })

batter_df  = (pd.DataFrame(batter_rows)
              .drop_duplicates(["batter_id", "season", "team"])
              .reset_index(drop=True))
pitcher_df = (pd.DataFrame(pitcher_rows)
              .drop_duplicates(["pitcher_id", "season", "team"])
              .reset_index(drop=True))


def _fmt(s):
    parts = str(s).split(",")
    if len(parts) == 2:
        return f"{parts[1].strip()} {parts[0].strip()}"
    return str(s).strip()


pitcher_df["display_name"] = pitcher_df["raw_name"].apply(_fmt)

print(f"  → {len(batter_df)} batter rows, {len(pitcher_df)} pitcher rows")

batter_df.to_parquet(DATA_DIR / "meta_batters.parquet", engine="pyarrow", index=False)
pitcher_df.to_parquet(DATA_DIR / "meta_pitchers.parquet", engine="pyarrow", index=False)
print("  Saved meta_batters.parquet and meta_pitchers.parquet")

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Pre-aggregate zone stats (feeds Tab 1 fast path)
# ═══════════════════════════════════════════════════════════════════

print("\nPre-aggregating zone statistics for Tab 1…")

agg_parts = []
agg_cols = ["pitch_type", "zone", "stand", "p_throws", "balls", "strikes",
            "description", "launch_speed", "launch_angle",
            "estimated_woba_using_speedangle"]

for yr, paths in STATCAST_FILES.items():
    for path in paths:
        print(f"  Aggregating {path.name}…")
        try:
            df2 = pd.read_parquet(path, engine="pyarrow", columns=agg_cols)
        except Exception as e:
            print(f"    Skipping {path.name}: {e}")
            continue

        df2["game_year"]   = yr
        df2["count_state"] = df2["balls"].astype(str) + "-" + df2["strikes"].astype(str)
        desc = df2["description"].fillna("")
        df2["is_swing"]   = desc.isin(SWING_EV).astype("int8")
        df2["is_whiff"]   = desc.isin(WHIFF_EV).astype("int8")
        df2["is_contact"] = desc.isin(CONTACT_EV).astype("int8")
        ls = pd.to_numeric(df2["launch_speed"], errors="coerce")
        la = pd.to_numeric(df2["launch_angle"], errors="coerce")
        df2["is_barrel"] = ((ls >= 98) & la.between(26, 30)).fillna(False).astype("int8")
        df2["is_hh"]     = (ls >= 95).fillna(False).astype("int8")
        df2["is_gb"]     = (la < 10).fillna(False).astype("int8")
        df2 = df2[df2["zone"].between(1, 14)]
        agg_parts.append(df2)

if not agg_parts:
    raise SystemExit("No data could be aggregated — check file paths/columns.")

full = pd.concat(agg_parts, ignore_index=True)
grp = full.groupby(
    ["game_year", "zone", "pitch_type", "p_throws", "stand", "count_state"],
    dropna=False, as_index=False,
).agg(
    total     = ("is_swing",   "count"),
    swings    = ("is_swing",   "sum"),
    whiffs    = ("is_whiff",   "sum"),
    contacts  = ("is_contact", "sum"),
    barrels   = ("is_barrel",  "sum"),
    hard_hits = ("is_hh",      "sum"),
    gbs       = ("is_gb",      "sum"),
    batted    = ("launch_speed", "count"),
    avg_ev    = ("launch_speed", "mean"),
    avg_la    = ("launch_angle", "mean"),
    avg_xwoba = ("estimated_woba_using_speedangle", "mean"),
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

print("\n✅ Precompute complete. Commit these 3 files to GitHub, "
      "next to your statcast_*.parquet files:")
print("   meta_batters.parquet")
print("   meta_pitchers.parquet")
print("   zone_stats_agg.parquet")
