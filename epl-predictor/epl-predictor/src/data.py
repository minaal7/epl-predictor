"""
Step 1: collect, clean and merge Premier League results + betting odds.

Primary source: football-data.co.uk season CSVs (E0 = Premier League).
Fallback source: a GitHub mirror of the same football-data.co.uk data
(xgabora/Club-Football-Match-Data-2000-2025), used when football-data.co.uk
is unreachable. Both are normalised into ONE schema:

    date, season, home, away, fthg, ftag, ftr,
    hs, as_, hst, ast,              # shots / shots on target
    odds_h, odds_d, odds_a,         # market-average pre-match odds
    max_h, max_d, max_a             # best available pre-match odds
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN_PATH = ROOT / "data" / "matches_clean.csv"

FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
MIRROR_URL = ("https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/"
              "main/data/Matches.csv")

# One canonical spelling per club (sources disagree over the years)
TEAM_ALIASES = {
    "Nottm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Sheffield Utd": "Sheffield United",
    "Wolverhampton": "Wolves",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Leicester City": "Leicester",
}

COLUMNS = ["date", "season", "home", "away", "fthg", "ftag", "ftr",
           "hs", "as_", "hst", "ast",
           "odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a"]


def season_code(start_year: int) -> str:
    """2023 -> '2324' (football-data.co.uk URL format)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_of(date: pd.Series) -> pd.Series:
    """Season = start year. Aug 1 cut-off handles the Covid 2019-20 season,
    which finished on 26 July 2020."""
    return date.dt.year - (date.dt.month < 8).astype(int)


# ----------------------------------------------------------------- sources
def _first_present(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for c in cols:
        if c in df.columns:
            out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
    return out


def load_football_data(first_season: int, last_season: int) -> pd.DataFrame:
    frames = []
    for yr in range(first_season, last_season + 1):
        url = FD_URL.format(code=season_code(yr))
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 404 and yr == last_season:
            print(f"  {season_code(yr)} not published yet, skipping")
            continue
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.content.decode("latin-1")))
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
        (RAW / f"E0_{season_code(yr)}.csv").write_bytes(r.content)
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
            "home": raw["HomeTeam"], "away": raw["AwayTeam"],
            "fthg": raw["FTHG"], "ftag": raw["FTAG"], "ftr": raw["FTR"],
            "hs": raw.get("HS"), "as_": raw.get("AS"),
            "hst": raw.get("HST"), "ast": raw.get("AST"),
            # market average odds (column names changed in 2019-20)
            "odds_h": _first_present(raw, ["AvgH", "BbAvH", "B365H"]),
            "odds_d": _first_present(raw, ["AvgD", "BbAvD", "B365D"]),
            "odds_a": _first_present(raw, ["AvgA", "BbAvA", "B365A"]),
            "max_h": _first_present(raw, ["MaxH", "BbMxH"]),
            "max_d": _first_present(raw, ["MaxD", "BbMxD"]),
            "max_a": _first_present(raw, ["MaxA", "BbMxA"]),
        })
        frames.append(df)
        print(f"  football-data.co.uk {season_code(yr)}: {len(df)} matches")
    return pd.concat(frames, ignore_index=True)


def load_mirror() -> pd.DataFrame:
    path = RAW / "club_matches_mirror.csv"
    if not path.exists():
        print("  downloading GitHub mirror (~45 MB)…")
        r = requests.get(MIRROR_URL, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["Division"] == "E0"]
    return pd.DataFrame({
        "date": pd.to_datetime(raw["MatchDate"]),
        "home": raw["HomeTeam"], "away": raw["AwayTeam"],
        "fthg": raw["FTHome"], "ftag": raw["FTAway"], "ftr": raw["FTResult"],
        "hs": raw["HomeShots"], "as_": raw["AwayShots"],
        "hst": raw["HomeTarget"], "ast": raw["AwayTarget"],
        "odds_h": raw["OddHome"], "odds_d": raw["OddDraw"], "odds_a": raw["OddAway"],
        "max_h": raw["MaxHome"], "max_d": raw["MaxDraw"], "max_a": raw["MaxAway"],
    })


# ----------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("home", "away"):
        df[c] = df[c].astype(str).str.strip().replace(TEAM_ALIASES)
    df = df.dropna(subset=["date"])
    df["season"] = season_of(df["date"])
    num = ["fthg", "ftag", "hs", "as_", "hst", "ast",
           "odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")

    # Result consistency: recompute FTR from the score where both exist
    played = df["fthg"].notna() & df["ftag"].notna()
    df.loc[played, "ftr"] = np.select(
        [df.loc[played, "fthg"] > df.loc[played, "ftag"],
         df.loc[played, "fthg"] < df.loc[played, "ftag"]], ["H", "A"], "D")

    # Odds sanity: anything <= 1.0 is a data error
    for c in ["odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a"]:
        df.loc[df[c] <= 1.0, c] = np.nan

    df = (df.drop_duplicates(subset=["date", "home", "away"])
            .sort_values(["date", "home"]).reset_index(drop=True))
    return df[COLUMNS]


def build_dataset(first_season: int = 2005, last_season: int = 2026,
                  source: str = "auto") -> pd.DataFrame:
    """Download (or reuse) data, clean it, save data/matches_clean.csv.

    Seasons before the modelling window are kept as warm-up history for the
    Elo ratings and rolling features.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    df = None
    if source in ("auto", "football-data"):
        try:
            print("Trying football-data.co.uk …")
            df = load_football_data(first_season, last_season)
        except Exception as e:  # noqa: BLE001
            if source == "football-data":
                raise
            print(f"  unavailable ({e.__class__.__name__}); using GitHub mirror")
    if df is None:
        df = load_mirror()
    df = clean(df)
    df = df[(df.season >= first_season) & (df.season <= last_season)]
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {len(df):,} matches ({df.season.min()}–{df.season.max()}) to {CLEAN_PATH}")
    return df


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_PATH, parse_dates=["date"])


if __name__ == "__main__":
    build_dataset()
