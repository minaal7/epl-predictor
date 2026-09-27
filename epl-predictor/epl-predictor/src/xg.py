"""
Expected-goals features.

Two options:
1. Real xG from Understat (shot-level model, 2014-15 onwards). Run
   `python -m src.xg --fetch` on a machine that can reach understat.com. It
   writes data/understat_xg.csv, which the pipeline merges automatically.
2. A shot-based xG proxy (always available). A Poisson regression of goals on
   shots on target and off-target shots, fitted ONLY on seasons before the
   modelling window (2005-06 to 2013-14) so it can't leak information into the
   training or test years. It converts shot volume into "expected goals" units.

Real xG is the better feature because it knows shot quality (a penalty vs a
30-yard hopeful). The proxy only knows volume and accuracy.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
UNDERSTAT_PATH = ROOT / "data" / "understat_xg.csv"

UNDERSTAT_NAMES = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Newcastle United": "Newcastle", "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom", "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR", "Sheffield United": "Sheffield United",
    "Tottenham": "Tottenham", "Leicester": "Leicester", "Brighton": "Brighton",
}


def fit_proxy(matches: pd.DataFrame, fit_seasons=range(2005, 2014)) -> dict:
    """Fit goals ~ exp(b0 + b1*SoT + b2*off-target shots) on pre-window seasons."""
    m = matches[matches.season.isin(list(fit_seasons))].dropna(subset=["hs", "hst", "fthg"])
    X = np.vstack([
        np.c_[m.hst, m.hs - m.hst],          # home attack
        np.c_[m.ast, m.as_ - m.ast],         # away attack
    ])
    y = np.r_[m.fthg, m.ftag]
    reg = PoissonRegressor(alpha=1e-6, max_iter=1000).fit(X, y)
    return {"intercept": float(reg.intercept_),
            "sot": float(reg.coef_[0]), "off": float(reg.coef_[1]),
            "n_fit": int(len(m))}


def add_xg(matches: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    """Adds home_xg / away_xg. Uses Understat where present, proxy elsewhere."""
    df = matches.copy()

    def proxy(sot, shots):
        return np.exp(coefs["intercept"] + coefs["sot"] * sot + coefs["off"] * (shots - sot))

    df["home_xg"] = proxy(df.hst, df.hs)
    df["away_xg"] = proxy(df.ast, df.as_)
    df["xg_source"] = "proxy"

    if UNDERSTAT_PATH.exists():
        us = pd.read_csv(UNDERSTAT_PATH, parse_dates=["date"])
        us["date"] = us["date"].dt.normalize()
        df = df.merge(us[["date", "home", "away", "us_home_xg", "us_away_xg"]],
                      on=["date", "home", "away"], how="left")
        hit = df.us_home_xg.notna()
        df.loc[hit, "home_xg"] = df.loc[hit, "us_home_xg"]
        df.loc[hit, "away_xg"] = df.loc[hit, "us_away_xg"]
        df.loc[hit, "xg_source"] = "understat"
        df = df.drop(columns=["us_home_xg", "us_away_xg"])
        print(f"Merged Understat xG for {hit.sum():,} of {len(df):,} matches")
    return df


# --------------------------------------------------------------- Understat
def fetch_understat(first_season: int = 2014, last_season: int = 2026) -> pd.DataFrame:
    """Download match-level xG from Understat (needs internet access to understat.com).

    Understat has changed how it serves data before, so this tries the JSON
    endpoint first and falls back to parsing the JSON embedded in the page.
    """
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})
    rows = []
    for yr in range(first_season, last_season + 1):
        data = None
        try:
            r = s.get(f"https://understat.com/getLeagueData/EPL/{yr}", timeout=30)
            r.raise_for_status()
            data = r.json()["dates"]
        except Exception:  # noqa: BLE001
            html = s.get(f"https://understat.com/league/EPL/{yr}", timeout=30).text
            m = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", html)
            if m:
                data = json.loads(m.group(1).encode().decode("unicode_escape"))
        if not data:
            print(f"  {yr}: no data")
            continue
        for g in data:
            if not g.get("isResult"):
                continue
            rows.append({
                "date": pd.to_datetime(g["datetime"]).normalize(),
                "home": UNDERSTAT_NAMES.get(g["h"]["title"], g["h"]["title"]),
                "away": UNDERSTAT_NAMES.get(g["a"]["title"], g["a"]["title"]),
                "us_home_xg": float(g["xG"]["h"]), "us_away_xg": float(g["xG"]["a"]),
            })
        print(f"  {yr}: {sum(1 for g in data if g.get('isResult'))} matches")
    out = pd.DataFrame(rows)
    out.to_csv(UNDERSTAT_PATH, index=False)
    print(f"Saved {len(out):,} rows to {UNDERSTAT_PATH}")
    return out


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch_understat()
