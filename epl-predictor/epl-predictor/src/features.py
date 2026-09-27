"""
Step 2: feature engineering.

Golden rule: every feature for a match is computed ONLY from matches that
finished before it kicked off (everything is shift(1)-ed). The same code path
builds features for hypothetical future fixtures in the app: we append a row
with no result and read off its features.

Feature families
  Elo rating        own implementation: home advantage, goal-difference
                    multiplier, summer regression to the mean, promoted
                    teams start at the average of the teams they replaced
  Rolling form      last 5 matches: points, goals for/against, goal diff,
                    shots on target, xG for/against
  Smoothed form     exponentially weighted (span 10) goal diff and xG diff
  Venue form        last 5 HOME matches for the home side, last 5 AWAY
                    matches for the away side
  Season form       points per game so far this season
  Schedule          rest days since the previous league match
  Newcomer flag     team is in its first season back in the division
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ELO_K = 20.0
ELO_HFA = 60.0          # home advantage in Elo points
ELO_REGRESS = 0.2       # fraction pulled back toward 1500 each summer
SPELL_GAP_DAYS = 150    # a longer gap between league games = relegated & back
PRIOR_SEASONS = range(2005, 2016)   # seasons used to build newcomer priors

STATS = ["pts", "gf", "ga", "gd", "sotf", "sota", "xgf", "xga", "xgd"]
VENUE_STATS = ["pts", "gd", "xgd"]

FEATURES = [
    "elo_diff", "elo_exp",
    "d_pts_r5", "d_gd_r5", "d_xgf_r5", "d_xga_r5", "d_sotf_r5", "d_sota_r5",
    "d_gd_ewm", "d_xgd_ewm",
    "h_pts_v5", "a_pts_v5", "h_xgd_v5", "a_xgd_v5",
    "d_ppg_season",
    "h_rest", "a_rest",
    "h_new", "a_new",
]
LABELS = {"H": 0, "D": 1, "A": 2}


# ------------------------------------------------------------------- Elo
def _goal_mult(gd: float) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pre-match Elo for every row. Rows without a result get a rating but
    don't update anything (future fixtures)."""
    df = df.sort_values("date").copy()
    ratings: dict[str, float] = {}
    season_teams = df.groupby("season")[["home", "away"]].agg(
        lambda s: set(s)).apply(lambda r: r.home | r.away, axis=1).to_dict()

    cur_season = None
    elo_h, elo_a = np.empty(len(df)), np.empty(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        if row.season != cur_season:
            prev_teams = season_teams.get(cur_season, set())
            new_teams = season_teams[row.season]
            # summer regression toward the mean
            ratings = {t: r + ELO_REGRESS * (1500 - r) for t, r in ratings.items()}
            relegated = prev_teams - new_teams
            promoted = new_teams - prev_teams
            if prev_teams and relegated:
                start = np.mean([ratings[t] for t in relegated])
            else:
                start = 1500.0
            for t in promoted:
                ratings[t] = start
            cur_season = row.season
        rh, ra = ratings.setdefault(row.home, 1500.0), ratings.setdefault(row.away, 1500.0)
        elo_h[i], elo_a[i] = rh, ra
        if pd.isna(row.fthg):
            continue
        exp_h = 1 / (1 + 10 ** (-(rh + ELO_HFA - ra) / 400))
        score_h = 1.0 if row.fthg > row.ftag else 0.5 if row.fthg == row.ftag else 0.0
        delta = ELO_K * _goal_mult(row.fthg - row.ftag) * (score_h - exp_h)
        ratings[row.home] = rh + delta
        ratings[row.away] = ra - delta
    df["elo_h"], df["elo_a"] = elo_h, elo_a
    return df, ratings


# ------------------------------------------------------------- long table
def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    common = ["mid", "date", "season"]
    home = pd.DataFrame({
        **{c: df[c] for c in common}, "team": df.home, "venue": 1,
        "gf": df.fthg, "ga": df.ftag, "sotf": df.hst, "sota": df.ast,
        "xgf": df.home_xg, "xga": df.away_xg})
    away = pd.DataFrame({
        **{c: df[c] for c in common}, "team": df.away, "venue": 0,
        "gf": df.ftag, "ga": df.fthg, "sotf": df.ast, "sota": df.hst,
        "xgf": df.away_xg, "xga": df.home_xg})
    long = pd.concat([home, away], ignore_index=True)
    long["gd"] = long.gf - long.ga
    long["xgd"] = long.xgf - long.xga
    long["pts"] = np.select([long.gd > 0, long.gd == 0], [3.0, 1.0], 0.0)
    long.loc[long.gf.isna(), "pts"] = np.nan
    return long.sort_values(["team", "date", "mid"]).reset_index(drop=True)


def _rolling_features(long: pd.DataFrame) -> pd.DataFrame:
    g_team = long.groupby("team")
    long["rest"] = g_team["date"].diff().dt.days.clip(2, 14).fillna(14)
    gap = g_team["date"].diff().dt.days.fillna(9999)
    long["spell"] = (gap > SPELL_GAP_DAYS).astype(int).groupby(long.team).cumsum()
    first_season = long.groupby(["team", "spell"])["season"].transform("min")
    first_in_data = long.season == long.season.min()
    long["new"] = ((long.season == first_season) & ~first_in_data).astype(float)

    g = long.groupby(["team", "spell"])
    for s in STATS:
        shifted = g[s].shift(1)
        long[f"{s}_r5"] = shifted.groupby([long.team, long.spell]).transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        long[f"{s}_ewm"] = shifted.groupby([long.team, long.spell]).transform(
            lambda x: x.ewm(span=10, ignore_na=True).mean())

    gv = long.groupby(["team", "spell", "venue"])
    for s in VENUE_STATS:
        shifted = gv[s].shift(1)
        long[f"{s}_v5"] = shifted.groupby([long.team, long.spell, long.venue]).transform(
            lambda x: x.rolling(5, min_periods=1).mean())

    gs = long.groupby(["team", "season"])
    long["ppg_season"] = gs["pts"].transform(lambda x: x.shift(1).expanding().mean())
    return long


def _fill_newcomer_prior(long: pd.DataFrame) -> pd.DataFrame:
    """A team with no history yet (first game back in the division) gets the
    average stats of newcomers from the PRIOR seasons (pre-modelling window, so
    no leakage). Better than the league median: promoted sides are weaker."""
    cols = [c for c in long.columns if c.endswith(("_r5", "_ewm", "_v5"))] + ["ppg_season"]
    newc = long[(long.new == 1) & long.season.isin(PRIOR_SEASONS)]
    # what did newcomers actually do? (use their raw match stats)
    base = {s: newc[s].mean() for s in STATS}
    for c in cols:
        stat = c.rsplit("_", 1)[0] if c != "ppg_season" else "pts"
        long[c] = long[c].fillna(base[stat])
    return long


# ------------------------------------------------------------ public API
def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """matches must contain home_xg/away_xg (see xg.add_xg). Returns one row
    per match with FEATURES + target + odds columns."""
    df = matches.sort_values(["date", "home"]).reset_index(drop=True).copy()
    df["mid"] = np.arange(len(df))
    df, _ = compute_elo(df)
    df = df.sort_values("mid").reset_index(drop=True)

    long = _fill_newcomer_prior(_rolling_features(_to_long(df)))
    keep = [c for c in long.columns if c.endswith(("_r5", "_ewm", "_v5"))] + \
        ["ppg_season", "rest", "new"]
    h = long[long.venue == 1].set_index("mid")[keep].add_prefix("h_")
    a = long[long.venue == 0].set_index("mid")[keep].add_prefix("a_")
    out = df.join(h, on="mid").join(a, on="mid")

    out["elo_diff"] = out.elo_h - out.elo_a
    out["elo_exp"] = 1 / (1 + 10 ** (-(out.elo_diff + ELO_HFA) / 400))
    for s in STATS:
        out[f"d_{s}_r5"] = out[f"h_{s}_r5"] - out[f"a_{s}_r5"]
        out[f"d_{s}_ewm"] = out[f"h_{s}_ewm"] - out[f"a_{s}_ewm"]
    out["d_ppg_season"] = out.h_ppg_season - out.a_ppg_season
    out["y"] = out.ftr.map(LABELS)

    # bookmaker implied probabilities (normalised to remove the margin)
    inv = 1 / out[["odds_h", "odds_d", "odds_a"]].to_numpy()
    out["overround"] = inv.sum(1)
    out[["bk_h", "bk_d", "bk_a"]] = inv / inv.sum(1, keepdims=True)
    return out


def features_for_fixture(matches: pd.DataFrame, home: str, away: str,
                         date: pd.Timestamp | None = None,
                         rest_home: int | None = None,
                         rest_away: int | None = None) -> pd.DataFrame:
    """Features for a hypothetical match, using all history before `date`."""
    date = date or (matches.date.max() + pd.Timedelta(days=7))
    season = int(date.year - (date.month < 8))
    fx = pd.DataFrame([{"date": date, "season": season, "home": home, "away": away}])
    hist = matches[matches.date < date]
    feats = build_features(pd.concat([hist, fx], ignore_index=True))
    row = feats[(feats.date == date) & (feats.home == home) & (feats.away == away)].copy()
    if rest_home is not None:
        row["h_rest"] = rest_home
    if rest_away is not None:
        row["a_rest"] = rest_away
    return row
