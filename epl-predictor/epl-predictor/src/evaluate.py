"""
Steps 4 and 5: evaluation and betting simulation.

Metrics (probabilities ordered H, D, A):
  log loss   the headline metric: punishes confident wrong answers
  accuracy   % where the most likely outcome happened
  Brier      mean squared error of the probability vector
  RPS        ranked probability score, the standard football metric: it
             treats H > D > A as ordered, so calling a draw when the home
             side wins is "less wrong" than calling an away win
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ["H", "D", "A"]


def log_loss(y, P):
    P = np.clip(P, 1e-15, 1)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def brier(y, P):
    Y = np.eye(3)[y]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def rps(y, P):
    Y = np.eye(3)[y]
    cp, cy = np.cumsum(P, 1)[:, :2], np.cumsum(Y, 1)[:, :2]
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / 2))


def accuracy(y, P):
    return float(np.mean(P.argmax(1) == y))


def score(y, P) -> dict:
    return {"log_loss": log_loss(y, P), "accuracy": accuracy(y, P),
            "brier": brier(y, P), "rps": rps(y, P)}


def paired_bootstrap_logloss(y, P_model, P_ref, n_boot=5000, seed=0) -> dict:
    """Is the model's log loss really different from the reference (bookies)?
    Returns mean difference (model - ref; negative = model better) and a 95% CI."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    lm = -np.log(np.clip(P_model[idx, y], 1e-15, 1))
    lr = -np.log(np.clip(P_ref[idx, y], 1e-15, 1))
    d = lm - lr
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "p_model_better": float(np.mean(np.array(boots) < 0))}


def calibration_table(y, P, outcome: int, bins=10) -> pd.DataFrame:
    p = P[:, outcome]
    hit = (y == outcome).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    t = pd.DataFrame({"bin": b, "p": p, "hit": hit}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("hit", "mean"), n=("hit", "size"))
    return t.reset_index(drop=True)


# ------------------------------------------------------------- betting
def simulate_betting(df: pd.DataFrame, P: np.ndarray, odds_cols=("odds_h", "odds_d", "odds_a"),
                     edge=0.05, stake="flat", kelly_frac=0.25, max_odds=None) -> dict:
    """Value betting: back the outcome with the largest expected value if the
    model's EV (p * odds - 1) exceeds `edge`. At most one bet per match.

    stake="flat"  -> 1 unit per bet, report profit and ROI
    stake="kelly" -> fractional Kelly on a bankroll starting at 100
    """
    O = df[list(odds_cols)].to_numpy(float)
    y = df["y"].to_numpy(int)
    ev = P * O - 1
    if max_odds:
        ev = np.where(O > max_odds, -np.inf, ev)
    pick = np.nanargmax(np.where(np.isnan(ev), -np.inf, ev), axis=1)
    best_ev = ev[np.arange(len(ev)), pick]
    bet = best_ev > edge
    odds_taken = O[np.arange(len(O)), pick]
    won = pick == y

    ledger = pd.DataFrame({"date": df["date"].to_numpy(), "home": df["home"].to_numpy(),
                           "away": df["away"].to_numpy(), "pick": np.array(OUTCOMES)[pick],
                           "p_model": P[np.arange(len(P)), pick], "odds": odds_taken,
                           "ev": best_ev, "won": won})[bet].reset_index(drop=True)
    if stake == "flat":
        ledger["stake"] = 1.0
        ledger["pnl"] = np.where(ledger.won, ledger.odds - 1, -1.0)
        ledger["cum_pnl"] = ledger.pnl.cumsum()
        n = len(ledger)
        profit = float(ledger.pnl.sum()) if n else 0.0
        rng = np.random.default_rng(0)
        boots = ([ledger.pnl.to_numpy()[rng.integers(0, n, n)].mean() for _ in range(5000)]
                 if n else [0.0])
        return {"bets": n, "win_rate": float(ledger.won.mean()) if n else 0.0,
                "avg_odds": float(ledger.odds.mean()) if n else 0.0,
                "profit_units": profit, "roi": profit / n if n else 0.0,
                "roi_ci_low": float(np.percentile(boots, 2.5)),
                "roi_ci_high": float(np.percentile(boots, 97.5)),
                "ledger": ledger}
    # fractional Kelly
    bank, hist = 100.0, []
    for r in ledger.itertuples():
        f = max(0.0, (r.p_model * r.odds - 1) / (r.odds - 1)) * kelly_frac
        s = bank * f
        bank += s * (r.odds - 1) if r.won else -s
        hist.append(bank)
    ledger["bankroll"] = hist
    return {"bets": len(ledger), "final_bankroll": bank, "ledger": ledger}


def bookmaker_margin_hurdle(df: pd.DataFrame) -> float:
    """Average bookmaker margin: the edge a bettor must overcome just to break even."""
    return float((df.overround - 1).mean())
