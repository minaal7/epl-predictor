"""
Step 3: models.

  Baseline      always predict a home win; probabilities = training class
                frequencies (so log loss is defined)
  Logistic      multinomial logistic regression on standardised features
  XGBoost       gradient-boosted trees, shallow and heavily regularised
  Dixon-Coles   (stretch goal) bivariate Poisson scoreline model with the
                Dixon-Coles low-score correction and exponential time decay
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ------------------------------------------------------------ baselines
class HomeWinBaseline:
    def fit(self, X, y):
        self.p_ = np.bincount(y, minlength=3) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))

    def predict(self, X):
        return np.zeros(len(X), dtype=int)   # always "H"


# -------------------------------------------------------- logistic / xgb
def make_logreg(C: float = 0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def make_xgb(**overrides):
    params = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                  n_estimators=300, learning_rate=0.03, max_depth=2,
                  min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, tree_method="hist", random_state=42, n_jobs=4)
    params.update(overrides)
    return XGBClassifier(**params)


# ---------------------------------------------------------- Dixon-Coles
def _tau(x, y, mu, nu, rho):
    """Dixon-Coles correction for 0-0, 1-0, 0-1, 1-1."""
    t = np.ones_like(mu)
    t = np.where((x == 0) & (y == 0), 1 - mu * nu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + nu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.clip(t, 1e-10, None)


class DixonColes:
    """Attack/defence ratings per team, fitted by weighted maximum likelihood.

    log E[home goals] = home_adv + attack[home] + defence[away]
    log E[away goals] =            attack[away] + defence[home]

    Ratings are shrunk toward a prior: 0 (average) for established teams and
    a "promoted side" prior for newcomers, which matters early in a season
    when a promoted team has almost no top-flight data.
    """

    def __init__(self, xi: float = 0.0019, window_days: int = 3 * 365,
                 ridge: float = 5.0, max_goals: int = 10,
                 newcomer_prior: tuple[float, float] = (-0.25, 0.2)):
        self.xi, self.window_days, self.ridge = xi, window_days, ridge
        self.max_goals, self.newcomer_prior = max_goals, newcomer_prior

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp,
            newcomers: set[str] | None = None):
        m = matches[(matches.date < as_of) &
                    (matches.date >= as_of - pd.Timedelta(days=self.window_days))]
        m = m.dropna(subset=["fthg", "ftag"])
        newcomers = newcomers or set()
        teams = sorted(set(m.home) | set(m.away) | newcomers)
        self.teams_ = teams
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi, ai = m.home.map(idx).to_numpy(), m.away.map(idx).to_numpy()
        x, y = m.fthg.to_numpy(int), m.ftag.to_numpy(int)
        w = np.exp(-self.xi * (as_of - m.date).dt.days.to_numpy())

        att_c = np.array([self.newcomer_prior[0] if t in newcomers else 0.0 for t in teams])
        def_c = np.array([self.newcomer_prior[1] if t in newcomers else 0.0 for t in teams])

        def nll(p):
            att, dfn, home, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            mu = np.exp(home + att[hi] + dfn[ai])
            nu = np.exp(att[ai] + dfn[hi])
            ll = (np.log(_tau(x, y, mu, nu, rho))
                  + x * np.log(mu) - mu + y * np.log(nu) - nu)
            pen = 0.5 * self.ridge * (np.sum((att - att_c) ** 2) + np.sum((dfn - def_c) ** 2))
            return -(w * ll).sum() + pen

        def grad(p):  # analytic Poisson part; tau part is small, done numerically
            att, dfn, home, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            mu = np.exp(home + att[hi] + dfn[ai])
            nu = np.exp(att[ai] + dfn[hi])
            rh, ra = w * (x - mu), w * (y - nu)
            g = np.zeros_like(p)
            g[:n] -= np.bincount(hi, rh, n) + np.bincount(ai, ra, n)
            g[n:2 * n] -= np.bincount(ai, rh, n) + np.bincount(hi, ra, n)
            g[2 * n] -= rh.sum()
            g[:n] += self.ridge * (att - att_c)
            g[n:2 * n] += self.ridge * (dfn - def_c)
            # tau terms (only low scores): finite differences on the few affected params
            low = (x <= 1) & (y <= 1)
            if low.any():
                eps = 1e-6

                def tau_ll(pp):
                    a_, d_, h_, r_ = pp[:n], pp[n:2 * n], pp[2 * n], pp[2 * n + 1]
                    mu_ = np.exp(h_ + a_[hi[low]] + d_[ai[low]])
                    nu_ = np.exp(a_[ai[low]] + d_[hi[low]])
                    return (w[low] * np.log(_tau(x[low], y[low], mu_, nu_, r_))).sum()
                base = tau_ll(p)
                for j in range(len(p)):
                    pj = p.copy()
                    pj[j] += eps
                    g[j] -= (tau_ll(pj) - base) / eps
            return g

        p0 = np.r_[att_c, def_c, 0.25, -0.05]
        bounds = [(None, None)] * (2 * n + 1) + [(-0.2, 0.2)]
        res = minimize(nll, p0, jac=grad, method="L-BFGS-B", bounds=bounds)
        self.att_ = dict(zip(teams, res.x[:n]))
        self.def_ = dict(zip(teams, res.x[n:2 * n]))
        self.home_, self.rho_ = res.x[2 * n], res.x[2 * n + 1]
        self.as_of_ = as_of
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        a0, d0 = self.newcomer_prior
        mu = np.exp(self.home_ + self.att_.get(home, a0) + self.def_.get(away, d0))
        nu = np.exp(self.att_.get(away, a0) + self.def_.get(home, d0))
        return float(mu), float(nu)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        mu, nu = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        M = np.outer(poisson.pmf(g, mu), poisson.pmf(g, nu))
        M[0, 0] *= 1 - mu * nu * self.rho_
        M[0, 1] *= 1 + mu * self.rho_
        M[1, 0] *= 1 + nu * self.rho_
        M[1, 1] *= 1 - self.rho_
        return M / M.sum()

    def predict_proba_pair(self, home: str, away: str) -> np.ndarray:
        M = self.score_matrix(home, away)
        return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])

    def to_dict(self) -> dict:
        return {"att": self.att_, "def": self.def_, "home": float(self.home_),
                "rho": float(self.rho_), "as_of": str(self.as_of_.date()),
                "xi": self.xi, "max_goals": self.max_goals,
                "newcomer_prior": list(self.newcomer_prior)}

    @classmethod
    def from_dict(cls, d: dict) -> "DixonColes":
        m = cls(xi=d["xi"], max_goals=d["max_goals"],
                newcomer_prior=tuple(d["newcomer_prior"]))
        m.att_, m.def_ = d["att"], d["def"]
        m.home_, m.rho_ = d["home"], d["rho"]
        m.as_of_ = pd.Timestamp(d["as_of"])
        m.teams_ = sorted(m.att_)
        return m


def newcomer_prior_from(matches: pd.DataFrame, seasons=range(2005, 2016)) -> tuple[float, float]:
    """log(goals scored / conceded by promoted sides vs league average), estimated
    only from pre-modelling seasons."""
    m = matches[matches.season.isin(list(seasons))]
    teams_by = m.groupby("season").apply(lambda s: set(s.home) | set(s.away)).to_dict()
    gf = ga = n = 0.0
    for s in list(seasons)[1:]:
        new = teams_by[s] - teams_by[s - 1]
        ms = m[m.season == s]
        for side, opp, gcol, ocol in (("home", "away", "fthg", "ftag"), ("away", "home", "ftag", "fthg")):
            sel = ms[ms[side].isin(new)]
            gf += sel[gcol].sum()
            ga += sel[ocol].sum()
            n += len(sel)
    league = (m.fthg.mean() + m.ftag.mean()) / 2
    return float(np.log(gf / n / league)), float(np.log(ga / n / league))


def newcomers_for_season(matches: pd.DataFrame, season: int) -> set[str]:
    cur = matches[matches.season == season]
    prev = matches[matches.season == season - 1]
    return (set(cur.home) | set(cur.away)) - (set(prev.home) | set(prev.away))


def dixon_coles_walk_forward(matches: pd.DataFrame, test: pd.DataFrame,
                             prior: tuple[float, float], refit_days: int = 7,
                             **kw) -> np.ndarray:
    """Predict each test match with a model fitted only on matches before its
    week. Refit every `refit_days`."""
    test = test.reset_index(drop=True)   # output rows align with input order
    out = np.zeros((len(test), 3))
    start = test.date.min().normalize()
    blocks = ((test.date - start).dt.days // refit_days).to_numpy()
    for b in np.unique(blocks):
        rows = np.where(blocks == b)[0]
        as_of = start + pd.Timedelta(days=int(b) * refit_days)
        season = int(test.iloc[rows[0]].season)
        dc = DixonColes(newcomer_prior=prior, **kw).fit(
            matches, as_of, newcomers_for_season(matches, season))
        for r in rows:
            out[r] = dc.predict_proba_pair(test.iloc[r].home, test.iloc[r].away)
    return out
