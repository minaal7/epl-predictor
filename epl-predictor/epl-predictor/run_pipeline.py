"""
End-to-end pipeline:  python run_pipeline.py [--source auto|football-data|mirror]

1. collect + clean data           -> data/matches_clean.csv
2. xG + features                  -> data/features.csv
3. tune on a validation season    (train 2016-17..2021-22, validate 2022-23)
4. walk-forward test              (2023-24, 2024-25, 2025-26; each trained on
                                   every earlier season from 2016-17)
5. betting simulation             on the test seasons
6. fit final models on everything -> models/, reports/
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import features as F  # noqa: E402
from src import models as M  # noqa: E402
from src import xg as X  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = D.ROOT
FIG = ROOT / "reports" / "figures"
FIRST_TRAIN = 2016
VAL_SEASON = 2022
TEST_SEASONS = [2023, 2024, 2025]
MAIN_TEST = 2025

COLORS = {"Home-win baseline": "#A7ADB5", "Logistic regression": "#2F6DB5",
          "XGBoost": "#1E5B3A", "Dixon-Coles": "#C7771A", "Ensemble": "#B3261E",
          "Bookmakers": "#6B4E9B"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "figure.dpi": 130})


def season_label(s: int) -> str:
    return f"{s}-{(s + 1) % 100:02d}"


def main(source: str):
    # ---------------------------------------------------------------- 1 + 2
    matches = D.build_dataset(first_season=2005, last_season=2026, source=source)
    coefs = X.fit_proxy(matches)
    matches = X.add_xg(matches, coefs)
    feats = F.build_features(matches)
    feats.to_csv(ROOT / "data" / "features.csv", index=False)
    played = feats[feats.y.notna()].copy()
    played["y"] = played.y.astype(int)
    feat = F.FEATURES

    def split(train_seasons, test_season):
        tr = played[played.season.isin(train_seasons)]
        te = played[played.season == test_season].sort_values(["date", "home"], kind="stable")
        return tr, te

    # ---------------------------------------------------------------- 3 tune
    tr, va = split(range(FIRST_TRAIN, VAL_SEASON), VAL_SEASON)
    lr_scores = {}
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        m = M.make_logreg(C).fit(tr[feat], tr.y)
        lr_scores[C] = E.log_loss(va.y.to_numpy(), m.predict_proba(va[feat]))
    best_C = min(lr_scores, key=lr_scores.get)

    xgb_scores = {}
    for depth in [2, 3]:
        for mcw in [10, 30]:
            m = M.make_xgb(max_depth=depth, min_child_weight=mcw, n_estimators=1500,
                           early_stopping_rounds=100)
            m.fit(tr[feat], tr.y, eval_set=[(va[feat], va.y)], verbose=False)
            xgb_scores[(depth, mcw)] = (m.best_score, m.best_iteration)
    (best_depth, best_mcw), (_, best_iter) = min(xgb_scores.items(), key=lambda kv: kv[1][0])
    xgb_params = dict(max_depth=best_depth, min_child_weight=best_mcw,
                      n_estimators=max(50, int(best_iter * 1.1)))
    print(f"Tuned: logistic C={best_C}, xgb {xgb_params}")

    dc_prior = M.newcomer_prior_from(matches)
    print(f"Dixon-Coles newcomer prior (attack, defence): {np.round(dc_prior, 3)}")

    # ---------------------------------------------------------------- 4 test
    preds = []
    for ts in TEST_SEASONS:
        tr, te = split(range(FIRST_TRAIN, ts), ts)
        y = te.y.to_numpy()
        out = te[["date", "season", "home", "away", "fthg", "ftag", "y",
                  "odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a",
                  "bk_h", "bk_d", "bk_a", "overround"]].copy()
        base = M.HomeWinBaseline().fit(tr[feat], tr.y.to_numpy())
        lr = M.make_logreg(best_C).fit(tr[feat], tr.y)
        xgb = M.make_xgb(**xgb_params).fit(tr[feat], tr.y)
        dc = M.dixon_coles_walk_forward(matches, te, dc_prior)
        for name, P in [("base", base.predict_proba(te)), ("lr", lr.predict_proba(te[feat])),
                        ("xgb", xgb.predict_proba(te[feat])), ("dc", dc)]:
            out[[f"{name}_h", f"{name}_d", f"{name}_a"]] = P
        # simple ensemble: equal-weight average of the three real models
        out[["ens_h", "ens_d", "ens_a"]] = np.mean(
            [out[[f"{k}_h", f"{k}_d", f"{k}_a"]].to_numpy() for k in ("lr", "xgb", "dc")], axis=0)
        preds.append(out)
        print(f"  {season_label(ts)}: trained on {len(tr):,} matches, tested on {len(te)}  "
              f"(LR {E.log_loss(y, out[['lr_h','lr_d','lr_a']].to_numpy()):.4f}, "
              f"XGB {E.log_loss(y, out[['xgb_h','xgb_d','xgb_a']].to_numpy()):.4f}, "
              f"DC {E.log_loss(y, dc):.4f}, "
              f"bookies {E.log_loss(y, out[['bk_h','bk_d','bk_a']].to_numpy()):.4f})")
    preds = pd.concat(preds, ignore_index=True)
    preds.to_csv(ROOT / "reports" / "test_predictions.csv", index=False)

    names = {"base": "Home-win baseline", "lr": "Logistic regression", "xgb": "XGBoost",
             "dc": "Dixon-Coles", "ens": "Ensemble", "bk": "Bookmakers"}

    def P_of(df, k):
        return df[[f"{k}_h", f"{k}_d", f"{k}_a"]].to_numpy()

    rows = []
    for scope, df in [(season_label(s), preds[preds.season == s]) for s in TEST_SEASONS] + \
            [("All 3 seasons", preds)]:
        y = df.y.to_numpy()
        for k, nm in names.items():
            s = E.score(y, P_of(df, k))
            if k == "base":   # "always home" accuracy, not the frequency argmax
                s["accuracy"] = float(np.mean(y == 0))
            rows.append({"scope": scope, "model": nm, **s, "n": len(df)})
    metrics = pd.DataFrame(rows)
    print("\n", metrics[metrics.scope == season_label(MAIN_TEST)].round(4).to_string(index=False))
    print("\n", metrics[metrics.scope == "All 3 seasons"].round(4).to_string(index=False))

    boot = {}
    for k in ["lr", "xgb", "dc", "ens"]:
        boot[names[k]] = E.paired_bootstrap_logloss(preds.y.to_numpy(), P_of(preds, k),
                                                    P_of(preds, "bk"))

    # ------------------------------------------------------------- 5 betting
    bets = {}
    main = preds[preds.season == MAIN_TEST]
    for k in ["lr", "xgb", "dc", "ens"]:
        for odds_name, cols in [("average", ("odds_h", "odds_d", "odds_a")),
                                ("best", ("max_h", "max_d", "max_a"))]:
            for edge in [0.0, 0.05, 0.10]:
                for scope, df in [(season_label(MAIN_TEST), main), ("All 3 seasons", preds)]:
                    r = E.simulate_betting(df, P_of(df, k), odds_cols=cols, edge=edge)
                    bets[(names[k], odds_name, edge, scope)] = r
    bet_table = pd.DataFrame([{"model": m, "odds": o, "edge": e, "scope": s,
                               **{kk: v for kk, v in r.items() if kk != "ledger"}}
                              for (m, o, e, s), r in bets.items()])
    led = bets[("Ensemble", "average", 0.05, "All 3 seasons")]["ledger"]
    by_pick = led.groupby("pick").agg(bets=("pnl", "size"), profit=("pnl", "sum"),
                                      roi=("pnl", "mean"), avg_odds=("odds", "mean"))
    print("\nEnsemble bets by outcome (5% edge, average odds, 3 seasons):\n", by_pick.round(3))
    print("\nBetting, 5% edge:\n", bet_table[bet_table.edge == 0.05].round(3).to_string(index=False))

    # -------------------------------------------------------------- figures
    make_figures(preds, metrics, bets, names, P_of)

    # ------------------------------------------------ 6 final models for app
    final_tr = played[played.season >= FIRST_TRAIN]
    lr_f = M.make_logreg(best_C).fit(final_tr[feat], final_tr.y)
    xgb_f = M.make_xgb(**xgb_params).fit(final_tr[feat], final_tr.y)
    xgb_f.get_booster().feature_names = feat
    cur_season = int(matches.season.max())
    as_of = matches.dropna(subset=["fthg"]).date.max() + pd.Timedelta(days=1)
    dc_f = M.DixonColes(newcomer_prior=dc_prior).fit(
        matches, as_of, M.newcomers_for_season(matches, cur_season))
    joblib.dump(lr_f, ROOT / "models" / "logreg.joblib")
    joblib.dump(xgb_f, ROOT / "models" / "xgb.joblib")
    (ROOT / "models" / "dixon_coles.json").write_text(json.dumps(dc_f.to_dict(), indent=1))

    imp = pd.Series(xgb_f.get_booster().get_score(importance_type="gain")).reindex(feat).fillna(0)
    lr_coef = pd.DataFrame(lr_f[-1].coef_, index=["H", "D", "A"], columns=feat).T

    results = {
        "data": {"matches_total": int(len(matches)),
                 "matches_modelled": int((played.season >= FIRST_TRAIN).sum()),
                 "first_season": season_label(int(matches.season.min())),
                 "last_complete_season": season_label(MAIN_TEST),
                 "latest_match": str(matches.dropna(subset=["fthg"]).date.max().date()),
                 "xg_source": matches.xg_source.value_counts().to_dict(),
                 "xg_proxy": coefs},
        "tuning": {"logreg_C": best_C, "logreg_val_logloss": lr_scores,
                   "xgb": xgb_params,
                   "xgb_val": {f"depth={d},mcw={c}": v[0] for (d, c), v in xgb_scores.items()},
                   "dc_newcomer_prior": dc_prior},
        "metrics": metrics.to_dict(orient="records"),
        "bootstrap_vs_bookies": boot,
        "ensemble_bets_by_pick": by_pick.reset_index().to_dict(orient="records"),
        "betting": bet_table.to_dict(orient="records"),
        "bookmaker_margin": {season_label(s): E.bookmaker_margin_hurdle(preds[preds.season == s])
                             for s in TEST_SEASONS},
        "outcome_rates_test": preds.y.value_counts(normalize=True).sort_index().tolist(),
        "feature_importance_gain": imp.sort_values(ascending=False).to_dict(),
        "logreg_coefficients": lr_coef.round(4).to_dict(),
    }
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print("\nSaved models/ and reports/results.json")


def make_figures(preds, metrics, bets, names, P_of):
    FIG.mkdir(parents=True, exist_ok=True)
    y = preds.y.to_numpy()

    # calibration (home-win and away-win probability), pooled test seasons
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (o, lab) in zip(axes, enumerate(["Home win", "Draw", "Away win"])):
        ax.plot([0, 1], [0, 1], ls="--", c="#888", lw=1)
        for k in ["lr", "xgb", "dc", "ens", "bk"]:
            t = E.calibration_table(y, P_of(preds, k), o, bins=10 if o != 1 else 30)
            t = t[t.n >= 15]
            ax.plot(t.predicted, t.observed, marker="o", ms=4, lw=1.6,
                    label=names[k], c=COLORS[names[k]])
        ax.set_title(lab)
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, .9 if o != 1 else .45)
        ax.set_ylim(0, .9 if o != 1 else .45)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Calibration on 1,140 held-out matches (2023-24 to 2025-26)", x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)

    # log loss by season
    fig, ax = plt.subplots(figsize=(8, 4.8))
    m = metrics[metrics.scope != "All 3 seasons"]
    seasons = m.scope.unique()
    order = ["Logistic regression", "XGBoost", "Dixon-Coles", "Ensemble", "Bookmakers"]
    w = .16
    for i, nm in enumerate(order):
        v = m[m.model == nm].set_index("scope").loc[seasons, "log_loss"]
        ax.bar(np.arange(len(seasons)) + (i - 2) * w, v, w, label=nm, color=COLORS[nm])
    base = m[m.model == "Home-win baseline"].set_index("scope").loc[seasons, "log_loss"]
    ax.scatter(np.arange(len(seasons)), base, marker="_", s=900, c="#555",
               label="Home-win baseline", zorder=3)
    ax.set_xticks(range(len(seasons)), seasons)
    ax.set_ylim(.9, 1.1)
    ax.set_ylabel("Log loss (lower is better)")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.1))
    ax.set_title("Log loss by test season", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "logloss_by_season.png")
    plt.close(fig)

    # betting P&L, main test season, 5% edge
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, odds in zip(axes, ["average", "best"]):
        for k in ["lr", "xgb", "dc", "ens"]:
            r = bets[(names[k], odds, 0.05, season_label(MAIN_TEST))]
            led = r["ledger"]
            if len(led):
                ax.plot(pd.to_datetime(led.date), led.cum_pnl, lw=1.8, c=COLORS[names[k]],
                        label=f"{names[k]} ({r['bets']} bets, ROI {r['roi']:+.1%})")
        ax.axhline(0, c="#444", lw=.8)
        ax.set_title(f"{odds.capitalize()} market odds", loc="left")
        ax.set_ylabel("Cumulative profit (units, 1 unit per bet)")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Value betting, {season_label(MAIN_TEST)}: bet when model EV > 5%", x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "betting_pnl.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "football-data", "mirror"])
    main(ap.parse_args().source)
