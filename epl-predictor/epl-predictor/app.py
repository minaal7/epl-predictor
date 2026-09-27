"""
Streamlit app:  streamlit run app.py

Pick a home and an away side and see win/draw/loss probabilities from each
model, a Dixon-Coles scoreline grid, and (optionally) how the prediction
compares with a bookmaker's odds. A second tab summarises how the models did
against the bookmakers on held-out seasons.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data as D
from src import features as F
from src import xg as X
from src.models import DixonColes

ROOT = Path(__file__).resolve().parent
HOME_C, DRAW_C, AWAY_C, BOOK_C, INK = "#1E5B3A", "#7A8290", "#C7771A", "#6B4E9B", "#16201A"

st.set_page_config(page_title="Premier League predictor", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label { font-family: 'Source Sans 3', system-ui, sans-serif; }
h1, h2, h3, .board-num, .board-team { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif !important; letter-spacing: .01em; }
h1 { font-weight: 700 !important; font-size: 2.6rem !important; margin-bottom: 0 !important; }
.sub { color: #5B665E; font-size: 1.02rem; margin-top: .1rem; max-width: 70ch; }
.board { margin: 1.2rem 0 .4rem; }
.board-bar { display: flex; height: 92px; border-radius: 6px; overflow: hidden; }
.board-seg { display: flex; flex-direction: column; justify-content: center; padding: 0 16px;
             color: #fff; min-width: 64px; transition: flex-grow .5s ease; }
.board-num { font-size: 2.6rem; font-weight: 700; line-height: 1; }
.board-lab { font-size: .9rem; opacity: .9; }
.board-teams { display: flex; justify-content: space-between; margin-bottom: .35rem; }
.board-team { font-size: 1.7rem; font-weight: 600; color: #16201A; }
.chip { display: inline-block; width: 26px; height: 26px; border-radius: 4px; margin-right: 4px;
        text-align: center; line-height: 26px; font-weight: 600; font-size: .85rem; color: #fff; }
.W { background: #1E5B3A; } .D { background: #7A8290; } .L { background: #A33A2B; }
.note { color: #5B665E; font-size: .9rem; }
@media (prefers-reduced-motion: reduce) { .board-seg { transition: none; } }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_everything():
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    matches = X.add_xg(D.load_clean(), results["data"]["xg_proxy"])
    lr = joblib.load(ROOT / "models" / "logreg.joblib")
    xgb = joblib.load(ROOT / "models" / "xgb.joblib")
    dc = DixonColes.from_dict(json.loads((ROOT / "models" / "dixon_coles.json").read_text()))
    return results, matches, lr, xgb, dc


@st.cache_data(show_spinner="Crunching recent form…")
def fixture_features(home: str, away: str, rest_h: int, rest_a: int) -> pd.DataFrame:
    _, matches, *_ = load_everything()
    date = matches.date.max() + pd.Timedelta(days=1)
    return F.features_for_fixture(matches, home, away, date, rest_h, rest_a)


def last_five(matches: pd.DataFrame, team: str) -> list[tuple[str, str]]:
    m = matches[((matches.home == team) | (matches.away == team)) & matches.fthg.notna()]
    out = []
    for r in m.sort_values("date").tail(5).itertuples():
        home = r.home == team
        gf, ga = (r.fthg, r.ftag) if home else (r.ftag, r.fthg)
        res = "W" if gf > ga else "D" if gf == ga else "L"
        opp = r.away if home else r.home
        out.append((res, f"{'v' if home else '@'} {opp} {int(gf)}-{int(ga)} ({r.date:%d %b %Y})"))
    return out


results, matches, lr, xgb, dc = load_everything()
latest = matches[matches.fthg.notna()].date.max()
cur_season = int(matches.season.max())
teams = sorted(set(matches[matches.season == cur_season].home) |
               set(matches[matches.season == cur_season].away))

st.markdown("# Premier League match predictor")
st.markdown(f"<p class='sub'>Win, draw and loss probabilities from three models trained on "
            f"{results['data']['matches_modelled']:,} Premier League matches, tested against "
            f"the bookmakers. Form is up to date as of {latest:%d %B %Y}.</p>",
            unsafe_allow_html=True)

tab_pred, tab_perf = st.tabs(["Predict a match", "How the models did"])

# =========================================================== predict tab
with tab_pred:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away_opts = [t for t in teams if t != home]
    away = c2.selectbox("Away team", away_opts,
                        index=away_opts.index("Chelsea") if "Chelsea" in away_opts else 0)
    with st.expander("Rest days since each team's last league match"):
        r1, r2 = st.columns(2)
        rest_h = r1.slider(f"{home}", 2, 14, 7)
        rest_a = r2.slider(f"{away}", 2, 14, 7)

    row = fixture_features(home, away, rest_h, rest_a)
    Xf = row[F.FEATURES]
    probs = {
        "Logistic regression": lr.predict_proba(Xf)[0],
        "XGBoost": xgb.predict_proba(Xf)[0],
        "Dixon-Coles": dc.predict_proba_pair(home, away),
    }
    probs["Ensemble"] = np.mean(list(probs.values()), axis=0)
    ph, pd_, pa = probs["Ensemble"]

    st.markdown(f"""
    <div class="board" role="img" aria-label="{home} win {ph:.0%}, draw {pd_:.0%}, {away} win {pa:.0%}">
      <div class="board-teams"><span class="board-team">{home}</span><span class="board-team">{away}</span></div>
      <div class="board-bar">
        <div class="board-seg" style="flex-grow:{ph:.4f};background:{HOME_C}">
          <span class="board-num">{ph:.0%}</span><span class="board-lab">{home} win</span></div>
        <div class="board-seg" style="flex-grow:{pd_:.4f};background:{DRAW_C}">
          <span class="board-num">{pd_:.0%}</span><span class="board-lab">Draw</span></div>
        <div class="board-seg" style="flex-grow:{pa:.4f};background:{AWAY_C};text-align:right;align-items:flex-end">
          <span class="board-num">{pa:.0%}</span><span class="board-lab">{away} win</span></div>
      </div>
    </div>
    <p class="note">Ensemble: the average of the logistic regression, XGBoost and Dixon-Coles
    models, the best-scoring combination on held-out seasons.</p>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### Each model's view")
        tbl = pd.DataFrame(probs, index=[f"{home} win", "Draw", f"{away} win"]).T
        st.dataframe(tbl.style.format("{:.1%}"), width="stretch")

        mu, nu = dc.expected_goals(home, away)
        st.markdown("### Form going in")
        for team, elo_col in ((home, "elo_h"), (away, "elo_a")):
            form = last_five(matches, team)
            chips = "".join(f"<span class='chip {r}' title='{t}'>{r}</span>" for r, t in form)
            st.markdown(f"**{team}** &nbsp; Elo {row[elo_col].iloc[0]:.0f}<br>{chips}",
                        unsafe_allow_html=True)
        st.markdown(
            f"<p class='note'>Last five league matches, most recent on the right. Hover a chip for "
            f"the score. Rolling xG difference: {home} {row.h_xgd_r5.iloc[0]:+.2f}, "
            f"{away} {row.a_xgd_r5.iloc[0]:+.2f} per game.</p>", unsafe_allow_html=True)

    with right:
        st.markdown("### Most likely scorelines")
        M = dc.score_matrix(home, away)[:6, :6]
        text = [[f"{v:.0%}" if v >= .005 else "" for v in r] for r in M]
        fig = go.Figure(go.Heatmap(
            z=M, x=[str(i) for i in range(6)], y=[str(i) for i in range(6)], text=text,
            texttemplate="%{text}", colorscale=[[0, "#F6F7F4"], [1, HOME_C]], showscale=False,
            hovertemplate=f"{home} %{{y}} – %{{x}} {away}<br>%{{z:.1%}}<extra></extra>"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=f"{away} goals", yaxis_title=f"{home} goals",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=INK))
        st.plotly_chart(fig, width="stretch")
        top = np.dstack(np.unravel_index(np.argsort(-M.ravel())[:3], M.shape))[0]
        st.markdown(f"<p class='note'>Expected goals: {home} {mu:.2f}, {away} {nu:.2f}. "
                    f"Top scores: " + ", ".join(f"{i}-{j} ({M[i, j]:.0%})" for i, j in top) + ".</p>",
                    unsafe_allow_html=True)

    st.markdown("### Compare with a bookmaker")
    st.markdown("<p class='note'>Enter decimal odds to see the bookmaker's implied probabilities "
                "and whether the model sees value. Leave at 0 to skip.</p>", unsafe_allow_html=True)
    o1, o2, o3 = st.columns(3)
    oh = o1.number_input(f"{home} win", min_value=0.0, value=0.0, step=0.05, format="%.2f")
    od = o2.number_input("Draw", min_value=0.0, value=0.0, step=0.05, format="%.2f")
    oa = o3.number_input(f"{away} win", min_value=0.0, value=0.0, step=0.05, format="%.2f")
    if min(oh, od, oa) > 1:
        inv = np.array([1 / oh, 1 / od, 1 / oa])
        implied = inv / inv.sum()
        ev = probs["Ensemble"] * np.array([oh, od, oa]) - 1
        comp = pd.DataFrame({"Bookmaker (margin removed)": implied, "Model": probs["Ensemble"],
                             "Model expected value per £1": ev},
                            index=[f"{home} win", "Draw", f"{away} win"])
        st.dataframe(comp.style.format({"Bookmaker (margin removed)": "{:.1%}", "Model": "{:.1%}",
                                        "Model expected value per £1": "{:+.2f}"}),
                     width="stretch")
        best = int(np.argmax(ev))
        msg = (f"Bookmaker margin: {inv.sum() - 1:.1%}. ")
        if ev[best] > .05:
            msg += (f"The model rates **{comp.index[best]}** as value (EV {ev[best]:+.0%}). On held-out "
                    "seasons, bets like this did not make a reliable profit, so treat it as a curiosity.")
        else:
            msg += "No outcome clears the 5% value threshold the backtest used."
        st.markdown(msg)

# ============================================================ perf tab
with tab_perf:
    met = pd.DataFrame(results["metrics"])
    last = results["data"]["last_complete_season"]
    st.markdown("### Held-out results")
    st.markdown(
        "<p class='note'>Each season was predicted by models trained only on earlier seasons "
        "(from 2016-17). No random shuffling, so no future data leaks into training.</p>",
        unsafe_allow_html=True)
    scope = st.radio("Test period", [last, "All 3 seasons"], horizontal=True)
    show = (met[met.scope == scope][["model", "log_loss", "accuracy", "rps", "brier"]]
            .rename(columns={"model": "Model", "log_loss": "Log loss", "accuracy": "Accuracy",
                             "rps": "RPS", "brier": "Brier"}).set_index("Model"))
    st.dataframe(show.style.format({"Log loss": "{:.4f}", "Accuracy": "{:.1%}",
                                    "RPS": "{:.4f}", "Brier": "{:.4f}"}),
                 width="stretch")
    st.markdown("<p class='note'>Lower is better for log loss, RPS and Brier.</p>",
                unsafe_allow_html=True)

    b = results["bootstrap_vs_bookies"]["Ensemble"]
    st.markdown(
        f"Over all three test seasons the ensemble's log loss was **{b['diff']:+.4f}** versus the "
        f"bookmakers (95% bootstrap interval {b['ci_low']:+.4f} to {b['ci_high']:+.4f}). "
        "The models close most of the gap between the naive baseline and the market, "
        "but the market is still better.")

    st.image(str(ROOT / "reports" / "figures" / "logloss_by_season.png"))
    st.image(str(ROOT / "reports" / "figures" / "calibration.png"))

    st.markdown("### Would it have made money?")
    bt = pd.DataFrame(results["betting"])
    bt = bt[(bt.edge == 0.05) & (bt.scope == scope)][
        ["model", "odds", "bets", "win_rate", "avg_odds", "profit_units", "roi", "roi_ci_low", "roi_ci_high"]]
    bt.columns = ["Model", "Odds used", "Bets", "Win rate", "Avg odds", "Profit (units)",
                  "ROI", "ROI 95% low", "ROI 95% high"]
    st.dataframe(bt.set_index("Model").style.format(
        {"Win rate": "{:.1%}", "Avg odds": "{:.2f}", "Profit (units)": "{:+.1f}",
         "ROI": "{:+.1%}", "ROI 95% low": "{:+.1%}", "ROI 95% high": "{:+.1%}"}),
        width="stretch")
    st.markdown("<p class='note'>Strategy: 1 unit on the outcome with the highest expected value "
                "whenever the model's edge over the odds exceeds 5%. Every ROI interval spans zero.</p>",
                unsafe_allow_html=True)
    st.image(str(ROOT / "reports" / "figures" / "betting_pnl.png"))
