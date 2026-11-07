"""Game prediction engine: opponent-adjusted EPA ratings, anchored to the market.

The ratings model alone forecasts margins slightly worse than the closing line
(see research/FINDINGS.md), so predictions are built as

    final = market + w * (model - market)

with w fit walk-forward on past seasons. The model's main jobs are filling in
games the market doesn't price and nudging the market where they disagree.
Every rating used for a game in week W is fit only on games before week W.
"""
import math
import os

import numpy as np
import pandas as pd

from cfbsite import ratings as R
from cfbsite import priors as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_season(season):
    d = os.path.join(ROOT, "site_data", str(season))
    games = pd.read_parquet(os.path.join(d, "games.parquet"))
    tg = pd.read_parquet(os.path.join(d, "team_games.parquet"))
    teams = pd.read_parquet(os.path.join(d, "teams.parquet"))
    pri = P.load_or_build(season, teams)
    return games, tg, teams, pri


class Ratings:
    """Ratings as of a cutoff: fit on team-games from weeks < before_week."""

    def __init__(self, tg, games, teams, prior, before_week=None):
        sub = tg if before_week is None else tg[tg["week"] < before_week]
        self.fbs = set(teams["team_id"])
        self.prior = prior or {}
        self.n_games = sub["event_id"].nunique()
        rt, prm = R.fit(sub, games, fbs_ids=self.fbs, prior=self.prior)
        a, b, _ = R.points_map(sub if len(sub) >= 200 else tg_fallback(tg))
        self.off = dict(zip(rt["team_id"], rt["off"]))
        self.dfn = dict(zip(rt["team_id"], rt["def"]))
        self.mu, self.hfa, self.a, self.b = prm["mu"], prm["hfa"], a, b

    def _rating(self, team_id):
        tid = team_id if team_id in self.fbs else R.FCS
        if tid in self.off:
            return self.off[tid], self.dfn[tid]
        return self.prior.get(tid, (0.0, 0.0))   # FBS team with no games yet

    def project(self, home_id, away_id, neutral=False):
        ho, hd = self._rating(home_id)
        ao, ad = self._rating(away_id)
        h = 0.0 if neutral else 1.0
        h_epa = self.mu + ho + ad + self.hfa * h
        a_epa = self.mu + ao + hd - self.hfa * h
        hp, ap = self.a + self.b * h_epa, self.a + self.b * a_epa
        return hp, ap


_FALLBACK = {}


def tg_fallback(tg):
    """EPA->points map needs a decent sample; early weeks borrow the full table
    of the previous season if available, else the current table."""
    return _FALLBACK.get("tg", tg)


def predict_rows(rows, rt):
    out = []
    for r in rows.itertuples(index=False):
        hp, ap = rt.project(r.home_id, r.away_id, bool(r.neutral))
        out.append((hp - ap, hp + ap, hp, ap))
    return pd.DataFrame(out, columns=["model_margin", "model_total", "model_home_pts", "model_away_pts"],
                        index=rows.index)


def walk_forward(season, first_week=2, verbose=True):
    """Predict every completed game of a season using only earlier weeks."""
    games, tg, teams, pri = load_season(season)
    done = games[games["completed"].astype(bool)].copy()
    out = []
    for wk in sorted(done["week"].unique()):
        if wk < first_week:
            continue
        rows = done[done["week"] == wk]
        if tg[tg["week"] < wk]["event_id"].nunique() < 40:
            continue
        rt = Ratings(tg, games, teams, pri, before_week=wk)
        p = predict_rows(rows, rt)
        out.append(pd.concat([rows, p], axis=1).assign(season=season))
        if verbose:
            print(f"   {season} wk {wk}: {len(rows)} games (fit on {rt.n_games})")
    df = pd.concat(out, ignore_index=True)
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["market_margin"] = -df["spread"]
    df["market_total"] = df["total"]
    return df


# ------------------------------------------------------------------ calibration
def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def win_prob(margin, sigma):
    return phi(margin / sigma)


def fit_blend(df, model_col, market_col, actual_col):
    """Least-squares w in  actual - market = w * (model - market)."""
    d = df.dropna(subset=[model_col, market_col, actual_col])
    x = (d[model_col] - d[market_col]).to_numpy()
    y = (d[actual_col] - d[market_col]).to_numpy()
    w = float((x * y).sum() / (x * x).sum())
    return max(0.0, min(1.0, w))


def fit_sigma(margins, home_won):
    """MLE of the normal sd turning a projected margin into a win probability."""
    best, best_ll = None, -np.inf
    for s in np.arange(8.0, 24.01, 0.1):
        p = np.clip([win_prob(m, s) for m in margins], 1e-6, 1 - 1e-6)
        ll = np.sum(home_won * np.log(p) + (1 - home_won) * np.log(1 - p))
        if ll > best_ll:
            best, best_ll = float(s), ll
    return best
