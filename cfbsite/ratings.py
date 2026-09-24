"""Opponent-adjusted ratings (ridge regression on game-level EPA per play).

For every team-game:  offense EPA/play = mu + off[team] + def[opponent] + hfa*home
off[t] = how much better than average t's offense is, having removed the quality
of the defenses it faced; def[t] = EPA/play t's defense ALLOWS vs average (lower
is better). Ridge shrinkage matters early in the season, when each team has only
2-4 games; alpha is chosen by cross-validation on held-out games.

Ratings are converted to points with a fitted line (game points ~ EPA/play), so a
team's POWER rating reads as "points better than an average FBS team on a
neutral field", and a projected spread falls out for any pair.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _design(g, teams):
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(g), 2 * n + 1))
    for r, (t, o, h) in enumerate(zip(g["team_id"], g["opp_id"], g["home_ind"])):
        X[r, idx[t]] = 1.0
        X[r, n + idx[o]] = 1.0
        X[r, 2 * n] = h
    return X


FCS = "FCS"


def fit(tg, games, metric="epa", min_plays=15, alphas=(10, 30, 100, 300, 1000, 3000),
        fbs_ids=None, prior=None):
    g = tg[tg[f"o_{metric}_n" if metric == "epa" else "o_succ_n"] >= min_plays].copy()
    # Pool every FCS team into one opponent. Individually they each have one
    # game, so ridge shrinks them to "average" and the model then credits the
    # blowout to home field (HFA came out ~12 pts before pooling).
    if fbs_ids is not None:
        for c in ("team_id", "opp_id"):
            g[c] = np.where(g[c].isin(fbs_ids), g[c], FCS)
    num = "o_epa" if metric == "epa" else "o_success"
    den = "o_epa_n" if metric == "epa" else "o_succ_n"
    g["y"] = g[num] / g[den]
    g["w"] = g[den]
    # shrink toward the preseason prior instead of toward zero: fit deviations
    prior = prior or {}
    po = g["team_id"].map(lambda t: prior.get(t, (0.0, 0.0))[0]).astype(float)
    pdf = g["opp_id"].map(lambda t: prior.get(t, (0.0, 0.0))[1]).astype(float)
    g["y"] = g["y"] - po - pdf
    neutral = games.set_index("event_id")["neutral"].to_dict()
    g["home_ind"] = [0.0 if neutral.get(e) else (1.0 if ha == "home" else -1.0)
                     for e, ha in zip(g["event_id"], g["home_away"])]
    teams = sorted(set(g["team_id"]) | set(g["opp_id"]))
    X = _design(g, teams)

    # choose alpha: 5-fold by game
    ev = g["event_id"].unique()
    rng = np.random.default_rng(3)
    fold = dict(zip(ev, rng.integers(0, 5, len(ev))))
    f = g["event_id"].map(fold).to_numpy()
    best, best_err = alphas[0], np.inf
    for a in alphas:
        err = 0.0
        for k in range(5):
            tr, te = f != k, f == k
            m = Ridge(alpha=a).fit(X[tr], g["y"].to_numpy()[tr], sample_weight=g["w"].to_numpy()[tr])
            err += np.average((m.predict(X[te]) - g["y"].to_numpy()[te]) ** 2, weights=g["w"].to_numpy()[te])
        if err < best_err:
            best, best_err = a, err
    m = Ridge(alpha=best).fit(X, g["y"], sample_weight=g["w"])
    n = len(teams)
    out = pd.DataFrame({"team_id": teams, "off": m.coef_[:n], "def": m.coef_[n:2 * n]})
    out["prior_off"] = out["team_id"].map(lambda t: prior.get(t, (0.0, 0.0))[0])
    out["prior_def"] = out["team_id"].map(lambda t: prior.get(t, (0.0, 0.0))[1])
    out["off"] += out["prior_off"]
    out["def"] += out["prior_def"]
    return out, {"mu": float(m.intercept_), "hfa": float(m.coef_[2 * n]), "alpha": best}


def points_map(tg):
    """Game points ~ a + b * (offense EPA/play). Fit on team-games."""
    g = tg[tg["o_epa_n"] >= 15]
    x = (g["o_epa"] / g["o_epa_n"]).to_numpy()
    y = g["points"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return float(a), float(b), float(r)


def project(home, away, params, neutral=False):
    """Projected (home_pts, away_pts) from rating rows with off/def/mu."""
    mu, hfa, a, b = params["mu"], params["hfa"], params["pts_a"], params["pts_b"]
    h = 0.0 if neutral else 1.0
    h_epa = mu + home["adj_off"] + away["adj_def"] + hfa * h
    a_epa = mu + away["adj_off"] + home["adj_def"] - hfa * h
    return a + b * h_epa, a + b * a_epa
