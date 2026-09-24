"""Expected points (EP) model and expected points added (EPA).

EP(state) = expected value of the NEXT score in the half, from the offense's
point of view, given down, distance, yards to the end zone and time left in
the half. Trained on ~1.1M scrimmage plays from CFBD 2018-2025 (the local
data_cache). EPA for a play = EP after - EP before, where "after" is:
  * the points scored, if the play changed the score (+7 TD incl. PAT, -7 for
    a defensive TD, -2 for a safety);
  * -EP of the new offense's state, if possession changed;
  * EP of the resulting state otherwise; 0 at the end of a half.

    python -m cfbsite.ep          # train, validate vs CFBD's own EPA, save
"""
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(HERE, "ep_model.joblib")

SCRIMMAGE = {"Rush", "Pass Reception", "Pass Incompletion", "Passing Touchdown",
             "Rushing Touchdown", "Sack", "Pass Interception Return", "Interception",
             "Fumble Recovery (Own)", "Fumble Recovery (Opponent)",
             "Interception Return Touchdown", "Fumble Return Touchdown",
             "Pass Interception", "Pass Completion", "Fumble"}
FEATURES = ["yards_to_goal", "down", "distance", "half_secs", "goal_to_go"]

_model = None


def features(ytg, down, dist, half_secs):
    ytg = np.clip(np.asarray(ytg, float), 1, 99)
    dist = np.clip(np.asarray(dist, float), 1, 30)
    return pd.DataFrame({
        "yards_to_goal": ytg,
        "down": np.clip(np.asarray(down, float), 1, 4),
        "distance": dist,
        "half_secs": np.clip(np.asarray(half_secs, float), 0, 1800),
        "goal_to_go": (dist >= ytg).astype(float),
    })[FEATURES]


def _load_cfbd_plays():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "data_cache", "plays_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        for p in d:
            clk = p.get("clock") or {}
            rows.append((p["gameId"], int(p["id"]), p.get("period") or 0, clk.get("minutes") or 0,
                         clk.get("seconds") or 0, p.get("offense"), p.get("home"),
                         p.get("offenseScore") or 0, p.get("defenseScore") or 0,
                         p.get("down") or 0, p.get("distance") or 0, p.get("yardsToGoal"),
                         p.get("playType"), p.get("ppa")))
    df = pd.DataFrame(rows, columns=["game", "pid", "period", "mm", "ss", "offense", "home",
                                     "off_sc", "def_sc", "down", "distance", "ytg", "ptype", "ppa"])
    return df.drop_duplicates(["game", "pid"]).sort_values(["game", "pid"]).reset_index(drop=True)


def next_score_labels(df):
    """Points of the next score in the same half, from each play's offense view."""
    home_off = df["offense"] == df["home"]
    df["home_sc"] = np.where(home_off, df["off_sc"], df["def_sc"])
    df["away_sc"] = np.where(home_off, df["def_sc"], df["off_sc"])
    df["half"] = np.where(df["period"] <= 2, 1, np.where(df["period"] <= 4, 2, 3))
    lab = np.zeros(len(df))
    for (_, _), g in df.groupby(["game", "half"], sort=False):
        idx = g.index.to_numpy()
        hs, as_ = g["home_sc"].to_numpy(), g["away_sc"].to_numpy()
        # score change AT play k (scores are recorded after the play)
        prev_h = np.r_[hs[0] if g["half"].iloc[0] == 1 else hs[0], hs[:-1]]
        prev_a = np.r_[as_[0], as_[:-1]]
        dh, da = hs - prev_h, as_ - prev_a
        # the first play of a half carries no prior; treat as no change
        dh[0] = da[0] = 0
        change = (dh != 0) | (da != 0)
        # next change at or after each play: scan backwards
        nxt_h = np.zeros(len(g)); nxt_a = np.zeros(len(g))
        cur_h = cur_a = 0.0
        for k in range(len(g) - 1, -1, -1):
            if change[k]:
                cur_h, cur_a = dh[k], da[k]
            nxt_h[k], nxt_a[k] = cur_h, cur_a
        is_home = (g["offense"] == g["home"]).to_numpy()
        lab[idx] = np.where(is_home, nxt_h - nxt_a, nxt_a - nxt_h)
    # a TD that shows +6 (PAT recorded on a later play) counts as 7
    lab = np.where(np.abs(lab) == 6, np.sign(lab) * 7, lab)
    return np.clip(lab, -8, 8)


def train(verbose=True):
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _load_cfbd_plays()
    df["label"] = next_score_labels(df)
    df["half_secs"] = np.where(df["period"].isin([1, 3]), 900, 0) + df["mm"] * 60 + df["ss"]
    s = df[df["ptype"].isin(SCRIMMAGE) & df["down"].between(1, 4) & df["ytg"].between(1, 99)
           & (df["period"] <= 4)].copy()
    X = features(s["ytg"], s["down"], s["distance"], s["half_secs"]).set_index(s.index)
    games = s["game"].unique()
    rng = np.random.default_rng(7)
    test_games = set(rng.choice(games, size=len(games) // 5, replace=False))
    te = s["game"].isin(test_games)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=48,
                                      min_samples_leaf=200, random_state=7)
    m.fit(X[~te], s.loc[~te, "label"])
    pred = m.predict(X[te])
    if verbose:
        mae = np.abs(pred - s.loc[te, "label"]).mean()
        base = np.abs(s.loc[te, "label"].mean() - s.loc[te, "label"]).mean()
        print(f"   trained on {int((~te).sum()):,} plays; held-out games MAE {mae:.3f} vs {base:.3f} baseline")
    m.fit(X, s["label"])                           # final fit on everything
    joblib.dump(m, MODEL_PATH, compress=3)
    global _model
    _model = m
    if verbose:
        _validate(s)
    return m


def _validate(s):
    """Compare our EPA with CFBD's published EPA on the same plays."""
    s = s.copy()
    s["ep0"] = ep(s["ytg"], s["down"], s["distance"], s["half_secs"])
    g = s.groupby("game", sort=False)
    s["n_ep"] = g["ep0"].shift(-1)
    s["n_off"] = g["offense"].shift(-1)
    s["n_half"] = g["period"].shift(-1)
    home_off = s["offense"] == s["home"]
    tot = s["home_sc"] + s["away_sc"]
    s["scored"] = g["home_sc"].shift(-1).add(g["away_sc"].shift(-1)) != tot  # proxy only
    epa = np.where(s["n_off"] == s["offense"], s["n_ep"], -s["n_ep"]) - s["ep0"]
    ok = s["ppa"].notna() & pd.notna(epa) & (s["ptype"].isin({"Rush", "Pass Reception", "Pass Incompletion", "Sack"}))
    r = np.corrcoef(epa[ok], s.loc[ok, "ppa"])[0, 1]
    print(f"   EP by down at own 25, 1st&10, full half: "
          f"{ep([75], [1], [10], [1800])[0]:+.2f}  | at opp 25: {ep([25], [1], [10], [1800])[0]:+.2f}"
          f"  | opp 1, 1st&goal: {ep([1], [1], [1], [1800])[0]:+.2f}")
    print(f"   corr(our EPA, CFBD EPA) on {int(ok.sum()):,} non-scoring run/pass plays: {r:.3f}")


def model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def ep(ytg, down, dist, half_secs):
    return model().predict(features(ytg, down, dist, half_secs))


if __name__ == "__main__":
    train()
