"""Walk-forward backtest with honest betting metrics.

Every prediction for (season S, week W) is made by a model trained only on
games that had already finished before that week. Nothing is fit on the
future, and — per wf/dataset.py — no feature saw its own game.

We model MARGIN and TOTAL as regressions rather than classifying "cover",
because the quantity that matters is how far our number sits from the
market's. That difference is the edge, and it's what we threshold on.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import dataset as ds

JUICE = -110.0
BREAKEVEN = 110 / 210  # 0.5238


def _payout(odds):
    """Profit on a 1-unit win at American odds."""
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


WIN_UNIT = _payout(JUICE)


def make_model(seed=0):
    # Handles NaN natively, which matters: week-1 games have no current-season form.
    return HistGradientBoostingRegressor(
        max_depth=3, max_iter=300, learning_rate=0.05,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def walk_forward(df, features, target, min_train=400, model_fn=make_model, verbose=True):
    """Predict each week using only strictly-earlier games."""
    df = df.sort_values(["season", "week", "start_date"]).reset_index(drop=True)
    df = df.dropna(subset=[target])
    keys = df[["season", "week"]].drop_duplicates().values.tolist()

    preds = np.full(len(df), np.nan)
    n_fit = 0
    for season, week in keys:
        te = (df["season"] == season) & (df["week"] == week)
        tr = (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
        if tr.sum() < min_train:
            continue
        m = model_fn()
        m.fit(df.loc[tr, features], df.loc[tr, target])
        preds[te.values] = m.predict(df.loc[te, features])
        n_fit += 1

    df = df.copy()
    df["pred"] = preds
    out = df.dropna(subset=["pred"]).copy()
    if verbose:
        print(f"   walk-forward: {n_fit} weekly refits, {len(out)} graded predictions "
              f"({out['season'].min()}-{out['season'].max()})")
    return out


# ------------------------------------------------------------------ metrics


def _record(win, loss, push, unit_profit):
    n = win + loss
    rate = win / n if n else 0.0
    roi = unit_profit / (win + loss + push) if (win + loss + push) else 0.0
    # z against the -110 break-even
    z = (rate - BREAKEVEN) / np.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n) if n else 0.0
    return {"W": win, "L": loss, "P": push, "n": n, "win_pct": rate * 100,
            "units": unit_profit, "roi": roi * 100, "z": z}


def evaluate_spread(res, thresholds=(0, 1, 2, 3, 4, 6)):
    """res needs: pred (model home margin), spread, margin."""
    res = res.copy()
    res["edge"] = res["pred"] - res["market_margin"]   # + => we like the home side

    rows = []
    for t in thresholds:
        sel = res[res["edge"].abs() >= t]
        w = l = p = 0
        profit = 0.0
        for _, r in sel.iterrows():
            back_home = r["edge"] > 0
            # ats_diff > 0 means home covered
            d = r["ats_diff"] if back_home else -r["ats_diff"]
            if d > 0:
                w += 1; profit += WIN_UNIT
            elif d < 0:
                l += 1; profit -= 1.0
            else:
                p += 1
        rows.append({"min_edge": t, **_record(w, l, p, profit)})
    return pd.DataFrame(rows)


def evaluate_total(res, thresholds=(0, 1, 2, 3, 4, 6)):
    res = res.copy()
    res["edge"] = res["pred"] - res["over_under"]      # + => we like the over
    rows = []
    for t in thresholds:
        sel = res[res["edge"].abs() >= t]
        w = l = p = 0
        profit = 0.0
        for _, r in sel.iterrows():
            back_over = r["edge"] > 0
            d = r["total_diff"] if back_over else -r["total_diff"]
            if d > 0:
                w += 1; profit += WIN_UNIT
            elif d < 0:
                l += 1; profit -= 1.0
            else:
                p += 1
        rows.append({"min_edge": t, **_record(w, l, p, profit)})
    return pd.DataFrame(rows)


def baselines(df):
    """What you'd get without a model at all."""
    out = []
    d = df.dropna(subset=["ats_diff"])
    for name, mask_home in [("always home", True), ("always away", False)]:
        v = d["ats_diff"] if mask_home else -d["ats_diff"]
        w, l, p = int((v > 0).sum()), int((v < 0).sum()), int((v == 0).sum())
        out.append({"strategy": name, **_record(w, l, p, w * WIN_UNIT - l)})

    fav_home = d["spread"] < 0
    v = np.where(fav_home, d["ats_diff"], -d["ats_diff"])
    w, l, p = int((v > 0).sum()), int((v < 0).sum()), int((v == 0).sum())
    out.append({"strategy": "always favorite", **_record(w, l, p, w * WIN_UNIT - l)})

    t = df.dropna(subset=["total_diff"])
    for name, sign in [("always over", 1), ("always under", -1)]:
        v = t["total_diff"] * sign
        w, l, p = int((v > 0).sum()), int((v < 0).sum()), int((v == 0).sum())
        out.append({"strategy": name, **_record(w, l, p, w * WIN_UNIT - l)})
    return pd.DataFrame(out)


def accuracy_vs_market(res, market_col, actual_col):
    """Does the model forecast the raw number better than the line does?"""
    m = res.dropna(subset=["pred", market_col, actual_col])
    model_mae = (m["pred"] - m[actual_col]).abs().mean()
    market_mae = (m[market_col] - m[actual_col]).abs().mean()
    blend = 0.5 * m["pred"] + 0.5 * m[market_col]
    blend_mae = (blend - m[actual_col]).abs().mean()
    return {"model_mae": model_mae, "market_mae": market_mae,
            "blend_mae": blend_mae, "n": len(m)}


def _fmt(df_metrics):
    d = df_metrics.copy()
    for c in ("win_pct", "roi", "units", "z"):
        if c in d.columns:
            d[c] = d[c].map(lambda x: f"{x:+.2f}" if c in ("units", "z") else f"{x:.2f}")
    return d.to_string(index=False)


def report(years=(2019, 2020, 2021, 2022, 2023, 2024, 2025), features=None, verbose=True):
    features = features or ds.ALL_FEATURES
    print("=" * 74)
    print("WALK-FORWARD BACKTEST")
    print("=" * 74)
    df = ds.build(list(years), verbose=verbose)
    feats = [f for f in features if f in df.columns]

    print("\n--- No-model baselines ---")
    print(_fmt(baselines(df)))
    print(f"\n  break-even at {JUICE:.0f} is {BREAKEVEN*100:.2f}%")

    print("\n--- Spread model (target: home margin) ---")
    res_m = walk_forward(df, feats, "margin", verbose=verbose)
    acc = accuracy_vs_market(res_m, "market_margin", "margin")
    print(f"   MAE  model={acc['model_mae']:.2f}  market={acc['market_mae']:.2f}  "
          f"50/50 blend={acc['blend_mae']:.2f}  (n={acc['n']})")
    if acc["model_mae"] > acc["market_mae"]:
        print("   -> model is a WORSE margin forecaster than the closing line.")
    print(_fmt(evaluate_spread(res_m)))

    print("\n--- Total model (target: combined points) ---")
    res_t = walk_forward(df, feats, "total_points", verbose=verbose)
    acc_t = accuracy_vs_market(res_t, "over_under", "total_points")
    print(f"   MAE  model={acc_t['model_mae']:.2f}  market={acc_t['market_mae']:.2f}  "
          f"50/50 blend={acc_t['blend_mae']:.2f}  (n={acc_t['n']})")
    print(_fmt(evaluate_total(res_t)))

    print("\n" + "=" * 74)
    print("Read the z column, not the win %. |z| < 2 is noise, and with several")
    print("thresholds tested you should want more than that before risking money.")
    print("=" * 74)
    return df, res_m, res_t


if __name__ == "__main__":
    import sys
    yrs = [int(a) for a in sys.argv[1:]] or [2019, 2021, 2022, 2023, 2024, 2025]
    report(yrs)
