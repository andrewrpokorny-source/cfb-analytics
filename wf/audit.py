"""Leakage audit. Run this after touching wf/dataset.py.

The premise: a closing spread is a very efficient forecast. After you subtract
it, what's left (`ats_diff`) should be close to unpredictable noise. So if any
pre-kickoff feature shows a large correlation with `ats_diff`, the overwhelmingly
likely explanation is that the feature saw the result — not that we found an edge
worth 20 points of correlation.

This is the check that would have caught the 2025 SRS bug immediately:
end-of-season SRS correlates with ats_diff because it contains the outcome.
"""
import sys

import numpy as np
import pandas as pd

from . import dataset as ds
from .cache import fetch

# |corr| with ats_diff above this is treated as a leak until proven otherwise.
LEAK_THRESHOLD = 0.10


def _corr(a, b):
    m = pd.notna(a) & pd.notna(b)
    if m.sum() < 50:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def check_elo_boundary(year=2025, week=5, verbose=True):
    """Re-derive which side of the game /ratings/elo?week=W sits on."""
    eW = {x["team"]: x["elo"] for x in fetch("/ratings/elo", {"year": year, "week": week})}
    eP = {x["team"]: x["elo"] for x in fetch("/ratings/elo", {"year": year, "week": week - 1})}
    eN = {x["team"]: x["elo"] for x in fetch("/ratings/elo", {"year": year, "week": week + 1})}
    games = fetch("/games", {"year": year, "seasonType": "regular", "week": week})

    d_incl, d_next, mar = [], [], []
    for g in games:
        if not g.get("completed"):
            continue
        h = g.get("homeTeam") or g.get("home_team")
        a = g.get("awayTeam") or g.get("away_team")
        hp = g.get("homePoints") if g.get("homePoints") is not None else g.get("home_points")
        ap = g.get("awayPoints") if g.get("awayPoints") is not None else g.get("away_points")
        if hp is None or ap is None:
            continue
        for t, m in ((h, hp - ap), (a, ap - hp)):
            if t in eW and t in eP and t in eN:
                d_incl.append(eW[t] - eP[t])
                d_next.append(eN[t] - eW[t])
                mar.append(m)

    c_incl = np.corrcoef(d_incl, mar)[0, 1]
    c_next = np.corrcoef(d_next, mar)[0, 1]
    if verbose:
        print(f"  corr(elo[W]-elo[W-1], week-W margin) = {c_incl:+.3f}   "
              f"(high => elo[W] CONTAINS week W)")
        print(f"  corr(elo[W+1]-elo[W], week-W margin) = {c_next:+.3f}   "
              f"(high => elo[W] precedes week W)")
    leaky_at_w = c_incl > 0.3
    if verbose:
        verdict = "elo[W] is POST-week-W -> dataset must use week W-1" if leaky_at_w \
            else "elo[W] is PRE-week-W -> week W would be safe"
        print(f"  => {verdict}")
    return leaky_at_w


def run(years=(2023, 2024, 2025)):
    print("=" * 72)
    print("LEAKAGE AUDIT")
    print("=" * 72)

    print("\n[1] Elo week boundary")
    leaky = check_elo_boundary()
    assert leaky, "Elo boundary flipped — dataset.py's asof=W-1 assumption needs revisiting"
    print("  OK: dataset.py uses week W-1, which is the safe side.")

    print("\n[2] Building dataset...")
    df = ds.build(list(years), verbose=True)
    assert not df.empty, "empty dataset"

    print("\n[3] Structural checks")
    assert (df["asof_week"] == df["week"] - 1).all(), "asof_week must be week-1"
    print("  OK: every row's as-of window is strictly before its own week.")

    wk1 = df[df["week"] == 1]
    adv_cols = [f"{s}_{suf}" for s in ("home", "away") for suf, _ in ds.ADV_FIELDS]
    n_leaked = wk1[adv_cols].notna().sum().sum()
    assert n_leaked == 0, f"week-1 rows carry {n_leaked} current-season stat values"
    print(f"  OK: all {len(wk1)} week-1 rows have NaN current-season stats.")

    # cumulative play counts must not decrease as the season progresses
    bad = 0
    for team in df["home_team"].unique()[:60]:
        sub = df[(df["home_team"] == team)].sort_values(["season", "week"])
        for season, grp in sub.groupby("season"):
            v = grp["home_plays"].dropna().values
            if len(v) > 1 and (np.diff(v) < 0).any():
                bad += 1
    assert bad == 0, f"{bad} teams show shrinking cumulative plays"
    print("  OK: cumulative play counts are monotone within a season.")

    print("\n[4] Spread sign convention")
    c, n = _corr(df["market_margin"], df["margin"])
    print(f"  corr(market implied home margin, actual home margin) = {c:+.3f}  (n={n})")
    assert c > 0.6, "spread sign convention looks inverted"
    mae = (df["market_margin"] - df["margin"]).abs().mean()
    print(f"  market MAE on margin = {mae:.2f} pts")
    print("  OK: -spread is the market's implied home margin.")

    print("\n[5] Feature vs. post-market residual (the leak canary)")
    print(f"    flagging |corr| > {LEAK_THRESHOLD:.2f} against ats_diff\n")
    print(f"    {'feature':24s} {'vs ats_diff':>12s} {'vs margin':>11s} {'n':>7s}")
    print("    " + "-" * 58)

    flags = []
    for f in ds.ALL_FEATURES:
        if f not in df.columns:
            continue
        c_ats, n = _corr(df[f], df["ats_diff"])
        c_mar, _ = _corr(df[f], df["margin"])
        mark = ""
        if not np.isnan(c_ats) and abs(c_ats) > LEAK_THRESHOLD:
            mark = "  <== FLAG"
            flags.append((f, c_ats))
        print(f"    {f:24s} {c_ats:+12.3f} {c_mar:+11.3f} {n:7d}{mark}")

    # The known-bad feature, shown for contrast: same-season SRS.
    print("\n[6] Control: the feature that caused the original bug")
    srs_now = {}
    for y in years:
        recs = fetch("/ratings/srs", {"year": y})
        for r in recs:
            if isinstance(r, dict) and r.get("team"):
                srs_now[(y, r["team"])] = r.get("rating")
    same_season = df.apply(
        lambda r: (srs_now.get((r["season"], r["home_team"]), np.nan) or np.nan)
        - (srs_now.get((r["season"], r["away_team"]), np.nan) or np.nan), axis=1)
    c_leak, n_leak = _corr(same_season, df["ats_diff"])
    c_safe, _ = _corr(df["prior_srs_diff"], df["ats_diff"])
    print(f"    same-season srs_diff vs ats_diff = {c_leak:+.3f}  (n={n_leak})  <-- leaked")
    print(f"    prior-season srs_diff vs ats_diff = {c_safe:+.3f}          <-- safe")
    print("    The gap between those two lines is the entire 58.7% mirage.")

    print("\n" + "=" * 72)
    if flags:
        print(f"RESULT: {len(flags)} feature(s) flagged — investigate before trusting any backtest:")
        for f, c in sorted(flags, key=lambda x: -abs(x[1])):
            print(f"   {f:24s} corr={c:+.3f}")
    else:
        print("RESULT: clean. No pre-kickoff feature shows suspicious residual correlation.")
    print("=" * 72)
    return df, flags


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] or [2023, 2024, 2025]
    run(yrs)
