"""Pre-registered edge sweep with false-discovery control.

Searching many hypotheses against the same data is how the original 58.7%
happened. So this module is built around three rules:

  1. Every hypothesis is declared BEFORE it is run, in HYPOTHESES below.
  2. Every result is reported, including the failures. No cherry-picking.
  3. p-values get Benjamini-Hochberg FDR correction for the whole family,
     and anything surviving must ALSO hold up per-season.

A rule is only interesting if it survives all three. Read `q` and
`seasons_positive`, not `win_pct`.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import dataset as ds
from .cache import fetch

BREAKEVEN = 110 / 210          # -110
WIN_UNIT = 100 / 110


# ------------------------------------------------------------------ stats


def grade(df, back_home, market="spread"):
    """Grade a set of bets. back_home is a boolean Series aligned to df."""
    if market == "spread":
        diff = np.where(back_home, df["ats_diff"], -df["ats_diff"])
    else:  # totals: back_home==True means back the OVER
        diff = np.where(back_home, df["total_diff"], -df["total_diff"])
    w = int((diff > 0).sum())
    l = int((diff < 0).sum())
    p = int((diff == 0).sum())
    return w, l, p


def summarise(w, l, p, label, note=""):
    n = w + l
    if n == 0:
        return None
    rate = w / n
    z = (rate - BREAKEVEN) / np.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n)
    pval = stats.binomtest(w, n, BREAKEVEN, alternative="two-sided").pvalue
    roi = (w * WIN_UNIT - l) / (n + p) * 100
    return {"hypothesis": label, "W": w, "L": l, "P": p, "n": n,
            "win_pct": rate * 100, "roi": roi, "z": z, "p": pval, "note": note}


def by_season(df, back_home, market="spread"):
    """How many seasons was this rule above break-even? Guards one-season flukes."""
    pos = tot = 0
    for s, grp in df.groupby("season"):
        bh = back_home.loc[grp.index]
        w, l, _ = grade(grp, bh, market)
        if w + l >= 25:
            tot += 1
            pos += int(w / (w + l) > BREAKEVEN)
    return pos, tot


def test_rule(df, mask, back_home, label, market="spread", min_n=100):
    """Evaluate one pre-registered rule."""
    sub = df[mask].dropna(subset=["ats_diff"] if market == "spread" else ["total_diff"])
    if len(sub) < min_n:
        return None
    bh = back_home.loc[sub.index] if hasattr(back_home, "loc") else pd.Series(back_home, index=sub.index)
    w, l, p = grade(sub, bh, market)
    r = summarise(w, l, p, label)
    if r is None:
        return None
    pos, tot = by_season(sub, bh, market)
    r["seasons_positive"] = f"{pos}/{tot}"
    r["_pos"], r["_tot"] = pos, tot
    return r


def fdr(results, q=0.10):
    """Benjamini-Hochberg. Adds q-value and `survives` flag."""
    res = [r for r in results if r]
    if not res:
        return pd.DataFrame()
    d = pd.DataFrame(res).sort_values("p").reset_index(drop=True)
    m = len(d)
    d["rank"] = d.index + 1
    d["bh_crit"] = q * d["rank"] / m
    # step-up: largest rank where p <= crit
    below = d["p"] <= d["bh_crit"]
    cutoff = d.loc[below, "rank"].max() if below.any() else 0
    d["survives_fdr"] = d["rank"] <= cutoff
    d["q_value"] = (d["p"] * m / d["rank"]).clip(upper=1.0)[::-1].cummin()[::-1]
    return d


# ------------------------------------------------------------------ data


def add_venue_context(df):
    """Travel distance, altitude change, dome, capacity — all knowable pre-kickoff."""
    teams = fetch("/teams/fbs", {"year": 2024})
    loc = {}
    for t in teams if isinstance(teams, list) else []:
        lc = t.get("location") or {}
        if lc.get("latitude") is not None:
            loc[t["school"]] = (lc.get("latitude"), lc.get("longitude"),
                                lc.get("elevation"), bool(lc.get("dome")),
                                lc.get("capacity"))

    def great_circle(a, b):
        if a is None or b is None:
            return np.nan
        lat1, lon1 = np.radians(a[0]), np.radians(a[1])
        lat2, lon2 = np.radians(b[0]), np.radians(b[1])
        d = 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 +
                                  np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
        return 3956 * d

    trav, alt, dome, cap = [], [], [], []
    for h, a in zip(df["home_team"], df["away_team"]):
        H, A = loc.get(h), loc.get(a)
        trav.append(great_circle(A, H) if (H and A) else np.nan)
        try:
            alt.append(float(H[2]) - float(A[2]) if (H and A and H[2] and A[2]) else np.nan)
        except (TypeError, ValueError):
            alt.append(np.nan)
        dome.append(int(H[3]) if H else np.nan)
        cap.append(H[4] if H else np.nan)
    df["travel_miles"] = trav
    df["altitude_diff"] = alt
    df["home_dome"] = dome
    df["home_capacity"] = pd.to_numeric(pd.Series(cap), errors="coerce").values
    return df


def add_schedule_context(df):
    """Look-ahead / let-down / revenge — all derived from the schedule only."""
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    strength = {}
    for _, r in df.iterrows():
        strength[(r["season"], r["home_team"])] = r.get("home_prior_sp")
        strength[(r["season"], r["away_team"])] = r.get("away_prior_sp")

    sched = {}
    for _, r in df.iterrows():
        for me, opp in ((r["home_team"], r["away_team"]), (r["away_team"], r["home_team"])):
            sched.setdefault((r["season"], me), []).append((r["week"], opp))
    for k in sched:
        sched[k].sort()

    def neighbour(season, team, week, offset):
        games = sched.get((season, team), [])
        for i, (w, opp) in enumerate(games):
            if w == week:
                j = i + offset
                if 0 <= j < len(games):
                    return strength.get((season, games[j][1]))
                return np.nan
        return np.nan

    for side in ("home", "away"):
        nxt, prv = [], []
        for _, r in df.iterrows():
            t = r[f"{side}_team"]
            nxt.append(neighbour(r["season"], t, r["week"], +1))
            prv.append(neighbour(r["season"], t, r["week"], -1))
        df[f"{side}_next_opp_sp"] = nxt
        df[f"{side}_prev_opp_sp"] = prv
        cur = pd.to_numeric(df[f"{'away' if side=='home' else 'home'}_prior_sp"], errors="coerce")
        # look-ahead: next opponent much stronger than today's
        df[f"{side}_lookahead"] = pd.to_numeric(pd.Series(nxt), errors="coerce").values - cur
        df[f"{side}_letdown"] = pd.to_numeric(pd.Series(prv), errors="coerce").values - cur
    return df


def add_market_context(df, years):
    """Book dispersion and open->close movement."""
    best_h, best_a, disp, disp_t, nb = {}, {}, {}, {}, {}
    for y in years:
        for st in ("regular",):
            for g in fetch("/lines", {"year": y, "seasonType": st}) or []:
                ls = [b for b in (g.get("lines") or []) if b.get("provider") in ds.VALID_BOOKS]
                sp = [b["spread"] for b in ls if b.get("spread") is not None]
                to = [b["overUnder"] for b in ls if b.get("overUnder") is not None]
                gid = str(g["id"])
                if sp:
                    # best number for a home bettor is the largest (most points)
                    best_h[gid] = max(sp)
                    best_a[gid] = min(sp)
                    disp[gid] = max(sp) - min(sp)
                    nb[gid] = len(sp)
                if to:
                    disp_t[gid] = max(to) - min(to)
    df["best_spread_home"] = df["game_id"].map(best_h)
    df["best_spread_away"] = df["game_id"].map(best_a)
    df["spread_dispersion"] = df["game_id"].map(disp)
    df["total_dispersion"] = df["game_id"].map(disp_t)
    df["n_books_seen"] = df["game_id"].map(nb)
    return df


def load_universe(years, verbose=True):
    df = ds.build(list(years), verbose=verbose)
    if verbose:
        print("   adding venue / schedule / market context...")
    df = add_venue_context(df)
    df = add_schedule_context(df)
    df = add_market_context(df, years)
    df["abs_spread"] = df["spread"].abs()
    df["home_fav"] = (df["spread"] < 0).astype(int)
    df["start_dt"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    df["dow"] = df["start_dt"].dt.dayofweek       # 0=Mon .. 5=Sat, 6=Sun
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["conf_game"] = (df["home_conf"] == df["away_conf"]).astype(int)
    return df


# ------------------------------------------------------------------ hypotheses
#
# PRE-REGISTERED. Each entry: (label, market, fn(df) -> (mask, back_home))
# back_home True = bet the home side (spread) / the OVER (totals).
# Declared before running; every one is reported regardless of outcome.

def _s(df, v):
    return pd.Series(v, index=df.index)


HYPOTHESES = [
    # --- A. classic public-bias spots -------------------------------------
    ("A1  home underdog", "spread",
     lambda d: (d.spread > 0, _s(d, True))),
    ("A2  road favorite (fade)", "spread",
     lambda d: (d.spread > 0, _s(d, False))),
    ("A3  big favorite >=14 (fade)", "spread",
     lambda d: (d.abs_spread >= 14, d.spread > 0)),
    ("A4  big favorite >=21 (fade)", "spread",
     lambda d: (d.abs_spread >= 21, d.spread > 0)),
    ("A5  big favorite >=28 (fade)", "spread",
     lambda d: (d.abs_spread >= 28, d.spread > 0)),
    ("A6  small spread <=3 back home", "spread",
     lambda d: (d.abs_spread <= 3, _s(d, True))),
    ("A7  neutral site back favorite", "spread",
     lambda d: (d.neutral_site == 1, d.spread < 0)),

    # --- B. rest / schedule -----------------------------------------------
    ("B1  home off bye (>=10d rest)", "spread",
     lambda d: (d.home_rest >= 10, _s(d, True))),
    ("B2  away off bye (>=10d rest)", "spread",
     lambda d: (d.away_rest >= 10, _s(d, False))),
    ("B3  rest edge >=4 days", "spread",
     lambda d: (d.rest_diff.abs() >= 4, d.rest_diff > 0)),
    ("B4  home on short week (<=5d)", "spread",
     lambda d: (d.home_rest <= 5, _s(d, False))),
    ("B5  look-ahead: home faces tougher next wk", "spread",
     lambda d: (d.home_lookahead >= 10, _s(d, False))),
    ("B6  let-down: home off tougher opponent", "spread",
     lambda d: (d.home_letdown >= 10, _s(d, False))),

    # --- C. travel / venue -------------------------------------------------
    ("C1  away travels >1500mi", "spread",
     lambda d: (d.travel_miles > 1500, _s(d, True))),
    ("C2  away travels >2000mi", "spread",
     lambda d: (d.travel_miles > 2000, _s(d, True))),
    ("C3  altitude edge home >3000ft", "spread",
     lambda d: (d.altitude_diff > 3000, _s(d, True))),
    ("C4  home dome", "spread",
     lambda d: (d.home_dome == 1, _s(d, True))),
    ("C5  big venue (cap>80k)", "spread",
     lambda d: (d.home_capacity > 80000, _s(d, True))),

    # --- D. calendar -------------------------------------------------------
    ("D1  weeknight (non-Saturday)", "spread",
     lambda d: (d.is_saturday == 0, _s(d, True))),
    ("D2  week 1-2 back home", "spread",
     lambda d: (d.week <= 2, _s(d, True))),
    ("D3  week 1-2 back favorite", "spread",
     lambda d: (d.week <= 2, d.spread < 0)),
    ("D4  late season (wk>=11) back dog", "spread",
     lambda d: (d.week >= 11, d.spread > 0)),
    ("D5  conference game back home", "spread",
     lambda d: (d.conf_game == 1, _s(d, True))),
    ("D6  non-conference back favorite", "spread",
     lambda d: (d.conf_game == 0, d.spread < 0)),

    # --- E. line movement --------------------------------------------------
    ("E1  follow line move (>=1pt)", "spread",
     lambda d: (d.line_move.abs() >= 1, d.line_move < 0)),
    ("E2  fade line move (>=1pt)", "spread",
     lambda d: (d.line_move.abs() >= 1, d.line_move > 0)),
    ("E3  follow big move (>=3pt)", "spread",
     lambda d: (d.line_move.abs() >= 3, d.line_move < 0)),
    ("E4  fade big move (>=3pt)", "spread",
     lambda d: (d.line_move.abs() >= 3, d.line_move > 0)),

    # --- F. book disagreement ---------------------------------------------
    ("F1  books disagree >=1pt, back home", "spread",
     lambda d: (d.spread_dispersion >= 1, _s(d, True))),
    ("F2  books disagree >=1pt, back fav", "spread",
     lambda d: (d.spread_dispersion >= 1, d.spread < 0)),

    # --- G. totals ---------------------------------------------------------
    ("G1  all games UNDER", "total",
     lambda d: (d.over_under.notna(), _s(d, False))),
    ("G2  high total >=60 UNDER", "total",
     lambda d: (d.over_under >= 60, _s(d, False))),
    ("G3  low total <=45 OVER", "total",
     lambda d: (d.over_under <= 45, _s(d, True))),
    ("G4  dome OVER", "total",
     lambda d: (d.home_dome == 1, _s(d, True))),
    ("G5  altitude >3000ft OVER", "total",
     lambda d: (d.altitude_diff > 3000, _s(d, True))),
    ("G6  week 1-2 UNDER", "total",
     lambda d: (d.week <= 2, _s(d, False))),
    ("G7  late season (wk>=11) UNDER", "total",
     lambda d: (d.week >= 11, _s(d, False))),
    ("G8  books disagree >=1pt on total, UNDER", "total",
     lambda d: (d.total_dispersion >= 1, _s(d, False))),
    ("G9  big favorite >=21 OVER", "total",
     lambda d: (d.abs_spread >= 21, _s(d, True))),
    ("G10 weeknight UNDER", "total",
     lambda d: (d.is_saturday == 0, _s(d, False))),
]


def run(years=range(2016, 2026), q=0.10, verbose=True):
    print("=" * 96)
    print(f"PRE-REGISTERED EDGE SWEEP — {len(HYPOTHESES)} hypotheses, "
          f"BH-FDR at q={q}")
    print("=" * 96)
    df = load_universe(years, verbose=verbose)
    print(f"\n   universe: {len(df)} games, {df.season.min()}-{df.season.max()}\n")

    results = []
    for label, market, fn in HYPOTHESES:
        try:
            mask, bh = fn(df)
            mask = mask.fillna(False) if hasattr(mask, "fillna") else mask
            r = test_rule(df, mask, bh, label, market=market)
            if r:
                r["market"] = market
                results.append(r)
            elif verbose:
                print(f"   (skipped, too few games: {label})")
        except Exception as e:
            print(f"   ! {label}: {e}")

    d = fdr(results, q=q)
    if d.empty:
        print("no testable hypotheses")
        return d

    show = d[["hypothesis", "market", "W", "L", "P", "n", "win_pct", "roi", "z",
              "p", "q_value", "seasons_positive", "survives_fdr"]].copy()
    for c, f in [("win_pct", "{:.2f}"), ("roi", "{:+.2f}"), ("z", "{:+.2f}"),
                 ("p", "{:.4f}"), ("q_value", "{:.3f}")]:
        show[c] = show[c].map(f.format)
    print(show.to_string(index=False))

    print("\n" + "=" * 96)
    surv = d[d.survives_fdr]
    if surv.empty:
        best = d.iloc[0]
        print(f"RESULT: nothing survives FDR correction at q={q}.")
        print(f"  Best raw p-value was {best['hypothesis']} (p={best['p']:.4f}, "
              f"q={best['q_value']:.3f}, seasons above break-even {best['seasons_positive']}).")
        print(f"  With {len(d)} hypotheses tested, expect ~{len(d)*0.05:.1f} to hit p<0.05 by chance alone.")
    else:
        print(f"RESULT: {len(surv)} survived FDR. Now check per-season consistency:")
        for _, r in surv.iterrows():
            flag = "" if r["_pos"] >= r["_tot"] - 1 else "   <-- INCONSISTENT across seasons"
            print(f"  {r['hypothesis']:44s} {r['win_pct']:.2f}%  q={r['q_value']:.3f}  "
                  f"seasons {r['seasons_positive']}{flag}")
    print("=" * 96)
    return d


if __name__ == "__main__":
    import sys
    yrs = [int(a) for a in sys.argv[1:]] or list(range(2016, 2026))
    run(yrs)
