"""Paper trading with closing-line-value tracking.

CLV is the point of this module. Results are mostly variance — you need
hundreds of bets before a win rate means anything. But CLV tells you within
weeks whether you are getting a better number than the market's final word,
and consistently beating the close is the precondition for winning long term.

Workflow:
    python -m wf.paper log     2026 1      # record picks at the number you'd take
    python -m wf.paper close   2026 1      # capture closing lines after kickoff
    python -m wf.paper grade   2026        # settle results
    python -m wf.paper report  2026        # CLV + ROI

Nothing here risks money. The ledger is a CSV.
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import odds, shop
from .cache import fetch

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "paper_ledger.csv")

COLUMNS = ["bet_id", "logged_at", "season", "week", "game", "home", "away",
           "commence", "market", "side", "line_taken", "price_taken", "book",
           "fair_line_at_log", "ev_at_log", "n_books_at_log", "strategy",
           "close_line", "close_price", "clv_points", "clv_pct",
           "home_score", "away_score", "result", "profit_units",
           "price_assumed", "middle_id", "note"]


_STR_COLS = ["bet_id", "logged_at", "game", "home", "away", "commence",
             "market", "side", "book", "strategy", "result"]


def _load():
    if os.path.exists(LEDGER):
        df = pd.read_csv(LEDGER)
    else:
        df = pd.DataFrame(columns=COLUMNS)
    # keep object dtype on text columns so assigning e.g. "LOSS" into an
    # all-NaN float column does not raise in future pandas
    for c in _STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object")
    return df


def _save(df):
    df.to_csv(LEDGER, index=False)
    print(f"   ledger -> {LEDGER} ({len(df)} rows)")


# ------------------------------------------------------------ strategies
#
# Only the three candidates that survived the alpha hunt (see research/FINDINGS.md).
# NONE is proven. They are logged to gather evidence, not because they win.

MAX_UNDER_PRICE = -112   # the unders edge is +1.5% at -115 and negative at -120


def _price_ok(price, floor):
    """True if American `price` pays at least as well as `floor` (e.g. -112)."""
    return shop.to_decimal(price) >= shop.to_decimal(floor) - 1e-9


def strategy_high_total_under(offers, min_total=58.0, **_):
    """Candidate #2: UNDER on high totals. Mechanism: posted totals over-extrapolate
    (actual points rise only ~0.9 per posted point; the shortfall is 2nd-half).
    54.3% at OU>=60 2017-25, CI still contains break-even. Price-gated at -112:
    with a real price worse than that, it is not logged."""
    if offers.empty:
        return offers
    m = ((offers["market"] == "total") & (offers["side"] == "UNDER")
         & (offers["fair_line"] >= min_total))
    out = offers[m].copy()
    real = ~out["price_assumed"]
    ok = out["best_price"].map(lambda p: _price_ok(p, MAX_UNDER_PRICE))
    out = out[~real | ok]
    out["strategy"] = f"high_total_under_{min_total:g}"
    out["note"] = ""
    return out


def strategy_open_field_dog(offers, features=None, threshold=0.5, **_):
    """Candidate #3: back the underdog when the favourite plays in the higher
    open-field-yards environment. Lane reported 56.2% (2018-25); rebuilt from
    scratch in Sep 2026 it reproduces at only ~53.4% above a 0.5 edge (z<1).
    Weakest of the three."""
    if offers.empty or features is None or features.empty:
        return offers.iloc[0:0]
    f = features.set_index("game")
    sp = offers[offers["market"] == "spread"].copy()
    sp = sp[sp["game"].isin(f.index)]
    if sp.empty:
        return sp
    sp["ofyd_mis"] = sp["game"].map(f["ofyd_mis"])
    sp["home_spread"] = sp["game"].map(f["spread"])
    sp = sp.dropna(subset=["ofyd_mis", "home_spread"])
    sp = sp[sp["home_spread"] != 0]
    home_fav = sp["home_spread"] < 0
    fav_env = np.where(home_fav, sp["ofyd_mis"], -sp["ofyd_mis"])
    dog = np.where(home_fav, sp["away"], sp["home"])
    out = sp[(sp["side"] == dog) & (fav_env >= threshold)].copy()
    out["strategy"] = f"open_field_dog_{threshold:g}"
    out["note"] = ""
    return out


def strategy_shop_value(offers, min_gain=0.01, min_books=4, **_):
    """Pure execution test: the best book beats the median by >=1% EV.

    Needs a real multi-book consensus. On 2-3 books the de-vigged fair price
    comes from the very books being shopped, so the best offer always looks
    +EV; run unguarded on 2025 wk5 CFBD data this went 22-46 (32.4%)."""
    if offers.empty:
        return offers
    out = offers[(offers["shop_gain"] >= min_gain)
                 & (offers["n_books"] >= min_books)
                 & (~offers["both_sides_pos"])
                 & (~offers["price_assumed"])].copy()
    out["strategy"] = f"shop_gain_{min_gain:g}"
    out["note"] = ""
    return out


STRATEGIES = {
    "high_total_under": strategy_high_total_under,
    "open_field_dog": strategy_open_field_dog,
    "shop_value": strategy_shop_value,
}


def _live_features(season, week):
    """As-of-week features for this week's games, keyed like the quotes."""
    try:
        from . import dataset as ds
        f = ds.build_season(season, "regular", verbose=False)
    except Exception as e:  # API down / quota
        print(f"   features unavailable ({e.__class__.__name__}); open_field_dog skipped")
        return pd.DataFrame()
    if f.empty:
        return f
    f = f[f["week"] == week].copy()
    f["game"] = f["away_team"] + " @ " + f["home_team"]
    return f[["game", "ofyd_mis", "spread"]]


def _drop_started(offers, lead_minutes=0):
    """Keep only games that have not kicked off yet."""
    if offers.empty or "commence" not in offers.columns:
        return offers
    now = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=lead_minutes)
    ts = pd.to_datetime(offers["commence"], errors="coerce", utc=True)
    future = ts.notna() & (ts > now)
    dropped = len(offers) - int(future.sum())
    if dropped:
        print(f"   skipped {dropped} offers on games already under way / finished")
    return offers[future].copy()


def _middle_rows(quotes, season, week, now, min_width=2.5):
    """Candidate #1: cross-book total middles, logged as two linked legs."""
    m = shop.find_middles(quotes, min_width=min_width)
    if m.empty:
        return []
    ts = pd.to_datetime(m["commence"], errors="coerce", utc=True)
    m = m[ts > pd.Timestamp.now(tz="UTC")]
    rows = []
    for _, r in m.iterrows():
        mid = f"{season}-{week}-{r['game']}-mid{r['over_line']:g}-{r['under_line']:g}"
        for side, book, line, price in (("OVER", r["over_book"], r["over_line"], r["over_price"]),
                                        ("UNDER", r["under_book"], r["under_line"], r["under_price"])):
            rows.append({
                "bet_id": f"{mid}-{side}", "logged_at": now, "season": season, "week": week,
                "game": r["game"], "home": r["home"], "away": r["away"], "commence": r["commence"],
                "market": "total", "side": side, "line_taken": line, "price_taken": price,
                "book": book, "fair_line_at_log": r["fair_line"], "ev_at_log": r["ev_per_middle"],
                "n_books_at_log": np.nan, "strategy": f"total_middle_{min_width:g}",
                "price_assumed": bool(r["price_assumed"]), "middle_id": mid,
                "note": f"width {r['width']:g}",
            })
    if rows:
        print(f"   total_middle: {len(rows)//2} middles")
    return rows


# ------------------------------------------------------------ commands


def cmd_log(season, week, strategies=None, prefer="theoddsapi"):
    """Record the picks you would place, at the best number available now."""
    quotes = odds.get_quotes(season, week, prefer=prefer)
    if not quotes:
        print("   no quotes available; nothing logged")
        return
    # min_edge=-1: strategies decide, not the EV filter. With a real consensus
    # an UNDER at the fair number is ~-2% EV and the old default dropped it.
    offers = shop.best_offers(quotes, min_edge=-1.0)
    if offers.empty:
        print("   no priceable offers")
        return

    # A "pick" on a game that already kicked off is not a pick, it is hindsight.
    # CFB week 1 spans ~12 days, so mid-week runs will always see finished games.
    offers = _drop_started(offers)
    if offers.empty:
        print("   every game in this week has already kicked off; nothing to log")
        return

    features = _live_features(season, week) if (strategies is None or "open_field_dog" in strategies) else None
    picks = []
    for name in (strategies or STRATEGIES):
        sel = STRATEGIES[name](offers, features=features)
        if not sel.empty:
            picks.append(sel)
            print(f"   {name}: {len(sel)} picks")

    new = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    now = datetime.now(timezone.utc).isoformat()
    rows = _middle_rows(quotes, season, week, now)
    for _, r in new.iterrows():
        rows.append({
            "bet_id": f"{season}-{week}-{r['game']}-{r['market']}-{r['side']}-{r['strategy']}",
            "logged_at": now, "season": season, "week": week,
            "game": r["game"], "home": r["home"], "away": r["away"],
            "commence": r["commence"], "market": r["market"], "side": r["side"],
            "line_taken": r["best_line"], "price_taken": r["best_price"],
            "book": r["best_book"], "fair_line_at_log": r["fair_line"],
            "ev_at_log": r["ev_best"], "n_books_at_log": r["n_books"],
            "strategy": r["strategy"],
            "price_assumed": bool(r.get("price_assumed", False)),
            "note": r.get("note", ""),
        })

    if not rows:
        print("   no picks matched any strategy")
        return
    led = _load()
    add = pd.DataFrame(rows).reindex(columns=COLUMNS)
    add = add[~add["bet_id"].isin(set(led["bet_id"]))]
    if add.empty:
        print("   all picks already logged")
        return
    merged = add if led.empty else pd.concat([led, add], ignore_index=True)
    _save(merged)
    print(f"   logged {len(add)} new paper bets")


def cmd_close(season, week, prefer="theoddsapi"):
    """Capture the closing number for logged bets. Run just before kickoff."""
    led = _load()
    if led.empty:
        print("   empty ledger")
        return
    target = led[(led.season == season) & (led.week == week) & led.close_line.isna()]
    if target.empty:
        print("   nothing pending for that week")
        return

    quotes = odds.get_quotes(season, week, prefer=prefer)
    if not quotes:
        print("   no quotes; try again before kickoff")
        return
    cons = shop.consensus(quotes)

    n = 0
    for idx, r in target.iterrows():
        f = cons.get((r["game"], r["market"], r["side"]))
        if not f:
            continue

        if r["market"] == "ml":
            # Moneylines have no "line" — CLV is a price difference, so measure
            # it as the gap between the fair probability you beat and the
            # closing fair probability.
            close_p = f.get("fair_prob")
            if close_p is None or pd.isna(r["price_taken"]):
                continue
            taken_p = shop.to_implied(int(r["price_taken"]))
            led.at[idx, "close_price"] = shop.to_american(close_p)
            led.at[idx, "clv_pct"] = float(close_p - taken_p)
            n += 1
            continue

        close_line = f.get("fair_line")
        if close_line is None:
            continue
        led.at[idx, "close_line"] = close_line
        taken = r["line_taken"]
        if pd.notna(taken):
            # CLV in points, signed so that positive = you got the better number
            if r["market"] == "total" and r["side"] == "UNDER":
                clv = taken - close_line          # under wants a higher number
            elif r["market"] == "total":
                clv = close_line - taken          # over wants a lower number
            else:
                clv = taken - close_line          # spread: more points is better
            led.at[idx, "clv_points"] = clv
            kind = "spread" if r["market"] == "spread" else "total"
            # convert points of CLV into probability, via the empirical distribution
            led.at[idx, "clv_pct"] = (shop.sf(kind, -abs(clv)) - shop.sf(kind, 0)) * np.sign(clv)
        n += 1

    _save(led)
    print(f"   captured closing lines for {n} bets")

    # CLV is only meaningful if the close was genuinely observed later than the
    # log. Same-snapshot gives clv = best-minus-median, positive by construction.
    graded = led[(led.season == season) & (led.week == week) & led.clv_points.notna()]
    if len(graded) and (graded.clv_points > 0).mean() > 0.95:
        print("   *** WARNING: >95% of bets 'beat the close'. That is what you get")
        print("       when log and close read the SAME snapshot — CLV then just")
        print("       re-measures the shopping gap. Run `close` shortly before")
        print("       kickoff, hours or days after `log`, for a real reading. ***")


def cmd_grade(season):
    """Settle bets against final scores."""
    led = _load()
    if led.empty:
        print("   empty ledger")
        return
    pending = led[(led.season == season) & led.result.isna()]
    if pending.empty:
        print("   nothing to grade")
        return

    scores = {}
    for st in ("regular", "postseason"):
        for g in fetch("/games", {"year": season, "seasonType": st}) or []:
            if not g.get("completed"):
                continue
            h = g.get("homeTeam") or g.get("home_team")
            a = g.get("awayTeam") or g.get("away_team")
            hp = g.get("homePoints") if g.get("homePoints") is not None else g.get("home_points")
            ap = g.get("awayPoints") if g.get("awayPoints") is not None else g.get("away_points")
            if None in (h, a, hp, ap):
                continue
            scores[f"{a} @ {h}"] = (hp, ap)

    n = 0
    for idx, r in pending.iterrows():
        s = scores.get(r["game"])
        if not s:
            continue
        hp, ap = s
        led.at[idx, "home_score"], led.at[idx, "away_score"] = hp, ap
        line, side, market = r["line_taken"], r["side"], r["market"]

        if market == "spread":
            margin = (hp - ap) if side == r["home"] else (ap - hp)
            v = margin + line
        elif market == "total":
            v = (hp + ap) - line
            if side == "UNDER":
                v = -v
        else:  # moneyline
            won = (hp > ap) if side == r["home"] else (ap > hp)
            v = 1 if won else -1

        res = "PUSH" if v == 0 else ("WIN" if v > 0 else "LOSS")
        led.at[idx, "result"] = res
        led.at[idx, "profit_units"] = (0.0 if res == "PUSH"
                                       else shop.to_decimal(r["price_taken"]) if res == "WIN"
                                       else -1.0)
        n += 1
    _save(led)
    print(f"   graded {n} bets")


def cmd_report(season=None):
    led = _load()
    if led.empty:
        print("   empty ledger — run `log` first")
        return
    if season:
        led = led[led.season == season]

    print("=" * 78)
    print(f"PAPER LEDGER {'— season ' + str(season) if season else ''}")
    print("=" * 78)
    print(f"  logged: {len(led)}   with closing line: {led.close_line.notna().sum()}"
          f"   graded: {led.result.notna().sum()}")

    clv = led.dropna(subset=["clv_points"])
    if not clv.empty:
        print(f"\n  --- CLOSING LINE VALUE (the number that matters early) ---")
        for strat, g in clv.groupby("strategy"):
            beat = (g.clv_points > 0).mean() * 100
            print(f"    {strat:28s} n={len(g):4d}  mean CLV={g.clv_points.mean():+.3f} pts  "
                  f"beat close {beat:.1f}%")
        beat_all = (clv.clv_points > 0).mean() * 100
        print(f"    {'ALL':28s} n={len(clv):4d}  mean CLV={clv.clv_points.mean():+.3f} pts  "
              f"beat close {beat_all:.1f}%")
        print("\n    Positive mean CLV is the leading indicator. Negative CLV with a")
        print("    winning record means you got lucky, not sharp.")

    if "price_assumed" in led.columns:
        pa = led["price_assumed"].astype(str).str.lower().eq("true")
        if pa.any():
            print(f"\n  !! {int(pa.sum())} of {len(led)} bets carry an ASSUMED -110 price (CFBD has")
            print("     no prices). Their ROI below is not a real number. Set ODDS_API_KEY.")

    done = led.dropna(subset=["result"])
    if "middle_id" in done.columns and done["middle_id"].notna().any():
        mids = done[done["middle_id"].notna()].groupby("middle_id")
        full = [g for _, g in mids if len(g) == 2]
        if full:
            hit = sum(1 for g in full if (g["result"] == "WIN").all())
            pnl = sum(g["profit_units"].sum() for g in full)
            print(f"\n  --- TOTAL MIDDLES (scored per pair) ---")
            print(f"    {len(full)} middles, {hit} hit both legs ({hit/len(full)*100:.1f}%), "
                  f"P&L {pnl:+.2f}u on {2*len(full)}u staked")
            print("    historical: 9.2% hit, +8.6% per middle (2019-25, n=261)")
    if not done.empty:
        print(f"\n  --- RESULTS (noisy; needs hundreds of bets) ---")
        for strat, g in done.groupby("strategy"):
            w = int((g.result == "WIN").sum()); l = int((g.result == "LOSS").sum())
            p = int((g.result == "PUSH").sum())
            n = w + l
            roi = g.profit_units.sum() / len(g) * 100 if len(g) else 0
            rate = w / n * 100 if n else 0
            se = (np.sqrt(0.5238 * 0.4762 / n) * 100) if n else 0
            print(f"    {strat:28s} {w}-{l}-{p}  {rate:.1f}%  roi={roi:+.2f}%  "
                  f"(1 SD = {se:.1f} pts)")
    print("=" * 78)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    args = [int(a) for a in sys.argv[2:] if a.isdigit()]
    if cmd == "log":
        cmd_log(args[0], args[1])
    elif cmd == "close":
        cmd_close(args[0], args[1])
    elif cmd == "grade":
        cmd_grade(args[0])
    elif cmd == "report":
        cmd_report(args[0] if args else None)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
