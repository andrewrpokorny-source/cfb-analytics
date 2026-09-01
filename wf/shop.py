"""Line shopping engine.

Naively "take the biggest number" is wrong: a book offering +7 at -130 can be
worse than +6.5 at -105. So every offer is priced against the no-vig consensus
using the empirical CFB margin distribution (wf/margin_dist.json, built from
5,605 games), and the best offer is the one with the highest expected value.

Measured ceiling on CFBD's 3 books: +1.00 ROI pts on spreads, +2.9-4.0 on
moneylines. A real 10-15 book feed should beat that — that is the point of
wiring The Odds API in.
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

_DIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "margin_dist.json")
with open(_DIST_PATH) as f:
    _DIST = json.load(f)

KEY_NUMBERS = [3, 7, 10, 14, 6, 4, 17, 21]


# ------------------------------------------------------------ odds helpers


def to_decimal(american):
    """American odds -> profit multiplier per 1 unit staked."""
    if american is None:
        return np.nan
    return american / 100.0 if american > 0 else 100.0 / abs(american)


def to_implied(american):
    """American odds -> implied probability (includes vig)."""
    if american is None:
        return np.nan
    return 100.0 / (american + 100.0) if american > 0 else abs(american) / (abs(american) + 100.0)


def to_american(prob):
    """Probability -> fair American odds (no vig)."""
    if prob is None or not np.isfinite(prob) or not (0 < prob < 1):
        return np.nan
    return -100.0 * prob / (1 - prob) if prob >= 0.5 else 100.0 * (1 - prob) / prob


def devig_pair(p1, p2):
    """Remove the book's margin from a two-way market."""
    s = p1 + p2
    if not np.isfinite(s) or s <= 0:
        return np.nan, np.nan
    return p1 / s, p2 / s


def sf(residual_kind, x):
    """P(residual > x) from the empirical distribution."""
    d = _DIST[residual_kind]
    return float(np.interp(x, d["grid"], d["sf"]))


def push_prob(residual_kind, x):
    """P(residual == x) — non-zero only at whole numbers, but material there.
    Ignoring it treats every push as a loss and understates EV on integer lines."""
    d = _DIST[residual_kind]
    if abs(x - round(x)) > 1e-9:
        return 0.0
    try:
        return float(d["pmf"][d["grid"].index(round(float(x), 1))])
    except (ValueError, KeyError):
        return 0.0


def cover_prob(kind, fair_line, offered_line):
    """P(this side wins) given the consensus fair line and the line on offer.

    kind='spread': lines are from the bettor's side, fair_line is consensus.
    Residual e = actual_margin + fair_line has mean ~0, sd ~15.45. The side
    wins if actual_margin + offered_line > 0, i.e. e > fair_line - offered_line.
    """
    return sf(kind, fair_line - offered_line)


# ------------------------------------------------------------ consensus


def consensus(quotes):
    """Fair (no-vig) line and probability per game/market/side."""
    by = defaultdict(list)
    for q in quotes:
        by[(q.game_key, q.market)].append(q)

    fair = {}
    for (gk, market), qs in by.items():
        sides = sorted({q.side for q in qs})
        if market == "ml":
            if len(sides) != 2:
                continue
            a, b = sides
            pa = np.median([to_implied(q.price) for q in qs if q.side == a])
            pb = np.median([to_implied(q.price) for q in qs if q.side == b])
            fa, fb = devig_pair(pa, pb)
            fair[(gk, market, a)] = {"fair_prob": fa, "fair_line": None}
            fair[(gk, market, b)] = {"fair_prob": fb, "fair_line": None}
        else:
            for s in sides:
                pts = [q.line for q in qs if q.side == s and q.line is not None]
                if not pts:
                    continue
                fair[(gk, market, s)] = {"fair_line": float(np.median(pts)),
                                         "fair_prob": None}
    return fair


def price_offer(q, fair):
    """Expected value of one offer, in units per 1 staked."""
    f = fair.get((q.game_key, q.market, q.side))
    if not f:
        return None
    dec = to_decimal(q.price)
    if not np.isfinite(dec):
        return None

    p_push = 0.0

    if q.market == "ml":
        p = f["fair_prob"]

    elif q.market == "spread":
        fair_line, L = f["fair_line"], q.line
        if fair_line is None or L is None:
            return None
        # Works for both sides: the spread residual is symmetric (mean -0.014),
        # so the away side's mirror probability comes out right by symmetry.
        p = cover_prob("spread", fair_line, L)
        p_push = push_prob("spread", fair_line - L)

    else:  # totals — residual mean is +0.368, so direction must be explicit
        fair_line, L = f["fair_line"], q.line
        if fair_line is None or L is None:
            return None
        p_over = sf("total", L - fair_line)      # actual total exceeds the offered line
        p = p_over if q.side == "OVER" else 1.0 - p_over
        p_push = push_prob("total", L - fair_line)
        p -= p_push if q.side == "UNDER" else 0.0   # 1-sf already includes the mass

    if p is None or not np.isfinite(p):
        return None
    # A push returns the stake, so it is neither a win nor a loss.
    p_loss = max(0.0, 1.0 - p - p_push)
    return p * dec - p_loss


def crosses_key_number(fair_line, offered_line):
    """Did shopping move the number across a key number? That is where value is."""
    if fair_line is None or offered_line is None:
        return []
    lo, hi = sorted([abs(fair_line), abs(offered_line)])
    return [k for k in KEY_NUMBERS if lo < k <= hi]


def best_offers(quotes, min_edge=0.0):
    """Best available offer per game/market/side, with EV and shopping gain."""
    fair = consensus(quotes)
    by = defaultdict(list)
    for q in quotes:
        by[(q.game_key, q.market, q.side)].append(q)

    rows = []
    for key, qs in by.items():
        gk, market, side = key
        priced = [(q, price_offer(q, fair)) for q in qs]
        priced = [(q, ev) for q, ev in priced if ev is not None]
        if not priced:
            continue
        priced.sort(key=lambda x: -x[1])
        best_q, best_ev = priced[0]
        median_ev = float(np.median([ev for _, ev in priced]))
        f = fair.get(key, {})
        rows.append({
            "game": gk, "home": best_q.home_team, "away": best_q.away_team,
            "commence": best_q.commence, "market": market, "side": side,
            "best_book": best_q.book, "best_line": best_q.line, "best_price": best_q.price,
            "fair_line": f.get("fair_line"), "fair_prob": f.get("fair_prob"),
            "n_books": len(priced),
            # With few books the "fair" line is barely more than the best book's
            # own number, so EV is mostly an artifact. Flag it rather than imply
            # precision we do not have.
            "consensus_ok": len(priced) >= 4,
            "ev_best": best_ev, "ev_median": median_ev,
            "shop_gain": best_ev - median_ev,
            "key_crossed": ",".join(str(k) for k in
                                    crosses_key_number(f.get("fair_line"), best_q.line)) or "",
            "all_books": "; ".join(
                f"{q.book} {q.line if q.line is not None else ''}{q.price:+d}"
                for q, _ in priced),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Both sides of a two-way market showing +EV is either a genuine middle
    # (books far enough apart that both can win) or, more often, a consensus
    # too thin to trust. Either way the caller needs to know.
    two_way = df[df["market"].isin(["spread", "total"])]
    pos = two_way[two_way["ev_best"] > 0].groupby(["game", "market"]).size()
    both = {k for k, v in pos.items() if v >= 2}
    df["both_sides_pos"] = [(g, m) in both for g, m in zip(df["game"], df["market"])]
    df["middle_pts"] = np.where(
        df["both_sides_pos"] & (df["market"] == "spread"),
        (df["best_line"].abs() - df["fair_line"].abs()).abs() * 2, np.nan)

    return df[df["ev_best"] >= min_edge].sort_values("ev_best", ascending=False).reset_index(drop=True)


def report(quotes, top=25, min_edge=0.0):
    df = best_offers(quotes, min_edge=min_edge)
    if df.empty:
        print("no priceable offers")
        return df

    nb = df["n_books"].mean()
    print(f"\n{len(df)} game/market/side combinations, mean {nb:.1f} books each")
    print(f"mean shopping gain: {df['shop_gain'].mean()*100:+.2f}% EV "
          f"(best vs median book)")
    crossed = (df["key_crossed"] != "").sum()
    print(f"key-number crossings available: {crossed} ({crossed/len(df)*100:.1f}%)")

    if nb < 4:
        print(f"\n  *** WARNING: only {nb:.1f} books per market. The 'fair' line is then")
        print(f"      little more than the best book's own number, so EV is inflated by")
        print(f"      construction. Treat these figures as unusable for sizing — set")
        print(f"      ODDS_API_KEY to get a real 10-15 book consensus. ***")
    nbad = int(df["both_sides_pos"].sum())
    if nbad:
        print(f"\n  {nbad} rows show +EV on BOTH sides of the same market — genuine middles")
        print(f"      if the books really are that far apart, thin consensus otherwise.")

    show = df.head(top)[["game", "market", "side", "best_book", "best_line",
                         "best_price", "fair_line", "n_books", "ev_best",
                         "shop_gain", "key_crossed", "both_sides_pos"]].copy()
    show["ev_best"] = show["ev_best"].map(lambda x: f"{x*100:+.2f}%")
    show["shop_gain"] = show["shop_gain"].map(lambda x: f"{x*100:+.2f}%")
    print("\n" + show.to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    from . import odds
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    q = odds.get_quotes(year, week)
    report(q)
