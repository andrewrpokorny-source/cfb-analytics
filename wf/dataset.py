"""Build a walk-forward-safe feature table.

THE CONTRACT
------------
For a game played in (season S, week W), a feature may only be derived from
information that existed before kickoff. Verified empirically on 2026-08-04:

  * /ratings/elo?year=S&week=W   is POST-week-W.
        corr(elo[W] - elo[W-1], week-W margin) = +0.824
    So a week-W game must use elo at week W-1. week=0 is the preseason rating.

  * /stats/season/advanced?startWeek=1&endWeek=W is INCLUSIVE of week W.
    So a week-W game must use endWeek = W-1. Week 1 games get no
    current-season stats at all (correctly NaN).

  * /ratings/srs?year=S and /ratings/sp?year=S are FULL-SEASON aggregates.
    They are never safe within season S. We use season S-1 only.

  * /talent and /player/returning are preseason-published and safe for S.

Anything added here must state which of those buckets it falls in.
Run `python -m wf.audit` after any change to this file.
"""
import numpy as np
import pandas as pd

from .cache import fetch

# Books we accept a line from, in rough order of trust.
VALID_BOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars", "ESPN Bet",
               "Bovada", "PointsBet", "BetRivers", "Unibet"]

# Advanced-stat paths we pull, as (column suffix, dotted path).
ADV_FIELDS = [
    ("off_ppa", "offense.ppa"),
    ("off_sr", "offense.successRate"),
    ("off_expl", "offense.explosiveness"),
    ("off_ppo", "offense.pointsPerOpportunity"),
    ("off_stuff", "offense.stuffRate"),
    ("def_ppa", "defense.ppa"),
    ("def_sr", "defense.successRate"),
    ("def_expl", "defense.explosiveness"),
    ("def_ppo", "defense.pointsPerOpportunity"),
    ("def_stuff", "defense.stuffRate"),
    ("plays", "offense.plays"),
]

# Which bucket each feature family comes from — asserted by wf/audit.py.
PROVENANCE = {
    "elo": "current season, week W-1 (pre-kickoff)",
    "adv": "current season, endWeek W-1 (pre-kickoff)",
    "prior": "season S-1 full-season aggregate (safe)",
    "pre": "preseason publication for season S (safe)",
    "market": "betting line (safe)",
    "sched": "schedule-derived (safe)",
}


def american_to_decimal(odds):
    """American odds -> decimal payout multiplier (profit per 1 unit staked)."""
    if not odds:
        return np.nan
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def decimal_to_american(dec):
    """Inverse of american_to_decimal."""
    if dec is None or np.isnan(dec) or dec <= 0:
        return np.nan
    return dec * 100.0 if dec >= 1 else -100.0 / dec


def american_median(odds_list):
    """Median of American odds, computed in decimal space where it is well defined."""
    dec = [american_to_decimal(o) for o in odds_list if o]
    dec = [x for x in dec if not np.isnan(x)]
    if not dec:
        return np.nan
    return float(decimal_to_american(float(np.median(dec))))


def american_best(odds_list):
    """Best available price for a bettor = highest decimal payout."""
    dec = [american_to_decimal(o) for o in odds_list if o]
    dec = [x for x in dec if not np.isnan(x)]
    if not dec:
        return np.nan
    return float(decimal_to_american(max(dec)))


def _dig(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return np.nan
        cur = cur.get(part)
    return cur if isinstance(cur, (int, float)) else np.nan


def _team_of(rec):
    return rec.get("team") or rec.get("school")


# ---------------------------------------------------------------- loaders


def _elo_by_week(year, max_week):
    """elo[w][team] — rating AFTER week w. Consumers must ask for w-1."""
    out = {}
    for w in range(0, max_week + 1):
        recs = fetch("/ratings/elo", {"year": year, "week": w})
        out[w] = {r["team"]: r["elo"] for r in recs if r.get("elo") is not None} if isinstance(recs, list) else {}
    return out


def _adv_by_endweek(year, max_week):
    """adv[w][team] — cumulative stats THROUGH week w. Consumers must ask for w-1."""
    out = {}
    for w in range(1, max_week + 1):
        recs = fetch("/stats/season/advanced",
                     {"year": year, "startWeek": 1, "endWeek": w})
        if not isinstance(recs, list):
            out[w] = {}
            continue
        out[w] = {
            r["team"]: {suf: _dig(r, path) for suf, path in ADV_FIELDS}
            for r in recs if r.get("team")
        }
    return out


def _prior_season_maps(year):
    """Full-season ratings from year-1. Safe because that season is closed."""
    py = year - 1
    srs = fetch("/ratings/srs", {"year": py})
    sp = fetch("/ratings/sp", {"year": py})
    elo = fetch("/ratings/elo", {"year": py})
    return {
        "prior_srs": {r["team"]: r.get("rating") for r in srs if isinstance(r, dict) and r.get("team")} if isinstance(srs, list) else {},
        "prior_sp": {r["team"]: r.get("rating") for r in sp if isinstance(r, dict) and r.get("team")} if isinstance(sp, list) else {},
        "prior_sp_off": {r["team"]: _dig(r, "offense.rating") for r in sp if isinstance(r, dict) and r.get("team")} if isinstance(sp, list) else {},
        "prior_sp_def": {r["team"]: _dig(r, "defense.rating") for r in sp if isinstance(r, dict) and r.get("team")} if isinstance(sp, list) else {},
        "prior_elo": {r["team"]: r.get("elo") for r in elo if isinstance(r, dict) and r.get("team")} if isinstance(elo, list) else {},
    }


def _preseason_maps(year):
    """Published before the season starts — safe for season `year`."""
    tal = fetch("/talent", {"year": year})
    ret = fetch("/player/returning", {"year": year})
    return {
        "talent": {_team_of(r): r.get("talent") for r in tal if isinstance(r, dict)} if isinstance(tal, list) else {},
        "ret_ppa": {r["team"]: r.get("percentPPA") for r in ret if isinstance(r, dict) and r.get("team")} if isinstance(ret, list) else {},
        "ret_usage": {r["team"]: r.get("usage") for r in ret if isinstance(r, dict) and r.get("team")} if isinstance(ret, list) else {},
    }


def _lines_map(year, season_type):
    """game_id -> consensus market line (median across accepted books)."""
    recs = fetch("/lines", {"year": year, "seasonType": season_type})
    out = {}
    if not isinstance(recs, list):
        return out
    for g in recs:
        books = [b for b in (g.get("lines") or []) if b.get("provider") in VALID_BOOKS]
        if not books:
            continue
        spreads = [b["spread"] for b in books if b.get("spread") is not None]
        totals = [b["overUnder"] for b in books if b.get("overUnder") is not None]
        opens = [b["spreadOpen"] for b in books if b.get("spreadOpen") is not None]
        h_ml = [b["homeMoneyline"] for b in books if b.get("homeMoneyline")]
        a_ml = [b["awayMoneyline"] for b in books if b.get("awayMoneyline")]
        out[str(g["id"])] = {
            "spread": float(np.median(spreads)) if spreads else np.nan,
            "over_under": float(np.median(totals)) if totals else np.nan,
            "spread_open": float(np.median(opens)) if opens else np.nan,
            # NEVER median American odds directly: the scale is discontinuous at
            # +/-100, so median(+180, -110) = 35, which is not a price at all.
            # Convert to decimal, aggregate there, convert back.
            "home_ml": american_median(h_ml),
            "away_ml": american_median(a_ml),
            "n_books": len(books),
        }
    return out


# ---------------------------------------------------------------- builder


def build_season(year, season_type="regular", verbose=True):
    """One row per completed game, with strictly pre-kickoff features."""
    games = fetch("/games", {"year": year, "seasonType": season_type})
    if not isinstance(games, list) or not games:
        return pd.DataFrame()

    norm = []
    for g in games:
        home = g.get("homeTeam") or g.get("home_team")
        away = g.get("awayTeam") or g.get("away_team")
        hp = g.get("homePoints") if g.get("homePoints") is not None else g.get("home_points")
        ap = g.get("awayPoints") if g.get("awayPoints") is not None else g.get("away_points")
        if not home or not away:
            continue
        norm.append({
            "game_id": str(g["id"]), "season": year, "week": g.get("week"),
            "start_date": g.get("startDate") or g.get("start_date"),
            "home_team": home, "away_team": away,
            "home_points": hp, "away_points": ap,
            "completed": bool(g.get("completed")),
            "neutral_site": bool(g.get("neutralSite") or g.get("neutral_site")),
            "home_conf": g.get("homeConference"), "away_conf": g.get("awayConference"),
            "home_div": g.get("homeClassification"), "away_div": g.get("awayClassification"),
        })

    gdf = pd.DataFrame(norm)
    if gdf.empty:
        return gdf
    gdf["week"] = pd.to_numeric(gdf["week"], errors="coerce")
    gdf = gdf.dropna(subset=["week"])
    gdf["week"] = gdf["week"].astype(int)
    max_week = int(gdf["week"].max())

    if verbose:
        print(f"   {year}: {len(gdf)} games, weeks 1-{max_week} — loading as-of features...")

    elo = _elo_by_week(year, max_week)
    adv = _adv_by_endweek(year, max_week)
    prior = _prior_season_maps(year)
    pre = _preseason_maps(year)
    lines = _lines_map(year, season_type)

    # Rest days: needs full schedule ordering, including not-yet-played games.
    gdf["_dt"] = pd.to_datetime(gdf["start_date"], errors="coerce", utc=True)
    last_played = {}
    for _, r in gdf.sort_values("_dt").iterrows():
        for t in (r["home_team"], r["away_team"]):
            last_played.setdefault(t, [])
            last_played[t].append(r["_dt"])
    rest_lookup = {}
    for t, dates in last_played.items():
        ds = sorted(dates)
        for i, d in enumerate(ds):
            rest_lookup[(t, d)] = (d - ds[i - 1]).days if i > 0 else np.nan

    rows = []
    for _, g in gdf.iterrows():
        w = g["week"]
        asof = w - 1                      # <-- the contract, in one line
        e = elo.get(asof, {})
        a = adv.get(asof, {}) if asof >= 1 else {}
        mk = lines.get(g["game_id"], {})

        row = {
            "game_id": g["game_id"], "season": year, "week": w,
            "season_type": season_type,
            "start_date": g["start_date"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "home_conf": g["home_conf"], "away_conf": g["away_conf"],
            "home_div": g["home_div"], "away_div": g["away_div"],
            "completed": g["completed"],
            "home_points": g["home_points"], "away_points": g["away_points"],
            "neutral_site": int(g["neutral_site"]),
            "asof_week": asof,
        }

        # market
        row.update({
            "spread": mk.get("spread", np.nan),
            "over_under": mk.get("over_under", np.nan),
            "spread_open": mk.get("spread_open", np.nan),
            "home_ml": mk.get("home_ml", np.nan),
            "away_ml": mk.get("away_ml", np.nan),
            "n_books": mk.get("n_books", 0),
        })

        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            row[f"{side}_elo"] = e.get(team, np.nan)
            ta = a.get(team, {})
            for suf, _ in ADV_FIELDS:
                row[f"{side}_{suf}"] = ta.get(suf, np.nan)
            row[f"{side}_prior_srs"] = prior["prior_srs"].get(team, np.nan)
            row[f"{side}_prior_sp"] = prior["prior_sp"].get(team, np.nan)
            row[f"{side}_prior_sp_off"] = prior["prior_sp_off"].get(team, np.nan)
            row[f"{side}_prior_sp_def"] = prior["prior_sp_def"].get(team, np.nan)
            row[f"{side}_prior_elo"] = prior["prior_elo"].get(team, np.nan)
            row[f"{side}_talent"] = pre["talent"].get(team, np.nan)
            row[f"{side}_ret_ppa"] = pre["ret_ppa"].get(team, np.nan)
            row[f"{side}_ret_usage"] = pre["ret_usage"].get(team, np.nan)
            row[f"{side}_rest"] = rest_lookup.get((team, g["_dt"]), np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- derived diffs (home minus away) ----
    def diff(name, col):
        df[name] = pd.to_numeric(df[f"home_{col}"], errors="coerce") - \
                   pd.to_numeric(df[f"away_{col}"], errors="coerce")

    diff("elo_diff", "elo")
    diff("prior_srs_diff", "prior_srs")
    diff("prior_sp_diff", "prior_sp")
    diff("prior_elo_diff", "prior_elo")
    diff("talent_diff", "talent")
    diff("ret_ppa_diff", "ret_ppa")
    diff("rest_diff", "rest")

    # net efficiency: my offense minus their defense
    for a_side, b_side in (("home", "away"), ("away", "home")):
        df[f"{a_side}_net_ppa"] = (pd.to_numeric(df[f"{a_side}_off_ppa"], errors="coerce")
                                   - pd.to_numeric(df[f"{b_side}_def_ppa"], errors="coerce"))
        df[f"{a_side}_net_sr"] = (pd.to_numeric(df[f"{a_side}_off_sr"], errors="coerce")
                                  - pd.to_numeric(df[f"{b_side}_def_sr"], errors="coerce"))
    df["net_ppa_diff"] = df["home_net_ppa"] - df["away_net_ppa"]
    df["net_sr_diff"] = df["home_net_sr"] - df["away_net_sr"]

    # line movement (market disagreement with itself)
    df["line_move"] = df["spread"] - df["spread_open"]

    # ---- targets (only meaningful for completed games) ----
    hp = pd.to_numeric(df["home_points"], errors="coerce")
    ap = pd.to_numeric(df["away_points"], errors="coerce")
    df["margin"] = hp - ap                       # home perspective
    df["total_points"] = hp + ap
    # CFBD `spread` is home-perspective, negative = home favored,
    # so the market's implied home margin is -spread.
    df["market_margin"] = -df["spread"]
    df["ats_diff"] = df["margin"] + df["spread"]  # >0 home covered
    df["home_cover"] = np.where(df["ats_diff"] > 0, 1,
                                np.where(df["ats_diff"] < 0, 0, np.nan))
    df["total_diff"] = df["total_points"] - df["over_under"]
    df["over"] = np.where(df["total_diff"] > 0, 1,
                          np.where(df["total_diff"] < 0, 0, np.nan))
    df["home_win"] = np.where(hp > ap, 1, np.where(hp < ap, 0, np.nan))

    return df


def build(years, season_type="regular", fbs_only=True, need_line=True, verbose=True):
    """Assemble multiple seasons into one modelling table."""
    frames = [build_season(y, season_type, verbose) for y in years]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    if fbs_only:
        df = df[(df["home_div"] == "fbs") & (df["away_div"] == "fbs")]
    if need_line:
        df = df.dropna(subset=["spread", "over_under"])

    df = df.sort_values(["season", "week", "start_date"]).reset_index(drop=True)
    if verbose:
        print(f"   -> {len(df)} games with lines across {df['season'].nunique()} seasons")
    return df


# Feature groups the models consume.
MARKET = ["spread", "over_under"]
FORM = ["elo_diff", "net_ppa_diff", "net_sr_diff",
        "home_off_ppa", "home_def_ppa", "away_off_ppa", "away_def_ppa",
        "home_off_sr", "home_def_sr", "away_off_sr", "away_def_sr",
        "home_off_expl", "away_off_expl", "home_off_ppo", "away_off_ppo"]
PRIOR = ["prior_srs_diff", "prior_sp_diff", "prior_elo_diff",
         "home_prior_sp_off", "home_prior_sp_def",
         "away_prior_sp_off", "away_prior_sp_def"]
PRESEASON = ["talent_diff", "ret_ppa_diff", "home_talent", "away_talent"]
SITUATION = ["neutral_site", "rest_diff", "week"]

ALL_FEATURES = MARKET + FORM + PRIOR + PRESEASON + SITUATION
NO_MARKET = FORM + PRIOR + PRESEASON + SITUATION
