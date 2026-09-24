"""ESPN public scoreboard: schedule, live DraftKings lines WITH prices, results.

Keyless and free. Deliberately imports nothing from the rest of wf/ (which
needs a CFBD key at import time), so the Streamlit app can use it directly.

What it gives that CFBD does not:
  * real prices on spreads and totals (CFBD has none — see price_assumed)
  * current AND opening numbers, so line movement / CLV are measurable
  * results that keep flowing when the CFBD monthly quota is exhausted
What it does not give: more than one book. Middles and shopping still need
a multi-book feed (ODDS_API_KEY).
"""
import re
from datetime import datetime, timedelta, timezone

import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
FBS = 80  # ESPN group id for FBS
FCS = 81  # games vs FCS teams are outside every backtest (FBS-vs-FBS only)


def _num(x):
    """'+2.5' / 'o50.5' / 'u50.5' / '-108' / 'EVEN' -> float, or None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    if s in ("even", "ev", "pk", "pick"):
        return 100.0 if s.startswith("ev") else 0.0
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _get(params, timeout=30, group=FBS):
    r = requests.get(SCOREBOARD, params={"groups": group, "limit": 400, **params}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _fcs_team_ids(params):
    """ESPN's FBS board includes FBS-vs-FCS games; collect the FCS teams so
    they can be flagged. Returns empty set if the lookup fails."""
    try:
        d = _get(params, group=FCS)
    except Exception:
        return set()
    return {x["team"]["id"] for e in d.get("events", [])
            for x in e["competitions"][0]["competitors"] if x.get("team", {}).get("id")}


def _tag_fbs(rows, fcs_ids):
    for r in rows:
        r["fbs_vs_fbs"] = not ({r.pop("_home_id", None), r.pop("_away_id", None)} & fcs_ids)
    return rows


def team_name(comp):
    """School name without mascot ('Coastal Carolina'), close to CFBD naming."""
    t = comp.get("team", {})
    return t.get("location") or t.get("shortDisplayName") or t.get("displayName")


def normalise(name):
    """Loose key for matching team names across sources."""
    s = (name or "").lower().replace("&", "and")
    s = re.sub(r"\bst\.?\b", "state", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _side(odds, market, side, which):
    node = ((odds.get(market) or {}).get(side) or {}).get(which) or {}
    return _num(node.get("line")), _num(node.get("odds"))


def lines_from_odds(odds):
    """ESPN odds/pickcenter item -> flat line dict (home-perspective spread).

    Same structure on the live scoreboard and in a game summary's pickcenter,
    which is where closing lines survive after the game is final."""
    out = {"book": (odds.get("provider") or {}).get("name")}
    for which in ("close", "open"):          # ESPN calls the current number "close"
        tag = "" if which == "close" else "_open"
        hl, hp = _side(odds, "pointSpread", "home", which)
        _, ap = _side(odds, "pointSpread", "away", which)
        ol, op = _side(odds, "total", "over", which)
        ul, up = _side(odds, "total", "under", which)
        _, hml = _side(odds, "moneyline", "home", which)
        _, aml = _side(odds, "moneyline", "away", which)
        out.update({f"spread{tag}": hl, f"home_spread_price{tag}": hp,
                    f"away_spread_price{tag}": ap,
                    f"total{tag}": ol if ol is not None else ul,
                    f"over_price{tag}": op, f"under_price{tag}": up,
                    f"home_ml{tag}": hml, f"away_ml{tag}": aml})
    if out.get("spread") is None:
        out["spread"] = _num(odds.get("spread"))
    if out.get("total") is None:
        out["total"] = _num(odds.get("overUnder"))
    return out


def parse_event(ev):
    """One ESPN event -> flat dict (lines are home-perspective like CFBD)."""
    c = ev["competitions"][0]
    comps = {x["homeAway"]: x for x in c["competitors"]}
    home, away = comps.get("home", {}), comps.get("away", {})
    st = c.get("status", {}).get("type", {})
    row = {
        "event_id": ev.get("id"),
        "commence": ev.get("date"),
        "home": team_name(home), "away": team_name(away),
        "home_rank": (home.get("curatedRank") or {}).get("current"),
        "away_rank": (away.get("curatedRank") or {}).get("current"),
        "state": st.get("state"),            # pre / in / post
        "completed": bool(st.get("completed")),
        "status": st.get("shortDetail") or st.get("description"),
        "home_score": _num(home.get("score")), "away_score": _num(away.get("score")),
        "neutral": bool(c.get("neutralSite")),
        "venue": (c.get("venue") or {}).get("fullName"),
        "tv": ", ".join(b.get("names", [""])[0] for b in c.get("broadcasts", []) if b.get("names")),
        "book": None,
        "_home_id": home.get("team", {}).get("id"), "_away_id": away.get("team", {}).get("id"),
    }
    row["game"] = f"{row['away']} @ {row['home']}"
    odds = (c.get("odds") or [None])[0]
    if odds:
        row.update(lines_from_odds(odds))
    return row


def current_week():
    """The week ESPN is currently showing (the upcoming slate mid-week)."""
    d = _get({})
    yr, wk = d.get("season", {}).get("year"), (d.get("week") or {}).get("number")
    rows = [parse_event(e) for e in d.get("events", [])]
    return yr, wk, _tag_fbs(rows, _fcs_team_ids({"dates": yr, "week": wk}))


def week(season, week_no, season_type=2):
    params = {"dates": season, "week": week_no, "seasontype": season_type}
    d = _get(params)
    return _tag_fbs([parse_event(e) for e in d.get("events", [])], _fcs_team_ids(params))


def on_date(day):
    """All FBS events on a calendar date (UTC date string 'YYYY-MM-DD' or datetime)."""
    if isinstance(day, datetime):
        day = day.strftime("%Y%m%d")
    d = _get({"dates": str(day).replace("-", "")[:8]})
    return [parse_event(e) for e in d.get("events", [])]   # ids unused here


def find_result(game, commence):
    """Final score for a ledger game ('Away @ Home'), searched around its kickoff.

    ESPN files games by US-Eastern date, so a 7:30pm ET kickoff has a UTC date one
    day later; check both neighbouring days.
    """
    try:
        away, home = [normalise(x) for x in game.split(" @ ")]
    except ValueError:
        return None
    ts = commence if isinstance(commence, datetime) else datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    for delta in (0, -1, 1):
        for r in on_date(ts + timedelta(days=delta)):
            if normalise(r["home"]) == home and normalise(r["away"]) == away and r["completed"]:
                return r
    return None
