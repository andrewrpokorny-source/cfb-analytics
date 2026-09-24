"""Live odds feed, normalised across providers.

Two backends:
  * theoddsapi — the real one. 10-15 books incl. Pinnacle. Needs ODDS_API_KEY.
  * cfbd      — fallback. Only 3 books, so shopping value is capped (see
                wf/README_SHOPPING.md). Fine for wiring things up.

Everything downstream consumes `Quote` objects, so swapping backends changes
nothing else.
"""
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import requests

from .cache import fetch as cfbd_fetch
from .dataset import clean_books

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_ncaaf"


@dataclass
class Quote:
    """One book's price on one side of one market."""
    game_key: str          # "AwayTeam @ HomeTeam"
    home_team: str
    away_team: str
    commence: str          # ISO8601
    book: str
    market: str            # 'spread' | 'total' | 'ml'
    side: str              # team name, or 'OVER' / 'UNDER'
    line: float | None     # points; None for moneyline
    price: int             # American odds
    fetched_at: str
    price_assumed: bool = False   # True when the feed carries no price (CFBD)

    def to_dict(self):
        return asdict(self)


def _now():
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------ name matching

_SUFFIXES = [
    "Fighting Irish", "Crimson Tide", "Golden Gophers", "Nittany Lions",
    "Scarlet Knights", "Yellow Jackets", "Demon Deacons", "Mountaineers",
    "Cornhuskers", "Bulldogs", "Wildcats", "Tigers", "Cardinals", "Aggies",
    "Cowboys", "Buckeyes", "Wolverines", "Sooners", "Longhorns", "Volunteers",
    "Gamecocks", "Razorbacks", "Commodores", "Rebels", "Hurricanes",
    "Seminoles", "Gators", "Trojans", "Bruins", "Ducks", "Beavers", "Huskies",
    "Cougars", "Utes", "Buffaloes", "Sun Devils", "Jayhawks", "Cyclones",
    "Horned Frogs", "Red Raiders", "Bears", "Mustangs", "Golden Hurricane",
    "Knights", "Owls", "Bulls", "Blazers", "Chanticleers", "Panthers",
    "Eagles", "Falcons", "Rams", "Broncos", "Rainbow Warriors", "Spartans",
    "Hawkeyes", "Badgers", "Boilermakers", "Hoosiers", "Terrapins",
    "Fighting Illini", "Wolfpack", "Tar Heels", "Blue Devils", "Cavaliers",
    "Hokies", "Orange", "Golden Bears", "Cardinal", "Sun Belt", "Jaguars",
    "Warhawks", "Ragin' Cajuns", "Red Wolves", "Trojans", "Bobcats",
    "Thundering Herd", "Green Wave", "Pirates", "Minutemen", "Huskies",
]


def strip_mascot(name):
    """'Georgia Bulldogs' -> 'Georgia'. The Odds API sends full names; CFBD doesn't."""
    if not name:
        return name
    for suf in sorted(_SUFFIXES, key=len, reverse=True):
        if name.endswith(" " + suf):
            return name[: -(len(suf) + 1)].strip()
    return name.strip()


def build_alias_map(year):
    """CFBD school names + common variants, for joining feed names to CFBD."""
    teams = cfbd_fetch("/teams/fbs", {"year": year}) or []
    alias = {}
    for t in teams:
        school = t.get("school")
        if not school:
            continue
        for key in filter(None, [school, t.get("alt_name1"), t.get("alt_name2"),
                                 t.get("alt_name3"), t.get("abbreviation")]):
            alias[key.lower()] = school
        mascot = t.get("mascot")
        if mascot:
            alias[f"{school} {mascot}".lower()] = school
    return alias


def resolve_team(name, alias):
    if not name:
        return name
    n = name.strip()
    if n.lower() in alias:
        return alias[n.lower()]
    s = strip_mascot(n)
    if s.lower() in alias:
        return alias[s.lower()]
    return s


# ------------------------------------------------------------ the odds api


def fetch_theoddsapi(api_key=None, markets=("spreads", "totals", "h2h"),
                     regions="us", year=None, verbose=True):
    """Live odds from The Odds API. Returns [] and explains itself if unconfigured."""
    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        if verbose:
            print("   ODDS_API_KEY not set — see wf/README_SHOPPING.md for setup.")
        return []

    url = f"{ODDS_API_BASE}/sports/{SPORT}/odds"
    params = {"apiKey": api_key, "regions": regions,
              "markets": ",".join(markets), "oddsFormat": "american"}
    try:
        res = requests.get(url, params=params, timeout=45)
    except requests.RequestException as e:
        print(f"   ! odds api request failed: {e}")
        return []

    if res.status_code != 200:
        print(f"   ! odds api HTTP {res.status_code}: {res.text[:200]}")
        return []

    if verbose:
        rem = res.headers.get("x-requests-remaining")
        used = res.headers.get("x-requests-used")
        if rem is not None:
            print(f"   odds api quota: {used} used, {rem} remaining")

    alias = build_alias_map(year or datetime.now(timezone.utc).year)
    quotes = parse_theoddsapi(res.json(), alias)
    if verbose:
        print(f"   fetched {len(quotes)} quotes across "
              f"{len({q.game_key for q in quotes})} games, "
              f"{len({q.book for q in quotes})} books")
    return quotes


def parse_theoddsapi(payload, alias=None):
    """Parse a v4 /odds response into Quotes. Separated from the HTTP call so
    it can be tested against a fixture without an API key."""
    alias = alias or {}
    now = _now()
    quotes = []
    for ev in payload:
        home = resolve_team(ev.get("home_team"), alias)
        away = resolve_team(ev.get("away_team"), alias)
        gk = f"{away} @ {home}"
        commence = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = (bk.get("title") or bk.get("key") or "").strip()
            for mk in bk.get("markets", []):
                kind = {"spreads": "spread", "totals": "total", "h2h": "ml"}.get(mk.get("key"))
                if not kind:
                    continue
                for oc in mk.get("outcomes", []):
                    nm = oc.get("name")
                    if kind == "total":
                        side = nm.upper() if nm else None
                        if side not in ("OVER", "UNDER"):
                            continue
                    else:
                        side = resolve_team(nm, alias)
                    price = oc.get("price")
                    if price is None:
                        continue
                    quotes.append(Quote(gk, home, away, commence, book, kind,
                                        side, oc.get("point"), int(price), now))
    return quotes


# ------------------------------------------------------------ cfbd fallback


def fetch_cfbd(year, week, season_type="regular", verbose=True):
    """Fallback feed. Only 3 books and no per-side spread pricing."""
    lines = cfbd_fetch("/lines", {"year": year, "week": week, "seasonType": season_type}) or []
    now = _now()
    quotes = []
    for g in lines:
        home, away = g.get("homeTeam"), g.get("awayTeam")
        if not home or not away:
            continue
        gk = f"{away} @ {home}"
        commence = g.get("startDate")
        # dedupe "DraftKings"/"Draft Kings" and drop home/away-swapped rows
        for b in clean_books(g.get("lines") or []):
            book = b.get("provider")
            sp, ou = b.get("spread"), b.get("overUnder")
            # CFBD carries NO spread or totals price. -110 is a placeholder so
            # the lines can be compared, and every such quote is flagged
            # price_assumed=True so nothing downstream mistakes it for a price.
            if sp is not None:
                quotes.append(Quote(gk, home, away, commence, book, "spread", home, float(sp), -110, now, True))
                quotes.append(Quote(gk, home, away, commence, book, "spread", away, -float(sp), -110, now, True))
            if ou is not None:
                quotes.append(Quote(gk, home, away, commence, book, "total", "OVER", float(ou), -110, now, True))
                quotes.append(Quote(gk, home, away, commence, book, "total", "UNDER", float(ou), -110, now, True))
            if b.get("homeMoneyline"):
                quotes.append(Quote(gk, home, away, commence, book, "ml", home, None, int(b["homeMoneyline"]), now))
            if b.get("awayMoneyline"):
                quotes.append(Quote(gk, home, away, commence, book, "ml", away, None, int(b["awayMoneyline"]), now))
    if verbose:
        print(f"   [cfbd fallback] {len(quotes)} quotes, "
              f"{len({q.book for q in quotes})} books — shopping value is capped here")
    return quotes


def get_quotes(year, week=None, prefer="theoddsapi", verbose=True):
    """Preferred feed, falling back to CFBD."""
    if prefer == "theoddsapi":
        q = fetch_theoddsapi(year=year, verbose=verbose)
        if q:
            return q
        if verbose:
            print("   falling back to CFBD lines...")
    if week is None:
        raise ValueError("CFBD fallback needs a week")
    return fetch_cfbd(year, week, verbose=verbose)


def save_quotes(quotes, path):
    with open(path, "w") as f:
        json.dump([q.to_dict() for q in quotes], f, indent=1)


def load_quotes(path):
    with open(path) as f:
        return [Quote(**{**{'price_assumed': False}, **d}) for d in json.load(f)]
