"""ESPN data access for the stats site. Keyless.

Completed-game summaries never change, so they are cached on disk forever
(espn_cache/, gitignored). Schedules and standings are always refetched.
Nothing here imports wf.config, so it runs without any API key.
"""
import json
import os
import time

import requests

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
STANDINGS = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "espn_cache")


def _get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"ESPN request failed: {url} {params}")


def fbs_teams(season):
    """FBS teams with conference, from ESPN standings (recurses into divisions)."""
    d = _get(STANDINGS, {"group": 80, "season": season})
    out = []

    def walk(node, conf_name, conf_abbr):
        name = node.get("name") or conf_name
        abbr = node.get("abbreviation") or conf_abbr
        top_conf = conf_name or name
        top_abbr = conf_abbr or abbr
        for e in (node.get("standings") or {}).get("entries", []):
            t = e["team"]
            stats = {s.get("name"): s.get("displayValue") for s in e.get("stats", [])}
            out.append({
                "team_id": str(t["id"]),
                "team": t.get("location") or t.get("shortDisplayName"),
                "display_name": t.get("displayName"),
                "abbr": t.get("abbreviation"),
                "logo": (t.get("logos") or [{}])[0].get("href"),
                "conference": top_conf, "conf_abbr": top_abbr,
                "overall": stats.get("overall"),
                "conf_record": None,
            })
        for c in node.get("children", []):
            walk(c, top_conf, top_abbr)

    for conf in d.get("children", []):
        walk(conf, conf.get("name"), conf.get("abbreviation"))
    # ESPN's standings for a conference have a separate "vs. Conf." record
    seen, uniq = set(), []
    for t in out:
        if t["team_id"] not in seen:
            seen.add(t["team_id"])
            uniq.append(t)
    return uniq


def team_directory():
    """All teams ESPN knows (logos, colours) — covers FCS opponents too."""
    d = _get(f"{SITE}/teams", {"limit": 1000})
    out = {}
    for x in d["sports"][0]["leagues"][0]["teams"]:
        t = x["team"]
        out[str(t["id"])] = {
            "team_id": str(t["id"]),
            "team": t.get("location") or t.get("shortDisplayName"),
            "display_name": t.get("displayName"),
            "abbr": t.get("abbreviation"),
            "color": "#" + (t.get("color") or "666666"),
            "alt_color": "#" + (t.get("alternateColor") or "cccccc"),
            "logo": (t.get("logos") or [{}])[0].get("href"),
        }
    return out


def scoreboard(season, week, season_type=2):
    return _get(f"{SITE}/scoreboard",
                {"groups": 80, "limit": 400, "dates": season, "week": week, "seasontype": season_type})


def current_week():
    d = _get(f"{SITE}/scoreboard", {"groups": 80, "limit": 400})
    return d.get("season", {}).get("year"), (d.get("week") or {}).get("number"), d


def summary(event_id, completed):
    """Game summary (box score + play-by-play). Cached to disk once final."""
    path = os.path.join(CACHE, f"summary_{event_id}.json")
    if completed and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    s = _get(f"{SITE}/summary", {"event": event_id})
    if completed:
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w") as f:
            json.dump(s, f)
    return s
