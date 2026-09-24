"""Player attribution for ESPN play-by-play, and player EPA aggregates.

ESPN plays carry no athlete IDs, only text, in two formats:
  "Keelon Russell pass complete to Daniel Hill for 7 yds"            (full names)
  "No Huddle-Shotgun #7 K.Jackson pass complete short left to #4 I.Cisse ..."
The jersey format is resolved to full names through the game's box score
(team, jersey -> athlete). Players are keyed by ESPN athlete id when resolved.
"""
import re

import numpy as np
import pandas as pd

_CLOCK = re.compile(r"^\(\s*\d+:\d+\s*\)\s*")
_FORMATION = re.compile(r"^(?:No Huddle[- ]?)?(?:Shotgun|Pistol|Under Center|Wildcat|No Huddle)?[- ]*")
_PENALTY = re.compile(r"\s+[A-Z][\w&'\. ]*? Penalty,.*$")
_SUFFIX = r"(?: (?:Jr\.?|Sr\.?|II|III|IV|V))?"
_UP = r"[A-ZÀ-ÖØ-Þ]"                    # capital, incl. accented (K.Colón)
_LOW = r"[^\W\d_'\.\-]|['\.\-]"          # any letter or ' . -
_NAME = _UP + r"(?:" + _LOW + r")*(?: " + _UP + r"(?:" + _LOW + r")*)*?" + _SUFFIX
_ACTOR = re.compile(r"(?:#(\d+)\s+)?(" + _NAME + r")\s+(pass|sacked|run|rush|scramble|scrambles|kneel)")
_TARGET = re.compile(r"\bto\s+(?:#(\d+)\s+)?(" + _NAME + r")(?=\s+(?:for|caught|at|thrown|in|\(|[a-z])|,|\.|$)")


# scoring-summary format ESPN uses on touchdowns:
#   "Jaden Reddell 14 Yd pass from Gunner Stockton (Peyton Woodring Kick)"
#   "Damon Ferguson Jr. 13 Yd Run (Antonio Chadha Kick)"
_TD_PASS = re.compile(r"^(" + _NAME + r") \d+ Yd pass from (" + _NAME + r")(?=\s*\(|\s*$|,| for)")
_TD_RUN = re.compile(r"^(" + _NAME + r") \d+ Yd (?:Run|Rush)")


def parse(text):
    """-> (actor_jersey, actor_name, verb, target_jersey, target_name)."""
    t = _CLOCK.sub("", text or "")
    m = _TD_PASS.match(t)
    if m:
        return None, m.group(2).strip(), "pass", None, m.group(1).strip()
    m = _TD_RUN.match(t)
    if m:
        return None, m.group(1).strip(), "run", None, None
    t = _PENALTY.sub("", t)
    t = _FORMATION.sub("", t, count=1)
    m = _ACTOR.search(t)
    if not m:
        return None
    tj = tn = None
    if m.group(3) == "pass":
        m2 = _TARGET.search(t, m.end())
        if m2:
            tj, tn = m2.group(1), m2.group(2).strip()
    return m.group(1), m.group(2).strip(), m.group(3), tj, tn


OFFENSE_GROUPS = ("passing", "rushing", "receiving")


def roster(summary):
    """{team_id: {group: {"jersey": {num: (id, name)}}, "name": {norm: (id, name)}}}

    College teams reuse jersey numbers on offense and defense, so jersey lookups
    are kept PER STAT GROUP (a passer is looked up among passers first). Only
    offensive groups are indexed at all."""
    out = {}
    for t in (summary.get("boxscore") or {}).get("players", []):
        tid = str(t["team"]["id"])
        r = out.setdefault(tid, {"name": {}, **{g: {} for g in OFFENSE_GROUPS}})
        for g in t.get("statistics", []):
            grp = g.get("name")
            if grp not in OFFENSE_GROUPS:
                continue
            for a in g.get("athletes", []):
                ath = a.get("athlete") or {}
                aid, nm, jer = ath.get("id"), ath.get("displayName"), ath.get("jersey")
                if not aid or not nm:
                    continue
                if jer:
                    r[grp][str(jer)] = (str(aid), nm)
                r["name"][_norm(nm)] = (str(aid), nm)
                for k in _short_keys(nm):
                    r["name"].setdefault(k, (str(aid), nm))
    return out


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _last(nm):
    parts = [x for x in re.split(r"[\s\.]+", nm or "") if x]
    while len(parts) > 1 and parts[-1].lower() in _SUFFIXES:
        parts.pop()
    return _norm(parts[-1]) if parts else ""


def _short_keys(nm):
    """'Keelon Russell' -> 'krussell' (matches 'K.Russell')."""
    parts = [x for x in re.split(r"[\s\.]+", nm or "") if x]
    if len(parts) < 2:
        return []
    return [_norm(parts[0][:1] + _last(nm))]


def _norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


ROLE_GROUP = {"pass": "passing", "sacked": "passing", "run": "rushing", "rush": "rushing",
              "scramble": "rushing", "scrambles": "rushing", "kneel": "rushing", "target": "receiving"}


def resolve(ros, team_id, jersey, name, role=None):
    """-> (player_key, display_name). Falls back to the raw text name."""
    r = ros.get(str(team_id), {})
    want = _last(name)
    groups = [ROLE_GROUP.get(role)] if role in ROLE_GROUP else []
    groups += [g for g in OFFENSE_GROUPS if g not in groups]
    if jersey:
        for g in groups:
            hit = r.get(g, {}).get(jersey)
            if hit and (not want or _last(hit[1]) == want):
                return hit
    k = _norm(name)
    if k in r.get("name", {}):
        return r["name"][k]
    for k2 in _short_keys(name.replace(".", ". ") if name else name):
        if k2 in r.get("name", {}):
            return r["name"][k2]
    return (f"{team_id}:{k}", name)


def aggregate(plays, teams, games_played):
    """Season player tables from the per-play log.

    plays: DataFrame with team_id, kind, epa, success, yds, td, turnover,
           actor_id, actor, verb, target_id, target, complete.
    Returns {"passing": df, "rushing": df, "receiving": df}.
    """
    p = plays.copy()
    team_name = dict(zip(teams["team_id"], teams["team"]))
    logo = dict(zip(teams["team_id"], teams["logo"]))
    out = {}

    def finish(df, key_n):
        df["team"] = df["team_id"].map(team_name)
        df["logo"] = df["team_id"].map(logo)
        df["games"] = df["team_id"].map(games_played)
        return df[df["team"].notna()].sort_values(key_n, ascending=False)

    # passing: dropbacks = pass attempts + sacks, credited to the passer
    db = p[(p["kind"] == "pass") & p["actor_id"].notna()]
    if len(db):
        g = db.groupby(["actor_id", "team_id"])
        d = g.agg(player=("actor", "last"), dropbacks=("epa", "size"), epa=("epa", "sum"),
                  success=("success", "mean"), sacks=("verb", lambda v: (v == "sacked").sum()),
                  att=("verb", lambda v: (v == "pass").sum()), comp=("complete", "sum"),
                  pass_yds=("pass_yds", "sum"), td=("td", "sum"), ints=("interception", "sum")).reset_index()
        d["epa_per"] = d["epa"] / d["dropbacks"]
        d["comp_pct"] = d["comp"] / d["att"].where(d["att"] > 0)
        d["ypa"] = d["pass_yds"] / d["att"].where(d["att"] > 0)
        out["passing"] = finish(d, "dropbacks")
    # rushing: designed runs + scrambles
    ru = p[(p["kind"] == "rush") & p["actor_id"].notna()]
    if len(ru):
        g = ru.groupby(["actor_id", "team_id"])
        d = g.agg(player=("actor", "last"), carries=("epa", "size"), epa=("epa", "sum"),
                  success=("success", "mean"), yds=("yds", "sum"), td=("td", "sum"),
                  explosive=("yds", lambda v: (v >= 12).sum())).reset_index()
        d["epa_per"] = d["epa"] / d["carries"]
        d["ypc"] = d["yds"] / d["carries"]
        out["rushing"] = finish(d, "carries")
    # receiving: targets on pass attempts with a named target
    tg = p[(p["kind"] == "pass") & (p["verb"] == "pass") & p["target_id"].notna()]
    if len(tg):
        g = tg.groupby(["target_id", "team_id"])
        d = g.agg(player=("target", "last"), targets=("epa", "size"), epa=("epa", "sum"),
                  success=("success", "mean"), rec=("complete", "sum"), yds=("pass_yds", "sum"),
                  td=("td", "sum")).reset_index()
        d["epa_per"] = d["epa"] / d["targets"]
        d["catch_pct"] = d["rec"] / d["targets"]
        d["ypt"] = d["yds"] / d["targets"]
        out["receiving"] = finish(d, "targets")
    return out
