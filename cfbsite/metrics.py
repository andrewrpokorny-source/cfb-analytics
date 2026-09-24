"""Turn one ESPN game summary into per-team offense/defense stat lines.

Definitions (standard in college football analytics):

* Scrimmage play: a run or a pass dropback (incl. sacks and interceptions).
  Kickoffs, punts, field goals, penalties, timeouts are excluded.
* Garbage time: plays are dropped when the margin at the snap exceeds
  38 (Q2), 28 (Q3) or 22 (Q4). Rates are computed on non-garbage plays.
* Success: gains >= 50% of the distance on 1st down, >= 70% on 2nd,
  100% on 3rd/4th. Touchdowns succeed; turnovers fail.
* Explosive: a scrimmage play gaining 20+ yards.
* Havoc (defense): sacks + tackles for loss + takeaways, per defensive play.
* Points per drive: TD = 7, FG = 3, over drives excluding end-of-half kneels.
"""
import re

RUSH = {"Rush", "Rushing Touchdown"}
PASS = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack",
        "Interception", "Pass Interception Return", "Interception Return Touchdown",
        "Pass Interception Return Touchdown"}
FUMBLE = {"Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble"}
GARBAGE = {2: 38, 3: 28, 4: 22}
INT_TYPES = {"Interception", "Pass Interception Return", "Interception Return Touchdown",
             "Pass Interception Return Touchdown"}
# ESPN's per-play isTurnover flag is unreliable (most interceptions are False,
# lost fumbles appear as "Fumble Recovery (Own)"). The drive result is reliable.
TO_DRIVE_RESULTS = {"interception", "fumble", "interception return touchdown",
                    "fumble return touchdown", "interception touchdown", "fumble touchdown"}

_PASS_TXT = re.compile(r"\b(pass|sacked|scramble)\b", re.I)
_RUN_TXT = re.compile(r"\b(rush|run)\b", re.I)
_ST_TXT = re.compile(r"\b(kickoff|punt|field goal)\b", re.I)


def play_kind(p):
    t = p.get("type", {}).get("text", "")
    if t in RUSH:
        return "rush"
    if t in PASS:
        return "pass"
    if t in FUMBLE:
        txt = p.get("text", "")
        if _ST_TXT.search(txt):
            return None
        if _PASS_TXT.search(txt):
            return "pass"
        if _RUN_TXT.search(txt):
            return "rush"
    return None


def is_success(down, dist, yds, td, turnover):
    if turnover:
        return False
    if td:
        return True
    if not down or dist is None:
        return None
    need = {1: 0.5, 2: 0.7}.get(down, 1.0) * dist
    return yds >= need


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def game_rows(summary):
    """Two dicts (home, away) of raw counts for one completed game, or []."""
    hdr = summary.get("header", {}).get("competitions", [{}])[0]
    comps = {c["homeAway"]: c for c in hdr.get("competitors", [])}
    if "home" not in comps or "away" not in comps:
        return []
    ids = {"home": str(comps["home"]["team"]["id"]), "away": str(comps["away"]["team"]["id"])}
    side_of = {v: k for k, v in ids.items()}
    score = {k: _num(comps[k].get("score"), 0) for k in ("home", "away")}

    blank = lambda: {k: 0.0 for k in (
        "plays", "yards", "success", "succ_n", "explosive", "rush", "rush_yds", "pass",
        "pass_yds", "sacks", "tfl", "turnovers", "third_att", "third_conv",
        "drives", "drive_pts", "rz_trips", "rz_td", "td", "fg", "plays_all")}
    off = {"home": blank(), "away": blank()}

    home_sc = away_sc = 0.0
    drives = summary.get("drives", {}).get("previous", [])
    for d in drives:
        dteam = str((d.get("team") or {}).get("id", ""))
        dside = side_of.get(dteam)
        reached_rz = False
        dres = (d.get("displayResult") or d.get("result") or "").lower()
        scrim_idx = [i for i, p in enumerate(d.get("plays", [])) if play_kind(p)]
        to_idx = scrim_idx[-1] if (dres in TO_DRIVE_RESULTS and scrim_idx) else None
        for i, p in enumerate(d.get("plays", [])):
            period = (p.get("period") or {}).get("number", 1)
            start = p.get("start") or {}
            oside = side_of.get(str((start.get("team") or {}).get("id", "")), dside)
            margin = abs(home_sc - away_sc)
            garbage = margin > GARBAGE.get(period, 999)
            kind = play_kind(p)
            ytg = start.get("yardsToEndzone")
            if kind and oside and ytg is not None and ytg <= 20:
                reached_rz = True
            if kind and oside:
                o = off[oside]
                o["plays_all"] += 1
                if not garbage:
                    yds = _num(p.get("statYardage"), 0.0)
                    td = "Touchdown" in p.get("type", {}).get("text", "") and not p.get("isTurnover")
                    ptype = p.get("type", {}).get("text", "")
                    to = (i == to_idx) or ptype in INT_TYPES or ptype == "Fumble Recovery (Opponent)"
                    down, dist = start.get("down"), _num(start.get("distance"))
                    o["plays"] += 1
                    o["yards"] += yds
                    s = is_success(down, dist, yds, td, to)
                    if s is not None:
                        o["succ_n"] += 1
                        o["success"] += int(s)
                    o["explosive"] += int(yds >= 20)
                    o[kind] += 1
                    o[f"{kind}_yds"] += yds
                    if p.get("type", {}).get("text") == "Sack":
                        o["sacks"] += 1
                    elif kind == "rush" and yds < 0:
                        o["tfl"] += 1
                    if down == 3:
                        o["third_att"] += 1
                        o["third_conv"] += int(td or (dist is not None and yds >= dist and not to))
            home_sc = _num(p.get("homeScore"), home_sc)
            away_sc = _num(p.get("awayScore"), away_sc)
        if dside:
            res = (d.get("displayResult") or d.get("result") or "").lower()
            n_off = d.get("offensivePlays") or 0
            o = off[dside]
            o["turnovers"] += int(res in TO_DRIVE_RESULTS)   # whole game, incl. garbage time
            if res in ("kickoff",) or (("end of" in res) and n_off <= 3):
                continue
            o["drives"] += 1
            pts = 7 if res == "touchdown" else 3 if res == "field goal" else 0
            o["drive_pts"] += pts
            o["td"] += int(pts == 7)
            o["fg"] += int(pts == 3)
            if reached_rz:
                o["rz_trips"] += 1
                o["rz_td"] += int(pts == 7)

    # Box-score extras (penalties, possession) — whole game, no garbage filter
    box = {}
    for t in summary.get("boxscore", {}).get("teams", []):
        st = {x["name"]: x.get("displayValue") for x in t.get("statistics", [])}
        box[side_of.get(str(t["team"]["id"]))] = st

    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        o, dfn = off[side], off[opp]
        pen = (box.get(side, {}).get("totalPenaltiesYards") or "0-0").split("-")
        top = box.get(side, {}).get("possessionTime") or ""
        rows.append({
            "event_id": str(summary.get("header", {}).get("id")),
            "team_id": ids[side], "opp_id": ids[opp], "home_away": side,
            "points": score[side], "opp_points": score[opp],
            "win": int(score[side] > score[opp]),
            **{f"o_{k}": v for k, v in o.items()},
            **{f"d_{k}": v for k, v in dfn.items()},
            "penalties": _num(pen[0], 0.0), "penalty_yds": _num(pen[-1], 0.0),
            "top_min": (lambda m: _num(m[0], 0) + _num(m[1], 0) / 60 if len(m) == 2 else None)(top.split(":")),
        })
    return rows
