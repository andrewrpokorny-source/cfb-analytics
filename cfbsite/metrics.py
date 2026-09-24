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

import numpy as np

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
DEF_TD_TYPES = {"Interception Return Touchdown", "Fumble Return Touchdown",
                "Pass Interception Return Touchdown", "Fumble Recovery (Opponent) Touchdown",
                "Interception Touchdown", "Fumble Touchdown"}
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


def _half_secs(period, clock):
    try:
        mm, ss = str((clock or {}).get("displayValue", "0:00")).split(":")
        secs = int(mm) * 60 + int(ss)
    except ValueError:
        secs = 0
    return secs + (900 if period in (1, 3) else 0)


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
        "drives", "drive_pts", "rz_trips", "rz_td", "td", "fg", "plays_all",
        # EPA family (non-garbage scrimmage plays)
        "epa", "epa_n", "rush_epa", "pass_epa", "early_epa", "early_n",
        "succ_epa", "sd_n", "sd_succ", "pd_n", "pd_succ", "stuffed",
        # drive context
        "start_ytg_sum", "opp_n", "opp_pts",
        # neutral-situation play calling (|margin| <= 8, Q1-3)
        "neutral_n", "neutral_pass")}
    off = {"home": blank(), "away": blank()}

    drives = summary.get("drives", {}).get("previous", [])
    ep_rows = []   # (side, kind, succ, down, before-state, after-state | points)

    # ---- pass 1: drives (order-independent) ----------------------------------
    to_plays, flat = set(), []
    for d in drives:
        dteam = str((d.get("team") or {}).get("id", ""))
        dside = side_of.get(dteam)
        plays = d.get("plays", [])
        scrim = [p for p in plays if play_kind(p)]
        res = (d.get("displayResult") or d.get("result") or "").lower()
        if res in TO_DRIVE_RESULTS and scrim:
            to_plays.add(scrim[-1].get("id"))
        for p in plays:
            flat.append((dside, p))
        if not dside:
            continue
        o = off[dside]
        o["turnovers"] += int(res in TO_DRIVE_RESULTS)   # whole game, incl. garbage time
        n_off = d.get("offensivePlays") or 0
        if res in ("kickoff",) or (("end of" in res) and n_off <= 3):
            continue
        ytgs = [(pl.get("start") or {}).get("yardsToEndzone") for pl in scrim]
        ytgs = [y for y in ytgs if y is not None]
        pts = 7 if res == "touchdown" else 3 if res == "field goal" else 0
        o["drives"] += 1
        o["drive_pts"] += pts
        o["td"] += int(pts == 7)
        o["fg"] += int(pts == 3)
        if ytgs:
            o["start_ytg_sum"] += ytgs[0]
            if min(ytgs) <= 40:                      # a scoring opportunity
                o["opp_n"] += 1
                o["opp_pts"] += pts
            if min(ytgs) <= 20:
                o["rz_trips"] += 1
                o["rz_td"] += int(pts == 7)

    # ---- pass 2: plays in true chronological order -----------------------------
    # ESPN's drive list is sometimes out of order (overtime drives interleaved
    # with the 2nd quarter, a score typo of 1,414). Sort by period, then ESPN's
    # sequence number, and only let the running score move UP by a plausible
    # amount, so the garbage-time margin is never computed from a glitch.
    def _seq(item):
        p = item[1]
        try:
            sq = int(p.get("sequenceNumber") or 0)
        except (TypeError, ValueError):
            sq = 0
        return ((p.get("period") or {}).get("number", 1), sq)

    flat.sort(key=_seq)
    home_sc = away_sc = 0.0
    for dside, p in flat:
        period = (p.get("period") or {}).get("number", 1)
        start = p.get("start") or {}
        oside = side_of.get(str((start.get("team") or {}).get("id", "")), dside)
        margin = abs(home_sc - away_sc)
        garbage = margin > GARBAGE.get(period, 999)
        kind = play_kind(p)
        if kind and oside and period <= 4:
            o = off[oside]
            o["plays_all"] += 1
            if not garbage:
                ptype = p.get("type", {}).get("text", "")
                yds = _num(p.get("statYardage"), 0.0)
                def_td = ptype in DEF_TD_TYPES
                to = (p.get("id") in to_plays) or ptype in INT_TYPES \
                    or ptype == "Fumble Recovery (Opponent)" or def_td
                td = "Touchdown" in ptype and not to
                down, dist = start.get("down"), _num(start.get("distance"))
                ytg = start.get("yardsToEndzone")
                o["plays"] += 1
                o["yards"] += yds
                s = is_success(down, dist, yds, td, to)
                if s is not None:
                    o["succ_n"] += 1
                    o["success"] += int(s)
                o["explosive"] += int(yds >= 20)
                o["stuffed"] += int(kind == "rush" and yds <= 0)
                # standard vs passing downs (Connelly): 2nd & 8+, 3rd/4th & 5+
                passing_down = (down == 2 and (dist or 0) >= 8) or (down in (3, 4) and (dist or 0) >= 5)
                if s is not None:
                    k = "pd" if passing_down else "sd"
                    o[f"{k}_n"] += 1
                    o[f"{k}_succ"] += int(s)
                if period <= 3 and margin <= 8:
                    o["neutral_n"] += 1
                    o["neutral_pass"] += int(kind == "pass")
                # EPA: scoring decided by play TYPE, never by score deltas
                end = p.get("end") or {}
                before = (ytg, down, dist, _half_secs(period, p.get("clock")))
                if td:
                    after = ("pts", 7.0)
                elif def_td:
                    after = ("pts", -7.0)
                elif end.get("down") and end.get("yardsToEndzone") is not None:
                    same = str((end.get("team") or {}).get("id", "")) == ids[oside]
                    after = ("state", same, (end.get("yardsToEndzone"), end.get("down"),
                                             _num(end.get("distance"), 10), before[3]))
                else:
                    after = None
                if down and ytg is not None and after is not None:
                    ep_rows.append((oside, kind, s, down, before, after))
                o[kind] += 1
                o[f"{kind}_yds"] += yds
                if ptype == "Sack":
                    o["sacks"] += 1
                elif kind == "rush" and yds < 0:
                    o["tfl"] += 1
                if down == 3:
                    o["third_att"] += 1
                    o["third_conv"] += int(td or (dist is not None and yds >= dist and not to))
        # running score: monotone, plausible steps only
        nh, na = _num(p.get("homeScore"), home_sc), _num(p.get("awayScore"), away_sc)
        if home_sc <= nh <= home_sc + 9:
            home_sc = nh
        if away_sc <= na <= away_sc + 9:
            away_sc = na

    if ep_rows:
        from . import ep as epm
        states = [r[4] for r in ep_rows] + [r[5][2] for r in ep_rows if r[5][0] == "state"]
        vals = epm.ep([x[0] for x in states], [x[1] for x in states], [x[2] for x in states],
                      [x[3] for x in states])
        n = len(ep_rows)
        before_ep, after_iter = vals[:n], iter(vals[n:])
        for (side_, kind, succ, down, _, after), b in zip(ep_rows, before_ep):
            if after[0] == "pts":
                a_ep = after[1]
            else:
                v = next(after_iter)
                a_ep = v if after[1] else -v
            e = float(np.clip(a_ep - b, -10, 10))
            o = off[side_]
            o["epa"] += e
            o["epa_n"] += 1
            o[f"{kind}_epa"] += e
            if down in (1, 2):
                o["early_epa"] += e
                o["early_n"] += 1
            if succ:
                o["succ_epa"] += e

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
