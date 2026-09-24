"""Preseason priors for the power ratings.

Fitted on 786 team-season pairs (2018-2025, CFBD advanced stats):
    next offense EPA/play (centered) = 0.381 * last season's + 0.017 * talent z-score
    next defense EPA/play (centered) = 0.394 * last season's - 0.016 * talent z-score
R^2 0.22 / 0.24. Returning production added ~0.002 R^2, so it is left out.
Early in the season the ridge ratings shrink toward these instead of toward
"average"; as games accumulate, the season's own data takes over.
"""
import json
import os

import numpy as np

from wf.espn import normalise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COEF = {"o_prev": 0.381, "o_tal": 0.017, "d_prev": 0.394, "d_tal": -0.016}


def _cfbd(endpoint, params):
    from wf.cache import _key   # key only; never triggers a network call
    p = os.path.join(ROOT, "data_cache", _key(endpoint, params) + ".json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def build(season, teams):
    """{espn team_id: (prior_off, prior_def)} in centered EPA/play units."""
    adv = None
    for ew in (16, 15, 14, 13):
        adv = _cfbd("/stats/season/advanced", {"year": season - 1, "startWeek": 1, "endWeek": ew})
        if adv:
            break
    if not adv:
        return {}
    tal = _cfbd("/talent", {"year": season}) or []
    o = {normalise(r["team"]): r["offense"].get("ppa") for r in adv}
    d = {normalise(r["team"]): r["defense"].get("ppa") for r in adv}
    t = {normalise(r.get("team") or r.get("school")): r.get("talent") for r in tal}
    ov = [v for v in o.values() if v is not None]
    dv = [v for v in d.values() if v is not None]
    tv = [v for v in t.values() if v is not None]
    om, dm = np.mean(ov), np.mean(dv)
    tm, ts = (np.mean(tv), np.std(tv)) if tv else (0, 1)
    out = {}
    for tid, name in zip(teams["team_id"], teams["team"]):
        k = normalise(name)
        if o.get(k) is None:
            continue
        tz = ((t[k] - tm) / ts) if t.get(k) is not None else 0.0
        out[tid] = (COEF["o_prev"] * (o[k] - om) + COEF["o_tal"] * tz,
                    COEF["d_prev"] * (d[k] - dm) + COEF["d_tal"] * tz)
    return out
