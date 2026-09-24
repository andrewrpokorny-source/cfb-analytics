"""Closing lines for past seasons, from the local CFBD cache.

ESPN's game summaries stop carrying odds after a while (0 of 874 games in
2024, 57 of 888 in 2025), so past seasons take lines from the CFBD /lines
responses cached during the research phase. Read-only and keyless: it reads
data_cache/ directly and never imports wf.config. Same book hygiene as the
live pipeline: duplicate provider spellings merged, home/away-swapped rows
dropped, consensus = median across books.
"""
import json
import os

import numpy as np

from wf.espn import normalise
from .priors import ROOT, _key


def _clean(lines):
    seen, out = set(), []
    for b in lines:
        c = "".join((b.get("provider") or "").split()).casefold()
        if not c or c in seen or c in ("teamrankings", "numberfire"):   # model projections, not markets
            continue
        sp, hml, aml = b.get("spread"), b.get("homeMoneyline"), b.get("awayMoneyline")
        if sp is not None and hml and aml and abs(sp) >= 1 and (sp < 0) != (hml < aml):
            continue
        seen.add(c)
        out.append(b)
    return out


def load(season):
    """{(home_norm, away_norm): (spread, total)} for a season, or {}."""
    path = os.path.join(ROOT, "data_cache", _key("/lines", {"year": season, "seasonType": "regular"}) + ".json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for g in data:
        books = _clean(g.get("lines") or [])
        sp = [b["spread"] for b in books if b.get("spread") is not None]
        to = [b["overUnder"] for b in books if b.get("overUnder") is not None]
        if not sp and not to:
            continue
        out[(normalise(g.get("homeTeam")), normalise(g.get("awayTeam")))] = (
            float(np.median(sp)) if sp else None, float(np.median(to)) if to else None)
    return out
