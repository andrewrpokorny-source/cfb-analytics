"""Predictions for a day's FBS slate.

    python -m engine.today            # today (US/Eastern)
    python -m engine.today 2026-11-07

Writes predictions/<date>.csv and predictions/<date>.md. Parameters below were
fit walk-forward on 2024-25 and checked out-of-sample on 2026 weeks 2-9:

  margin: model weight 0.027 -> blend MAE 11.161 vs market 11.165 (no real edge)
  total:  model weight 0.217 -> blend MAE 12.395 vs market 12.405
  win prob sigma 14.2 -> 2026 Brier 0.1727 (market alone 0.1726)

So these are market-anchored forecasts, not betting picks.
"""
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cfbsite import source
from wf.espn import parse_event
from engine import core

W_MARGIN, W_TOTAL, SIGMA, TOTAL_SD = 0.027, 0.217, 14.2, 15.6
ET = ZoneInfo("America/New_York")


def slate(day):
    season, week, d = source.current_week()
    rows = []
    for ev in d.get("events", []):
        r = parse_event(ev)
        local = pd.Timestamp(r["commence"]).tz_convert(ET)
        if local.date().isoformat() != day:
            continue
        comps = ev["competitions"][0]["competitors"]
        ids = {c["homeAway"]: str(c["team"]["id"]) for c in comps}
        r.update({"home_id": ids.get("home"), "away_id": ids.get("away"),
                  "kickoff_et": local.strftime("%-I:%M %p"), "_ko": local})
        rows.append(r)
    return season, week, pd.DataFrame(rows)


def predict(day):
    season, week, g = slate(day)
    games, tg, teams, pri = core.load_season(season)
    rt = core.Ratings(tg, games, teams, pri)          # all completed games so far
    fbs = set(teams["team_id"])
    p = core.predict_rows(g, rt)
    g = pd.concat([g, p], axis=1)
    g["market_margin"] = -g["spread"]
    has_mkt = g["spread"].notna()
    g["margin"] = np.where(has_mkt, g["market_margin"] + W_MARGIN * (g["model_margin"] - g["market_margin"]),
                           g["model_margin"])
    has_tot = g["total"].notna()
    g["pred_total"] = np.where(has_tot, g["total"] + W_TOTAL * (g["model_total"] - g["total"]), g["model_total"])
    g["home_win_prob"] = [core.win_prob(m, SIGMA) for m in g["margin"]]
    g["over_prob"] = [core.phi((t - m) / TOTAL_SD) if pd.notna(m) else np.nan
                      for t, m in zip(g["pred_total"], g["total"])]
    g["home_pts"] = (g["pred_total"] + g["margin"]) / 2
    g["away_pts"] = (g["pred_total"] - g["margin"]) / 2
    g["fbs_vs_fbs"] = g["home_id"].isin(fbs) & g["away_id"].isin(fbs)
    g["source"] = np.where(has_mkt, "market+model", "model only")
    # Tracking candidate from research (unproven, paper-trade only): UNDER when
    # total >= 58 and the under is priced -112 or better.
    g["watch_under58"] = has_tot & (g["total"] >= 58) & (g["under_price"].fillna(-999) >= -112) & g["fbs_vs_fbs"]
    g = g.sort_values(["_ko", "game"])
    return season, week, g


def fmt_line(margin, home, away):
    if abs(margin) < 0.25:
        return "pick'em"
    return f"{home} -{margin:.1f}" if margin > 0 else f"{away} -{-margin:.1f}"


def write(day, season, week, g):
    import os
    out = os.path.join(core.ROOT, "predictions")
    os.makedirs(out, exist_ok=True)
    cols = ["kickoff_et", "away", "home", "neutral", "state", "spread", "total", "model_margin", "model_total",
            "margin", "pred_total", "away_pts", "home_pts", "home_win_prob", "over_prob", "source",
            "watch_under58", "event_id", "fbs_vs_fbs"]
    g[cols].round(3).to_csv(os.path.join(out, f"{day}.csv"), index=False)
    lines = [f"# CFB predictions: {day} (season {season}, week {week})", "",
             f"Generated {datetime.now(ET).strftime('%Y-%m-%d %-I:%M %p ET')}. "
             "Market-anchored forecasts: the ratings model does not beat the closing line on margins, "
             "so the projected line is ~the market with a tiny model nudge; totals lean 22% on the model. "
             "Win probabilities are calibrated (out-of-sample Brier 0.173 on 2026). These are not betting picks.", "",
             "| Kick (ET) | Matchup | Projected score | Proj. line | Market | Win prob | Proj. total | Market total |",
             "|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        fav_wp = max(r.home_win_prob, 1 - r.home_win_prob)
        fav = r.home if r.home_win_prob >= 0.5 else r.away
        mkt = "—" if pd.isna(r.spread) else fmt_line(-r.spread, r.home, r.away)
        mt = "—" if pd.isna(r.total) else f"{r.total:g}"
        at = "vs" if r.neutral else "@"
        flag = " ⚑" if r.watch_under58 else ""
        lines.append(f"| {r.kickoff_et} | {r.away} {at} {r.home} | {r.away} {r.away_pts:.0f}, {r.home} {r.home_pts:.0f} "
                     f"| {fmt_line(r.margin, r.home, r.away)} | {mkt} | {fav} {fav_wp:.0%} | {r.pred_total:.1f} | {mt}{flag} |")
    lines += ["", "⚑ = high-total UNDER tracking candidate (total ≥ 58, under priced -112 or better). "
              "Unproven: 95% CI includes break-even. Paper-trade only."]
    with open(os.path.join(out, f"{day}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.join(out, f"{day}.md")


if __name__ == "__main__":
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now(ET).date().isoformat()
    season, week, g = predict(day)
    path = write(day, season, week, g)
    print(open(path).read())
