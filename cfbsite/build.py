"""Build the stats snapshots the site reads.

    python -m cfbsite.build            # current season, all completed weeks
    python -m cfbsite.build 2026 1 4   # season, first week, last week

Writes site_data/<season>/:
  teams.parquet        FBS teams: conference, logo, colours
  games.parquet        every game involving an FBS team (scores, lines, status)
  team_games.parquet   one row per team per completed game (raw counts)
  team_season.parquet  season stats per FBS team + national rank per stat
  meta.json            build time and coverage
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import metrics, source

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (column, label, higher_is_better, format) — the stats the site shows and ranks.
STATS = [
    ("win_pct", "Win %", True, "pct"),
    ("ppg", "Points per game", True, "1f"),
    ("opp_ppg", "Points allowed per game", False, "1f"),
    ("margin_pg", "Scoring margin", True, "+1f"),
    ("o_ypp", "Yards per play", True, "2f"),
    ("o_sr", "Success rate", True, "pct"),
    ("o_expl", "Explosive play rate (20+ yds)", True, "pct"),
    ("o_ppd", "Points per drive", True, "2f"),
    ("o_rush_ypc", "Yards per rush", True, "2f"),
    ("o_pass_ypd", "Yards per dropback", True, "2f"),
    ("o_rush_rate", "Run rate", None, "pct"),
    ("o_third", "3rd-down conversion", True, "pct"),
    ("o_rz_td", "Red-zone TD rate", True, "pct"),
    ("o_havoc_allowed", "Havoc allowed", False, "pct"),
    ("o_plays_pg", "Plays per game", None, "1f"),
    ("d_ypp", "Yards per play allowed", False, "2f"),
    ("d_sr", "Success rate allowed", False, "pct"),
    ("d_expl", "Explosive plays allowed", False, "pct"),
    ("d_ppd", "Points per drive allowed", False, "2f"),
    ("d_rush_ypc", "Yards per rush allowed", False, "2f"),
    ("d_pass_ypd", "Yards per dropback allowed", False, "2f"),
    ("d_third", "3rd-down conversion allowed", False, "pct"),
    ("d_rz_td", "Red-zone TD rate allowed", False, "pct"),
    ("d_havoc", "Havoc rate (defense)", True, "pct"),
    ("to_margin_pg", "Turnover margin per game", True, "+2f"),
    ("penalty_yds_pg", "Penalty yards per game", False, "1f"),
    ("net_ypp", "Net yards per play", True, "+2f"),
    ("net_sr", "Net success rate", True, "+pct"),
]
STAT_META = {c: {"label": l, "higher_better": h, "fmt": f} for c, l, h, f in STATS}


def _div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def season_table(tg, teams):
    """Aggregate per-game counts to season rates, then rank within FBS."""
    g = tg.groupby("team_id")
    s = g.agg({c: "sum" for c in tg.columns if c.startswith(("o_", "d_"))
               or c in ("points", "opp_points", "win", "penalties", "penalty_yds")})
    s["games"] = g.size()
    s["losses"] = s["games"] - s["win"]
    for side in ("o", "d"):
        s[f"{side}_ypp"] = _div(s[f"{side}_yards"], s[f"{side}_plays"])
        s[f"{side}_sr"] = _div(s[f"{side}_success"], s[f"{side}_succ_n"])
        s[f"{side}_expl"] = _div(s[f"{side}_explosive"], s[f"{side}_plays"])
        s[f"{side}_ppd"] = _div(s[f"{side}_drive_pts"], s[f"{side}_drives"])
        s[f"{side}_rush_ypc"] = _div(s[f"{side}_rush_yds"], s[f"{side}_rush"])
        s[f"{side}_pass_ypd"] = _div(s[f"{side}_pass_yds"], s[f"{side}_pass"])
        s[f"{side}_rush_rate"] = _div(s[f"{side}_rush"], s[f"{side}_plays"])
        s[f"{side}_third"] = _div(s[f"{side}_third_conv"], s[f"{side}_third_att"])
        s[f"{side}_rz_td"] = _div(s[f"{side}_rz_td"], s[f"{side}_rz_trips"])
        s[f"{side}_plays_pg"] = s[f"{side}_plays_all"] / s["games"]
    s["o_havoc_allowed"] = _div(s["o_sacks"] + s["o_tfl"] + s["o_turnovers"], s["o_plays"])
    s["d_havoc"] = _div(s["d_sacks"] + s["d_tfl"] + s["d_turnovers"], s["d_plays"])
    s["win_pct"] = s["win"] / s["games"]
    s["ppg"] = s["points"] / s["games"]
    s["opp_ppg"] = s["opp_points"] / s["games"]
    s["margin_pg"] = s["ppg"] - s["opp_ppg"]
    s["to_margin_pg"] = (s["d_turnovers"] - s["o_turnovers"]) / s["games"]
    s["penalty_yds_pg"] = s["penalty_yds"] / s["games"]
    s["net_ypp"] = s["o_ypp"] - s["d_ypp"]
    s["net_sr"] = s["o_sr"] - s["d_sr"]
    s["record"] = s["win"].astype(int).astype(str) + "-" + s["losses"].astype(int).astype(str)

    s = s.reset_index()
    s = s[s["team_id"].isin(teams["team_id"])].copy()   # rank FBS teams only
    for col, m in STAT_META.items():
        if m["higher_better"] is None:
            continue
        s[f"{col}_rank"] = s[col].rank(ascending=not m["higher_better"], method="min")
        # percentile: 100 = best in FBS
        s[f"{col}_pctl"] = 100 * (1 - (s[f"{col}_rank"] - 1) / max(len(s) - 1, 1))
    keep = ["team_id", "games", "record", "win", "losses"] + list(STAT_META) + \
           [c for c in s.columns if c.endswith(("_rank", "_pctl"))]
    return s[keep].merge(teams, on="team_id", how="left")


def build(season=None, first_week=1, last_week=None, verbose=True):
    cur_season, cur_week, _ = source.current_week()
    season = season or cur_season
    last_week = last_week or (cur_week if season == cur_season else 16)

    teams = pd.DataFrame(source.fbs_teams(season))
    directory = source.team_directory()
    teams["color"] = teams["team_id"].map(lambda i: directory.get(i, {}).get("color", "#666666"))
    teams["alt_color"] = teams["team_id"].map(lambda i: directory.get(i, {}).get("alt_color", "#cccccc"))
    fbs_ids = set(teams["team_id"])
    if verbose:
        print(f"   {season}: {len(teams)} FBS teams, weeks {first_week}-{last_week}")

    from wf.espn import lines_from_odds, parse_event   # flat rows incl. lines
    games, team_rows = [], []
    for wk in range(first_week, last_week + 1):
        sb = source.scoreboard(season, wk)
        n_done = 0
        for ev in sb.get("events", []):
            row = parse_event(ev)
            comps = ev["competitions"][0]["competitors"]
            ids = {c["homeAway"]: str(c["team"]["id"]) for c in comps}
            row.update({"week": wk, "home_id": ids.get("home"), "away_id": ids.get("away")})
            row["fbs_vs_fbs"] = ids.get("home") in fbs_ids and ids.get("away") in fbs_ids
            row.pop("_home_id", None); row.pop("_away_id", None)
            games.append(row)
            if row["completed"]:
                s = source.summary(row["event_id"], completed=True)
                pc = (s.get("pickcenter") or [None])[0]
                if pc and row.get("spread") is None:   # scoreboard drops odds once final
                    row.update({k: v for k, v in lines_from_odds(pc).items() if v is not None})
                for r in metrics.game_rows(s):
                    r.update({"week": wk, "commence": row["commence"], "opp_fbs": r["opp_id"] in fbs_ids})
                    team_rows.append(r)
                n_done += 1
        if verbose:
            print(f"     week {wk}: {len(sb.get('events', []))} games, {n_done} final")

    games = pd.DataFrame(games)
    tg = pd.DataFrame(team_rows)
    ts = season_table(tg, teams) if len(tg) else pd.DataFrame()

    out = os.path.join(ROOT, "site_data", str(season))
    os.makedirs(out, exist_ok=True)
    teams.to_parquet(os.path.join(out, "teams.parquet"), index=False)
    games.to_parquet(os.path.join(out, "games.parquet"), index=False)
    tg.to_parquet(os.path.join(out, "team_games.parquet"), index=False)
    ts.to_parquet(os.path.join(out, "team_season.parquet"), index=False)
    meta = {"season": season, "weeks": [first_week, last_week],
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fbs_teams": len(teams), "games": int(len(games)),
            "final_games": int(games["completed"].sum()) if len(games) else 0,
            "stats": STAT_META}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"   -> {out}: {meta['final_games']} final games, {len(ts)} teams ranked")
    return meta


if __name__ == "__main__":
    build(*[int(x) for x in sys.argv[1:]])
