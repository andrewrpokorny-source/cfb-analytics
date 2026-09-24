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

# (column, label, higher_is_better, format, category) — shown and ranked on the site.
# higher_is_better=None -> descriptive (shown, not ranked).
STATS = [
    # overview
    ("power", "Power rating (pts vs avg team)", True, "+1f", "Ratings"),
    ("adj_off", "Adj. offense (EPA/play)", True, "+3f", "Ratings"),
    ("adj_def", "Adj. defense (EPA/play allowed)", False, "+3f", "Ratings"),
    ("sos", "Strength of schedule (avg opp. power)", True, "+1f", "Ratings"),
    ("win_pct", "Win %", True, "pct", "Results"),
    ("ppg", "Points per game", True, "1f", "Results"),
    ("opp_ppg", "Points allowed per game", False, "1f", "Results"),
    ("margin_pg", "Scoring margin", True, "+1f", "Results"),
    ("to_margin_pg", "Turnover margin per game", True, "+2f", "Results"),
    ("penalty_yds_pg", "Penalty yards per game", False, "1f", "Results"),
    # offense
    ("o_epa", "EPA per play", True, "+3f", "Offense"),
    ("o_rush_epa", "EPA per rush", True, "+3f", "Offense"),
    ("o_pass_epa", "EPA per dropback", True, "+3f", "Offense"),
    ("o_early_epa", "Early-down EPA (1st & 2nd)", True, "+3f", "Offense"),
    ("o_sr", "Success rate", True, "pct", "Offense"),
    ("o_sd_sr", "Standard-downs success rate", True, "pct", "Offense"),
    ("o_pd_sr", "Passing-downs success rate", True, "pct", "Offense"),
    ("o_isoppp", "Explosiveness (EPA per successful play)", True, "2f", "Offense"),
    ("o_expl", "Explosive play rate (20+ yds)", True, "pct", "Offense"),
    ("o_ypp", "Yards per play", True, "2f", "Offense"),
    ("o_rush_ypc", "Yards per rush", True, "2f", "Offense"),
    ("o_pass_ypd", "Yards per dropback", True, "2f", "Offense"),
    ("o_stuff", "Stuff rate (runs for <= 0)", False, "pct", "Offense"),
    ("o_third", "3rd-down conversion", True, "pct", "Offense"),
    ("o_havoc_allowed", "Havoc allowed", False, "pct", "Offense"),
    # finishing & field position
    ("o_ppd", "Points per drive", True, "2f", "Drives"),
    ("o_pts_opp", "Points per scoring opportunity", True, "2f", "Drives"),
    ("o_rz_td", "Red-zone TD rate", True, "pct", "Drives"),
    ("o_start", "Avg. starting field position (yds to go)", False, "1f", "Drives"),
    ("d_ppd", "Points per drive allowed", False, "2f", "Drives"),
    ("d_pts_opp", "Points per opportunity allowed", False, "2f", "Drives"),
    ("d_rz_td", "Red-zone TD rate allowed", False, "pct", "Drives"),
    ("d_start", "Opp. starting field position (yds to go)", True, "1f", "Drives"),
    # defense
    ("d_epa", "EPA per play allowed", False, "+3f", "Defense"),
    ("d_rush_epa", "EPA per rush allowed", False, "+3f", "Defense"),
    ("d_pass_epa", "EPA per dropback allowed", False, "+3f", "Defense"),
    ("d_early_epa", "Early-down EPA allowed", False, "+3f", "Defense"),
    ("d_sr", "Success rate allowed", False, "pct", "Defense"),
    ("d_sd_sr", "Standard-downs success allowed", False, "pct", "Defense"),
    ("d_pd_sr", "Passing-downs success allowed", False, "pct", "Defense"),
    ("d_isoppp", "Explosiveness allowed", False, "2f", "Defense"),
    ("d_expl", "Explosive plays allowed (20+)", False, "pct", "Defense"),
    ("d_ypp", "Yards per play allowed", False, "2f", "Defense"),
    ("d_rush_ypc", "Yards per rush allowed", False, "2f", "Defense"),
    ("d_pass_ypd", "Yards per dropback allowed", False, "2f", "Defense"),
    ("d_stuff", "Stuff rate (defense)", True, "pct", "Defense"),
    ("d_third", "3rd-down conversion allowed", False, "pct", "Defense"),
    ("d_havoc", "Havoc rate (sacks + TFL + takeaways)", True, "pct", "Defense"),
    # style
    ("o_neutral_pass", "Neutral-situation pass rate", None, "pct", "Style"),
    ("o_rush_rate", "Run rate (all plays)", None, "pct", "Style"),
    ("o_plays_pg", "Plays per game", None, "1f", "Style"),
    ("net_ypp", "Net yards per play", True, "+2f", "Results"),
    ("net_sr", "Net success rate", True, "+pct", "Results"),
]
STAT_META = {c: {"label": l, "higher_better": h, "fmt": f, "cat": cat} for c, l, h, f, cat in STATS}


def _div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def season_table(tg, teams, games):
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
    for side in ("o", "d"):
        s[f"{side}_epa_pp"] = _div(s[f"{side}_epa"], s[f"{side}_epa_n"])
        s[f"{side}_rush_epa_pp"] = _div(s[f"{side}_rush_epa"], s[f"{side}_rush"])
        s[f"{side}_pass_epa_pp"] = _div(s[f"{side}_pass_epa"], s[f"{side}_pass"])
        s[f"{side}_early_epa_pp"] = _div(s[f"{side}_early_epa"], s[f"{side}_early_n"])
        s[f"{side}_sd_sr"] = _div(s[f"{side}_sd_succ"], s[f"{side}_sd_n"])
        s[f"{side}_pd_sr"] = _div(s[f"{side}_pd_succ"], s[f"{side}_pd_n"])
        s[f"{side}_isoppp"] = _div(s[f"{side}_succ_epa"], s[f"{side}_success"])
        s[f"{side}_stuff"] = _div(s[f"{side}_stuffed"], s[f"{side}_rush"])
        s[f"{side}_pts_opp"] = _div(s[f"{side}_opp_pts"], s[f"{side}_opp_n"])
        s[f"{side}_start"] = _div(s[f"{side}_start_ytg_sum"], s[f"{side}_drives"])
        s[f"{side}_neutral_pass"] = _div(s[f"{side}_neutral_pass"], s[f"{side}_neutral_n"])
        # per-play EPA columns take the display names
        for k in ("epa", "rush_epa", "pass_epa", "early_epa"):
            s[f"{side}_{k}"] = s[f"{side}_{k}_pp"]
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

    # opponent-adjusted ratings over ALL teams (FCS opponents included), then
    # rank FBS only
    from . import ratings
    from . import priors
    season = int(pd.to_datetime(tg["commence"]).dt.year.mode()[0])
    pri = priors.build(season, teams)
    rt, prm = ratings.fit(tg, games, fbs_ids=set(teams["team_id"]), prior=pri)
    prm["prior_teams"] = len(pri)
    a, b, r = ratings.points_map(tg)
    prm.update({"pts_a": a, "pts_b": b, "pts_r": r})
    s = s.merge(rt.rename(columns={"off": "adj_off_raw", "def": "adj_def_raw"}), on="team_id", how="left")
    s["adj_off"] = prm["mu"] + s["adj_off_raw"]
    s["adj_def"] = prm["mu"] + s["adj_def_raw"]
    s["power"] = b * (s["adj_off_raw"] - s["adj_def_raw"])
    power = dict(zip(s["team_id"], s["power"]))
    fcs_power = b * float(rt.loc[rt["team_id"] == ratings.FCS, "off"].sum()
                          - rt.loc[rt["team_id"] == ratings.FCS, "def"].sum())
    opp = tg.groupby("team_id")["opp_id"].apply(list)
    s["sos"] = s["team_id"].map(lambda t: np.nanmean([power.get(o, fcs_power) for o in opp.get(t, [])]))
    prm["fcs_power"] = fcs_power
    s = s[s["team_id"].isin(teams["team_id"])].copy()   # rank FBS teams only
    for col, m in STAT_META.items():
        if m["higher_better"] is None:
            continue
        s[f"{col}_rank"] = s[col].rank(ascending=not m["higher_better"], method="min")
        # percentile: 100 = best in FBS
        s[f"{col}_pctl"] = 100 * (1 - (s[f"{col}_rank"] - 1) / max(len(s) - 1, 1))
    keep = ["team_id", "games", "record", "win", "losses"] + list(STAT_META) + \
           [c for c in s.columns if c.endswith(("_rank", "_pctl"))]
    return s[keep + ["adj_off_raw", "adj_def_raw", "prior_off", "prior_def"]].merge(teams, on="team_id", how="left"), prm


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
    ts, prm = season_table(tg, teams, games) if len(tg) else (pd.DataFrame(), {})

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
            "stats": STAT_META, "ratings": prm}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if verbose:
        print(f"   -> {out}: {meta['final_games']} final games, {len(ts)} teams ranked")
    return meta


if __name__ == "__main__":
    build(*[int(x) for x in sys.argv[1:]])
