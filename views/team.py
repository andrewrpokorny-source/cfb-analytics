import numpy as np
import pandas as pd
import streamlit as st

from cfbsite import ui

snap = ui.snapshot()
if snap is None:
    st.error("Stats snapshot missing — run `python -m cfbsite.build`.")
    st.stop()
meta, ss, tg, games, teams = (snap["meta"], snap["season_stats"], snap["team_games"],
                              snap["games"], snap["teams"])
n_teams = len(ss)
names = dict(zip(teams["team_id"], teams["team"]))

# ---------------------------------------------------------------- pick a team
order = ss.sort_values("team")
ids = list(order["team_id"])
labels = {r["team_id"]: f"{r['team']}  ·  {r['conf_abbr'].upper() if isinstance(r['conf_abbr'], str) else ''}"
          for _, r in order.iterrows()}
q = st.query_params.get("team")
idx = ids.index(q) if q in ids else 0
tid = st.selectbox("Team", ids, index=idx, format_func=lambda i: labels[i])
if tid != q:
    st.query_params["team"] = tid
t = ss.set_index("team_id").loc[tid].to_dict()

# ---------------------------------------------------------------- header
h1, h2 = st.columns([1, 6])
with h1:
    if t.get("logo"):
        st.image(t["logo"], width=96)
with h2:
    st.title(t["team"])
    conf = teams.set_index("team_id").loc[tid, "conference"]
    st.caption(f"{conf} · {t['record']} · through {int(t['games'])} games")

k = st.columns(4)
for col, c in zip(k, ["margin_pg", "net_ypp", "o_ppd", "d_ppd"]):
    m = meta["stats"][c]
    col.metric(m["label"], ui.fmt(t.get(c), m["fmt"]), f"{ui.ordinal(t[c + '_rank'])} of {n_teams}",
               delta_color="off")

# ---------------------------------------------------------------- profile
st.subheader("Profile vs FBS")
st.caption("Percentile among all FBS teams: 100 = best, 50 = median (dotted line). "
           "Blue = better than median, orange = worse. Garbage time excluded.")
o_col, d_col = st.columns(2)
with o_col:
    st.markdown("**Offense**")
    st.plotly_chart(ui.percentile_bars(t, ui.OFFENSE, meta, n_teams), width="stretch",
                    config={"displayModeBar": False})
with d_col:
    st.markdown("**Defense**")
    st.plotly_chart(ui.percentile_bars(t, ui.DEFENSE, meta, n_teams), width="stretch",
                    config={"displayModeBar": False})

# ---------------------------------------------------------------- schedule
st.subheader("Schedule & results")
g = games[(games["home_id"] == tid) | (games["away_id"] == tid)].sort_values("commence")
rows, upcoming = [], []
for _, r in g.iterrows():
    home = r["home_id"] == tid
    opp = r["away"] if home else r["home"]
    opp_id = r["away_id"] if home else r["home_id"]
    ko = pd.to_datetime(r["commence"], utc=True).tz_convert("America/New_York")
    my_line = r["spread"] if home else (-r["spread"] if pd.notna(r["spread"]) else np.nan)
    res = ats = ou = ""
    if r["completed"]:
        me, them = (r["home_score"], r["away_score"]) if home else (r["away_score"], r["home_score"])
        res = f"{'W' if me > them else 'L'} {int(me)}–{int(them)}"
        if pd.notna(my_line):
            v = me + my_line - them
            ats = "Cover" if v > 0 else ("Push" if v == 0 else "No cover")
        if pd.notna(r.get("total")):
            tot = me + them
            ou = "Over" if tot > r["total"] else ("Push" if tot == r["total"] else "Under")
    else:
        upcoming.append(r["event_id"])
    rows.append({
        "Wk": int(r["week"]), "Date": ko.strftime("%b %-d"),
        "Opponent": ("vs " if home else "@ ") + str(opp) + ("" if opp_id in names else " (FCS)"),
        "Result": res or ko.strftime("%a %-I:%M %p"),
        "Line": ui.line(my_line), "vs spread": ats,
        "Total": ui.line(r.get("total"), False), "O/U": ou,
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Lines are DraftKings closing numbers via ESPN, from this team's side.")
if upcoming:
    st.page_link("views/matchup.py", label="Preview the next game →", query_params={"game": str(upcoming[0])})

# ---------------------------------------------------------------- game log
st.subheader("Game by game")
mine = tg[tg["team_id"] == tid]
if len(mine):
    st.plotly_chart(ui.game_log_chart(mine, names), width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------- all stats
with st.expander("Every stat, with national rank"):
    rows = []
    for c, m in meta["stats"].items():
        rk = t.get(f"{c}_rank")
        rows.append({"Stat": m["label"], "Value": ui.fmt(t.get(c), m["fmt"]),
                     "FBS rank": "" if rk is None or pd.isna(rk) else f"{int(rk)} of {n_teams}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
