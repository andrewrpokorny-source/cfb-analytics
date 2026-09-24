import pandas as pd
import streamlit as st

from cfbsite import ui

snap = ui.snapshot()
if snap is None:
    st.error("Stats snapshot missing — run `python -m cfbsite.build`.")
    st.stop()
meta, ss, tg, teams = snap["meta"], snap["season_stats"], snap["team_games"], snap["teams"]
n_teams = len(ss)
by_id = ss.set_index("team_id")
names = dict(zip(teams["team_id"], teams["team"]))

# ---------------------------------------------------------------- pick a game
try:
    yr, wk, board = ui.live_week()
except Exception:
    board = pd.DataFrame()

game_id = st.query_params.get("game")
if not board.empty:
    opts = board.sort_values("commence")
    labels = {r["event_id"]: f"{ui.rank_label(r.get('away_rank'))}{r['away']} @ "
                             f"{ui.rank_label(r.get('home_rank'))}{r['home']}"
              for _, r in opts.iterrows()}
    ids = list(labels)
    idx = ids.index(game_id) if game_id in ids else 0
    choice = st.selectbox("Game", ids, index=idx, format_func=lambda i: labels[i])
    if choice != game_id:
        st.query_params["game"] = choice
        game_id = choice
if not game_id:
    st.info("Pick a game from **This week**.")
    st.stop()

s = ui.live_summary(game_id)
comp = s["header"]["competitions"][0]
side = {c["homeAway"]: c for c in comp["competitors"]}
H, A = side["home"], side["away"]
hid, aid = str(H["team"]["id"]), str(A["team"]["id"])


def team_bits(c):
    t = c["team"]
    rec = next((r.get("summary") for r in c.get("record", []) if r.get("type") in ("total", "overall")), "")
    return {"name": t.get("location") or t.get("displayName"), "abbr": t.get("abbreviation") or t.get("location"), "logo": (t.get("logos") or [{}])[0].get("href"),
            "rank": c.get("rank"), "record": rec}


hb, ab = team_bits(H), team_bits(A)
status = comp.get("status", {}).get("type", {})
ko = pd.to_datetime(comp.get("date"), utc=True).tz_convert("America/New_York")
venue = (s.get("gameInfo") or {}).get("venue", {}).get("fullName", "")

# ---------------------------------------------------------------- header
l, mid, r = st.columns([3, 2, 3])
with l:
    if ab["logo"]:
        st.image(ab["logo"], width=84)
    st.markdown(f"### {ui.rank_label(ab['rank'])}{ab['name']}")
    st.caption(f"{ab['record']} · away")
with mid:
    st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1.5rem'>@</div>",
                unsafe_allow_html=True)
    if status.get("state") in ("in", "post"):
        st.markdown(f"<div style='text-align:center;font-size:1.6rem'><b>{A.get('score','')}–{H.get('score','')}</b></div>"
                    f"<div style='text-align:center'>{status.get('shortDetail','')}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:center'>{ko:%a %b %-d · %-I:%M %p ET}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;color:gray;font-size:.85rem'>{venue}</div>", unsafe_allow_html=True)
with r:
    if hb["logo"]:
        st.image(hb["logo"], width=84)
    st.markdown(f"### {ui.rank_label(hb['rank'])}{hb['name']}")
    st.caption(f"{hb['record']} · home")

# ---------------------------------------------------------------- the line
odds = (s.get("pickcenter") or [{}])[0] if s.get("pickcenter") else {}
row_live = board[board["event_id"] == game_id].iloc[0] if not board.empty and (board["event_id"] == game_id).any() else None
if row_live is not None and row_live.get("spread") is not None and not pd.isna(row_live.get("spread")):
    g = row_live
    sp = g["spread"]
    fav = f"{hb['name']} {ui.line(sp)}" if sp < 0 else (f"{ab['name']} {ui.line(-sp)}" if sp > 0 else "Pick'em")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spread", fav, None if pd.isna(g.get("spread_open")) or g["spread_open"] == sp
              else f"opened {ui.line(g['spread_open'])} ({hb['name']})", delta_color="off")
    c2.metric("Total", ui.line(g.get("total"), False), None if pd.isna(g.get("total_open")) or g["total_open"] == g["total"]
              else f"opened {ui.line(g['total_open'], False)}", delta_color="off")
    c3.metric("Moneyline", f"{ab['abbr']} {ui.price(g.get('away_ml'))} / {hb['abbr']} {ui.price(g.get('home_ml'))}")
    st.caption("DraftKings via ESPN. Implied score from the line: "
               + (f"{ab['name']} {(g['total'] + sp) / 2:.1f}, {hb['name']} {(g['total'] - sp) / 2:.1f}."
                  if not pd.isna(g.get("total")) else "n/a"))

if hid not in by_id.index or aid not in by_id.index:
    st.info("One of these teams isn't FBS (or hasn't played yet), so there's no stat profile to compare.")
    st.stop()
h, a = by_id.loc[hid].to_dict(), by_id.loc[aid].to_dict()

# ---------------------------------------------------------------- model projection
from cfbsite import ratings as rmod
prm = meta.get("ratings", {})
if prm and not any(pd.isna(x.get("adj_off_raw")) for x in (h, a)):
    neutral = bool((s.get("header", {}).get("competitions", [{}])[0]).get("neutralSite"))
    hp, ap = rmod.project({"adj_off": h["adj_off_raw"], "adj_def": h["adj_def_raw"]},
                          {"adj_off": a["adj_off_raw"], "adj_def": a["adj_def_raw"]}, prm, neutral=neutral)
    margin = hp - ap
    wp_home = ui.win_prob(margin)
    st.subheader("Model projection")
    p1, p2, p3 = st.columns(3)
    p1.metric("Projected score", f"{ab['abbr']} {ap:.0f} – {hb['abbr']} {hp:.0f}")
    fav_name, fav_by = (hb["name"], margin) if margin >= 0 else (ab["name"], -margin)
    mkt = None
    if row_live is not None and pd.notna(row_live.get("spread")):
        mkt = row_live["spread"]
    p2.metric("Model line", f"{fav_name} -{fav_by:.1f}",
              None if mkt is None else f"market: {hb['name'] if mkt < 0 else ab['name']} {-abs(mkt):+g}",
              delta_color="off")
    p3.metric("Win probability", f"{hb['abbr']} {wp_home:.0%} / {ab['abbr']} {1 - wp_home:.0%}")
    st.caption("From opponent-adjusted power ratings (see Power ratings). The betting market is more "
               "accurate than this model — on completed games it misses by ~11 points vs ~12 — so "
               "read a big gap as \"the market knows something the stats don't\", not as a pick.")

# ---------------------------------------------------------------- tale of the tape
st.subheader("Tale of the tape")
tape = ["record", "power", "adj_off", "adj_def", "sos", "o_epa", "d_epa", "o_sr", "d_sr",
        "o_pts_opp", "d_pts_opp", "margin_pg", "to_margin_pg", "penalty_yds_pg"]
rows = []
for c in tape:
    if c == "record":
        rows.append({"Stat": "Record", ab["name"]: a["record"], hb["name"]: h["record"], "Edge": ""})
        continue
    m = meta["stats"][c]
    ar, hr = a.get(f"{c}_rank"), h.get(f"{c}_rank")
    edge = "" if pd.isna(ar) or pd.isna(hr) or ar == hr else (ab["name"] if ar < hr else hb["name"])
    rows.append({"Stat": m["label"], ab["name"]: ui.stat_cell(a, c, meta),
                 hb["name"]: ui.stat_cell(h, c, meta), "Edge": edge})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"National rank among {n_teams} FBS teams in parentheses. Through {int(max(a['games'], h['games']))} games; "
           "early-season numbers include games against FCS opponents.")

# ---------------------------------------------------------------- unit vs unit
st.subheader("Unit vs unit")
st.caption("Percentile vs all FBS (100 = best). A long offense bar against a short defense bar is a mismatch.")
t1, t2 = st.tabs([f"When {ab['name']} has the ball", f"When {hb['name']} has the ball"])
with t1:
    st.plotly_chart(ui.matchup_bars(a, h, ab["name"], hb["name"], ui.BLUE, ui.ORANGE, meta),
                    width="stretch", config={"displayModeBar": False})
with t2:
    st.plotly_chart(ui.matchup_bars(h, a, hb["name"], ab["name"], ui.ORANGE, ui.BLUE, meta),
                    width="stretch", config={"displayModeBar": False})

# biggest mismatches, in words
edges = []
for oc, dc in ui.MATCH:
    for off, dfn, on, dn in ((a, h, ab["name"], hb["name"]), (h, a, hb["name"], ab["name"])):
        op, dp = off.get(f"{oc}_pctl"), dfn.get(f"{dc}_pctl")
        if op is None or dp is None or pd.isna(op) or pd.isna(dp):
            continue
        edges.append((op - dp, on, dn, ui.MATCH_LABEL[oc], op, dp))
edges.sort(key=lambda x: -abs(x[0]))
st.markdown("**Biggest mismatches**")
for gap, on, dn, lab, op, dp in edges[:4]:
    who = f"{on} offense" if gap > 0 else f"{dn} defense"
    st.markdown(f"- **{lab}:** edge **{who}** — {on} offense {ui.ordinal(round(op))} percentile vs {dn} defense {ui.ordinal(round(dp))}")

# ---------------------------------------------------------------- recent form
st.subheader("Recent form")
f1, f2 = st.columns(2)
for col, tid, nm in ((f1, aid, ab["name"]), (f2, hid, hb["name"])):
    with col:
        st.markdown(f"**{nm}**")
        g = tg[tg["team_id"] == tid].sort_values("week", ascending=False)
        st.dataframe(pd.DataFrame({
            "Wk": g["week"].astype(int),
            "Opponent": [("vs " if ha == "home" else "@ ") + names.get(o, "FCS opp.") for o, ha in zip(g["opp_id"], g["home_away"])],
            "Result": [f"{'W' if w else 'L'} {int(p)}–{int(q)}" for w, p, q in zip(g["win"], g["points"], g["opp_points"])],
            "Off EPA/play": (g["o_epa"] / g["o_epa_n"]).round(3),
            "Def EPA/play": (g["d_epa"] / g["d_epa_n"]).round(3),
            "Success": (100 * g["o_success"] / g["o_succ_n"]).round(1).astype(str) + "%",
        }), hide_index=True, width="stretch")

cc1, cc2 = st.columns(2)
cc1.page_link("views/team.py", label=f"Full {ab['name']} profile →", query_params={"team": aid})
cc2.page_link("views/team.py", label=f"Full {hb['name']} profile →", query_params={"team": hid})
