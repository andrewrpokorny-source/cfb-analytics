import streamlit as st

from cfbsite import ui

snap = ui.snapshot(st.session_state.get("season"))
if snap is None or not snap.get("players"):
    st.error("Player data missing — run `python -m cfbsite.build`.")
    st.stop()
pl, ss, meta = snap["players"], snap["season_stats"], snap["meta"]
games_played = int(ss["games"].max())

st.title("🏃 Player leaders")
st.caption(
    "Expected points added, player by player, parsed from every play of the season. **EPA per play** "
    "measures how much a player's plays raised his team's expected points — it credits a 6-yard "
    "gain on 3rd-and-5 far more than on 3rd-and-15. Garbage time excluded. Quarterbacks are charged "
    "with sacks; scrambles count as runs."
)

c1, c2, c3 = st.columns([2, 2, 2])
sort_label = c1.selectbox("Rank by", ["Total EPA", "EPA per play"])
confs = ["All FBS"] + sorted(ss["conference"].dropna().unique())
conf = c2.selectbox("Conference", confs)
search = c3.text_input("Find a player or team")

tabs = st.tabs([v["label"] for v in ui.PLAYER_VIEWS.values()])
for tab, (kind, v) in zip(tabs, ui.PLAYER_VIEWS.items()):
    with tab:
        df = pl.get(kind)
        if df is None or df.empty:
            st.info("No data.")
            continue
        default_min = max(1, int(v["min_pg"] * games_played * 0.75))
        mn = st.slider(f"Minimum {v['vol']}", 1, int(df[v["vol"]].max()), default_min, key=f"min_{kind}")
        d = df[df[v["vol"]] >= mn]
        if conf != "All FBS":
            d = d[d["team_id"].isin(ss.loc[ss["conference"] == conf, "team_id"])]
        if search:
            q = search.lower()
            d = d[d["player"].str.lower().str.contains(q) | d["team"].str.lower().str.contains(q)]
        tbl, cfg = ui.player_table(d, kind, sort_by="epa" if sort_label == "Total EPA" else "epa_per")
        tbl.insert(0, "#", range(1, len(tbl) + 1))
        st.dataframe(tbl, hide_index=True, width="stretch", column_config=cfg,
                     height=min(36 * (len(tbl) + 1) + 4, 900))
        st.caption(f"{len(d)} players with at least {mn} {v['vol']}.")
