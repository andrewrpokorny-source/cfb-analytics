import pandas as pd
import streamlit as st

from cfbsite import ui

snap = ui.snapshot(st.session_state.get("season"))
if snap is None:
    st.error("Stats snapshot missing — run `python -m cfbsite.build`.")
    st.stop()
meta, ss = snap["meta"], snap["season_stats"]
ranked = {c: m for c, m in meta["stats"].items() if m["higher_better"] is not None}

st.title("📊 National leaders")
c0, c1, c2, c3 = st.columns([2, 3, 2, 2])
cats = sorted({m.get("cat", "Other") for m in ranked.values()}, key=lambda c: ["Ratings", "Offense", "Defense", "Drives", "Results"].index(c) if c in ["Ratings", "Offense", "Defense", "Drives", "Results"] else 9)
cat = c0.selectbox("Category", cats)
opts = [c for c, m in ranked.items() if m.get("cat") == cat]
stat = c1.selectbox("Stat", opts, format_func=lambda c: ranked[c]["label"])
confs = ["All FBS"] + sorted(ss["conference"].dropna().unique())
conf = c2.selectbox("Conference", confs)
worst = c3.toggle("Show worst first", value=False)

d = ss if conf == "All FBS" else ss[ss["conference"] == conf]
d = d.sort_values(f"{stat}_rank", ascending=not worst)
m = ranked[stat]
tbl = pd.DataFrame({
    "Rank": d[f"{stat}_rank"].astype(int),
    "Logo": d["logo"],
    "Team": d["team"],
    "Conf": d["conf_abbr"].str.upper(),
    "Record": d["record"],
    m["label"]: [ui.fmt(v, m["fmt"]) for v in d[stat]],
    "Percentile": d[f"{stat}_pctl"].round(0),
    "team_id": d["team_id"],
})
ids = tbl.pop("team_id")
sel = st.dataframe(
    tbl, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    height=min(36 * (len(tbl) + 1) + 4, 1100),
    column_config={
        "Logo": st.column_config.ImageColumn("", width="small"),
        "Percentile": st.column_config.ProgressColumn("FBS percentile", min_value=0, max_value=100, format="%d"),
    })
if sel and sel.selection and sel.selection.rows:
    st.switch_page("views/team.py", query_params={"team": str(ids.iloc[sel.selection.rows[0]])})
st.caption(f"{'Lower is better' if not m['higher_better'] else 'Higher is better'}. "
           f"Rank is among all {len(ss)} FBS teams, even when filtered to a conference. "
           "Click a team to open its page. Garbage time excluded from per-play stats.")
