import pandas as pd
import streamlit as st

from cfbsite import ui

snap = ui.snapshot()
if snap is None:
    st.error("Stats snapshot missing — run `python -m cfbsite.build`.")
    st.stop()
meta, ss = snap["meta"], snap["season_stats"]
prm = meta.get("ratings", {})

st.title("⚡ Power ratings")
st.caption(
    "Opponent-adjusted efficiency. Each offense's EPA per play is adjusted for the defenses it "
    "faced, and each defense for the offenses it faced, with a home-field term. **Power** is "
    "points better than an average FBS team on a neutral field. Early in the season the ratings "
    "lean on a preseason prior (last season's efficiency and recruiting talent), and the "
    "season's own games take over as they accumulate."
)

c1, c2 = st.columns([2, 3])
conf = c1.selectbox("Highlight a conference", ["None"] + sorted(ss["conference"].dropna().unique()))
st.plotly_chart(ui.efficiency_map(ss, conf=None if conf == "None" else conf), width="stretch",
                config={"displayModeBar": False})

d = ss.sort_values("power_rank")
if conf != "None":
    d = d[d["conference"] == conf]
tbl = pd.DataFrame({
    "Rank": d["power_rank"].astype(int),
    "Logo": d["logo"],
    "Team": d["team"],
    "Conf": d["conf_abbr"].str.upper(),
    "Record": d["record"],
    "Power": d["power"].round(1),
    "Adj. off": [f"{v:+.3f} ({int(r)})" for v, r in zip(d["adj_off"], d["adj_off_rank"])],
    "Adj. def": [f"{v:+.3f} ({int(r)})" for v, r in zip(d["adj_def"], d["adj_def_rank"])],
    "SOS": [f"{v:+.1f} ({int(r)})" for v, r in zip(d["sos"], d["sos_rank"])],
    "team_id": d["team_id"],
})
ids = tbl.pop("team_id")
sel = st.dataframe(
    tbl, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    height=min(36 * (len(tbl) + 1) + 4, 1100),
    column_config={
        "Logo": st.column_config.ImageColumn("", width="small"),
        "Power": st.column_config.ProgressColumn(
            "Power (pts)", min_value=float(ss["power"].min()), max_value=float(ss["power"].max()),
            format="%+.1f"),
    })
if sel and sel.selection and sel.selection.rows:
    st.switch_page("views/team.py", query_params={"team": str(ids.iloc[sel.selection.rows[0]])})

with st.expander("How good are these ratings?"):
    st.markdown(
        f"""
- **Built from:** every non-garbage-time play of the season, scored with an expected-points
  model trained on ~1M plays from 2018–25 (agrees with CFBD's published EPA at r = 0.82).
- **Against the betting market:** on upcoming games, the spreads these ratings imply correlate
  **r ≈ 0.84** with the DraftKings line. The market is still more accurate: on completed games the
  ratings miss the final margin by about 12 points on average, against 11 for the closing line.
  Treat the model line as context, not a pick.
- **Home field** is estimated from the data (currently ≈{2 * prm.get('hfa', 0) * prm.get('pts_b', 0) / 2:.1f}
  pts per side, {2 * prm.get('hfa', 0) * prm.get('pts_b', 0):.1f} total swing) and is inflated early in
  the season by home games against FCS teams.
- **Numbers in parentheses** are national ranks among {len(ss)} FBS teams. SOS is the average power
  rating of opponents played, with FCS opponents counted at {prm.get('fcs_power', 0):+.1f}.
"""
    )
