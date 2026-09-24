import pandas as pd
import streamlit as st

from cfbsite import ui

snap = ui.snapshot()

try:
    yr, wk, board = ui.live_week()
except Exception as e:
    st.error(f"Couldn't reach ESPN ({e.__class__.__name__}). Try again shortly.")
    st.stop()

st.title(f"🗓️ {yr} · Week {wk}")
st.caption("Click a game to open its matchup preview. Lines are DraftKings via ESPN, "
           "refreshed every 10 minutes.")

records = {}
if snap is not None:
    ss = snap["season_stats"]
    records = dict(zip(ss["team"], ss["record"]))

c1, c2, c3 = st.columns([2, 2, 3])
conf_filter = c1.selectbox("Games involving", ["All FBS", "Top 25", "Power 4"], index=0)
hide_fcs = c2.toggle("Hide games vs FCS", value=True)
search = c3.text_input("Find a team", placeholder="e.g. Georgia")

b = board.copy()
if hide_fcs and "fbs_vs_fbs" in b:
    b = b[b["fbs_vs_fbs"]]
if conf_filter == "Top 25":
    b = b[(b["home_rank"].fillna(99) <= 25) | (b["away_rank"].fillna(99) <= 25)]
elif conf_filter == "Power 4" and snap is not None:
    p4 = set(snap["teams"].loc[snap["teams"]["conf_abbr"].isin(["sec", "big10", "big12", "acc"]), "team"])
    b = b[b["home"].isin(p4) | b["away"].isin(p4)]
if search:
    s = search.lower()
    b = b[b["home"].str.lower().str.contains(s) | b["away"].str.lower().str.contains(s)]

rows = []
for _, g in b.iterrows():
    ko = pd.to_datetime(g["commence"], utc=True).tz_convert("America/New_York")
    away = f"{ui.rank_label(g.get('away_rank'))}{g['away']}"
    home = f"{ui.rank_label(g.get('home_rank'))}{g['home']}"
    if g.get("state") == "post":
        status = f"Final: {int(g['away_score'])}–{int(g['home_score'])}"
    elif g.get("state") == "in":
        status = f"LIVE {int(g['away_score'])}–{int(g['home_score'])} · {g.get('status', '')}"
    else:
        status = ko.strftime("%a %-I:%M %p ET")
    fav = ""
    sp = g.get("spread")
    if sp is not None and not pd.isna(sp):
        fav = f"{g['home']} {ui.line(sp)}" if sp < 0 else (f"{g['away']} {ui.line(-sp)}" if sp > 0 else "Pick'em")
    rows.append({
        "event_id": g["event_id"], "_ko": ko,
        "When": status,
        "Away": away, "Rec (A)": records.get(g["away"], ""),
        "Home": home + (" (N)" if g.get("neutral") else ""), "Rec (H)": records.get(g["home"], ""),
        "Favorite": fav,
        "Total": ui.line(g.get("total"), signed=False),
        "TV": g.get("tv") or "",
    })

if not rows:
    st.info("No games match those filters.")
    st.stop()

tbl = pd.DataFrame(rows).sort_values("_ko").reset_index(drop=True)
event_ids = tbl.pop("event_id")
tbl = tbl.drop(columns="_ko")
sel = st.dataframe(tbl, hide_index=True, width="stretch", on_select="rerun",
                   selection_mode="single-row", height=min(36 * (len(tbl) + 1) + 4, 1100))

picked = sel.selection.rows if sel and sel.selection else []
if picked:
    st.switch_page("views/matchup.py", query_params={"game": str(event_ids.iloc[picked[0]])})

st.caption(f"{len(tbl)} games shown. Records and stats cover completed weeks "
           f"(updated {snap['meta']['built_at'][:10] if snap else 'n/a'}).")
