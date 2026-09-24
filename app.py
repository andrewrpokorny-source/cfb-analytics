"""CFB Stats — college football team pages, matchup previews, and the betting lab.

Deployed on Streamlit Community Cloud from this repo. Reads stats snapshots in
site_data/ (built by `python -m cfbsite.build`) plus live ESPN data. Needs no
API keys. Public: no password (removed 2026-09-23 at the owner's request).
"""
import streamlit as st

st.set_page_config(page_title="CFB Stats", page_icon="🏈", layout="wide")


PAGES = {
    "home": st.Page("views/home.py", title="This week", icon="🗓️", default=True),
    "matchup": st.Page("views/matchup.py", title="Matchup preview", icon="⚔️", url_path="matchup"),
    "teams": st.Page("views/team.py", title="Teams", icon="🏈", url_path="team"),
    "ratings": st.Page("views/ratings.py", title="Power ratings", icon="⚡", url_path="ratings"),
    "players": st.Page("views/players.py", title="Player leaders", icon="🏃", url_path="players"),
    "leaders": st.Page("views/leaders.py", title="National leaders", icon="📊", url_path="leaders"),
    "betting": st.Page("views/betting.py", title="Betting lab", icon="🧪", url_path="betting"),
}
st.session_state["PAGES"] = PAGES

# ---- season picker (drives Teams, Power ratings, Players, Leaders) ----
import os
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_data")
seasons = sorted((int(x) for x in os.listdir(_base) if x.isdigit()), reverse=True) if os.path.isdir(_base) else []
if seasons:
    q = st.query_params.get("season")
    default = int(q) if q and q.isdigit() and int(q) in seasons else seasons[0]
    season = st.sidebar.selectbox("Season", seasons, index=seasons.index(default))
    st.session_state["season"] = season
    if season != seasons[0]:
        st.query_params["season"] = str(season)
        st.sidebar.caption(f"Viewing {season}. *This week* and *Matchup preview* always show the "
                           "current season.")
    elif "season" in st.query_params:
        del st.query_params["season"]

nav = st.navigation(list(PAGES.values()))
nav.run()
st.caption("Data: ESPN (scores, play-by-play, DraftKings lines). Stats computed by this site. "
           "For information and entertainment only — not betting advice.")
