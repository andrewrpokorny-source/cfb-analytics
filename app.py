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
    "leaders": st.Page("views/leaders.py", title="National leaders", icon="📊", url_path="leaders"),
    "betting": st.Page("views/betting.py", title="Betting lab", icon="🧪", url_path="betting"),
}
st.session_state["PAGES"] = PAGES
st.navigation(list(PAGES.values())).run()
st.caption("Data: ESPN (scores, play-by-play, DraftKings lines). Stats computed by this site. "
           "For information and entertainment only — not betting advice.")
