"""CFB Stats — college football team pages, matchup previews, and the betting lab.

Deployed on Streamlit Community Cloud from this repo. Reads stats snapshots in
site_data/ (built by `python -m cfbsite.build`) plus live ESPN data. Needs no
API keys. The password lives in Streamlit Cloud app settings (Secrets).
"""
import hmac

import streamlit as st

st.set_page_config(page_title="CFB Stats", page_icon="🏈", layout="wide")


def _password():
    try:
        expected = st.secrets.get("password")
    except Exception:
        expected = None
    if not expected:
        st.title("🔒 Not configured")
        st.error("No password is set. Add `password = \"...\"` under "
                 "App settings → Secrets in Streamlit Community Cloud.")
        return False
    if st.session_state.get("authed"):
        return True

    def entered():
        ok = hmac.compare_digest(st.session_state.get("pw", ""), str(expected))
        st.session_state["authed"] = ok
        st.session_state.pop("pw", None)

    st.title("🔒 CFB Stats")
    st.text_input("Password", type="password", on_change=entered, key="pw")
    if st.session_state.get("authed") is False:
        st.error("Incorrect password")
    return False


if not _password():
    st.stop()

PAGES = {
    "home": st.Page("views/home.py", title="This week", icon="🗓️", default=True),
    "matchup": st.Page("views/matchup.py", title="Matchup preview", icon="⚔️", url_path="matchup"),
    "teams": st.Page("views/team.py", title="Teams", icon="🏈", url_path="team"),
    "leaders": st.Page("views/leaders.py", title="National leaders", icon="📊", url_path="leaders"),
    "betting": st.Page("views/betting.py", title="Betting lab", icon="🧪", url_path="betting"),
}
st.session_state["PAGES"] = PAGES
st.navigation(list(PAGES.values())).run()
