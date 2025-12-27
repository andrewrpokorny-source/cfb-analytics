import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="CFB Quant Engine", page_icon="🏈", layout="wide")
st.title("🏈 CFB Quant Engine: Triple Threat Dashboard")

load_dotenv()

# --- UTILS ---
@st.cache_data(ttl=300) 
def fetch_scores():
    # Fetch live scores to grade pending games if they finish
    api_key = os.getenv("CFBD_API_KEY") or st.secrets.get("CFBD_API_KEY")
    if not api_key: return {}

    headers = {"Authorization": f"Bearer {api_key}"}
    games_dict = {}
    
    try:
        # Check Regular Season & Postseason
        for p in [{"year": 2025, "seasonType": "regular"}, {"year": 2025, "seasonType": "postseason"}]:
            res = requests.get("https://api.collegefootballdata.com/games", headers=headers, params=p)
            if res.status_code == 200:
                for g in res.json(): games_dict[str(g['id'])] = g
        return games_dict
    except: return {}

@st.cache_data(ttl=0)
def load_data():
    try:
        df = pd.read_csv("live_predictions.csv")
        # Normalize GameID
        if 'GameID' in df.columns:
            df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        return df
    except: return pd.DataFrame()

def color_conf(val):
    # Highlight high confidence
    try:
        score = float(str(val).strip('%'))
        if score >= 60: return 'background-color: #1b5e20; color: white' 
        if score >= 55: return 'background-color: #4caf50; color: black' 
    except: pass
    return ''

def color_result(val):
    if val == 'WIN': return 'color: #2e7d32; font-weight: bold'
    if val == 'LOSS': return 'color: #d32f2f; font-weight: bold'
    return 'color: gray'

# --- MAIN LOGIC ---
df = load_data()
scores = fetch_scores()

if df.empty:
    st.warning("⚠️ No predictions found. (live_predictions.csv is empty)")
    st.stop()

graded = []
upcoming = []

for _, row in df.iterrows():
    gid = str(row.get("GameID"))
    api_data = scores.get(gid, {})
    
    is_completed = False
    h_score = 0
    a_score = 0
    
    # Check Completion (API or Manual)
    if api_data.get('status') == 'completed':
        is_completed = True
        h_score = api_data.get('home_points', 0)
        a_score = api_data.get('away_points', 0)
    elif pd.notnull(row.get('Manual_HomeScore')):
        is_completed = True
        h_score = row['Manual_HomeScore']
        a_score = row['Manual_AwayScore']

    if is_completed:
        # --- GRADING ---
        winner = row['HomeTeam'] if h_score > a_score else row['AwayTeam']
        
        # Grade Spread
        spread_res = "PUSH"
        if pd.notnull(row.get('Pick_Line')):
            pick_team = row.get('Pick_Team')
            line = float(row.get('Pick_Line', 0))
            if pick_team == row['HomeTeam']: margin = (h_score - a_score) + line
            else: margin = (a_score - h_score) + line
            if margin > 0: spread_res = "WIN"
            elif margin < 0: spread_res = "LOSS"

        # Grade Total
        total_res = "PUSH"
        if pd.notnull(row.get('Pick_Total')):
            actual_total = h_score + a_score
            target = float(row['Pick_Total'])
            side = row.get('Pick_Side')
            if side == 'OVER': total_res = "WIN" if actual_total > target else "LOSS"
            elif side == 'UNDER': total_res = "WIN" if actual_total < target else "LOSS"
            if actual_total == target: total_res = "PUSH"

        graded.append({
            "Date": row.get('StartDate', 'N/A')[:10],
            "Game": f"{row['AwayTeam']} {a_score} - {h_score} {row['HomeTeam']}",
            "Pick (SU)": row.get('Moneyline Pick'),
            "Res (SU)": "WIN" if row.get('Moneyline Pick') == winner else "LOSS",
            "Pick (Spr)": row.get('Spread Pick'),
            "Res (Spr)": spread_res,
            "Pick (Tot)": row.get('Total Pick'),
            "Res (Tot)": total_res
        })
        
    else:
        # --- UPCOMING ---
        # NO FILTERING: We show everything that isn't graded.
        ts = row.get('StartDate')
        try:
            dt = pd.to_datetime(ts)
            if dt.tzinfo is None: dt = dt.tz_localize('UTC')
            fmt_time = dt.tz_convert('US/Eastern').strftime('%a %I:%M %p')
        except: fmt_time = "TBD"
        
        row['Time'] = fmt_time
        upcoming.append(row)

# --- DISPLAY ---
t1, t2 = st.tabs(["🔮 Forecast Board", "📜 Performance History"])

with t1:
    if upcoming:
        up_df = pd.DataFrame(upcoming)
        # Sort by date (Ascending) so today/tomorrow are at the top
        if 'StartDate' in up_df.columns: 
            up_df = up_df.sort_values('StartDate', ascending=True)
        
        st.markdown(f"### 📅 Upcoming Schedule ({len(up_df)} Games)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 🏆 Straight Up")
            st.dataframe(up_df[['Time', 'Game', 'Moneyline Pick', 'Moneyline Conf']].style.map(color_conf, subset=['Moneyline Conf']), hide_index=True)
        with c2:
            st.markdown("#### ⚖️ Spread")
            st.dataframe(up_df[['Game', 'Spread Pick', 'Spread Conf']].style.map(color_conf, subset=['Spread Conf']), hide_index=True)
        with c3:
            st.markdown("#### ↕️ Total")
            st.dataframe(up_df[['Game', 'Total Pick', 'Total Conf']].style.map(color_conf, subset=['Total Conf']), hide_index=True)
    else:
        st.info("No upcoming games found in the database.")

with t2:
    if graded:
        res_df = pd.DataFrame(graded)
        wins = len(res_df[res_df['Res (Spr)'] == 'WIN'])
        losses = len(res_df[res_df['Res (Spr)'] == 'LOSS'])
        st.metric("Spread Record", f"{wins}-{losses}")
        st.dataframe(res_df.style.map(color_result, subset=['Res (SU)', 'Res (Spr)', 'Res (Tot)']), hide_index=True, use_container_width=True)
    else:
        st.info("No graded history yet.")