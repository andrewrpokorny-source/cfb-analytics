import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone

# --- CONFIG ---
st.set_page_config(page_title="CFB Quant Engine", page_icon="🏈", layout="wide")
st.title("🏈 CFB Quant Engine: Triple Threat Dashboard")

@st.cache_data(ttl=300) 
def fetch_scores():
    try:
        api_key = st.secrets["CFBD_API_KEY"]
        headers = {"Authorization": f"Bearer {api_key}"}
        res_reg = requests.get("https://api.collegefootballdata.com/games", 
                               headers=headers, params={"year": 2025, "seasonType": "regular"})
        res_post = requests.get("https://api.collegefootballdata.com/games", 
                                headers=headers, params={"year": 2025, "seasonType": "postseason"})
        games_dict = {}
        if res_reg.status_code == 200:
            for g in res_reg.json(): games_dict[str(g['id'])] = g
        if res_post.status_code == 200:
            for g in res_post.json(): games_dict[str(g['id'])] = g
        return games_dict
    except: return {}

@st.cache_data(ttl=0)
def load_picks():
    try:
        df = pd.read_csv("live_predictions.csv")
        if 'GameID' in df.columns:
            df = df.dropna(subset=['GameID'])
            df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        return df
    except: return pd.DataFrame()

df = load_picks()
scores = fetch_scores()

graded_results = []
upcoming_games = []
now_utc = datetime.now(timezone.utc)

if not df.empty:
    for _, row in df.iterrows():
        gid = str(row.get("GameID"))
        game = scores.get(gid)
        
        is_completed = False
        home_score, away_score = 0, 0
        date_str = "N/A"
        
        if game and game.get('status') == 'completed':
            is_completed = True
            home_score = game.get('home_points', 0)
            away_score = game.get('away_points', 0)
            date_str = game.get('start_date', 'N/A')[:10]
        elif 'Manual_HomeScore' in row and pd.notnull(row['Manual_HomeScore']):
            try:
                if float(row['Manual_HomeScore']) >= 0:
                    is_completed = True
                    home_score = int(float(row['Manual_HomeScore']))
                    away_score = int(float(row['Manual_AwayScore']))
                    date_str = str(row.get('Manual_Date', 'N/A'))
            except: pass

        # --- A. HISTORY GRADING ---
        if is_completed:
            winner = row['HomeTeam'] if home_score > away_score else row['AwayTeam']
            
            # Grade Spread
            pick_team = row.get('Pick_Team', '')
            try: line = float(row.get('Pick_Line', 0))
            except: line = 0.0
            margin = (home_score - away_score) if pick_team == row['HomeTeam'] else (away_score - home_score)
            diff = margin + line
            spr_res = "WIN" if diff > 0 else "LOSS" if diff < 0 else "PUSH"

            # Grade Total
            try: pick_total = float(row.get('Pick_Total', 0))
            except: pick_total = 0.0
            pick_side = row.get('Pick_Side', '')
            tot_score = home_score + away_score
            if tot_score == pick_total: tot_res = "PUSH"
            elif pick_side == "OVER": tot_res = "WIN" if tot_score > pick_total else "LOSS"
            else: tot_res = "WIN" if tot_score < pick_total else "LOSS"

            # Grade Straight Up
            ml_pick = row.get('Moneyline Pick', 'N/A')
            ml_res = "WIN" if ml_pick == winner else "LOSS" if ml_pick != 'N/A' else "N/A"

            graded_results.append({
                "Date": date_str,
                "Game": f"{row['AwayTeam']} {away_score} - {home_score} {row['HomeTeam']}",
                "Winner": winner,
                "Straight Up Pick": ml_pick,
                "SU Result": ml_res,
                "Spread Pick": row['Spread Pick'],
                "Spread Result": spr_res,
                "Total Pick": row['Total Pick'],
                "Total Result": tot_res
            })

        # --- B. UPCOMING GAMES ---
        else:
            new_row = row.copy()
            # 1. Try Live API Date
            start_str = game.get('start_date') if game else None
            # 2. Fallback to Database Date
            if not start_str:
                start_str = row.get('StartDate')
            
            show_game = True
            if start_str and pd.notnull(start_str):
                new_row['Kickoff_Sort'] = start_str
                try:
                    dt = pd.to_datetime(start_str)
                    if dt.tzinfo is None: dt = dt.tz_localize('UTC')
                    if dt < now_utc: show_game = False
                    new_row['Time'] = dt.tz_convert('US/Eastern').strftime('%a %I:%M %p')
                except: new_row['Time'] = "Err"
            else:
                new_row['Kickoff_Sort'] = "2099-12-31"
                new_row['Time'] = "TBD"
            
            if show_game:
                new_row['Source'] = str(row.get('Spread Book', 'DK')).replace("DraftKings", "DK")
                upcoming_games.append(new_row)

# --- DISPLAY ---
tab1, tab2 = st.tabs(["🔮 Forecast Board", "📜 Performance History"])

def color_conf(val):
    try: score = float(val.strip('%'))
    except: return ''
    if score >= 60: return 'background-color: #2e7d32; color: white'
    if score >= 55: return 'background-color: #4caf50; color: black'
    return ''

with tab1:
    st.markdown("### 📅 Three-Panel Forecast")
    if upcoming_games:
        up_df = pd.DataFrame(upcoming_games)
        if 'Kickoff_Sort' in up_df.columns:
            up_df = up_df.sort_values(by=['Kickoff_Sort'], ascending=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 🏆 Straight Up Winner")
            st.dataframe(up_df[['Time', 'Game', 'Moneyline Pick', 'Moneyline Conf']].style.map(color_conf, subset=['Moneyline Conf']), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("#### ⚖️ Spread")
            st.dataframe(up_df[['Game', 'Spread Pick', 'Spread Conf']].style.map(color_conf, subset=['Spread Conf']), hide_index=True, use_container_width=True)
        with c3:
            st.markdown("#### ↕️ Total")
            st.dataframe(up_df[['Game', 'Total Pick', 'Total Conf']].style.map(color_conf, subset=['Total Conf']), hide_index=True, use_container_width=True)

    else:
        st.info("No active games found.")

with tab2:
    if graded_results:
        res_df = pd.DataFrame(graded_results)
        res_df = res_df.sort_values(by='Date', ascending=False)
        su_wins = len(res