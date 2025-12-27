import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="CFB Quant Engine", page_icon="🏈", layout="wide")
st.title("🏈 CFB Quant Engine: Triple Threat Dashboard")

# Load environment variables for local dev
load_dotenv()

@st.cache_data(ttl=300) 
def fetch_scores():
    # 1. Try Local .env
    api_key = os.getenv("CFBD_API_KEY")
    
    # 2. Try Streamlit Secrets (Cloud)
    if not api_key:
        try: api_key = st.secrets["CFBD_API_KEY"]
        except: pass
        
    if not api_key:
        st.error("❌ API Key Not Found. Check .env or secrets.toml")
        return {}

    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # Fetch Regular Season
        res_reg = requests.get("https://api.collegefootballdata.com/games", 
                               headers=headers, params={"year": 2025, "seasonType": "regular"})
        
        # Fetch Postseason
        res_post = requests.get("https://api.collegefootballdata.com/games", 
                                headers=headers, params={"year": 2025, "seasonType": "postseason"})
        
        games_dict = {}
        if res_reg.status_code == 200:
            for g in res_reg.json(): games_dict[str(g['id'])] = g
        if res_post.status_code == 200:
            for g in res_post.json(): games_dict[str(g['id'])] = g
            
        return games_dict
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}

@st.cache_data(ttl=0)
def load_picks():
    try:
        df = pd.read_csv("live_predictions.csv")
        # Ensure GameID is a clean string
        if 'GameID' in df.columns:
            df = df.dropna(subset=['GameID'])
            df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        return df
    except:
        return pd.DataFrame()

# --- MAIN LOGIC ---
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
        
        # CHECK 1: Live API Status
        if game and game.get('status') == 'completed':
            is_completed = True
            home_score = game.get('home_points', 0)
            away_score = game.get('away_points', 0)
            date_str = game.get('start_date', 'N/A')[:10]
            
        # CHECK 2: Manual Backfill Status (Fallback)
        elif 'Manual_HomeScore' in row and pd.notnull(row['Manual_HomeScore']):
            try:
                # Check if score exists (it might be 0, so check not null)
                h = row['Manual_HomeScore']
                if pd.notnull(h) and str(h).strip() != '':
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
            
            # Calculate Margin relative to Home Team
            real_margin = home_score - away_score
            
            # If we picked Home, we need (Margin + Spread) > 0
            if pick_team == row['HomeTeam']:
                adj_margin = real_margin + line
            # If we picked Away, we need (AwayScore - HomeScore - Spread) > 0
            # Or simpler: Did the pick cover?
            # Let's align with standard logic:
            else:
                # Picked Away. 
                # e.g. Away +6.5. Away loses by 6 (-6 margin). -6 + 6.5 = +0.5 (Win).
                # Reverse the margin for Away perspective
                adj_margin = (away_score - home_score) + line

            if adj_margin > 0: spr_res = "WIN"
            elif adj_margin < 0: spr_res = "LOSS"
            else: spr_res = "PUSH"

            # Grade Total
            try: pick_total = float(row.get('Pick_Total', 0))
            except: pick_total = 0.0
            pick_side = row.get('Pick_Side', '')
            tot_score = home_score + away_score
            
            if tot_score == pick_total: tot_res = "PUSH"
            elif pick_side == "OVER": tot_res = "WIN" if tot_score > pick_total else "LOSS"
            elif pick_side == "UNDER": tot_res = "WIN" if tot_score < pick_total else "LOSS"
            else: tot_res = "N/A"

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
            start_str = game.get('start_date') if game else row.get('StartDate')
            
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
        
        su_wins = len(res_df[res_df['SU Result'] == 'WIN'])
        su_loss = len(res_df[res_df['SU Result'] == 'LOSS'])
        su_pct = (su_wins/(su_wins+su_loss)*100) if (su_wins+su_loss) > 0 else 0
        
        s_wins = len(res_df[res_df['Spread Result'] == 'WIN'])
        s_loss = len(res_df[res_df['Spread Result'] == 'LOSS'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Straight Up Record", f"{su_wins}-{su_loss}", f"{su_pct:.1f}%")
        m2.metric("Spread Record", f"{s_wins}-{s_loss}")
        m3.metric("Total Record", f"See Table")

        def color_res(val):
            return 'color: green; font-weight: bold' if val == 'WIN' else 'color: red; font-weight: bold' if val == 'LOSS' else 'color: gray'

        st.divider()
        st.dataframe(res_df[['Date', 'Game', 'Straight Up Pick', 'SU Result', 'Spread Pick', 'Spread Result', 'Total Pick', 'Total Result']].style.map(color_res, subset=['SU Result', 'Spread Result', 'Total Result']), hide_index=True, use_container_width=True)
    else:
        st.info("No history yet.")