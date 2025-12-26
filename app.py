import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone

# --- 1. SETUP & REBRANDING ---
st.set_page_config(page_title="CFB Quant Engine", page_icon="🏈", layout="wide")
st.title("🏈 CFB Quant Engine")

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
    except:
        return {}

@st.cache_data(ttl=0)
def load_picks():
    try:
        df = pd.read_csv("live_predictions.csv")
        # Clean up GameID column
        if 'GameID' in df.columns:
            df = df.dropna(subset=['GameID'])
            df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        return df
    except:
        return pd.DataFrame()

df = load_picks()
scores = fetch_scores()

# --- 2. PROCESSING LOOP ---
graded_results = []
upcoming_games = []

# Get current time in UTC to filter out past games
now_utc = datetime.now(timezone.utc)

if not df.empty:
    for _, row in df.iterrows():
        gid = str(row.get("GameID"))
        game = scores.get(gid)
        
        is_completed = False
        home_score = 0
        away_score = 0
        date_str = "N/A"

        # --- A. DETERMINE STATUS (API vs MANUAL) ---
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
            except: 
                pass

        # --- B. GRADING LOGIC ---
        if is_completed:
            # Spread Result
            pick_team = row['Pick_Team']
            try: raw_home_spread = float(row['Pick_Line'])
            except: raw_home_spread = 0.0
            
            if pick_team == row['HomeTeam']:
                margin = (home_score - away_score) + raw_home_spread
            else:
                margin = (away_score - home_score) - raw_home_spread
            
            if margin == 0: spread_res = "PUSH"
            elif margin > 0: spread_res = "WIN"
            else: spread_res = "LOSS"

            # Total Result
            total_res = "N/A"
            pick_side = row.get('Pick_Side')
            try: pick_total = float(row.get('Pick_Total', 0))
            except: pick_total = 0.0
            
            if pick_total > 0:
                total_score = home_score + away_score
                if total_score == pick_total: total_res = "PUSH"
                elif pick_side == "OVER": total_res = "WIN" if total_score > pick_total else "LOSS"
                else: total_res = "WIN" if total_score < pick_total else "LOSS"

            graded_results.append({
                "Game": f"{row['AwayTeam']} {away_score} - {home_score} {row['HomeTeam']}",
                "Date": date_str,
                "Spread Forecast": f"{row['Spread Pick']}", # Renamed from "Spread Bet"
                "Spread Result": spread_res,
                "Total Forecast": f"{row['Total Pick']}",   # Renamed from "Total Bet"
                "Total Result": total_res
            })
        
        # --- C. UPCOMING GAMES ---
        else:
            new_row = row.copy()
            start_str = None
            if game:
                start_str = game.get('start_date') or game.get('startDate')
            
            show_game = True
            if start_str:
                new_row['Kickoff_Sort'] = start_str
                try:
                    dt = pd.to_datetime(start_str)
                    if dt.tzinfo is None: dt = dt.tz_localize('UTC')
                    if dt < now_utc: show_game = False
                    dt_et = dt.tz_convert('US/Eastern')
                    new_row['Time'] = dt_et.strftime('%a %I:%M %p')
                except: 
                    new_row['Time'] = "Date Error"
            else:
                new_row['Kickoff_Sort'] = "2099-12-31"
                new_row['Time'] = "Date Missing" if game else "TBD"
            
            if show_game:
                new_row['Source'] = str(row.get('Spread Book', '')).replace("DraftKings", "DK").replace("Bovada", "Bov")
                upcoming_games.append(new_row)

# --- 3. UI DISPLAY ---
tab1, tab2 = st.tabs(["🔮 Forecast Board", "📜 Performance History"]) # Renamed Tabs

with tab1:
    st.markdown("### 📅 Active Model Outputs")
    def color_confidence(val):
        try: score = float(val.strip('%'))
        except: return ''
        if score >= 60.0: return 'background-color: #2e7d32; color: white'
        elif score >= 55.0: return 'background-color: #4caf50; color: black'
        elif score <= 52.5: return 'background-color: #ef5350; color: white'
        return ''

    if upcoming_games:
        up_df = pd.DataFrame(upcoming_games)
        if 'Kickoff_Sort' in up_df.columns:
            up_df = up_df.sort_values(by=['Kickoff_Sort', 'Spread_Conf_Raw'], ascending=[True, False])

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Spread Model")
            st.dataframe(up_df[['Time', 'Game', 'Source', 'Spread Pick', 'Spread Conf']].style.map(color_confidence, subset=['Spread Conf']), hide_index=True, use_container_width=True)
        with col2:
            st.caption("Totals Model")
            st.dataframe(up_df[['Time', 'Game', 'Source', 'Total Pick', 'Total Conf']].style.map(color_confidence, subset=['Total Conf']), hide_index=True, use_container_width=True)
    else:
        st.info("No upcoming games found. (All active games are in progress or completed)")

with tab2:
    if graded_results:
        res_df = pd.DataFrame(graded_results)
        res_df = res_df.sort_values(by='Date', ascending=False)
        
        s_wins = len(res_df[res_df['Spread Result'] == 'WIN'])
        s_loss = len(res_df[res_df['Spread Result'] == 'LOSS'])
        s_total = s_wins + s_loss
        s_pct = (s_wins / s_total * 100) if s_total > 0 else 0.0
        
        t_wins = len(res_df[res_df['Total Result'] == 'WIN'])
        t_loss = len(res_df[res_df['Total Result'] == 'LOSS'])
        t_total = t_wins + t_loss
        t_pct = (t_wins / t_total * 100) if t_total > 0 else 0.0

        st.markdown("### 📊 Model Accuracy")
        m1, m2, m3 = st.columns(3)
        m1.metric("Spread Record", f"{s_wins}-{s_loss}", f"{s_pct:.1f}%")
        m2.metric("Total Record", f"{t_wins}-{t_loss}", f"{t_pct:.1f}%")
        m3.metric("Graded Games", len(res_df))
        
        st.divider()
        def color_history(val):
            if val == "WIN": return 'color: green; font-weight: bold'
            if val == "LOSS": return 'color: red; font-weight: bold'
            return 'color: gray'

        st.dataframe(res_df.style.map(color_history, subset=['Spread Result', 'Total Result']), hide_index=True, use_container_width=True)
    else:
        st.info("History will populate as games finish.")
