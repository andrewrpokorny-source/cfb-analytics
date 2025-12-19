import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="CFB Quant Engine", page_icon="🏈", layout="wide")
st.title("🏈 CFB Algorithmic Betting Engine")

# --- 1. SETUP & DATA LOADING ---
@st.cache_data(ttl=300) 
def fetch_scores():
    try:
        api_key = st.secrets["CFBD_API_KEY"]
        headers = {"Authorization": f"Bearer {api_key}"}
        res = requests.get("https://api.collegefootballdata.com/games", 
                           headers=headers, 
                           params={"year": 2025, "seasonType": "postseason"})
        if res.status_code == 200:
            return {str(g['id']): g for g in res.json()}
    except:
        pass
    return {}

@st.cache_data(ttl=0)
def load_picks():
    try:
        df = pd.read_csv("live_predictions.csv")
        # Clean IDs
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

if not df.empty:
    for _, row in df.iterrows():
        gid = str(row.get("GameID"))
        game = scores.get(gid)
        
        # --- HYBRID SCORE LOOKUP ---
        # 1. Try API first
        if game and game.get('status') == 'completed':
            home_score = game.get('home_points', 0)
            away_score = game.get('away_points', 0)
            date_str = game.get('start_date', 'N/A')[:10]
            is_completed = True
        
        # 2. Try Manual Backfill (from CSV columns)
        elif 'Manual_HomeScore' in row and pd.notnull(row['Manual_HomeScore']):
            home_score = int(row['Manual_HomeScore'])
            away_score = int(row['Manual_AwayScore'])
            date_str = str(row['Manual_Date'])
            is_completed = True
        
        else:
            is_completed = False

        # --- GRADING LOGIC ---
        if is_completed:
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

            # Total Logic (Skip for manual backfill if missing)
            total_res = "N/A"
            if 'Manual_HomeScore' not in row or pd.isnull(row['Manual_HomeScore']):
                total_score = home_score + away_score
                pick_side = row['Pick_Side'] 
                try: pick_total = float(row['Pick_Total'])
                except: pick_total = 0.0
                
                if total_score == pick_total: total_res = "PUSH"
                elif pick_side == "OVER": total_res = "WIN" if total_score > pick_total else "LOSS"
                else: total_res = "WIN" if total_score < pick_total else "LOSS"

            graded_results.append({
                "Game": f"{row['AwayTeam']} {away_score} - {home_score} {row['HomeTeam']}",
                "Date": date_str,
                "Spread Bet": f"{row['Spread Pick']}",
                "Spread Result": spread_res,
                "Total Bet": f"{row['Total Pick']}",
                "Total Result": total_res
            })
        
        # --- UPCOMING ---
        else:
            new_row = row.copy()
            start_str = game.get('start_date') if game else None
            
            if start_str:
                new_row['Kickoff_Sort'] = start_str
                try:
                    dt = pd.to_datetime(start_str)
                    if dt.tzinfo is None: dt = dt.tz_localize('UTC')
                    dt_et = dt.tz_convert('US/Eastern')
                    new_row['Time'] = dt_et.strftime('%a %I:%M %p')
                except:
                    new_row['Time'] = "Date Error"
            else:
                new_row['Kickoff_Sort'] = "2099-12-31"
                new_row['Time'] = "TBD"
            upcoming_games.append(new_row)

# --- 3. UI ---
tab1, tab2 = st.tabs(["🔮 Betting Board", "📜 Performance History"])

with tab1:
    st.markdown("### 📅 Active & Upcoming Games")
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
            st.caption("Spread Picks")
            st.dataframe(up_df[['Time', 'Game', 'Spread Pick', 'Spread Book', 'Spread Conf']].style.map(color_confidence, subset=['Spread Conf']), hide_index=True, use_container_width=True)
        with col2:
            st.caption("Totals Picks")
            st.dataframe(up_df[['Time', 'Game', 'Total Pick', 'Total Book', 'Total Conf']].style.map(color_confidence, subset=['Total Conf']), hide_index=True, use_container_width=True)
    else:
        st.info("No upcoming games found.")

with tab2:
    if graded_results:
        res_df = pd.DataFrame(graded_results)
        res_df = res_df.sort_values(by='Date', ascending=False)
        
        s_wins = len(res_df[res_df['Spread Result'] == 'WIN'])
        s_loss = len(res_df[res_df['Spread Result'] == 'LOSS'])
        s_total = s_wins + s_loss
        s_pct = (s_wins / s_total * 100) if s_total > 0 else 0.0

        st.markdown("### 📊 ROI Tracker")
        m1, m2 = st.columns(2)
        m1.metric("Spread Record", f"{s_wins}-{s_loss}", f"{s_pct:.1f}% Win Rate")
        m2.metric("Total Graded Games", len(res_df))
        
        st.divider()
        def color_history(val):
            if val == "WIN": return 'color: green; font-weight: bold'
            if val == "LOSS": return 'color: red; font-weight: bold'
            return 'color: gray'

        st.dataframe(res_df.style.map(color_history, subset=['Spread Result']), hide_index=True, use_container_width=True)
    else:
        st.warning("No completed games found.")