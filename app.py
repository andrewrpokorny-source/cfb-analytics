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
            return {g['id']: g for g in res.json()}
    except:
        return {}
    return {}

@st.cache_data
def load_picks():
    try:
        return pd.read_csv("live_predictions.csv")
    except:
        return pd.DataFrame()

df = load_picks()
scores = fetch_scores()

# --- 2. GRADING LOGIC ---
graded_results = []
upcoming_games = []

if not df.empty:
    for _, row in df.iterrows():
        gid = row.get("GameID")
        game = scores.get(gid)
        
        # IF GAME IS COMPLETED -> GRADE IT
        if game and game['status'] == 'completed':
            home_score = game.get('home_points', 0)
            away_score = game.get('away_points', 0)
            
            # Grade Spread
            pick_team = row['Pick_Team']
            raw_home_spread = row['Pick_Line'] 
            
            if pick_team == row['HomeTeam']:
                margin = (home_score - away_score) + raw_home_spread
            else:
                margin = (away_score - home_score) - raw_home_spread
            
            if margin == 0: spread_res = "PUSH"
            elif margin > 0: spread_res = "WIN"
            else: spread_res = "LOSS"

            # Grade Total
            total_score = home_score + away_score
            pick_side = row['Pick_Side'] 
            pick_total = row['Pick_Total']
            
            if total_score == pick_total: total_res = "PUSH"
            elif pick_side == "OVER": total_res = "WIN" if total_score > pick_total else "LOSS"
            else: total_res = "WIN" if total_score < pick_total else "LOSS"

            graded_results.append({
                "Game": f"{row['AwayTeam']} {away_score} - {home_score} {row['HomeTeam']}",
                "Date": game.get('start_date', 'N/A')[:10],
                "Spread Bet": f"{row['Spread Pick']}",
                "Spread Result": spread_res,
                "Total Bet": f"{row['Total Pick']}",
                "Total Result": total_res
            })
        
        # IF GAME IS NOT COMPLETED -> ADD TO BOARD
        else:
            upcoming_games.append(row)

# --- 3. CREATE TABS ---
tab1, tab2 = st.tabs(["🔮 Betting Board", "📜 Performance History"])

# --- TAB 1: UPCOMING GAMES ---
with tab1:
    st.markdown("### Active & Upcoming Games")
    
    def color_confidence(val):
        try:
            score = float(val.strip('%'))
        except: return ''
        if score >= 60.0: return 'background-color: #2e7d32; color: white'
        elif score >= 55.0: return 'background-color: #4caf50; color: black'
        elif score <= 52.5: return 'background-color: #ef5350; color: white'
        return ''

    if upcoming_games:
        up_df = pd.DataFrame(upcoming_games)
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("Spread Edges")
            st.dataframe(
                up_df[['Game', 'Spread Pick', 'Spread Book', 'Spread Conf']].style.map(color_confidence, subset=['Spread Conf']),
                use_container_width=True, hide_index=True
            )
        with col2:
            st.caption("Totals Edges")
            st.dataframe(
                up_df[['Game', 'Total Pick', 'Total Book', 'Total Conf']].style.map(color_confidence, subset=['Total Conf']),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("No upcoming games found. Check back later!")

# --- TAB 2: HISTORY & METRICS ---
with tab2:
    if graded_results:
        res_df = pd.DataFrame(graded_results)
        
        # --- CALCULATE METRICS ---
        s_wins = len(res_df[res_df['Spread Result'] == 'WIN'])
        s_loss = len(res_df[res_df['Spread Result'] == 'LOSS'])
        s_push = len(res_df[res_df['Spread Result'] == 'PUSH'])
        s_total = s_wins + s_loss
        s_pct = (s_wins / s_total * 100) if s_total > 0 else 0.0

        t_wins = len(res_df[res_df['Total Result'] == 'WIN'])
        t_loss = len(res_df[res_df['Total Result'] == 'LOSS'])
        t_push = len(res_df[res_df['Total Result'] == 'PUSH'])
        t_total = t_wins + t_loss
        t_pct = (t_wins / t_total * 100) if t_total > 0 else 0.0

        # --- DISPLAY SCOREBOARD ---
        st.markdown("### 📊 ROI Tracker")
        m1, m2, m3 = st.columns(3)
        m1.metric("Spread Record", f"{s_wins}-{s_loss}-{s_push}", f"{s_pct:.1f}% Win Rate")
        m2.metric("Total Record", f"{t_wins}-{t_loss}-{t_push}", f"{t_pct:.1f}% Win Rate")
        m3.metric("Total Graded Games", len(res_df))
        
        st.divider()
        
        # --- DISPLAY HISTORY TABLE ---
        st.markdown("### 📜 Game Log")
        
        def color_history(val):
            if val == "WIN": return 'color: green; font-weight: bold'
            if val == "LOSS": return 'color: red; font-weight: bold'
            return 'color: gray'

        st.dataframe(
            res_df.style.map(color_history, subset=['Spread Result', 'Total Result']),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No completed games to grade yet.")