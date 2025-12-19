import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="CFB Quant Grader", page_icon="🏈", layout="wide")
st.title("🏈 CFB Algorithmic Betting Engine")

# --- 1. SETUP & DATA LOADING ---
@st.cache_data(ttl=300) 
def fetch_scores():
    try:
        api_key = st.secrets["CFBD_API_KEY"]
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # CHANGED: Removed 'week=1'. Now fetches ALL Postseason games.
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

if not df.empty and scores:
    for _, row in df.iterrows():
        gid = row.get("GameID")
        game = scores.get(gid)
        
        if game and game['status'] == 'completed':
            home_score = game.get('home_points', 0)
            away_score = game.get('away_points', 0)
            
            # Grade Spread
            pick_team = row['Pick_Team']
            line = row['Pick_Line']
            if pick_team == row['HomeTeam']:
                margin = home_score - away_score
            else:
                margin = away_score - home_score
            
            if (margin + line) == 0: spread_res = "✋ PUSH"
            elif (margin + line) > 0: spread_res = "✅ WIN"
            else: spread_res = "❌ LOSS"

            # Grade Total
            total_score = home_score + away_score
            pick_side = row['Pick_Side'] 
            pick_total = row['Pick_Total']
            
            if total_score == pick_total: total_res = "✋ PUSH"
            elif pick_side == "OVER": total_res = "✅ WIN" if total_score > pick_total else "❌ LOSS"
            else: total_res = "✅ WIN" if total_score < pick_total else "❌ LOSS"

            graded_results.append({
                "Game": f"{row['AwayTeam']} {away_score} - {home_score} {row['HomeTeam']}",
                "Spread Bet": f"{row['Spread Pick']}",
                "Spread Result": spread_res,
                "Total Bet": f"{row['Total Pick']}",
                "Total Result": total_res
            })

# --- 3. DISPLAY ---

# A. Report Card
if graded_results:
    st.markdown(f"### 📝 Graded Results ({len(graded_results)} Games)")
    results_df = pd.DataFrame(graded_results)
    
    def color_results(val):
        if "WIN" in val: return 'color: green; font-weight: bold'
        if "LOSS" in val: return 'color: red; font-weight: bold'
        return 'color: gray'

    st.dataframe(
        results_df.style.map(color_results, subset=['Spread Result', 'Total Result']),
        use_container_width=True, hide_index=True
    )
    st.divider()

# B. Upcoming Board
st.subheader("🔮 Upcoming Predictions")

def color_confidence(val):
    try:
        score = float(val.strip('%'))
    except: return ''
    if score >= 60.0: return 'background-color: #2e7d32; color: white'
    elif score >= 55.0: return 'background-color: #4caf50; color: black'
    elif score <= 52.5: return 'background-color: #ef5350; color: white'
    return ''

if not df.empty:
    # Filter out completed games from the "Upcoming" view
    # We do this by checking if the GameID is in the 'graded_results' list? 
    # Simpler: Just rely on the user visually distinguishing them, or filter logic:
    # (Optional: Filter df to exclude GIDs that are in scores and completed)
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Spread Edges")
        st.dataframe(
            df[['Game', 'Spread Pick', 'Spread Book', 'Spread Conf']].style.map(color_confidence, subset=['Spread Conf']),
            use_container_width=True, hide_index=True
        )
    with col2:
        st.caption("Totals Edges")
        st.dataframe(
            df[['Game', 'Total Pick', 'Total Book', 'Total Conf']].style.map(color_confidence, subset=['Total Conf']),
            use_container_width=True, hide_index=True
        )
else:
    st.info("No predictions found.")