import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="CFB Quant Engine", page_icon="📈", layout="wide")
st.title("📈 CFB Quant Engine: Trading Terminal")

# Constants for "Smart" Math
# -110 odds implies a 52.38% break-even point.
# If our model is >52.38%, we have a mathematical "Edge" (Positive EV).
BREAK_EVEN_PROB = 0.5238

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
def load_data():
    try:
        df = pd.read_csv("live_predictions.csv")
        if 'GameID' in df.columns:
            df = df.dropna(subset=['GameID'])
            df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        return df
    except:
        return pd.DataFrame()

# --- 2. QUANT LOGIC (EV Only) ---
def calculate_ev(confidence_str):
    """
    Parses confidence string and calculates Edge.
    Edge = Model Probability - Break Even Probability (52.38%)
    """
    try:
        prob = float(confidence_str.strip('%')) / 100.0
    except:
        return 0.0, 0.0

    # Calculate Edge (How much better is this bet than a coin flip?)
    edge = prob - BREAK_EVEN_PROB
    
    return prob, edge

# --- 3. PROCESSING LOOP ---
df = load_data()
scores = fetch_scores()
graded_results = []
active_plays = []

now_utc = datetime.now(timezone.utc)

if not df.empty:
    for _, row in df.iterrows():
        gid = str(row.get("GameID"))
        game = scores.get(gid)
        
        # Determine Status
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

        # --- PATH A: HISTORY (GRADED) ---
        if is_completed:
            pick_team = row['Pick_Team']
            try: line = float(row['Pick_Line'])
            except: line = 0.0
            
            margin = (home_score - away_score) if pick_team == row['HomeTeam'] else (away_score - home_score)
            diff = margin + line
            res = "WIN" if diff > 0 else "LOSS" if diff < 0 else "PUSH"
            
            graded_results.append({
                "Date": date_str,
                "Game": f"{row['AwayTeam']} {away_score} - {home_score} {row['HomeTeam']}",
                "Pick": row['Spread Pick'],
                "Result": res
            })
            
        # --- PATH B: ACTIVE (QUANT ANALYSIS) ---
        else:
            start_str = game.get('start_date') if game else None
            show_game = True
            time_display = "TBD"
            kickoff_sort = "2099-12-31"
            
            if start_str:
                kickoff_sort = start_str
                try:
                    dt = pd.to_datetime(start_str)
                    if dt.tzinfo is None: dt = dt.tz_localize('UTC')
                    if dt < now_utc: show_game = False
                    time_display = dt.tz_convert('US/Eastern').strftime('%a %I:%M %p')
                except: pass

            if show_game:
                # Calculate Metrics (No Kelly, just EV)
                s_prob, s_edge = calculate_ev(row['Spread Conf'])
                t_prob, t_edge = calculate_ev(row['Total Conf'])
                
                # SPREAD ENTRY
                active_plays.append({
                    "Time": time_display, "Sort": kickoff_sort,
                    "Game": row['Game'],
                    "Type": "Spread",
                    "Pick": row['Spread Pick'],
                    "Model %": s_prob,
                    "Edge": s_edge
                })
                # TOTAL ENTRY
                active_plays.append({
                    "Time": time_display, "Sort": kickoff_sort,
                    "Game": row['Game'],
                    "Type": "Total",
                    "Pick": row['Total Pick'],
                    "Model %": t_prob,
                    "Edge": t_edge
                })

# --- 4. DISPLAY UI ---
tab1, tab2 = st.tabs(["💎 Value Board (Quant)", "📜 History"])

with tab1:
    st.markdown("### ⚡ Positive EV Opportunities")
    st.caption("Filters: Only showing plays where the Model Probability > 52.4% (Positive Expected Value vs -110 odds).")
    
    if active_plays:
        q_df = pd.DataFrame(active_plays)
        
        # FILTER: Strict +EV only (Edge > 0)
        ev_df = q_df[q_df['Edge'] > 0].copy()
        
        if not ev_df.empty:
            ev_df = ev_df.sort_values(by=['Edge', 'Sort'], ascending=[False, True])
            
            # Formatting
            ev_df['Model %'] = ev_df['Model %'].map('{:.1%}'.format)
            ev_df['Edge'] = ev_df['Edge'].map('{:.2%}'.format)
            
            def color_edge(val):
                # Green intensity based on value
                try: score = float(val.strip('%'))
                except: return ''
                if score > 5.0: return 'background-color: #1b5e20; color: white' # Huge Value (>5% edge)
                if score > 2.5: return 'background-color: #2e7d32; color: white' # Great Value (>2.5% edge)
                if score > 0.0: return 'background-color: #e8f5e9; color: black' # Marginal Value (>0% edge)
                return ''

            st.dataframe(
                ev_df[['Time', 'Game', 'Type', 'Pick', 'Model %', 'Edge']].style.map(color_edge, subset=['Edge']), 
                hide_index=True, 
                use_container_width=True
            )
        else:
            st.warning("⚠️ No +EV plays found. The model does not see an edge in the current lines (all confidence < 52.4%).")
            with st.expander("Show all raw predictions (Negative EV)"):
                st.dataframe(q_df[['Time', 'Game', 'Type', 'Pick', 'Model %', 'Edge']])
            
    else:
        st.info("No active games.")

with tab2:
    if graded_results:
        h_df = pd.DataFrame(graded_results)
        st.dataframe(h_df, hide_index=True, use_container_width=True)
    else:
        st.info("History empty.")