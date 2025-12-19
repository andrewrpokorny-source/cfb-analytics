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
        # Fetch ALL Postseason games
        res = requests.get("https://api.collegefootballdata.com/games", 
                           headers=headers, 
                           params={"year": 2025, "seasonType": "postseason"})
        if res.status_code == 200:
            return {int(g['id']): g for g in res.json()}
    except Exception as e:
        st.error(f"API Connection Error: {e}")
    return {}

@st.cache_data(ttl=0)
def load_picks():
    try:
        df = pd.read_csv("live_predictions.csv")
        # Ensure GameID is integer for matching
        if 'GameID' in df.columns:
            df = df.dropna(subset=['GameID'])
            df['GameID'] = df['GameID'].astype(int)
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
        gid = int(row.get("GameID"))
        game = scores.get(gid)
        
        # A. COMPLETED GAMES (History)
        if game and game.get('status') == 'completed':
            home_score = game.get('home_points', 0)
            away_score = game.get('away_points', 0)
            
            # Grade Spread
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

            # Grade Total
            total_score = home_score + away_score
            pick_side = row['Pick_Side'] 
            try: pick_total = float(row['Pick_Total'])
            except: pick_total = 0.0
            
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
        
        # B. UPCOMING GAMES (Board)
        else:
            new_row = row.copy()
            start_str = game.get('start_date') if game else None
            
            if start_str:
                new_row['Kickoff_Sort'] = start_str
                try:
                    # Bulletproof Parsing with Pandas
                    dt = pd.to_datetime(start_str)
                    # Convert to Eastern Time (UTC-5)
                    dt_et = dt.tz_convert('US/Eastern')
                    new_row['Time'] = dt_et.strftime('%a %I:%M %p') # "Fri 08:00 PM"
                except:
                    # Fallback if timezone conversion fails
                    new_row['Time'] = dt.strftime('%a %I:%M %p')
            else:
                new_row['Kickoff_Sort'] = "2099-12-31"
                new_row['Time'] = "Time TBD"
                
            upcoming_games.append(new_row)

# --- 3. DISPLAY UI ---
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
        
        # Sort: Earliest Kickoff First
        if 'Kickoff_Sort' in up_df.columns:
            up_df = up_df.sort_values(by=['Kickoff_Sort', 'Spread_Conf_Raw'], ascending=[True, False])

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Spread Picks (Sorted by Time)")
            st.dataframe(
                up_df[['Time', 'Game', 'Spread Pick', 'Spread Book', 'Spread Conf']].style.map(color_confidence, subset=['Spread Conf']),
                use_container_width=True, hide_index=True
            )
        with col2:
            st.caption("Totals Picks")
            st.dataframe(
                up_df[['Time', 'Game', 'Total Pick', 'Total Book', 'Total Conf']].style.map(color_confidence, subset=['Total Conf']),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("No upcoming games found.")

with tab2:
    if graded_results:
        res_df = pd.DataFrame(graded_results)
        res_df = res_df.sort_values(by='Date', ascending=False)
        
        # Calculate Records
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

        st.markdown("### 📊 ROI Tracker")
        m1, m2, m3 = st.columns(3)
        m1.metric("Spread Record", f"{s_wins}-{s_loss}-{s_push}", f"{s_pct:.1f}%")
        m2.metric("Total Record", f"{t_wins}-{t_loss}-{t_push}", f"{t_pct:.1f}%")
        m3.metric("Graded Games", len(res_df))
        
        st.divider()
        
        # Color Logic for Win/Loss
        def color_history(val):
            if val == "WIN": return 'color: green; font-weight: bold'
            if val == "LOSS": return 'color: red; font-weight: bold'
            return 'color: gray'

        st.dataframe(
            res_df.style.map(color_history, subset=['Spread Result', 'Total Result']),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("No completed games found to grade.")

# --- DEBUG FOOTER ---
with st.expander("⚙️ System Status"):
    st.write(f"**API Games Fetched:** {len(scores)}")
    if len(scores) == 0:
        st.error("⚠️ API returned 0 games. Check your API Key in Settings > Secrets.")
    st.write(f"**CSV Picks Loaded:** {len(df)}")