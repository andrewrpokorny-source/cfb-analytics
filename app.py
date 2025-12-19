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
            # Force Key to String for reliable matching
            return {str(g['id']): g for g in res.json()}
    except Exception as e:
        st.error(f"API Connection Error: {e}")
    return {}

@st.cache_data(ttl=0)
def load_picks():
    try:
        df = pd.read_csv("live_predictions.csv")
        # Clean IDs: Force to string, remove decimal suffixes like ".0"
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
            
            # --- ROBUST DATE PARSING ---
            # Try both snake_case (standard) and camelCase (rare API variance)
            start_str = None
            if game:
                start_str = game.get('start_date') or game.get('startDate')
            
            if start_str:
                new_row['Kickoff_Sort'] = start_str
                try:
                    # 1. Convert string to Datetime Object
                    dt = pd.to_datetime(start_str)
                    
                    # 2. Handle Timezones (API is usually UTC)
                    if dt.tzinfo is None:
                        dt = dt.tz_localize('UTC')
                    
                    # 3. Convert to Eastern Time
                    dt_et = dt.tz_convert('US/Eastern')
                    
                    # 4. Format nicely
                    new_row['Time'] = dt_et.strftime('%a %I:%M %p') # e.g., "Sat 12:00 PM"
                except:
                    # If parsing fails, just show the raw string truncated
                    new_row['Time'] = str(start_str)[:16]
            else:
                new_row['Kickoff_Sort'] = "2099-12-31"
                new_row['Time'] = "Date Missing" # Changed from TBD to diagnose
                
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
        
        # Sort by Date first, then Confidence
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
with st.expander("⚙️ System Status (Deep Debug)"):
    st.write(f"**API Games Fetched:** {len(scores)}")
    st.write(f"**CSV Picks Loaded:** {len(df)}")
    
    if not df.empty and len(scores) > 0:
        sample_id = str(df.iloc[0]['GameID'])
        match = scores.get(sample_id)
        
        st.write(f"**Sample ID:** `{sample_id}`")
        if match:
            st.success("✅ Game Object Found")
            # Show the raw date field to verify the key name
            st.write(f"**Raw Date Value:** `{match.get('start_date')}`")
            st.write(f"**All Keys Available:** {list(match.keys())[:5]}...") 
        else:
            st.error("❌ Match Failed for sample row.")