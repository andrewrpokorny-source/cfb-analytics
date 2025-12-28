import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="CFB Quant Engine", page_icon="🏈", layout="wide")
st.title("🏈 CFB Quant Engine: Triple Threat Dashboard")

load_dotenv()

# --- 1. LOAD DATA ---
@st.cache_data(ttl=0)
def load_data():
    if not os.path.exists("live_predictions.csv"):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv("live_predictions.csv")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    
    if 'GameID' in df.columns:
        df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        
    if 'Manual_HomeScore' not in df.columns:
        df['Manual_HomeScore'] = pd.NA
        df['Manual_AwayScore'] = pd.NA
        
    return df

df = load_data()

# --- 2. DIAGNOSTICS ---
with st.expander("🛠️ Data Diagnostics", expanded=False):
    st.write(f"Rows: {len(df)}")
    if not df.empty: st.dataframe(df.head())

# --- 3. MAIN DISPLAY ---
if df.empty:
    st.warning("⚠️ Waiting for data... (CSV is empty)")
    st.stop()

if 'StartDate' in df.columns:
    df = df.sort_values(by='StartDate', ascending=True)

# --- 4. PRE-CALCULATE GRADES ---
graded_rows = []

for _, row in df.iterrows():
    if pd.notna(row['Manual_HomeScore']):
        h_score = float(row['Manual_HomeScore'])
        a_score = float(row['Manual_AwayScore'])
        winner = row['HomeTeam'] if h_score > a_score else row['AwayTeam']
        
        # SU
        su_pick = row.get('Moneyline Pick')
        su_res = "WIN" if su_pick == winner else "LOSS"
        
        # Spread
        spread_res = "PUSH"
        if pd.notna(row.get('Pick_Line')):
            pick_team = row.get('Pick_Team')
            line = float(row.get('Pick_Line', 0))
            if pick_team == row['HomeTeam']: margin = (h_score - a_score) + line
            else: margin = (a_score - h_score) + line
            if margin > 0: spread_res = "WIN"
            elif margin < 0: spread_res = "LOSS"
            
        # Total
        total_res = "PUSH"
        if pd.notna(row.get('Pick_Total')):
            actual_total = h_score + a_score
            target = float(row.get('Pick_Total', 0))
            side = row.get('Pick_Side')
            if side == 'OVER': total_res = "WIN" if actual_total > target else "LOSS"
            elif side == 'UNDER': total_res = "WIN" if actual_total < target else "LOSS"
            if actual_total == target: total_res = "PUSH"

        new_row = row.copy()
        new_row['Res (SU)'] = su_res
        new_row['Res (Spr)'] = spread_res
        new_row['Res (Tot)'] = total_res
        new_row['Pick (SU)'] = su_pick
        new_row['Pick (Spr)'] = row.get('Spread Pick')
        new_row['Pick (Tot)'] = row.get('Total Pick')
        new_row['Date'] = str(row.get('StartDate'))[:10]
        new_row['Game'] = f"{row['AwayTeam']} {int(a_score)} - {int(h_score)} {row['HomeTeam']}"
        
        graded_rows.append(new_row)

if graded_rows:
    graded_df = pd.DataFrame(graded_rows)
else:
    graded_df = pd.DataFrame()

upcoming_df = df[df['Manual_HomeScore'].isna()].copy()

t1, t2, t3 = st.tabs(["🔮 Forecast Board", "📜 Performance History", "💰 Bankroll Simulator"])

def color_conf(val):
    try:
        s = float(str(val).strip('%'))
        if s >= 60: return 'background-color: #1b5e20; color: white'
        if s >= 55: return 'background-color: #4caf50; color: black'
    except: pass
    return ''

def color_result_cell(val):
    if val == 'WIN': return 'background-color: #c8e6c9; color: #1b5e20; font-weight: bold'
    if val == 'LOSS': return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold'
    if val == 'PUSH': return 'background-color: #e0e0e0; color: #424242'
    return ''

with t1:
    st.subheader(f"Upcoming Games ({len(upcoming_df)})")
    if not upcoming_df.empty:
        cols = ['StartDate', 'Game', 'Moneyline Pick', 'Moneyline Conf', 
                'Spread Pick', 'Spread Conf', 'Total Pick', 'Total Conf']
        valid_cols = [c for c in cols if c in upcoming_df.columns]
        st.dataframe(upcoming_df[valid_cols].style.map(color_conf, subset=[c for c in ['Moneyline Conf', 'Spread Conf', 'Total Conf'] if c in valid_cols]), use_container_width=True, hide_index=True)
    else:
        st.info("No upcoming games found.")

with t2:
    if not graded_df.empty:
        display_df = graded_df.sort_values(by='StartDate', ascending=False)
        def get_record(df, res_col):
            if res_col not in df.columns: return "0-0-0", 0.0
            wins = len(df[df[res_col] == 'WIN'])
            losses = len(df[df[res_col] == 'LOSS'])
            pushes = len(df[df[res_col] == 'PUSH'])
            total = wins + losses
            pct = (wins / total * 100) if total > 0 else 0.0
            return f"{wins}-{losses}-{pushes}", pct

        rec_su, pct_su = get_record(display_df, 'Res (SU)')
        rec_spr, pct_spr = get_record(display_df, 'Res (Spr)')
        rec_tot, pct_tot = get_record(display_df, 'Res (Tot)')

        st.markdown("### 📊 Performance Report")
        m1, m2, m3 = st.columns(3)
        m1.metric("🏆 Straight Up", rec_su, f"{pct_su:.1f}%")
        m2.metric("⚖️ Spread", rec_spr, f"{pct_spr:.1f}%")
        m3.metric("↕️ Total", rec_tot, f"{pct_tot:.1f}%")
        st.divider()
        hist_cols = ['Date', 'Game', 'Pick (SU)', 'Res (SU)', 'Pick (Spr)', 'Res (Spr)', 'Pick (Tot)', 'Res (Tot)']
        valid_hist_cols = [c for c in hist_cols if c in display_df.columns]
        st.dataframe(display_df[valid_hist_cols].style.map(color_result_cell, subset=[c for c in ['Res (SU)', 'Res (Spr)', 'Res (Tot)'] if c in valid_hist_cols]), use_container_width=True, hide_index=True)
    else:
        st.info("No graded games yet.")

with t3:
    if not graded_df.empty:
        st.markdown("### 📈 Bankroll Simulator (Precise Odds)")
        wager = st.number_input("Enter Bet Amount ($)", min_value=10, value=100, step=10)
        st.caption(f"Simulation: ${wager} per game. Uses EXACT historical odds for Moneyline.")
        
        sim_df = graded_df.sort_values(by='StartDate', ascending=True).copy()
        
        # EXACT PAYOUT CALCULATOR
        def calc_pnl(row, pick_type):
            res_col = f"Res ({pick_type})"
            res = row.get(res_col)
            
            if res == 'LOSS': return -float(wager)
            if res == 'PUSH': return 0.0
            
            # WINS
            if pick_type == 'SU':
                # Use real odds if available, else +100 fallback
                odds = row.get('Pick_ML_Odds')
                if pd.isna(odds) or odds == 0: odds = 100
                
                if odds > 0: return wager * (odds / 100)
                else: return wager * (100 / abs(odds))
                
            else:
                # Spread/Total assumed -110 standard
                return wager * (100/110)

        sim_df['Profit_SU'] = sim_df.apply(lambda r: calc_pnl(r, 'SU'), axis=1)
        sim_df['Profit_Spread'] = sim_df.apply(lambda r: calc_pnl(r, 'Spr'), axis=1)
        sim_df['Profit_Total'] = sim_df.apply(lambda r: calc_pnl(r, 'Tot'), axis=1)

        sim_df['Bankroll_SU'] = sim_df['Profit_SU'].cumsum()
        sim_df['Bankroll_Spread'] = sim_df['Profit_Spread'].cumsum()
        sim_df['Bankroll_Total'] = sim_df['Profit_Total'].cumsum()
        
        st.line_chart(sim_df[['Date', 'Bankroll_SU', 'Bankroll_Spread', 'Bankroll_Total']].set_index('Date'))
        
        b1, b2, b3 = st.columns(3)
        b1.metric("SU Net Profit", f"${sim_df['Profit_SU'].sum():,.2f}")
        b2.metric("Spread Net Profit", f"${sim_df['Profit_Spread'].sum():,.2f}")
        b3.metric("Total Net Profit", f"${sim_df['Profit_Total'].sum():,.2f}")
        
    else:
        st.info("No history available.")