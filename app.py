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
    
    # Cleanup GameID
    if 'GameID' in df.columns:
        df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        
    # Safety Check
    if 'Manual_HomeScore' not in df.columns:
        df['Manual_HomeScore'] = pd.NA
        df['Manual_AwayScore'] = pd.NA
        
    return df

df = load_data()

# --- 2. DIAGNOSTICS (Closed by default) ---
with st.expander("🛠️ Data Diagnostics", expanded=False):
    st.write(f"Rows: {len(df)}")
    if not df.empty: st.dataframe(df.head())

# --- 3. MAIN DISPLAY ---
if df.empty:
    st.warning("⚠️ Waiting for data... (CSV is empty)")
    st.stop()

# Force Sort by Date
if 'StartDate' in df.columns:
    df = df.sort_values(by='StartDate', ascending=True)

# Split Data
upcoming = df[df['Manual_HomeScore'].isna()].copy()
graded = df[df['Manual_HomeScore'].notna()].copy()

# Sort History in Reverse Chronological Order (Newest first)
if not graded.empty:
    graded = graded.sort_values(by='StartDate', ascending=False)

t1, t2 = st.tabs(["🔮 Forecast Board", "📜 Performance History"])

def color_conf(val):
    try:
        s = float(str(val).strip('%'))
        if s >= 60: return 'background-color: #1b5e20; color: white'
        if s >= 55: return 'background-color: #4caf50; color: black'
    except: pass
    return ''

def color_result_cell(val):
    # Visual cues for W/L
    if val == 'WIN': return 'background-color: #c8e6c9; color: #1b5e20; font-weight: bold' # Light Green
    if val == 'LOSS': return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold' # Light Red
    if val == 'PUSH': return 'background-color: #e0e0e0; color: #424242'
    return ''

with t1:
    st.subheader(f"Upcoming Games ({len(upcoming)})")
    if not upcoming.empty:
        cols = ['StartDate', 'Game', 'Moneyline Pick', 'Moneyline Conf', 
                'Spread Pick', 'Spread Conf', 'Total Pick', 'Total Conf']
        valid_cols = [c for c in cols if c in upcoming.columns]
        
        st.dataframe(
            upcoming[valid_cols].style.map(color_conf, subset=[c for c in ['Moneyline Conf', 'Spread Conf', 'Total Conf'] if c in valid_cols]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No upcoming games found.")

with t2:
    if not graded.empty:
        # --- CALCULATE STATS ---
        # Helper to get record
        def get_record(df, res_col):
            wins = len(df[df[res_col] == 'WIN'])
            losses = len(df[df[res_col] == 'LOSS'])
            pushes = len(df[df[res_col] == 'PUSH'])
            total = wins + losses
            pct = (wins / total * 100) if total > 0 else 0.0
            return f"{wins}-{losses}-{pushes}", pct

        rec_su, pct_su = get_record(graded, 'Res (SU)')
        rec_spr, pct_spr = get_record(graded, 'Res (Spr)')
        rec_tot, pct_tot = get_record(graded, 'Res (Tot)')

        # --- SCOREBOARD ---
        st.markdown("### 📊 Performance Report (Since Dec 1)")
        m1, m2, m3 = st.columns(3)
        m1.metric("🏆 Straight Up", rec_su, f"{pct_su:.1f}%")
        m2.metric("⚖️ Spread", rec_spr, f"{pct_spr:.1f}%")
        m3.metric("↕️ Total", rec_tot, f"{pct_tot:.1f}%")

        st.divider()

        # --- DETAILED TABLE ---
        # Select clean columns for history
        hist_cols = ['Date', 'Game', 'Pick (SU)', 'Res (SU)', 'Pick (Spr)', 'Res (Spr)', 'Pick (Tot)', 'Res (Tot)']
        # Only use cols that exist (in case backfill didn't create them all)
        valid_hist_cols = [c for c in hist_cols if c in graded.columns]
        
        st.dataframe(
            graded[valid_hist_cols].style.map(color_result_cell, subset=[c for c in ['Res (SU)', 'Res (Spr)', 'Res (Tot)'] if c in graded.columns]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No graded games yet.")