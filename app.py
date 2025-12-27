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
    
    df = pd.read_csv("live_predictions.csv")
    
    # Cleanup GameID
    if 'GameID' in df.columns:
        df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        
    # --- CRITICAL FIX: Ensure 'Manual_HomeScore' exists ---
    # The real engine doesn't output this column, so we add it if missing.
    if 'Manual_HomeScore' not in df.columns:
        df['Manual_HomeScore'] = pd.NA
        df['Manual_AwayScore'] = pd.NA
        
    return df

df = load_data()

# --- 2. DIAGNOSTICS (Toggle closed by default now) ---
with st.expander("🛠️ Data Diagnostics", expanded=False):
    st.write(f"Rows: {len(df)}")
    if not df.empty: st.dataframe(df.head())

# --- 3. MAIN DISPLAY ---
if df.empty:
    st.warning("⚠️ Waiting for data...")
    st.stop()

# Force Sort by Date
if 'StartDate' in df.columns:
    df = df.sort_values(by='StartDate', ascending=True)

# Split Data
# If Manual Score is missing (NaN), it's Upcoming. Otherwise, it's History.
upcoming = df[df['Manual_HomeScore'].isna()].copy()
graded = df[df['Manual_HomeScore'].notna()].copy()

t1, t2 = st.tabs(["🔮 Forecast Board", "📜 Performance History"])

def color_conf(val):
    try:
        s = float(str(val).strip('%'))
        if s >= 60: return 'background-color: #1b5e20; color: white'
        if s >= 55: return 'background-color: #4caf50; color: black'
    except: pass
    return ''

with t1:
    st.subheader(f"Upcoming Games ({len(upcoming)})")
    if not upcoming.empty:
        # Define the exact columns we want to show
        cols = ['StartDate', 'Game', 'Moneyline Pick', 'Moneyline Conf', 
                'Spread Pick', 'Spread Conf', 'Total Pick', 'Total Conf']
        
        # Only show columns that actually exist in the file
        valid_cols = [c for c in cols if c in upcoming.columns]
        
        # Display
        st.dataframe(
            upcoming[valid_cols].style.map(color_conf, subset=[c for c in ['Moneyline Conf', 'Spread Conf', 'Total Conf'] if c in valid_cols]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No upcoming games found.")

with t2:
    st.subheader(f"History ({len(graded)})")
    if not graded.empty:
        st.dataframe(graded, use_container_width=True)
    else:
        st.info("No graded games yet.")