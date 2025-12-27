import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="CFB Quant Engine", page_icon="🏈", layout="wide")
st.title("🏈 CFB Quant Engine: Triple Threat Dashboard")

load_dotenv()

# --- 1. LOAD DATA (Force Read) ---
@st.cache_data(ttl=0)
def load_data():
    if not os.path.exists("live_predictions.csv"):
        return pd.DataFrame()
    
    # Read CSV without manipulating it first
    df = pd.read_csv("live_predictions.csv")
    
    # Basic cleanup
    if 'GameID' in df.columns:
        df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
    return df

df = load_data()

# --- 2. DIAGNOSTIC PANEL (Top of Screen) ---
with st.expander("🛠️ Data Diagnostics (Open if Board is Empty)", expanded=True):
    st.write(f"**Rows in CSV:** {len(df)}")
    if not df.empty:
        st.dataframe(df.head(3))
    else:
        st.error("CSV is empty! Run 'python3 predict.py' again.")

# --- 3. MAIN DISPLAY ---
if df.empty:
    st.stop()

# Force 'StartDate' sorting if it exists, otherwise ignore
if 'StartDate' in df.columns:
    df = df.sort_values(by='StartDate', ascending=True)

# Separate Graded vs Upcoming
# Logic: If it has a Manual Score, it's graded. Otherwise, it's Upcoming.
upcoming = df[df['Manual_HomeScore'].isna()].copy()
graded = df[df['Manual_HomeScore'].notna()].copy()

t1, t2 = st.tabs(["🔮 Forecast Board", "📜 Performance History"])

def color_conf(val):
    # Simple highlighter for confidence
    try:
        s = str(val).replace('%', '')
        if float(s) >= 60: return 'background-color: #1b5e20; color: white'
        if float(s) >= 55: return 'background-color: #4caf50; color: black'
    except: pass
    return ''

with t1:
    st.subheader(f"Upcoming Games ({len(upcoming)})")
    if not upcoming.empty:
        # We manually build the columns to ensure they exist
        display_cols = ['Game', 'Moneyline Pick', 'Moneyline Conf', 'Spread Pick', 'Spread Conf', 'Total Pick', 'Total Conf']
        # Filter strictly to columns that actually exist in the CSV
        valid_cols = [c for c in display_cols if c in upcoming.columns]
        
        # Display the table raw and unfiltered
        st.dataframe(
            upcoming[valid_cols].style.map(color_conf, subset=[c for c in ['Moneyline Conf', 'Spread Conf', 'Total Conf'] if c in valid_cols]), 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("No upcoming games found in data.")

with t2:
    st.subheader(f"History ({len(graded)})")
    if not graded.empty:
        st.dataframe(graded, use_container_width=True)
    else:
        st.info("No graded games yet.")