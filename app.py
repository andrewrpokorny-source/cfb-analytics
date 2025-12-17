import streamlit as st
import pandas as pd

# 1. Page Config
st.set_page_config(
    page_title="CFB Quant Engine",
    page_icon="🏈",
    layout="wide"
)

# 2. Title
st.title("🏈 CFB Algorithmic Betting Engine")
st.markdown("### Powered by Decay Metrics, SRS, and Talent Composites")

# 3. Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("live_predictions.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# 4. Color Logic for Confidence
def color_confidence(val):
    """
    Colors the cell green if confidence is high (>55%), 
    red if low (<53%).
    Expects string input like "58.2%"
    """
    if pd.isna(val):
        return ''
    
    # Clean string to float
    try:
        score = float(val.strip('%'))
    except:
        return ''

    color = ''
    if score >= 60.0:
        color = 'background-color: #2e7d32; color: white' # Dark Green
    elif score >= 55.0:
        color = 'background-color: #4caf50; color: black' # Green
    elif score <= 52.5:
        color = 'background-color: #ef5350; color: white' # Red
        
    return color

if df.empty:
    st.error("No predictions found. Please run the pipeline locally and push the CSV.")
else:
    # --- SECTION 1: TOP SPREAD PICKS ---
    st.divider()
    st.header("🎯 Top Spread Edges")
    st.caption("Ranked by Model Confidence. Green = Bet, Red = Pass.")

    # Sort by raw confidence if available, otherwise just display
    spread_cols = ['Game', 'Spread Pick', 'Spread Book', 'Spread Conf']
    
    # We create a styler object to apply colors
    st.dataframe(
        df[spread_cols].style.map(color_confidence, subset=['Spread Conf']),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Spread Book": st.column_config.ImageColumn("Book") # Optional: could map logos later
        }
    )

    # --- SECTION 2: TOP TOTALS PICKS ---
    st.divider()
    st.header("📉 Top Totals Edges")
    st.caption("Over/Under Opportunities.")

    total_cols = ['Game', 'Total Pick', 'Total Book', 'Total Conf']
    
    st.dataframe(
        df[total_cols].style.map(color_confidence, subset=['Total Conf']),
        use_container_width=True,
        hide_index=True
    )

    # --- SIDEBAR INFO ---
    with st.sidebar:
        st.header("⚙️ Strategy")
        st.info("""
        **Data Source:**
        US Regulated Books Only
        (DraftKings, FanDuel, MGM, etc.)
        """)
        
        st.write("""
        **Legend:**
        * 🟢 **>60%:** Max Bet (1.5u)
        * 🟢 **>55%:** Standard Bet (1.0u)
        * ⚪ **53-55%:** Lean (0.5u)
        * 🔴 **<53%:** NO BET
        """)
        st.caption("Last Updated via GitHub Pipeline")