cat << 'EOF' > inspect_game.py
import pandas as pd
import requests
import os
import math
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
YEAR = 2025

def get_data(endpoint, params):
    res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
    return res.json()

def main():
    print(f"--- 🕵️ INSPECTING DATA FOR {YEAR} ---")
    
    # 1. Fetch Ratings
    print("   -> Downloading Talent & SRS ratings...")
    talent = get_data("/talent", {"year": YEAR})
    srs = get_data("/ratings/srs", {"year": YEAR})
    
    # Convert to Dictionaries
    talent_map = {x['school']: x['talent'] for x in talent}
    srs_map = {x['team']: x['rating'] for x in srs}
    
    # 2. Check Specific Teams
    # Update these names if needed to match what you see in the app
    team_a = "Indiana"
    team_b = "Alabama"
    
    t_a = talent_map.get(team_a, 0)
    s_a = srs_map.get(team_a, 0)
    
    t_b = talent_map.get(team_b, 0)
    s_b = srs_map.get(team_b, 0)
    
    print(f"\n🏈 MATCHUP ANALYSIS: {team_a} vs {team_b}")
    print(f"   {team_a}: Talent={t_a}, SRS={s_a}")
    print(f"   {team_b}: Talent={t_b}, SRS={s_b}")
    
    # 3. Re-run the Math
    # Talent Diff / 20 is the current formula
    talent_edge = (t_a - t_b) / 20.0
    srs_edge = s_a - s_b
    hfa = 1.0 # Assuming Team A is Home
    
    total_edge = srs_edge + talent_edge + hfa
    
    # Sigmoid
    try: prob = 1 / (1 + math.exp(-1 * total_edge / 7.5))
    except: prob = 0.5
    
    print(f"\n🧮 THE MATH:")
    print(f"   SRS Edge ({team_a}):     {srs_edge:.2f} pts")
    print(f"   Talent Edge ({team_a}):  {talent_edge:.2f} pts ( This is likely the culprit! )")
    print(f"   Home Field:            +{hfa} pts")
    print(f"   --------------------------------")
    print(f"   TOTAL EDGE:            {total_edge:.2f} pts")
    print(f"   {team_a} Win Prob:       {prob:.1%}")
    print(f"   {team_b} Win Prob:       {1-prob:.1%}")

if __name__ == "__main__":
    main()
EOF

