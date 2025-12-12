import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

def get_data(endpoint, params):
    url = f"https://api.collegefootballdata.com{endpoint}"
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        time.sleep(0.5)
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def main():
    print("--- 🌟 INJECTING TEAM TALENT COMPOSITE 🌟 ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv("cfb_training_data_final.csv")
        print(f"Loaded {len(df)} games.")
    except FileNotFoundError:
        print("Error: cfb_training_data_final.csv not found. Run power.py first!")
        return

    # 2. Fetch Talent
    years = df['season'].unique()
    all_talent = []
    
    print("Fetching 247Sports Talent Composite...")

    for year in years:
        talent = get_data("/talent", {"year": year})
        if talent:
            df_t = pd.json_normalize(talent)
            
            # Smart Column Check
            if 'school' in df_t.columns:
                team_col = 'school'
            elif 'team' in df_t.columns:
                team_col = 'team'
            else:
                continue
                
            if 'talent' in df_t.columns:
                # Standardize to just [team, season, score]
                df_t = df_t[[team_col, 'year', 'talent']].rename(columns={
                    team_col: 'team', 
                    'year': 'season',
                    'talent': 'talent_score'
                })
                all_talent.append(df_t)
            
    if not all_talent:
        print("Failed to fetch talent data.")
        return

    df_talent = pd.concat(all_talent, ignore_index=True)
    
    # 3. Surgical Merge (No Prefixes to avoid duplicates)
    print("Merging Talent Scores...")
    
    # HOME TEAM MERGE
    # We rename the talent dataframe temporarily to match the main dataframe keys
    home_talent = df_talent.rename(columns={
        'team': 'home_team', 
        'talent_score': 'home_talent_score'
    })
    # Merge on home_team AND season
    df = pd.merge(df, home_talent, on=['home_team', 'season'], how='left')
    
    # AWAY TEAM MERGE
    away_talent = df_talent.rename(columns={
        'team': 'away_team', 
        'talent_score': 'away_talent_score'
    })
    df = pd.merge(df, away_talent, on=['away_team', 'season'], how='left')
    
    # Fill missing values for small schools
    df['home_talent_score'] = df['home_talent_score'].fillna(10.0)
    df['away_talent_score'] = df['away_talent_score'].fillna(10.0)
    
    # 4. Save
    output_filename = "cfb_training_data_ultimate.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nSUCCESS: Saved ultimate dataset to {output_filename}")
    print("New Columns: ['home_talent_score', 'away_talent_score']")

if __name__ == "__main__":
    main()