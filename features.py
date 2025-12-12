import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

def get_data(endpoint, params):
    try:
        response = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        time.sleep(0.5) 
        return response.json()
    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
        return []

def calculate_weighted_decay(df_stats):
    df_stats = df_stats.sort_values(by=['team', 'season', 'week'])
    
    # REQUIRED METRICS (We force these to exist)
    metrics = [
        'offense.ppa', 'defense.ppa',
        'offense.rushing.ppa', 'defense.rushing.ppa',
        'offense.rushing.successRate', 'defense.rushing.successRate',
        'offense.passing.ppa', 'defense.passing.ppa',
        'offense.passing.successRate', 'defense.passing.successRate'
    ]
    
    # 1. Force columns to exist (Fill missing with 0.0)
    for m in metrics:
        if m not in df_stats.columns:
            df_stats[m] = 0.0
        else:
            df_stats[m] = df_stats[m].fillna(0.0)
            
    # 2. Calculate Decay
    for metric in metrics:
        col_name = f"decay_{metric}"
        df_stats[col_name] = df_stats.groupby(['team', 'season'])[metric].transform(
            lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
        )
    
    return df_stats

def main():
    print("--- 🚀 BUILDING GRANULAR DECAY FEATURES (ROBUST) 🚀 ---")
    
    try:
        df_master = pd.read_csv("cfb_training_data_ultimate.csv")
        print(f"Loaded {len(df_master)} games from Ultimate dataset.")
    except FileNotFoundError:
        print("Error: cfb_training_data_ultimate.csv not found.")
        return

    print("Fetching game-level stats...")
    years = df_master['season'].unique()
    all_game_stats = []

    for year in years:
        stats = get_data("/stats/game/advanced", {"year": year})
        if stats:
            df = pd.json_normalize(stats)
            df['week'] = pd.to_numeric(df['week'])
            df['season'] = year
            all_game_stats.append(df)

    if not all_game_stats: return

    df_stats_raw = pd.concat(all_game_stats, ignore_index=True)
    
    print("Applying EWMA to Rushing/Passing splits...")
    df_decay = calculate_weighted_decay(df_stats_raw)
    
    # Keep only the new decay columns
    cols_to_keep = ['team', 'season', 'week'] + [c for c in df_decay.columns if 'decay_' in c]
    df_features = df_decay[cols_to_keep].copy()

    print("Merging granular features...")
    
    # Surgical Merge (Home)
    rename_home = {c: f"home_{c}" for c in df_features.columns if 'decay_' in c}
    rename_home['team'] = 'home_team'
    home_feats = df_features.rename(columns=rename_home)
    
    df_master = pd.merge(df_master, home_feats, on=['home_team', 'season', 'week'], how='left')
    
    # Surgical Merge (Away)
    rename_away = {c: f"away_{c}" for c in df_features.columns if 'decay_' in c}
    rename_away['team'] = 'away_team'
    away_feats = df_features.rename(columns=rename_away)
    
    df_master = pd.merge(df_master, away_feats, on=['away_team', 'season', 'week'], how='left')

    # Final cleanup of NaNs
    decay_cols = [c for c in df_master.columns if 'decay_' in c]
    df_master[decay_cols] = df_master[decay_cols].fillna(0.0)
    
    output_filename = "cfb_training_data_granular.csv"
    df_master.to_csv(output_filename, index=False)
    print(f"\nSUCCESS: Saved GRANULAR dataset to {output_filename}")
    
    # DEBUG CHECK
    print("\nVerifying Columns:")
    check_col = 'home_decay_offense.rushing.ppa'
    if check_col in df_master.columns:
        print(f"✅ {check_col} exists!")
    else:
        print(f"❌ {check_col} is STILL MISSING. Something is wrong.")

if __name__ == "__main__":
    main()