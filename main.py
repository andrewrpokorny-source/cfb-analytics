import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Check your .env file.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

def get_data(endpoint, params):
    """
    Generic helper to fetch data from CFBD API with rate limiting.
    """
    url = f"https://api.collegefootballdata.com{endpoint}"
    try:
        print(f"Fetching {endpoint} with params {params}...")
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        time.sleep(0.5) 
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def normalize_game_columns(df):
    """
    Helper to standardize column names (handling snake_case vs camelCase).
    """
    # Map common camelCase variations to snake_case
    rename_map = {
        'homeTeam': 'home_team',
        'awayTeam': 'away_team',
        'homePoints': 'home_points',
        'awayPoints': 'away_points',
        'homeScore': 'home_points', # Sometimes it's called score
        'awayScore': 'away_points'
    }
    df = df.rename(columns=rename_map)
    return df

def fetch_season_data(years):
    all_games = []
    all_lines = []
    all_stats = []

    for year in years:
        # --- 1. FETCH GAMES (Scores) ---
        for season_type in ["regular", "postseason"]:
            games = get_data("/games", {"year": year, "seasonType": season_type})
            
            if games:
                df_games = pd.DataFrame(games)
                
                # FIX: Normalize columns before selecting
                df_games = normalize_game_columns(df_games)
                
                # Check if we have the required columns now
                required_cols = ['id', 'season', 'week', 'home_team', 'away_team', 'home_points', 'away_points']
                
                # Safety Check: If columns are still missing, print debug info and skip
                missing_cols = [c for c in required_cols if c not in df_games.columns]
                if missing_cols:
                    print(f"\n⚠️ WARNING: Missing columns in {year} {season_type} games: {missing_cols}")
                    print(f"Available columns: {df_games.columns.tolist()[:10]}...") 
                    continue
                
                # Keep only finished games
                if 'completed' in df_games.columns:
                    df_games = df_games[df_games['completed'] == True]
                
                df_games = df_games[required_cols].copy()
                all_games.append(df_games)

            # --- 2. FETCH BETTING LINES ---
            lines = get_data("/lines", {"year": year, "seasonType": season_type})
            
            if lines:
                processed_lines = []
                for game in lines:
                    game_id = game.get('id')
                    lines_list = game.get('lines', [])
                    if lines_list:
                        # Grab the first available provider
                        line_data = lines_list[0] 
                        processed_lines.append({
                            'id': game_id,
                            'spread': line_data.get('spread'),
                            'overUnder': line_data.get('overUnder')
                        })
                
                if processed_lines:
                    all_lines.append(pd.DataFrame(processed_lines))

        # --- 3. FETCH ADVANCED STATS (Features) ---
        stats = get_data("/stats/season/advanced", {"year": year})
        if stats:
            df_stats = pd.json_normalize(stats)
            df_stats['season'] = year
            all_stats.append(df_stats)

    # --- MERGE EVERYTHING ---
    print("\nProcessing and Merging Data...")
    
    if not all_games: 
        print("No game data found. Exiting.")
        return None
        
    final_games = pd.concat(all_games, ignore_index=True)
    
    final_lines = pd.DataFrame()
    if all_lines:
        final_lines = pd.concat(all_lines, ignore_index=True)
        
    final_stats = pd.concat(all_stats, ignore_index=True)

    # Merge Games + Lines
    master_df = pd.merge(final_games, final_lines, on='id', how='left')

    # Merge Home Team Stats
    final_stats_home = final_stats.add_prefix("home_")
    master_df = pd.merge(
        master_df, 
        final_stats_home, 
        left_on=['season', 'home_team'], 
        right_on=['home_season', 'home_team'], 
        how='left'
    )

    # Merge Away Team Stats
    final_stats_away = final_stats.add_prefix("away_")
    master_df = pd.merge(
        master_df, 
        final_stats_away, 
        left_on=['season', 'away_team'], 
        right_on=['away_season', 'away_team'], 
        how='left'
    )

    return master_df

if __name__ == "__main__":
    YEARS_TO_FETCH = [2024, 2025]
    
    print(f"Starting extraction for years: {YEARS_TO_FETCH}")
    df = fetch_season_data(YEARS_TO_FETCH)
    
    if df is not None and not df.empty:
        # Drop games where we couldn't find stats
        # We use 'home_offense.ppa' (Predictive Points Added) as a proxy for 'stats exist'
        df_clean = df.dropna(subset=['home_offense.ppa', 'away_offense.ppa'])
        
        # --- CALCULATE TARGETS ---
        df_clean['target_home_win'] = (df_clean['home_points'] > df_clean['away_points']).astype(int)
        
        # Check if spread exists before calculating cover
        if 'spread' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['spread'])
            # Logic: If Home (-7) scores 28 and Away scores 10. 28 + (-7) = 21 > 10. Cover = True.
            df_clean['target_home_cover'] = ((df_clean['home_points'] + df_clean['spread']) > df_clean['away_points']).astype(int)
        
        if 'overUnder' in df_clean.columns:
             df_clean['target_over'] = ((df_clean['home_points'] + df_clean['away_points']) > df_clean['overUnder']).astype(int)

        print("\n--- DATA PREVIEW ---")
        cols_to_show = ['season', 'week', 'home_team', 'away_team', 'home_points', 'target_home_win']
        # Only show cols that actually exist
        cols_to_show = [c for c in cols_to_show if c in df_clean.columns]
        print(df_clean[cols_to_show].tail())
        
        filename = "cfb_training_data_24_25.csv"
        df_clean.to_csv(filename, index=False)
        print(f"\nSUCCESS: Saved {len(df_clean)} games to {filename}")
    else:
        print("Failed to acquire data.")