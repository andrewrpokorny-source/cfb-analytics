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
    print("--- 🔋 INJECTING POWER RANKINGS (SRS) 🔋 ---")
    
    # 1. Load your Momentum Data (The most recent version)
    try:
        df = pd.read_csv("cfb_training_data_with_momentum.csv")
        print(f"Loaded {len(df)} games.")
    except FileNotFoundError:
        print("Error: cfb_training_data_with_momentum.csv not found. Run features.py first!")
        return

    # 2. Fetch SRS Ratings for all years in the data
    years = df['season'].unique()
    all_ratings = []
    
    print("Fetching SRS (Simple Rating System) data...")
    print("This metric mathematically adjusts for Strength of Schedule.")

    for year in years:
        # Fetch SRS from API
        ratings = get_data("/ratings/srs", {"year": year})
        
        if ratings:
            df_r = pd.DataFrame(ratings)
            df_r['season'] = year
            # We only need the total rating (which combines offense/defense/SOS)
            df_r = df_r[['team', 'season', 'rating']].rename(columns={'rating': 'srs_rating'})
            all_ratings.append(df_r)
            
    if not all_ratings:
        print("Failed to fetch ratings.")
        return

    df_ratings = pd.concat(all_ratings, ignore_index=True)
    
    # 3. Merge SRS into Master Data
    print("Merging Power Ratings into Training Data...")
    
    # Merge Home Team SRS
    df = pd.merge(
        df,
        df_ratings.add_prefix("home_"), 
        left_on=['home_team', 'season'],
        right_on=['home_team', 'home_season'], 
        how='left'
    )
    
    # Merge Away Team SRS
    df = pd.merge(
        df,
        df_ratings.add_prefix("away_"),
        left_on=['away_team', 'season'],
        right_on=['away_team', 'away_season'], 
        how='left'
    )
    
    # Fill missing values (FCS teams might not have an SRS rating) with a low default
    # An average FBS team is 0.0. A bad FCS team is -15.0.
    df['home_srs_rating'] = df['home_srs_rating'].fillna(-10.0)
    df['away_srs_rating'] = df['away_srs_rating'].fillna(-10.0)
    
    # 4. Save
    output_filename = "cfb_training_data_final.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nSUCCESS: Saved final dataset to {output_filename}")
    print("New Columns: ['home_srs_rating', 'away_srs_rating']")

if __name__ == "__main__":
    main()