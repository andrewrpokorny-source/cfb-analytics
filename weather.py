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
        time.sleep(0.3)
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def main():
    print("--- 🌪️ FETCHING WEATHER DATA 🌪️ ---")
    
    # 1. Load Existing Data
    try:
        df = pd.read_csv("cfb_training_data_ultimate.csv")
        print(f"Loaded {len(df)} games.")
    except FileNotFoundError:
        print("Error: cfb_training_data_ultimate.csv not found. Run talent.py first!")
        return

    # 2. Fetch Weather for Each Season
    years = df['season'].unique()
    all_weather = []
    
    print("Fetching game weather (Wind, Temp)...")

    for year in years:
        # Fetch weather for the entire season
        # Note: 'seasonType' param is often needed, we'll try 'regular' and 'postseason'
        for stype in ['regular', 'postseason']:
            weather_data = get_data("/games/weather", {"year": year, "seasonType": stype})
            
            if weather_data:
                df_w = pd.DataFrame(weather_data)
                
                # Keep relevant columns
                # 'weatherConditionCode' is numeric (1=Clear, 2=Cloudy, etc). 
                # We care most about numeric physical effects: windSpeed, temperature
                cols_to_keep = ['id', 'temperature', 'windSpeed', 'weatherConditionCode']
                
                # Filter for cols that exist
                existing_cols = [c for c in cols_to_keep if c in df_w.columns]
                df_w = df_w[existing_cols].copy()
                
                # Rename 'id' to 'id_game' or just merge on 'id' if your master uses 'id'
                # Your master uses 'id' for game_id
                all_weather.append(df_w)
            
    if not all_weather:
        print("Failed to fetch weather data.")
        return

    df_weather = pd.concat(all_weather, ignore_index=True)
    
    # Remove duplicates (sometimes API returns same game in multiple queries)
    df_weather = df_weather.drop_duplicates(subset=['id'])
    
    # 3. Merge Weather into Master Data
    print("Merging Weather Data...")
    
    # Merge on Game ID
    df = pd.merge(df, df_weather, on='id', how='left')
    
    # 4. Clean Missing Data (Domed stadiums or missing records)
    # If weather is missing, assume "Average/Indoor" conditions
    # Temp: 70F, Wind: 0mph
    df['temperature'] = df['temperature'].fillna(70.0)
    df['windSpeed'] = df['windSpeed'].fillna(0.0)
    
    # 5. Save
    output_filename = "cfb_training_data_weather.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nSUCCESS: Saved dataset to {output_filename}")
    print("New Columns: ['temperature', 'windSpeed']")

if __name__ == "__main__":
    main()