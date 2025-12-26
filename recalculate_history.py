import pandas as pd
import joblib
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"

def get_data(endpoint, params):
    try:
        response = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except:
        return []

def build_decay_lookup(year):
    print("   -> Fetching team stats for feature generation...")
    stats = get_data("/stats/game/advanced", {"year": year})
    if not stats: return {}
    
    df = pd.json_normalize(stats)
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'])
        df = df.sort_values(['team', 'season', 'week'])
    
    metrics = ['offense.ppa', 'offense.successRate', 'offense.explosiveness', 'defense.ppa', 'defense.successRate', 'defense.explosiveness']
    lookup = {}
    
    for team, group in df.groupby('team'):
        team_mom = {}
        for m in metrics:
            if m in group.columns:
                team_mom[f"decay_{m}"] = group[m].ewm(span=3, adjust=False).mean().iloc[-1]
            else:
                team_mom[f"decay_{m}"] = 0.0
        lookup[team] = team_mom
    return lookup

def main():
    print("--- 🧠 RECALCULATING HISTORICAL TOTALS ---")
    
    # 1. Load Model
    try:
        model_total = joblib.load("model_total.pkl")
        model_spread = joblib.load("model_spread_tuned.pkl") # Needed for feature names
    except:
        print("❌ Models not found. Run pipeline first.")
        return

    # 2. Load History
    if not os.path.exists(HISTORY_FILE):
        print("❌ No history file found.")
        return
    
    df = pd.read_csv(HISTORY_FILE)
    
    # 3. Fetch Stats Context
    YEAR = 2025
    srs_data = get_data("/ratings/srs", {"year": YEAR})
    talent_data = get_data("/talent", {"year": YEAR})
    srs_map = {x['team']: x['rating'] for x in srs_data}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data}
    decay_map = build_decay_lookup(YEAR)
    
    updates_count = 0
    
    print("   -> Re-running model on backfilled games...")
    
    for index, row in df.iterrows():
        # Only target the "Backfill" rows that have the dummy totals
        if row.get('Spread Book') == 'Backfill':
            
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            # Reconstruct Features
            h_d = decay_map.get(home)
            a_d = decay_map.get(away)
            
            # If we can't find stats (e.g. FCS teams), skip
            if not h_d or not a_d:
                continue
                
            # Build Feature Row (Standard Line 55.5)
            model_row = {
                'spread': float(row['Pick_Line']), # Use the spread we already picked
                'overUnder': 55.5, # The standard line we used for backfill
                'home_talent_score': talent_map.get(home, 10), 
                'away_talent_score': talent_map.get(away, 10),
                'home_srs_rating': srs_map.get(home, -5), 
                'away_srs_rating': srs_map.get(away, -5),
                **{f"home_{k}":v for k,v in h_d.items()}, 
                **{f"away_{k}":v for k,v in a_d.items()}
            }
            
            # Predict
            feat_cols = model_spread.feature_names_in_
            try:
                features = pd.DataFrame([model_row])[feat_cols]
                over_prob = model_total.predict_proba(features)[0][1]
                
                # Logic: Over vs Under
                pick_side = "OVER" if over_prob > 0.5 else "UNDER"
                conf = max(over_prob, 1 - over_prob)
                
                # Update DataFrame
                df.at[index, 'Total Pick'] = f"{pick_side} 55.5"
                df.at[index, 'Pick_Side'] = pick_side
                df.at[index, 'Total Conf'] = f"{conf:.1%}"
                df.at[index, 'Total_Conf_Raw'] = conf
                
                updates_count += 1
                
            except Exception as e:
                print(f"Skipping {home} vs {away}: {e}")

    # 4. Save
    df.to_csv(HISTORY_FILE, index=False)
    print(f"✅ Successfully updated {updates_count} historical games with REAL model predictions.")

if __name__ == "__main__":
    main()