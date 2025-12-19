import pandas as pd
import joblib
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"

# 1. Map Shorthand Names to API Official Names
NAME_FIXES = {
    "Jax State": "Jacksonville State",
    "Kennesaw St": "Kennesaw State",
    "W. Michigan": "Western Michigan",
    "SC State": "South Carolina State",
    "Missouri St": "Missouri State",
    "Arkansas St": "Arkansas State",
    "N. Illinois": "Northern Illinois",
    "E. Michigan": "Eastern Michigan"
}

def get_data(endpoint, params):
    try:
        response = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        return response.json()
    except: return []

def build_decay_lookup(year):
    print("   -> Fetching 2025 stats...")
    stats = get_data("/stats/game/advanced", {"year": year})
    if not stats: return {}
    df = pd.json_normalize(stats)
    
    # Simple lookup dict
    lookup = {}
    metrics = ['offense.ppa', 'offense.successRate', 'offense.explosiveness', 'defense.ppa', 'defense.successRate', 'defense.explosiveness']
    
    for team, group in df.groupby('team'):
        # Take the average of the last 3 games as the "Decay" proxy
        # (Simplified for backfill speed)
        team_mom = {}
        for m in metrics:
            if m in group.columns:
                team_mom[f"decay_{m}"] = group[m].mean()
            else:
                team_mom[f"decay_{m}"] = 0.0
        lookup[team] = team_mom
    return lookup

def main():
    print("--- 🔧 FIXING NAMES & RECALCULATING TOTALS ---")
    
    # Load
    try:
        model_total = joblib.load("model_total.pkl")
        model_spread = joblib.load("model_spread_tuned.pkl") # For column names
    except:
        print("❌ Models missing. Run pipeline.")
        return

    if not os.path.exists(HISTORY_FILE):
        print("❌ History file missing.")
        return
    
    df = pd.read_csv(HISTORY_FILE)
    
    # Context
    YEAR = 2025
    srs_data = get_data("/ratings/srs", {"year": YEAR})
    talent_data = get_data("/talent", {"year": YEAR})
    
    srs_map = {x['team']: x['rating'] for x in srs_data}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data}
    decay_map = build_decay_lookup(YEAR)
    
    fixed_count = 0
    
    for index, row in df.iterrows():
        # Only touch Backfill rows
        if row.get('Spread Book') == 'Backfill':
            
            # 1. Fix Names
            h = row['HomeTeam']
            a = row['AwayTeam']
            
            if h in NAME_FIXES: h = NAME_FIXES[h]
            if a in NAME_FIXES: a = NAME_FIXES[a]
            
            # Save fixed names back to DF so they look professional
            df.at[index, 'HomeTeam'] = h
            df.at[index, 'AwayTeam'] = a
            df.at[index, 'Game'] = f"{a} @ {h}"

            # 2. Re-Predict
            h_d = decay_map.get(h)
            a_d = decay_map.get(a)
            
            if not h_d or not a_d:
                print(f"⚠️ Stats still missing for {a} @ {h} - Keeping default.")
                continue

            model_row = {
                'spread': float(row['Pick_Line']),
                'overUnder': 55.5, # Standard line
                'home_talent_score': talent_map.get(h, 10), 
                'away_talent_score': talent_map.get(a, 10),
                'home_srs_rating': srs_map.get(h, -5), 
                'away_srs_rating': srs_map.get(a, -5),
                **{f"home_{k}":v for k,v in h_d.items()}, 
                **{f"away_{k}":v for k,v in a_d.items()}
            }
            
            # Fill missing columns with 0
            for col in model_spread.feature_names_in_:
                if col not in model_row: model_row[col] = 0.0

            try:
                features = pd.DataFrame([model_row])[model_spread.feature_names_in_]
                over_prob = model_total.predict_proba(features)[0][1]
                
                pick_side = "OVER" if over_prob > 0.5 else "UNDER"
                conf = max(over_prob, 1 - over_prob)
                
                # UPDATE THE CSV
                df.at[index, 'Total Pick'] = f"{pick_side} 55.5"
                df.at[index, 'Pick_Side'] = pick_side
                df.at[index, 'Total Conf'] = f"{conf:.1%}"
                df.at[index, 'Total_Conf_Raw'] = conf
                
                fixed_count += 1
            except Exception as e:
                print(f"Error predicting {h} vs {a}: {e}")

    df.to_csv(HISTORY_FILE, index=False)
    print(f"✅ Fixed & Recalculated {fixed_count} games.")

if __name__ == "__main__":
    main()