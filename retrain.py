import os
import time
import pandas as pd
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# The exact features your engine uses
FEATURES = [
    'spread', 'overUnder',
    'home_talent_score', 'away_talent_score',
    'home_srs_rating', 'away_srs_rating'
]

def fetch_with_retry(endpoint, params):
    url = f"https://api.collegefootballdata.com{endpoint}"
    for attempt in range(1, 4):
        try:
            res = requests.get(url, headers=HEADERS, params=params)
            if res.status_code == 200: return res.json()
            elif res.status_code == 429:
                print(f"   ⚠️ Rate Limit. Cooling down ({10*attempt}s)...")
                time.sleep(10 * attempt)
        except: time.sleep(5)
    return []

def main():
    print("--- 🧠 TRAINING ON COMPLETED GAMES (INCLUDING BOWLS) ---")
    
    all_games = []
    
    # 1. Fetch 2024 and 2025 Data
    for year in [2024, 2025]:
        print(f"   -> Fetching {year} season data...")
        games = fetch_with_retry("/games", {"year": year, "seasonType": "both"})
        lines = fetch_with_retry("/lines", {"year": year, "seasonType": "both"})
        srs = fetch_with_retry("/ratings/srs", {"year": year})
        talent = fetch_with_retry("/talent", {"year": year})
        
        # Maps
        line_map = {}
        if isinstance(lines, list):
            for g in lines:
                if g.get('lines'): line_map[str(g['id'])] = g['lines'][0]
        
        srs_map = {x['team']: x['rating'] for x in srs} if isinstance(srs, list) else {}
        tal_map = {x.get('school', x.get('team')): x['talent'] for x in talent} if isinstance(talent, list) else {}

        if isinstance(games, list):
            for g in games:
                if not g.get('completed'): continue 
                
                gid = str(g['id'])
                
                # --- UNIVERSAL KEY HANDLING ---
                # Checks both snake_case (home_team) and camelCase (homeTeam)
                home = g.get('home_team') or g.get('homeTeam')
                away = g.get('away_team') or g.get('awayTeam')
                h_pts = g.get('home_points') or g.get('homePoints')
                a_pts = g.get('away_points') or g.get('awayPoints')
                
                if not home or not away or h_pts is None or a_pts is None: continue

                # Get Stats
                h_srs, a_srs = srs_map.get(home, 0), srs_map.get(away, 0)
                h_tal, a_tal = tal_map.get(home, 10), tal_map.get(away, 10)
                
                # Get Lines
                line_data = line_map.get(gid, {})
                spread = line_data.get('spread')
                total = line_data.get('overUnder')
                
                if spread is None or total is None: continue
                
                all_games.append({
                    'spread': spread,
                    'overUnder': total,
                    'home_talent_score': h_tal, 'away_talent_score': a_tal,
                    'home_srs_rating': h_srs, 'away_srs_rating': a_srs,
                    'home_points': h_pts,
                    'away_points': a_pts
                })

    # 2. Prepare Training Data
    if not all_games:
        print("❌ No data found to train on.")
        return

    df = pd.DataFrame(all_games)
    print(f"   -> Found {len(df)} games with valid data.")
    
    # Define Targets
    df['target_cover'] = ((df['home_points'] + df['spread']) > df['away_points']).astype(int)
    df['target_over'] = ((df['home_points'] + df['away_points']) > df['overUnder']).astype(int)
    
    X = df[FEATURES]
    
    # 3. Train Models
    print("   -> Training Spread Model...")
    model_spread = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model_spread.fit(X, df['target_cover'])
    
    print("   -> Training Total Model...")
    model_total = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_total.fit(X, df['target_over'])
    
    # 4. Save
    model_spread.feature_names_in_ = FEATURES
    model_total.feature_names_in_ = FEATURES
    
    joblib.dump(model_spread, "model_spread_tuned.pkl")
    joblib.dump(model_total, "model_total.pkl")
    
    print("✅ SUCCESS: Models updated with latest bowl results!")

if __name__ == "__main__":
    main()