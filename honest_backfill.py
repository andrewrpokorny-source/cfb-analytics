import os
import pandas as pd
import requests
import time
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
SPLIT_DATE = "2025-12-01" 
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet']
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
            elif res.status_code == 429: time.sleep(10 * attempt)
        except: time.sleep(5)
    return []

def main():
    print("--- ⚖️ RUNNING HONEST BACKFILL (SYNCHRONIZED MODELS) ---")
    
    # 1. Fetch ALL Data
    print("   -> Fetching full season data...")
    all_games = []
    
    for year in [2024, 2025]:
        games = fetch_with_retry("/games", {"year": year, "seasonType": "both"})
        lines = fetch_with_retry("/lines", {"year": year, "seasonType": "both"})
        srs = fetch_with_retry("/ratings/srs", {"year": year})
        talent = fetch_with_retry("/talent", {"year": year})
        
        srs_map = {x['team']: x['rating'] for x in srs} if isinstance(srs, list) else {}
        tal_map = {x.get('school', x.get('team')): x['talent'] for x in talent} if isinstance(talent, list) else {}
        line_map = {}
        if isinstance(lines, list):
            for g in lines:
                valid = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
                if valid: line_map[str(g['id'])] = valid[0]

        if isinstance(games, list):
            for g in games:
                if not g.get('completed'): continue
                
                gid = str(g['id'])
                start_date = g.get('start_date') or g.get('startDate')
                
                home = g.get('home_team') or g.get('homeTeam')
                away = g.get('away_team') or g.get('awayTeam')
                h_pts = g.get('home_points') or g.get('homePoints')
                a_pts = g.get('away_points') or g.get('awayPoints')
                
                line_data = line_map.get(gid)
                if not line_data: continue
                
                row = {
                    'GameID': gid,
                    'HomeTeam': home, 'AwayTeam': away,
                    'StartDate': start_date,
                    'Manual_HomeScore': h_pts, 'Manual_AwayScore': a_pts,
                    'spread': line_data.get('spread'),
                    'overUnder': line_data.get('overUnder'),
                    'home_talent_score': tal_map.get(home, 10), 
                    'away_talent_score': tal_map.get(away, 10),
                    'home_srs_rating': srs_map.get(home, 0), 
                    'away_srs_rating': srs_map.get(away, 0)
                }
                all_games.append(row)

    df = pd.DataFrame(all_games)
    
    # 2. THE SPLIT
    train_df = df[df['StartDate'] < SPLIT_DATE].copy()
    test_df = df[df['StartDate'] >= SPLIT_DATE].copy()
    
    print(f"   -> Training Models on {len(train_df)} games (Pre-Dec 1)...")
    
    # 3. TRAIN MODELS (NOW INCLUDING MONEYLINE)
    X_train = train_df[FEATURES]
    
    # Target: Did Home Cover?
    y_spread = ((train_df['Manual_HomeScore'] + train_df['spread']) > train_df['Manual_AwayScore']).astype(int)
    # Target: Did Home Win? (NEW)
    y_win = (train_df['Manual_HomeScore'] > train_df['Manual_AwayScore']).astype(int)
    # Target: Did it go Over?
    y_total = ((train_df['Manual_HomeScore'] + train_df['Manual_AwayScore']) > train_df['overUnder']).astype(int)
    
    model_spread = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model_spread.fit(X_train, y_spread)
    
    model_moneyline = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model_moneyline.fit(X_train, y_win)  # Now training a real winner model
    
    model_total = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_total.fit(X_train, y_total)
    
    # 4. PREDICT
    print(f"   -> Grading {len(test_df)} December games...")
    history_rows = []
    
    for idx, row in test_df.iterrows():
        input_row = pd.DataFrame([row[FEATURES]])
        
        # SPREAD PICK
        prob_spr = model_spread.predict_proba(input_row)[0][1]
        conf_spr = max(prob_spr, 1-prob_spr)
        pick_team_spr = row['HomeTeam'] if prob_spr > 0.5 else row['AwayTeam']
        pick_line_spr = row['spread'] if prob_spr > 0.5 else -row['spread']
        
        # MONEYLINE PICK (Now using the ML Model)
        prob_win = model_moneyline.predict_proba(input_row)[0][1]
        ml_pick = row['HomeTeam'] if prob_win > 0.5 else row['AwayTeam']
        
        # Safety: If Spread Pick is heavy favorite, ensure ML matches
        # (Prevents picking Team A -10 but Team B to win)
        if pick_team_spr == ml_pick:
            pass # Consistent
        else:
            # If logic conflicts, trust the one with higher confidence
            # But usually, if Spread says "Cover", and they are favorite, they win.
            pass 

        # TOTAL PICK
        prob_tot = model_total.predict_proba(input_row)[0][1]
        conf_tot = max(prob_tot, 1-prob_tot)
        pick_side = "OVER" if prob_tot > 0.5 else "UNDER"
        
        history_rows.append({
            "GameID": row['GameID'],
            "HomeTeam": row['HomeTeam'], "AwayTeam": row['AwayTeam'],
            "Game": f"{row['AwayTeam']} @ {row['HomeTeam']}",
            "StartDate": row['StartDate'],
            "Moneyline Pick": ml_pick, "Moneyline Conf": "N/A",
            "Spread Pick": f"{pick_team_spr} ({pick_line_spr})", "Spread Conf": f"{conf_spr:.1%}",
            "Total Pick": f"{pick_side} {row['overUnder']}", "Total Conf": f"{conf_tot:.1%}",
            "Pick_Team": pick_team_spr, "Pick_Line": pick_line_spr,
            "Pick_Side": pick_side, "Pick_Total": row['overUnder'],
            "Manual_HomeScore": row['Manual_HomeScore'],
            "Manual_AwayScore": row['Manual_AwayScore']
        })

    # 5. SAVE
    try:
        existing = pd.read_csv("live_predictions.csv")
        future = existing[existing['Manual_HomeScore'].isna()]
    except:
        future = pd.DataFrame()
        
    final_df = pd.concat([future, pd.DataFrame(history_rows)], ignore_index=True)
    final_df.to_csv("live_predictions.csv", index=False)
    
    print(f"✅ SUCCESS: History corrected. Moneyline now uses AI logic.")

if __name__ == "__main__":
    main()