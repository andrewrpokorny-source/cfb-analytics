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
SPLIT_DATE = "2025-12-01" # Train on everything before this, Test on everything after
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
    print("--- ⚖️ RUNNING HONEST BACKFILL (NO CHEATING) ---")
    
    # 1. Fetch ALL Data (Training + Testing together)
    print("   -> Fetching full season data...")
    all_games = []
    
    # Grab 2024 (Historical) + 2025 (Current)
    for year in [2024, 2025]:
        games = fetch_with_retry("/games", {"year": year, "seasonType": "both"})
        lines = fetch_with_retry("/lines", {"year": year, "seasonType": "both"})
        srs = fetch_with_retry("/ratings/srs", {"year": year})
        talent = fetch_with_retry("/talent", {"year": year})
        
        # Build Maps
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
                
                # Universal Key Handling
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
    print(f"   -> Total Games Found: {len(df)}")

    # 2. THE SPLIT (Crucial Step)
    # Train = Before Dec 1st
    # Test  = After Dec 1st
    train_df = df[df['StartDate'] < SPLIT_DATE].copy()
    test_df = df[df['StartDate'] >= SPLIT_DATE].copy()
    
    print(f"   -> Training Set: {len(train_df)} games (Pre-Dec 1)")
    print(f"   -> Testing Set:  {len(test_df)} games (Post-Dec 1)")
    
    # 3. Train "Honest" Models
    print("   -> Training temporary models on past data only...")
    
    # Targets
    train_df['target_cover'] = ((train_df['Manual_HomeScore'] + train_df['spread']) > train_df['Manual_AwayScore']).astype(int)
    train_df['target_over'] = ((train_df['Manual_HomeScore'] + train_df['Manual_AwayScore']) > train_df['overUnder']).astype(int)
    
    X_train = train_df[FEATURES]
    
    model_spread = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model_spread.fit(X_train, train_df['target_cover'])
    
    model_total = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_total.fit(X_train, train_df['target_over'])
    
    # 4. Predict the "Future" (The Test Set)
    print("   -> Grading December games...")
    history_rows = []
    
    for idx, row in test_df.iterrows():
        # Prepare Input
        input_row = pd.DataFrame([row[FEATURES]])
        
        # Predict Spread
        prob_spr = model_spread.predict_proba(input_row)[0][1]
        conf_spr = max(prob_spr, 1-prob_spr)
        pick_team = row['HomeTeam'] if prob_spr > 0.5 else row['AwayTeam']
        pick_line = row['spread'] if prob_spr > 0.5 else -row['spread']
        
        # Predict Total
        prob_tot = model_total.predict_proba(input_row)[0][1]
        conf_tot = max(prob_tot, 1-prob_tot)
        pick_side = "OVER" if prob_tot > 0.5 else "UNDER"
        
        # Moneyline (Heuristic)
        ml_pick = row['HomeTeam'] if (row['home_srs_rating'] + row['home_talent_score']/200) > (row['away_srs_rating'] + row['away_talent_score']/200) else row['AwayTeam']
        
        history_rows.append({
            "GameID": row['GameID'],
            "HomeTeam": row['HomeTeam'], "AwayTeam": row['AwayTeam'],
            "Game": f"{row['AwayTeam']} @ {row['HomeTeam']}",
            "StartDate": row['StartDate'],
            "Moneyline Pick": ml_pick, "Moneyline Conf": "N/A",
            "Spread Pick": f"{pick_team} ({pick_line})", "Spread Conf": f"{conf_spr:.1%}",
            "Total Pick": f"{pick_side} {row['overUnder']}", "Total Conf": f"{conf_tot:.1%}",
            "Pick_Team": pick_team, "Pick_Line": pick_line,
            "Pick_Side": pick_side, "Pick_Total": row['overUnder'],
            "Manual_HomeScore": row['Manual_HomeScore'],
            "Manual_AwayScore": row['Manual_AwayScore']
        })

    # 5. Merge & Save
    # We keep the FUTURE predictions from live_predictions.csv but overwrite the HISTORY
    try:
        existing = pd.read_csv("live_predictions.csv")
        future = existing[existing['Manual_HomeScore'].isna()]
    except:
        future = pd.DataFrame()
        
    final_df = pd.concat([future, pd.DataFrame(history_rows)], ignore_index=True)
    final_df.to_csv("live_predictions.csv", index=False)
    
    print(f"✅ SUCCESS: Honest History generated. {len(history_rows)} games graded.")

if __name__ == "__main__":
    main()