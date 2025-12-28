import os
import pandas as pd
import requests
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet']

# V2 FEATURES (EPA Enriched)
FEATURES = [
    'spread', 'overUnder', 
    'home_talent_score', 'away_talent_score', 
    'home_srs_rating', 'away_srs_rating',
    'home_off_epa', 'away_off_epa',
    'home_def_epa', 'away_def_epa',
    'home_success_rate', 'away_success_rate'
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
    print("--- 💰 RUNNING V2 PROFIT SIMULATION (EPA MODEL) 💰 ---")
    
    # 1. Fetch Data
    print("   -> Fetching full season data with Advanced Stats...")
    all_games = []
    
    for year in [2024, 2025]:
        games = fetch_with_retry("/games", {"year": year, "seasonType": "both"})
        lines = fetch_with_retry("/lines", {"year": year, "seasonType": "both"})
        srs = fetch_with_retry("/ratings/srs", {"year": year})
        talent = fetch_with_retry("/talent", {"year": year})
        adv = fetch_with_retry("/stats/season/advanced", {"year": year, "excludeGarbageTime": "true"})
        
        # Build Maps
        srs_map = {x['team']: x['rating'] for x in srs} if isinstance(srs, list) else {}
        tal_map = {x.get('school', x.get('team')): x['talent'] for x in talent} if isinstance(talent, list) else {}
        
        adv_map = {}
        if isinstance(adv, list):
            for t in adv:
                adv_map[t['team']] = {
                    'off_epa': t['offense']['ppa'],
                    'def_epa': t['defense']['ppa'],
                    'success_rate': t['offense']['successRate']
                }

        line_map = {}
        if isinstance(lines, list):
            for g in lines:
                valid = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
                if valid: line_map[str(g['id'])] = valid[0]

        if isinstance(games, list):
            for g in games:
                if not g.get('completed'): continue
                
                gid = str(g['id'])
                home = g.get('home_team') or g.get('homeTeam')
                away = g.get('away_team') or g.get('awayTeam')
                h_pts = g.get('home_points') or g.get('homePoints')
                a_pts = g.get('away_points') or g.get('awayPoints')
                start_date = g.get('start_date') or g.get('startDate')

                if not home or not away or h_pts is None: continue

                line_data = line_map.get(gid)
                if not line_data: continue

                # Get EPA Stats
                h_adv = adv_map.get(home, {'off_epa': 0, 'def_epa': 0, 'success_rate': 0})
                a_adv = adv_map.get(away, {'off_epa': 0, 'def_epa': 0, 'success_rate': 0})

                all_games.append({
                    'StartDate': start_date,
                    'Manual_HomeScore': h_pts, 'Manual_AwayScore': a_pts,
                    'spread': line_data.get('spread'),
                    'overUnder': line_data.get('overUnder'),
                    'home_talent_score': tal_map.get(home, 10), 
                    'away_talent_score': tal_map.get(away, 10),
                    'home_srs_rating': srs_map.get(home, 0), 
                    'away_srs_rating': srs_map.get(away, 0),
                    # V2 Features
                    'home_off_epa': h_adv['off_epa'], 'away_off_epa': a_adv['off_epa'],
                    'home_def_epa': h_adv['def_epa'], 'away_def_epa': a_adv['def_epa'],
                    'home_success_rate': h_adv['success_rate'], 'away_success_rate': a_adv['success_rate']
                })

    df = pd.DataFrame(all_games)
    print(f"   -> Analyzing {len(df)} total games...")

    # 2. TIME SERIES SPLIT (The "Honest" Test)
    # We train on games BEFORE Dec 1, and test on games AFTER Dec 1
    SPLIT_DATE = "2025-12-01"
    train_df = df[df['StartDate'] < SPLIT_DATE].copy()
    test_df = df[df['StartDate'] >= SPLIT_DATE].copy()
    
    print(f"   -> Training on {len(train_df)} games (Pre-Dec 1)...")
    print(f"   -> Testing on {len(test_df)} games (Post-Dec 1)...")
    
    # Train Spread Model
    X_train = train_df[FEATURES]
    y_train = ((train_df['Manual_HomeScore'] + train_df['spread']) > train_df['Manual_AwayScore']).astype(int)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 3. RUN SIMULATION
    print("\n--- 📊 PERFORMANCE REPORT (V2) ---")
    bets = 0
    wins = 0
    losses = 0
    bankroll = 0.0
    
    for _, row in test_df.iterrows():
        # Predict
        input_row = pd.DataFrame([row])[FEATURES]
        prob = model.predict_proba(input_row)[0][1]
        conf = max(prob, 1-prob)
        
        # Strategy: Bet if Conf > 55%
        if conf > 0.55:
            bets += 1
            
            # Did Home Cover?
            home_covered = (row['Manual_HomeScore'] + row['spread']) > row['Manual_AwayScore']
            
            # Our Pick
            pick_home = (prob > 0.5)
            
            if pick_home == home_covered:
                wins += 1
                bankroll += 90.91 # Win $90.91 on $100 bet (-110 odds)
            else:
                losses += 1
                bankroll -= 100.00 # Lose $100

    win_rate = (wins/bets*100) if bets > 0 else 0.0
    roi = (bankroll / (bets * 100) * 100) if bets > 0 else 0.0

    print(f"Total Bets Placed: {bets}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}% (V1 Benchmark: 56.9%)")
    print(f"Net Profit: ${bankroll:.2f}")
    print(f"ROI: {roi:.1f}%")
    
    if win_rate > 56.9:
        print("\n✅ RESULT: V2 IS BETTER! KEEP THE UPGRADE.")
    elif win_rate < 56.9:
        print("\n⚠️ RESULT: V2 IS WORSE. REVERT TO V1.")
    else:
        print("\n😐 RESULT: TIED. (No Edge Gained)")

if __name__ == "__main__":
    main()