import os
import pandas as pd
import joblib
import requests
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"
YEAR = 2025
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet']

def fetch_with_retry(endpoint, params):
    url = f"https://api.collegefootballdata.com{endpoint}"
    for attempt in range(1, 4): 
        try:
            res = requests.get(url, headers=HEADERS, params=params)
            if res.status_code == 200: return res.json()
            elif res.status_code == 429: time.sleep(15 * attempt)
        except: time.sleep(5)
    return []

def main():
    print("--- 🔮 FORECAST ENGINE (LOGIC ENFORCED) ---")
    
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        model_win = joblib.load("model_winner.pkl")
        feat_cols = model_spread.feature_names_in_
    except: print("❌ Models missing. Run retrain.py first."); return

    # 1. FETCH
    print("   -> Fetching schedule...")
    games = []
    lines = []
    scenarios = [
        {"seasonType": "postseason", "week": 1},
        {"seasonType": "regular", "week": 16},
        {"seasonType": "regular", "week": 17},
        {"seasonType": "regular", "week": 15}
    ]
    for s in scenarios:
        g = fetch_with_retry("/games", {"year": YEAR, **s})
        l = fetch_with_retry("/lines", {"year": YEAR, **s})
        if isinstance(g, list): games.extend(g)
        if isinstance(l, list): lines.extend(l)

    unique_games = {g['id']: g for g in games}
    games = list(unique_games.values())
    
    lines_map = {}
    for g in lines:
        valid = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        lines_map[str(g['id'])] = valid

    # 2. STATS
    srs = fetch_with_retry("/ratings/srs", {"year": YEAR})
    talent = fetch_with_retry("/talent", {"year": YEAR})
    srs_map = {x['team']: x['rating'] for x in srs} if isinstance(srs, list) else {}
    tal_map = {x.get('school', x.get('team')): x['talent'] for x in talent} if isinstance(talent, list) else {}

    # 3. PREDICT
    predictions = []
    
    if games:
        print(f"   -> Processing {len(games)} potential games...")
        for g in games:
            if not isinstance(g, dict) or g.get('completed'): continue
            gid = str(g.get('id'))
            
            home = g.get('home_team') or g.get('homeTeam')
            away = g.get('away_team') or g.get('awayTeam')
            if not home or not away: continue

            game_lines = lines_map.get(gid, [])
            if not game_lines: continue 

            best_spread = {"conf": 0.0, "pick": "Pending"}
            best_total = {"conf": 0.0, "pick": "Pending"}
            best_ml = {"conf": 0.0, "pick": "Pending"}

            for line in game_lines:
                spread_val = line.get('spread')
                total_val = line.get('overUnder')
                if spread_val is None or total_val is None: continue

                row = {
                    'spread': spread_val,
                    'overUnder': total_val,
                    'home_talent_score': tal_map.get(home, 10), 
                    'away_talent_score': tal_map.get(away, 10),
                    'home_srs_rating': srs_map.get(home, 0), 
                    'away_srs_rating': srs_map.get(away, 0)
                }
                input_df = pd.DataFrame([row])[feat_cols]

                # SPREAD
                prob = model_spread.predict_proba(input_df)[0][1]
                conf = max(prob, 1-prob)
                if conf > best_spread['conf']:
                    p_team = home if prob > 0.5 else away
                    p_line = spread_val if prob > 0.5 else -spread_val
                    best_spread = {"conf": conf, "pick": f"{p_team} ({p_line})", "pick_team": p_team, "pick_line": p_line}

                # TOTAL
                prob = model_total.predict_proba(input_df)[0][1]
                conf = max(prob, 1-prob)
                if conf > best_total['conf']:
                    side = "OVER" if prob > 0.5 else "UNDER"
                    best_total = {"conf": conf, "pick": f"{side} {total_val}", "pick_side": side, "pick_val": total_val}
                    
                # MONEYLINE
                prob = model_win.predict_proba(input_df)[0][1]
                conf = max(prob, 1-prob)
                if conf > best_ml['conf']:
                    ml_team = home if prob > 0.5 else away
                    best_ml = {"conf": conf, "pick": ml_team}

            # --- LOGIC ENFORCEMENT ---
            if best_spread['conf'] > 0:
                # Rule: If Spread Pick is Favorite (Negative Line), ML must match
                if best_spread.get('pick_line', 0) < 0:
                    if best_ml['pick'] != best_spread['pick_team']:
                        # Override ML with Spread pick
                        best_ml['pick'] = best_spread['pick_team']

                predictions.append({
                    "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
                    "StartDate": g.get('start_date') or g.get('startDate'),
                    "Moneyline Pick": best_ml['pick'], "Moneyline Conf": f"{best_ml['conf']:.1%}", 
                    "Spread Pick": best_spread['pick'], "Spread Conf": f"{best_spread['conf']:.1%}", 
                    "Total Pick": best_total['pick'], "Total Conf": f"{best_total['conf']:.1%}",
                    "Pick_Team": best_spread.get('pick_team'), "Pick_Line": best_spread.get('pick_line'),
                    "Pick_Side": best_total.get('pick_side'), "Pick_Total": best_total.get('pick_val')
                })

    if predictions:
        try:
            old_df = pd.read_csv(HISTORY_FILE)
            history = old_df[old_df['Manual_HomeScore'].notna()]
        except: history = pd.DataFrame()
        
        final_df = pd.concat([pd.DataFrame(predictions), history], ignore_index=True)
        final_df.to_csv(HISTORY_FILE, index=False)
        print(f"✅ SUCCESS: Updated predictions with logic enforcement.")
    else:
        print("⚠️ No games found with valid odds.")

if __name__ == "__main__":
    main()