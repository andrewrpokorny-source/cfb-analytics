import os
import pandas as pd
import joblib
import requests
import json
import math
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
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                print(f"   ⚠️ API Rate Limit. Waiting {20 * attempt}s...")
                time.sleep(20 * attempt)
            else:
                return []
        except:
            time.sleep(5)
    return []

def calculate_win_prob(home_srs, away_srs, home_talent, away_talent):
    talent_diff = (home_talent - away_talent) / 200.0
    srs_diff = home_srs - away_srs
    try: prob = 1 / (1 + math.exp(-1 * (srs_diff + talent_diff) / 7.5))
    except: prob = 0.5
    return prob

def main():
    print("--- 🔮 FORECAST ENGINE (ROBUST) ---")
    
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        feat_cols = model_spread.feature_names_in_
    except: print("❌ Models missing."); return

    # Fetch Postseason + Week 16/17 (Wide Net)
    print("   -> Fetching active schedule...")
    games = fetch_with_retry("/games", {"year": YEAR, "seasonType": "postseason"})
    lines = fetch_with_retry("/lines", {"year": YEAR, "seasonType": "postseason"})
    
    # Simple Map
    lines_map = {}
    if isinstance(lines, list):
        for g in lines:
            valid = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
            lines_map[str(g['id'])] = valid

    # Stats
    srs = fetch_with_retry("/ratings/srs", {"year": YEAR})
    talent = fetch_with_retry("/talent", {"year": YEAR})
    
    srs_map = {x['team']: x['rating'] for x in srs} if isinstance(srs, list) else {}
    tal_map = {x.get('school', x.get('team')): x['talent'] for x in talent} if isinstance(talent, list) else {}

    predictions = []
    
    if isinstance(games, list):
        print(f"   -> Scanning {len(games)} games...")
        for g in games:
            if not isinstance(g, dict) or g.get('completed'): continue
            
            gid = str(g.get('id'))
            home, away = g.get('home_team'), g.get('away_team')
            if not home or not away: continue

            h_srs, a_srs = srs_map.get(home, 0), srs_map.get(away, 0)
            h_tal, a_tal = tal_map.get(home, 10), tal_map.get(away, 10)
            
            base_row = {
                'home_talent_score': h_tal, 'away_talent_score': a_tal,
                'home_srs_rating': h_srs, 'away_srs_rating': a_srs,
                **{c: 0.0 for c in feat_cols if 'decay' in c}
            }
            
            ml_prob = calculate_win_prob(h_srs, a_srs, h_tal, a_tal)
            ml_pick = home if ml_prob > 0.5 else away
            ml_conf = max(ml_prob, 1 - ml_prob)

            game_lines = lines_map.get(gid, [])
            best_spread = {"conf": 0.0, "pick": "Pending", "book": "TBD"}
            best_total = {"conf": 0.0, "pick": "Pending", "book": "TBD"}

            if game_lines:
                for line in game_lines:
                    if line.get('spread'):
                        row = base_row.copy()
                        row['spread'] = line.get('spread')
                        row['overUnder'] = line.get('overUnder', 55.5)
                        for c in feat_cols: 
                            if c not in row: row[c] = 0.0
                        
                        prob = model_spread.predict_proba(pd.DataFrame([row])[feat_cols])[0][1]
                        conf = max(prob, 1-prob)
                        if conf > best_spread['conf']:
                            p_team = home if prob > 0.5 else away
                            p_line = line.get('spread') if prob > 0.5 else -1 * line.get('spread')
                            best_spread = {"conf": conf, "book": line.get('provider'), "pick": f"{p_team} ({p_line})", "raw_line": p_line, "pick_team": p_team}
                    
                    if line.get('overUnder'):
                        row = base_row.copy()
                        row['spread'] = line.get('spread', 0.0)
                        row['overUnder'] = line.get('overUnder')
                        for c in model_total.feature_names_in_:
                            if c not in row: row[c] = 0.0
                        prob = model_total.predict_proba(pd.DataFrame([row])[model_total.feature_names_in_])[0][1]
                        conf = max(prob, 1-prob)
                        if conf > best_total['conf']:
                            side = "OVER" if prob > 0.5 else "UNDER"
                            best_total = {"conf": conf, "book": line.get('provider'), "pick": f"{side} {line.get('overUnder')}", "pick_side": side, "pick_val": line.get('overUnder')}

            predictions.append({
                "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
                "StartDate": g.get('start_date'),
                "Moneyline Pick": ml_pick, "Moneyline Conf": f"{ml_conf:.1%}", 
                "Spread Pick": best_spread['pick'], "Spread Conf": f"{best_spread['conf']:.1%}",
                "Total Pick": best_total['pick'], "Total Conf": f"{best_total['conf']:.1%}"
            })

    if predictions:
        pd.DataFrame(predictions).to_csv(HISTORY_FILE, index=False)
        print(f"✅ SUCCESS: Saved {len(predictions)} forecasts.")
    else:
        print("⚠️ No active games found.")

if __name__ == "__main__":
    main()