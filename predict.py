import os
import pandas as pd
import joblib
import requests
import json
import math
import time
from dotenv import load_dotenv

# --- 1. SETUP ---
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"
YEAR = 2025
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet']

# --- 2. UTILS ---
def fetch_data(endpoint, params):
    try:
        res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        return res.json() if res.status_code == 200 else []
    except: return []

def calculate_win_prob(home_srs, away_srs, home_talent, away_talent):
    # Performance Override (200.0 Divisor)
    talent_diff = (home_talent - away_talent) / 200.0
    srs_diff = home_srs - away_srs
    try: prob = 1 / (1 + math.exp(-1 * (srs_diff + talent_diff) / 7.5))
    except: prob = 0.5
    return prob

def main():
    print("--- 🔮 RUNNING FORECAST ENGINE (CLASSIC) ---")
    
    # 1. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        feat_cols = model_spread.feature_names_in_
    except: print("❌ Models missing."); return

    # 2. Fetch Active Schedule (Postseason + Late Regular)
    print("   -> Fetching active games...")
    games_post = fetch_data("/games", {"year": YEAR, "seasonType": "postseason"})
    lines_post = fetch_data("/lines", {"year": YEAR, "seasonType": "postseason"})
    
    # Also grab Week 16/17 just in case
    games_reg = fetch_data("/games", {"year": YEAR, "seasonType": "regular", "week": 16})
    lines_reg = fetch_data("/lines", {"year": YEAR, "seasonType": "regular", "week": 16})
    
    games_data = games_post + games_reg
    lines_data = lines_post + lines_reg
    
    # Fetch Stats
    srs_data = fetch_data("/ratings/srs", {"year": YEAR})
    talent_data = fetch_data("/talent", {"year": YEAR})
    
    srs_map = {x['team']: x['rating'] for x in srs_data} if isinstance(srs_data, list) else {}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data} if isinstance(talent_data, list) else {}
    
    lines_map = {}
    if isinstance(lines_data, list):
        for g in lines_data:
            valid = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
            lines_map[str(g['id'])] = valid

    predictions = []
    seen_ids = set()

    print(f"   -> Processing {len(games_data)} games...")

    for g in games_data:
        if not isinstance(g, dict): continue
        gid = str(g.get('id'))
        
        # Deduplicate
        if gid in seen_ids: continue
        seen_ids.add(gid)
        
        # Skip completed games (Forecast only)
        if g.get('completed'): continue
        
        home, away = g.get('home_team'), g.get('away_team')
        if not home or not away: continue

        h_srs, a_srs = srs_map.get(home, 0), srs_map.get(away, 0)
        h_tal, a_tal = talent_map.get(home, 10), talent_map.get(away, 10)
        
        base_row = {
            'home_talent_score': h_tal, 'away_talent_score': a_tal,
            'home_srs_rating': h_srs, 'away_srs_rating': a_srs,
            **{c: 0.0 for c in feat_cols if 'decay' in c} # Default stats
        }
        
        # Moneyline
        ml_prob = calculate_win_prob(h_srs, a_srs, h_tal, a_tal)
        ml_pick = home if ml_prob > 0.5 else away
        ml_conf = max(ml_prob, 1 - ml_prob)

        # Spread/Total
        game_lines = lines_map.get(gid, [])
        best_spread = {"conf": 0.0, "pick": "Pending", "book": "TBD"}
        best_total = {"conf": 0.0, "pick": "Pending", "book": "TBD"}

        if game_lines:
            for line in game_lines:
                # Spread
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
                
                # Total
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
            "Moneyline Pick": ml_pick, "Moneyline Conf": f"{ml_conf:.1%}", "Moneyline_Conf_Raw": ml_conf,
            "Spread Pick": best_spread['pick'], "Spread Book": best_spread['book'], "Spread Conf": f"{best_spread['conf']:.1%}",
            "Total Pick": best_total['pick'], "Total Book": best_total['book'], "Total Conf": f"{best_total['conf']:.1%}",
            "Pick_Team": best_spread.get('pick_team'), "Pick_Line": best_spread.get('raw_line'),
            "Pick_Side": best_total.get('pick_side'), "Pick_Total": best_total.get('pick_val')
        })

    # 3. Save
    if predictions:
        new_df = pd.DataFrame(predictions)
        # Sort by date so upcoming games are top
        new_df.sort_values(by="StartDate", inplace=True)
        new_df.to_csv(HISTORY_FILE, index=False)
        print(f"✅ SUCCESS: Generated {len(new_df)} upcoming forecasts.")
    else:
        print("⚠️ No active games found.")

if __name__ == "__main__":
    main()