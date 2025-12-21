import os
import pandas as pd
import joblib
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"

def get_data(endpoint, params):
    # Simple fetch with retry
    for i in range(3):
        try:
            time.sleep(1)
            res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
            if res.status_code == 200: 
                return res.json()
        except: 
            time.sleep(2)
    return []

def main():
    print("--- 🕰️ RECONSTRUCTING HISTORY (DEC 14 - DEC 19) ---")
    
    # 1. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except:
        print("❌ Models not found. Please run the training pipeline first.")
        return

    # 2. Fetch COMPLETED Postseason Games
    print("   -> Fetching completed bowl games...")
    games = get_data("/games", {"year": 2025, "seasonType": "postseason"})
    
    # --- ERROR FIX: Use .get() and check type ---
    completed_games = []
    if isinstance(games, list):
        completed_games = [g for g in games if isinstance(g, dict) and g.get('status') == 'completed']
    else:
        print(f"   ⚠️ Unexpected API response format: {type(games)}")
        return
    
    if not completed_games:
        print("   (No completed postseason games found yet. Check back later!)")
        return

    # 3. Fetch Context (Stats/Lines)
    print("   -> Fetching stats context...")
    stats = get_data("/stats/game/advanced", {"year": 2025})
    lines = get_data("/lines", {"year": 2025, "seasonType": "postseason"})
    talent = get_data("/talent", {"year": 2025})
    srs = get_data("/ratings/srs", {"year": 2025})

    # Helpers
    stats_df = pd.json_normalize(stats)
    decay_map = {} 
    if not stats_df.empty:
        metrics = ['offense.ppa', 'offense.successRate', 'offense.explosiveness', 'defense.ppa', 'defense.successRate', 'defense.explosiveness']
        for team, group in stats_df.groupby('team'):
            t_mom = {}
            for m in metrics:
                if m in group.columns: t_mom[f"decay_{m}"] = group[m].mean()
                else: t_mom[f"decay_{m}"] = 0.0
            decay_map[team] = t_mom

    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent}
    srs_map = {x['team']: x['rating'] for x in srs}
    lines_map = {str(g['id']): g['lines'] for g in lines if 'id' in g}

    new_rows = []
    print(f"   -> Processing {len(completed_games)} games...")

    for g in completed_games:
        gid = str(g['id'])
        home, away = g['home_team'], g['away_team']
        
        # Get Line (Prefer DraftKings)
        game_lines = lines_map.get(gid, [])
        valid_line = next((l for l in game_lines if l.get('provider') == 'DraftKings'), None)
        if not valid_line and game_lines: valid_line = game_lines[0] # Fallback
        
        if not valid_line: continue
        
        # Build Features
        h_d = decay_map.get(home)
        a_d = decay_map.get(away)
        if not h_d or not a_d: continue

        row = {
            'spread': valid_line.get('spread', 0),
            'overUnder': valid_line.get('overUnder', 55.5),
            'home_talent_score': talent_map.get(home, 10), 'away_talent_score': talent_map.get(away, 10),
            'home_srs_rating': srs_map.get(home, -5), 'away_srs_rating': srs_map.get(away, -5),
            **{f"home_{k}":v for k,v in h_d.items()}, **{f"away_{k}":v for k,v in a_d.items()}
        }
        
        # Predict Spread
        try:
            feats = pd.DataFrame([row])[model_spread.feature_names_in_]
            cover_prob = model_spread.predict_proba(feats)[0][1]
            
            if cover_prob > 0.5:
                pick_team, my_line = home, row['spread']
            else:
                pick_team, my_line = away, -1 * row['spread']
            
            fmt_line = f"+{my_line}" if my_line > 0 else f"{my_line}"
            
            # Predict Total
            over_prob = model_total.predict_proba(feats)[0][1]
            pick_side = "OVER" if over_prob > 0.5 else "UNDER"
            
            new_rows.append({
                "GameID": gid, "HomeTeam": home, "AwayTeam": away,
                "Game": f"{away} @ {home}",
                "Spread Pick": f"{pick_team} ({fmt_line})", "Spread Book": "Backfill",
                "Spread Conf": f"{max(cover_prob, 1-cover_prob):.1%}", "Spread_Conf_Raw": max(cover_prob, 1-cover_prob),
                "Pick_Team": pick_team, "Pick_Line": row['spread'],
                "Total Pick": f"{pick_side} {row['overUnder']}", "Total Book": "Backfill",
                "Total Conf": f"{max(over_prob, 1-over_prob):.1%}", "Total_Conf_Raw": max(over_prob, 1-over_prob),
                "Pick_Side": pick_side, "Pick_Total": row['overUnder']
            })
        except: pass

    # Append to CSV
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if os.path.exists(HISTORY_FILE):
            old_df = pd.read_csv(HISTORY_FILE)
            # Combine and remove duplicates based on GameID
            combined = pd.concat([new_df, old_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['GameID'], keep='first')
            combined.to_csv(HISTORY_FILE, index=False)
        else:
            new_df.to_csv(HISTORY_FILE, index=False)
        print(f"✅ Restored {len(new_rows)} historical games.")
    else:
        print("No suitable history found to restore.")

if __name__ == "__main__":
    main()