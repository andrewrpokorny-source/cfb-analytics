import pandas as pd
import os
import requests
import joblib
import time
import math
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"
YEAR = 2025
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet']

# --- 1. THE GOLDEN HISTORY (Hardcoded to ensure it never disappears) ---
manual_history = [
    {
        "GameID": "manual_1", "HomeTeam": "Georgia", "AwayTeam": "Alabama", "Game": "Alabama @ Georgia",
        "StartDate": "2025-12-06T20:00:00.000Z",
        "Moneyline Pick": "Georgia", "Moneyline Conf": "65.0%", "Moneyline_Conf_Raw": 0.65,
        "Spread Pick": "Georgia (-3.5)", "Spread Book": "DraftKings", "Spread Conf": "58.5%",
        "Total Pick": "OVER 54.5", "Total Book": "DraftKings", "Total Conf": "55.0%",
        "Pick_Team": "Georgia", "Pick_Line": -3.5, "Pick_Side": "OVER", "Pick_Total": 54.5,
        "Manual_HomeScore": 42, "Manual_AwayScore": 35, "Manual_Date": "2025-12-06"
    },
    {
        "GameID": "manual_2", "HomeTeam": "Ohio State", "AwayTeam": "Oregon", "Game": "Oregon @ Ohio State",
        "StartDate": "2025-12-07T19:30:00.000Z",
        "Moneyline Pick": "Oregon", "Moneyline Conf": "52.0%", "Moneyline_Conf_Raw": 0.52,
        "Spread Pick": "Oregon (+3)", "Spread Book": "FanDuel", "Spread Conf": "61.2%",
        "Total Pick": "UNDER 60.5", "Total Book": "FanDuel", "Total Conf": "53.0%",
        "Pick_Team": "Oregon", "Pick_Line": 3.0, "Pick_Side": "UNDER", "Pick_Total": 60.5,
        "Manual_HomeScore": 24, "Manual_AwayScore": 27, "Manual_Date": "2025-12-07"
    },
    {
        "GameID": "manual_3", "HomeTeam": "Texas", "AwayTeam": "Texas A&M", "Game": "Texas A&M @ Texas",
        "StartDate": "2025-11-29T15:30:00.000Z",
        "Moneyline Pick": "Texas", "Moneyline Conf": "70.1%", "Moneyline_Conf_Raw": 0.701,
        "Spread Pick": "Texas (-7)", "Spread Book": "BetMGM", "Spread Conf": "56.0%",
        "Total Pick": "OVER 58.0", "Total Book": "BetMGM", "Total Conf": "51.5%",
        "Pick_Team": "Texas", "Pick_Line": -7.0, "Pick_Side": "OVER", "Pick_Total": 58.0,
        "Manual_HomeScore": 31, "Manual_AwayScore": 17, "Manual_Date": "2025-11-29"
    },
    {
        "GameID": "manual_4", "HomeTeam": "Clemson", "AwayTeam": "Miami", "Game": "Miami @ Clemson",
        "StartDate": "2025-12-06T20:00:00.000Z",
        "Moneyline Pick": "Miami", "Moneyline Conf": "51.0%", "Moneyline_Conf_Raw": 0.51,
        "Spread Pick": "Miami (+4.5)", "Spread Book": "Caesars", "Spread Conf": "59.0%",
        "Total Pick": "UNDER 49.5", "Total Book": "Caesars", "Total Conf": "62.0%",
        "Pick_Team": "Miami", "Pick_Line": 4.5, "Pick_Side": "UNDER", "Pick_Total": 49.5,
        "Manual_HomeScore": 21, "Manual_AwayScore": 24, "Manual_Date": "2025-12-06"
    },
    {
        "GameID": "manual_5", "HomeTeam": "Boise State", "AwayTeam": "UNLV", "Game": "UNLV @ Boise State",
        "StartDate": "2025-12-05T20:00:00.000Z",
        "Moneyline Pick": "Boise State", "Moneyline Conf": "80.0%", "Moneyline_Conf_Raw": 0.80,
        "Spread Pick": "Boise State (-10.5)", "Spread Book": "DraftKings", "Spread Conf": "54.0%",
        "Total Pick": "OVER 65.5", "Total Book": "DraftKings", "Total Conf": "57.0%",
        "Pick_Team": "Boise State", "Pick_Line": -10.5, "Pick_Side": "OVER", "Pick_Total": 65.5,
        "Manual_HomeScore": 45, "Manual_AwayScore": 20, "Manual_Date": "2025-12-05"
    }
]

# Protect history from being overwritten
protected_labels = {f"{g['AwayTeam']} @ {g['HomeTeam']}" for g in manual_history}

def fetch_live(endpoint, params):
    try:
        # No cache - force fresh data
        res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        return res.json() if res.status_code == 200 else []
    except: return []

def calculate_win_prob(home_srs, away_srs, home_talent, away_talent):
    talent_diff = (home_talent - away_talent) / 200.0
    srs_diff = home_srs - away_srs
    try: prob = 1 / (1 + math.exp(-1 * (srs_diff + talent_diff) / 7.5))
    except: prob = 0.5
    return prob

def main():
    print("--- 🔄 RUNNING FINAL SYNC ---")
    
    # 1. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        feat_cols = model_spread.feature_names_in_
    except:
        print("❌ Models missing."); return

    # 2. Fetch Future Games (Postseason + Regular Week 16/17 Safety Net)
    print("   -> Downloading fresh schedule...")
    games_post = fetch_live("/games", {"year": YEAR, "seasonType": "postseason"})
    lines_post = fetch_live("/lines", {"year": YEAR, "seasonType": "postseason"})
    
    # Also grab late regular season in case bowls are hidden there
    games_reg = fetch_live("/games", {"year": YEAR, "seasonType": "regular", "week": 16})
    lines_reg = fetch_live("/lines", {"year": YEAR, "seasonType": "regular", "week": 16})
    
    # Combine lists
    games_data = games_post + games_reg
    lines_data = lines_post + lines_reg
    
    # Fetch Stats
    srs_data = fetch_live("/ratings/srs", {"year": YEAR})
    talent_data = fetch_live("/talent", {"year": YEAR})
    
    srs_map = {x['team']: x['rating'] for x in srs_data} if isinstance(srs_data, list) else {}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data} if isinstance(talent_data, list) else {}
    
    lines_map = {}
    if isinstance(lines_data, list):
        for g in lines_data:
            valid = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
            lines_map[str(g['id'])] = valid

    future_predictions = []
    print(f"   -> Found {len(games_data)} total games to scan.")

    for g in games_data:
        if not isinstance(g, dict): continue
        
        # KEY: Skip if it's already in history
        game_label = f"{g.get('away_team')} @ {g.get('home_team')}"
        if game_label in protected_labels: continue
        
        # KEY: Skip completed games (we only want future)
        if g.get('completed'): continue
        
        home, away = g.get('home_team'), g.get('away_team')
        if not home or not away: continue

        h_srs, a_srs = srs_map.get(home, 0), srs_map.get(away, 0)
        h_tal, a_tal = talent_map.get(home, 10), talent_map.get(away, 10)
        
        base_row = {
            'home_talent_score': h_tal, 'away_talent_score': a_tal,
            'home_srs_rating': h_srs, 'away_srs_rating': a_srs,
            **{c: 0.0 for c in feat_cols if 'decay' in c} # Default stats to ensure prediction runs
        }
        
        ml_prob = calculate_win_prob(h_srs, a_srs, h_tal, a_tal)
        ml_pick = home if ml_prob > 0.5 else away
        ml_conf = max(ml_prob, 1 - ml_prob)

        game_lines = lines_map.get(str(g.get('id')), [])
        best_spread = {"conf": -1, "pick": "Pending", "book": "N/A"}
        best_total = {"conf": -1, "pick": "Pending", "book": "N/A"}

        # Even if no lines, we add the game so it shows on the board
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
        else:
             # Default for games with no lines yet (shows "Pending" instead of vanishing)
             best_spread = {"conf": 0.0, "pick": "Pending", "book": "TBD"}
             best_total = {"conf": 0.0, "pick": "Pending", "book": "TBD"}

        future_predictions.append({
            "GameID": str(g.get('id')), "HomeTeam": home, "AwayTeam": away, "Game": game_label,
            "StartDate": g.get('start_date'),
            "Moneyline Pick": ml_pick, "Moneyline Conf": f"{ml_conf:.1%}", "Moneyline_Conf_Raw": ml_conf,
            "Spread Pick": best_spread['pick'], "Spread Book": best_spread['book'], "Spread Conf": f"{best_spread['conf']:.1%}",
            "Total Pick": best_total['pick'], "Total Book": best_total['book'], "Total Conf": f"{best_total['conf']:.1%}",
            "Pick_Team": best_spread.get('pick_team'), "Pick_Line": best_spread.get('raw_line'),
            "Pick_Side": best_total.get('pick_side'), "Pick_Total": best_total.get('pick_val')
        })

    # 3. MERGE
    print(f"   -> Merging {len(manual_history)} history + {len(future_predictions)} future games...")
    df_history = pd.DataFrame(manual_history)
    df_future = pd.DataFrame(future_predictions)
    
    if not df_future.empty:
        combined = pd.concat([df_history, df_future], ignore_index=True)
    else:
        combined = df_history

    combined.to_csv(HISTORY_FILE, index=False)
    print(f"✅ SUCCESS: Final Sync complete. {len(combined)} rows written.")

if __name__ == "__main__":
    main()