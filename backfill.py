import os
import pandas as pd
import joblib
import requests
import time
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_CUTOFF = "2025-12-01"
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet']

def fetch_with_retry(endpoint, params):
    url = f"https://api.collegefootballdata.com{endpoint}"
    for attempt in range(1, 4):
        try:
            res = requests.get(url, headers=HEADERS, params=params)
            if res.status_code == 200: return res.json()
            elif res.status_code == 429:
                time.sleep(10 * attempt)
        except: time.sleep(5)
    return []

def main():
    print("--- 📜 RUNNING HISTORICAL BACKFILL (SINCE DEC 1) ---")
    
    # 1. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        feat_cols = model_spread.feature_names_in_
    except:
        print("❌ Models missing. Run retrain.py first.")
        return

    # 2. Fetch Games (Weeks 14, 15, 16, Postseason)
    all_games = []
    all_lines = []
    
    # December spans late regular season + postseason
    scenarios = [
        {"year": 2025, "seasonType": "regular", "week": 14},
        {"year": 2025, "seasonType": "regular", "week": 15},
        {"year": 2025, "seasonType": "regular", "week": 16},
        {"year": 2025, "seasonType": "postseason"}
    ]
    
    print("   -> Fetching historical data...")
    for s in scenarios:
        g = fetch_with_retry("/games", s)
        l = fetch_with_retry("/lines", s)
        if isinstance(g, list): all_games.extend(g)
        if isinstance(l, list): all_lines.extend(l)

    # 3. Process Data
    lines_map = {}
    for g in all_lines:
        valid = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        if valid: lines_map[str(g['id'])] = valid[0] # Take first valid book

    # Fetch Stats (We need these to run the model)
    srs = fetch_with_retry("/ratings/srs", {"year": 2025})
    talent = fetch_with_retry("/talent", {"year": 2025})
    srs_map = {x['team']: x['rating'] for x in srs} if isinstance(srs, list) else {}
    tal_map = {x.get('school', x.get('team')): x['talent'] for x in talent} if isinstance(talent, list) else {}

    history_rows = []
    
    print(f"   -> Analyzing {len(all_games)} games...")
    
    for g in all_games:
        # Filter by Date
        start_date = g.get('start_date') or g.get('startDate')
        if not start_date or start_date < HISTORY_CUTOFF: continue
        
        # Must be completed to be "History"
        if not g.get('completed'): continue
        
        gid = str(g['id'])
        home = g.get('home_team') or g.get('homeTeam')
        away = g.get('away_team') or g.get('awayTeam')
        h_score = g.get('home_points') or g.get('homePoints')
        a_score = g.get('away_points') or g.get('awayPoints')
        
        # Get Line
        line_data = lines_map.get(gid)
        if not line_data: continue # Skip games with no odds
        
        spread = line_data.get('spread')
        total = line_data.get('overUnder')
        
        # Prepare Model Inputs
        h_srs, a_srs = srs_map.get(home, 0), srs_map.get(away, 0)
        h_tal, a_tal = tal_map.get(home, 10), tal_map.get(away, 10)
        
        row = {
            'spread': spread,
            'overUnder': total,
            'home_talent_score': h_tal, 'away_talent_score': a_tal,
            'home_srs_rating': h_srs, 'away_srs_rating': a_srs,
            **{c: 0.0 for c in feat_cols if 'decay' in c}
        }
        
        # RUN MODELS
        # Spread
        prob_spread = model_spread.predict_proba(pd.DataFrame([row])[feat_cols])[0][1]
        conf_spread = max(prob_spread, 1-prob_spread)
        pick_team_spr = home if prob_spread > 0.5 else away
        pick_line_spr = spread if prob_spread > 0.5 else -spread
        spread_display = f"{pick_team_spr} ({pick_line_spr})"
        
        # Total
        prob_total = model_total.predict_proba(pd.DataFrame([row])[feat_cols])[0][1]
        conf_total = max(prob_total, 1-prob_total)
        pick_side = "OVER" if prob_total > 0.5 else "UNDER"
        total_display = f"{pick_side} {total}"
        
        # Moneyline (Simple Logic)
        ml_pick = home if (h_srs + h_tal/200) > (a_srs + a_tal/200) else away
        
        history_rows.append({
            "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
            "StartDate": start_date,
            "Moneyline Pick": ml_pick, "Moneyline Conf": "N/A",
            "Spread Pick": spread_display, "Spread Conf": f"{conf_spread:.1%}",
            "Total Pick": total_display, "Total Conf": f"{conf_total:.1%}",
            "Pick_Team": pick_team_spr, "Pick_Line": pick_line_spr,
            "Pick_Side": pick_side, "Pick_Total": total,
            "Manual_HomeScore": h_score, "Manual_AwayScore": a_score # This triggers the "History" tab
        })

    # 4. Merge with Existing Future Predictions
    print(f"   -> Found {len(history_rows)} completed games.")
    
    try:
        existing = pd.read_csv("live_predictions.csv")
        # Keep only UPCOMING games from the existing file (where score is empty)
        if 'Manual_HomeScore' in existing.columns:
            future = existing[existing['Manual_HomeScore'].isna()]
        else:
            future = existing
    except:
        future = pd.DataFrame()
        
    # Combine
    hist_df = pd.DataFrame(history_rows)
    final_df = pd.concat([future, hist_df], ignore_index=True)
    
    # Save
    final_df.to_csv("live_predictions.csv", index=False)
    print(f"✅ SUCCESS: Database now has {len(hist_df)} history games and {len(future)} upcoming games.")

if __name__ == "__main__":
    main()