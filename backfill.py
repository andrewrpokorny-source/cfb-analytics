import os
import pandas as pd
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"

# STRICTLY US REGULATED BOOKS + CONSENSUS (as backup for old games)
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'consensus']

def get_data(endpoint, params):
    try:
        response = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️ Error fetching {endpoint}: {e}")
        return []

def cleanup_csv():
    """Removes broken rows (None @ None) from the CSV."""
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        initial_len = len(df)
        # Drop rows where Game is NaN or contains "None"
        df = df.dropna(subset=['Game'])
        df = df[~df['Game'].astype(str).str.contains("None")]
        df.to_csv(HISTORY_FILE, index=False)
        removed = initial_len - len(df)
        if removed > 0:
            print(f"🧹 Cleaned up {removed} broken rows from history file.")

def main():
    print("--- ⏳ SMART BACKFILL (REPAIR & UPDATE) ---")
    
    # 1. Clean broken data first
    cleanup_csv()
    
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except:
        print("❌ Models not found. Run ./run_pipeline.sh first.")
        return
    
    # 2. Fetch Data
    print("   -> Fetching Postseason games & lines...")
    games = get_data("/games", {"year": 2025, "seasonType": "postseason"})
    lines_data = get_data("/lines", {"year": 2025, "seasonType": "postseason"})
    
    # Build a lookup for lines: GameID -> Best Line
    lines_map = {}
    for g in lines_data:
        gid = g.get('id')
        # Find first valid book
        valid_lines = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        if valid_lines:
            # Prefer DraftKings/FanDuel, fallback to consensus
            lines_map[gid] = valid_lines[0]

    backfilled = []
    
    for g in games:
        # We ONLY want COMPLETED games that we might have missed
        if not g.get('completed'): continue
        
        gid = g.get('id')
        home = g.get('home_team')
        away = g.get('away_team')
        
        # Safety Check
        if not home or not away: continue

        # Get Line
        line = lines_map.get(gid)
        if not line:
            # If no line exists, we can't grade it. Skip.
            continue
            
        spread_val = line.get('spread')
        total_val = line.get('overUnder')
        
        if spread_val is None: continue

        # 3. Re-Predict (To get the "Conf" scores)
        # Simplified features since we are backfilling past events
        row = {
            'spread': spread_val,
            'overUnder': total_val if total_val else 55.5,
            'home_talent_score': 10, 'away_talent_score': 10,
            'home_srs_rating': 0, 'away_srs_rating': 0
        }
        # Fill missing features with 0
        for col in model_spread.feature_names_in_:
            if col not in row: row[col] = 0.0

        features = pd.DataFrame([row])[model_spread.feature_names_in_]
        cover_prob = model_spread.predict_proba(features)[0][1]
        over_prob = model_total.predict_proba(features)[0][1]
        
        # 4. Format Logic (New Precision Logic)
        if cover_prob > 0.5:
            pick_team = home
            pick_line = spread_val
        else:
            pick_team = away
            pick_line = -1 * spread_val

        fmt_line = f"+{pick_line}" if pick_line > 0 else f"{pick_line}"

        backfilled.append({
            "GameID": gid,
            "HomeTeam": home,
            "AwayTeam": away,
            "Game": f"{away} @ {home}",
            "Spread Pick": f"{pick_team} ({fmt_line})",
            "Spread Book": line.get('provider'),
            "Spread Conf": "N/A (Backfill)",
            "Spread_Conf_Raw": 0.5,
            "Pick_Team": pick_team,
            "Pick_Line": spread_val, # Raw Home Spread
            "Total Pick": f"{'OVER' if over_prob > 0.5 else 'UNDER'} {total_val}",
            "Total Book": line.get('provider'),
            "Total Conf": "N/A",
            "Total_Conf_Raw": 0.5,
            "Pick_Side": "OVER" if over_prob > 0.5 else "UNDER",
            "Pick_Total": total_val
        })

    # 5. Save and Merge
    if backfilled:
        new_df = pd.DataFrame(backfilled)
        
        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
            # Combine
            combined = pd.concat([new_df, history_df], ignore_index=True)
            # Deduplicate by GameID (Keep the one we just generated if needed, or prefer existing)
            combined = combined.drop_duplicates(subset=['GameID'], keep='last')
        else:
            combined = new_df
            
        combined.to_csv(HISTORY_FILE, index=False)
        print(f"✅ Successfully backfilled {len(backfilled)} completed games.")
    else:
        print("No missing completed games found.")

if __name__ == "__main__":
    main()