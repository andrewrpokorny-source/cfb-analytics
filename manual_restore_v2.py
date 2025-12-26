import pandas as pd
import joblib
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"

# --- OFFICIAL BOWL RESULTS (Dec 19 - Dec 26) ---
COMPLETED_GAMES = [
    {"date": "2025-12-19", "home": "Western Michigan", "away": "Kennesaw State", "h_score": 41, "a_score": 6},
    {"date": "2025-12-19", "home": "NC State", "away": "Memphis", "h_score": 31, "a_score": 7},
    {"date": "2025-12-20", "home": "Oklahoma", "away": "Alabama", "h_score": 24, "a_score": 34},
    {"date": "2025-12-20", "home": "Texas A&M", "away": "Miami", "h_score": 3, "a_score": 10},
    {"date": "2025-12-20", "home": "Ole Miss", "away": "Tulane", "h_score": 41, "a_score": 10},
    {"date": "2025-12-20", "home": "Oregon", "away": "James Madison", "h_score": 51, "a_score": 34},
    {"date": "2025-12-22", "home": "Boise State", "away": "Washington State", "h_score": 21, "a_score": 34}, 
    {"date": "2025-12-23", "home": "Louisville", "away": "Toledo", "h_score": 27, "a_score": 22},
    {"date": "2025-12-23", "home": "Western Kentucky", "away": "Southern Mississippi", "h_score": 27, "a_score": 16},
    {"date": "2025-12-23", "home": "UNLV", "away": "Ohio", "h_score": 10, "a_score": 17},
    {"date": "2025-12-24", "home": "Hawaii", "away": "California", "h_score": 35, "a_score": 31}
]

def get_data(endpoint, params):
    try:
        time.sleep(1)
        res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        if res.status_code == 200: return res.json()
    except: pass
    return []

def main():
    print("--- 🎄 HOLIDAY BOWL RESTORE (DEC 20-26) ---")
    
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except:
        print("❌ Models not found.")
        return

    print("   -> Fetching context for grading...")
    # Fetch lines/stats to build a valid prediction row
    stats = get_data("/stats/game/advanced", {"year": 2025})
    talent = get_data("/talent", {"year": 2025})
    srs = get_data("/ratings/srs", {"year": 2025})
    lines = get_data("/lines", {"year": 2025, "seasonType": "postseason"})
    
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent}
    srs_map = {x['team']: x['rating'] for x in srs}
    
    # Map lines by Team Name
    team_lines_map = {}
    for g in lines:
        for l in g.get('lines', []):
            team_lines_map[g['homeTeam']] = l
            team_lines_map[g['awayTeam']] = l
            break

    stats_df = pd.json_normalize(stats)
    decay_map = {}
    if not stats_df.empty:
        for team, group in stats_df.groupby('team'):
            t_mom = {}
            for m in ['offense.ppa', 'offense.successRate']:
                if m in group.columns: t_mom[f"decay_{m}"] = group[m].mean()
                else: t_mom[f"decay_{m}"] = 0.0
            decay_map[team] = t_mom

    new_rows = []
    print(f"   -> Injecting {len(COMPLETED_GAMES)} missing games...")

    for g in COMPLETED_GAMES:
        home, away = g['home'], g['away']
        
        # Build Stats
        h_d = decay_map.get(home, {})
        a_d = decay_map.get(away, {})
        
        # Build Line
        line_data = team_lines_map.get(home, team_lines_map.get(away))
        spread_val = float(line_data.get('spread', -3.0)) if line_data else -3.0
        total_val = float(line_data.get('overUnder', 55.5)) if line_data else 55.5
        
        row = {
            'spread': spread_val, 'overUnder': total_val,
            'home_talent_score': talent_map.get(home, 10), 'away_talent_score': talent_map.get(away, 10),
            'home_srs_rating': srs_map.get(home, -5), 'away_srs_rating': srs_map.get(away, -5),
            **{f"home_{k}":v for k,v in h_d.items()}, **{f"away_{k}":v for k,v in a_d.items()}
        }
        
        for col in model_spread.feature_names_in_:
            if col not in row: row[col] = 0.0

        # Predict
        feats = pd.DataFrame([row])[model_spread.feature_names_in_]
        cover_prob = model_spread.predict_proba(feats)[0][1]
        over_prob = model_total.predict_proba(feats)[0][1]
        
        if cover_prob > 0.5: pick_team, my_line = home, row['spread']
        else: pick_team, my_line = away, -1 * row['spread']
        pick_side = "OVER" if over_prob > 0.5 else "UNDER"
        
        # Create Record
        dummy_id = f"MANUAL_v2_{g['date'].replace('-','')}_{home[:3]}"
        new_rows.append({
            "GameID": dummy_id, "HomeTeam": home, "AwayTeam": away,
            "Game": f"{away} @ {home}",
            "Spread Pick": f"{pick_team} ({'+' if my_line > 0 else ''}{my_line})", 
            "Spread Book": "Restored_v2", "Spread Conf": f"{max(cover_prob, 1-cover_prob):.1%}", 
            "Spread_Conf_Raw": max(cover_prob, 1-cover_prob),
            "Pick_Team": pick_team, "Pick_Line": row['spread'],
            "Total Pick": f"{pick_side} {row['overUnder']}", 
            "Total Book": "Restored_v2", "Total Conf": f"{max(over_prob, 1-over_prob):.1%}", 
            "Total_Conf_Raw": max(over_prob, 1-over_prob),
            "Pick_Side": pick_side, "Pick_Total": row['overUnder'],
            "Manual_Date": g['date'],
            "Manual_HomeScore": g['h_score'],
            "Manual_AwayScore": g['a_score']
        })
        print(f"      ✅ Injected: {away} ({g['a_score']}) - {home} ({g['h_score']})")

    # Save
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if os.path.exists(HISTORY_FILE):
            old_df = pd.read_csv(HISTORY_FILE)
            combined = pd.concat([new_df, old_df], ignore_index=True)
            # Smart Dedup
            combined = combined.drop_duplicates(subset=['HomeTeam', 'AwayTeam'], keep='first')
            combined.to_csv(HISTORY_FILE, index=False)
        else:
            new_df.to_csv(HISTORY_FILE, index=False)
        print(f"\n💾 SUCCESS: Added {len(new_rows)} recent bowl games to history.")

if __name__ == "__main__":
    main()
