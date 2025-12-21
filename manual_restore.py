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

# --- 1. THE TRUTH DATA (Hardcoded Results) ---
# Since API is returning NULL scores, we provide them manually.
COMPLETED_GAMES = [
    {"date": "2025-12-16", "home": "Troy", "away": "Jacksonville State", "h_score": 13, "a_score": 17},
    {"date": "2025-12-17", "home": "South Florida", "away": "Old Dominion", "h_score": 10, "a_score": 24},
    {"date": "2025-12-17", "home": "Louisiana", "away": "Delaware", "h_score": 13, "a_score": 20},
    {"date": "2025-12-18", "home": "Arkansas State", "away": "Missouri State", "h_score": 34, "a_score": 28},
    {"date": "2025-12-19", "home": "Western Kentucky", "away": "Sam Houston State", "h_score": 41, "a_score": 28}
]

def get_data(endpoint, params):
    try:
        time.sleep(1)
        res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        if res.status_code == 200: return res.json()
    except: pass
    return []

def main():
    print("--- 🛠️ MANUAL HISTORY RESTORATION ---")
    
    # 1. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except:
        print("❌ Models not found.")
        return

    # 2. Fetch Context (Stats/Talent) needed for the Model
    print("   -> Fetching Team Stats & Talent (for predictions)...")
    stats = get_data("/stats/game/advanced", {"year": 2025})
    talent = get_data("/talent", {"year": 2025})
    srs = get_data("/ratings/srs", {"year": 2025})
    
    # Fetch Lines (We try to find real lines, or default if missing)
    lines_post = get_data("/lines", {"year": 2025, "seasonType": "postseason", "week": 1})
    lines_reg = get_data("/lines", {"year": 2025, "seasonType": "regular", "week": 16})
    all_lines = lines_post + lines_reg
    
    # Mappers
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent}
    srs_map = {x['team']: x['rating'] for x in srs}
    
    # Map lines by Team Name because Game IDs might mismatch in our manual list
    team_lines_map = {}
    for g in all_lines:
        for l in g.get('lines', []):
            if l.get('provider') == 'DraftKings':
                team_lines_map[g['homeTeam']] = l
                team_lines_map[g['awayTeam']] = l
                break

    # Build Decay Stats
    stats_df = pd.json_normalize(stats)
    decay_map = {}
    if not stats_df.empty:
        for team, group in stats_df.groupby('team'):
            t_mom = {}
            for m in ['offense.ppa', 'offense.successRate', 'offense.explosiveness', 'defense.ppa', 'defense.successRate', 'defense.explosiveness']:
                if m in group.columns: t_mom[f"decay_{m}"] = group[m].mean()
                else: t_mom[f"decay_{m}"] = 0.0
            decay_map[team] = t_mom

    new_rows = []
    print(f"   -> Re-generating predictions for {len(COMPLETED_GAMES)} games...")

    for g in COMPLETED_GAMES:
        home, away = g['home'], g['away']
        
        # 1. Get Stats
        h_d = decay_map.get(home)
        a_d = decay_map.get(away)
        
        # Fallback for "Sam Houston State" vs "Sam Houston" naming diffs
        if not a_d and "State" in away: a_d = decay_map.get(away.replace(" State", ""))
        if not h_d and "State" in home: h_d = decay_map.get(home.replace(" State", ""))

        if not h_d or not a_d:
            print(f"      ⚠️ Stats missing for {away} @ {home}. Skipping.")
            continue
            
        # 2. Find Line (or default if API fails)
        line_data = team_lines_map.get(home, team_lines_map.get(away))
        spread_val = float(line_data.get('spread', -3.0)) if line_data else -3.0
        total_val = float(line_data.get('overUnder', 55.5)) if line_data else 55.5
        
        # 3. Build Row
        row = {
            'spread': spread_val,
            'overUnder': total_val,
            'home_talent_score': talent_map.get(home, 10), 
            'away_talent_score': talent_map.get(away, 10),
            'home_srs_rating': srs_map.get(home, -5), 
            'away_srs_rating': srs_map.get(away, -5),
            **{f"home_{k}":v for k,v in h_d.items()}, 
            **{f"away_{k}":v for k,v in a_d.items()}
        }
        
        # Fill missing model columns
        for col in model_spread.feature_names_in_:
            if col not in row: row[col] = 0.0

        # 4. Predict
        feats = pd.DataFrame([row])[model_spread.feature_names_in_]
        cover_prob = model_spread.predict_proba(feats)[0][1]
        over_prob = model_total.predict_proba(feats)[0][1]
        
        if cover_prob > 0.5: pick_team, my_line = home, row['spread']
        else: pick_team, my_line = away, -1 * row['spread']
        
        pick_side = "OVER" if over_prob > 0.5 else "UNDER"
        
        # 5. Create Record (With Manual Scores!)
        # We use a Dummy GameID based on date to prevent duplicates
        dummy_id = f"MANUAL_{g['date'].replace('-','')}_{home[:3]}"
        
        new_rows.append({
            "GameID": dummy_id, "HomeTeam": home, "AwayTeam": away,
            "Game": f"{away} @ {home}",
            "Spread Pick": f"{pick_team} ({'+' if my_line > 0 else ''}{my_line})", 
            "Spread Book": "Restored", "Spread Conf": f"{max(cover_prob, 1-cover_prob):.1%}", 
            "Spread_Conf_Raw": max(cover_prob, 1-cover_prob),
            "Pick_Team": pick_team, "Pick_Line": row['spread'],
            "Total Pick": f"{pick_side} {row['overUnder']}", 
            "Total Book": "Restored", "Total Conf": f"{max(over_prob, 1-over_prob):.1%}", 
            "Total_Conf_Raw": max(over_prob, 1-over_prob),
            "Pick_Side": pick_side, "Pick_Total": row['overUnder'],
            # INJECT MANUAL SCORES HERE
            "Manual_Date": g['date'],
            "Manual_HomeScore": g['h_score'],
            "Manual_AwayScore": g['a_score']
        })
        print(f"      ✅ Restored: {away} ({g['a_score']}) - {home} ({g['h_score']})")

    # 6. Save
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if os.path.exists(HISTORY_FILE):
            old_df = pd.read_csv(HISTORY_FILE)
            combined = pd.concat([new_df, old_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['GameID'], keep='first')
            combined.to_csv(HISTORY_FILE, index=False)
        else:
            new_df.to_csv(HISTORY_FILE, index=False)
        print(f"\n💾 SUCCESS: Manually restored {len(new_rows)} completed games.")

if __name__ == "__main__":
    main()