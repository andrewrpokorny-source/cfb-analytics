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

def get_data(endpoint, params):
    try:
        # Polite delay
        time.sleep(1) 
        res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        if res.status_code == 200: return res.json()
        print(f"⚠️ API Error {res.status_code} for {endpoint}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    return []

def main():
    print("--- 📢 VERBOSE BACKFILL (LOUD MODE) ---")
    
    # 1. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        print("✅ Models loaded successfully.")
    except:
        print("❌ Models NOT found. Make sure .pkl files are in this folder.")
        return

    # 2. Fetch Completed Games
    print("   -> Fetching Postseason Games...")
    games = get_data("/games", {"year": 2025, "seasonType": "postseason"})
    
    # Robust Status Check
    completed_games = [g for g in games if isinstance(g, dict) and g.get('status') == 'completed']
    print(f"   -> Found {len(completed_games)} completed games in API.")

    if not completed_games:
        print("   🛑 Stopping: No completed games found. (Check API Key?)")
        return

    # 3. Fetch Context
    print("   -> Fetching Lines & Stats...")
    lines = get_data("/lines", {"year": 2025, "seasonType": "postseason"})
    stats = get_data("/stats/game/advanced", {"year": 2025})
    talent = get_data("/talent", {"year": 2025})
    srs = get_data("/ratings/srs", {"year": 2025})
    
    # Mappers
    lines_map = {str(g['id']): g['lines'] for g in lines if 'id' in g}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent}
    srs_map = {x['team']: x['rating'] for x in srs}
    
    stats_df = pd.json_normalize(stats)
    decay_map = {}
    if not stats_df.empty:
        for team, group in stats_df.groupby('team'):
            t_mom = {}
            for m in ['offense.ppa', 'offense.successRate']: # Simplified for speed
                if m in group.columns: t_mom[f"decay_{m}"] = group[m].mean()
                else: t_mom[f"decay_{m}"] = 0.0
            decay_map[team] = t_mom

    new_rows = []
    
    # 4. Processing Loop
    print("\n   -> analyzing games...")
    for g in completed_games:
        gid = str(g['id'])
        home, away = g['home_team'], g['away_team']
        print(f"      🔎 Checking {away} @ {home}...", end=" ")
        
        # FIND LINES (With Fallback)
        game_lines = lines_map.get(gid, [])
        valid_line = None
        
        # Priority 1: DraftKings (Any spelling)
        for l in game_lines:
            prov = l.get('provider', '')
            if 'Draft' in prov or 'Kings' in prov:
                valid_line = l
                break
        
        # Priority 2: Any available line
        if not valid_line and game_lines:
            valid_line = game_lines[0]
            
        if not valid_line:
            print("❌ No lines found. Skipping.")
            continue

        # Get Stats
        h_d = decay_map.get(home)
        a_d = decay_map.get(away)
        if not h_d or not a_d:
            print("⚠️ Missing stats. Skipping.")
            continue

        # Build Data
        row = {
            'spread': float(valid_line.get('spread', 0)),
            'overUnder': float(valid_line.get('overUnder', 55.5)),
            'home_talent_score': talent_map.get(home, 10), 
            'away_talent_score': talent_map.get(away, 10),
            'home_srs_rating': srs_map.get(home, -5), 
            'away_srs_rating': srs_map.get(away, -5),
            **{f"home_{k}":v for k,v in h_d.items()}, 
            **{f"away_{k}":v for k,v in a_d.items()}
        }
        
        # Fill missing model columns with 0
        for col in model_spread.feature_names_in_:
            if col not in row: row[col] = 0.0

        # Predict
        try:
            feats = pd.DataFrame([row])[model_spread.feature_names_in_]
            cover_prob = model_spread.predict_proba(feats)[0][1]
            over_prob = model_total.predict_proba(feats)[0][1]
            
            # Spread Logic
            if cover_prob > 0.5:
                pick_team, my_line = home, row['spread']
            else:
                pick_team, my_line = away, -1 * row['spread']
            fmt_line = f"+{my_line}" if my_line > 0 else f"{my_line}"

            # Total Logic
            pick_side = "OVER" if over_prob > 0.5 else "UNDER"

            new_rows.append({
                "GameID": gid, "HomeTeam": home, "AwayTeam": away,
                "Game": f"{away} @ {home}",
                "Spread Pick": f"{pick_team} ({fmt_line})", 
                "Spread Book": "Backfill",
                "Spread Conf": f"{max(cover_prob, 1-cover_prob):.1%}", 
                "Spread_Conf_Raw": max(cover_prob, 1-cover_prob),
                "Pick_Team": pick_team, 
                "Pick_Line": row['spread'],
                "Total Pick": f"{pick_side} {row['overUnder']}", 
                "Total Book": "Backfill",
                "Total Conf": f"{max(over_prob, 1-over_prob):.1%}", 
                "Total_Conf_Raw": max(over_prob, 1-over_prob),
                "Pick_Side": pick_side, 
                "Pick_Total": row['overUnder']
            })
            print("✅ Added!")
            
        except Exception as e:
            print(f"❌ Error predicting: {e}")

    # 5. Save to CSV
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if os.path.exists(HISTORY_FILE):
            try:
                old_df = pd.read_csv(HISTORY_FILE)
                # Ensure GameID is string for both to match correctly
                new_df['GameID'] = new_df['GameID'].astype(str)
                old_df['GameID'] = old_df['GameID'].astype(str)
                
                combined = pd.concat([new_df, old_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['GameID'], keep='first')
                combined.to_csv(HISTORY_FILE, index=False)
                print(f"\n💾 SUCCESS: Wrote {len(new_rows)} backfilled games to {HISTORY_FILE}")
            except Exception as e:
                print(f"\n❌ Error saving CSV: {e}")
        else:
            new_df.to_csv(HISTORY_FILE, index=False)
            print(f"\n💾 SUCCESS: Created new {HISTORY_FILE} with {len(new_rows)} games.")
    else:
        print("\n⚠️ Loop finished but NO games were added. (Check 'Skipping' messages above)")

if __name__ == "__main__":
    main()