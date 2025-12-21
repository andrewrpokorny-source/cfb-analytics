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
        time.sleep(1)
        res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        if res.status_code == 200: return res.json()
        print(f"⚠️ API Error {res.status_code} for {endpoint}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    return []

def main():
    print("--- 🎯 TARGETED BACKFILL (WEEK 1 SPECIFIC) ---")
    
    # 1. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except:
        print("❌ Models missing.")
        return

    # 2. Fetch Specifically Postseason Week 1
    print("   -> Fetching Postseason Week 1 Games...")
    games_w1 = get_data("/games", {"year": 2025, "seasonType": "postseason", "week": 1})
    
    # Also grab Regular Season Week 15/16 just in case
    print("   -> Fetching Late Regular Season Games...")
    games_reg = get_data("/games", {"year": 2025, "seasonType": "regular", "week": 16})
    
    all_games = games_w1 + games_reg
    print(f"   -> Found {len(all_games)} total games in search window.")

    # Filter for COMPLETED only
    # Note: We verify 'home_points' exists to ensure the score is final
    completed_games = [
        g for g in all_games 
        if isinstance(g, dict) 
        and g.get('status') == 'completed'
        and g.get('home_points') is not None
    ]
    
    print(f"   -> Identified {len(completed_games)} COMPLETED games ready for grading.")

    if not completed_games:
        print("   🛑 Still found 0 completed games. (This is weird - check the dates manually?)")
        if all_games:
            print(f"      Example game status: {all_games[0].get('status')}")
        return

    # 3. Fetch Lines & Stats
    print("   -> Fetching Lines & Stats Context...")
    # Fetch lines for both contexts
    lines_post = get_data("/lines", {"year": 2025, "seasonType": "postseason", "week": 1})
    lines_reg = get_data("/lines", {"year": 2025, "seasonType": "regular", "week": 16})
    all_lines = lines_post + lines_reg
    
    stats = get_data("/stats/game/advanced", {"year": 2025})
    talent = get_data("/talent", {"year": 2025})
    srs = get_data("/ratings/srs", {"year": 2025})
    
    # Mappers
    lines_map = {str(g['id']): g['lines'] for g in all_lines if 'id' in g}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent}
    srs_map = {x['team']: x['rating'] for x in srs}
    
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
    
    # 4. Predict & Grade
    print("\n   -> Re-simulating predictions...")
    for g in completed_games:
        gid = str(g['id'])
        home, away = g['home_team'], g['away_team']
        
        # FIND LINE
        game_lines = lines_map.get(gid, [])
        valid_line = None
        for l in game_lines:
            prov = l.get('provider', '')
            if 'Draft' in prov or 'Kings' in prov: valid_line = l; break
        if not valid_line and game_lines: valid_line = game_lines[0]
            
        if not valid_line: continue

        # BUILD ROW
        h_d = decay_map.get(home)
        a_d = decay_map.get(away)
        if not h_d or not a_d: continue

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
        
        for col in model_spread.feature_names_in_:
            if col not in row: row[col] = 0.0

        try:
            feats = pd.DataFrame([row])[model_spread.feature_names_in_]
            cover_prob = model_spread.predict_proba(feats)[0][1]
            over_prob = model_total.predict_proba(feats)[0][1]
            
            if cover_prob > 0.5: pick_team, my_line = home, row['spread']
            else: pick_team, my_line = away, -1 * row['spread']
            
            pick_side = "OVER" if over_prob > 0.5 else "UNDER"

            new_rows.append({
                "GameID": gid, "HomeTeam": home, "AwayTeam": away,
                "Game": f"{away} @ {home}",
                "Spread Pick": f"{pick_team} ({'+' if my_line > 0 else ''}{my_line})", 
                "Spread Book": "Backfill", "Spread Conf": f"{max(cover_prob, 1-cover_prob):.1%}", 
                "Spread_Conf_Raw": max(cover_prob, 1-cover_prob),
                "Pick_Team": pick_team, "Pick_Line": row['spread'],
                "Total Pick": f"{pick_side} {row['overUnder']}", 
                "Total Book": "Backfill", "Total Conf": f"{max(over_prob, 1-over_prob):.1%}", 
                "Total_Conf_Raw": max(over_prob, 1-over_prob),
                "Pick_Side": pick_side, "Pick_Total": row['overUnder']
            })
            print(f"      ✅ Processed {away} @ {home}")
            
        except: pass

    # 5. Save
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if os.path.exists(HISTORY_FILE):
            try:
                old_df = pd.read_csv(HISTORY_FILE)
                # Important: Convert GameID to string to ensure matching
                new_df['GameID'] = new_df['GameID'].astype(str)
                old_df['GameID'] = old_df['GameID'].astype(str)
                
                combined = pd.concat([new_df, old_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['GameID'], keep='first')
                combined.to_csv(HISTORY_FILE, index=False)
                print(f"\n💾 MERGED {len(new_rows)} games into history.")
            except:
                new_df.to_csv(HISTORY_FILE, index=False)
                print(f"\n💾 SAVED new history file with {len(new_rows)} games.")
        else:
            new_df.to_csv(HISTORY_FILE, index=False)
            print(f"\n💾 CREATED history file with {len(new_rows)} games.")

if __name__ == "__main__":
    main()