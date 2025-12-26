import os
import pandas as pd
import joblib
import requests
import datetime
import time
import json
from dotenv import load_dotenv

# --- 1. SETUP ---
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

VALID_BOOKS = [
    'DraftKings', 'Draft Kings', 'FanDuel', 'BetMGM', 'Caesars', 
    'PointsBet', 'BetRivers', 'Unibet'
]
HISTORY_FILE = "live_predictions.csv"

# TEAM NAME MAPPING (For Auto-Cleaning)
TEAM_MAP = {
    "USF": "South Florida",
    "Ole Miss": "Mississippi",
    "LSU": "Louisiana State",
    "UConn": "Connecticut",
    "UMass": "Massachusetts",
    "Southern Miss": "Southern Mississippi",
    "UL Monroe": "Louisiana Monroe",
    "UL Lafayette": "Louisiana"
}

# --- 2. CHECKPOINT SYSTEM ---
def fetch_with_cache(filename, endpoint, params, max_age_hours=1):
    """
    Checks for local JSON first. If missing/stale, fetches from API and saves it.
    """
    if os.path.exists(filename):
        last_modified = os.path.getmtime(filename)
        if (time.time() - last_modified) < (max_age_hours * 3600):
            print(f"   📦 Loading {endpoint} from local cache...")
            try:
                with open(filename, 'r') as f: return json.load(f)
            except: pass
    
    url = f"https://api.collegefootballdata.com{endpoint}"
    retries = 3
    
    for attempt in range(retries):
        try:
            time.sleep(1.5) # Polite pause
            response = requests.get(url, headers=HEADERS, params=params)
            
            if response.status_code == 429:
                time.sleep(60)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            with open(filename, 'w') as f: json.dump(data, f)
            return data
            
        except Exception as e:
            print(f"   ⚠️ API Error: {e}")
            time.sleep(2)
            
    return []

def get_current_week(year, season_type):
    # Simplified week finder
    try:
        cal = fetch_with_cache(f"cache_calendar_{year}.json", "/calendar", {"year": year})
        today = datetime.datetime.now().isoformat()
        for w in cal:
            if w['seasonType'] == season_type and w['firstGameStart'] <= today <= w['lastGameStart']:
                return w['week']
        # If between weeks, look ahead
        for w in cal:
            if w['seasonType'] == season_type and w['firstGameStart'] > today:
                return w['week']
    except: pass
    return 1

def build_decay_lookup(year):
    print("   -> Fetching Advanced Stats...")
    stats = fetch_with_cache(f"cache_stats_{year}.json", "/stats/game/advanced", {"year": year})
    if not stats: return {}
    
    df = pd.json_normalize(stats)
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'])
        df = df.sort_values(['team', 'season', 'week'])
    
    metrics = ['offense.ppa', 'offense.successRate', 'defense.ppa', 'defense.successRate']
    lookup = {}
    
    for team, group in df.groupby('team'):
        team_mom = {}
        for m in metrics:
            if m in group.columns:
                team_mom[f"decay_{m}"] = group[m].ewm(span=3, adjust=False).mean().iloc[-1]
            else:
                team_mom[f"decay_{m}"] = 0.0
        lookup[team] = team_mom
    return lookup

# --- 3. MAIN EXECUTION ---
def main():
    print("--- 🏈 CFB QUANT ENGINE: PREDICT & CLEAN ---")
    YEAR = 2025
    SEASON_TYPE = "postseason"
    
    # A. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except: 
        print("❌ Models not found. Ensure .pkl files are present.")
        return

    WEEK = get_current_week(YEAR, SEASON_TYPE)
    print(f"   -> Analyizing {SEASON_TYPE} Week {WEEK}...")

    # B. Fetch Data
    games_data = fetch_with_cache(f"cache_games_w{WEEK}.json", "/games", {"year": YEAR, "seasonType": SEASON_TYPE, "week": WEEK})
    lines_data = fetch_with_cache(f"cache_lines_w{WEEK}.json", "/lines", {"year": YEAR, "seasonType": SEASON_TYPE, "week": WEEK})
    srs_data = fetch_with_cache(f"cache_srs_{YEAR}.json", "/ratings/srs", {"year": YEAR})
    talent_data = fetch_with_cache(f"cache_talent_{YEAR}.json", "/talent", {"year": YEAR})
    
    srs_map = {x['team']: x['rating'] for x in srs_data}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data}
    decay_map = build_decay_lookup(YEAR)

    # C. Organize Lines
    shopping_cart = {}
    for g in lines_data:
        valid_lines = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        shopping_cart[g['id']] = valid_lines

    current_week_preds = []
    
    # D. Generate Predictions
    print(f"   -> Running models on {len(games_data)} games...")
    for g in games_data:
        if g.get('completed'): continue # Skip finished games
        
        gid = str(g.get('id'))
        home, away = g.get('home_team'), g.get('away_team')
        
        lines = shopping_cart.get(int(gid), [])
        if not lines: continue
        
        h_d = decay_map.get(home)
        a_d = decay_map.get(away)
        if not h_d or not a_d: continue

        # Build Feature Row
        base_row = {
            'home_talent_score': talent_map.get(home, 10), 'away_talent_score': talent_map.get(away, 10),
            'home_srs_rating': srs_map.get(home, -5), 'away_srs_rating': srs_map.get(away, -5),
            **{f"home_{k}":v for k,v in h_d.items()}, **{f"away_{k}":v for k,v in a_d.items()}
        }

        # Find Best Value
        best_spread = {"conf": -1, "pick": "N/A", "book": "N/A", "raw_spread": 0, "pick_team": ""}
        best_total = {"conf": -1, "pick": "N/A", "book": "N/A", "raw_total": 0, "pick_side": ""}

        for line in lines:
            home_spread = line.get('spread')
            total_val = line.get('overUnder')
            book = line.get('provider')

            if home_spread is not None:
                row = base_row.copy()
                row['spread'] = home_spread
                row['overUnder'] = total_val if total_val else 55.5
                
                # Normalize features
                for col in model_spread.feature_names_in_:
                    if col not in row: row[col] = 0.0
                
                features = pd.DataFrame([row])[model_spread.feature_names_in_]
                cover_prob = model_spread.predict_proba(features)[0][1]
                
                s_conf = max(cover_prob, 1-cover_prob)
                if s_conf > best_spread['conf']:
                    pick_team = home if cover_prob > 0.5 else away
                    my_line = home_spread if cover_prob > 0.5 else -1 * home_spread
                    best_spread = {
                        "conf": s_conf, "book": book, 
                        "pick": f"{pick_team} ({'+' if my_line > 0 else ''}{my_line})",
                        "raw_spread": home_spread, "pick_team": pick_team
                    }

            if total_val is not None:
                row = base_row.copy()
                row['spread'] = home_spread if home_spread else 0.0
                row['overUnder'] = total_val
                
                for col in model_total.feature_names_in_:
                    if col not in row: row[col] = 0.0
                    
                features = pd.DataFrame([row])[model_total.feature_names_in_]
                over_prob = model_total.predict_proba(features)[0][1]
                
                t_conf = max(over_prob, 1-over_prob)
                if t_conf > best_total['conf']:
                    pick_side = "OVER" if over_prob > 0.5 else "UNDER"
                    best_total = {
                        "conf": t_conf, "book": book,
                        "pick": f"{pick_side} {total_val}",
                        "raw_total": total_val, "pick_side": pick_side
                    }

        if best_spread['conf'] != -1:
            current_week_preds.append({
                "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
                "Spread Pick": best_spread['pick'], "Spread Book": best_spread['book'],
                "Spread Conf": f"{best_spread['conf']:.1%}", "Spread_Conf_Raw": best_spread['conf'],
                "Pick_Team": best_spread['pick_team'], "Pick_Line": best_spread['raw_spread'],
                "Total Pick": best_total['pick'], "Total Book": best_total['book'],
                "Total Conf": f"{best_total['conf']:.1%}", "Total_Conf_Raw": best_total['conf'],
                "Pick_Side": best_total['pick_side'], "Pick_Total": best_total['raw_total']
            })

    # E. MERGE & CLEAN (Consolidated Logic)
    if current_week_preds:
        new_df = pd.DataFrame(current_week_preds)
        new_df['GameID'] = new_df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
            history_df['GameID'] = history_df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
            combined_df = pd.concat([new_df, history_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # 1. Normalize Team Names (Fixes USF/South Florida)
        combined_df['HomeTeam'] = combined_df['HomeTeam'].replace(TEAM_MAP)
        combined_df['AwayTeam'] = combined_df['AwayTeam'].replace(TEAM_MAP)
        
        # 2. Drop Duplicates (The "Smart" Way)
        # We prefer the 'new' prediction if dates match, but keep 'manual' history if present
        combined_df = combined_df.drop_duplicates(subset=['GameID'], keep='first')
        
        # 3. Final Safety Dedup by Team Matchup
        combined_df = combined_df.drop_duplicates(subset=['HomeTeam', 'AwayTeam'], keep='first')

        combined_df.to_csv(HISTORY_FILE, index=False)
        print(f"✅ SUCCESS: Updated database with {len(current_week_preds)} new predictions.")
    else:
        print("   (No active games found to predict)")

if __name__ == "__main__":
    main()