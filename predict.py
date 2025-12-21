import os
import pandas as pd
import joblib
import requests
import datetime
import time
import json
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

VALID_BOOKS = [
    'DraftKings', 'Draft Kings', 'FanDuel', 'BetMGM', 'Caesars', 
    'PointsBet', 'BetRivers', 'Unibet'
]
HISTORY_FILE = "live_predictions.csv"

# 2. CHECKPOINT SYSTEM (Saves progress to avoid API crashes)
def fetch_with_cache(filename, endpoint, params, max_age_hours=1):
    """
    Checks for a local JSON file first. If valid/recent, uses that.
    If not, fetches from API and saves it. 
    """
    # Check if cache exists and is fresh
    if os.path.exists(filename):
        last_modified = os.path.getmtime(filename)
        if (time.time() - last_modified) < (max_age_hours * 3600):
            print(f"   📦 Loading {endpoint} from local cache...")
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except:
                print("      (Cache corrupted, re-fetching...)")
    
    # If no cache, fetch from API with strict backoff
    url = f"https://api.collegefootballdata.com{endpoint}"
    retries = 5
    
    for attempt in range(retries):
        try:
            time.sleep(2) # Polite pause
            response = requests.get(url, headers=HEADERS, params=params)
            
            if response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                print(f"   🛑 Rate Limit Hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            # SAVE TO CACHE
            with open(filename, 'w') as f:
                json.dump(data, f)
            
            return data
            
        except Exception as e:
            print(f"   ⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(5)
            
    print(f"❌ Failed to fetch {endpoint}")
    return []

def get_current_week(year, season_type):
    try:
        cal = fetch_with_cache(f"cache_calendar_{year}.json", "/calendar", {"year": year})
        today = datetime.datetime.now().isoformat()
        
        for week_obj in cal:
            if week_obj['seasonType'] == season_type:
                if week_obj['firstGameStart'] <= today <= week_obj['lastGameStart']:
                    return week_obj['week']
        # Fallback to upcoming
        for week_obj in cal:
            if week_obj['seasonType'] == season_type:
                if week_obj['firstGameStart'] > today:
                    return week_obj['week']
    except: pass
    return 1

def build_decay_lookup(year):
    print("   -> Fetching Stats (cached)...")
    stats = fetch_with_cache(f"cache_stats_{year}.json", "/stats/game/advanced", {"year": year})
    
    if not stats: return {}
    
    df = pd.json_normalize(stats)
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'])
        df = df.sort_values(['team', 'season', 'week'])
    
    metrics = ['offense.ppa', 'offense.successRate', 'offense.explosiveness', 'defense.ppa', 'defense.successRate', 'defense.explosiveness']
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

def main():
    print("--- 🏈 CFB PREDICTOR (FINAL STABLE VERSION) 🏈 ---")
    YEAR = 2025
    SEASON_TYPE = "postseason"
    
    # 1. Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except: 
        print("❌ Models not found.")
        return

    # 2. Dynamic Week
    WEEK = get_current_week(YEAR, SEASON_TYPE)
    print(f"   -> Processing {SEASON_TYPE} Week {WEEK}")

    # 3. Data Fetching (With Checkpoints)
    games_data = fetch_with_cache(f"cache_games_w{WEEK}.json", "/games", {"year": YEAR, "seasonType": SEASON_TYPE, "week": WEEK})
    lines_data = fetch_with_cache(f"cache_lines_w{WEEK}.json", "/lines", {"year": YEAR, "seasonType": SEASON_TYPE, "week": WEEK})
    srs_data = fetch_with_cache(f"cache_srs_{YEAR}.json", "/ratings/srs", {"year": YEAR})
    talent_data = fetch_with_cache(f"cache_talent_{YEAR}.json", "/talent", {"year": YEAR})
    
    srs_map = {x['team']: x['rating'] for x in srs_data}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data}
    decay_map = build_decay_lookup(YEAR)

    # 4. Processing
    shopping_cart = {}
    for g in lines_data:
        valid_lines = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        shopping_cart[g['id']] = valid_lines

    current_week_preds = []
    games = pd.DataFrame(games_data).rename(columns={'homeTeam': 'home_team', 'awayTeam': 'away_team'}).to_dict('records')

    print(f"   -> Generating Predictions for {len(games)} games...")

    for g in games:
        if g.get('completed'): continue
        gid = g.get('id')
        home, away = g.get('home_team'), g.get('away_team')
        
        lines = shopping_cart.get(gid, [])
        if not lines: continue
        
        h_d, a_d = decay_map.get(home), decay_map.get(away)
        if not h_d or not a_d: continue

        base_row = {
            'home_talent_score': talent_map.get(home, 10), 'away_talent_score': talent_map.get(away, 10),
            'home_srs_rating': srs_map.get(home, -5), 'away_srs_rating': srs_map.get(away, -5),
            **{f"home_{k}":v for k,v in h_d.items()}, **{f"away_{k}":v for k,v in a_d.items()}
        }

        best_spread = {"conf": -1, "book": "N/A", "pick": "N/A", "raw_spread": 0, "pick_team": ""}
        best_total = {"conf": -1, "book": "N/A", "pick": "N/A", "raw_total": 0, "pick_side": ""}

        for line in lines:
            home_spread_val = line.get('spread')
            total_val = line.get('overUnder')
            book = line.get('provider')

            if home_spread_val is not None:
                row = base_row.copy()
                row['spread'] = home_spread_val
                row['overUnder'] = total_val if total_val else 55.0
                feat_cols = model_spread.feature_names_in_
                features = pd.DataFrame([row])[feat_cols]
                cover_prob = model_spread.predict_proba(features)[0][1]
                s_conf = max(cover_prob, 1-cover_prob)
                if s_conf > best_spread['conf']:
                    if cover_prob > 0.5:
                        pick_team = home
                        my_line = home_spread_val 
                    else:
                        pick_team = away
                        my_line = -1 * home_spread_val
                    fmt_line = f"+{my_line}" if my_line > 0 else f"{my_line}"
                    best_spread = {"conf": s_conf, "book": book, "pick": f"{pick_team} ({fmt_line})", "raw_spread": home_spread_val, "pick_team": pick_team}

            if total_val is not None:
                row = base_row.copy()
                row['spread'] = home_spread_val if home_spread_val else 0.0 
                row['overUnder'] = total_val
                feat_cols = model_spread.feature_names_in_
                features = pd.DataFrame([row])[feat_cols]
                over_prob = model_total.predict_proba(features)[0][1]
                t_conf = max(over_prob, 1-over_prob)
                if t_conf > best_total['conf']:
                    best_total = {"conf": t_conf, "book": book, "pick": f"{'OVER' if over_prob > 0.5 else 'UNDER'} {total_val}", "raw_total": total_val, "pick_side": "OVER" if over_prob > 0.5 else "UNDER"}

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

    # 5. DATABASE MERGE (DUPLICATE PROOF)
    if current_week_preds:
        new_df = pd.DataFrame(current_week_preds)
        
        # Ensure New IDs are strict strings
        new_df['GameID'] = new_df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        if os.path.exists(HISTORY_FILE):
            try:
                history_df = pd.read_csv(HISTORY_FILE)
                # Ensure Old IDs are strict strings
                history_df['GameID'] = history_df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
                
                # Combine and remove duplicates based on GameID
                combined_df = pd.concat([new_df, history_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['GameID'], keep='first')
            except:
                combined_df = new_df
        else:
            combined_df = new_df

        combined_df.to_csv(HISTORY_FILE, index=False)
        print(f"\n✅ SUCCESS: Updated {HISTORY_FILE} with active games.")
    else:
        print("No active games found for this week.")

if __name__ == "__main__":
    main()