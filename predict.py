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

# Normalization Map
TEAM_MAP = {
    "USF": "South Florida", "Ole Miss": "Mississippi", "LSU": "Louisiana State",
    "UConn": "Connecticut", "UMass": "Massachusetts", 
    "Southern Miss": "Southern Mississippi", "UL Monroe": "Louisiana Monroe", 
    "UL Lafayette": "Louisiana"
}

# --- 2. CACHING ---
def fetch_with_cache(filename, endpoint, params, max_age_hours=1):
    if os.path.exists(filename):
        if (time.time() - os.path.getmtime(filename)) < (max_age_hours * 3600):
            try:
                with open(filename, 'r') as f: return json.load(f)
            except: pass
    
    url = f"https://api.collegefootballdata.com{endpoint}"
    for attempt in range(3):
        try:
            time.sleep(1.5)
            response = requests.get(url, headers=HEADERS, params=params)
            if response.status_code == 429:
                time.sleep(60); continue
            response.raise_for_status()
            data = response.json()
            with open(filename, 'w') as f: json.dump(data, f)
            return data
        except: time.sleep(2)
    return []

def build_decay_lookup(year):
    print("   -> Fetching Advanced Stats...")
    stats = fetch_with_cache(f"cache_stats_{year}.json", "/stats/game/advanced", {"year": year})
    if not stats: return {}
    df = pd.json_normalize(stats)
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'])
        df = df.sort_values(['team', 'season', 'week'])
    
    lookup = {}
    metrics = ['offense.ppa', 'offense.successRate', 'defense.ppa', 'defense.successRate']
    for team, group in df.groupby('team'):
        team_mom = {}
        for m in metrics:
            val = group[m].ewm(span=3, adjust=False).mean().iloc[-1] if m in group.columns else 0.0
            team_mom[f"decay_{m}"] = val
        lookup[team] = team_mom
    return lookup

# --- 3. MAIN EXECUTION ---
def main():
    print("--- 🏈 CFB QUANT ENGINE: ROBUST POSTSEASON ---")
    YEAR = 2025
    
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        feat_cols = model_spread.feature_names_in_
    except: 
        print("❌ Models not found."); return

    # Fetch Data
    games_data = fetch_with_cache(f"cache_games_post_{YEAR}.json", "/games", {"year": YEAR, "seasonType": "postseason"})
    lines_data = fetch_with_cache(f"cache_lines_post_{YEAR}.json", "/lines", {"year": YEAR, "seasonType": "postseason"})
    srs_data = fetch_with_cache(f"cache_srs_{YEAR}.json", "/ratings/srs", {"year": YEAR})
    talent_data = fetch_with_cache(f"cache_talent_{YEAR}.json", "/talent", {"year": YEAR})
    
    srs_map = {x['team']: x['rating'] for x in srs_data}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data}
    decay_map = build_decay_lookup(YEAR)

    shopping_cart = {}
    for g in lines_data:
        valid_lines = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        shopping_cart[g['id']] = valid_lines

    current_week_preds = []
    print(f"   -> Scanning {len(games_data)} games...")
    
    skipped_completed = 0
    generated_count = 0

    for g in games_data:
        if g.get('completed'): 
            skipped_completed += 1
            continue
        
        gid = str(g.get('id'))
        home, away = g.get('home_team'), g.get('away_team')
        
        # ROBUST STATS: Default to empty dict if missing (Don't skip!)
        h_d = decay_map.get(home, {})
        a_d = decay_map.get(away, {})
        
        # Base Features
        base_row = {
            'home_talent_score': talent_map.get(home, 10), 'away_talent_score': talent_map.get(away, 10),
            'home_srs_rating': srs_map.get(home, -5), 'away_srs_rating': srs_map.get(away, -5),
            **{f"home_{k}":v for k,v in h_d.items()}, **{f"away_{k}":v for k,v in a_d.items()}
        }

        # --- 1. ALWAYS PREDICT STRAIGHT UP (Even if no lines) ---
        ml_row = base_row.copy()
        ml_row['spread'] = 0.0
        ml_row['overUnder'] = 55.5
        for col in feat_cols:
            if col not in ml_row: ml_row[col] = 0.0
        
        try:
            ml_probs = model_spread.predict_proba(pd.DataFrame([ml_row])[feat_cols])[0]
            ml_pick = home if ml_probs[1] > 0.5 else away
            ml_conf = max(ml_probs[1], 1 - ml_probs[1])
        except:
            continue

        # --- 2. PREDICT SPREAD/TOTAL (If lines exist) ---
        lines = shopping_cart.get(int(gid), [])
        
        best_spread = {"conf": -1, "pick": "N/A", "book": "N/A"}
        best_total = {"conf": -1, "pick": "N/A", "book": "N/A"}

        for line in lines:
            # Spread
            if line.get('spread') is not None:
                row = base_row.copy()
                row['spread'] = line.get('spread')
                row['overUnder'] = line.get('overUnder', 55.5)
                for col in feat_cols:
                    if col not in row: row[col] = 0.0
                
                sp_prob = model_spread.predict_proba(pd.DataFrame([row])[feat_cols])[0][1]
                s_conf = max(sp_prob, 1-sp_prob)
                
                if s_conf > best_spread['conf']:
                    p_team = home if sp_prob > 0.5 else away
                    p_line = line.get('spread') if sp_prob > 0.5 else -1 * line.get('spread')
                    fmt = f"+{p_line}" if p_line > 0 else f"{p_line}"
                    best_spread = {"conf": s_conf, "book": line.get('provider'), "pick": f"{p_team} ({fmt})", "raw_line": p_line, "pick_team": p_team}

            # Total
            if line.get('overUnder') is not None:
                row = base_row.copy()
                row['spread'] = line.get('spread', 0.0)
                row['overUnder'] = line.get('overUnder')
                for col in model_total.feature_names_in_:
                    if col not in row: row[col] = 0.0
                t_prob = model_total.predict_proba(pd.DataFrame([row])[model_total.feature_names_in_])[0][1]
                t_conf = max(t_prob, 1-t_prob)
                if t_conf > best_total['conf']:
                    side = "OVER" if t_prob > 0.5 else "UNDER"
                    best_total = {"conf": t_conf, "book": line.get('provider'), "pick": f"{side} {line.get('overUnder')}", "pick_side": side, "pick_val": line.get('overUnder')}

        # Add Prediction (Even if lines are missing, we add the ML pick)
        current_week_preds.append({
            "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
            "Moneyline Pick": ml_pick, "Moneyline Conf": f"{ml_conf:.1%}", "Moneyline_Conf_Raw": ml_conf,
            "Spread Pick": best_spread['pick'], "Spread Book": best_spread['book'], "Spread Conf": f"{best_spread['conf']:.1%}",
            "Total Pick": best_total['pick'], "Total Book": best_total['book'], "Total Conf": f"{best_total['conf']:.1%}",
            "Pick_Team": best_spread.get('pick_team', ''), "Pick_Line": best_spread.get('raw_line', 0),
            "Pick_Side": best_total.get('pick_side', ''), "Pick_Total": best_total.get('pick_val', 0)
        })
        generated_count += 1

    print(f"   -> Skipped {skipped_completed} completed games.")
    
    if current_week_preds:
        new_df = pd.DataFrame(current_week_preds)
        new_df['GameID'] = new_df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
            history_df['GameID'] = history_df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
            combined_df = pd.concat([new_df, history_df], ignore_index=True)
        else:
            combined_df = new_df
        
        combined_df['HomeTeam'] = combined_df['HomeTeam'].replace(TEAM_MAP)
        combined_df['AwayTeam'] = combined_df['AwayTeam'].replace(TEAM_MAP)
        combined_df = combined_df.drop_duplicates(subset=['GameID'], keep='first')
        combined_df = combined_df.drop_duplicates(subset=['HomeTeam', 'AwayTeam'], keep='first')
        
        combined_df.to_csv(HISTORY_FILE, index=False)
        print(f"✅ SUCCESS: Updated database with {generated_count} active predictions.")
    else:
        print("⚠️ Warning: No predictions generated. Is the API returning games?")

if __name__ == "__main__":
    main()