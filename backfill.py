import os
import pandas as pd
import joblib
import requests
import json
import math
import time
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
HISTORY_FILE = "live_predictions.csv"
YEAR = 2025

VALID_BOOKS = [
    'DraftKings', 'Draft Kings', 'FanDuel', 'BetMGM', 'Caesars', 
    'PointsBet', 'BetRivers', 'Unibet'
]

# Normalization Map
TEAM_MAP = {
    "USF": "South Florida", "Ole Miss": "Mississippi", "LSU": "Louisiana State",
    "UConn": "Connecticut", "UMass": "Massachusetts", 
    "Southern Miss": "Southern Mississippi", "UL Monroe": "Louisiana Monroe", 
    "UL Lafayette": "Louisiana"
}

def fetch_data(endpoint, params):
    try:
        time.sleep(0.5)
        res = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        return res.json()
    except: return []

def calculate_win_prob(home_srs, away_srs, home_talent, away_talent):
    # Performance Override Formula
    talent_diff = (home_talent - away_talent) / 200.0
    srs_diff = home_srs - away_srs
    total_edge = srs_diff + talent_diff
    try: prob = 1 / (1 + math.exp(-1 * total_edge / 7.5))
    except: prob = 0.5
    return prob

def main():
    print("--- 🕰️ RUNNING HISTORY BACKFILL ---")
    
    # 1. Load Models
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        feat_cols = model_spread.feature_names_in_
    except:
        print("❌ Models missing. Run train_models.py first.")
        return

    # 2. Fetch Context Data (Stats, SRS, Talent)
    print("   -> Fetching season stats...")
    stats = fetch_data("/stats/game/advanced", {"year": YEAR})
    srs = fetch_data("/ratings/srs", {"year": YEAR})
    talent = fetch_data("/talent", {"year": YEAR})
    
    # Build Maps
    srs_map = {x['team']: x['rating'] for x in srs}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent}
    
    # Decay Stats Map
    decay_map = {}
    if stats:
        df_stats = pd.json_normalize(stats)
        # Ensure numeric
        metrics = ['offense.ppa', 'offense.successRate', 'defense.ppa', 'defense.successRate']
        for team, group in df_stats.groupby('team'):
            t_mom = {}
            for m in metrics:
                if m in group.columns:
                    # Simple average for backfill is sufficient
                    t_mom[f"decay_{m}"] = group[m].mean()
                else:
                    t_mom[f"decay_{m}"] = 0.0
            decay_map[team] = t_mom

    # 3. Define Time Periods to Backfill
    # We want Conf Champ (Week 15) and Postseason
    periods = [
        {"type": "regular", "week": 15},
        {"type": "postseason", "week": 1}
    ]

    new_rows = []

    for p in periods:
        print(f"   -> Processing {p['type']} week {p['week']}...")
        games = fetch_data("/games", {"year": YEAR, "seasonType": p['type'], "week": p['week']})
        lines_data = fetch_data("/lines", {"year": YEAR, "seasonType": p['type'], "week": p['week']})
        
        # Map lines to Game ID
        lines_map = {}
        for g in lines_data:
            valid_lines = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
            lines_map[g['id']] = valid_lines

        for g in games:
            # IMPORTANT: Only process COMPLETED games for backfill
            if not g.get('completed'):
                continue

            gid = str(g['id'])
            home = g.get('home_team')
            away = g.get('away_team')
            start_date = g.get('start_date')

            # Skip if stats missing
            h_d = decay_map.get(home, {})
            a_d = decay_map.get(away, {})
            if not h_d or not a_d: continue

            # Build Features
            h_srs, a_srs = srs_map.get(home, 0), srs_map.get(away, 0)
            h_tal, a_tal = talent_map.get(home, 10), talent_map.get(away, 10)

            base_row = {
                'home_talent_score': h_tal, 'away_talent_score': a_tal,
                'home_srs_rating': h_srs, 'away_srs_rating': a_srs,
                **{f"home_{k}":v for k,v in h_d.items()}, **{f"away_{k}":v for k,v in a_d.items()}
            }

            # A. Moneyline
            ml_prob = calculate_win_prob(h_srs, a_srs, h_tal, a_tal)
            ml_pick = home if ml_prob > 0.5 else away
            ml_conf = max(ml_prob, 1 - ml_prob)

            # B. Spread/Total
            game_lines = lines_map.get(int(gid), [])
            best_spread = {"conf": -1, "pick": "N/A", "book": "N/A"}
            best_total = {"conf": -1, "pick": "N/A", "book": "N/A"}
            
            # If no lines found (common for old games in API), skip spread/total but keep ML
            if not game_lines:
                # Add just the Moneyline prediction
                new_rows.append({
                    "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
                    "StartDate": start_date,
                    "Moneyline Pick": ml_pick, "Moneyline Conf": f"{ml_conf:.1%}",
                    "Spread Pick": "N/A", "Spread Conf": "0%",
                    "Total Pick": "N/A", "Total Conf": "0%"
                })
                continue

            for line in game_lines:
                # Spread
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
                        best_spread = {
                            "conf": conf, "book": line.get('provider'),
                            "pick": f"{p_team} ({'+' if p_line > 0 else ''}{p_line})",
                            "raw_line": p_line, "pick_team": p_team
                        }

                # Total
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
                        best_total = {
                            "conf": conf, "book": line.get('provider'),
                            "pick": f"{side} {line.get('overUnder')}",
                            "pick_side": side, "pick_val": line.get('overUnder')
                        }

            new_rows.append({
                "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
                "StartDate": start_date,
                "Moneyline Pick": ml_pick, "Moneyline Conf": f"{ml_conf:.1%}", "Moneyline_Conf_Raw": ml_conf,
                "Spread Pick": best_spread['pick'], "Spread Book": best_spread['book'], "Spread Conf": f"{best_spread['conf']:.1%}",
                "Total Pick": best_total['pick'], "Total Book": best_total['book'], "Total Conf": f"{best_total['conf']:.1%}",
                "Pick_Team": best_spread.get('pick_team'), "Pick_Line": best_spread.get('raw_line'),
                "Pick_Side": best_total.get('pick_side'), "Pick_Total": best_total.get('pick_val')
            })

    # 4. Save/Append
    if new_rows:
        backfill_df = pd.DataFrame(new_rows)
        backfill_df['GameID'] = backfill_df['GameID'].astype(str)
        backfill_df['HomeTeam'] = backfill_df['HomeTeam'].replace(TEAM_MAP)
        backfill_df['AwayTeam'] = backfill_df['AwayTeam'].replace(TEAM_MAP)

        if os.path.exists(HISTORY_FILE):
            existing_df = pd.read_csv(HISTORY_FILE)
            existing_df['GameID'] = existing_df['GameID'].astype(str)
            combined = pd.concat([backfill_df, existing_df], ignore_index=True)
        else:
            combined = backfill_df

        # Dedup (keep first)
        combined = combined.drop_duplicates(subset=['GameID'], keep='first')
        combined.to_csv(HISTORY_FILE, index=False)
        print(f"✅ SUCCESS: Injected {len(new_rows)} historical games into the database.")
    else:
        print("⚠️ No historical games found to backfill.")

if __name__ == "__main__":
    main()