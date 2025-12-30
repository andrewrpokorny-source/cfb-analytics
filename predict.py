import os
import pandas as pd
import joblib
import time
from datetime import datetime
from api import fetch_with_retry
from config import HISTORY_FILE, VALID_BOOKS

YEAR = 2025

def main():
    print("--- 🏈 CFB QUANT ENGINE: DAILY UPDATE ---")
    
    # 1. LOAD & GRADE EXISTING HISTORY
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if 'GameID' in df.columns:
            df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
            
        if 'Manual_HomeScore' not in df.columns:
            df['Manual_HomeScore'] = pd.NA
            df['Manual_AwayScore'] = pd.NA

        pending_mask = df['Manual_HomeScore'].isna() & df['GameID'].notna()
        
        if pending_mask.any():
            print(f"   -> Checking scores for {pending_mask.sum()} pending games...")
            
            score_map = {}
            for stype in ["regular", "postseason"]:
                print(f"      -> Fetching {stype} scores from API...")
                games = fetch_with_retry("/games", {"year": YEAR, "seasonType": stype})
                
                if isinstance(games, list):
                    count = 0
                    for g in games:
                        # KEY FIX: Robust check for points keys
                        if g.get('completed'):
                            h_pts = g.get('home_points') if g.get('home_points') is not None else g.get('homePoints')
                            a_pts = g.get('away_points') if g.get('away_points') is not None else g.get('awayPoints')
                            
                            # Only add if we actually found numbers
                            if h_pts is not None and a_pts is not None:
                                score_map[str(g['id'])] = {'h': h_pts, 'a': a_pts}
                                count += 1
                    print(f"         Found {count} completed games.")
            
            graded_count = 0
            for idx, row in df[pending_mask].iterrows():
                gid = str(row['GameID'])
                if gid in score_map:
                    s = score_map[gid]
                    df.at[idx, 'Manual_HomeScore'] = s['h']
                    df.at[idx, 'Manual_AwayScore'] = s['a']
                    graded_count += 1
                    print(f"      ✅ Graded: {row['Game']} ({int(s['a'])}-{int(s['h'])})")
            
            if graded_count > 0:
                print(f"   -> Updated {graded_count} games in history.")
            else:
                print("   ⚠️ No matching scores found for pending games.")
    else:
        df = pd.DataFrame()

    # 2. PREDICTION LOGIC (Unchanged)
    print("   -> Scanning for new matchups...")
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        model_win = joblib.load("model_winner.pkl")
        feat_cols = model_spread.feature_names_in_
    except: 
        if not df.empty: df.to_csv(HISTORY_FILE, index=False)
        print("❌ Models missing."); return

    games = []; lines = []
    scenarios = [{"seasonType": "postseason", "week": 1}, {"seasonType": "regular", "week": 16}, {"seasonType": "regular", "week": 17}]
    
    for s in scenarios:
        g = fetch_with_retry("/games", {"year": YEAR, **s})
        l = fetch_with_retry("/lines", {"year": YEAR, **s})
        if isinstance(g, list): games.extend(g)
        if isinstance(l, list): lines.extend(l)

    existing_ids = df['GameID'].astype(str).tolist() if not df.empty and 'GameID' in df.columns else []
    lines_map = {}
    for g in lines:
        valid = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        lines_map[str(g['id'])] = valid

    srs = fetch_with_retry("/ratings/srs", {"year": YEAR})
    talent = fetch_with_retry("/talent", {"year": YEAR})
    srs_map = {x['team']: x['rating'] for x in srs} if isinstance(srs, list) else {}
    tal_map = {x.get('school', x.get('team')): x['talent'] for x in talent} if isinstance(talent, list) else {}

    new_predictions = []
    if games:
        for g in games:
            if not isinstance(g, dict) or g.get('completed'): continue
            gid = str(g.get('id'))
            if gid in existing_ids: continue
            
            home = g.get('home_team') or g.get('homeTeam')
            away = g.get('away_team') or g.get('awayTeam')
            if not home or not away: continue

            game_lines = lines_map.get(gid, [])
            if not game_lines: continue 

            best_spread = {"conf": 0.0, "pick": "Pending"}
            best_total = {"conf": 0.0, "pick": "Pending"}
            best_ml = {"conf": 0.0, "pick": "Pending"}
            active_odds = None

            for line in game_lines:
                spread_val = line.get('spread')
                total_val = line.get('overUnder')
                h_ml = line.get('homeMoneyline')
                a_ml = line.get('awayMoneyline')
                if spread_val is None or total_val is None: continue

                row = {
                    'spread': spread_val,
                    'overUnder': total_val,
                    'home_talent_score': tal_map.get(home, 10), 
                    'away_talent_score': tal_map.get(away, 10),
                    'home_srs_rating': srs_map.get(home, 0), 
                    'away_srs_rating': srs_map.get(away, 0)
                }
                input_df = pd.DataFrame([row])[feat_cols]

                # PREDICTIONS
                prob = model_spread.predict_proba(input_df)[0][1]
                conf = max(prob, 1-prob)
                if conf > best_spread['conf']:
                    p_team = home if prob > 0.5 else away
                    p_line = spread_val if prob > 0.5 else -spread_val
                    best_spread = {"conf": conf, "pick": f"{p_team} ({p_line})", "pick_team": p_team, "pick_line": p_line}

                prob = model_total.predict_proba(input_df)[0][1]
                conf = max(prob, 1-prob)
                if conf > best_total['conf']:
                    side = "OVER" if prob > 0.5 else "UNDER"
                    best_total = {"conf": conf, "pick": f"{side} {total_val}", "pick_side": side, "pick_val": total_val}
                    
                prob = model_win.predict_proba(input_df)[0][1]
                conf = max(prob, 1-prob)
                if conf > best_ml['conf']:
                    ml_team = home if prob > 0.5 else away
                    best_ml = {"conf": conf, "pick": ml_team}
                    active_odds = h_ml if ml_team == home else a_ml

            # Logic Enforcement
            if best_spread['conf'] > 0:
                if best_spread.get('pick_line', 0) < 0:
                    if best_ml['pick'] != best_spread['pick_team']:
                        best_ml['pick'] = best_spread['pick_team']
                        for line in game_lines:
                            h_ml, a_ml = line.get('homeMoneyline'), line.get('awayMoneyline')
                            active_odds = h_ml if best_ml['pick'] == home else a_ml
                            break

                new_predictions.append({
                    "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
                    "StartDate": g.get('start_date') or g.get('startDate'),
                    "Moneyline Pick": best_ml['pick'], "Moneyline Conf": f"{best_ml['conf']:.1%}", 
                    "Spread Pick": best_spread['pick'], "Spread Conf": f"{best_spread['conf']:.1%}", 
                    "Total Pick": best_total['pick'], "Total Conf": f"{best_total['conf']:.1%}",
                    "Pick_Team": best_spread.get('pick_team'), "Pick_Line": best_spread.get('pick_line'),
                    "Pick_Side": best_total.get('pick_side'), "Pick_Total": best_total.get('pick_val'),
                    "Pick_ML_Odds": active_odds
                })

    if new_predictions:
        print(f"   -> Added {len(new_predictions)} new forecasts.")
        final_df = pd.concat([pd.DataFrame(new_predictions), df], ignore_index=True)
    else:
        final_df = df
    
    final_df.to_csv(HISTORY_FILE, index=False)
    print("✅ SUCCESS: Database updated.")

if __name__ == "__main__":
    main()