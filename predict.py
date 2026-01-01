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

        # 1.5 CALCULATE BETTING RESULTS
        if 'Manual_HomeScore' in df.columns and 'Manual_AwayScore' in df.columns:
            # Ensure numeric columns
            df['Manual_HomeScore'] = pd.to_numeric(df['Manual_HomeScore'], errors='coerce')
            df['Manual_AwayScore'] = pd.to_numeric(df['Manual_AwayScore'], errors='coerce')
            
            # Filter for rows that have scores but might not have results
            scored_mask = df['Manual_HomeScore'].notna() & df['Manual_AwayScore'].notna()
            
            if scored_mask.any():
                print(f"   -> Grading {scored_mask.sum()} completed games...")
                
                for idx, row in df[scored_mask].iterrows():
                    h_score = row['Manual_HomeScore']
                    a_score = row['Manual_AwayScore']
                    
                    # --- GRADE SPREAD ---
                    # Pick: "Team (Line)" e.g., "Louisiana Tech (-9.5)"
                    # We stored "Pick_Team" and "Pick_Line" separately for easier grading
                    p_team = row.get('Pick_Team')
                    p_line = row.get('Pick_Line')
                    
                    if pd.notna(p_team) and pd.notna(p_line):
                        # Calculate Margin from perspective of Pick Team
                        if p_team == row['HomeTeam']:
                            actual_margin = h_score - a_score
                        else:
                            actual_margin = a_score - h_score
                            
                        # Compare to spread line (e.g. -9.5)
                        # If Actual Margin (9) + Line (-9.5) > 0 ?? No.
                        # Standard Logic: If Team is -9.5, they must win by > 9.5. 
                        # So Margin (9) > 9.5? False.
                        # If Team is +3.5, they can lose by 3. Actual Margin (-3) > - (-3.5)? 
                        
                        # Simplified: Score + Line > Opponent Score?
                        # If Pick is Home: (Home + Line) > Away
                        if p_team == row['HomeTeam']:
                            cover = (h_score + p_line) > a_score
                            push = (h_score + p_line) == a_score
                        else:
                            cover = (a_score + p_line) > h_score
                            push = (a_score + p_line) == h_score
                            
                        res = "PUSH" if push else ("WIN" if cover else "LOSS")
                        df.at[idx, 'Spread_Result'] = res

                    # --- GRADE TOTAL ---
                    p_side = row.get('Pick_Side') # OVER / UNDER
                    p_total = row.get('Pick_Total')
                    
                    if pd.notna(p_side) and pd.notna(p_total):
                        total_score = h_score + a_score
                        if p_side == "OVER":
                            res = "WIN" if total_score > p_total else ("LOSS" if total_score < p_total else "PUSH")
                        else: # UNDER
                            res = "WIN" if total_score < p_total else ("LOSS" if total_score > p_total else "PUSH")
                        df.at[idx, 'Total_Result'] = res
                        
                    # --- GRADE MONEYLINE ---
                    p_ml_pick = row.get('Moneyline Pick')
                    if pd.notna(p_ml_pick):
                        winner = row['HomeTeam'] if h_score > a_score else row['AwayTeam']
                        if p_ml_pick == winner:
                            df.at[idx, 'ML_Result'] = "WIN"
                        else:
                            df.at[idx, 'ML_Result'] = "LOSS"

    else:
        df = pd.DataFrame()

    # 2. PREDICTION LOGIC
    print("   -> Scanning for new matchups...")
    try:
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
        model_win = joblib.load("model_winner.pkl")
        feat_cols = model_spread.feature_names_in_
    except: 
        if not df.empty: df.to_csv(HISTORY_FILE, index=False)
        print("❌ Models missing."); return

    # REFRESH LOGIC: Keep only completed games in history, regenerate all pending
    if not df.empty and 'Manual_HomeScore' in df.columns:
        # Keep rows where game is already graded (Score exists)
        # OR keep rows that are clearly past (to preserve history even if not graded? No, usually we grade them.)
        # Let's just keep graded ones.
        graded_mask = df['Manual_HomeScore'].notna()
        completed_games_df = df[graded_mask].copy()
        
        # Determine IDs we should NOT predict again (Completed IDs)
        existing_ids = completed_games_df['GameID'].astype(str).tolist()
        
        # We will append new predictions to this 'clean' history
        df = completed_games_df 
        print(f"   -> Refreshing pending lines (Keeping {len(df)} graded games)...")
    else:
        existing_ids = []

    games = []; lines = []
    scenarios = [{"seasonType": "postseason", "week": 1}, {"seasonType": "regular", "week": 16}, {"seasonType": "regular", "week": 17}]
    
    for s in scenarios:
        g = fetch_with_retry("/games", {"year": YEAR, **s})
        l = fetch_with_retry("/lines", {"year": YEAR, **s})
        if isinstance(g, list): games.extend(g)
        if isinstance(l, list): lines.extend(l)    

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

            # Calculate Consensus (Median) Lines
            spreads = [l.get('spread') for l in game_lines if l.get('spread') is not None]
            totals = [l.get('overUnder') for l in game_lines if l.get('overUnder') is not None]

            if not spreads or not totals: continue

            import statistics
            median_spread = statistics.median(spreads)
            median_total = statistics.median(totals)

            # Find best available odds for the median line (or closest to it)
            # For now, we just use the median line as the "Market" truth
            
            row = {
                'spread': median_spread,
                'overUnder': median_total,
                'home_talent_score': tal_map.get(home, 10), 
                'away_talent_score': tal_map.get(away, 10),
                'home_srs_rating': srs_map.get(home, 0), 
                'away_srs_rating': srs_map.get(away, 0)
            }
            input_df = pd.DataFrame([row])[feat_cols]

            # PREDICTIONS
            # 1. Spread
            prob = model_spread.predict_proba(input_df)[0][1] # Prob Home Covers
            conf_spread = max(prob, 1-prob)
            p_spread_team = home if prob > 0.5 else away
            p_spread_line = median_spread if prob > 0.5 else -median_spread
            
            # 2. Total
            prob = model_total.predict_proba(input_df)[0][1] # Prob Over
            conf_total = max(prob, 1-prob)
            p_total_side = "OVER" if prob > 0.5 else "UNDER"
            
            # 3. Moneyline
            prob = model_win.predict_proba(input_df)[0][1] # Prob Home Win
            conf_ml = max(prob, 1-prob)
            p_ml_team = home if prob > 0.5 else away
            
            # Find best odds for the moneyline pick
            active_odds = None
            for line in game_lines:
                h_ml = line.get('homeMoneyline')
                a_ml = line.get('awayMoneyline')
                if p_ml_team == home and h_ml:
                    if active_odds is None or h_ml > active_odds: active_odds = h_ml
                elif p_ml_team == away and a_ml:
                    if active_odds is None or a_ml > active_odds: active_odds = a_ml

            # Store Prediction
            new_predictions.append({
                "GameID": gid, "HomeTeam": home, "AwayTeam": away, "Game": f"{away} @ {home}",
                "StartDate": g.get('start_date') or g.get('startDate'),
                "Moneyline Pick": p_ml_team, "Moneyline Conf": f"{conf_ml:.1%}", 
                "Spread Pick": f"{p_spread_team} ({p_spread_line})", "Spread Conf": f"{conf_spread:.1%}", 
                "Total Pick": f"{p_total_side} {median_total}", "Total Conf": f"{conf_total:.1%}",
                "Pick_Team": p_spread_team, "Pick_Line": p_spread_line,
                "Pick_Side": p_total_side, "Pick_Total": median_total,
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