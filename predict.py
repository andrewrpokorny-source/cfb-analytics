import os
import pandas as pd
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet']
HISTORY_FILE = "live_predictions.csv"

def get_data(endpoint, params):
    try:
        response = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
        return []

def build_decay_lookup(year):
    print("   -> Fetching full season stats...")
    stats = get_data("/stats/game/advanced", {"year": year})
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
    print("--- 🏈 CFB PREDICTOR (DATABASE MODE) 🏈 ---")
    YEAR = 2025; SEASON_TYPE = "postseason"; WEEK = 1
    
    try:
        model_win = joblib.load("model_winner.pkl")
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except: 
        print("Models not found.")
        return

    print(f"Fetching Data for {YEAR} Bowl Season...")
    games_data = get_data("/games", {"year": YEAR, "seasonType": SEASON_TYPE, "week": WEEK})
    lines_data = get_data("/lines", {"year": YEAR, "seasonType": SEASON_TYPE, "week": WEEK})
    srs_data = get_data("/ratings/srs", {"year": YEAR})
    talent_data = get_data("/talent", {"year": YEAR})
    
    srs_map = {x['team']: x['rating'] for x in srs_data}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data}
    decay_map = build_decay_lookup(YEAR)

    shopping_cart = {}
    for g in lines_data:
        valid_lines = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        shopping_cart[g['id']] = valid_lines

    current_week_preds = []
    games = pd.DataFrame(games_data).rename(columns={'homeTeam': 'home_team', 'awayTeam': 'away_team'}).to_dict('records')

    print(f"   -> Shopping across {len(VALID_BOOKS)} legitimate books...")

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
            spread_val = line.get('spread')
            total_val = line.get('overUnder')
            book = line.get('provider')

            if spread_val is not None:
                row = base_row.copy()
                row['spread'] = spread_val
                row['overUnder'] = total_val if total_val else 55.0
                feat_cols = model_spread.feature_names_in_
                features = pd.DataFrame([row])[feat_cols]
                cover_prob = model_spread.predict_proba(features)[0][1]
                s_conf = max(cover_prob, 1-cover_prob)
                if s_conf > best_spread['conf']:
                    best_spread = {
                        "conf": s_conf, "book": book, "pick": f"{home if cover_prob > 0.5 else away} ({spread_val})",
                        "raw_spread": spread_val, "pick_team": home if cover_prob > 0.5 else away
                    }

            if total_val is not None:
                row = base_row.copy()
                row['spread'] = spread_val if spread_val else 0.0 
                row['overUnder'] = total_val
                feat_cols = model_spread.feature_names_in_
                features = pd.DataFrame([row])[feat_cols]
                over_prob = model_total.predict_proba(features)[0][1]
                t_conf = max(over_prob, 1-over_prob)
                if t_conf > best_total['conf']:
                    best_total = {
                        "conf": t_conf, "book": book, "pick": f"{'OVER' if over_prob > 0.5 else 'UNDER'} {total_val}",
                        "raw_total": total_val, "pick_side": "OVER" if over_prob > 0.5 else "UNDER"
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

    # --- DATABASE MERGE LOGIC ---
    if current_week_preds:
        new_df = pd.DataFrame(current_week_preds)
        
        # 1. Load History if exists
        if os.path.exists(HISTORY_FILE):
            try:
                history_df = pd.read_csv(HISTORY_FILE)
                # 2. Combine: Put NEWEST predictions at the top
                combined_df = pd.concat([new_df, history_df], ignore_index=True)
                # 3. Deduplicate: Keep the NEWEST version of any GameID
                # (This ensures if lines updated today, we see the new one)
                combined_df = combined_df.drop_duplicates(subset=['GameID'], keep='first')
            except Exception as e:
                print(f"Error reading history, starting fresh: {e}")
                combined_df = new_df
        else:
            combined_df = new_df

        # Save back to CSV
        combined_df.to_csv(HISTORY_FILE, index=False)
        print(f"\n✅ Database Updated: {len(combined_df)} total tracked games.")
        
        # Show Top Picks
        print("\n" + "="*50)
        print("🎯 TOP SPREAD EDGES (Current Week)")
        print("="*50)
        print(new_df.sort_values("Spread_Conf_Raw", ascending=False)[['Game', 'Spread Pick', 'Spread Conf']].head(10).to_string(index=False))
    else:
        print("No new games found.")

if __name__ == "__main__":
    main()