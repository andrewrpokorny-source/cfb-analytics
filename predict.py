import os
import pandas as pd
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# STRICTLY US REGULATED BOOKS
# Removed: Bovada, BetOnline
VALID_BOOKS = [
    'DraftKings', 
    'FanDuel', 
    'BetMGM', 
    'Caesars', 
    'PointsBet', 
    'BetRivers', 
    'Unibet'
]

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
    print("--- 🏈 CFB PREDICTOR (US REGULATED BOOKS ONLY) 🏈 ---")
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
        # Filter strictly for the VALID_BOOKS list
        valid_lines = [l for l in g.get('lines', []) if l.get('provider') in VALID_BOOKS]
        shopping_cart[g['id']] = valid_lines

    final_predictions = []
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

        best_spread = {"conf": -1, "book": "N/A", "pick": "N/A"}
        best_total = {"conf": -1, "book": "N/A", "pick": "N/A"}

        # SHOP EVERY BOOK
        for line in lines:
            spread_val = line.get('spread')
            total_val = line.get('overUnder')
            book = line.get('provider')

            # 1. Evaluate SPREAD
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
                        "conf": s_conf,
                        "book": book,
                        "pick": f"{home if cover_prob > 0.5 else away} ({spread_val})"
                    }

            # 2. Evaluate TOTAL
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
                        "conf": t_conf,
                        "book": book,
                        "pick": f"{'OVER' if over_prob > 0.5 else 'UNDER'} {total_val}"
                    }

        if best_spread['conf'] != -1:
            final_predictions.append({
                "Game": f"{away} @ {home}",
                "Spread Pick": best_spread['pick'],
                "Spread Book": best_spread['book'],
                "Spread Conf": f"{best_spread['conf']:.1%}",
                "Spread_Conf_Raw": best_spread['conf'],
                "Total Pick": best_total['pick'],
                "Total Book": best_total['book'],
                "Total Conf": f"{best_total['conf']:.1%}",
                "Total_Conf_Raw": best_total['conf']
            })

    if final_predictions:
        df = pd.DataFrame(final_predictions)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print("\n" + "="*50)
        print("🎯 TOP SPREAD EDGES (Legit Books Only)")
        print("="*50)
        spread_view = df.sort_values("Spread_Conf_Raw", ascending=False)
        print(spread_view[['Game', 'Spread Pick', 'Spread Book', 'Spread Conf']].head(20).to_string(index=False))

        print("\n" + "="*50)
        print("📉 TOP TOTALS EDGES (Over/Under)")
        print("="*50)
        total_view = df.sort_values("Total_Conf_Raw", ascending=False)
        print(total_view[['Game', 'Total Pick', 'Total Book', 'Total Conf']].head(20).to_string(index=False))
        
        df.to_csv("live_predictions.csv", index=False)
    else:
        print("No lines found. (API might be returning only Consensus or Offshore lines for these specific games).")

if __name__ == "__main__":
    main()