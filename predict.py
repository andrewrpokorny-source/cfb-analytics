import os
import pandas as pd
import joblib
import requests
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

def get_data(endpoint, params):
    try:
        response = requests.get(f"https://api.collegefootballdata.com{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
        return []

def build_decay_lookup(year):
    """
    Fetches ALL stats for the year in ONE call and calculates
    current momentum for every team at once.
    """
    print("   -> Fetching full season stats (Batch Request)...")
    stats = get_data("/stats/game/advanced", {"year": year})
    
    if not stats:
        return {}
        
    df = pd.json_normalize(stats)
    
    # Sort by week to ensure the 'decay' calculation flows chronologically
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'])
        df = df.sort_values(['team', 'season', 'week'])
    
    metrics = [
        'offense.ppa', 'offense.successRate', 'offense.explosiveness',
        'defense.ppa', 'defense.successRate', 'defense.explosiveness'
    ]
    
    # Calculate EWMA for every team in one pandas operation
    # We DON'T shift here because we want the "Final" value entering the upcoming game
    print("   -> Calculating math decay for all 130+ teams...")
    lookup = {}
    
    grouped = df.groupby('team')
    
    for team, group in grouped:
        team_momentum = {}
        for m in metrics:
            if m in group.columns:
                # Calculate EWMA (Span=3) and take the very last value (Current Form)
                val = group[m].ewm(span=3, adjust=False).mean().iloc[-1]
                team_momentum[f"decay_{m}"] = val
            else:
                team_momentum[f"decay_{m}"] = 0.0
        lookup[team] = team_momentum
        
    return lookup

def main():
    print("--- 🏈 CFB PREDICTOR (OPTIMIZED SPEED) 🏈 ---")
    
    # SETTINGS
    YEAR = 2025
    SEASON_TYPE = "postseason"
    WEEK = 1
    
    # 1. Load Models
    print("1. Loading Models...")
    try:
        model_win = joblib.load("model_winner.pkl")
        model_spread = joblib.load("model_spread_tuned.pkl")
        model_total = joblib.load("model_total.pkl")
    except FileNotFoundError:
        print("Error: Models not found. Run model.py first!")
        return
    
    # 2. Fetch Data Layers (Batch)
    print(f"2. Fetching Data for {YEAR}...")
    games_data = get_data("/games", {"year": YEAR, "seasonType": SEASON_TYPE, "week": WEEK})
    lines_data = get_data("/lines", {"year": YEAR, "seasonType": SEASON_TYPE, "week": WEEK})
    srs_data = get_data("/ratings/srs", {"year": YEAR})
    talent_data = get_data("/talent", {"year": YEAR})
    
    # 3. Build Fast Lookups
    print("3. Building Data Lookups...")
    lines_map = {x['id']: x['lines'][0] for x in lines_data if x.get('lines')}
    srs_map = {x['team']: x['rating'] for x in srs_data}
    talent_map = {x.get('school', x.get('team')): x['talent'] for x in talent_data}
    
    # THE SPEED UP: Build the Momentum Lookup once
    decay_map = build_decay_lookup(YEAR)

    predictions = []
    games = pd.DataFrame(games_data).rename(columns={'homeTeam': 'home_team', 'awayTeam': 'away_team'}).to_dict('records')
    
    print(f"\nAnalyzing {len(games)} matchups...")

    for g in games:
        if g.get('completed'): continue
        game_id = g.get('id')
        home, away = g.get('home_team'), g.get('away_team')
        
        line = lines_map.get(game_id)
        if not line: continue
        
        # Fast Lookup (No API calls inside loop!)
        h_decay = decay_map.get(home)
        a_decay = decay_map.get(away)
        
        # Skip if missing stats
        if not h_decay or not a_decay: continue

        # BUILD ROW
        row = {
            'spread': line.get('spread'),
            'overUnder': line.get('overUnder'),
            
            # CONTEXT
            'home_talent_score': talent_map.get(home, 10.0), 
            'away_talent_score': talent_map.get(away, 10.0),
            'home_srs_rating': srs_map.get(home, -5.0), 
            'away_srs_rating': srs_map.get(away, -5.0),
            
            # DECAY (Smart Memory)
            **{f"home_{k}":v for k,v in h_decay.items()},
            **{f"away_{k}":v for k,v in a_decay.items()}
        }
        
        # Ensure column order
        feat_cols = model_spread.feature_names_in_
        features = pd.DataFrame([row])[feat_cols]

        # PREDICT
        cover_prob = model_spread.predict_proba(features)[0][1]
        win_prob = model_win.predict_proba(features)[0][1]
        over_prob = model_total.predict_proba(features)[0][1]
        
        pred_spread = home if cover_prob > 0.5 else away
        conf_spread = max(cover_prob, 1-cover_prob)
        
        pred_total = "OVER" if over_prob > 0.5 else "UNDER"
        conf_total = max(over_prob, 1-over_prob)

        predictions.append({
            "Game": f"{away} @ {home}",
            "Spread": line.get('spread'),
            "Pick": f"{pred_spread} ({line.get('spread')})",
            "Conf": f"{conf_spread:.1%}",
            "Winner": f"{home if win_prob > 0.5 else away} ({max(win_prob, 1-win_prob):.0%})",
            "Total": f"{pred_total} {line.get('overUnder')} ({conf_total:.1%})"
        })

    # OUTPUT
    if predictions:
        df = pd.DataFrame(predictions).sort_values("Conf", ascending=False)
        print("\n--- 🎯 LIVE MODEL PREDICTIONS ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
        df.to_csv("live_predictions.csv", index=False)
    else:
        print("No valid games found.")

if __name__ == "__main__":
    main()