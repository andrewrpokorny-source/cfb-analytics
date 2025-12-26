import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")

def main():
    print("--- 🕵️ API DIAGNOSTIC (POSTSEASON 2025) ---")
    if not API_KEY:
        print("❌ Error: No API Key found in .env")
        return

    HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    
    # Check specifically for the games in question
    url = "https://api.collegefootballdata.com/games"
    params = {"year": 2025, "seasonType": "postseason"}
    
    try:
        print("   -> Pinging API for game data...")
        res = requests.get(url, headers=HEADERS, params=params)
        
        if res.status_code != 200:
            print(f"❌ API Error: {res.status_code}")
            return
            
        games = res.json()
        
        # Filter for Dec 20-26
        target_games = [g for g in games if "2025-12-2" in g.get('start_date', '')]
        
        print(f"\n--- INSPECTING {len(target_games)} RECENT GAMES ---")
        
        if not target_games:
            print("   (No games found in this date range)")
            
        for g in target_games:
            home = g.get('home_team')
            away = g.get('away_team')
            h_pts = g.get('home_points')
            a_pts = g.get('away_points')
            
            print(f"\n🏈 {away} @ {home}")
            print(f"   Score:  {a_pts} - {h_pts}")
            
            if h_pts is None:
                print("   ⚠️ VERDICT: MISSING DATA (Score is Null)")
            else:
                print("   ✅ VERDICT: DATA EXISTS")

    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    main()
