import os
import requests
import pandas as pd
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    print("❌ Error: CFBD_API_KEY not found in .env")
    exit()

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

def main():
    print("--- 🕵️ LINE SHOPPING AUDIT ---")
    
    # 2. Fetch ALL raw line data
    print("Fetching raw betting data from API...")
    url = "https://api.collegefootballdata.com/lines"
    params = {"year": 2025, "seasonType": "postseason"} 
    
    try:
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"API Failed: {e}")
        return

    # 3. Analyze what we found
    all_providers = set()
    games_with_multiple_books = 0
    total_lines_seen = 0
    
    print(f"\nScanning {len(data)} games for betting lines...")
    
    for game in data:
        lines = game.get('lines', [])
        if not lines: continue
        
        providers = [l['provider'] for l in lines]
        all_providers.update(providers)
        total_lines_seen += len(lines)
        
        if len(providers) > 1:
            games_with_multiple_books += 1
            # Print a sample of a multi-book game to prove it exists
            if games_with_multiple_books == 1:
                print(f"\n✅ PROOF OF LIFE: Found multiple books for {game['homeTeam']} vs {game['awayTeam']}")
                for l in lines:
                    print(f"   - {l['provider']}: Spread {l.get('spread')} | Total {l.get('overUnder')}")

    # 4. The Verdict
    print("\n" + "="*40)
    print("📢 AUDIT RESULTS")
    print("="*40)
    print(f"Total Lines Scanned: {total_lines_seen}")
    print(f"Games with >1 Book:  {games_with_multiple_books}")
    print(f"Unique Books Found:  {sorted(list(all_providers))}")
    print("="*40)
    
    # 5. Check against your Filter
    YOUR_VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet']
    print("\n🔍 MISSING BOOKS (Found in API but blocked by your filter):")
    blocked = [p for p in all_providers if p not in YOUR_VALID_BOOKS]
    if blocked:
        for b in blocked:
            print(f"   ❌ {b}")
    else:
        print("   (None - Your filter matches everything the API sees)")

if __name__ == "__main__":
    main()