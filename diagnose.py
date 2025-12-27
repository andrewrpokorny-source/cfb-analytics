import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

def main():
    print("--- 🕵️ API DIAGNOSTIC SCAN ---")
    
    # Check 1: Credentials
    if not API_KEY:
        print("❌ Error: No API Key found in .env")
        return
    print("✅ API Key loaded.")

    # Check 2: Scan for COMPLETED games in 2025
    print("\nScanning 2025 Schedule for completed games...")
    found_any = False
    
    # Scan Regular Season Weeks 1-16
    for week in range(1, 17):
        res = requests.get("https://api.collegefootballdata.com/games", 
                           headers=HEADERS, 
                           params={"year": 2025, "seasonType": "regular", "week": week})
        
        if res.status_code == 200:
            games = res.json()
            completed = [g for g in games if g.get('completed')]
            if completed:
                print(f"   -> Regular Week {week}: Found {len(completed)} COMPLETED games.")
                found_any = True
                # Print sample
                print(f"      (Sample: {completed[0]['away_team']} vs {completed[0]['home_team']} - Score: {completed[0]['away_points']}-{completed[0]['home_points']})")

    # Scan Postseason
    res = requests.get("https://api.collegefootballdata.com/games", 
                       headers=HEADERS, 
                       params={"year": 2025, "seasonType": "postseason"})
    if res.status_code == 200:
        games = res.json()
        completed = [g for g in games if g.get('completed')]
        if completed:
            print(f"   -> Postseason: Found {len(completed)} COMPLETED games.")
            found_any = True
            print(f"      (Sample: {completed[0]['away_team']} vs {completed[0]['home_team']})")

    if not found_any:
        print("\n❌ CRITICAL: The API returned 0 completed games for 2025.")
        print("   This means your backfill script has nothing to download.")
        print("   Are you sure the simulation has advanced to the end of the season?")
    else:
        print("\n✅ DIAGNOSIS COMPLETE: Use the week numbers found above in your backfill.py script.")

if __name__ == "__main__":
    main()