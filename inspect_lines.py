import os
from api import fetch_with_retry
from dotenv import load_dotenv
import json
import statistics

load_dotenv()

def inspect():
    # Fetch 2025 Postseason to find Oregon
    print("Fetching lines for 2025 postseason...")
    lines = fetch_with_retry("/lines", {"year": 2025, "seasonType": "postseason", "week": 1})
    
    if not lines:
        print("No lines found.")
        return

    found = False
    for game in lines:
        home = game.get('homeTeam')
        away = game.get('awayTeam')
        if 'Oregon' in home or 'Oregon' in away:
            found = True
            print(f"\nFOUND GAME: {away} @ {home} (ID: {game['id']})")
            print("-" * 50)
            
            raw_lines = game.get('lines', [])
            valid_spreads = []
            
            for l in raw_lines:
                prov = l.get('provider')
                spread = l.get('spread')
                print(f"Provider: {prov:15} | Spread: {spread}")
                if spread is not None:
                    valid_spreads.append(spread)
            
            if valid_spreads:
                med = statistics.median(valid_spreads)
                print("-" * 50)
                print(f"CALCULATED MEDIAN: {med}")
            else:
                print("No spreads available.")
    
    if not found:
        print("Oregon game not found in 2025 Postseason Week 1.")

if __name__ == "__main__":
    inspect()
