import os
from api import fetch_with_retry
from dotenv import load_dotenv
import json

load_dotenv()

def inspect():
    print("Fetching lines for 2024 Regular Season Week 10...")
    # NOTE: Using 2024 because 2025 Regular Season might not have lines if it's early 2026? 
    # Wait, today is Jan 1 2026. The 2025 Regular Season just finished.
    # So 2025 Week 10 should have data.
    lines = fetch_with_retry("/lines", {"year": 2025, "seasonType": "regular", "week": 10})
    
    if not lines:
        print("No lines found.")
        return

    all_providers = set()

    for game in lines:
        for line in game.get('lines', []):
            all_providers.add(line.get('provider'))

    print(f"\nAll Available Providers found: {sorted(list(all_providers))}")

if __name__ == "__main__":
    inspect()
