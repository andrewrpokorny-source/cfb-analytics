import requests
import time
from config import HEADERS

BASE_URL = "https://api.collegefootballdata.com"

def fetch_with_retry(endpoint, params):
    """
    Fetch data from CFBD API with rate limiting and retries.
    Used for batch processes where we want to be robust.
    """
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(1, 4): 
        try:
            res = requests.get(url, headers=HEADERS, params=params)
            if res.status_code == 200: 
                return res.json()
            elif res.status_code == 429: 
                print(f"      ⚠️ Rate limit hit. Sleeping {10 * attempt}s...")
                time.sleep(10 * attempt)
            else:
                 # For other errors, we might want to just continue retry or log
                 pass
        except Exception as e: 
            time.sleep(5)
    return []

def get_data(endpoint, params):
    """
    Simpler wrapper for API calls, often used where we want to print status.
    Maintains compatibility with previous simpler implementations.
    """
    url = f"{BASE_URL}{endpoint}"
    try:
        # Optional: verify if we want to print here or leave it to the caller.
        # The original code printed "Fetching..." often.
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        time.sleep(0.5) 
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []
