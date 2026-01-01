import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Check your .env file.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# Constants
VALID_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 'BetRivers', 'Unibet', 'Bovada', 'ESPN Bet']
HISTORY_FILE = "live_predictions.csv"
HISTORY_CUTOFF = "2025-12-01"
