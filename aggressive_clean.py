import pandas as pd
import os

HISTORY_FILE = "live_predictions.csv"

def main():
    print("--- ☢️ AGGRESSIVE CLEANUP (MATCHUP BASED) ---")
    
    if not os.path.exists(HISTORY_FILE):
        print("❌ No file found.")
        return

    # 1. Load
    df = pd.read_csv(HISTORY_FILE)
    original_count = len(df)
    
    # 2. THE NUCLEAR FIX:
    # Instead of looking at IDs, we look at the actual TEAMS.
    # If "Western Kentucky" plays "Sam Houston State" twice, delete the older one.
    # We keep='last' to preserve the most recent (likely official API) update.
    df_clean = df.drop_duplicates(subset=['HomeTeam', 'AwayTeam'], keep='last')
    
    # 3. Save
    df_clean.to_csv(HISTORY_FILE, index=False)
    
    final_count = len(df_clean)
    removed = original_count - final_count
    
    print(f"✅ Scanned {original_count} rows.")
    print(f"🗑️ Removed {removed} duplicates based on Team Matchups.")
    print(f"💾 Saved clean database.")

if __name__ == "__main__":
    main()
