import pandas as pd
import os

HISTORY_FILE = "live_predictions.csv"

# MAPPING: Nickname -> Official Name
TEAM_MAP = {
    "USF": "South Florida",
    "Ole Miss": "Mississippi",
    "LSU": "Louisiana State",
    "UConn": "Connecticut",
    "UMass": "Massachusetts",
    "Southern Miss": "Southern Mississippi",
    "UL Monroe": "Louisiana Monroe",
    "UL Lafayette": "Louisiana"
}

def main():
    print("--- 🧠 SMART NORMALIZATION & CLEANUP ---")
    
    if not os.path.exists(HISTORY_FILE):
        print("❌ No file found.")
        return

    # 1. Load Data
    df = pd.read_csv(HISTORY_FILE)
    original_count = len(df)
    
    # 2. NORMALIZE TEAM NAMES
    # This converts "USF" -> "South Florida" in both columns
    print("   -> Normalizing team names...")
    df['HomeTeam'] = df['HomeTeam'].replace(TEAM_MAP)
    df['AwayTeam'] = df['AwayTeam'].replace(TEAM_MAP)
    
    # 3. DEDUPLICATE (The Nuclear Option)
    # Now that "USF" matches "South Florida", this will catch the duplicate
    df_clean = df.drop_duplicates(subset=['HomeTeam', 'AwayTeam'], keep='last')
    
    # 4. Save
    df_clean.to_csv(HISTORY_FILE, index=False)
    
    final_count = len(df_clean)
    removed = original_count - final_count
    
    print(f"✅ Scanned {original_count} rows.")
    print(f"   (Fixed {removed} duplicates caused by naming mismatches)")
    print(f"💾 Saved clean database.")

if __name__ == "__main__":
    main()
