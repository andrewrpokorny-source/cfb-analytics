# Create the file with the code inside
cat << 'EOF' > clean_duplicates.py
import pandas as pd
import os

HISTORY_FILE = "live_predictions.csv"

def main():
    print("--- 🧹 CLEANING DUPLICATE ROWS ---")
    
    if not os.path.exists(HISTORY_FILE):
        print("❌ No file found to clean.")
        return

    # 1. Load Data
    df = pd.read_csv(HISTORY_FILE)
    original_count = len(df)
    
    # 2. STANDARDIZE GameID
    # Force everything to String, remove decimals like '.0'
    df['GameID'] = df['GameID'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # 3. Drop Duplicates
    # We keep the 'first' occurrence (usually the newest)
    df_clean = df.drop_duplicates(subset=['GameID'], keep='first')
    
    final_count = len(df_clean)
    removed = original_count - final_count
    
    # 4. Save
    df_clean.to_csv(HISTORY_FILE, index=False)
    
    print(f"✅ Scanning {original_count} rows...")
    print(f"🗑️ Removed {removed} duplicates.")
    print(f"💾 Saved clean file with {final_count} unique games.")

if __name__ == "__main__":
    main()
EOF
