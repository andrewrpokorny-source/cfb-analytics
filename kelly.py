import pandas as pd
import os

def calculate_kelly():
    print("--- 💰 KELLY CRITERION BET SIZER 💰 ---")
    
    # 1. Load Predictions
    if not os.path.exists("live_predictions.csv"):
        print("Error: live_predictions.csv not found. Run predict.py first!")
        return
        
    df = pd.read_csv("live_predictions.csv")
    
    # 2. Settings
    BANKROLL = 1000  # Example Bankroll ($1,000)
    KELLY_FRACTION = 0.25 # "Quarter Kelly" is industry standard for safety
    
    print(f"Bankroll: ${BANKROLL}")
    print(f"Strategy: Quarter Kelly (Aggressiveness: {KELLY_FRACTION*100}%)")
    print("-" * 60)
    print(f"{'GAME':<30} | {'PICK':<20} | {'CONF':<6} | {'WAGER':<8}")
    print("-" * 60)
    
    total_wagered = 0
    
    for index, row in df.iterrows():
        # Clean string "58.4%" -> float 0.584
        try:
            win_prob = float(row['Conf'].strip('%')) / 100.0
        except:
            continue
            
        # FILTER: Only bet if edge is significant (>53%)
        if win_prob <= 0.53:
            continue
            
        # KELLY FORMULA
        # f* = (bp - q) / b
        # b = odds (1 to 1 for spread, so b=0.91 if -110 odds)
        # We assume standard -110 juice (b = 0.9091)
        b = 0.9091
        p = win_prob
        q = 1 - p
        
        f_star = (b * p - q) / b
        
        # Apply Safety Fraction (Quarter Kelly)
        fraction_to_bet = f_star * KELLY_FRACTION
        
        # Calculate Dollar Amount
        wager = BANKROLL * fraction_to_bet
        
        if wager > 0:
            total_wagered += wager
            print(f"{row['Game']:<30} | {row['Pick']:<20} | {row['Conf']:<6} | ${wager:.2f}")

    print("-" * 60)
    print(f"TOTAL EXPOSURE: ${total_wagered:.2f}")
    print(f"% OF BANKROLL: {(total_wagered/BANKROLL)*100:.1f}%")

if __name__ == "__main__":
    calculate_kelly()