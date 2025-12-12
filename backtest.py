import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def run_backtest():
    print("--- 💰 RUNNING PROFIT SIMULATION 💰 ---")
    
    # 1. Load Data (Same as model.py)
    df = pd.read_csv("cfb_training_data_24_25.csv")
    
    features = [
        'spread', 'overUnder',
        'home_offense.ppa', 'home_offense.successRate', 'home_offense.explosiveness',
        'home_defense.ppa', 'home_defense.successRate', 'home_defense.explosiveness',
        'away_offense.ppa', 'away_offense.successRate', 'away_offense.explosiveness',
        'away_defense.ppa', 'away_defense.successRate', 'away_defense.explosiveness'
    ]
    
    df_clean = df.dropna(subset=features + ['target_home_cover']).copy()
    X = df_clean[features]
    y = df_clean['target_home_cover']
    
    # 2. Re-create the Test Split (CRITICAL: Must use same random_state=42)
    # This ensures we are testing on data the model has NEVER seen.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Load the Spread Model
    model = joblib.load("model_spread.pkl")
    
    # 4. Get Probabilities
    probs = model.predict_proba(X_test)[:, 1] # Probability of Home Cover
    
    # Create a Results DataFrame
    results = X_test.copy()
    results['actual_result'] = y_test
    results['model_prob'] = probs
    
    # 5. SIMULATE BETTING STRATEGY
    # Strategy: Only bet if model confidence is > 55%
    CONFIDENCE_THRESHOLD = 0.55
    BET_SIZE = 100 # Dollar amount per bet
    ODDS = -110 # Standard Vegas odds (implied prob 52.38%)
    
    # Logic: 
    # If prob > 0.55 -> Bet Home
    # If prob < 0.45 -> Bet Away (because Away prob is > 0.55)
    
    bets = []
    balance = 0
    balance_history = [0]
    wins = 0
    losses = 0
    
    print(f"\nSimulating bets on {len(results)} unseen games...")
    print(f"Strategy: Bet ${BET_SIZE} when confidence > {CONFIDENCE_THRESHOLD:.0%}")
    
    for idx, row in results.iterrows():
        prob_home = row['model_prob']
        actual_home_cover = row['actual_result']
        
        bet_placed = False
        won_bet = False
        
        # Bet Home?
        if prob_home >= CONFIDENCE_THRESHOLD:
            bet_placed = True
            if actual_home_cover == 1:
                profit = BET_SIZE * (100/110) # Win $90.90 on $100 bet
                wins += 1
                won_bet = True
            else:
                profit = -BET_SIZE
                losses += 1
                
        # Bet Away? (Home Prob <= 1 - Threshold)
        elif prob_home <= (1 - CONFIDENCE_THRESHOLD):
            bet_placed = True
            # If Home didn't cover (0), then Away covered (1)
            if actual_home_cover == 0:
                profit = BET_SIZE * (100/110)
                wins += 1
                won_bet = True
            else:
                profit = -BET_SIZE
                losses += 1
                
        if bet_placed:
            balance += profit
            balance_history.append(balance)
            
    # 6. REPORT CARD
    total_bets = wins + losses
    if total_bets > 0:
        win_rate = wins / total_bets
        roi = (balance / (total_bets * BET_SIZE)) * 100
        
        print("\n--- 📊 PERFORMANCE REPORT ---")
        print(f"Total Bets Placed: {total_bets}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Win Rate: {win_rate:.1%} (Target: >52.4%)")
        print(f"Net Profit: ${balance:.2f}")
        print(f"ROI: {roi:.1f}%")
        
        if balance > 0:
            print("\n✅ STATUS: PROFITABLE MODEL")
        else:
            print("\n❌ STATUS: NEEDS TUNING")
    else:
        print("No bets met the confidence threshold.")

if __name__ == "__main__":
    run_backtest()