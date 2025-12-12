import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_equity_curve():
    # 1. Setup Data (Same as backtest)
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
    
    # 2. Test Split (Must be same random_state to match your backtest!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load("model_spread.pkl")
    probs = model.predict_proba(X_test)[:, 1]
    
    # 3. Re-Calculate Bets
    results = X_test.copy()
    results['actual_result'] = y_test
    results['model_prob'] = probs
    
    balance = 0
    balance_history = [0]
    
    CONFIDENCE_THRESHOLD = 0.55
    BET_SIZE = 100
    
    print("Re-calculating betting history for plot...")
    
    for idx, row in results.iterrows():
        prob_home = row['model_prob']
        actual_home_cover = row['actual_result']
        profit = 0
        
        # Bet Home
        if prob_home >= CONFIDENCE_THRESHOLD:
            if actual_home_cover == 1:
                profit = BET_SIZE * (100/110) 
            else:
                profit = -BET_SIZE
        # Bet Away
        elif prob_home <= (1 - CONFIDENCE_THRESHOLD):
            if actual_home_cover == 0:
                profit = BET_SIZE * (100/110)
            else:
                profit = -BET_SIZE
        
        # Only record if a bet was actually made
        if profit != 0:
            balance += profit
            balance_history.append(balance)
            
    # 4. PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(balance_history, color='green', linewidth=2, label='Strategy Profit')
    
    # Add a zero line
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.title(f"Bankroll Growth (55% Threshold)\nFinal Profit: ${balance:.2f} | Win Rate: 56.9%", fontsize=14)
    plt.xlabel("Number of Bets Placed", fontsize=12)
    plt.ylabel("Profit ($)", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    
    print(f"Final Balance: ${balance:.2f}")
    plt.show()

if __name__ == "__main__":
    plot_equity_curve()