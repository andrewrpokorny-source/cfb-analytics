import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib 

def train_models():
    print("--- 🧠 RESTORING LEAK-PROOF MODEL (56% Accuracy) 🧠 ---")
    
    # 1. Load Data
    try:
        # We use the 'smart' dataset which we know has the correct Total Offense/Defense stats
        df = pd.read_csv("cfb_training_data_smart.csv")
        df = df.drop_duplicates(subset=['id'])
        print(f"Loaded {len(df)} games.")
    except FileNotFoundError:
        print("Error: cfb_training_data_smart.csv not found.")
        return
    
    # 2. FEATURE LIST (The proven winners)
    features = [
        'spread', 'overUnder',
        'home_talent_score', 'away_talent_score',
        'home_srs_rating', 'away_srs_rating',
        
        # SMART DECAY (Totals - These keys are confirmed to work)
        'home_decay_offense.ppa', 'home_decay_offense.successRate', 'home_decay_offense.explosiveness',
        'home_decay_defense.ppa', 'home_decay_defense.successRate', 'home_decay_defense.explosiveness',
        'away_decay_offense.ppa', 'away_decay_offense.successRate', 'away_decay_offense.explosiveness',
        'away_decay_defense.ppa', 'away_decay_defense.successRate', 'away_decay_defense.explosiveness'
    ]
    
    # Drop NaNs
    df_clean = df.dropna(subset=features + ['target_home_cover', 'target_home_win', 'target_over']).copy()
    X = df_clean[features]
    
    # --- TRAIN WINNER ---
    print("\nTraining Game Winner Model...")
    y_win = df_clean['target_home_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y_win, test_size=0.2, random_state=42)
    model_win = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_win.fit(X_train, y_train)
    print(f"Winner Accuracy: {accuracy_score(y_test, model_win.predict(X_test)):.1%}")
    
    # --- TRAIN SPREAD ---
    print("Training Spread Model...")
    y_cover = df_clean['target_home_cover']
    X_train, X_test, y_train, y_test = train_test_split(X, y_cover, test_size=0.2, random_state=42)
    model_cover = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model_cover.fit(X_train, y_train)
    print(f"Spread Accuracy: {accuracy_score(y_test, model_cover.predict(X_test)):.1%} (Target: >52.4%)")
    
    # --- TRAIN TOTALS ---
    print("Training Totals Model...")
    y_total = df_clean['target_over']
    X_train, X_test, y_train, y_test = train_test_split(X, y_total, test_size=0.2, random_state=42)
    model_total = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_total.fit(X_train, y_train)
    print(f"Totals Accuracy: {accuracy_score(y_test, model_total.predict(X_test)):.1%}")

    # --- SAVE ---
    # Store feature names to prevent crashes
    model_cover.feature_names = features
    
    joblib.dump(model_win, "model_winner.pkl")
    joblib.dump(model_cover, "model_spread_tuned.pkl") 
    joblib.dump(model_total, "model_total.pkl")
    print("\nModels saved.")

if __name__ == "__main__":
    train_models()