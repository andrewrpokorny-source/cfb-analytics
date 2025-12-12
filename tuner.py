import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def tune_spread_model():
    print("--- 🔧 STARTING HYPERPARAMETER TUNING 🔧 ---")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv("cfb_training_data_24_25.csv")
    
    features = [
        'spread', 'overUnder',
        'home_offense.ppa', 'home_offense.successRate', 'home_offense.explosiveness',
        'home_defense.ppa', 'home_defense.successRate', 'home_defense.explosiveness',
        'away_offense.ppa', 'away_offense.successRate', 'away_offense.explosiveness',
        'away_defense.ppa', 'away_defense.successRate', 'away_defense.explosiveness'
    ]
    
    # Filter valid data
    df_clean = df.dropna(subset=features + ['target_home_cover']).copy()
    X = df_clean[features]
    y = df_clean['target_home_cover']
    
    # 2. Split Data (Keep the test set separate so we don't cheat!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Define the "Grid" of possibilities
    # The machine will try EVERY combination of these settings.
    param_grid = {
        'n_estimators': [50, 100, 200, 300],        # Number of trees
        'max_depth': [3, 5, 7, 10],                 # How deep each tree can go (lower = less overfitting)
        'min_samples_split': [2, 5, 10],            # Minimum data points to create a split
        'min_samples_leaf': [1, 2, 4]               # Minimum data points at the end of a branch
    }
    
    # 4. Setup the Grid Search
    # cv=5 means "5-Fold Cross Validation". It trains 5 times on different chunks of data 
    # to ensure the score isn't just luck.
    rf = RandomForestClassifier(random_state=42)
    
    print(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])} different model combinations...")
    print("This might take a minute...")
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    
    grid_search.fit(X_train, y_train)
    
    # 5. The Results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("\n--- ✅ TUNING COMPLETE ---")
    print(f"Best Training Accuracy (CV): {best_score:.1%}")
    print("Best Parameters found:")
    print(best_params)
    
    # 6. Validate on the Test Set (The Moment of Truth)
    best_model = grid_search.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    
    print(f"\nTest Set Accuracy (Unseen Data): {test_acc:.1%}")
    
    # 7. Save the Tuned Model
    joblib.dump(best_model, "model_spread_tuned.pkl")
    print("Saved optimized model to 'model_spread_tuned.pkl'")

if __name__ == "__main__":
    tune_spread_model()