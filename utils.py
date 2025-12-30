import pandas as pd

def normalize_game_columns(df):
    """
    Helper to standardize column names (handling snake_case vs camelCase).
    """
    # Map common camelCase variations to snake_case
    rename_map = {
        'homeTeam': 'home_team',
        'awayTeam': 'away_team',
        'homePoints': 'home_points',
        'awayPoints': 'away_points',
        'homeScore': 'home_points', # Sometimes it's called score
        'awayScore': 'away_points'
    }
    df = df.rename(columns=rename_map)
    return df
