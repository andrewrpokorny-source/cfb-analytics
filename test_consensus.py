import statistics

def test_consensus():
    print("Testing Consensus Logic...")
    
    # Sample lines from different providers (simulated)
    # Scenario 1: Oregon -2.5 (User's example where market is moving)
    # Let's say:
    # DK: -2.5
    # FD: -1.5
    # Bovada: -1.5
    # ESPN Bet: -1.0
    # MGM: -1.5
    
    game_lines = [
        {'provider': 'DraftKings', 'spread': -2.5, 'overUnder': 55.0},
        {'provider': 'FanDuel', 'spread': -1.5, 'overUnder': 55.5},
        {'provider': 'Bovada', 'spread': -1.5, 'overUnder': 54.5},
        {'provider': 'ESPN Bet', 'spread': -1.0, 'overUnder': 55.0},
        {'provider': 'BetMGM', 'spread': -1.5, 'overUnder': 55.0},
    ]
    
    spreads = [l.get('spread') for l in game_lines if l.get('spread') is not None]
    totals = [l.get('overUnder') for l in game_lines if l.get('overUnder') is not None]
    
    print(f"Spreads: {spreads}")
    
    median_spread = statistics.median(spreads)
    print(f"Median Spread: {median_spread}")
    
    assert median_spread == -1.5, f"Expected -1.5, got {median_spread}"
    
    print(f"Totals: {totals}")
    median_total = statistics.median(totals)
    print(f"Median Total: {median_total}")
    
    assert median_total == 55.0, f"Expected 55.0, got {median_total}"
    
    print("✅ Consensus Logic Verified.")

if __name__ == "__main__":
    test_consensus()
