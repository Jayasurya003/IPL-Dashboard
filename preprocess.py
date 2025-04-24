import pandas as pd
import numpy as np

def preprocess_it20():
    df = pd.read_csv("It20_ball_by_ball.csv")

    # Rename columns to standard format
    df.columns = df.columns.str.strip().str.lower()

    # Extract and format basic fields
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['season'] = df['date'].dt.year
    df.rename(columns={
        'match id': 'match_id',
        'bat first': 'team1',
        'bat second': 'team2',
        'batter': 'batter',
        'bowler': 'bowler',
        'batter runs': 'batsman_runs',
        'wicket': 'is_wicket',
        'venue': 'venue',
        'winner': 'winner',
        'target score': 'target_runs',
        'player out': 'player_out'
    }, inplace=True)

    df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)

    # Generate match-level summary
    match_level = df.groupby("match_id").agg({
        "season": "first",
        "date": "first",
        "venue": "first",
        "team1": "first",
        "team2": "first",
        "winner": "first",
        "target_runs": "first"
    }).reset_index()

    # Save files
    match_level.to_csv("matches_it20.csv", index=False)
    df.to_csv("deliveries_it20.csv", index=False)

    # Merge for dashboard use
    merged = df.merge(
        match_level[["match_id", "season", "winner", "venue", "team1", "team2"]],
        on="match_id", how="left"
    )
    merged.to_csv("preprocessed_it20_data.csv", index=False)

    print("✅ Preprocessing complete!")
    print("- matches_it20.csv")
    print("- deliveries_it20.csv")
    print("- preprocessed_it20_data.csv")

if __name__ == "__main__":
    preprocess_it20()
