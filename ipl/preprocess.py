# preprocess.py
import pandas as pd

def preprocess_data(deliveries_file="deliveries.csv", matches_file="matches.csv"):
    # Load datasets
    try:
        deliveries_df = pd.read_csv(deliveries_file)
        matches_df = pd.read_csv(matches_file)
    except FileNotFoundError as e:
        print(f"Error: One of the files not found - {e}")
        return None

    # Rename 'id' in matches_df to 'match_id' for consistency
    matches_df.rename(columns={"id": "match_id"}, inplace=True)

    # Standardize 'season' column in matches_df to numeric (extract first year from formats like "2007/08")
    def standardize_season(season):
        if isinstance(season, str) and "/" in season:
            return int(season.split("/")[0])  # Take the first year (e.g., 2007 from "2007/08")
        try:
            return int(season)  # Direct integer seasons (e.g., 2023)
        except (ValueError, TypeError):
            return None  # Handle invalid entries

    matches_df["season"] = matches_df["season"].apply(standardize_season)

    # Merge datasets on 'match_id'
    merged_df = pd.merge(deliveries_df, matches_df, on="match_id", how="left")

    # Handle missing values
    merged_df.fillna({"winner": "No Result", "result": "No Result", "result_margin": 0, "season": merged_df["season"].median()}, inplace=True)

    # Use 'date' from matches.csv directly (no _x or _y since deliveries.csv has no date)
    merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")

    # Save preprocessed data
    merged_df.to_csv("preprocessed_ipl_data.csv", index=False)
    print("Preprocessed data saved as 'preprocessed_ipl_data.csv'")
    return merged_df

if __name__ == "__main__":
    preprocess_data("deliveries.csv", "matches.csv")