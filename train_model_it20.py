import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# ===== Load and clean data =====
matches_df = pd.read_csv("matches_it20.csv")
deliveries_df = pd.read_csv("deliveries_it20.csv")

matches_df.columns = matches_df.columns.str.strip().str.lower()
deliveries_df.columns = deliveries_df.columns.str.strip().str.lower()
matches_df.rename(columns={'match id': 'match_id'}, inplace=True)
deliveries_df.rename(columns={'match id': 'match_id'}, inplace=True)
matches_df['match_id'] = matches_df['match_id'].astype(str)
deliveries_df['match_id'] = deliveries_df['match_id'].astype(str)

print("✅ Data loaded.")

# ===== Handle missing venue =====
if 'venue' not in matches_df.columns:
    if 'ground' in matches_df.columns:
        matches_df.rename(columns={'ground': 'venue'}, inplace=True)
    elif 'location' in matches_df.columns:
        matches_df.rename(columns={'location': 'venue'}, inplace=True)
    else:
        raise ValueError("❌ 'venue' column not found in matches_df.")

venue_stats = deliveries_df.merge(matches_df[['match_id', 'venue']], on='match_id', how='left')
if 'venue_y' in venue_stats.columns:
    venue_stats.rename(columns={'venue_y': 'venue'}, inplace=True)

# ===== Precompute Stats =====
print("📊 Precomputing stats...")

batsman_vs_bowler = deliveries_df.groupby(["batter", "bowler"]).agg({
    "batsman_runs": "sum", "ball": "count", "is_wicket": "sum"
}).reset_index()
batsman_vs_bowler["strike_rate"] = (batsman_vs_bowler["batsman_runs"] / batsman_vs_bowler["ball"] * 100).fillna(0)

batsman_venue = venue_stats.groupby(["batter", "venue"])["batsman_runs"].mean().reset_index(name="avg_runs_venue")
bowler_venue = venue_stats[venue_stats["is_wicket"] == 1].groupby(["bowler", "venue"]).size().reset_index(name="avg_wickets_venue")
global_batsman_avg = deliveries_df["batsman_runs"].mean()
global_bowler_avg = deliveries_df["is_wicket"].mean()

team_venue_wins = matches_df.groupby(["team1", "venue"]).apply(lambda x: (x["winner"] == x["team1"]).mean()).reset_index(name="team1_win_rate_venue")
team_venue_wins = pd.concat([
    team_venue_wins.rename(columns={"team1": "team", "team1_win_rate_venue": "win_rate_venue"}),
    matches_df.groupby(["team2", "venue"]).apply(lambda x: (x["winner"] == x["team2"]).mean()).reset_index(name="win_rate_venue").rename(columns={"team2": "team"})
]).groupby(["team", "venue"]).mean().reset_index()

team_h2h = matches_df.groupby(["team1", "team2"]).apply(lambda x: (x["winner"] == x["team1"]).mean()).reset_index(name="t1_win_rate_h2h")

if "toss_decision" not in matches_df.columns:
    matches_df["toss_decision"] = "field"

# Toss data missing in IT20 dataset
print("⚠️ 'toss_winner' column not found. Using default toss factor = 0.5")

# ===== Recent Form Function =====
def get_recent_form(player, match_id, is_bowler=False):
    player_data = deliveries_df[deliveries_df["match_id"] < match_id].sort_values("match_id", ascending=False)
    if is_bowler:
        recent = player_data[player_data["bowler"] == player].groupby("match_id")["is_wicket"].sum().head(5)
        return np.nanmean(np.minimum(recent, 5)) if not recent.empty else global_bowler_avg
    else:
        recent = player_data[player_data["batter"] == player].groupby("match_id")["batsman_runs"].sum().head(5)
        return np.nanmean(np.minimum(recent, 50)) if not recent.empty else global_batsman_avg

# ===== Feature Computation =====
def compute_team_features(t1_bat, t1_bowl, t2_bat, t2_bowl, venue, toss_decision, match_id, team1, team2):
    t1_vs_t2 = batsman_vs_bowler[(batsman_vs_bowler["batter"].isin(t1_bat)) & (batsman_vs_bowler["bowler"].isin(t2_bowl))]
    t2_vs_t1 = batsman_vs_bowler[(batsman_vs_bowler["batter"].isin(t2_bat)) & (batsman_vs_bowler["bowler"].isin(t1_bowl))]

    t1_runs_vs_t2 = np.nanmean(t1_vs_t2["batsman_runs"]) if not t1_vs_t2.empty else global_batsman_avg
    t1_sr_vs_t2 = np.nanmean(t1_vs_t2["strike_rate"]) if not t1_vs_t2.empty else 100
    t1_dismissals_vs_t2 = np.nanmean(t1_vs_t2["is_wicket"]) if not t1_vs_t2.empty else global_bowler_avg

    t2_runs_vs_t1 = np.nanmean(t2_vs_t1["batsman_runs"]) if not t2_vs_t1.empty else global_batsman_avg
    t2_sr_vs_t1 = np.nanmean(t2_vs_t1["strike_rate"]) if not t2_vs_t1.empty else 100
    t2_dismissals_vs_t1 = np.nanmean(t2_vs_t1["is_wicket"]) if not t2_vs_t1.empty else global_bowler_avg

    t1_batsmen_venue = batsman_venue[(batsman_venue["batter"].isin(t1_bat)) & (batsman_venue["venue"] == venue)]["avg_runs_venue"].mean()
    t2_batsmen_venue = batsman_venue[(batsman_venue["batter"].isin(t2_bat)) & (batsman_venue["venue"] == venue)]["avg_runs_venue"].mean()
    t1_bowlers_venue = bowler_venue[(bowler_venue["bowler"].isin(t1_bowl)) & (bowler_venue["venue"] == venue)]["avg_wickets_venue"].mean()
    t2_bowlers_venue = bowler_venue[(bowler_venue["bowler"].isin(t2_bowl)) & (bowler_venue["venue"] == venue)]["avg_wickets_venue"].mean()

    t1_batsmen_venue = t1_batsmen_venue if not np.isnan(t1_batsmen_venue) else global_batsman_avg
    t2_batsmen_venue = t2_batsmen_venue if not np.isnan(t2_batsmen_venue) else global_batsman_avg
    t1_bowlers_venue = t1_bowlers_venue if not np.isnan(t1_bowlers_venue) else global_bowler_avg
    t2_bowlers_venue = t2_bowlers_venue if not np.isnan(t2_bowlers_venue) else global_bowler_avg

    t1_team_venue_win = team_venue_wins[(team_venue_wins["team"] == team1) & (team_venue_wins["venue"] == venue)]["win_rate_venue"].mean()
    t2_team_venue_win = team_venue_wins[(team_venue_wins["team"] == team2) & (team_venue_wins["venue"] == venue)]["win_rate_venue"].mean()
    t1_team_venue_win = t1_team_venue_win if not np.isnan(t1_team_venue_win) else 0.5
    t2_team_venue_win = t2_team_venue_win if not np.isnan(t2_team_venue_win) else 0.5

    t1_batsmen_form = np.nanmean([get_recent_form(p, match_id) for p in t1_bat])
    t2_batsmen_form = np.nanmean([get_recent_form(p, match_id) for p in t2_bat])
    t1_bowlers_form = np.nanmean([get_recent_form(p, match_id, True) for p in t1_bowl])
    t2_bowlers_form = np.nanmean([get_recent_form(p, match_id, True) for p in t2_bowl])

    toss_factor = 0.5  # default (no toss_winner info)

    h2h = team_h2h[(team_h2h["team1"] == team1) & (team_h2h["team2"] == team2)]["t1_win_rate_h2h"].mean()
    h2h = h2h if not np.isnan(h2h) else 0.5

    return [
        t1_runs_vs_t2, t1_sr_vs_t2, t1_dismissals_vs_t2, t1_batsmen_venue, t1_bowlers_venue, t1_team_venue_win, t1_batsmen_form, t1_bowlers_form,
        t2_runs_vs_t1, t2_sr_vs_t1, t2_dismissals_vs_t1, t2_batsmen_venue, t2_bowlers_venue, t2_team_venue_win, t2_batsmen_form, t2_bowlers_form,
        toss_factor, h2h
    ]

# ===== Feature Generation =====
print("⚙️ Generating features (with progress bar)...")
X, y = [], []

for i, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Matches"):
    match_deliveries = deliveries_df[deliveries_df["match_id"] == row["match_id"]]
    t1_bat = match_deliveries["batter"].unique().tolist()[:6]
    t1_bowl = match_deliveries["bowler"].unique().tolist()[:5]
    t2_bat = match_deliveries[~match_deliveries["batter"].isin(t1_bat)]["batter"].unique().tolist()[:6]
    t2_bowl = match_deliveries[~match_deliveries["bowler"].isin(t1_bowl)]["bowler"].unique().tolist()[:5]
    features = compute_team_features(t1_bat, t1_bowl, t2_bat, t2_bowl, row["venue"], row["toss_decision"], row["match_id"], row["team1"], row["team2"])
    X.append(features)
    y.append(1 if row["winner"] == row["team1"] else 0)

# ===== Train Model =====
print("🧠 Training model...")
X_df = pd.DataFrame(X, columns=[
    "t1_runs_vs_t2", "t1_sr_vs_t2", "t1_dismissals_vs_t2", "t1_batsmen_venue", "t1_bowlers_venue", "t1_team_venue_win", "t1_batsmen_form", "t1_bowlers_form",
    "t2_runs_vs_t1", "t2_sr_vs_t1", "t2_dismissals_vs_t1", "t2_batsmen_venue", "t2_bowlers_venue", "t2_team_venue_win", "t2_batsmen_form", "t2_bowlers_form",
    "toss_factor", "t1_win_rate_h2h"
])
X_df["match_id"] = matches_df["match_id"]
X_df["winner"] = y
X_df.to_csv("cricket_features_it20.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(X_df.drop(columns=["match_id", "winner"]), y, test_size=0.2, random_state=42)
model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# ===== Evaluation =====
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("📈 Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")

with open("xgb_model_it20.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as xgb_model_it20.pkl")
