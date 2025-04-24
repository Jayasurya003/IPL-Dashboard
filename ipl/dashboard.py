import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score  # Fixed imports
import os
import pickle

from fpdf import FPDF
import tempfile
import base64

def generate_full_pdf_report(title, sections):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title(title)
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    pdf.ln(10)

    for section_title, section_lines in sections:
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(200, 10, txt=section_title, ln=True)
        pdf.set_font("Arial", size=11)
        for line in section_lines:
            pdf.multi_cell(0, 10, txt=line)
        pdf.ln(5)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)

    with open(temp_file.name, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f'data:application/octet-stream;base64,{encoded}'


print("Starting script...")

# Function to load data and preprocess if needed
def load_or_preprocess_data():
    preprocessed_file = "preprocessed_ipl_data.csv"
    if not os.path.exists(preprocessed_file):
        print("Preprocessed file not found. Generating preprocessed data...")
        from preprocess import preprocess_data
        preprocess_data()
    else:
        print(f"Preprocessed file '{preprocessed_file}' found. Skipping preprocessing...")
    
    print("Loading data...")
    matches_df = pd.read_csv("matches.csv")
    deliveries_df = pd.read_csv("deliveries.csv")
    matches_df.rename(columns={"id": "match_id"}, inplace=True)
    df = pd.read_csv(preprocessed_file, low_memory=False)
    return matches_df, deliveries_df, df

# Function to precompute all stats
def precompute_stats(deliveries_df, matches_df):
    print("Precomputing batsman vs bowler stats...")
    batsman_vs_bowler = deliveries_df.groupby(["batter", "bowler"]).agg({
        "batsman_runs": "sum", "ball": "count", "is_wicket": "sum"
    }).reset_index()
    batsman_vs_bowler["strike_rate"] = (batsman_vs_bowler["batsman_runs"] / batsman_vs_bowler["ball"] * 100).fillna(0)

    print("Precomputing venue stats...")
    venue_stats = deliveries_df.merge(matches_df[["match_id", "venue"]], on="match_id")
    batsman_venue = venue_stats.groupby(["batter", "venue"])["batsman_runs"].mean().reset_index(name="avg_runs_venue")
    bowler_venue = venue_stats[venue_stats["is_wicket"] == 1].groupby(["bowler", "venue"]).size().reset_index(name="avg_wickets_venue")
    global_batsman_avg = deliveries_df["batsman_runs"].mean()
    global_bowler_avg = deliveries_df["is_wicket"].mean()

    print("Precomputing team venue win rates...")
    team_venue_wins = matches_df.groupby(["team1", "venue"]).apply(lambda x: (x["winner"] == x["team1"]).mean()).reset_index(name="team1_win_rate_venue")
    team_venue_wins = pd.concat([
        team_venue_wins.rename(columns={"team1": "team", "team1_win_rate_venue": "win_rate_venue"}),
        matches_df.groupby(["team2", "venue"]).apply(lambda x: (x["winner"] == x["team2"]).mean()).reset_index(name="win_rate_venue").rename(columns={"team2": "team"})
    ]).groupby(["team", "venue"]).mean().reset_index()

    print("Precomputing team head-to-head stats...")
    team_h2h = matches_df.groupby(["team1", "team2"]).apply(lambda x: (x["winner"] == x["team1"]).mean()).reset_index(name="t1_win_rate_h2h")

    print("Precomputing toss impact...")
    toss_impact = matches_df.groupby(["venue", "toss_decision"]).apply(
        lambda x: (x["winner"] == x["toss_winner"]).mean()
    ).reset_index(name="toss_win_rate")

    return batsman_vs_bowler, batsman_venue, bowler_venue, team_venue_wins, team_h2h, toss_impact, global_batsman_avg, global_bowler_avg

# Function to train and save model, or load if it exists
def get_model(matches_df, deliveries_df, batsman_vs_bowler, batsman_venue, bowler_venue, team_venue_wins, team_h2h, toss_impact, global_batsman_avg, global_bowler_avg):
    model_file = "xgb_model.pkl"
    if not os.path.exists(model_file):
        print("Model file not found. Training model...")
        X = []
        y = []
        for i, row in matches_df.iterrows():
            if i % 100 == 0:
                print(f"Processing match {i}/{len(matches_df)}")
            match_deliveries = deliveries_df[deliveries_df["match_id"] == row["match_id"]]
            team1_batsmen = match_deliveries["batter"].unique().tolist()[:6]
            team1_bowlers = match_deliveries["bowler"].unique().tolist()[:5]
            team2_batsmen = match_deliveries[~match_deliveries["batter"].isin(team1_batsmen)]["batter"].unique().tolist()[:6]
            team2_bowlers = match_deliveries[~match_deliveries["bowler"].isin(team1_bowlers)]["bowler"].unique().tolist()[:5]
            features = compute_team_features(team1_batsmen, team1_bowlers, team2_batsmen, team2_bowlers, row["venue"], row["toss_decision"], row["match_id"], row["team1"], row["team2"])
            X.append(features)
            y.append(1 if row["winner"] == row["team1"] else 0)

        print("Building feature DataFrame...")
        X = pd.DataFrame(X, columns=[
            "t1_runs_vs_t2", "t1_sr_vs_t2", "t1_dismissals_vs_t2", "t1_batsmen_venue", "t1_bowlers_venue", "t1_team_venue_win", "t1_batsmen_form", "t1_bowlers_form",
            "t2_runs_vs_t1", "t2_sr_vs_t1", "t2_dismissals_vs_t1", "t2_batsmen_venue", "t2_bowlers_venue", "t2_team_venue_win", "t2_batsmen_form", "t2_bowlers_form",
            "toss_factor", "t1_win_rate_h2h"
        ]).fillna(0)
        y = np.array(y)

        print("Splitting data...")
        train_mask = matches_df["season"].apply(lambda x: int(str(x).split("/")[0]) if "/" in str(x) else int(x)) < 2023
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]

        print("Training model...")
        model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for the positive class (Team 1 wins)

        # Calculate and print all scoring metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print("Model Performance Metrics:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"ROC-AUC: {roc_auc:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

        # Save the trained model
        try:
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved to '{model_file}'")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print(f"Model file '{model_file}' found. Loading model...")
        with open(model_file, "rb") as f:
            model = pickle.load(f)
    
    return model

# Function to compute recent form (unchanged)
def get_recent_form(player, match_id, is_bowler=False):
    player_data = deliveries_df[deliveries_df["match_id"] < match_id].sort_values("match_id", ascending=False)
    if is_bowler:
        recent = player_data[player_data["bowler"] == player].groupby("match_id")["is_wicket"].sum().head(5)
        recent = np.minimum(recent, 5)  # Cap at 5 wickets
        return np.nanmean(recent) if not recent.empty else global_bowler_avg
    else:
        recent = player_data[player_data["batter"] == player].groupby("match_id")["batsman_runs"].sum().head(5)
        recent = np.minimum(recent, 50)  # Cap at 50 runs
        return np.nanmean(recent) if not recent.empty else global_batsman_avg

# Function to compute team features (unchanged)
def compute_team_features(team1_batsmen, team1_bowlers, team2_batsmen, team2_bowlers, venue, toss_decision, match_id=None, team1=None, team2=None):
    team1_players = team1_batsmen + team1_bowlers
    team2_players = team2_batsmen + team2_bowlers

    t1_vs_t2 = batsman_vs_bowler[
        (batsman_vs_bowler["batter"].isin(team1_batsmen)) & 
        (batsman_vs_bowler["bowler"].isin(team2_bowlers))
    ]
    t1_runs_vs_t2 = np.nanmean(t1_vs_t2["batsman_runs"]) if not t1_vs_t2.empty else global_batsman_avg
    t1_sr_vs_t2 = np.nanmean(t1_vs_t2["strike_rate"]) if not t1_vs_t2.empty else 100
    t1_dismissals_vs_t2 = np.nanmean(t1_vs_t2["is_wicket"]) if not t1_vs_t2.empty else global_bowler_avg

    t2_vs_t1 = batsman_vs_bowler[
        (batsman_vs_bowler["batter"].isin(team2_batsmen)) & 
        (batsman_vs_bowler["bowler"].isin(team1_bowlers))
    ]
    t2_runs_vs_t1 = np.nanmean(t2_vs_t1["batsman_runs"]) if not t2_vs_t1.empty else global_batsman_avg
    t2_sr_vs_t1 = np.nanmean(t2_vs_t1["strike_rate"]) if not t2_vs_t1.empty else 100
    t2_dismissals_vs_t1 = np.nanmean(t2_vs_t1["is_wicket"]) if not t2_vs_t1.empty else global_bowler_avg

    t1_batsmen_venue_df = batsman_venue[
        (batsman_venue["batter"].isin(team1_batsmen)) & (batsman_venue["venue"] == venue)
    ]
    t1_batsmen_venue = np.nanmean(t1_batsmen_venue_df["avg_runs_venue"]) if not t1_batsmen_venue_df.empty else global_batsman_avg
    t2_batsmen_venue_df = batsman_venue[
        (batsman_venue["batter"].isin(team2_batsmen)) & (batsman_venue["venue"] == venue)
    ]
    t2_batsmen_venue = np.nanmean(t2_batsmen_venue_df["avg_runs_venue"]) if not t2_batsmen_venue_df.empty else global_batsman_avg
    t1_bowlers_venue_df = bowler_venue[
        (bowler_venue["bowler"].isin(team1_bowlers)) & (bowler_venue["venue"] == venue)
    ]
    t1_bowlers_venue = np.nanmean(t1_bowlers_venue_df["avg_wickets_venue"]) if not t1_bowlers_venue_df.empty else global_bowler_avg
    t2_bowlers_venue_df = bowler_venue[
        (bowler_venue["bowler"].isin(team2_bowlers)) & (bowler_venue["venue"] == venue)
    ]
    t2_bowlers_venue = np.nanmean(t2_bowlers_venue_df["avg_wickets_venue"]) if not t2_bowlers_venue_df.empty else global_bowler_avg

    t1_team = team1 if team1 else matches_df[matches_df["match_id"].isin(deliveries_df[deliveries_df["batter"].isin(team1_players)]["match_id"].unique())]["team1"].mode().iloc[0]
    t2_team = team2 if team2 else matches_df[matches_df["match_id"].isin(deliveries_df[deliveries_df["batter"].isin(team2_players)]["match_id"].unique())]["team2"].mode().iloc[0]
    t1_team_venue_win_df = team_venue_wins[
        (team_venue_wins["team"] == t1_team) & (team_venue_wins["venue"] == venue)
    ]
    t1_team_venue_win = np.nanmean(t1_team_venue_win_df["win_rate_venue"]) if not t1_team_venue_win_df.empty else 0.5
    t2_team_venue_win_df = team_venue_wins[
        (team_venue_wins["team"] == t2_team) & (team_venue_wins["venue"] == venue)
    ]
    t2_team_venue_win = np.nanmean(t2_team_venue_win_df["win_rate_venue"]) if not t2_team_venue_win_df.empty else 0.5

    t1_batsmen_form_vals = [get_recent_form(p, match_id) for p in team1_batsmen] if match_id else [global_batsman_avg] * len(team1_batsmen)
    t1_batsmen_form = np.nanmean(t1_batsmen_form_vals) if t1_batsmen_form_vals else global_batsman_avg
    t1_bowlers_form_vals = [get_recent_form(p, match_id, True) for p in team1_bowlers] if match_id else [global_bowler_avg] * len(team1_bowlers)
    t1_bowlers_form = np.nanmean(t1_bowlers_form_vals) if t1_bowlers_form_vals else global_bowler_avg
    t2_batsmen_form_vals = [get_recent_form(p, match_id) for p in team2_batsmen] if match_id else [global_batsman_avg] * len(team2_batsmen)
    t2_batsmen_form = np.nanmean(t2_batsmen_form_vals) if t2_batsmen_form_vals else global_batsman_avg
    t2_bowlers_form_vals = [get_recent_form(p, match_id, True) for p in team2_bowlers] if match_id else [global_bowler_avg] * len(team2_bowlers)
    t2_bowlers_form = np.nanmean(t2_bowlers_form_vals) if t2_bowlers_form_vals else global_bowler_avg

    toss_factor_df = toss_impact[
        (toss_impact["venue"] == venue) & 
        (toss_impact["toss_decision"] == toss_decision)
    ]
    toss_factor = np.nanmean(toss_factor_df["toss_win_rate"]) if not toss_factor_df.empty else 0.5

    h2h_df = team_h2h[(team_h2h["team1"] == t1_team) & (team_h2h["team2"] == t2_team)]
    t1_win_rate_h2h = h2h_df["t1_win_rate_h2h"].iloc[0] if not h2h_df.empty else 0.5

    return [
        t1_runs_vs_t2, t1_sr_vs_t2, t1_dismissals_vs_t2, t1_batsmen_venue, t1_bowlers_venue, t1_team_venue_win, t1_batsmen_form, t1_bowlers_form,
        t2_runs_vs_t1, t2_sr_vs_t1, t2_dismissals_vs_t1, t2_batsmen_venue, t2_bowlers_venue, t2_team_venue_win, t2_batsmen_form, t2_bowlers_form,
        toss_factor, t1_win_rate_h2h
    ]

# Load data
matches_df, deliveries_df, df = load_or_preprocess_data()

# Precompute stats
batsman_vs_bowler, batsman_venue, bowler_venue, team_venue_wins, team_h2h, toss_impact, global_batsman_avg, global_bowler_avg = precompute_stats(deliveries_df, matches_df)

# Load or train model
model = get_model(matches_df, deliveries_df, batsman_vs_bowler, batsman_venue, bowler_venue, team_venue_wins, team_h2h, toss_impact, global_batsman_avg, global_bowler_avg)

# Dash app setup (unchanged from here onward)
app = Dash(__name__, suppress_callback_exceptions=True)




styles = {
 "container": {"max-width": "900px", "margin": "0 auto", "padding": "20px", "font-family": "Arial, sans-serif"},
 "header": {"text-align": "center", "color": "#2c3e50", "padding": "20px 0", "background-color": "#ecf0f1"},
 "dropdown": {"width": "250px", "margin": "10px auto", "border-radius": "5px"},
 "table": {"border": "1px solid #bdc3c7", "width": "80%", "margin": "20px auto", "border-collapse": "collapse", "box-shadow": "0 2px 5px rgba(0,0,0,0.1)"},
 "th": {"background-color": "#34495e", "color": "white", "padding": "12px", "text-align": "left", "border": "1px solid #bdc3c7"},
 "td": {"padding": "12px", "border": "1px solid #bdc3c7", "background-color": "#fff"},
 "section": {"background-color": "#ffffff", "padding": "20px", "margin": "20px 0", "border-radius": "8px", "box-shadow": "0 2px 10px rgba(0,0,0,0.05)"},
 "label": {"font-weight": "bold", "margin-bottom": "5px", "display": "block"}
}

app.layout = html.Div([
 html.H1("IPL Analysis Dashboard", style=styles["header"]),
 html.Button("📄 Download Full Report", id="download-report-btn", n_clicks=0, style={"margin": "20px", "padding": "10px 20px", "background-color": "#27ae60", "color": "white", "border": "none", "border-radius": "5px"}),
html.Div(id="full-report-download"),


 html.Div([
     html.Div([
         html.Label("Select Season(s)", style=styles["label"]),
         dcc.Dropdown(
             id="season-filter",
             options=[{"label": str(s), "value": s} for s in df["season"].unique()],
             value=[df["season"].max()],
             multi=True,
             placeholder="Select Season(s)",
             style=styles["dropdown"]
         ),
         html.Label("Select Team(s)", style=styles["label"]),
         dcc.Dropdown(
             id="team-filter",
             options=[{"label": t, "value": t} for t in df["batting_team"].unique()],
             value=None,
             multi=True,
             placeholder="Select Team(s)",
             style=styles["dropdown"]
         )
     ], style={"width": "40%", "display": "inline-block", "vertical-align": "top", "padding": "10px"}),
     html.Div([
         html.Label("Select Stat Category", style=styles["label"]),
         dcc.Dropdown(
             id="stat-category",
             options=[
                 {"label": "Match Overview", "value": "match_overview"},
                 {"label": "Team Performance", "value": "team_performance"},
                 {"label": "Player Performance", "value": "player_performance"},
                 {"label": "Head-to-Head Stats", "value": "head_to_head"},
                 {"label": "Batsman vs Bowler", "value": "batsman_vs_bowler"},
                 {"label": "Individual Dismissal Stats", "value": "dismissal_stats"},
                 {"label": "Ball-by-Ball Insights", "value": "ball_by_ball"},
                 {"label": "Venue Analysis", "value": "venue_analysis"},
                 {"label": "Match Prediction", "value": "match_prediction"}
             ],
             value="match_overview",
             clearable=False,
             style=styles["dropdown"]
         )
     ], style={"width": "40%", "display": "inline-block", "vertical-align": "top", "padding": "10px"})
 ], style={"display": "flex", "justify-content": "space-around", "background-color": "#ecf0f1", "padding": "20px"}),

 html.Div(id="content-area", style=styles["section"])
], style=styles["container"])

def create_table(df, columns):
 data_columns = [col for col in columns if col != "Rank"]
 return html.Table([
     html.Thead(html.Tr([html.Th("Rank", style=styles["th"])] + [html.Th(col, style=styles["th"]) for col in data_columns])),
     html.Tbody([
         html.Tr([html.Td(str(i + 1), style=styles["td"])] + [html.Td(str(row[col]), style=styles["td"]) for col in data_columns])
         for i, (_, row) in enumerate(df.iterrows())
     ])
 ])

@app.callback(
 Output("content-area", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("stat-category", "value")]
)
def update_content_layout(selected_seasons, selected_teams, category):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]

 if category == "match_overview":
     return [
         html.H2("Match Overview", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Div([
             html.Label("Matches by Season", style=styles["label"]),
             dcc.Dropdown(id="matches-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="matches-by-season-table")
         ]),
         html.Div([
             html.Label("Toss Impact", style=styles["label"]),
             html.Div(id="toss-impact-table")
         ]),
         html.Div([
             html.Label("Highest Targets", style=styles["label"]),
             dcc.Dropdown(id="targets-rank-limit", options=[5, 10, 20], value=5, clearable=False, style=styles["dropdown"]),
             html.Div(id="highest-targets-table")
         ]),
         html.Div([
             html.Label("Close Finishes", style=styles["label"]),
             html.Div(id="close-finishes-table")
         ])
     ]

 elif category == "team_performance":
     return [
         html.H2("Team Performance", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Div([
             html.Label("Top Teams by Runs", style=styles["label"]),
             dcc.Dropdown(id="team-runs-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="top-teams-runs-table")
         ]),
         html.Div([
             html.Label("Team Win Rate", style=styles["label"]),
             dcc.Dropdown(id="team-wins-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="team-win-rate-table")
         ])
     ]

 elif category == "player_performance":
     return [
         html.H2("Player Performance", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Div([
             html.Label("Top Run Scorers", style=styles["label"]),
             dcc.Dropdown(id="scorers-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="top-run-scorers-table")
         ]),
         html.Div([
             html.Label("Top Wicket Takers", style=styles["label"]),
             dcc.Dropdown(id="wickets-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="top-wickets-table")
         ]),
         html.Div([
             html.Label("Top Bowlers by Dot Balls", style=styles["label"]),
             dcc.Dropdown(id="dot-balls-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="top-dot-bowlers-table")
         ]),
         html.Div([
             html.Label("Top Bowlers by Economy Rate", style=styles["label"]),
             dcc.Dropdown(id="economy-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="top-economy-bowlers-table")
         ]),
         html.Div([
             html.Label("Player of the Match Awards", style=styles["label"]),
             dcc.Dropdown(id="potm-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="potm-awards-table")
         ])
     ]

 elif category == "head_to_head":
     return [
         html.H2("Head-to-Head Stats", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Label("Select Player 1", style=styles["label"]),
         dcc.Dropdown(id="player1", options=[{"label": p, "value": p} for p in df["batter"].unique()], value=None, placeholder="Select Player 1", style=styles["dropdown"]),
         html.Label("Select Player 2", style=styles["label"]),
         dcc.Dropdown(id="player2", options=[{"label": p, "value": p} for p in df["batter"].unique()], value=None, placeholder="Select Player 2", style=styles["dropdown"]),
         html.Div(id="head-to-head-table")
     ]

 elif category == "batsman_vs_bowler":
     return [
         html.H2("Batsman vs Bowler Stats", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Label("Select Batsman", style=styles["label"]),
         dcc.Dropdown(id="batsman", options=[{"label": b, "value": b} for b in df["batter"].unique()], value=None, placeholder="Select Batsman", style=styles["dropdown"]),
         html.Label("Select Bowler", style=styles["label"]),
         dcc.Dropdown(id="bowler", options=[{"label": b, "value": b} for b in df["bowler"].unique()], value=None, placeholder="Select Bowler", style=styles["dropdown"]),
         html.Div(id="batsman-vs-bowler-table")
     ]

 elif category == "dismissal_stats":
     return [
         html.H2("Individual Dismissal Stats", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Label("Select Player", style=styles["label"]),
         dcc.Dropdown(id="dismissal-player", options=[{"label": p, "value": p} for p in df["batter"].unique()], value=None, placeholder="Select Player", style=styles["dropdown"]),
         html.Label("Select Rank Limit", style=styles["label"]),
         dcc.Dropdown(id="dismissal-rank-limit", options=[5, 10], value=5, clearable=False, style=styles["dropdown"]),
         html.Div(id="dismissal-stats-table")
     ]

 elif category == "ball_by_ball":
     return [
         html.H2("Ball-by-Ball Insights", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Div([
             html.Label("Runs per Over", style=styles["label"]),
             dcc.Dropdown(id="runs-over-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="runs-per-over-table")
         ]),
         html.Div([
             html.Label("Scoring Distribution", style=styles["label"]),
             html.Div(id="scoring-dist-table")
         ])
     ]

 elif category == "venue_analysis":
     return [
         html.H2("Venue Analysis", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Div([
             html.Label("Top Venues by Runs", style=styles["label"]),
             dcc.Dropdown(id="venue-runs-rank-limit", options=[5, 10, 20], value=10, clearable=False, style=styles["dropdown"]),
             html.Div(id="venue-runs-table")
         ])
     ]

 elif category == "match_prediction":
     return [
         html.H2("Match Prediction", style={"color": "#34495e", "margin-bottom": "20px"}),
         html.Div([
             html.Label("Team 1 Batsmen (6)", style=styles["label"]),
             dcc.Dropdown(
                 id="team1-batsmen",
                 options=[{"label": p, "value": p} for p in sorted(df["batter"].unique())],
                 multi=True,
                 placeholder="Select 6 Batsmen for Team 1",
                 style={"width": "100%", "margin": "10px 0"}
             ),
             html.Label("Team 1 Bowlers (5)", style=styles["label"]),
             dcc.Dropdown(
                 id="team1-bowlers",
                 options=[{"label": p, "value": p} for p in sorted(df["bowler"].unique())],
                 multi=True,
                 placeholder="Select 5 Bowlers for Team 1",
                 style={"width": "100%", "margin": "10px 0"}
             )
         ], style={"width": "45%", "display": "inline-block", "vertical-align": "top", "padding": "10px"}),
         html.Div([
             html.Label("Team 2 Batsmen (6)", style=styles["label"]),
             dcc.Dropdown(
                 id="team2-batsmen",
                 options=[{"label": p, "value": p} for p in sorted(df["batter"].unique())],
                 multi=True,
                 placeholder="Select 6 Batsmen for Team 2",
                 style={"width": "100%", "margin": "10px 0"}
             ),
             html.Label("Team 2 Bowlers (5)", style=styles["label"]),
             dcc.Dropdown(
                 id="team2-bowlers",
                 options=[{"label": p, "value": p} for p in sorted(df["bowler"].unique())],
                 multi=True,
                 placeholder="Select 5 Bowlers for Team 2",
                 style={"width": "100%", "margin": "10px 0"}
             )
         ], style={"width": "45%", "display": "inline-block", "vertical-align": "top", "padding": "10px"}),
         html.Label("Venue", style=styles["label"]),
         dcc.Dropdown(
             id="prediction-venue",
             options=[{"label": v, "value": v} for v in matches_df["venue"].unique()],
             value=None,
             placeholder="Select Venue",
             style=styles["dropdown"]
         ),
         html.Label("Toss Winner", style=styles["label"]),
         dcc.Dropdown(
             id="toss-winner",
             options=[{"label": "Team 1", "value": "Team 1"}, {"label": "Team 2", "value": "Team 2"}],
             value=None,
             placeholder="Select Toss Winner",
             style=styles["dropdown"]
         ),
         html.Label("Toss Decision", style=styles["label"]),
         dcc.Dropdown(
             id="toss-decision",
             options=[{"label": "Bat", "value": "bat"}, {"label": "Field", "value": "field"}],
             value=None,
             placeholder="Select Toss Decision",
             style=styles["dropdown"]
         ),
         html.Button("Predict", id="predict-button", n_clicks=0, style={"margin": "20px", "padding": "10px 20px", "background-color": "#34495e", "color": "white", "border": "none", "border-radius": "5px"}),
         html.Div(id="prediction-output")
     ]

 return html.Div("Select a category", style={"text-align": "center", "padding": "20px"})

@app.callback(
 Output("matches-by-season-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("matches-rank-limit", "value")]
)
def update_matches_by_season(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("season")["match_id"].nunique().reset_index().nlargest(limit, "match_id").rename(columns={"season": "Season", "match_id": "Matches"}),
     ["Rank", "Season", "Matches"]
 )

@app.callback(
 Output("toss-impact-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value")]
)
def update_toss_impact(selected_seasons, selected_teams):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 toss_wins = filtered_df.groupby("match_id").first().eval("toss_winner == winner").value_counts().reset_index()
 toss_wins.columns = ["Outcome", "Count"]
 toss_wins["Outcome"] = toss_wins["Outcome"].map({True: "Toss & Match Won", False: "Toss Won, Match Lost"})
 return create_table(toss_wins, ["Rank", "Outcome", "Count"])

@app.callback(
 Output("highest-targets-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("targets-rank-limit", "value")]
)
def update_highest_targets(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("match_id").first().reset_index().nlargest(limit, "target_runs")[["match_id", "target_runs", "team1", "team2"]].rename(columns={"match_id": "Match ID", "target_runs": "Target Runs", "team1": "Team 1", "team2": "Team 2"}),
     ["Rank", "Match ID", "Target Runs", "Team 1", "Team 2"]
 )

@app.callback(
 Output("close-finishes-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value")]
)
def update_close_finishes(selected_seasons, selected_teams):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     pd.DataFrame({
         "Category": ["Close Finishes (<10 runs/1 wicket)", "Others"],
         "Count": [
             filtered_df.groupby("match_id").first().reset_index().query("result_margin < 10 and result in ['runs', 'wickets']").shape[0],
             filtered_df["match_id"].nunique() - filtered_df.groupby("match_id").first().reset_index().query("result_margin < 10 and result in ['runs', 'wickets']").shape[0]
         ]
     }),
     ["Rank", "Category", "Count"]
 )

@app.callback(
 Output("top-teams-runs-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("team-runs-rank-limit", "value")]
)
def update_top_teams_runs(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("batting_team")["batsman_runs"].sum().nlargest(limit).reset_index().rename(columns={"batting_team": "Team", "batsman_runs": "Runs"}),
     ["Rank", "Team", "Runs"]
 )

@app.callback(
 Output("team-win-rate-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("team-wins-rank-limit", "value")]
)
def update_team_win_rate(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("winner")["match_id"].nunique().nlargest(limit).reset_index().rename(columns={"winner": "Team", "match_id": "Wins"}),
     ["Rank", "Team", "Wins"]
 )

@app.callback(
 Output("top-run-scorers-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("scorers-rank-limit", "value")]
)
def update_top_run_scorers(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("batter")["batsman_runs"].sum().nlargest(limit).reset_index().rename(columns={"batter": "Batter", "batsman_runs": "Runs"}),
     ["Rank", "Batter", "Runs"]
 )

@app.callback(
 Output("top-wickets-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("wickets-rank-limit", "value")]
)
def update_top_wickets(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df[filtered_df["is_wicket"] == 1].groupby("bowler").size().nlargest(limit).reset_index(name="Wickets").rename(columns={"bowler": "Bowler"}),
     ["Rank", "Bowler", "Wickets"]
 )

@app.callback(
 Output("top-dot-bowlers-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("dot-balls-rank-limit", "value")]
)
def update_top_dot_bowlers(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df[filtered_df["batsman_runs"] == 0].groupby("bowler").size().nlargest(limit).reset_index(name="Dot Balls").rename(columns={"bowler": "Bowler"}),
     ["Rank", "Bowler", "Dot Balls"]
 )

@app.callback(
 Output("top-economy-bowlers-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("economy-rank-limit", "value")]
)
def update_top_economy_bowlers(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("bowler").agg({"batsman_runs": "sum", "ball": "count"}).assign(
         overs=lambda x: x["ball"] / 6, economy=lambda x: x["batsman_runs"] / x["overs"]
     ).query("overs >= 50").nsmallest(limit, "economy").reset_index()[["bowler", "economy"]].rename(columns={"bowler": "Bowler", "economy": "Economy Rate"}),
     ["Rank", "Bowler", "Economy Rate"]
 )

@app.callback(
 Output("potm-awards-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("potm-rank-limit", "value")]
)
def update_potm_awards(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("player_of_match")["match_id"].nunique().nlargest(limit).reset_index().rename(columns={"player_of_match": "Player", "match_id": "Awards"}),
     ["Rank", "Player", "Awards"]
 )

@app.callback(
 Output("head-to-head-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("player1", "value"),
  Input("player2", "value")]
)
def update_head_to_head(selected_seasons, selected_teams, player1, player2):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 if player1 and player2:
     p1_data = filtered_df[filtered_df["batter"] == player1]
     p2_data = filtered_df[filtered_df["batter"] == player2]
     head_to_head = pd.DataFrame({
         "Player": [player1, player2],
         "Runs": [p1_data["batsman_runs"].sum(), p2_data["batsman_runs"].sum()],
         "Sixes": [p1_data[p1_data["batsman_runs"] == 6].shape[0], p2_data[p2_data["batsman_runs"] == 6].shape[0]],
         "Fours": [p1_data[p1_data["batsman_runs"] == 4].shape[0], p2_data[p2_data["batsman_runs"] == 4].shape[0]],
         "Strike Rate": [
             f"{(p1_data['batsman_runs'].sum() / p1_data.shape[0] * 100):.2f}" if p1_data.shape[0] > 0 else "0.00",
             f"{(p2_data['batsman_runs'].sum() / p2_data.shape[0] * 100):.2f}" if p2_data.shape[0] > 0 else "0.00"
         ],
         "Wickets": [
             filtered_df[(filtered_df["is_wicket"] == 1) & (filtered_df["bowler"] == player1)].shape[0],
             filtered_df[(filtered_df["is_wicket"] == 1) & (filtered_df["bowler"] == player2)].shape[0]
         ],
         "Matches": [p1_data["match_id"].nunique(), p2_data["match_id"].nunique()]
     })
     return create_table(head_to_head, ["Rank", "Player", "Runs", "Sixes", "Fours", "Strike Rate", "Wickets", "Matches"])
 return html.Table([html.Tr([html.Td("Select two players to compare", style=styles["td"])])], style=styles["table"])

@app.callback(
 Output("batsman-vs-bowler-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("batsman", "value"),
  Input("bowler", "value")]
)
def update_batsman_vs_bowler(selected_seasons, selected_teams, batsman, bowler):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 if batsman and bowler:
     vs_data = filtered_df[(filtered_df["batter"] == batsman) & (filtered_df["bowler"] == bowler)]
     vs_stats = pd.DataFrame({
         "Stat": ["Runs Scored", "Balls Faced", "Sixes", "Fours", "Dismissals", "Strike Rate"],
         "Value": [
             vs_data["batsman_runs"].sum(),
             vs_data.shape[0],
             vs_data[vs_data["batsman_runs"] == 6].shape[0],
             vs_data[vs_data["batsman_runs"] == 4].shape[0],
             vs_data[vs_data["is_wicket"] == 1].shape[0],
             f"{(vs_data['batsman_runs'].sum() / vs_data.shape[0] * 100):.2f}" if vs_data.shape[0] > 0 else "0.00"
         ]
     })
     return create_table(vs_stats, ["Rank", "Stat", "Value"])
 return html.Table([html.Tr([html.Td("Select a batsman and bowler", style=styles["td"])])], style=styles["table"])

@app.callback(
 Output("dismissal-stats-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("dismissal-player", "value"),
  Input("dismissal-rank-limit", "value")]
)
def update_dismissal_stats(selected_seasons, selected_teams, dismissal_player, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 if dismissal_player:
     dismissals = filtered_df[(filtered_df["batter"] == dismissal_player) & (filtered_df["is_wicket"] == 1)].groupby("bowler").size().nlargest(limit).reset_index(name="Dismissals")
     sixes_conceded = filtered_df[(filtered_df["batter"] == dismissal_player) & (filtered_df["batsman_runs"] == 6)].groupby("bowler").size().nlargest(limit).reset_index(name="Sixes")
     sr_data = filtered_df[filtered_df["batter"] == dismissal_player].groupby("bowler").agg({"batsman_runs": "sum", "ball": "count"})
     sr_data["Strike Rate"] = (sr_data["batsman_runs"] / sr_data["ball"] * 100).replace([float('inf'), -float('inf')], 0)
     highest_sr = sr_data["Strike Rate"].nlargest(limit).reset_index()
     highest_sr["Strike Rate"] = highest_sr["Strike Rate"].apply(lambda x: f"{x:.2f}")
     
     return [
         html.Div([
             html.Label("Most Dismissed By", style=styles["label"]),
             create_table(dismissals.rename(columns={"bowler": "Bowler"}), ["Rank", "Bowler", "Dismissals"])
         ]),
         html.Div([
             html.Label("Most Sixes Conceded By", style=styles["label"]),
             create_table(sixes_conceded.rename(columns={"bowler": "Bowler"}), ["Rank", "Bowler", "Sixes"])
         ]),
         html.Div([
             html.Label("Highest Strike Rate Against", style=styles["label"]),
             create_table(highest_sr.rename(columns={"bowler": "Bowler"}), ["Rank", "Bowler", "Strike Rate"])
         ])
     ]
 return html.Table([html.Tr([html.Td("Select a player", style=styles["td"])])], style=styles["table"])

@app.callback(
 Output("runs-per-over-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("runs-over-rank-limit", "value")]
)
def update_runs_per_over(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("over")["batsman_runs"].mean().nlargest(limit).reset_index().rename(columns={"over": "Over", "batsman_runs": "Average Runs"}),
     ["Rank", "Over", "Average Runs"]
 )

@app.callback(
 Output("scoring-dist-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value")]
)
def update_scoring_dist(selected_seasons, selected_teams):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df["batsman_runs"].value_counts().reset_index().rename(columns={"index": "Runs", "batsman_runs": "Count"}),
     ["Rank", "Runs", "Count"]
 )

@app.callback(
 Output("venue-runs-table", "children"),
 [Input("season-filter", "value"),
  Input("team-filter", "value"),
  Input("venue-runs-rank-limit", "value")]
)
def update_venue_runs(selected_seasons, selected_teams, limit):
 filtered_df = df
 if selected_seasons:
     filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
 if selected_teams:
     filtered_df = filtered_df[filtered_df["batting_team"].isin(selected_teams)]
 return create_table(
     filtered_df.groupby("venue")["batsman_runs"].sum().nlargest(limit).reset_index().rename(columns={"venue": "Venue", "batsman_runs": "Runs"}),
     ["Rank", "Venue", "Runs"]
 )

@app.callback(
    Output("prediction-output", "children"),
    [Input("predict-button", "n_clicks")],
    [State("team1-batsmen", "value"),
     State("team1-bowlers", "value"),
     State("team2-batsmen", "value"),
     State("team2-bowlers", "value"),
     State("prediction-venue", "value"),
     State("toss-winner", "value"),
     State("toss-decision", "value")]
)

def predict_match(n_clicks, team1_batsmen, team1_bowlers, team2_batsmen, team2_bowlers, venue, toss_winner, toss_decision):
 if n_clicks > 0 and team1_batsmen and team1_bowlers and team2_batsmen and team2_bowlers and venue and toss_winner and toss_decision:
     if len(team1_batsmen) != 6 or len(team1_bowlers) != 5 or len(team2_batsmen) != 6 or len(team2_bowlers) != 5:
         return html.Div("Please select exactly 6 batsmen and 5 bowlers for each team.", style={"color": "red"})
     
     features = compute_team_features(team1_batsmen, team1_bowlers, team2_batsmen, team2_bowlers, venue, toss_decision)
     probability = model.predict_proba([features])[0]
     prediction = model.predict([features])[0]

# probability[1] is Team 1 win chance (since y = 1 for team1 wins)
     prob_team1 = probability[1]
     prob_team2 = probability[0]

     winner = "Team 1" if prediction == 1 else "Team 2"
     return html.Div([
         html.H3(f"Predicted Winner: {winner}"),
         html.P(f"Probability - Team 1: {prob_team1:.2f}, Team 2: {prob_team2:.2f}")
     ], style={"text-align": "center", "margin-top": "20px"})
@app.callback(
    Output("full-report-download", "children"),
    [Input("download-report-btn", "n_clicks")],
    [State("season-filter", "value"),
     State("team-filter", "value"),
     State("stat-category", "value")]
)
# Only run preprocessing if file does NOT exist
def create_full_report(n_clicks, seasons, teams, category):
    if n_clicks == 0:
        return ""

    sections = []

    filtered_df = df
    if seasons:
        filtered_df = filtered_df[filtered_df["season"].isin(seasons)]
    if teams:
        filtered_df = filtered_df[filtered_df["batting_team"].isin(teams)]

    # Section: Filters
    sections.append(("Filters Applied", [
        f"Selected Season(s): {', '.join(map(str, seasons)) if seasons else 'All'}",
        f"Selected Team(s): {', '.join(teams) if teams else 'All'}",
        f"Category: {category}"
    ]))

    if category == "match_overview":
        season_counts = filtered_df.groupby("season")["match_id"].nunique()
        sections.append(("Match Overview", [f"{s}: {c} matches" for s, c in season_counts.items()]))

        toss_wins = filtered_df.groupby("match_id").first().eval("toss_winner == winner").value_counts().reset_index()
        toss_wins.columns = ["Outcome", "Count"]
        toss_wins["Outcome"] = toss_wins["Outcome"].map({True: "Toss & Match Won", False: "Toss Won, Match Lost"})
        sections.append(("Toss Impact", [f"{row['Outcome']}: {row['Count']}" for _, row in toss_wins.iterrows()]))

        highest_targets = filtered_df.groupby("match_id").first().reset_index().nlargest(5, "target_runs")
        highest_targets = highest_targets[["match_id", "target_runs", "team1", "team2"]]
        sections.append(("Highest Targets", [
            f"{row['match_id']}: {row['team1']} vs {row['team2']} - Target {row['target_runs']}"
            for _, row in highest_targets.iterrows()
        ]))

        close_df = filtered_df.groupby("match_id").first().reset_index()
        close_count = close_df.query("result_margin < 10 and result in ['runs', 'wickets']").shape[0]
        total_matches = close_df.shape[0]
        sections.append(("Close Finishes", [
            f"Close Matches: {close_count}",
            f"Other Matches: {total_matches - close_count}"
        ]))

    elif category == "team_performance":
        team_runs = filtered_df.groupby("batting_team")["batsman_runs"].sum().nlargest(10)
        sections.append(("Top Teams by Runs", [f"{team}: {runs} runs" for team, runs in team_runs.items()]))

        team_wins = filtered_df.groupby("winner")["match_id"].nunique().nlargest(10)
        sections.append(("Team Win Rates", [f"{team}: {wins} wins" for team, wins in team_wins.items()]))

    elif category == "player_performance":
        top_batters = filtered_df.groupby("batter")["batsman_runs"].sum().nlargest(10)
        sections.append(("Top Run Scorers", [f"{p}: {r} runs" for p, r in top_batters.items()]))

        top_wickets = filtered_df[filtered_df["is_wicket"] == 1].groupby("bowler").size().nlargest(10)
        sections.append(("Top Wicket Takers", [f"{b}: {w} wickets" for b, w in top_wickets.items()]))

        top_dot_balls = filtered_df[filtered_df["batsman_runs"] == 0].groupby("bowler").size().nlargest(10)
        sections.append(("Top Dot Ball Bowlers", [f"{b}: {c} dot balls" for b, c in top_dot_balls.items()]))

        econ_df = filtered_df.groupby("bowler").agg({"batsman_runs": "sum", "ball": "count"})
        econ_df["overs"] = econ_df["ball"] / 6
        econ_df["economy"] = econ_df["batsman_runs"] / econ_df["overs"]
        econ_df = econ_df.query("overs >= 50").nsmallest(10, "economy")
        sections.append(("Top Economy Bowlers", [f"{i}: {row['economy']:.2f}" for i, row in econ_df.iterrows()]))

        potm = filtered_df.groupby("player_of_match")["match_id"].nunique().nlargest(10)
        sections.append(("Player of the Match Awards", [f"{p}: {a} awards" for p, a in potm.items()]))

    elif category == "venue_analysis":
        top_venues = filtered_df.groupby("venue")["batsman_runs"].sum().nlargest(10)
        sections.append(("Top Venues by Runs", [f"{v}: {r} runs" for v, r in top_venues.items()]))

    elif category == "ball_by_ball":
        runs_over = filtered_df.groupby("over")["batsman_runs"].mean().nlargest(10)
        sections.append(("Avg Runs per Over", [f"Over {o}: {r:.2f} runs" for o, r in runs_over.items()]))

        scoring_dist = filtered_df["batsman_runs"].value_counts().sort_index()
        sections.append(("Scoring Distribution", [f"Runs {k}: {v} balls" for k, v in scoring_dist.items()]))

    elif category == "dismissal_stats":
        # Only meaningful when a player is selected, so show a message
        sections.append(("Dismissal Stats", ["Select a specific player to see detailed dismissal breakdowns."]))

    elif category == "batsman_vs_bowler":
        sections.append(("Batsman vs Bowler", ["Select a batsman and bowler to see their detailed head-to-head stats."]))

    elif category == "head_to_head":
        sections.append(("Head-to-Head", ["Select two players to compare."]))

    elif category == "match_prediction":
        sections.append(("Match Prediction", ["Run a prediction first to generate report output."]))

    # Generate and return the final PDF
    download_url = generate_full_pdf_report("IPL Dashboard Full Report", sections)
    return html.A("⬇️ Click here to download the full report", href=download_url, download="ipl_full_report.pdf", target="_blank", style={"margin-top": "15px", "display": "block"})

if __name__ == "__main__":
    app.run(debug=False)