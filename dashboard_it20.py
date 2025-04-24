

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fpdf import FPDF
import datetime
import tempfile

# === Load data and model ===
@st.cache_data
def load_data():
    deliveries = pd.read_csv("deliveries_it20.csv")
    matches = pd.read_csv("matches_it20.csv")
    return deliveries, matches

@st.cache_resource
def load_model():
    with open("xgb_model_it201.pkl", "rb") as f:
        model = pickle.load(f)
    return model

deliveries_df, matches_df = load_data()
model = load_model()

deliveries_df.columns = deliveries_df.columns.str.lower().str.strip()
matches_df.columns = matches_df.columns.str.lower().str.strip()

teams = sorted(matches_df['team1'].dropna().unique())
all_players = sorted(set(deliveries_df['batter'].unique()) | set(deliveries_df['bowler'].unique()))
venues = sorted(matches_df['venue'].dropna().unique())
toss_options = ["bat", "field"]

# === PDF Reporting Tool ===
def generate_pdf_report(content_lines, filename="it20_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("IT20 Cricket Dashboard Report")
    pdf.cell(200, 10, txt="IT20 Cricket Dashboard Report", ln=True, align="C")
    pdf.cell(200, 10, txt=datetime.datetime.now().strftime("Generated on: %Y-%m-%d %H:%M:%S"), ln=True, align="C")
    pdf.ln(10)
    for line in content_lines:
        pdf.multi_cell(0, 10, txt=line)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# === Sidebar ===
st.sidebar.title("IT20 Match Predictor & Stats")
mode = st.sidebar.radio("Choose Mode", ["Match Prediction", "Statistics"])

# === Precompute recent form ===
@st.cache_data
def compute_recent_form():
    batter_form = (
        deliveries_df.groupby(["batter", "match_id"])["batsman_runs"]
        .sum().reset_index().sort_values("match_id")
    )
    batter_form["recent_form"] = (
        batter_form.groupby("batter")["batsman_runs"]
        .rolling(window=5, min_periods=1).mean().reset_index(drop=True)
    )
    form_dict_bat = {(row["batter"], row["match_id"]): row["recent_form"] for _, row in batter_form.iterrows()}

    bowler_form = (
        deliveries_df.groupby(["bowler", "match_id"])["is_wicket"]
        .sum().reset_index().sort_values("match_id")
    )
    bowler_form["recent_form"] = (
        bowler_form.groupby("bowler")["is_wicket"]
        .rolling(window=5, min_periods=1).mean().reset_index(drop=True)
    )
    form_dict_bowl = {(row["bowler"], row["match_id"]): row["recent_form"] for _, row in bowler_form.iterrows()}
    return form_dict_bat, form_dict_bowl

form_bat, form_bowl = compute_recent_form()

# === Precompute stats ===
@st.cache_data
def compute_stats():
    bvbowler = deliveries_df.groupby(['batter', 'bowler']).agg({
        'batsman_runs': 'sum', 'ball': 'count', 'is_wicket': 'sum'
    }).reset_index()
    bvbowler['strike_rate'] = (bvbowler['batsman_runs'] / bvbowler['ball']) * 100
    bat_venue = deliveries_df.groupby(['batter', 'venue'])['batsman_runs'].mean().reset_index(name='avg_runs')
    bowl_venue = deliveries_df[deliveries_df['is_wicket'] == 1].groupby(['bowler', 'venue']).size().reset_index(name='wickets')
    team_venue = matches_df.groupby(['team1', 'venue']).apply(lambda x: (x['winner'] == x['team1']).mean()).reset_index(name='win_rate')
    h2h = matches_df.groupby(['team1', 'team2']).apply(lambda x: (x['winner'] == x['team1']).mean()).reset_index(name='win_rate')
    return bvbowler, bat_venue, bowl_venue, team_venue, h2h

bvbowler, batvenue, bowlvenue, teamvenue, h2h_stats = compute_stats()

# === Match Prediction ===
if mode == "Match Prediction":
    st.header("🏏 Predict Match Winner")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, key="team1")
        t1_bat = st.multiselect("Team 1 Batters (select 6)", all_players, key="t1_bat")
        t1_bowl = st.multiselect("Team 1 Bowlers (select 5)", all_players, key="t1_bowl")
    with col2:
        team2 = st.selectbox("Team 2", [t for t in teams if t != team1], key="team2")
        t2_bat = st.multiselect("Team 2 Batters (select 6)", all_players, key="t2_bat")
        t2_bowl = st.multiselect("Team 2 Bowlers (select 5)", all_players, key="t2_bowl")

    venue = st.selectbox("Venue", venues)
    toss_decision = st.selectbox("Toss Decision", toss_options)

    if st.button("Predict Winner"):
        if len(t1_bat) != 6 or len(t1_bowl) != 5 or len(t2_bat) != 6 or len(t2_bowl) != 5:
            st.error("Please select exactly 6 batters and 5 bowlers for both teams.")
        else:
            def avg(lst):
                lst = [x for x in lst if x is not None]
                return np.mean(lst) if len(lst) > 0 else 0

            def get_form(player_list, match_id, bowl=False):
                form_dict = form_bowl if bowl else form_bat
                return avg([form_dict.get((p, match_id), 0) for p in player_list])

            mid = matches_df['match_id'].max() + 1

            def get_stat(df, key1, key2, col):
                val = df[(df.iloc[:, 0].isin([key1])) & (df.iloc[:, 1] == key2)][col]
                return val.mean() if not val.empty else 0

            def vs_stat(batters, bowlers):
                vs = bvbowler[(bvbowler['batter'].isin(batters)) & (bvbowler['bowler'].isin(bowlers))]
                return avg(vs['batsman_runs']), avg(vs['strike_rate']), avg(vs['is_wicket'])

            t1_runs, t1_sr, t1_out = vs_stat(t1_bat, t2_bowl)
            t2_runs, t2_sr, t2_out = vs_stat(t2_bat, t1_bowl)
            t1_bv = avg([get_stat(batvenue, p, venue, 'avg_runs') for p in t1_bat])
            t2_bv = avg([get_stat(batvenue, p, venue, 'avg_runs') for p in t2_bat])
            t1_wv = avg([get_stat(bowlvenue, p, venue, 'wickets') for p in t1_bowl])
            t2_wv = avg([get_stat(bowlvenue, p, venue, 'wickets') for p in t2_bowl])
            t1_tw = get_stat(teamvenue, team1, venue, 'win_rate')
            t2_tw = get_stat(teamvenue, team2, venue, 'win_rate')
            t1_form = get_form(t1_bat, mid)
            t2_form = get_form(t2_bat, mid)
            t1_bf = get_form(t1_bowl, mid, True)
            t2_bf = get_form(t2_bowl, mid, True)
            h2h = get_stat(h2h_stats, team1, team2, 'win_rate')

            features = [[
                t1_runs, t1_sr, t1_out, t1_bv, t1_wv, t1_tw, t1_form, t1_bf,
                t2_runs, t2_sr, t2_out, t2_bv, t2_wv, t2_tw, t2_form, t2_bf,
                0.5, h2h
            ]]

            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][int(pred)]
            winner = team1 if pred == 1 else team2

            st.success(f"🏆 Predicted Winner: {winner} ({prob * 100:.2f}% confidence)")

            report_lines = [
                f"Predicted Winner: {winner} ({prob * 100:.2f}% confidence)",
                f"Team 1: {team1}, Batters: {', '.join(t1_bat)}, Bowlers: {', '.join(t1_bowl)}",
                f"Team 2: {team2}, Batters: {', '.join(t2_bat)}, Bowlers: {', '.join(t2_bowl)}",
                f"Venue: {venue}, Toss Decision: {toss_decision}",
                f"Head-to-Head Win Rate ({team1}): {h2h:.2f}"
            ]
            pdf_path = generate_pdf_report(report_lines)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download Prediction Report", f, file_name="match_prediction_report.pdf")

# === Statistics ===
elif mode == "Statistics":
    st.header("📊 Player & Team Statistics")
    stat_mode = st.radio("Choose Stat Type", ["Batting", "Bowling", "Venue", "Head-to-Head"])

    if stat_mode == "Batting":
        player = st.selectbox("Select Batter", sorted(deliveries_df['batter'].unique()))
        player_data = deliveries_df[deliveries_df['batter'] == player]
        runs = player_data.groupby('match_id')['batsman_runs'].sum()
        st.metric("Total Runs", int(runs.sum()))
        st.metric("Matches", len(runs))
        st.metric("Average", f"{runs.mean():.2f}")
        pdf_path = generate_pdf_report([
            f"Batting Report for {player}",
            f"Total Runs: {int(runs.sum())}",
            f"Matches: {len(runs)}",
            f"Average: {runs.mean():.2f}"
        ])
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Batting Report", f, file_name="batting_stats_report.pdf")

    elif stat_mode == "Bowling":
        player = st.selectbox("Select Bowler", sorted(deliveries_df['bowler'].unique()))
        player_data = deliveries_df[deliveries_df['bowler'] == player]
        wkts = player_data.groupby('match_id')['is_wicket'].sum()
        st.metric("Total Wickets", int(wkts.sum()))
        st.metric("Matches", len(wkts))
        st.metric("Avg Wickets/Match", f"{wkts.mean():.2f}")
        pdf_path = generate_pdf_report([
            f"Bowling Report for {player}",
            f"Total Wickets: {int(wkts.sum())}",
            f"Matches: {len(wkts)}",
            f"Avg Wickets/Match: {wkts.mean():.2f}"
        ])
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Bowling Report", f, file_name="bowling_stats_report.pdf")

    elif stat_mode == "Venue":
        venue = st.selectbox("Select Venue", venues)
        vdata = matches_df[matches_df['venue'] == venue]
        st.metric("Matches Played", len(vdata))
        most_wins = vdata['winner'].value_counts().idxmax()
        st.metric("Most Wins", most_wins)
        pdf_path = generate_pdf_report([
            f"Venue Report for {venue}",
            f"Matches Played: {len(vdata)}",
            f"Team with Most Wins: {most_wins}"
        ])
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Venue Report", f, file_name="venue_stats_report.pdf")

    elif stat_mode == "Head-to-Head":
        t1 = st.selectbox("Team 1", teams, key='h2h1')
        t2 = st.selectbox("Team 2", [t for t in teams if t != t1], key='h2h2')
        h2h_data = matches_df[((matches_df['team1'] == t1) & (matches_df['team2'] == t2)) |
                              ((matches_df['team1'] == t2) & (matches_df['team2'] == t1))]
        win1 = (h2h_data['winner'] == t1).sum()
        win2 = (h2h_data['winner'] == t2).sum()
        total = len(h2h_data)
        st.metric(f"{t1} Wins", win1)
        st.metric(f"{t2} Wins", win2)
        st.metric("Total Matches", total)
        pdf_path = generate_pdf_report([
            f"Head-to-Head Report: {t1} vs {t2}",
            f"{t1} Wins: {win1}",
            f"{t2} Wins: {win2}",
            f"Total Matches: {total}"
        ])
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download H2H Report", f, file_name="h2h_stats_report.pdf")