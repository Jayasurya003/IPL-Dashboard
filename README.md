# 🏏 Cricket Match Winner Prediction (IPL & T20)

This project predicts the winner between two cricket teams for **IPL** and **T20 International** matches using machine learning. It combines real-world cricket data, advanced engineered features, and classification models to generate accurate predictions and interactive dashboard outputs.

---

## 🚀 Features

- 🔮 Predict winners using pre-match team and venue context
- ⚙️ Trained XGBoost models using player and team statistics
- 📊 Streamlit dashboards for IPL, T20, and a unified view
- 📥 Feature-rich pipelines for IPL and T20 data with independent training

---

## 📊 Engineered Features

### T20 Features (Advanced Player-Level)
- `t1_form`, `t2_form`: Rolling avg of last 5 matches (batting/wicket form)
- `t1_win_rate_overall`, `t2_win_rate_overall`: Historical performance
- `batsman_vs_bowler`: Head-to-head stats
- `venue_familiarity`: Runs/wickets at a given venue
- `h2h_win_rate`: Win % of team1 over team2
- `toss_factor`: Fixed to 0.5 due to missing toss data

### IPL Features (Team-Level Stats)
- `t1_h2h_win_rate`: Historical win rate of team1 vs team2
- `t1_toss_advantage`: How often team1 wins when it wins the toss
- `t1_venue_win_rate`: team1 win rate at a venue
- Encoded: `team1`, `team2`, `venue`

---

## 🛠️ Setup Instructions

1. **Move to project folder:**
```bash
cd cap5771sp25-project
```

2. **Create virtual environment and activate it:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # For macOS/Linux
.venv\Scripts\activate   # For Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run T20 dashboard:**
```bash
streamlit run dashboard_it20.py
```

5. **Run IPL dashboard:**
```bash
cd ipl
source .venv/bin/activate
pip install -r requirements.txt
python3 dashboard.py
```

6. **Run Combined Dashboard:**
```bash
python3 dashboard_combined.py
```

---

## 📉 Evaluation Metrics (Milestone 3)

- Accuracy: 87%
- F1 Score: 87%
- ROC AUC: 96%

---

## 📉 Evaluation Metrics (Milestone 2)

### 📈 IPL Results
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.761    | 0.764     | 0.750  | 0.757    |
| Random Forest      | 0.642    | 0.636     | 0.648  | 0.642    |
| XGBoost            | 0.619    | 0.619     | 0.602  | 0.610    |

### 🌍 T20 Results
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.653    | 0.594     | 0.679  | 0.634    |
| XGBoost            | 0.648    | 0.595     | 0.635  | 0.615    |
| Random Forest      | 0.643    | 0.597     | 0.586  | 0.591    |

---

## 💡 What is Working

- ✅ Separate pipelines for IPL and T20 datasets
- ✅ Custom `preprocess.py` for rolling features, venue stats, and matchup insights
- ✅ `.pkl` model saving and reloading for consistent reuse
- ✅ Streamlit dashboards: `dashboard_it20.py`, `dashboard.py`, and `dashboard_combined.py`
- ✅ Same metrics used in training and test set for evaluation
- ✅ All major features engineered and integrated

---

## 📦 Tools & Tech Stack

| Category         | Tools Used                                  |
|------------------|----------------------------------------------|
| Programming      | Python                                       |
| Modeling         | XGBoost, Scikit-learn                        |
| UI & Visualization | Streamlit, Matplotlib, Seaborn            |
| Data             | Pandas, NumPy                                |
| Reporting        | fpdf (planned: auto PDF generation)         |

---

## 📄 Milestones

### ✅ Milestone 1:
- Original project (Airline Delay Prediction)

### ✅ Milestone 2:
- Switched to cricket prediction
- Team-level stats, encoded variables
- Used Logistic Regression, Random Forest, XGBoost

### ✅ Milestone 3:
- Player-level stats, rolling form, and venue familiarity
- Optimized XGBoost, 87% accuracy
- Streamlit dashboards per format + combined view
- PDF reporting planned

---

## 👥 Team
- GUTTAPATI JAYASURYA REDDY (T20 Pipeline & Dashboard)
- LOKESH MAKINENI (IPL Pipeline & Integration)

---

## 🎥 Video Links

**4-minute PPT presentation:**
https://youtu.be/s9mcpF-C7W4

**4-minute demo:**
https://youtu.be/bA_75s23IV0
