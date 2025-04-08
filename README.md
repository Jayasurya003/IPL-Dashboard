
# 🏏 Cricket Match Winner Prediction (IPL & T20)

This project predicts the winner between two cricket teams for **IPL** and **T20 International** matches using machine learning. It combines real-world cricket data, engineered features, and classification models to generate accurate predictions.

---

## 🔧 Project Setup

To run this project locally using **VS Code**:

```bash
# Step 1: Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate    # Windows

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch the notebook
code main.ipynb
```

---



## 🧠 Project Objective

To develop a machine learning pipeline that can predict the match winner using pre-match information (e.g., teams, venue, form, win rates).

**Target Variable:**  
`1` = team1 wins  
`0` = team2 wins  

**Models Used:**
- Logistic Regression
- Random Forest
- XGBoost

---

## 🔬 Features & Engineering

### IPL Features:
- `t1_h2h_win_rate`: Historical win rate of team1 vs team2.
- `t1_toss_advantage`: How often team1 wins when it wins the toss.
- `t1_venue_win_rate`: team1 win rate at a venue.
- Encoded: `team1`, `team2`, `venue`

### T20 Rolling Features:
- `t1_form`, `t2_form`: Win rate in last 5 games.
- `t1_win_rate_overall`, `t2_win_rate_overall`
- `venue_familiarity_t1`: Matches played at venue.
- Historical features only – no leakage.

---

## 📊 Evaluation Metrics

Used for both datasets:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### 📈 IPL Model Results
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.761    | 0.764     | 0.750  | 0.757    |
| Random Forest      | 0.642    | 0.636     | 0.648  | 0.642    |
| XGBoost            | 0.619    | 0.619     | 0.602  | 0.610    |

### 🌍 T20 Model Results
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.653    | 0.594     | 0.679  | 0.634    |
| XGBoost            | 0.648    | 0.595     | 0.635  | 0.615    |
| Random Forest      | 0.643    | 0.597     | 0.586  | 0.591    |

---

## 💡 Key Insights

- Logistic Regression consistently outperformed complex models.
- Rolling features captured team form and venue familiarity well.
- Time-aware splits prevented data leakage.

---

## 📊 Tech Stack

| Category       | Tools / Libraries              |
|----------------|-------------------------------|
| Programming    | Python                         |
| Data Analysis  | Pandas, NumPy                  |
| Visualization  | Seaborn, Matplotlib            |
| Modeling       | Scikit-learn, XGBoost          |
| Development    | VS Code with venv, previously Colab |

---

## 📝 Authors 

- Lokesh Makineni and Guttapati Jayasurya Reddy
- Project changed from Airline Delay to Cricket (approved by instructor)
- Milestone 2 covers EDA, Feature Engineering, Modeling, and Evaluation
- Final milestone includes dashboard and live predictions (ongoing)
