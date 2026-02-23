import fastf1
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression


# Enable Fastf1 caching
fastf1.Cache.enable_cache("f1_cache")

# Map 2026 drivers to teams 
DRIVER_TEAM_2026 = {
    "NOR": "McLaren",    "PIA": "McLaren",
    "LEC": "Ferrari",    "HAM": "Ferrari",
    "VER": "Red Bull",   "HAD": "Red Bull",
    "RUS": "Mercedes",   "ANT": "Mercedes",
    "SAI": "Williams",   "ALB": "Williams",
    "GAS": "Alpine",     "COL": "Alpine",
    "HUL": "Audi",       "BOR": "Audi",
    "BEA": "Haas",       "OCO": "Haas",
    "ALO": "Aston Martin","STR": "Aston Martin",
    "LAW": "Racing Bulls","LIN": "Racing Bulls",  
    "BOT": "Cadillac",   "PER": "Cadillac",
}

# Load FastF1 2025 Australian GP race + quali session data
session_r = fastf1.get_session(2025, 1, "R")
session_r.load(telemetry=False, weather=False, messages=False)

session_q = fastf1.get_session(2025, 1, "Q")
session_q.load(telemetry=False, weather=False, messages=False)

# Extract race time and DNF results
race_results = session_r.results[["Abbreviation", "TeamName", "Time", "ClassifiedPosition"]].copy()
race_results.rename(columns={"Abbreviation": "Driver"}, inplace=True)
race_results["TotalRaceTime_s"] = race_results["Time"].dt.total_seconds()
race_results["DNF"] = (~race_results["ClassifiedPosition"].apply(lambda x: str(x).isdigit())).astype(int)

# Feature engineering: 
# 1. Calculate DNF rate for each driver
race_results = session_r.results[["Abbreviation", "TeamName", "Time", "ClassifiedPosition"]].copy()
race_results.rename(columns={"Abbreviation": "Driver"}, inplace=True)
race_results["TotalRaceTime_s"] = race_results["Time"].dt.total_seconds()
race_results["DNF"] = (~race_results["ClassifiedPosition"].apply(lambda x: str(x).isdigit())).astype(int)

# 2. Calculate median lap time for each team, then rank

laps = session_r.laps[["Driver", "LapTime", "IsAccurate"]].copy()
laps.dropna(subset=["LapTime"], inplace=True)
laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

team_pace = (
    laps[laps["IsAccurate"]]
    .merge(race_results[["Driver", "TeamName"]], on="Driver")
    .groupby("TeamName")["LapTime_s"]
    .median()
    .rank()
    .reset_index()
    .rename(columns={"LapTime_s": "TeamPaceRank", "TeamName": "Team"})
)

team_pace["Team"] = team_pace["Team"].replace({"Kick Sauber": "Audi"})

# 3. Calculate qualifying gap to the pole position 2026
q_results = session_q.results[["Abbreviation", "Q1", "Q2", "Q3"]].copy()
q_results.rename(columns={"Abbreviation": "Driver"}, inplace=True)

q_results = session_q.results[["Abbreviation", "Q1", "Q2", "Q3"]].copy()
q_results.rename(columns={"Abbreviation": "Driver"}, inplace=True)

def best_q_time(row):
    for col in ["Q3", "Q2", "Q1"]:
        if pd.notna(row[col]) and row[col].total_seconds() > 0:
            return row[col].total_seconds()
    return np.nan

q_results["QualTime_s"] = q_results.apply(best_q_time, axis=1)
q_results["QualGapToPole_s"] = q_results["QualTime_s"] - q_results["QualTime_s"].min()
q_results = q_results[["Driver", "QualGapToPole_s"]]

# Merge features
driver_team_df = pd.DataFrame(list(DRIVER_TEAM_2026.items()), columns=["Driver", "Team"])

merged = (
    driver_team_df
    .merge(race_results[["Driver", "TotalRaceTime_s", "DNF"]], on="Driver", how="left")
    .merge(q_results, on="Driver", how="left")
    .merge(team_pace, on="Team", how="left")
)
if merged.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

#Rookie drivers with no historical data get assigned value based on team pace
for col in ["TotalRaceTime_s", "QualGapToPole_s"]:
    merged.loc[merged["Driver"] == "LIN", col] = \
        merged.loc[merged["Driver"] == "LAW", col].values[0] * 1.02
merged.loc[merged["Driver"] == "LIN", ["DNF", "TeamPaceRank"]] = \
    merged.loc[merged["Driver"] == "LAW", ["DNF", "TeamPaceRank"]].values[0]


merged.dropna(inplace=True)
print(f"Drivers with complete data: {len(merged)}")

# Train Gradient Boosting Model
FEATURES = ["QualGapToPole_s", "TotalRaceTime_s", "DNF", "TeamPaceRank"]
X = merged[FEATURES].values
y = merged["TotalRaceTime_s"].values

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
loo_scores = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
print(f"LOO-CV MAE: {-loo_scores.mean():.1f} ± {loo_scores.std():.1f} seconds")

model.fit(X, y)

# Predict and rank drivers
merged["PredictedRaceTime_s"] = model.predict(X)
merged = merged.sort_values("PredictedRaceTime_s").reset_index(drop=True)
merged.index += 1

print("\nPredicted 2026 Australian GP Finishing Order")
print("(using 2025 data as proxy — rerun after March 8 2026 with 2026 qualifying)\n")
print(merged[["Driver", "Team", "PredictedRaceTime_s"]].to_string())