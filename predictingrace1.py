import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable Fastf1 caching
fastf1.Cache.enable_cache("f1_cache")


# Load FastF1 2025 Australian GP race session
session_2025 = fastf1.get_session(2025, 3, "R")
session_2025.load()

# Extract Lap times
laps_2025 = session_2025.laps[["Driver", "LapTime"]].copy()
laps_2025.dropna(subset=["LapTime"], inplace=True)
laps_2025["LapTime (s)"] = laps_2025["LapTime"].dt.total_seconds()
# 2026 Qualifying Data
qualifying_2026 = pd.DataFrame({
    "Driver": ["LEC", "NOR", "VER", "RUS", "GAS", "BEA", "BOR", "ANT", "LIN", "SAI", "PIA", "OCO", "HAD", "BOT", "HUL", "PER", "STR", "ALO", "ALB", "LAW", "COL", "HAM"],
    "QualifyingTime (s)": [75.096, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755, 75.973, 75.980, 76.062, 76.4, 76.5, 76.6, 76.7, 76.8, 76.9, 77.0, 77.1, 77.2, 77.3, 77.4, 77.5]
})

# Merge 2026 Qualifying Data with 2025 Race Data
merged_data = qualifying_2026.merge(laps_2025, left_on="Driver", right_on="Driver", how="inner")
# Use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2026[["QualifyingTime (s)"]])
qualifying_2026["PredictedRaceTime (s)"] = predicted_lap_times
# Rank drivers by predicted race time
qualifying_2026 = qualifying_2026.sort_values(by="PredictedRaceTime (s)")
# Print final predictions
print ("\nPredicted 2026 Australian GP Winner\n")
print (qualifying_2026[["Driver", "PredictedRaceTime (s)"]])
# Evaluate Model
y_pred = model.predict(X_test)
print(f"Model Error : {mean_absolute_error(y_test, y_pred):.2f} seconds")