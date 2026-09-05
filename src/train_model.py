import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("../data/safiri_final_shipment_eta_dataset_250.csv")

print("Dataset shape:", df.shape)
print(df.head())


# ============================================================
# 2. CONVERT TIMESTAMPS
# ============================================================

time_columns = [
    "scheduled_departure",
    "estimated_departure",
    "actual_departure",
    "scheduled_arrival",
    "estimated_arrival",
    "actual_arrival"
]

for column in time_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")


# ============================================================
# 3. DEFINE ETA / DELAY
# ============================================================

# Final arrival delay:
#
# actual arrival - scheduled arrival
#
# Positive value = late
# Negative value = early

df["arrival_delay_hours"] = (
    df["actual_arrival"] -
    df["scheduled_arrival"]
).dt.total_seconds() / 3600


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Difference between latest estimated ETA
# and originally scheduled ETA.

df["eta_update_gap_hours"] = (
    df["estimated_arrival"] -
    df["scheduled_arrival"]
).dt.total_seconds() / 3600


# Departure time patterns

df["scheduled_departure_hour"] = (
    df["scheduled_departure"].dt.hour
)

df["scheduled_departure_dayofweek"] = (
    df["scheduled_departure"].dt.dayofweek
)


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

target = "arrival_delay_hours"

features = [
    "departure_delay_hours",
    "scheduled_transit_hours",
    "eta_update_gap_hours",
    "scheduled_departure_hour",
    "scheduled_departure_dayofweek",
    "ship",
    "depPort",
    "arrPort",
    "route"
]

X = df[features]
y = df[target]


# Remove rows with missing target/features

data = pd.concat([X, y], axis=1).dropna()

X = data[features]
y = data[target]


# ============================================================
# 6. NUMERICAL / CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "departure_delay_hours",
    "scheduled_transit_hours",
    "eta_update_gap_hours",
    "scheduled_departure_hour",
    "scheduled_departure_dayofweek"
]

categorical_features = [
    "ship",
    "depPort",
    "arrPort",
    "route"
]


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 8. RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. TRAIN MODEL
# ============================================================

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 11. PREDICTION
# ============================================================

predictions = pipeline.predict(X_test)


# ============================================================
# 12. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.4f} hours")
print(f"RMSE : {rmse:.4f} hours")
print(f"R²   : {r2:.4f}")


# ============================================================
# 13. BASELINE MODEL
# ============================================================

# Simple delay propagation assumption:
#
# Final arrival delay ≈ departure delay

baseline_predictions = X_test[
    "departure_delay_hours"
]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)


print("\n==============================")
print("BASELINE")
print("==============================")

print(
    f"Baseline MAE  : "
    f"{baseline_mae:.4f} hours"
)

print(
    f"Baseline RMSE : "
    f"{baseline_rmse:.4f} hours"
)


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

preprocessor_fitted = (
    pipeline.named_steps["preprocessor"]
)

rf_model = (
    pipeline.named_steps["model"]
)


feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)


importances = (
    rf_model.feature_importances_
)


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)


print("\n==============================")
print("TOP FEATURES")
print("==============================")

print(
    importance_df.head(15)
)


# ============================================================
# 15. PREDICTED ETA
# ============================================================

test_results = X_test.copy()

test_results["actual_delay_hours"] = (
    y_test.values
)

test_results["predicted_delay_hours"] = (
    predictions
)

test_results["absolute_error_hours"] = abs(
    test_results["actual_delay_hours"]
    -
    test_results["predicted_delay_hours"]
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

test_results.to_csv(
    "../data/predictions.csv",
    index=False
)

importance_df.to_csv(
    "../data/feature_importance.csv",
    index=False
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "../data/eta_prediction_model.pkl"
)


print("\nModel and results saved successfully.")