import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

from lightgbm import LGBMRegressor

df = pd.read_csv("test-data.csv")

# ---------------- CLEAN ---------------- #

df = df.drop(columns=["Name", "Location"], errors="ignore")

# Mileage
df["Mileage"] = df["Mileage"].astype(str).str.extract(r'(\d+\.?\d*)')
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

# Engine
df["Engine"] = df["Engine"].astype(str).str.replace(" CC", "", regex=False)
df["Engine"] = pd.to_numeric(df["Engine"], errors="coerce")

# Power
df["Power"] = df["Power"].astype(str).str.replace(" bhp", "", regex=False)
df["Power"] = pd.to_numeric(df["Power"], errors="coerce")

# Remove Price if exists
if "Price" in df.columns:
    df = df.drop(columns=["Price"])

# ---------------- FEATURE ENGINEERING ---------------- #

df["Car_Age"] = 2025 - df["Year"]

df["Engine_to_Power"] = df["Engine"] / (df["Power"] + 1)
df["KM_per_Year"] = df["Kilometers_Driven"] / (df["Car_Age"] + 1)

# ---------------- HANDLE MISSING ---------------- #

df = df.dropna(subset=["Mileage"])

num_cols = [
    "Year", "Kilometers_Driven", "Engine", "Power",
    "Seats", "Car_Age", "Engine_to_Power", "KM_per_Year"
]

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ["Fuel_Type", "Transmission", "Owner_Type"]

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---------------- FINAL FEATURES ---------------- #

features = [
    "Year",
    "Kilometers_Driven",
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
    "Engine",
    "Power",
    "Seats",
    "Car_Age",
    "Engine_to_Power",
    "KM_per_Year"
]

df = df[features + ["Mileage"]]

# ---------------- SAVE ---------------- #

df.to_csv("cleaned_data.csv", index=False)

print("✅ Cleaned dataset ready")

# ---------------- LOAD CLEAN DATA ---------------- #
df = pd.read_csv("cleaned_data.csv")

# ---------------- FEATURES & TARGET ---------------- #
X = df.drop("Mileage", axis=1)
y = df["Mileage"]

# Categorical & Numerical columns
cat_cols = ["Fuel_Type", "Transmission", "Owner_Type"]
num_cols = [col for col in X.columns if col not in cat_cols]

# ---------------- PREPROCESSOR ---------------- #
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# ---------------- MODEL ---------------- #
model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=40,
    min_child_samples=30,   # ↑ prevents overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
# ---------------- PIPELINE ---------------- #
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ---------------- TRAIN ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# ---------------- EVALUATION ---------------- #
preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"🔥 MAE: {mae:.2f}")
print(f"🔥 R2 Score: {r2:.2f}")

# ---------------- SAVE MODEL ---------------- #
joblib.dump(pipeline, "final_mileage_model.pkl")

print("✅ Model saved as final_mileage_model.pkl")