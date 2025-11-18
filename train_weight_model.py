import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- 1. Load config, features, and JSON ----------

CFG = json.load(open("config.json", "r"))
FEATURE_CSV = CFG["out_csv"]          # "features_area_diameter.csv"
JSON_PATH   = CFG["json_path"]        # "GroundTruth_All_388_Images.json"

df_feat = pd.read_csv(FEATURE_CSV)
print(f"[info] loaded features: {df_feat.shape[0]} rows")

J = json.load(open(JSON_PATH, "r"))
meas = J.get("Measurements", {})
if not meas:
    raise SystemExit("No Measurements found in JSON.")

# build a small table: plant_id -> FreshWeightShoot
rows = []
for pid, rec in meas.items():
    rows.append({
        "plant_id": pid,
        "W_g": rec.get("FreshWeightShoot", np.nan),
        "LeafArea_GT": rec.get("LeafArea", np.nan),
        "Diameter_GT": rec.get("Diameter", np.nan),
        "Variety": rec.get("Variety", "")
    })

df_labels = pd.DataFrame(rows)

# join our features with labels using plant_id
df = df_feat.merge(df_labels, on="plant_id", how="inner")
print(f"[info] merged features+labels: {df.shape[0]} rows")

# ---------- 2. Basic cleaning & quality filter ----------

# keep rows with valid weight and positive area/diameter
df = df[
    df["W_g"].notna()
    & (df["A_cm2"] > 0)
    & (df["D_cm"] > 0)
]

# optional: only use good masks
if "mask_density" in df.columns:
    df = df[(df["mask_density"].isna()) | (df["mask_density"] > 0.80)]
if "invalid_depth_ratio" in df.columns:
    df = df[(df["invalid_depth_ratio"].isna()) | (df["invalid_depth_ratio"] < 0.20)]

df = df.reset_index(drop=True)
print(f"[info] after filtering: {df.shape[0]} usable plants")

if df.shape[0] < 10:
    raise SystemExit("Too few samples after filtering – relax filters or check data.")

# ---------- 3. Build X, y for ALL-OMETRIC regression ----------

# Features: log(A_cm2), log(D_cm)
A = df["A_cm2"].values.astype(np.float64)
D = df["D_cm"].values.astype(np.float64)
W = df["W_g"].values.astype(np.float64)

# log-transform (allometric: ln W = α + β ln A + γ ln D)
logA = np.log(A)
logD = np.log(D)
logW = np.log(W)

X = np.column_stack([logA, logD])
y = logW

# ---------- 4. Train/test split ----------

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, random_state=42
)

print(f"[info] train size: {X_train.shape[0]}, test size: {X_test.shape[0]}")

# ---------- 5. Fit linear regression in log-space ----------

model = LinearRegression()
model.fit(X_train, y_train)

print("\n[model] ln(W) = alpha + beta_A * ln(A_cm2) + beta_D * ln(D_cm)")
print(f" alpha  = {model.intercept_:.4f}")
print(f" beta_A = {model.coef_[0]:.4f}")
print(f" beta_D = {model.coef_[1]:.4f}")

# ---------- 6. Evaluate on test set ----------

y_pred_test_log = model.predict(X_test)
# convert back to grams
W_true_test = np.exp(y_test)
W_pred_test = np.exp(y_pred_test_log)

mae = mean_absolute_error(W_true_test, W_pred_test)
r2  = r2_score(y_test, y_pred_test_log)

print("\n[test performance]")
print(f" MAE (grams) = {mae:.3f}")
print(f" R^2 (log-space) = {r2:.3f}")

# show a few example plants (test set)
print("\n[sample predictions on test plants]")
preview = pd.DataFrame({
    "plant_id": df_test["plant_id"].values,
    "Variety": df_test["Variety"].values,
    "W_true_g": W_true_test,
    "W_pred_g": W_pred_test,
    "A_cm2": df_test["A_cm2"].values,
    "D_cm": df_test["D_cm"].values,
}).reset_index(drop=True)

print(preview.head(10).to_string(index=False))

# ---------- 7. Save model parameters for later use ----------

params = {
    "alpha": float(model.intercept_),
    "beta_A": float(model.coef_[0]),
    "beta_D": float(model.coef_[1]),
}

with open("allometric_model.json", "w") as f:
    json.dump(params, f, indent=2)

print("\n[done] saved allometric_model.json with learned parameters.")
