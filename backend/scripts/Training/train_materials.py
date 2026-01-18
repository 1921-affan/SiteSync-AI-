import pandas as pd
import numpy as np
import joblib
import warnings
import os
import sys

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'backend', 'data', 'raw', 'train_materials_final.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🚀 STARTING MATERIAL TRAINING...")
print(f"📂 Loading Data from: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Data file not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

# ==========================================
# 2. FEATURE ENGINEERING (ALIGNED WITH ARCHITECTURE)
# ==========================================
# Features defined in System Architecture (Panel 1)
features = [
    "week_number",          # <--- Critical Temporal Feature
    "site_area_sqft", 
    "total_floors", 
    "floors_completed", 
    "construction_stage",
    "property_type",        # <--- Critical Complexity Feature
    "lost_days", 
    "weather_factor",       # Derived from weather
    "site_dimensions",      # Raw input 
    "work_pace", 
    "weather_condition"
]

# Note: is_complex_design is not in raw material CSV (it's in Labour CSV), 
# but property_type captures the same info.

targets = ["req_cement_bags", "req_steel_kg", "req_bricks_nos", "req_sand_tons"]

print(f"✅ Features Selected: {len(features)} inputs")
df_model = df[features + targets].copy()

# --- ENCODING ---
encoders = {}

# 1. Work Pace
df_model["work_pace_enc"] = df_model["work_pace"].map({"Normal": 0, "Fast_Track": 1})

# 2. Weather
weather_map = {"Normal": 0, "Light_Rain": 1, "High_Wind": 2, "Heatwave": 3, "Heavy_Rain": 4}
df_model["weather_enc"] = df_model["weather_condition"].map(weather_map)

# 3. Site Dimensions
le_site = LabelEncoder()
df_model["site_dim_enc"] = le_site.fit_transform(df_model["site_dimensions"])
encoders['site_dim'] = le_site

# 4. Property Type (NEW - Critical)
le_prop = LabelEncoder()
df_model["prop_type_enc"] = le_prop.fit_transform(df_model["property_type"])
encoders['prop_type'] = le_prop

# Drop raw categorical columns
df_model.drop(columns=["work_pace", "weather_condition", "site_dimensions", "property_type"], inplace=True)

# --- TARGET TRANSFORMATION (Log1p) ---
target_trans_cols = []
for col in targets:
    trans_col = f"{col}_log"
    df_model[trans_col] = np.log1p(df_model[col])
    target_trans_cols.append(trans_col)

X = df_model.drop(columns=targets + target_trans_cols)
y = df_model[target_trans_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. MODEL TRAINING (MAE OPTIMIZED)
# ==========================================
# User's optimized parameters logic
param_dist = {
    'estimator__n_estimators': [500, 1000, 1500],
    'estimator__learning_rate': [0.01, 0.03, 0.05],
    'estimator__num_leaves': [31, 50, 70],
    'estimator__max_depth': [-1, 10, 20],
    'estimator__min_child_samples': [20, 30, 50],
    'estimator__reg_alpha': [0, 0.1, 1.0],
    'estimator__reg_lambda': [0, 0.1, 1.0]
}

# Base model with L1 objective (MAE optimization)
lgbm = LGBMRegressor(objective='regression_l1', random_state=42, n_jobs=-1, verbose=-1)
model = MultiOutputRegressor(lgbm)

print("\n🔧 Running Randomized Search (MAE Optimized)...")
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10, # Reduced slightly for speed, increase if needed
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print(f"✅ Best Params: {search.best_params_}")

# ==========================================
# 4. EVALUATION & SAVING
# ==========================================
print("\n===== 🏁 PERFORMANCE (Test Set) =====")

pred_log = best_model.predict(X_test)
pred_orig = np.expm1(pred_log)
y_test_orig = np.expm1(y_test).values

targets_names = ["Cement (Bags)", "Steel (Kg)", "Bricks (Nos)", "Sand (Tons)"]

for i, name in enumerate(targets_names):
    mae = mean_absolute_error(y_test_orig[:, i], pred_orig[:, i])
    r2 = r2_score(y_test_orig[:, i], pred_orig[:, i])
    print(f"{name:15} | MAE: {mae:8.2f} | R²: {r2:.4f}")

# Save Artifacts
print("\n💾 Saving Models to backend/models/...")
joblib.dump(best_model, os.path.join(MODEL_DIR, "material_model.pkl"))
joblib.dump(encoders['site_dim'], os.path.join(MODEL_DIR, "site_dim_encoder.pkl"))
joblib.dump(encoders['prop_type'], os.path.join(MODEL_DIR, "prop_type_encoder.pkl")) # New

print("✅ Training Complete.")
