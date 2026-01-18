import pandas as pd
import numpy as np
import joblib
import warnings
import os
import sys

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'backend', 'data', 'raw', 'train_labour_final.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models')

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "labour_prediction_model.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "labour_model_features.pkl")

print("🚀 STARTING ULTIMATE LABOUR MODEL TRAINING...")
print(f"📂 Loading Data from: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Data file not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"📊 Dataset Loaded: {df.shape}")

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# A. One-Hot Encode Task Type
# IMPORTANT: We use dummy encoding because task_type is not ordinal (Concrete != 2 * Excavation)
df_encoded = pd.get_dummies(df, columns=['task_type'], drop_first=False)

# B. Define Base Features 
# Added `week_number` to ensure temporal alignment with user goals
base_features = [
    "week_number",        # <--- Added (Temporal Context)
    "site_area_sqft", 
    "construction_stage", 
    "floors_completed", 
    "is_complex_design", 
    "productivity_rate",  
    "req_cement_bags", 
    "req_bricks_nos", 
    "req_steel_kg", 
    "req_sand_tons"
]

# C. Add Task Columns Dynamically
# We will save this complete list so inference can match it 100%
task_cols = [col for col in df_encoded.columns if "task_type_" in col]
features = base_features + task_cols

targets = ["req_skilled_labour", "req_unskilled_labour"]

X = df_encoded[features]
y = df_encoded[targets]

print(f"✅ Features Selected: {len(features)}")
print(f"   Tasks Encoded: {task_cols}")

# --- TARGET TRANSFORMATION (Log1p) ---
target_trans_cols = []
for col in targets:
    trans_col = f"{col}_log"
    y[trans_col] = np.log1p(y[col]) # Log(1 + x)
    target_trans_cols.append(trans_col)

y_trans = y[target_trans_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y_trans, test_size=0.2, random_state=42)

# ==========================================
# 3. HYPERPARAMETER TUNING
# ==========================================
param_dist = {
    'estimator__n_estimators': [500, 1000, 1500],
    'estimator__learning_rate': [0.01, 0.03, 0.05],
    'estimator__num_leaves': [31, 50, 70],
    'estimator__min_child_samples': [10, 20, 30], 
    'estimator__reg_alpha': [0, 0.1, 0.5],        
    'estimator__colsample_bytree': [0.8, 1.0]     
}

# Base Model: LightGBM with L1 Loss (MAE focus)
lgbm = LGBMRegressor(objective='regression_l1', random_state=42, n_jobs=-1, verbose=-1)
model = MultiOutputRegressor(lgbm)

print("\n🔧 Running Deep Search (MAE Optimized)...")
search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist, 
    n_iter=15, # Restored to user preference
    scoring='neg_mean_absolute_error', 
    cv=3, 
    verbose=1, 
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print(f"✅ Best Params: {search.best_params_}")

# ==========================================
# 4. FINAL EVALUATION
# ==========================================
print("\n===== 🏁 LABOUR MODEL PERFORMANCE =====")
pred_log = best_model.predict(X_test)
pred_orig = np.expm1(pred_log)
y_test_orig = np.expm1(y_test).values
pred_rounded = np.ceil(pred_orig) # Round up to whole humans

targets_names = ["Skilled Labour", "Unskilled Labour"]

for i, name in enumerate(targets_names):
    mae = mean_absolute_error(y_test_orig[:, i], pred_rounded[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_orig[:, i], pred_orig[:, i]))
    r2 = r2_score(y_test_orig[:, i], pred_orig[:, i])
    print(f"{name:20} | MAE: {mae:.2f} Workers | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# ==========================================
# 5. SAVE ARTIFACTS
# ==========================================
joblib.dump(best_model, MODEL_SAVE_PATH)
joblib.dump(features, FEATURE_LIST_PATH) # Critical for Inference Column Order

print(f"\n💾 Model saved to: {MODEL_SAVE_PATH}")
print(f"💾 Feature List saved to: {FEATURE_LIST_PATH}")
