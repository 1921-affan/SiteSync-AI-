import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'backend', 'data', 'raw')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models')

RISK_DATA_PATH = os.path.join(DATA_DIR, "train_risk_final.csv")
LABOUR_DATA_PATH = os.path.join(DATA_DIR, "train_labour_final.csv")

print("🚀 STARTING RISK MODEL TRAINING (Tri-Model Architecture)...")
print(f"📂 Loading Data from: {DATA_DIR}")

if not os.path.exists(RISK_DATA_PATH) or not os.path.exists(LABOUR_DATA_PATH):
    print(f"❌ Error: Missing data files in {DATA_DIR}")
    sys.exit(1)

# ==========================================
# 2. DATA PREP (MERGE & ENCODE)
# ==========================================
df_risk = pd.read_csv(RISK_DATA_PATH)
df_labour = pd.read_csv(LABOUR_DATA_PATH)

# Aggregate Labour to Weekly level (Summing skilled/unskilled for the whole week)
# This aligns perfectly with the "Weekly Prediction" goal
df_labour_agg = df_labour.groupby(['project_id', 'week_number'])[
    ['req_skilled_labour', 'req_unskilled_labour']
].sum().reset_index()

# Merge on Week and Project ID
df_merged = pd.merge(df_risk, df_labour_agg, on=['project_id', 'week_number'], how='inner')

# Feature Engineering
# Material Availability Map
mat_map = {'Available': 0, 'Limited': 1, 'Shortage': 2}
df_merged['mat_avail_enc'] = df_merged['material_avail_status'].map(mat_map)

# Labour Availability Map
lab_map = {'Sufficient': 0, 'Moderate': 1, 'Critical': 2}
df_merged['lab_avail_enc'] = df_merged['labour_avail_status'].map(lab_map)

# Weather Map
weather_map = {'Normal': 0, 'Light_Rain': 1, 'High_Wind': 2, 'Heatwave': 3, 'Heavy_Rain': 4}
df_merged['weather_enc'] = df_merged['weather_condition'].map(weather_map)

print(f"✅ Data Prepared & Merged. Rows: {len(df_merged)}")

# ==========================================
# 3. DEFINE TUNING GRID (SHARED LOGIC)
# ==========================================
# 45 Fits as requested (15 Candidates * 3 Folds)
reg_param_dist = {
    'estimator__n_estimators': [500, 1000, 1500],
    'estimator__learning_rate': [0.01, 0.03, 0.05],
    'estimator__num_leaves': [31, 50, 70],
    'estimator__min_child_samples': [10, 20], 
    'estimator__reg_alpha': [0, 0.1, 0.5] 
}

# ==========================================
# 4. TUNE MODEL A: SHORTFALL ESTIMATOR
# ==========================================
print("\n🔹 [Model A] Tuning Shortfall Estimator...")
features_A = ['req_skilled_labour', 'req_unskilled_labour', 'lab_avail_enc', 'weather_enc']
target_A = 'labour_shortfall_est'

X_A = df_merged[features_A]
y_A = df_merged[target_A]
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

model_A = MultiOutputRegressor(LGBMRegressor(objective='regression_l1', random_state=42, n_jobs=-1, verbose=-1))

search_A = RandomizedSearchCV(
    model_A, 
    param_distributions=reg_param_dist, 
    n_iter=15,    # 15 Candidates
    scoring='neg_mean_absolute_error', 
    cv=3,         # 3 Folds -> Total 45 Fits
    verbose=1, n_jobs=-1, random_state=42
)
search_A.fit(X_train_A, y_train_A.to_frame())

best_model_A = search_A.best_estimator_
print(f"   ✅ Best Params A: {search_A.best_params_}")
pred_A = best_model_A.predict(X_test_A)
print(f"   Model A MAE: {mean_absolute_error(y_test_A, pred_A):.2f} Workers")
joblib.dump(best_model_A, os.path.join(MODEL_DIR, 'risk_model_shortfall.pkl'))

# ==========================================
# 5. TUNE MODEL B: RISK SCORER
# ==========================================
print("\n🔹 [Model B] Tuning Risk Scorer...")
features_B = [
    'labour_shortfall_est', 'progress_gap_pct', 'mat_avail_enc',
    'hist_material_delay_count', 'weather_impact_flag', 'construction_stage'
]
target_B = 'risk_score_debug'

X_B = df_merged[features_B]
y_B = df_merged[target_B]
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

model_B = MultiOutputRegressor(LGBMRegressor(objective='regression_l1', random_state=42, n_jobs=-1, verbose=-1))

search_B = RandomizedSearchCV(
    model_B, 
    param_distributions=reg_param_dist, 
    n_iter=15, 
    scoring='neg_mean_absolute_error', 
    cv=3, 
    verbose=1, n_jobs=-1, random_state=42
)
search_B.fit(X_train_B, y_train_B.to_frame())

best_model_B = search_B.best_estimator_
print(f"   ✅ Best Params B: {search_B.best_params_}")
pred_B = best_model_B.predict(X_test_B)
print(f"   Model B MAE: {mean_absolute_error(y_test_B, pred_B):.2f} Points")
joblib.dump(best_model_B, os.path.join(MODEL_DIR, 'risk_model_scorer.pkl'))

# ==========================================
# 6. TUNE MODEL C: RISK CLASSIFIER
# ==========================================
print("\n🔹 [Model C] Tuning Final Classifier...")
features_C = ['risk_score_debug', 'progress_gap_pct']
target_C = 'delay_risk'

X_C = df_merged[features_C]
y_C = df_merged[target_C]
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, random_state=42)

clf_param_dist = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05],
    'num_leaves': [31, 50, 70],
    'reg_alpha': [0, 0.1, 0.5]
}

search_C = RandomizedSearchCV(
    LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbose=-1), 
    param_distributions=clf_param_dist, 
    n_iter=15, 
    scoring='accuracy', 
    cv=3, 
    verbose=1, n_jobs=-1, random_state=42
)
search_C.fit(X_train_C, y_train_C)

best_model_C = search_C.best_estimator_
print(f"   ✅ Best Params C: {search_C.best_params_}")

pred_C = best_model_C.predict(X_test_C)
acc = accuracy_score(y_test_C, pred_C)
print(f"   Model C Accuracy: {acc:.2%}")
print(classification_report(y_test_C, pred_C, target_names=['Low', 'Medium', 'High']))

joblib.dump(best_model_C, os.path.join(MODEL_DIR, 'risk_model_classifier.pkl'))
print("\n✅ All 3 Optimized Models Saved.")
