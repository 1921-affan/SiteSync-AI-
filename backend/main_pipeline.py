import pandas as pd
import numpy as np
import joblib
import pulp
import os
import warnings
import sys

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CENTRAL CONFIGURATION
# ==========================================
# This is the ONLY place you need to change data to test different scenarios
USER_INPUTS = {
    "project_id": "PROJ-2025-001",
    "week_number": 15,
    "site_area_sqft": 2400,
    "total_floors": 4,
    "floors_completed": 2,
    "construction_stage": 3,     # 1:Foundation, 3:Structure, 5:Finishing
    "site_dimensions": "40x60",  # Must match encoder categories
    "is_complex_design": 0,      # 0=No, 1=Yes
    
    # Dynamic Weekly Inputs
    "task_type": "Concrete_Pour", # Context for Labour
    "productivity_rate": 100,     # Standard=100
    
    # Risk Factors (Simulated Site Status)
    "work_pace": "Normal",        # Normal / Fast_Track
    "weather_condition": "Light_Rain", # Normal, Light_Rain, Heavy_Rain, High_Wind
    "material_avail_status": "Limited",# Available, Limited, Shortage
    "labour_avail_status": "Moderate", # Sufficient, Moderate, Critical
    "progress_gap_pct": 12.0,     # % Behind Schedule
    "hist_material_delay_count": 2,
    "lost_days": 1
}

# Paths
MODEL_DIR = "models"
DATA_STATIC_DIR = "data/static"
OUTPUT_DIR = "outputs/pipeline_results"

print("🚀 STARTING CONSTRUCTION MANAGEMENT PIPELINE...")
print(f"   Project: {USER_INPUTS['project_id']} | Week: {USER_INPUTS['week_number']}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD ALL MODELS (Fail Fast)
# ==========================================
print("\n[1/5] Loading AI Models...")
try:
    # Model 1: Materials (Assuming these exist from previous steps)
    # If you haven't renamed them, ensure these match your actual filenames
    mat_model = joblib.load(os.path.join(MODEL_DIR, "material_model_final_optimized.pkl"))
    site_encoder = joblib.load(os.path.join(MODEL_DIR, "site_dimension_encoder.pkl"))
    
    # Model 2: Labour
    lab_model = joblib.load(os.path.join(MODEL_DIR, "labour_prediction_model.pkl"))
    lab_features = joblib.load(os.path.join(MODEL_DIR, "labour_model_features.pkl"))
    
    # Model 3: Risk
    risk_shortfall = joblib.load(os.path.join(MODEL_DIR, "risk_model_shortfall.pkl"))
    risk_scorer = joblib.load(os.path.join(MODEL_DIR, "risk_model_scorer.pkl"))
    risk_classifier = joblib.load(os.path.join(MODEL_DIR, "risk_model_classifier.pkl"))
    
    print("✅ All Models Loaded Successfully.")
except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Missing Model File.\n   {e}")
    print("   Please ensure all .pkl files are in the 'models/' folder.")
    sys.exit(1)

# ==========================================
# 3. STEP 1: PREDICT MATERIALS
# ==========================================
print("\n[2/5] Predicting Material Requirements...")

# Prepare Input
df_input = pd.DataFrame([USER_INPUTS])

# Encoders for Material Model
weather_map = {"Normal": 0, "Light_Rain": 1, "High_Wind": 2, "Heatwave": 3, "Heavy_Rain": 4}
df_input["work_pace_enc"] = df_input["work_pace"].map({"Normal": 0, "Fast_Track": 1})
df_input["weather_enc"] = df_input["weather_condition"].map(weather_map)
df_input["site_dim_enc"] = site_encoder.transform(df_input["site_dimensions"])
df_input["weather_factor"] = 0.85 # Simplified factor logic

mat_features = [
    "site_area_sqft", "total_floors", "floors_completed", "construction_stage",
    "lost_days", "weather_factor", "work_pace_enc", "weather_enc", "site_dim_enc"
]

# Predict
pred_mat_log = mat_model.predict(df_input[mat_features])
pred_mat = np.expm1(pred_mat_log)

# Store Results
results = USER_INPUTS.copy()
results["req_cement_bags"] = float(pred_mat[:, 0])
results["req_steel_kg"] = float(pred_mat[:, 1])
results["req_bricks_nos"] = float(pred_mat[:, 2])
results["req_sand_tons"] = float(pred_mat[:, 3])

print(f"   Cement: {results['req_cement_bags']:.1f} bags")
print(f"   Steel:  {results['req_steel_kg']:.1f} kg")

# ==========================================
# 4. STEP 2: PREDICT LABOUR
# ==========================================
print("\n[3/5] Predicting Labour Requirements...")

# Prepare Input (Needs Material Outputs + Task Type)
df_lab = pd.DataFrame(columns=lab_features)
df_lab.loc[0] = 0 # Init with zeros

# Map available features
cols_to_map = [
    "site_area_sqft", "construction_stage", "floors_completed", "is_complex_design",
    "req_cement_bags", "req_bricks_nos", "req_steel_kg", "req_sand_tons", "productivity_rate"
]
for col in cols_to_map:
    df_lab.loc[0, col] = results[col]

# One-Hot Encode Task
task_col = f"task_type_{USER_INPUTS['task_type']}"
if task_col in df_lab.columns:
    df_lab.loc[0, task_col] = 1

# Predict
pred_lab_log = lab_model.predict(df_lab)
pred_lab = np.expm1(pred_lab_log)

results["pred_skilled_labour"] = int(np.ceil(pred_lab[0][0]))
results["pred_unskilled_labour"] = int(np.ceil(pred_lab[0][1]))

print(f"   Skilled:   {results['pred_skilled_labour']} workers")
print(f"   Unskilled: {results['pred_unskilled_labour']} workers")

# ==========================================
# 5. STEP 3: ASSESS DELAY RISK
# ==========================================
print("\n[4/5] Assessing Project Risk...")

# Encodings for Risk Model
mat_status_map = {'Available': 0, 'Limited': 1, 'Shortage': 2}
lab_status_map = {'Sufficient': 0, 'Moderate': 1, 'Critical': 2}

# --- A. Predict Shortfall ---
df_risk_A = pd.DataFrame([{
    "req_skilled_labour": results["pred_skilled_labour"],
    "req_unskilled_labour": results["pred_unskilled_labour"],
    "lab_avail_enc": lab_status_map[USER_INPUTS['labour_avail_status']],
    "weather_enc": weather_map[USER_INPUTS['weather_condition']]
}])

# FIX 1: Extract scalar from array for shortfall
raw_shortfall = risk_shortfall.predict(df_risk_A)
if hasattr(raw_shortfall[0], '__len__'):
    shortfall_val = raw_shortfall[0][0]
else:
    shortfall_val = raw_shortfall[0]
pred_shortfall = float(max(0, shortfall_val))


# --- B. Predict Score ---
weather_impact_flag = 1 if USER_INPUTS['weather_condition'] in ['Light_Rain', 'Heavy_Rain', 'High_Wind'] else 0

# Define df_risk_B BEFORE using it
df_risk_B = pd.DataFrame([{
    "labour_shortfall_est": pred_shortfall,
    "progress_gap_pct": USER_INPUTS['progress_gap_pct'],
    "mat_avail_enc": mat_status_map[USER_INPUTS['material_avail_status']],
    "hist_material_delay_count": USER_INPUTS['hist_material_delay_count'],
    "weather_impact_flag": weather_impact_flag,
    "construction_stage": USER_INPUTS['construction_stage']
}])

# FIX 2: Extract scalar from array for score
raw_score = risk_scorer.predict(df_risk_B)
if hasattr(raw_score[0], '__len__'):
    score_val = raw_score[0][0]
else:
    score_val = raw_score[0]
pred_score = float(min(max(score_val, 0), 15))


# --- C. Classify Risk ---
df_risk_C = pd.DataFrame([{
    "risk_score_debug": pred_score,
    "progress_gap_pct": USER_INPUTS['progress_gap_pct']
}])
pred_class = risk_classifier.predict(df_risk_C)[0]
risk_labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

results["risk_score"] = round(pred_score, 2)
results["risk_class"] = risk_labels[pred_class]
results["est_shortfall"] = round(pred_shortfall, 1)

print(f"   Risk Score: {results['risk_score']}/15")
print(f"   Status:     {results['risk_class']} RISK")
# ==========================================
# 6. STEP 4: OPTIMIZE RESOURCES
# ==========================================
print("\n[5/5] Optimizing Procurement Plan...")

# Load Static Data
try:
    df_mat_cost = pd.read_csv(os.path.join(DATA_STATIC_DIR, "material_costs.csv"))
    df_lab_cost = pd.read_csv(os.path.join(DATA_STATIC_DIR, "labour_costs.csv"))
except FileNotFoundError:
    print("❌ Cost tables not found. Run create_cost_tables.py")
    sys.exit(1)

# Optimization Logic (Simplified for Pipeline - using same logic as standalone script)
risk_buffer = {"LOW": 1.0, "MEDIUM": 1.1, "HIGH": 1.2}[results["risk_class"]]
print(f"   Applying Safety Buffer: {int((risk_buffer-1)*100)}%")

# Setup Optimization Problem
prob = pulp.LpProblem("Weekly_Plan", pulp.LpMinimize)
plan_rows = []
total_cost = 0

# --- Material Optimization ---
mat_map = {
    "req_cement_bags": "Cement", "req_steel_kg": "Steel", 
    "req_bricks_nos": "Bricks", "req_sand_tons": "Sand"
}

for col, name in mat_map.items():
    target = results[col] * risk_buffer
    cost_data = df_mat_cost[df_mat_cost['material_name'] == name].iloc[0]
    
    qty_reg = pulp.LpVariable(f"Reg_{name}", lowBound=0)
    qty_exp = pulp.LpVariable(f"Exp_{name}", lowBound=0)
    
    prob += qty_reg + qty_exp >= target
    prob += qty_reg <= cost_data['max_supply_weekly_regular']
    
    total_cost += (qty_reg * cost_data['cost_regular_inr']) + (qty_exp * cost_data['cost_express_inr'])
    
    # Solve partial (PuLP accumulates constraints)
    # We solve once at the end, but we need variables accessible for reporting
    # To keep this script clean, we'll store the VAR OBJECTS in a list to read later
    plan_rows.append({
        "Type": "Material", "Name": name, "Target": target, 
        "Reg_Var": qty_reg, "Exp_Var": qty_exp, "Unit": cost_data['unit']
    })

# --- Labour Optimization ---
lab_map = {"pred_skilled_labour": "Skilled_Mason", "pred_unskilled_labour": "Unskilled_Helper"}

for col, role in lab_map.items():
    target = results[col] * risk_buffer
    cost_data = df_lab_cost[df_lab_cost['labour_role'] == role].iloc[0]
    
    hire_reg = pulp.LpVariable(f"Reg_{role}", lowBound=0, cat='Integer')
    hire_prem = pulp.LpVariable(f"Prem_{role}", lowBound=0, cat='Integer')
    
    prob += hire_reg + hire_prem >= target
    prob += hire_reg <= cost_data['max_regular_supply']
    
    weekly_wage_reg = cost_data['daily_wage_inr'] * 6
    weekly_wage_prem = (cost_data['daily_wage_inr'] * 1.5) * 6
    total_cost += (hire_reg * weekly_wage_reg) + (hire_prem * weekly_wage_prem)
    
    plan_rows.append({
        "Type": "Labour", "Name": role, "Target": target, 
        "Reg_Var": hire_reg, "Exp_Var": hire_prem, "Unit": "Workers"
    })

prob += total_cost
prob.solve()

# ==========================================
# 7. FINAL REPORT
# ==========================================
print("\n" + "="*50)
print(f"📋 FINAL EXECUTION PLAN | Risk: {results['risk_class']}")
print("="*50)
print(f"{'RESOURCE':<20} | {'TARGET':<10} | {'REGULAR':<10} | {'PREMIUM/OT':<10}")
print("-" * 55)

for row in plan_rows:
    reg_val = row['Reg_Var'].varValue
    exp_val = row['Exp_Var'].varValue
    print(f"{row['Name']:<20} | {row['Target']:<10.1f} | {reg_val:<10.1f} | {exp_val:<10.1f}")
    if exp_val > 0:
        print(f"   ⚠️  ALERT: Using Premium Sourcing for {row['Name']}")

print("-" * 55)
print(f"💰 TOTAL ESTIMATED WEEKLY COST: ₹ {pulp.value(prob.objective):,.2f}")

# Save Summary
results['total_cost_inr'] = pulp.value(prob.objective)
pd.DataFrame([results]).to_csv(f"{OUTPUT_DIR}/final_summary.csv", index=False)
print(f"\n✅ Full log saved to {OUTPUT_DIR}/final_summary.csv")
print("🏁 Pipeline execution completed successfully.") 