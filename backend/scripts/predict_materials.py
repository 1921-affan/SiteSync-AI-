import pandas as pd
import numpy as np
import joblib
import os
import sys

# Define Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

# Load Artifacts Global (Lazy Load)
model = None
le_site = None
le_prop = None

def load_artifacts():
    global model, le_site, le_prop
    if model is None:
        try:
            model = joblib.load(os.path.join(MODEL_DIR, "material_model.pkl"))
            le_site = joblib.load(os.path.join(MODEL_DIR, "site_dim_encoder.pkl"))
            le_prop = joblib.load(os.path.join(MODEL_DIR, "prop_type_encoder.pkl"))
        except FileNotFoundError as e:
            print(f"❌ Critical Error: Model files missing. {e}")
            sys.exit(1)

def predict_material_needs(inputs):
    """
    Predicts material quantities for the next week.
    
    inputs (dict):
        - week_number (int)
        - site_area_sqft (float)
        - total_floors (int)
        - floors_completed (int)
        - construction_stage (int)
        - property_type (str)
        - lost_days (int)
        - weather_condition (str) -> "Normal", "Light_Rain", etc.
        - site_dimensions (str) -> "30x40"
        - work_pace (str) -> "Normal"/"Fast_Track"
    """
    load_artifacts()
    
    # 1. Prepare DataFrame
    data = {
        "week_number": [inputs['week_number']],
        "site_area_sqft": [inputs['site_area_sqft']],
        "total_floors": [inputs['total_floors']],
        "floors_completed": [inputs['floors_completed']],
        "construction_stage": [inputs['construction_stage']],
        "lost_days": [inputs['lost_days']],
        # Calculate Weather Factor (Approximate logic from generation script)
        "weather_factor": [1.0] # Default, should be Refined if weather impact is critical input
    }
    
    df_pred = pd.DataFrame(data)

    # 2. Encode Categoricals
    # Weather
    weather_map = {"Normal": 0, "Light_Rain": 1, "High_Wind": 2, "Heatwave": 3, "Heavy_Rain": 4}
    df_pred["weather_enc"] = weather_map.get(inputs['weather_condition'], 0)

    # Site Dim
    try:
        df_pred["site_dim_enc"] = le_site.transform([inputs['site_dimensions']])[0]
    except ValueError:
        # Handle unseen labels (fallback to most common or 0)
        df_pred["site_dim_enc"] = 0 

    # Work Pace
    df_pred["work_pace_enc"] = 1 if inputs['work_pace'] == "Fast_Track" else 0

    # Property Type (CRITICAL NEW FEATURE)
    try:
        df_pred["prop_type_enc"] = le_prop.transform([inputs['property_type']])[0]
    except ValueError:
        print(f"⚠️ Warning: Unknown property type '{inputs['property_type']}'. Using default.")
        df_pred["prop_type_enc"] = 0

    # 3. Order Columns (Must match training exact order)
    # Features from train_materials.py:
    # [week_number, site_area_sqft, total_floors, floors_completed, construction_stage,
    #  prop_type_enc, lost_days, weather_factor, site_dim_enc, work_pace_enc, weather_enc]
    
    # Note: Training script dropped raw columns. We must pass exactly these:
    feature_order = [
        "week_number", "site_area_sqft", "total_floors", "floors_completed", 
        "construction_stage", "prop_type_enc", "lost_days", "weather_factor", 
        "site_dim_enc", "work_pace_enc", "weather_enc"
    ]
    
    X = df_pred[feature_order]

    # 4. Predict
    # Output is Log Transformed, so apply expm1
    pred_log = model.predict(X)
    pred_orig = np.expm1(pred_log) # Inverse Log

    # 5. Format Output
    # Order: Cement, Steel, Bricks, Sand
    return {
        "req_cement_bags": int(pred_orig[0][0]),
        "req_steel_kg": round(pred_orig[0][1], 2),
        "req_bricks_nos": int(pred_orig[0][2]),
        "req_sand_tons": round(pred_orig[0][3], 2)
    }

if __name__ == "__main__":
    # Test Run
    test_input = {
        "week_number": 5,
        "site_area_sqft": 1200,
        "total_floors": 3,
        "floors_completed": 1,
        "construction_stage": 3,
        "property_type": "2BHK",
        "lost_days": 0,
        "weather_condition": "Normal",
        "site_dimensions": "30x40",
        "work_pace": "Normal"
    }
    print(predict_material_needs(test_input))