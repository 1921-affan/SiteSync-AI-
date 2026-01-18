import pandas as pd
import numpy as np
import joblib
import os
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

# Global Artifacts
model = None
feature_columns = None

def load_artifacts():
    global model, feature_columns
    if model is None:
        try:
            model = joblib.load(os.path.join(MODEL_DIR, "labour_prediction_model.pkl"))
            feature_columns = joblib.load(os.path.join(MODEL_DIR, "labour_model_features.pkl"))
        except FileNotFoundError as e:
            print(f"❌ Error: Model files missing. {e}")
            sys.exit(1)

def predict_labour_needs(inputs):
    """
    Predicts Skilled and Unskilled labour.
    
    inputs (dict):
        - week_number (int)
        - site_area_sqft (float)
        - construction_stage (int)
        - floors_completed (int)
        - is_complex_design (int) -> 0 or 1
        - productivity_rate (float) -> e.g. 1.0 or 1.2
        - req_cement_bags (float)
        - req_bricks_nos (float)
        - req_steel_kg (float)
        - req_sand_tons (float)
        - task_type (str) -> e.g., "Concrete_Pour"
    """
    load_artifacts()
    
    # 1. Create Base DataFrame
    data = {
        "week_number": [inputs['week_number']],
        "site_area_sqft": [inputs['site_area_sqft']],
        "construction_stage": [inputs['construction_stage']],
        "floors_completed": [inputs['floors_completed']],
        "is_complex_design": [inputs['is_complex_design']],
        "productivity_rate": [inputs['productivity_rate']],
        "req_cement_bags": [inputs['req_cement_bags']],
        "req_bricks_nos": [inputs['req_bricks_nos']],
        "req_steel_kg": [inputs['req_steel_kg']],
        "req_sand_tons": [inputs['req_sand_tons']],
        "task_type": [inputs['task_type']] # Raw string
    }
    df = pd.DataFrame(data)
    
    # 2. Reconstruct One-Hot Encoding
    # We essentially simulate get_dummies but strictly enforce the training columns
    df_encoded = pd.get_dummies(df, columns=['task_type'])
    
    # 3. Align Columns (The Critical Step)
    # Add missing columns (filled with 0)
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Drop extra columns (if any unseen tasks appear)
    df_final = df_encoded[feature_columns]
    
    # 4. Predict
    pred_log = model.predict(df_final)
    pred_orig = np.expm1(pred_log) # Inverse Log
    
    # 5. Round Up (You can't hire 2.3 people)
    pred_rounded = np.ceil(pred_orig)
    
    return {
        "req_skilled_labour": int(pred_rounded[0][0]),
        "req_unskilled_labour": int(pred_rounded[0][1])
    }

if __name__ == "__main__":
    # Test
    test_input = {
        "week_number": 5,
        "site_area_sqft": 1200,
        "construction_stage": 3,
        "floors_completed": 1,
        "is_complex_design": 0,
        "productivity_rate": 1.0,
        "req_cement_bags": 50,
        "req_bricks_nos": 0,
        "req_steel_kg": 600,
        "req_sand_tons": 5,
        "task_type": "Concrete_Pour"
    }
    print(predict_labour_needs(test_input))