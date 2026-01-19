import pandas as pd
import numpy as np
import joblib
import os
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

# Global Models
model_shortfall = None
model_scorer = None
model_classifier = None

def load_risk_models():
    global model_shortfall, model_scorer, model_classifier
    if model_shortfall is None:
        try:
            model_shortfall = joblib.load(os.path.join(MODEL_DIR, "risk_model_shortfall.pkl"))
            model_scorer = joblib.load(os.path.join(MODEL_DIR, "risk_model_scorer.pkl"))
            model_classifier = joblib.load(os.path.join(MODEL_DIR, "risk_model_classifier.pkl"))
        except FileNotFoundError as e:
            print(f"❌ Error: Risk Model files missing. {e}")
            sys.exit(1)

def predict_risk_metrics(inputs):
    """
    Predicts Risk utilizing the Tri-Model Architecture.
    
    inputs (dict):
        # From Panel 2 (Labour Model Output)
        - req_skilled_labour (float)
        - req_unskilled_labour (float)
        
        # User/Site Inputs
        - material_avail_status (str) -> "Available", "Limited", "Shortage"
        - labour_avail_status (str) -> "Sufficient", "Moderate", "Critical"
        - weather_condition (str) -> "Normal", "Light_Rain", "Heavy_Rain", "High_Wind", "Heatwave"
        - progress_gap_pct (float) -> e.g. 12.5
        - hist_material_delay_count (int)
        - construction_stage (int)
        - weather_impact_flag (int) [Optional, auto-derived if missing]
    
    Returns:
        dict: {
            "labour_shortfall_est": float,
            "risk_score": float,
            "risk_class": str (Low/Medium/High)
        }
    """
    load_risk_models()
    
    # --- 1. PREPARE INPUTS & ENCODING ---
    
    # Mappings (Must match train_risk.py)
    mat_map = {'Available': 0, 'Limited': 1, 'Shortage': 2}
    lab_map = {'Sufficient': 0, 'Moderate': 1, 'Critical': 2}
    weather_map = {'Normal': 0, 'Light_Rain': 1, 'High_Wind': 2, 'Heatwave': 3, 'Heavy_Rain': 4}
    
    # Encode
    mat_enc = mat_map.get(inputs.get('material_avail_status', 'Available'), 0)
    lab_enc = lab_map.get(inputs.get('labour_avail_status', 'Sufficient'), 0)
    weather_enc = weather_map.get(inputs.get('weather_condition', 'Normal'), 0)
    
    # Derive Weather Impact Flag if not provided
    # Logic: High Wind(2), Heatwave(3), Heavy Rain(4) = Impact 1
    if 'weather_impact_flag' in inputs:
        weather_impact = inputs['weather_impact_flag']
    else:
        weather_impact = 1 if weather_enc >= 2 else 0

    # --- 2. MODEL A: SHORTFALL ESTIMATOR ---
    # Features: ['req_skilled_labour', 'req_unskilled_labour', 'lab_avail_enc', 'weather_enc']
    input_A = pd.DataFrame([{
        'req_skilled_labour': inputs['req_skilled_labour'],
        'req_unskilled_labour': inputs['req_unskilled_labour'],
        'lab_avail_enc': lab_enc,
        'weather_enc': weather_enc
    }])
    
    shortfall_pred = model_shortfall.predict(input_A)[0][0] # Output is often [[val]]
    # Ensure non-negative
    shortfall_est = max(0.0, float(shortfall_pred))
    
    # --- 3. MODEL B: RISK SCORER ---
    # Features: ['labour_shortfall_est', 'progress_gap_pct', 'mat_avail_enc', 
    #            'hist_material_delay_count', 'weather_impact_flag', 'construction_stage']
    input_B = pd.DataFrame([{
        'labour_shortfall_est': shortfall_est,
        'progress_gap_pct': inputs['progress_gap_pct'],
        'mat_avail_enc': mat_enc,
        'hist_material_delay_count': inputs['hist_material_delay_count'],
        'weather_impact_flag': weather_impact,
        'construction_stage': inputs['construction_stage']
    }])
    
    risk_score_pred = model_scorer.predict(input_B)[0][0]
    risk_score = max(0.0, float(risk_score_pred))
    
    # --- NUMERICAL MULTIPLIERS (User Defined) ---
    # 1. Schedule Gap: +0.25 pts per 1% Gap
    gap_val = max(0, inputs.get('progress_gap_pct', 0))
    risk_score += (gap_val * 0.25)
    
    # 2. Past Delays: +0.4 pts per Delay
    delays_val = max(0, inputs.get('hist_material_delay_count', 0))
    risk_score += (delays_val * 0.4)
    
    # Cap at 15
    risk_score = min(15.0, risk_score)
    
    # --- 4. MODEL C: CLASSIFIER ---
    # Features: ['risk_score_debug', 'progress_gap_pct']
    # Note: training script called it 'risk_score_debug', we keep name for consistency mapping
    input_C = pd.DataFrame([{
        'risk_score_debug': risk_score,
        'progress_gap_pct': inputs['progress_gap_pct']
    }])
    
    risk_class_pred = model_classifier.predict(input_C)[0] 
    
    # Map Integer Class back to String (0=Low, 1=Medium, 2=High)
    # Note: Based on train_risk_final.csv [0, 1, 2]
    class_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_class_str = class_map.get(int(risk_class_pred), "Unknown")
    
    # --- 5. RETURN DETAILED OUTPUT ---
    return {
        "labour_shortfall_est": round(shortfall_est, 2),
        "risk_score": round(risk_score, 2),
        "risk_class": risk_class_str
    }

if __name__ == "__main__":
    # Test Case
    test_inputs = {
        "req_skilled_labour": 15,
        "req_unskilled_labour": 40,
        "material_avail_status": "Limited",   # 1
        "labour_avail_status": "Moderate",    # 1
        "weather_condition": "High_Wind",     # 2 -> Impact 1
        "progress_gap_pct": 15.0,
        "hist_material_delay_count": 2,
        "construction_stage": 3
    }
    
    print("\n🧐 TEST PREDICTION:")
    results = predict_risk_metrics(test_inputs)
    print(results)