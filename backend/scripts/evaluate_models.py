import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, '..')
MODEL_DIR = os.path.join(BACKEND_DIR, 'models')
DATA_DIR = os.path.join(BACKEND_DIR, 'data', 'raw')

def evaluate_models():
    print("🚀 STARTING GRANULAR MODEL EVALUATION...")
    
    # ====================================================
    # 1. LABOUR MODEL EVALUATION
    # ====================================================
    print(f"\n👷 LABOUR MODEL METRICS")
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "labour_prediction_model.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "labour_model_features.pkl"))
        df = pd.read_csv(os.path.join(DATA_DIR, "train_labour_final.csv"))
        
        # Prep
        df_encoded = pd.get_dummies(df, columns=['task_type'])
        X_final = pd.DataFrame(0, index=df_encoded.index, columns=features)
        for col in features:
            if col in df_encoded.columns:
                X_final[col] = df_encoded[col]
        
        y = df[['req_skilled_labour', 'req_unskilled_labour']]
        
        # Predict
        y_pred = np.expm1(model.predict(X_final))
        y_test_np = y.values
        
        targets = ["Skilled Labour", "Unskilled Labour"]
        for i, name in enumerate(targets):
            r2 = r2_score(y_test_np[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test_np[:, i], y_pred[:, i])
            print(f"   🔹 {name}: R2={r2:.4f}, MAE={mae:.4f}")
            
    except Exception as e:
        print(f"❌ Labour Eval Failed: {e}")

    # ====================================================
    # 2. RISK MODEL EVALUATION (Tri-Model)
    # ====================================================
    print(f"\n🚨 RISK MODEL METRICS (Tri-Model Architecture)")
    try:
        # Load Data
        df_risk = pd.read_csv(os.path.join(DATA_DIR, "train_risk_final.csv"))
        df_labour = pd.read_csv(os.path.join(DATA_DIR, "train_labour_final.csv"))
        
        # --- MERGE LABOUR DATA (As done in training) ---
        # Aggregate Labour to Weekly level
        df_labour_agg = df_labour.groupby(['project_id', 'week_number'])[
            ['req_skilled_labour', 'req_unskilled_labour']
        ].sum().reset_index()
        
        # Merge
        df = pd.merge(df_risk, df_labour_agg, on=['project_id', 'week_number'], how='inner')
        
        # --- PREPROCESSING ---
        mat_map = {'Available': 0, 'Limited': 1, 'Shortage': 2}
        lab_map = {'Sufficient': 0, 'Moderate': 1, 'Critical': 2}
        
        # Apply maps safely
        df['mat_avail_enc'] = df['material_avail_status'].map(mat_map) if df['material_avail_status'].dtype=='O' else df['material_avail_status']
        df['lab_avail_enc'] = df['labour_avail_status'].map(lab_map) if df['labour_avail_status'].dtype=='O' else df['labour_avail_status']
        
        # Weather Map (Check training script logic)
        weather_map = {'Normal': 0, 'Light_Rain': 1, 'High_Wind': 2, 'Heatwave': 3, 'Heavy_Rain': 4}
        if df['weather_condition'].dtype=='O':
             df['weather_enc'] = df['weather_condition'].map(weather_map)
        else:
             df['weather_enc'] = df['weather_condition']
        
        # Create flag
        df['weather_impact_flag'] = df['weather_enc'].apply(lambda x: 1 if x >= 2 else 0)

        # ------------------------------------------------
        # MODEL A: SHORTFALL ESTIMATOR
        # ------------------------------------------------
        print("\n   🔹 [Model A] Shortfall Estimator:")
        model_A = joblib.load(os.path.join(MODEL_DIR, "risk_model_shortfall.pkl"))
        features_A = ['req_skilled_labour', 'req_unskilled_labour', 'lab_avail_enc', 'weather_enc']
        y_A = df['labour_shortfall_est']
        
        pred_A = model_A.predict(df[features_A])
        r2_A = r2_score(y_A, pred_A)
        mae_A = mean_absolute_error(y_A, pred_A)
        print(f"      - R2: {r2_A:.4f}")
        print(f"      - MAE: {mae_A:.4f} Workers")

        # ------------------------------------------------
        # MODEL B: RISK SCORER
        # ------------------------------------------------
        print("\n   🔹 [Model B] Risk Scorer:")
        model_B = joblib.load(os.path.join(MODEL_DIR, "risk_model_scorer.pkl"))
        features_B = ['labour_shortfall_est', 'progress_gap_pct', 'mat_avail_enc', 'hist_material_delay_count', 'weather_impact_flag', 'construction_stage']
        y_B = df['risk_score_debug']
        
        pred_B = model_B.predict(df[features_B])
        r2_B = r2_score(y_B, pred_B)
        mae_B = mean_absolute_error(y_B, pred_B)
        print(f"      - R2: {r2_B:.4f}")
        print(f"      - MAE: {mae_B:.4f} Points")

        # ------------------------------------------------
        # MODEL C: RISK CLASSIFIER
        # ------------------------------------------------
        print("\n   🔹 [Model C] Risk Classifier:")
        model_C = joblib.load(os.path.join(MODEL_DIR, "risk_model_classifier.pkl"))
        features_C = ['risk_score_debug', 'progress_gap_pct']
        y_C = df['delay_risk'] # 0,1,2
        
        pred_C = model_C.predict(df[features_C])
        report = classification_report(y_C, pred_C, target_names=['Low', 'Medium', 'High'])
        print(report)

    except Exception as e:
        print(f"❌ Risk Eval Failed: {e}")

if __name__ == "__main__":
    evaluate_models()
